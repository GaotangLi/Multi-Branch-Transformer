# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""TP=1 Marin May-recipe AdamH/MuonH optimizer.

This module intentionally implements one hybrid optimizer rather than chaining
independent Adam and Muon optimizers.  Megatron fuses semantically different
parameters into QGKV and SwiGLU tensors; keeping the update in one optimizer is
what lets those logical slices use different transformations while retaining a
single FP32 master parameter and a checkpointable state.

The implementation follows ``experiments/grug/moe/optimizer.py`` in Marin:

* hidden matrices use MuonH (momentum, quintic Newton--Schulz, Hyperball);
* the untied LM head uses AdamH;
* embeddings, routers, gates, biases, and norms use Adam at 3/13 of the
  Hyperball learning rate;
* each fused Q/K/V projection and each SwiGLU half preserves its own Frobenius
  norm.  The fused attention gate uses ordinary Adam.

Only unsharded logical matrices are exact here.  The Megatron factory therefore
rejects tensor parallelism, distributed optimizer/FSDP, and grouped-weight
layouts before constructing this optimizer.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


MARIN_HYPERBALL_ROLE = "marin_hyperball_role"
MARIN_SCALAR_LR_RATIO = 3.0 / 13.0

ROLE_ADAM = "adam"
ROLE_ADAMH = "adamh"
ROLE_MUONH = "muonh"
ROLE_QKV_MUONH = "qkv_muonh"
ROLE_QGKV_MIXED = "qgkv_mixed"
ROLE_SWIGLU_MUONH = "swiglu_muonh"

MARIN_HYPERBALL_ROLES = (
    ROLE_ADAM,
    ROLE_ADAMH,
    ROLE_MUONH,
    ROLE_QKV_MUONH,
    ROLE_QGKV_MIXED,
    ROLE_SWIGLU_MUONH,
)

_QUINTIC_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
_HYPERBALL_NORM_FLOOR = 1.0e-10


def classify_marin_hyperball_parameter(
    name: str, param: torch.nn.Parameter, model_config: Any
) -> str:
    """Assign every supported GPT parameter to one exact Marin update role.

    Name checks precede the shape fallback because embeddings and routers are
    matrices but Marin intentionally sends them to ordinary Adam.  A rank above
    three is rejected rather than silently flattening an unknown logical layout.
    """

    lower_name = name.lower()

    # An untied LM head is AdamH.  A tied weight appears under the embedding
    # name and is therefore deliberately treated as the token embedding below.
    if lower_name.endswith("output_layer.weight") or "lm_head.weight" in lower_name:
        if param.ndim != 2:
            raise ValueError(
                f"Marin AdamH LM head must be rank 2, got {name} with shape {tuple(param.shape)}"
            )
        return ROLE_ADAMH

    if (
        "word_embeddings" in lower_name
        or "position_embeddings" in lower_name
        or "token_embed" in lower_name
        or ".embedding." in lower_name
    ):
        return ROLE_ADAM

    if (
        "router" in lower_name
        or "router_bias" in lower_name
        or "attn_gate" in lower_name
        or "attention_gate" in lower_name
    ):
        return ROLE_ADAM

    if lower_name.endswith("linear_qkv.weight"):
        if param.ndim != 2:
            raise ValueError(
                f"Fused QKV weight must be rank 2, got {name} with shape {tuple(param.shape)}"
            )
        if getattr(model_config, "attention_output_gate", False):
            return ROLE_QGKV_MIXED
        return ROLE_QKV_MUONH

    is_swiglu_fc1 = bool(
        getattr(model_config, "gated_linear_unit", False)
        and re.search(r"linear_fc1\.weight\d*$", lower_name)
    )
    if is_swiglu_fc1:
        if param.ndim != 2 or param.shape[0] % 2 != 0:
            raise ValueError(
                "Fused SwiGLU FC1 must be a rank-2 tensor with an even output dimension, "
                f"got {name} with shape {tuple(param.shape)}"
            )
        return ROLE_SWIGLU_MUONH

    # Biases, normalization scales, and scalar controls use ordinary Adam.
    if param.ndim <= 1 or lower_name.endswith(".bias") or lower_name.endswith("_bias"):
        return ROLE_ADAM

    if param.ndim in (2, 3):
        return ROLE_MUONH

    raise ValueError(
        "No exact Marin Hyperball logical-matrix rule for "
        f"parameter {name!r} with shape {tuple(param.shape)}"
    )


def _matrix_norm(tensor: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(tensor.ndim - 2, tensor.ndim))
    return torch.linalg.vector_norm(tensor, dim=axes, keepdim=True)


def hyperball_project(
    param: torch.Tensor,
    direction: torch.Tensor,
    learning_rate: float | torch.Tensor,
    norm_floor: float = _HYPERBALL_NORM_FLOOR,
) -> torch.Tensor:
    """Return the Marin norm-preserving Hyperball parameter value.

    Rank-2 tensors preserve one Frobenius norm.  Rank-3 tensors are interpreted
    as a stack of matrices and preserve one norm per leading-axis slice.
    Zero parameters or exactly zero directions are exact no-ops.  A nonzero
    direction below ``norm_floor`` is still applied with the clamped divisor,
    matching Marin's canonical implementation.
    """

    if param.ndim not in (2, 3):
        raise ValueError(f"Hyperball expects rank 2 or 3, got shape {tuple(param.shape)}")
    if direction.shape != param.shape:
        raise ValueError(
            f"Hyperball direction shape {tuple(direction.shape)} does not match "
            f"parameter shape {tuple(param.shape)}"
        )

    param_norm = _matrix_norm(param)
    direction_norm = _matrix_norm(direction)
    candidate = param - learning_rate * direction * param_norm / direction_norm.clamp_min(
        norm_floor
    )
    candidate_norm = _matrix_norm(candidate)
    projected = candidate * param_norm / candidate_norm.clamp_min(norm_floor)
    # Guard exact zeros only.  In particular, NaN norms must select the
    # projected branch so canonical non-finite propagation is not masked as a
    # silent no-op by an ordered comparison (``NaN > 0`` is false).
    active = (param_norm != 0) & (direction_norm != 0)
    return torch.where(active, projected, param)


def quintic_newton_schulz(
    direction: torch.Tensor, steps: int = 5, eps: float = 1.0e-8
) -> torch.Tensor:
    """Marin's BF16 quintic Newton--Schulz direction for rank-2/3 tensors."""

    if direction.ndim not in (2, 3):
        raise ValueError(
            f"Newton--Schulz expects rank 2 or 3, got shape {tuple(direction.shape)}"
        )
    if steps < 1:
        raise ValueError(f"Newton--Schulz steps must be positive, got {steps}")

    original_dtype = direction.dtype
    work = direction.to(torch.bfloat16)
    work = work / (_matrix_norm(work) + eps)

    transposed = work.shape[-2] > work.shape[-1]
    if transposed:
        work = work.transpose(-2, -1)

    for index in range(steps):
        a, b, c = _QUINTIC_COEFFICIENTS[index % len(_QUINTIC_COEFFICIENTS)]
        gram = work @ work.transpose(-2, -1)
        polynomial = b * gram + c * (gram @ gram)
        work = a * work + polynomial @ work

    if transposed:
        work = work.transpose(-2, -1)

    # This scale is present in Marin's Grug Muon transform.  Hyperball removes
    # positive scalar magnitude mathematically, but retaining it matches the
    # finite-precision reference and makes this helper independently faithful.
    # Marin's raw JAX matrices are [fan_in, fan_out], whereas MCore stores
    # linear weights as [fan_out, fan_in].  Transpose the layout convention
    # when carrying over Grug Muon's finite-precision direction scale.
    fan_out, fan_in = work.shape[-2:]
    scale = math.sqrt(max(1.0, fan_out / fan_in))
    return work.to(original_dtype) * scale


class MarinHyperball(Optimizer):
    """Mixed Adam/AdamH/MuonH optimizer for unsharded Megatron parameters."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1.0e-8,
        momentum: float = 0.95,
        nesterov: bool = True,
        num_ns_steps: int = 5,
        muon_eps: float = 1.0e-8,
        qkv_split_shapes: Iterable[int] | None = None,
        scalar_lr_ratio: float = MARIN_SCALAR_LR_RATIO,
    ) -> None:
        if lr is None or lr < 0.0:
            raise ValueError(f"Invalid Hyperball learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid Adam betas: {betas}")
        if eps < 0.0 or muon_eps < 0.0:
            raise ValueError("Adam and Muon epsilons must be non-negative")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid Muon momentum: {momentum}")
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be positive, got {num_ns_steps}")
        if scalar_lr_ratio != MARIN_SCALAR_LR_RATIO:
            raise ValueError(
                "The exact Marin first-stage port fixes scalar Adam LR to 3/13 of Hyperball LR"
            )

        split_shapes = tuple(qkv_split_shapes or ())
        if split_shapes and (len(split_shapes) not in (3, 4) or min(split_shapes) <= 0):
            raise ValueError(f"Invalid Q(G)KV split shapes: {split_shapes}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "momentum": momentum,
            "nesterov": nesterov,
            "num_ns_steps": num_ns_steps,
            "muon_eps": muon_eps,
        }
        super().__init__(params, defaults)
        self.qkv_split_shapes = split_shapes
        self.scalar_lr_ratio = scalar_lr_ratio
        self._adam_step = 0

        for group in self.param_groups:
            role = group.get(MARIN_HYPERBALL_ROLE)
            if role not in MARIN_HYPERBALL_ROLES:
                raise ValueError(
                    f"Every Marin Hyperball parameter group needs a known role; got {role!r}"
                )
            self._validate_group(group)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state and restore Marin's transform-global Adam step count."""

        super().load_state_dict(state_dict)
        # MCore stores one ``common_step`` for optimizers whose per-parameter
        # Adam counts agree.  On load its generic wrapper restores that field
        # into *every* state, including pure Muon leaves that never owned an
        # Adam count.  If retained, the next update advances Adam leaves while
        # the injected Muon counts stay stale; the following sharded save then
        # rejects the inconsistent steps.  Strip the synthetic field by role
        # and reconstruct Marin's transform-global count only from leaves that
        # actually run an Adam transform.
        adam_roles = (ROLE_ADAM, ROLE_ADAMH, ROLE_QGKV_MIXED)
        steps = set()
        for group in self.param_groups:
            role = group[MARIN_HYPERBALL_ROLE]
            for param in group["params"]:
                param_state = self.state.get(param)
                if not param_state:
                    continue
                if role not in adam_roles:
                    param_state.pop("step", None)
                    continue
                if "step" not in param_state:
                    continue
                step = param_state["step"]
                if isinstance(step, torch.Tensor):
                    step = int(step.item())
                steps.add(int(step))
        if len(steps) > 1:
            raise ValueError(f"Marin Hyperball checkpoint has inconsistent Adam steps: {steps}")
        self._adam_step = next(iter(steps), 0)

    def _validate_group(self, group: dict[str, Any]) -> None:
        role = group[MARIN_HYPERBALL_ROLE]
        for param in group["params"]:
            if role in (ROLE_ADAM,):
                continue
            if role == ROLE_ADAMH and param.ndim != 2:
                raise ValueError(f"AdamH requires a matrix, got shape {tuple(param.shape)}")
            if role == ROLE_MUONH and param.ndim not in (2, 3):
                raise ValueError(f"MuonH requires rank 2/3, got shape {tuple(param.shape)}")
            if role == ROLE_SWIGLU_MUONH and (param.ndim != 2 or param.shape[0] % 2):
                raise ValueError(f"Invalid fused SwiGLU shape: {tuple(param.shape)}")
            if role in (ROLE_QKV_MUONH, ROLE_QGKV_MIXED) and param.ndim != 2:
                raise ValueError(f"Invalid fused Q(G)KV shape: {tuple(param.shape)}")

        if role == ROLE_QKV_MUONH and len(self.qkv_split_shapes) != 3:
            raise ValueError(
                f"QKV role requires three split widths, got {self.qkv_split_shapes}"
            )
        if role == ROLE_QGKV_MIXED and len(self.qkv_split_shapes) != 4:
            raise ValueError(
                f"QGKV role requires four split widths, got {self.qkv_split_shapes}"
            )

    def _init_group(self, group: dict[str, Any], skip_non_grad_params: bool = True) -> None:
        """Materialize all role-specific state (also used by torch_dist load)."""

        role = group[MARIN_HYPERBALL_ROLE]
        for param in group["params"]:
            if skip_non_grad_params and param.grad is None:
                continue
            state = self.state[param]
            if role in (ROLE_ADAM, ROLE_ADAMH, ROLE_QGKV_MIXED):
                state.setdefault("step", 0)
                state.setdefault("exp_avg", torch.zeros_like(param))
                state.setdefault("exp_avg_sq", torch.zeros_like(param))
            if role in (
                ROLE_MUONH,
                ROLE_QKV_MUONH,
                ROLE_QGKV_MIXED,
                ROLE_SWIGLU_MUONH,
            ):
                state.setdefault("momentum_buffer", torch.zeros_like(param))

    @staticmethod
    def _adam_direction(
        grad: torch.Tensor,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        step: int,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> torch.Tensor:
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        return (exp_avg / bias_correction1) / (
            (exp_avg_sq / bias_correction2).sqrt() + eps
        )

    @staticmethod
    def _muonh_value(
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        *,
        learning_rate: float,
        momentum: float,
        nesterov: bool,
        num_ns_steps: int,
        muon_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_buffer = momentum_buffer * momentum + grad
        direction = next_buffer * momentum + grad if nesterov else next_buffer
        orthogonal = quintic_newton_schulz(direction, steps=num_ns_steps, eps=muon_eps)
        return hyperball_project(param, orthogonal, learning_rate), next_buffer

    def _step_muonh_matrix(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        momentum_buffer: torch.Tensor,
        group: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._muonh_value(
            param,
            grad,
            momentum_buffer,
            learning_rate=group["lr"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            num_ns_steps=group["num_ns_steps"],
            muon_eps=group["muon_eps"],
        )

    def _step_qgkv(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
        has_gate: bool,
    ) -> None:
        widths = self.qkv_split_shapes
        total_width = sum(widths)
        if param.shape[0] % total_width:
            raise ValueError(
                f"Q(G)KV leading dimension {param.shape[0]} is not divisible by {total_width}"
            )
        num_query_groups = param.shape[0] // total_width
        view_shape = (num_query_groups, total_width, param.shape[-1])
        param_view = param.view(view_shape)
        grad_view = grad.view(view_shape)
        momentum_view = state["momentum_buffer"].view(view_shape)
        exp_avg_view = state.get("exp_avg")
        exp_avg_sq_view = state.get("exp_avg_sq")
        if exp_avg_view is not None:
            exp_avg_view = exp_avg_view.view(view_shape)
            exp_avg_sq_view = exp_avg_sq_view.view(view_shape)

        offset = 0
        for split_index, width in enumerate(widths):
            target = param_view[:, offset : offset + width, :]
            target_grad = grad_view[:, offset : offset + width, :]
            is_gate = has_gate and split_index == 1
            if is_gate:
                beta1, beta2 = group["betas"]
                direction = self._adam_direction(
                    target_grad,
                    exp_avg_view[:, offset : offset + width, :],
                    exp_avg_sq_view[:, offset : offset + width, :],
                    state["step"],
                    beta1,
                    beta2,
                    group["eps"],
                )
                target.add_(
                    direction,
                    alpha=-group["lr"] * self.scalar_lr_ratio,
                )
            else:
                target_buffer = momentum_view[:, offset : offset + width, :]
                matrix_shape = (-1, param.shape[-1])
                next_value, next_buffer = self._step_muonh_matrix(
                    target.reshape(matrix_shape),
                    target_grad.reshape(matrix_shape),
                    target_buffer.reshape(matrix_shape),
                    group,
                )
                target.copy_(next_value.view_as(target))
                target_buffer.copy_(next_buffer.view_as(target_buffer))
            offset += width

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Optax's Adam/AdamH transforms use one count shared by all leaves and
        # advance it on every optimizer invocation, including for a temporarily
        # missing leaf update.  Materialize and synchronize the per-parameter
        # copies that Megatron's checkpoint format expects.
        has_adam_role = any(
            group[MARIN_HYPERBALL_ROLE] in (ROLE_ADAM, ROLE_ADAMH, ROLE_QGKV_MIXED)
            for group in self.param_groups
        )
        if has_adam_role:
            self._adam_step += 1
            for group in self.param_groups:
                if group[MARIN_HYPERBALL_ROLE] in (
                    ROLE_ADAM,
                    ROLE_ADAMH,
                    ROLE_QGKV_MIXED,
                ):
                    self._init_group(group, skip_non_grad_params=False)
                    for param in group["params"]:
                        self.state[param]["step"] = self._adam_step

        for group in self.param_groups:
            role = group[MARIN_HYPERBALL_ROLE]
            self._init_group(group, skip_non_grad_params=True)
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MarinHyperball does not support sparse gradients")

                state = self.state[param]

                if role in (ROLE_ADAM, ROLE_ADAMH):
                    beta1, beta2 = group["betas"]
                    direction = self._adam_direction(
                        grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        beta1,
                        beta2,
                        group["eps"],
                    )
                    if role == ROLE_ADAM:
                        param.add_(direction, alpha=-group["lr"])
                    else:
                        param.copy_(hyperball_project(param, direction, group["lr"]))
                elif role == ROLE_MUONH:
                    next_value, next_buffer = self._step_muonh_matrix(
                        param, grad, state["momentum_buffer"], group
                    )
                    param.copy_(next_value)
                    state["momentum_buffer"].copy_(next_buffer)
                elif role == ROLE_SWIGLU_MUONH:
                    for target, target_grad, target_buffer in zip(
                        param.chunk(2, dim=0),
                        grad.chunk(2, dim=0),
                        state["momentum_buffer"].chunk(2, dim=0),
                    ):
                        next_value, next_buffer = self._step_muonh_matrix(
                            target, target_grad, target_buffer, group
                        )
                        target.copy_(next_value)
                        target_buffer.copy_(next_buffer)
                elif role in (ROLE_QKV_MUONH, ROLE_QGKV_MIXED):
                    self._step_qgkv(
                        param,
                        grad,
                        state,
                        group,
                        has_gate=role == ROLE_QGKV_MIXED,
                    )
                else:  # Constructor validation makes this unreachable.
                    raise RuntimeError(f"Unknown Marin Hyperball role: {role}")

        return loss


__all__ = [
    "MARIN_HYPERBALL_ROLE",
    "MARIN_HYPERBALL_ROLES",
    "MARIN_SCALAR_LR_RATIO",
    "MarinHyperball",
    "ROLE_ADAM",
    "ROLE_ADAMH",
    "ROLE_MUONH",
    "ROLE_QGKV_MIXED",
    "ROLE_QKV_MUONH",
    "ROLE_SWIGLU_MUONH",
    "classify_marin_hyperball_parameter",
    "hyperball_project",
    "quintic_newton_schulz",
]
