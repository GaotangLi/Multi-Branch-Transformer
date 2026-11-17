# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact single-node DP gradient safeguard for recurrent-depth supervision.

The implementation supports the no-pipeline, dense, single-node data-parallel
path.  It accumulates two FP32 gradients over every local microbatch, averages
them across the DP group, and therefore projects the exact *global-batch*
gradients:

``G1``
    The gradient of the unweighted first loop-exit NTP loss.

``Gmix``
    The gradient of the ordinary weighted multi-exit objective, computed from
    an independent forward graph on the exact same microbatch tensors.

After all gradient-accumulation microbatches, the closest correction to
``Gmix`` along ``G1`` that satisfies

``<G*, G1> >= rho * ||G1||^2``

is written back to Megatron's normal main-gradient buffers.  Gradient clipping,
AdamW, checkpointing, and the optimizer step then proceed unchanged.

Keeping the collector here (rather than in the language-model loss) is
important: Transformer Engine may fuse weight-gradient accumulation directly
into Megatron's FP32 ``main_grad`` buffers, bypassing ordinary ``param.grad``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from megatron.core import parallel_state


def r1_cone_projection_statistics(
    anchor_buffers: Sequence[Tensor],
    mixture_buffers: Sequence[Tensor],
    *,
    rho: float,
    eps: float,
) -> dict[str, Tensor]:
    """Return the scalar statistics for the R1 half-space projection.

    All dot products are accumulated in the buffers' dtype.  Production use
    requires FP32 buffers; allowing ordinary floating tensors here keeps the
    mathematical helper straightforward to unit test on CPU.
    """

    if len(anchor_buffers) != len(mixture_buffers) or not anchor_buffers:
        raise ValueError("anchor_buffers and mixture_buffers must be non-empty and aligned")
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError(f"rho must be finite and non-negative, got {rho}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    dot_terms = []
    anchor_sq_terms = []
    mixture_sq_terms = []
    for anchor, mixture in zip(anchor_buffers, mixture_buffers):
        if anchor.shape != mixture.shape or anchor.device != mixture.device:
            raise ValueError("aligned gradient buffers must have matching shapes and devices")
        if not anchor.is_floating_point() or not mixture.is_floating_point():
            raise TypeError("gradient buffers must be floating point")
        dot_terms.append(torch.dot(anchor.reshape(-1), mixture.reshape(-1)))
        anchor_sq_terms.append(torch.dot(anchor.reshape(-1), anchor.reshape(-1)))
        mixture_sq_terms.append(torch.dot(mixture.reshape(-1), mixture.reshape(-1)))

    dot = torch.stack(dot_terms).sum()
    anchor_sq = torch.stack(anchor_sq_terms).sum()
    mixture_sq = torch.stack(mixture_sq_terms).sum()
    denominator = anchor_sq + eps
    coefficient = torch.clamp(rho * anchor_sq - dot, min=0.0) / denominator
    effective_before = dot / denominator
    effective_after = (dot + coefficient * anchor_sq) / denominator
    cosine = dot / torch.sqrt((anchor_sq + eps) * (mixture_sq + eps))
    correction_norm_fraction = (
        coefficient * torch.sqrt(anchor_sq + eps) / torch.sqrt(mixture_sq + eps)
    )
    projected_sq = torch.clamp(
        mixture_sq + 2.0 * coefficient * dot + coefficient.square() * anchor_sq,
        min=0.0,
    )
    projected_norm_ratio = torch.sqrt(projected_sq + eps) / torch.sqrt(
        mixture_sq + eps
    )
    return {
        "dot": dot,
        "anchor_squared_norm": anchor_sq,
        "mixture_squared_norm": mixture_sq,
        "coefficient": coefficient,
        "effective_before": effective_before,
        "effective_after": effective_after,
        "cosine": cosine,
        "correction_norm_fraction": correction_norm_fraction,
        "projected_norm_ratio": projected_norm_ratio,
        "intervened": (coefficient > 0.0).to(dtype=dot.dtype),
    }


def project_r1_cone_(
    output_buffers: Sequence[Tensor],
    anchor_buffers: Sequence[Tensor],
    mixture_buffers: Sequence[Tensor],
    *,
    rho: float,
    eps: float,
) -> dict[str, Tensor]:
    """Write ``Gmix + correction * G1`` into ``output_buffers`` in place."""

    if len(output_buffers) != len(anchor_buffers):
        raise ValueError("output_buffers must align with the gradient buffers")
    statistics = r1_cone_projection_statistics(
        anchor_buffers,
        mixture_buffers,
        rho=rho,
        eps=eps,
    )
    coefficient = statistics["coefficient"]
    for output, anchor, mixture in zip(output_buffers, anchor_buffers, mixture_buffers):
        if output.shape != anchor.shape or output.device != anchor.device:
            raise ValueError("output gradient buffers must match the accumulated buffers")
        # addcmul accepts a zero-dimensional tensor and performs the broadcast
        # in one kernel, avoiding a full-sized ``coefficient * anchor`` temporary.
        torch.addcmul(mixture, anchor, coefficient, out=output)
    return statistics


@dataclass
class _CollectorState:
    """Persistent storage reused across optimizer steps."""

    current_grad_buffers: list[Tensor]
    anchor_accumulators: list[Tensor]
    mixture_accumulators: list[Tensor]
    rho: float
    eps: float
    first_exit_weight: float
    expected_microbatches: int = 0
    observed_microbatches: int = 0
    pending_anchor: bool = False
    active: bool = False
    last_metrics: dict[str, Tensor] | None = None

    @classmethod
    def create(
        cls,
        current_grad_buffers: Sequence[Tensor],
        *,
        rho: float,
        eps: float,
        first_exit_weight: float,
    ) -> "_CollectorState":
        current = list(current_grad_buffers)
        return cls(
            current_grad_buffers=current,
            anchor_accumulators=[torch.zeros_like(buffer) for buffer in current],
            mixture_accumulators=[torch.zeros_like(buffer) for buffer in current],
            rho=rho,
            eps=eps,
            first_exit_weight=first_exit_weight,
        )

    def matches(self, current_grad_buffers: Sequence[Tensor]) -> bool:
        current = list(current_grad_buffers)
        return len(current) == len(self.current_grad_buffers) and all(
            old.data_ptr() == new.data_ptr()
            and old.shape == new.shape
            and old.dtype == new.dtype
            and old.device == new.device
            for old, new in zip(self.current_grad_buffers, current)
        )

    def begin(
        self,
        expected_microbatches: int,
        *,
        rho: float,
        eps: float,
        first_exit_weight: float,
    ) -> None:
        if self.active:
            raise RuntimeError("loop-exit gradient safeguard collector is already active")
        if expected_microbatches < 1:
            raise ValueError("expected_microbatches must be positive")
        self.expected_microbatches = expected_microbatches
        self.observed_microbatches = 0
        self.pending_anchor = False
        self.rho = rho
        self.eps = eps
        self.first_exit_weight = first_exit_weight
        self.last_metrics = None
        for accumulator in self.anchor_accumulators:
            accumulator.zero_()
        for accumulator in self.mixture_accumulators:
            accumulator.zero_()
        self.active = True

    def _capture_and_zero(self, accumulators: Sequence[Tensor]) -> None:
        for accumulator, current in zip(accumulators, self.current_grad_buffers):
            accumulator.add_(current)
            current.zero_()

    def backward(self, objective: Tensor, *, phase: str, grad_scale_func=None) -> None:
        if not self.active:
            raise RuntimeError("loop-exit gradient safeguard collector was not started")
        if not isinstance(objective, Tensor) or objective.ndim != 0:
            raise RuntimeError("loop-exit gradient safeguard objectives must be scalar tensors")
        if self.observed_microbatches >= self.expected_microbatches:
            raise RuntimeError("received more safeguard microbatches than expected")

        scaled_objective = (
            grad_scale_func(objective) if grad_scale_func is not None else objective
        )
        if phase == "anchor":
            if self.pending_anchor:
                raise RuntimeError("received two anchor objectives without a mixture objective")
            torch.autograd.backward(scaled_objective)
            self._capture_and_zero(self.anchor_accumulators)
            self.pending_anchor = True
        elif phase == "mixture":
            if not self.pending_anchor:
                raise RuntimeError("mixture objective arrived before its anchor objective")
            torch.autograd.backward(scaled_objective)
            self._capture_and_zero(self.mixture_accumulators)
            self.pending_anchor = False
            self.observed_microbatches += 1
        else:
            raise ValueError(f"unknown loop-gradient safeguard phase: {phase!r}")

    def finalize(self) -> dict[str, Tensor]:
        if not self.active:
            raise RuntimeError("loop-exit gradient safeguard collector was not started")
        if self.observed_microbatches != self.expected_microbatches:
            raise RuntimeError(
                "loop-exit gradient safeguard microbatch count mismatch: "
                f"observed {self.observed_microbatches}, expected {self.expected_microbatches}"
            )
        if self.pending_anchor:
            raise RuntimeError("loop-exit gradient safeguard has an unmatched anchor objective")

        # Surgery must act on the global DP gradient.  Projecting independently
        # on each rank and then reducing is not equivalent because the
        # half-space projection is nonlinear.  Average both objective buffers
        # first.  The ordinary Megatron final reduce-scatter is intentionally
        # left enabled: every rank writes the same projected global gradient,
        # so a second average is mathematically an identity while preserving
        # the standard distributed-optimizer path.
        dp_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=True)
            if dist.is_initialized()
            else None
        )
        dp_world_size = dist.get_world_size(group=dp_group) if dp_group is not None else 1
        if dp_world_size > 1:
            for accumulator in (
                *self.anchor_accumulators,
                *self.mixture_accumulators,
            ):
                dist.all_reduce(accumulator, op=dist.ReduceOp.SUM, group=dp_group)
                accumulator.div_(dp_world_size)
        statistics = project_r1_cone_(
            self.current_grad_buffers,
            self.anchor_accumulators,
            self.mixture_accumulators,
            rho=self.rho,
            eps=self.eps,
        )
        deeper_weight = 1.0 - self.first_exit_weight
        if not 0.0 < deeper_weight <= 1.0:
            raise RuntimeError(
                "loop-exit gradient safeguard requires positive aggregate deeper-exit weight"
            )
        anchor_sq = statistics["anchor_squared_norm"]
        mixture_sq = statistics["mixture_squared_norm"]
        anchor_mixture_dot = statistics["dot"]
        auxiliary_dot = (
            anchor_mixture_dot - self.first_exit_weight * anchor_sq
        ) / deeper_weight
        auxiliary_sq = torch.clamp(
            (
                mixture_sq
                - 2.0 * self.first_exit_weight * anchor_mixture_dot
                + self.first_exit_weight**2 * anchor_sq
            )
            / deeper_weight**2,
            min=0.0,
        )
        auxiliary_cosine = auxiliary_dot / torch.sqrt(
            (anchor_sq + self.eps) * (auxiliary_sq + self.eps)
        )
        auxiliary_effective_r1 = auxiliary_dot / (anchor_sq + self.eps)
        # If the auxiliary geometry stayed fixed, this is the literal scalar
        # R1 weight needed to meet rho:
        #   w + (1-w) * auxiliary_effective_r1 >= rho.
        # It is diagnostic only; the actual update remains the minimum cone
        # correction above and adapts to the current full-gradient geometry.
        equivalent_normalized_r1_weight = torch.where(
            auxiliary_effective_r1 >= self.rho,
            torch.zeros_like(auxiliary_effective_r1),
            torch.clamp(
                (self.rho - auxiliary_effective_r1)
                / (1.0 - auxiliary_effective_r1 + self.eps),
                min=0.0,
                max=1.0,
            ),
        )
        self.last_metrics = {
            "optimizer/loop_r1_safeguard_effective_before": statistics[
                "effective_before"
            ].detach(),
            "optimizer/loop_r1_safeguard_effective_after": statistics[
                "effective_after"
            ].detach(),
            "optimizer/loop_r1_safeguard_cosine": statistics["cosine"].detach(),
            "optimizer/loop_r1_safeguard_correction_coefficient": statistics[
                "coefficient"
            ].detach(),
            "optimizer/loop_r1_safeguard_correction_norm_fraction": statistics[
                "correction_norm_fraction"
            ].detach(),
            "optimizer/loop_r1_safeguard_projected_norm_ratio": statistics[
                "projected_norm_ratio"
            ].detach(),
            "optimizer/loop_r1_safeguard_intervened": statistics["intervened"].detach(),
            "optimizer/loop_r1_safeguard_auxiliary_cosine": auxiliary_cosine.detach(),
            "optimizer/loop_r1_safeguard_auxiliary_effective_r1": (
                auxiliary_effective_r1.detach()
            ),
            "optimizer/loop_r1_safeguard_equivalent_normalized_r1_weight": (
                equivalent_normalized_r1_weight.detach()
            ),
        }
        self.active = False
        return self.last_metrics


def _ddp_grad_buffers(model: Any) -> list[Tensor]:
    ddp_config = getattr(model, "ddp_config", None)
    if ddp_config is None:
        raise RuntimeError("loop-exit gradient safeguard requires Megatron DDP")
    if getattr(ddp_config, "overlap_grad_reduce", False):
        raise RuntimeError(
            "loop-exit gradient safeguard requires overlap_grad_reduce=False; "
            "two objective backwards cannot mark the same DDP buckets ready twice"
        )
    if getattr(ddp_config, "use_megatron_fsdp", False):
        raise RuntimeError("loop-exit gradient safeguard does not yet support Megatron FSDP")
    expert_buffers = list(getattr(model, "expert_parallel_buffers", ()))
    if expert_buffers:
        raise RuntimeError("loop-exit gradient safeguard does not yet support expert buffers")

    parameter_buffers = getattr(model, "buffers", None)
    if parameter_buffers is None or callable(parameter_buffers):
        raise RuntimeError("Megatron DDP parameter buffers were not found")
    grad_buffers = [buffer.grad_data for buffer in parameter_buffers]
    if not grad_buffers:
        raise RuntimeError("Megatron DDP exposed no main-gradient buffers")
    if any(buffer.dtype != torch.float32 for buffer in grad_buffers):
        dtypes = sorted({str(buffer.dtype) for buffer in grad_buffers})
        raise RuntimeError(
            "loop-exit gradient safeguard requires FP32 main-gradient buffers; "
            f"got {dtypes}"
        )
    return grad_buffers


def begin_loop_exit_gradient_safeguard(
    model: Any,
    config: Any,
    *,
    num_microbatches: int,
) -> None:
    """Prepare persistent accumulators for one optimizer step."""

    if getattr(config, "calculate_per_token_loss", False):
        raise RuntimeError("loop-exit gradient safeguard requires calculate_per_token_loss=False")
    if getattr(config, "cuda_graph_impl", "none") != "none":
        raise RuntimeError("loop-exit gradient safeguard does not yet support CUDA graphs")
    if getattr(config, "num_moe_experts", None) is not None:
        raise RuntimeError("loop-exit gradient safeguard currently supports dense models only")

    grad_buffers = _ddp_grad_buffers(model)
    explicit_weights = getattr(config, "loop_depth_loss_weights", None)
    if explicit_weights is None:
        first_exit_weight = 1.0 / (1.0 + config.loop_depth_aux_loss_weight)
    else:
        first_exit_weight = float(explicit_weights[0]) / sum(explicit_weights)
    state = getattr(model, "_loop_exit_gradient_safeguard_state", None)
    if state is None or not state.matches(grad_buffers):
        state = _CollectorState.create(
            grad_buffers,
            rho=config.loop_exit_gradient_safeguard_rho,
            eps=config.loop_exit_gradient_safeguard_eps,
            first_exit_weight=first_exit_weight,
        )
        model._loop_exit_gradient_safeguard_state = state
    state.begin(
        num_microbatches,
        rho=config.loop_exit_gradient_safeguard_rho,
        eps=config.loop_exit_gradient_safeguard_eps,
        first_exit_weight=first_exit_weight,
    )
    config._loop_exit_gradient_safeguard_state = state


def backward_loop_exit_gradient_safeguard(
    objective: Tensor, config: Any, *, phase: str
) -> None:
    state = getattr(config, "_loop_exit_gradient_safeguard_state", None)
    if state is None:
        raise RuntimeError("loop-exit gradient safeguard backward has no active collector")
    state.backward(
        objective,
        phase=phase,
        grad_scale_func=getattr(config, "grad_scale_func", None),
    )


def finalize_loop_exit_gradient_safeguard(config: Any) -> None:
    state = getattr(config, "_loop_exit_gradient_safeguard_state", None)
    if state is None:
        raise RuntimeError("loop-exit gradient safeguard finalize has no active collector")
    state.finalize()
    # The collector's persistent tensors live only on the DDP wrapper.  Do not
    # leave them reachable from TransformerConfig, which may be serialized as
    # checkpoint metadata.
    delattr(config, "_loop_exit_gradient_safeguard_state")


def pop_loop_exit_gradient_safeguard_metrics(model: Any) -> dict[str, Tensor]:
    """Return one-step diagnostics without synchronizing them to the CPU."""

    state = getattr(model, "_loop_exit_gradient_safeguard_state", None)
    if state is None or state.last_metrics is None:
        return {}
    metrics = state.last_metrics
    state.last_metrics = None
    return metrics
