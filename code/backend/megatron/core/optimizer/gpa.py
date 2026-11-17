# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generalized Primal Averaging AdamW optimizer.

This module contains the fixed-coefficient GPA variant.  The optimizer keeps
``y`` in the parameter buffer, ``z`` in optimizer state, and reconstructs the
evaluation iterate ``x`` when :meth:`eval` is called.

Reference: https://arxiv.org/abs/2512.17131
"""

from collections.abc import Callable
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT


class FixedGPAParamSequence:
    """Shared fixed-coefficient GPA parameter-sequence lifecycle."""

    param_groups: list[dict[str, Any]]
    state: dict[torch.Tensor, dict[str, Any]]

    @staticmethod
    def validate_interp_coefficients(
        train_interp_coeff: float, eval_interp_coeff: float
    ) -> None:
        if not 0.0 < train_interp_coeff <= 1.0:
            raise ValueError(
                "train_interp_coeff must be in the range (0, 1], "
                f"got {train_interp_coeff}"
            )
        if not 0.0 <= eval_interp_coeff <= 1.0:
            raise ValueError(
                "eval_interp_coeff must be in the range [0, 1], "
                f"got {eval_interp_coeff}"
            )

    @torch.no_grad()
    def _initialize_gpa_buffers(self) -> None:
        for group in self.param_groups:
            group.setdefault("train_mode", False)
            for param in group["params"]:
                state = self.state[param]
                if "z" not in state:
                    state["z"] = param.detach().clone(memory_format=torch.preserve_format)

    def _require_gpa_train_mode(self) -> None:
        for group in self.param_groups:
            if group["params"] and not group["train_mode"]:
                raise RuntimeError(
                    f"{type(self).__name__}.step() requires train mode; "
                    "call optimizer.train() first"
                )

    @torch.no_grad()
    def train(self) -> None:
        """Put parameters at the ``y`` iterate used for gradient computation."""
        self.initialize_state()  # type: ignore[attr-defined]
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            mu_y = group["train_interp_coeff"]
            for param in group["params"]:
                param.lerp_(self.state[param]["z"], 1.0 - mu_y)
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self) -> None:
        """Put parameters at the ``x`` iterate used for evaluation."""
        self.initialize_state()  # type: ignore[attr-defined]
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            mu_y = group["train_interp_coeff"]
            for param in group["params"]:
                param.lerp_(self.state[param]["z"], 1.0 - 1.0 / mu_y)
            group["train_mode"] = False


class GPAAdamW(FixedGPAParamSequence, Optimizer):
    r"""AdamW with fixed-coefficient Generalized Primal Averaging.

    With ``mu_y=train_interp_coeff`` and ``mu_x=eval_interp_coeff``, the
    iterates are

    .. math::

        y_t &= \mu_y x_t + (1 - \mu_y) z_t, \\
        z_{t+1} &= z_t - \alpha_t u_t, \\
        x_{t+1} &= \mu_x x_t + (1 - \mu_x) z_{t+1}.

    The Adam-normalized update is ``u_t``.  Parameters must be switched to
    train mode before stepping and to eval mode before evaluation or saving a
    model checkpoint.

    Shared scalar state is stored in parameter groups.  Per-parameter state
    contains only model-shaped tensors so Megatron distributed checkpointing
    can shard it like ordinary Adam state.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        eps: float = 1e-8,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        use_wd_on_y: bool = False,
        train_interp_coeff: float = 0.7,
        eval_interp_coeff: float = 0.9967,
    ) -> None:
        self._validate_hyperparameters(
            lr, eps, beta1, beta2, weight_decay, train_interp_coeff, eval_interp_coeff
        )
        defaults = {
            "optimizer": "gpa_adamw",
            "lr": lr,
            "eps": eps,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "use_wd_on_y": use_wd_on_y,
            "train_interp_coeff": train_interp_coeff,
            "eval_interp_coeff": eval_interp_coeff,
            "step": 0,
            "train_mode": False,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _validate_hyperparameters(
        lr: float,
        eps: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        train_interp_coeff: float,
        eval_interp_coeff: float,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        FixedGPAParamSequence.validate_interp_coefficients(
            train_interp_coeff, eval_interp_coeff
        )

    @torch.no_grad()
    def initialize_state(self) -> None:
        """Initialize checkpointable state for every parameter, including no-grad params."""
        self._initialize_gpa_buffers()
        for group in self.param_groups:
            group.setdefault("step", 0)
            for param in group["params"]:
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    )
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        """Perform one GPA-AdamW update."""
        self.initialize_state()
        self._require_gpa_train_mode()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None and param.grad.is_sparse:
                    raise RuntimeError("GPAAdamW does not support sparse gradients")

        for group in self.param_groups:
            if not group["params"]:
                continue

            group["step"] += 1
            step = group["step"]
            lr = group["lr"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            use_wd_on_y = group["use_wd_on_y"]
            mu_y = group["train_interp_coeff"]
            avg_coeff = 1.0 - group["eval_interp_coeff"]

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                z = state["z"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = param.grad.detach()
                if grad.dtype != torch.float32:
                    grad = grad.float()

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                normalized_grad = (exp_avg / bias_correction1) / denom

                if weight_decay != 0.0:
                    if use_wd_on_y:
                        param.mul_(1.0 - lr * weight_decay)
                    else:
                        z.mul_(1.0 - lr * weight_decay)

                param.lerp_(z, avg_coeff)
                param.add_(normalized_grad, alpha=lr * (mu_y * (1.0 - avg_coeff) - 1.0))
                z.sub_(normalized_grad, alpha=lr)

        return loss
