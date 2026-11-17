# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small, stable W&B logging schema for pretraining runs.

Megatron's legacy logging emits one ``wandb.log`` call per scalar at the same
step.  Besides producing a flat dashboard, that pattern can leave sparse
history rows.  This module builds one ordered payload per training log event
and uses explicit optimizer-step/token axes for evaluation events that share a
training step.

Keep this module free of torch and Megatron imports so its schema and arithmetic
can be unit tested without initializing CUDA or distributed state.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from typing import Any


_LOSS_EMA: float | None = None
_LOSS_EMA_DECAY = 0.98


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def legacy_metrics_enabled() -> bool:
    """Whether to retain Megatron's old flat, per-scalar W&B calls."""

    return _env_bool("MEGATRON_WANDB_LEGACY_METRICS", False)


def rich_metrics_enabled() -> bool:
    """Whether to add cheap derived metrics to the stable W&B payload."""

    return _env_bool("MEGATRON_WANDB_RICH_METRICS", False)


def log_namespaced_metrics(
    wandb_writer: Any,
    payload: Mapping[str, Any],
    iteration: int,
) -> None:
    """Log a payload on the explicit optimizer-step axis.

    W&B's built-in ``step=`` is a global row counter.  Several independent
    Megatron probes legitimately emit rows for the same optimizer step, so
    using the built-in counter causes later probe rows to be discarded.  The
    namespaced schema instead uses ``train/optimizer_step`` as its x-axis.
    """

    if wandb_writer is None or not payload:
        return
    metrics = dict(payload)
    # The call-site iteration is authoritative.  Overwrite rather than trust a
    # stale axis value that may have been copied into a probe payload.
    metrics["train/optimizer_step"] = int(iteration)
    wandb_writer.log(metrics)


def reset_train_loss_ema() -> None:
    """Reset process-local smoothing state (normally once at ``wandb.init``)."""

    global _LOSS_EMA
    _LOSS_EMA = None


def configure_namespaced_metrics(wandb_writer: Any) -> None:
    """Register the grouped schema and its two meaningful x-axes."""

    reset_train_loss_ema()

    # W&B groups slash-delimited names in the workspace.  Alphabetically this
    # keeps loss first, followed by optimizer/performance/system/train.
    wandb_writer.define_metric("train/optimizer_step", hidden=True)
    wandb_writer.define_metric("optim/total_tokens", hidden=True)
    for namespace in (
        "act_rms",
        "act_rms_delta",
        "communication",
        "loss",
        "loop",
        "moe_probe",
        "mup",
        "optimizer",
        "optimizer_inventory",
        "perplexity",
        "optim",
        "performance",
        "slow_fast",
        "stability",
        "system",
        "train",
    ):
        wandb_writer.define_metric(
            f"{namespace}/*", step_metric="train/optimizer_step"
        )

    # The main curve is explicitly defined after the wildcard so its summary
    # and preferred axis take precedence.
    wandb_writer.define_metric(
        "loss/train", step_metric="train/optimizer_step", summary="min"
    )
    wandb_writer.define_metric(
        "loss/train_ema", step_metric="train/optimizer_step", summary="min"
    )
    wandb_writer.define_metric(
        "loss/train_vs_tokens", step_metric="optim/total_tokens", summary="min"
    )
    for split in ("validation", "test"):
        wandb_writer.define_metric(
            f"loss/{split}", step_metric="train/optimizer_step", summary="min"
        )
        wandb_writer.define_metric(
            f"perplexity/{split}", step_metric="train/optimizer_step", summary="min"
        )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "metric"


def _primary_loss(losses: Mapping[str, Any]) -> tuple[str | None, float | None]:
    if not losses:
        return None, None
    key = "lm loss" if "lm loss" in losses else next(iter(losses))
    return key, _as_float(losses[key])


def build_training_metrics(
    *,
    iteration: int,
    losses: Mapping[str, Any],
    window_losses: Mapping[str, Any] | None = None,
    learning_rate: Any,
    loss_scale: Any,
    grad_norm: Any,
    params_norm: Any,
    num_zeros_in_grad: Any,
    clip_grad: float | None,
    weight_decay: float | None,
    logical_consumed_samples: int,
    logical_batch_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    sequence_length: int,
    train_iters: int,
    elapsed_time_per_iteration: float,
    tflops_per_gpu: float,
    logical_world_size: int,
    window_iterations: int,
    skipped_iterations: int,
    nan_iterations: int,
    memory_metrics: Mapping[str, Any] | None = None,
    includes_initialization: bool = False,
) -> dict[str, float | int]:
    """Build one ordered payload for a training logging event."""

    global _LOSS_EMA

    metrics: dict[str, float | int] = {}
    primary_key, train_loss = _primary_loss(losses)
    if train_loss is not None:
        _LOSS_EMA = (
            train_loss
            if _LOSS_EMA is None
            else _LOSS_EMA_DECAY * _LOSS_EMA + (1.0 - _LOSS_EMA_DECAY) * train_loss
        )
        # Keep these first: they are the default curves a user looks for.
        metrics["loss/train"] = train_loss
        metrics["loss/train_ema"] = _LOSS_EMA
        metrics["loss/train_vs_tokens"] = train_loss
        if rich_metrics_enabled():
            metrics["perplexity/train"] = math.exp(min(20.0, train_loss))
            metrics["perplexity/train_ema"] = math.exp(min(20.0, _LOSS_EMA))
    if window_losses:
        _, window_train_loss = _primary_loss(window_losses)
        if window_train_loss is not None:
            metrics["loss/train_window"] = window_train_loss

    for key, value in losses.items():
        if key == primary_key:
            continue
        numeric_value = _as_float(value)
        if numeric_value is not None:
            metrics[f"loss/train_{_metric_slug(key)}"] = numeric_value

    total_tokens = int(logical_consumed_samples * sequence_length)
    tokens_per_update = int(logical_batch_size * sequence_length)
    metrics["train/optimizer_step"] = int(iteration)
    metrics["train/progress_fraction"] = min(1.0, iteration / max(1, train_iters))
    metrics["train/consumed_samples"] = int(logical_consumed_samples)
    metrics["train/global_batch_size"] = int(logical_batch_size)
    metrics["train/micro_batch_size_per_gpu"] = int(micro_batch_size)
    metrics["train/gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    metrics["train/sequence_length"] = int(sequence_length)
    metrics["train/tokens_per_update"] = tokens_per_update
    if rich_metrics_enabled():
        remaining_steps = max(0, int(train_iters) - int(iteration))
        metrics["train/logical_world_size"] = int(logical_world_size)
        metrics["train/remaining_steps"] = remaining_steps
        metrics["train/remaining_tokens"] = remaining_steps * tokens_per_update

    metrics["optim/total_tokens"] = total_tokens
    numeric_lr = _as_float(learning_rate)
    if numeric_lr is not None:
        metrics["optim/lr"] = numeric_lr
    numeric_grad_norm = _as_float(grad_norm)
    if numeric_grad_norm is not None:
        finite_grad = math.isfinite(numeric_grad_norm)
        metrics["optim/grad_norm"] = numeric_grad_norm
        metrics["optim/grad_norm_preclip"] = numeric_grad_norm
        metrics["optim/nonfinite_grad"] = float(not finite_grad)
        numeric_clip_grad = _as_float(clip_grad)
        if numeric_clip_grad is not None and numeric_clip_grad > 0.0:
            clip_coefficient = (
                min(1.0, numeric_clip_grad / (numeric_grad_norm + 1.0e-6))
                if finite_grad
                else 0.0
            )
            metrics["optim/grad_norm_postclip"] = (
                numeric_grad_norm * clip_coefficient if finite_grad else numeric_grad_norm
            )
            metrics["optim/clip_coefficient"] = clip_coefficient
            metrics["optim/clip_event"] = float(finite_grad and clip_coefficient < 1.0)
            metrics["optim/clip_threshold"] = numeric_clip_grad
    numeric_weight_decay = _as_float(weight_decay)
    if numeric_weight_decay is not None:
        metrics["optim/weight_decay"] = numeric_weight_decay
    numeric_loss_scale = _as_float(loss_scale)
    if numeric_loss_scale is not None:
        metrics["optim/loss_scale"] = numeric_loss_scale
    numeric_params_norm = _as_float(params_norm)
    if numeric_params_norm is not None:
        metrics["optim/params_norm"] = numeric_params_norm
    numeric_num_zeros = _as_float(num_zeros_in_grad)
    if numeric_num_zeros is not None:
        metrics["optim/num_zeros_in_grad"] = numeric_num_zeros
    metrics["optim/skipped_iterations_window"] = int(skipped_iterations)
    metrics["optim/nan_iterations_window"] = int(nan_iterations)

    iteration_seconds = max(float(elapsed_time_per_iteration), 1.0e-12)
    metrics["performance/iteration_time_ms"] = iteration_seconds * 1000.0
    metrics["performance/tokens_per_second"] = tokens_per_update / iteration_seconds
    metrics["performance/samples_per_second"] = logical_batch_size / iteration_seconds
    metrics["performance/tflops_per_gpu"] = float(tflops_per_gpu)
    metrics["performance/aggregate_pflops"] = (
        float(tflops_per_gpu) * int(logical_world_size) / 1000.0
    )
    metrics["performance/window_iterations"] = int(window_iterations)
    metrics["performance/includes_initialization"] = float(includes_initialization)
    if rich_metrics_enabled():
        metrics["performance/tokens_per_second_per_gpu"] = (
            tokens_per_update / iteration_seconds / max(1, int(logical_world_size))
        )
        metrics["performance/eta_hours_at_window_rate"] = (
            max(0, int(train_iters) - int(iteration)) * iteration_seconds / 3600.0
        )

    if memory_metrics:
        for key, value in memory_metrics.items():
            numeric_value = _as_float(value)
            if numeric_value is not None:
                metrics[key] = numeric_value

    return metrics


def build_evaluation_metrics(
    *,
    split: str,
    losses: Mapping[str, Any],
    iteration: int,
    consumed_samples: int,
    sequence_length: int,
    suffix: str = "",
) -> dict[str, float | int]:
    """Build grouped validation/test loss and perplexity metrics."""

    if split not in {"validation", "test"}:
        raise ValueError(f"unsupported evaluation split: {split}")
    label = split
    if suffix:
        label = f"{split}_{_metric_slug(suffix)}"

    metrics: dict[str, float | int] = {}
    primary_key, primary_loss = _primary_loss(losses)
    if primary_loss is not None:
        metrics[f"loss/{label}"] = primary_loss
        metrics[f"perplexity/{label}"] = math.exp(min(20.0, primary_loss))
    for key, value in losses.items():
        if key == primary_key:
            continue
        numeric_value = _as_float(value)
        if numeric_value is None:
            continue
        component = _metric_slug(key)
        metrics[f"loss/{label}_{component}"] = numeric_value
        metrics[f"perplexity/{label}_{component}"] = math.exp(min(20.0, numeric_value))

    metrics["train/optimizer_step"] = int(iteration)
    metrics["optim/total_tokens"] = int(consumed_samples * sequence_length)
    return metrics
