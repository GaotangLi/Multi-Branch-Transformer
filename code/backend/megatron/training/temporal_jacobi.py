# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Objective and accounting utilities for causal temporal Jacobi passes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def temporal_jacobi_loss_weights(
    num_passes: int,
    explicit_weights: Sequence[float],
    *,
    device=None,
    dtype=None,
) -> Tensor:
    """Validate and normalize one non-negative NTP weight per temporal pass."""

    if num_passes < 2:
        raise ValueError(f"temporal Jacobi requires at least two passes, got {num_passes}")
    values = [float(weight) for weight in explicit_weights]
    if len(values) != num_passes:
        raise ValueError(
            "explicit_weights must contain one weight per temporal pass; "
            f"got {len(values)} weights for {num_passes} passes"
        )
    if any(not math.isfinite(weight) or weight < 0.0 for weight in values):
        raise ValueError(f"explicit_weights must be finite and non-negative, got {values}")
    denominator = sum(values)
    if not denominator > 0.0:
        raise ValueError("explicit_weights must have a positive sum")
    if not values[-1] > 0.0:
        raise ValueError("the final temporal Jacobi pass must have positive weight")
    return torch.tensor(values, device=device, dtype=dtype) / denominator


def active_temporal_jacobi_pass_indices(
    num_passes: int, explicit_weights: Sequence[float]
) -> tuple[int, ...]:
    """Return zero-based passes whose codas/heads must be executed."""

    weights = temporal_jacobi_loss_weights(num_passes, explicit_weights)
    return tuple(index for index, weight in enumerate(weights.tolist()) if weight > 0.0)


def combine_temporal_jacobi_losses(
    active_pass_losses: Tensor,
    num_passes: int,
    explicit_weights: Sequence[float],
) -> Tensor:
    """Combine losses from positive-weight exits into a baseline-shaped tensor."""

    if active_pass_losses.ndim < 2:
        raise ValueError(
            "active_pass_losses must have an exit axis plus at least one token axis; "
            f"got shape {tuple(active_pass_losses.shape)}"
        )
    full_weights = temporal_jacobi_loss_weights(
        num_passes,
        explicit_weights,
        device=active_pass_losses.device,
        dtype=active_pass_losses.dtype,
    )
    active_weights = full_weights[full_weights > 0.0]
    if active_pass_losses.shape[0] != active_weights.shape[0]:
        raise ValueError(
            "active loss count does not match positive temporal pass weights; "
            f"got {active_pass_losses.shape[0]} losses and {active_weights.shape[0]} weights"
        )
    view_shape = (active_weights.shape[0],) + (1,) * (active_pass_losses.ndim - 1)
    return torch.sum(active_pass_losses * active_weights.view(view_shape), dim=0)


def temporal_jacobi_layer_execution_counts(
    num_layers: int,
    loop_layer_start: int,
    loop_layer_end: int,
    loop_repeats: int,
    temporal_passes: int,
    explicit_weights: Sequence[float],
) -> tuple[int, ...]:
    """Count shared physical-layer executions across all temporal passes."""

    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if loop_repeats < 2:
        raise ValueError(f"temporal Jacobi requires loop_repeats >= 2, got {loop_repeats}")
    if not 1 <= loop_layer_start <= loop_layer_end < num_layers:
        raise ValueError(
            "loop bounds must leave a non-empty coda and satisfy "
            "1 <= start <= end < num_layers"
        )
    active_exits = len(active_temporal_jacobi_pass_indices(temporal_passes, explicit_weights))
    prelude_width = loop_layer_start - 1
    loop_width = loop_layer_end - loop_layer_start + 1
    coda_width = num_layers - loop_layer_end
    return (
        (temporal_passes,) * prelude_width
        + (temporal_passes * loop_repeats,) * loop_width
        + (active_exits,) * coda_width
    )
