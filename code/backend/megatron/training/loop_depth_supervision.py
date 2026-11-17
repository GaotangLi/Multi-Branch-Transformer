# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Loss utilities for recurrent-depth language-model supervision."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from megatron.core.transformer.loop_depth import active_loop_depth_indices


def loop_layer_execution_counts(
    num_layers: int,
    loop_layer_start: int | None,
    loop_layer_end: int | None,
    loop_repeats: int,
    depth_supervision: bool,
    active_depth_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Count executions of every physical layer, including auxiliary codas.

    Loop bounds are one-based and inclusive.  A supervised non-final exit adds
    a coda traversal but never another prelude or loop-body traversal.
    """

    if num_layers < 1:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if loop_repeats < 1:
        raise ValueError(f"loop_repeats must be positive, got {loop_repeats}")
    counts = [1] * num_layers
    if loop_repeats == 1:
        if depth_supervision:
            raise ValueError("depth supervision requires loop_repeats > 1")
        return tuple(counts)
    if loop_layer_start is None or loop_layer_end is None:
        raise ValueError("loop bounds are required when loop_repeats > 1")
    if not 1 <= loop_layer_start <= loop_layer_end <= num_layers:
        raise ValueError(
            "loop bounds must satisfy 1 <= start <= end <= num_layers, got "
            f"{loop_layer_start}, {loop_layer_end}, {num_layers}"
        )

    extra_visits = loop_repeats - 1
    for layer_index in range(loop_layer_start - 1, loop_layer_end):
        counts[layer_index] += extra_visits
    if depth_supervision:
        active_depths = (
            tuple(range(loop_repeats))
            if active_depth_indices is None
            else tuple(active_depth_indices)
        )
        if not active_depths:
            raise ValueError("depth supervision requires at least one active exit")
        if len(set(active_depths)) != len(active_depths) or any(
            depth < 0 or depth >= loop_repeats for depth in active_depths
        ):
            raise ValueError(
                "active_depth_indices must be unique zero-based loop visits, got "
                f"{active_depths} for loop_repeats={loop_repeats}"
            )
        for layer_index in range(loop_layer_end, num_layers):
            # The recurrent trunk ends before Coda.  Run Coda exactly once for
            # each non-zero loss exit, including or excluding the final visit
            # according to its actual objective weight.
            counts[layer_index] = len(active_depths)
    return tuple(counts)


def loop_depth_loss_weights(
    num_depths: int,
    aux_loss_weight: float,
    explicit_weights: Sequence[float] | None = None,
    *,
    device=None,
    dtype=None,
) -> Tensor:
    """Return normalized weights for the no-loop exit and deeper exits.

    Explicit weights, when supplied, take precedence and are normalized to
    sum to one. Otherwise the first exit (one visit, hence the ordinary
    no-loop physical stack) has unnormalized weight one and
    ``aux_loss_weight`` is spread uniformly over all deeper recurrent exits.
    """

    if num_depths < 2:
        raise ValueError(f"loop depth supervision requires at least two depths, got {num_depths}")
    if explicit_weights is not None:
        values = [float(weight) for weight in explicit_weights]
        if len(values) != num_depths:
            raise ValueError(
                "explicit_weights must contain one weight per depth; "
                f"got {len(values)} weights for {num_depths} depths"
            )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in values):
            raise ValueError(f"explicit_weights must be finite and non-negative, got {values}")
        denominator = sum(values)
        if not denominator > 0.0:
            raise ValueError("explicit_weights must have a positive sum")
        return torch.tensor(values, device=device, dtype=dtype) / denominator
    if not math.isfinite(aux_loss_weight) or aux_loss_weight < 0.0:
        raise ValueError(
            f"aux_loss_weight must be finite and non-negative, got {aux_loss_weight}"
        )

    denominator = 1.0 + aux_loss_weight
    first_weight = 1.0 / denominator
    recurrent_weight = aux_loss_weight / ((num_depths - 1) * denominator)
    return torch.tensor(
        [first_weight] + [recurrent_weight] * (num_depths - 1),
        device=device,
        dtype=dtype,
    )


def combine_loop_depth_losses(
    depth_losses: Tensor,
    aux_loss_weight: float,
    explicit_weights: Sequence[float] | None = None,
) -> Tensor:
    """Combine ``[depth, ...]`` per-token losses into a baseline-shaped tensor."""

    if depth_losses.ndim < 2:
        raise ValueError(
            "depth_losses must have a leading depth axis plus at least one token axis; "
            f"got shape {tuple(depth_losses.shape)}"
        )
    weights = loop_depth_loss_weights(
        depth_losses.shape[0],
        aux_loss_weight,
        explicit_weights,
        device=depth_losses.device,
        dtype=depth_losses.dtype,
    )
    view_shape = (weights.shape[0],) + (1,) * (depth_losses.ndim - 1)
    return torch.sum(depth_losses * weights.view(view_shape), dim=0)


def combine_active_loop_depth_losses(
    active_depth_losses: Tensor,
    num_depths: int,
    aux_loss_weight: float,
    explicit_weights: Sequence[float] | None = None,
) -> Tensor:
    """Combine losses computed only for non-zero-weight loop exits.

    This is the execution-aware counterpart of :func:`combine_loop_depth_losses`.
    Zero-weight exits never need a Coda or vocabulary projection, so their loss
    tensors are intentionally absent rather than materialized as zeros.
    """

    if active_depth_losses.ndim < 2:
        raise ValueError(
            "active_depth_losses must have a leading exit axis plus at least one token axis; "
            f"got shape {tuple(active_depth_losses.shape)}"
        )
    active_indices = active_loop_depth_indices(
        num_depths,
        aux_loss_weight,
        explicit_weights,
    )
    if active_depth_losses.shape[0] != len(active_indices):
        raise ValueError(
            "active loss count does not match non-zero loop-depth weights; got "
            f"{active_depth_losses.shape[0]} losses for active depths "
            f"{tuple(index + 1 for index in active_indices)}"
        )
    full_weights = loop_depth_loss_weights(
        num_depths,
        aux_loss_weight,
        explicit_weights,
        device=active_depth_losses.device,
        dtype=active_depth_losses.dtype,
    )
    active_weights = full_weights[list(active_indices)]
    view_shape = (active_weights.shape[0],) + (1,) * (active_depth_losses.ndim - 1)
    return torch.sum(active_depth_losses * active_weights.view(view_shape), dim=0)


def ensemble_active_loop_depth_losses(
    active_depth_losses: Tensor,
    num_depths: int,
    aux_loss_weight: float,
    explicit_weights: Sequence[float] | None = None,
) -> Tensor:
    """Return exact per-token NLL of the weighted exit-probability mixture.

    Each input value is already ``-log Q_k(y)`` for the target token.  The
    mixture therefore needs only target probabilities, not materialized
    vocabulary distributions::

        -log(sum_k w_k * Q_k(y))
        = -logsumexp_k(log(w_k) - loss_k(y)).

    The computation is distinct from the training objective
    ``sum_k w_k * loss_k`` and is numerically stable for very small target
    probabilities.  Zero-weight exits are omitted by the execution-aware
    multi-exit path.
    """

    if active_depth_losses.ndim < 2:
        raise ValueError(
            "active_depth_losses must have a leading exit axis plus at least one token axis; "
            f"got shape {tuple(active_depth_losses.shape)}"
        )
    active_indices = active_loop_depth_indices(
        num_depths,
        aux_loss_weight,
        explicit_weights,
    )
    if active_depth_losses.shape[0] != len(active_indices):
        raise ValueError(
            "active loss count does not match non-zero loop-depth weights; got "
            f"{active_depth_losses.shape[0]} losses for active depths "
            f"{tuple(index + 1 for index in active_indices)}"
        )
    full_weights = loop_depth_loss_weights(
        num_depths,
        aux_loss_weight,
        explicit_weights,
        device=active_depth_losses.device,
        dtype=active_depth_losses.dtype,
    )
    active_weights = full_weights[list(active_indices)]
    view_shape = (active_weights.shape[0],) + (1,) * (active_depth_losses.ndim - 1)
    return -torch.logsumexp(
        torch.log(active_weights).view(view_shape) - active_depth_losses,
        dim=0,
    )
