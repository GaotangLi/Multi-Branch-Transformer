# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small, dependency-free helpers for recurrent loop-depth execution."""

from __future__ import annotations

import math
from collections.abc import Sequence


_UINT64_MASK = (1 << 64) - 1


def sample_uniform_loop_repeats(seed: int, optimizer_step: int, max_repeats: int) -> int:
    """Sample a reproducible uniform loop depth without touching training RNG state.

    ``optimizer_step`` is zero based.  The SplitMix64 finalizer makes every data
    parallel rank choose the same value from only globally shared scalars, while
    leaving CUDA/Python RNG streams (and therefore model stochasticity) unchanged.
    """

    if optimizer_step < 0:
        raise ValueError(f"optimizer_step must be non-negative, got {optimizer_step}")
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be positive, got {max_repeats}")

    value = (int(seed) + (optimizer_step + 1) * 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return 1 + value % max_repeats


def sampled_loop_depth_counts(
    seed: int,
    start_optimizer_step: int,
    num_optimizer_steps: int,
    max_repeats: int,
) -> tuple[int, ...]:
    """Return exact realized counts for a contiguous deterministic sample window."""

    if start_optimizer_step < 0:
        raise ValueError(
            f"start_optimizer_step must be non-negative, got {start_optimizer_step}"
        )
    if num_optimizer_steps < 0:
        raise ValueError(f"num_optimizer_steps must be non-negative, got {num_optimizer_steps}")
    if max_repeats < 1:
        raise ValueError(f"max_repeats must be positive, got {max_repeats}")

    counts = [0] * max_repeats
    for optimizer_step in range(
        start_optimizer_step, start_optimizer_step + num_optimizer_steps
    ):
        depth = sample_uniform_loop_repeats(seed, optimizer_step, max_repeats)
        counts[depth - 1] += 1
    return tuple(counts)


def loop_dual_r1_transition_iteration(train_iters: int, r1_fraction: float) -> int:
    """Return the first optimizer-step index that executes the full R2 objective.

    The transition is derived only from the immutable training horizon and an
    absolute, zero-based optimizer-step index.  Consequently a checkpoint resume
    does not need to restore any additional curriculum state.
    """

    if train_iters < 1:
        raise ValueError(f"train_iters must be positive, got {train_iters}")
    if not math.isfinite(r1_fraction) or not 0.0 <= r1_fraction < 1.0:
        raise ValueError(
            "loop_dual_r1_fraction must be finite and lie in [0, 1), got "
            f"{r1_fraction}"
        )
    return math.floor(train_iters * r1_fraction)


def loop_dual_r1_is_active(
    train_iters: int,
    r1_fraction: float,
    optimizer_step: int,
) -> bool:
    """Whether one zero-based optimizer step belongs to the R1 curriculum prefix."""

    if optimizer_step < 0 or optimizer_step >= train_iters:
        raise ValueError(
            "optimizer_step must lie in [0, train_iters); "
            f"got optimizer_step={optimizer_step}, train_iters={train_iters}"
        )
    return optimizer_step < loop_dual_r1_transition_iteration(train_iters, r1_fraction)


def loop_dual_runtime_exit_indices(
    train_iters: int,
    r1_fraction: float,
    optimizer_step: int,
    *,
    training: bool,
) -> tuple[int, ...] | None:
    """Select the R1 exit during the training prefix and full R2 otherwise.

    ``None`` means the configured R2 exits are retained.  Evaluation is always
    deterministic full R2, irrespective of the current training step.
    """

    if not training or r1_fraction == 0.0:
        # Validate the static contract even when no runtime override is needed.
        loop_dual_r1_transition_iteration(train_iters, r1_fraction)
        return None
    return (
        (0,)
        if loop_dual_r1_is_active(train_iters, r1_fraction, optimizer_step)
        else None
    )


def loop_dual_r1_step_counts(
    train_iters: int,
    r1_fraction: float,
    start_optimizer_step: int,
    num_optimizer_steps: int,
) -> tuple[int, int]:
    """Count R1 and R2 updates in a contiguous, half-open logging window."""

    transition = loop_dual_r1_transition_iteration(train_iters, r1_fraction)
    if start_optimizer_step < 0:
        raise ValueError(
            f"start_optimizer_step must be non-negative, got {start_optimizer_step}"
        )
    if num_optimizer_steps < 0:
        raise ValueError(
            f"num_optimizer_steps must be non-negative, got {num_optimizer_steps}"
        )
    end_optimizer_step = start_optimizer_step + num_optimizer_steps
    if end_optimizer_step > train_iters:
        raise ValueError(
            "curriculum window extends past train_iters; "
            f"got [{start_optimizer_step}, {end_optimizer_step}) for {train_iters}"
        )
    r1_steps = max(0, min(end_optimizer_step, transition) - start_optimizer_step)
    return r1_steps, num_optimizer_steps - r1_steps


def active_loop_depth_indices(
    num_depths: int,
    aux_loss_weight: float,
    explicit_weights: Sequence[float] | None = None,
) -> tuple[int, ...]:
    """Return zero-based loop visits whose NTP objective has non-zero weight.

    The recurrent trunk must execute every visit regardless of this result.
    Callers use the returned indices only to decide which visit states need a
    Coda traversal and language-model head.
    """

    if num_depths < 2:
        raise ValueError(f"loop depth supervision requires at least two depths, got {num_depths}")

    if explicit_weights is not None:
        weights = [float(weight) for weight in explicit_weights]
        if len(weights) != num_depths:
            raise ValueError(
                "explicit_weights must contain one weight per depth; "
                f"got {len(weights)} weights for {num_depths} depths"
            )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError(f"explicit_weights must be finite and non-negative, got {weights}")
        if not sum(weights) > 0.0:
            raise ValueError("explicit_weights must have a positive sum")
    else:
        if not math.isfinite(aux_loss_weight) or aux_loss_weight < 0.0:
            raise ValueError(
                f"aux_loss_weight must be finite and non-negative, got {aux_loss_weight}"
            )
        weights = [1.0] + [aux_loss_weight] * (num_depths - 1)

    return tuple(index for index, weight in enumerate(weights) if weight > 0.0)
