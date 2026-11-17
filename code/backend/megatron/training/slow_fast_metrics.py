# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sparse diagnostics for loop-specific slow/fast FFN adapters."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from megatron.core.transformer.mlp import MLP, SlowFastDownProjectionAdapter


def get_slow_fast_optimizer_group(optimizer: Any, args: Any) -> dict[str, Any] | None:
    """Find the dedicated adapter group before or after optimizer param remapping."""

    if optimizer is None:
        return None
    expected_max_lr = float(args.lr) * float(args.slow_fast_ffn_fast_lr_multiplier)
    for group in optimizer.param_groups:
        if any(
            getattr(parameter, "is_slow_fast_adapter_parameter", False)
            for parameter in group.get("params", ())
        ):
            return group
        # DistributedOptimizer may replace model Parameters with fp32 shards,
        # which do not retain custom Python attributes. The explicit LR
        # override still makes this a non-default, zero-decay schedule group.
        if (
            not group.get("default_config", True)
            and math.isclose(float(group.get("max_lr", math.nan)), expected_max_lr)
            and math.isclose(float(group.get("wd_mult", math.nan)), 0.0)
        ):
            return group
    return None


def _product_singular_values(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Compute singular values of ``left @ right`` without materializing it."""

    # left: [m, r], right: [r, n]. Thin QR reduces the SVD to at most r x r.
    _, left_r = torch.linalg.qr(left.float(), mode="reduced")
    _, right_r = torch.linalg.qr(right.float().T, mode="reduced")
    return torch.linalg.svdvals(left_r @ right_r.T)


def _adapter_statistics(
    adapter: SlowFastDownProjectionAdapter,
    base_weight: torch.Tensor,
    iteration: int,
) -> dict[str, float]:
    current_a = adapter.lora_a.detach()
    current_b = adapter.lora_b.detach()
    previous_a = adapter._audit_previous_a
    previous_b = adapter._audit_previous_b

    # Audit snapshots are not checkpoint state. On resume, avoid comparing the
    # loaded adapter with the random tensors that existed before checkpoint load.
    resumed_without_baseline = adapter._audit_previous_iteration == 0 and iteration > 1
    if previous_a is None or previous_b is None or resumed_without_baseline:
        previous_a = current_a
        previous_b = current_b

    scale = adapter.scaling
    update_singular = _product_singular_values(current_b * scale, current_a)
    change_left = torch.cat((current_b * scale, -previous_b * scale), dim=1)
    change_right = torch.cat((current_a, previous_a), dim=0)
    change_singular = _product_singular_values(change_left, change_right)

    update_fro = torch.linalg.vector_norm(update_singular)
    change_fro = torch.linalg.vector_norm(change_singular)
    spectral = update_singular[0] if update_singular.numel() else update_fro
    stable_rank = update_fro.square() / spectral.square().clamp_min(torch.finfo(torch.float32).tiny)
    base_fro = torch.linalg.vector_norm(base_weight.detach().float()).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    values = torch.stack(
        (
            update_fro,
            update_fro / base_fro,
            change_fro,
            change_fro / base_fro,
            spectral,
            stable_rank,
            current_a.float().square().mean().sqrt(),
            current_b.float().square().mean().sqrt(),
        )
    ).cpu().tolist()

    adapter._audit_previous_a = current_a.clone()
    adapter._audit_previous_b = current_b.clone()
    adapter._audit_previous_iteration = iteration
    keys = (
        "update_fro",
        "update_over_base_fro",
        "change_fro",
        "change_over_base_fro",
        "spectral_norm",
        "stable_rank",
        "a_rms",
        "b_rms",
    )
    return dict(zip(keys, values))


def collect_slow_fast_metrics(model: Iterable[torch.nn.Module], iteration: int) -> dict[str, float]:
    """Collect layer-wise and aggregate effective-fast-weight diagnostics."""

    layer_metrics: dict[int, dict[str, float]] = {}
    for model_chunk in model:
        for module in model_chunk.modules():
            if not isinstance(module, MLP) or module.slow_fast_adapter is None:
                continue
            adapter = module.slow_fast_adapter
            if adapter.layer_number in layer_metrics:
                raise RuntimeError(
                    f"Duplicate slow/fast adapter for physical layer {adapter.layer_number}."
                )
            layer_metrics[adapter.layer_number] = _adapter_statistics(
                adapter, module.linear_fc2.weight, iteration
            )

    payload: dict[str, float] = {}
    for layer_number, metrics in sorted(layer_metrics.items()):
        prefix = f"slow_fast/layer_{layer_number:02d}"
        payload.update({f"{prefix}/{name}": value for name, value in metrics.items()})

    if layer_metrics:
        for metric_name in next(iter(layer_metrics.values())):
            values = [metrics[metric_name] for metrics in layer_metrics.values()]
            payload[f"slow_fast/aggregate/{metric_name}_mean"] = sum(values) / len(values)
            payload[f"slow_fast/aggregate/{metric_name}_max"] = max(values)
    return payload


def maybe_log_slow_fast_metrics(
    *,
    model: Iterable[torch.nn.Module],
    args: Any,
    iteration: int,
    writer: Any,
    wandb_writer: Any,
    optimizer: Any,
) -> None:
    """Log an audit event when due; otherwise add no training-path work."""

    if not getattr(args, "slow_fast_ffn", False):
        return
    interval = int(args.slow_fast_ffn_audit_interval)
    train_iters = int(args.train_iters or iteration)
    if iteration != 1 and iteration % interval != 0 and iteration != train_iters:
        return
    if writer is None and wandb_writer is None:
        return

    payload = collect_slow_fast_metrics(model, iteration)
    payload["train/optimizer_step"] = iteration
    payload["optim/total_tokens"] = int(args.consumed_train_samples * args.seq_length)
    payload["slow_fast/config/rank"] = int(args.slow_fast_ffn_rank)
    payload["slow_fast/config/alpha"] = float(args.slow_fast_ffn_alpha)
    payload["slow_fast/config/fast_lr_multiplier"] = float(
        args.slow_fast_ffn_fast_lr_multiplier
    )
    payload["slow_fast/config/slow_end_visit"] = int(args.slow_fast_ffn_slow_end_visit)
    payload["slow_fast/config/fast_start_visit"] = int(args.slow_fast_ffn_fast_start_visit)
    fast_group = get_slow_fast_optimizer_group(optimizer, args)
    if fast_group is not None:
        payload["slow_fast/optim/lr"] = float(fast_group["lr"])
        payload["slow_fast/optim/weight_decay"] = float(fast_group.get("weight_decay", 0.0))

    if writer is not None:
        for name, value in payload.items():
            writer.add_scalar(name, value, iteration)
    if wandb_writer is not None:
        # All curves use the explicit optimizer-step metric configured at init.
        wandb_writer.log(payload)
