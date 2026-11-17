"""Explicit smoke-only checks of native Megatron gradients and optimizer updates.

Disabled by default, including formal training. No hooks, caches, or state are
installed into legacy models. These checks read actual DDP main_grad buffers.
"""
import json

import torch


@torch.no_grad()
def before_step(model, optimizer):
    records = []
    seen = set()
    for chunk in model:
        for name, module in chunk.named_modules():
            if not getattr(module, "is_kda_attention", False) or id(module) in seen:
                continue
            seen.add(id(module))
            snapshots = {}
            for key, parameter in module.named_parameters():
                gradient = getattr(parameter, "main_grad", None)
                if gradient is None:
                    raise RuntimeError(f"KDA diagnostic: missing native main_grad for {name}.{key}")
                if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                    raise RuntimeError(f"KDA diagnostic: nonfinite or zero main_grad for {name}.{key}")
                snapshots[key] = parameter.detach().clone()
            if module.A_log.dtype != torch.float32 or module.dt_bias.dtype != torch.float32:
                raise RuntimeError("Native optimizer must own FP32 KDA gate parameters.")
            records.append((name, module, snapshots))
    if len(records) != 6:
        raise RuntimeError(f"Expected six distinct, twice-shared KDA modules, found {len(records)}")
    active_lr = max(float(group["lr"]) for group in optimizer.param_groups)
    return records, active_lr


@torch.no_grad()
def after_step(evidence, successful, iteration):
    records, active_lr = evidence
    if not successful:
        raise RuntimeError("KDA smoke optimizer skipped the update.")
    counts = {}
    for name, module, snapshots in records:
        changed = []
        for key, parameter in module.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"KDA diagnostic: nonfinite updated parameter {name}.{key}")
            if not torch.equal(snapshots[key], parameter):
                changed.append(key)
        # The first warmup update can legitimately have LR=0. At positive LR
        # require a real projection AND gate update, not weight decay alone
        # being mistaken for evidence: all main_grad values were checked above.
        if active_lr > 0:
            if not any(key.startswith(("q_proj.", "k_proj.", "v_proj.", "o_proj.")) for key in changed):
                raise RuntimeError(f"KDA diagnostic: optimizer did not update projections in {name}")
            if not any(key.startswith(("f_proj.", "g_proj.", "b_proj.")) or key in ("A_log", "dt_bias") for key in changed):
                raise RuntimeError(f"KDA diagnostic: optimizer did not update gates in {name}")
        counts[name] = len(changed)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print("KDA_NATIVE_OPTIMIZER " + json.dumps(dict(
        status="PASS", rank=rank, iteration=iteration, learning_rate=active_lr,
        distinct_modules=len(records), main_grad_verified=True,
        positive_lr_updates_verified=active_lr > 0, changed_tensors=counts,
        max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_total_memory_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        memory_includes_smoke_parameter_snapshots=True,
    ), sort_keys=True), flush=True)
