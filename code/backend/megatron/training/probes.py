"""Training, stability, and system observability probes.

Lightweight metrics, intentionally small and dependency-free:

1. ``moe_probe/*/maxvio`` (log cadence)  - per-layer MoE overload
   violation, computed as ``max(tokens_per_expert) / mean(tokens_per_expert) - 1``.

2. ``moe_probe/summary/maxvio_mean`` (log cadence)  - mean maxvio across
   MoE layers.

3. ``layer_activation_rms`` (per eval)  - RMS of each transformer layer's
   output, measured on the validation batch via short-lived forward hooks.

4. ``layer_rms_delta`` (per eval)  - absolute difference and ratio of RMS
   between adjacent layers.

5. ``system/*`` and ``communication/*`` (log cadence) - step skew,
   process/GPU memory, NIC counters, collective payloads, and deferred CUDA
   event timings for communication lifetime and exposed waits.

Design notes:
- Metric 1 reads the routing map directly off the ``TopKRouter`` instances by
  registering a tiny forward hook. The
  hook body is one ``sum`` per layer per microbatch -- well under 0.01%
  of step compute.
- Metrics 3/4 only run during evaluation. Hooks are registered before the
  eval loop and removed after, so training fwd has zero overhead.
- All reductions across data-parallel ranks are int/float sums (or sum of
  squares) accumulated locally and reduced once per logging event.
- State lives on a module-level singleton (``_PROBES``); no global side
  effects until ``enable_probes`` is called.
"""

from __future__ import annotations

import logging
import math
import os
import resource
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from megatron.core import communication_probes
from megatron.training.wandb_metrics import log_namespaced_metrics


logger = logging.getLogger(__name__)


@dataclass
class _RouterState:
    """Per-router accumulator for metric 1."""

    counts: torch.Tensor  # int64 [num_experts], on cuda
    router: object = None
    hook_handle: object = None


@dataclass
class _RMSState:
    """Per-layer accumulator for metric 2."""

    sum_sq: torch.Tensor  # float64 scalar on cuda
    n: torch.Tensor  # int64 scalar on cuda
    hook_handle: object = None


@dataclass
class _BranchState:
    """Per-module eval accumulator for output RMS and outlier percentage."""

    kind: str
    layer_number: int | str
    sum_sq: torch.Tensor
    n: torch.Tensor
    outliers: torch.Tensor
    outlier_n: torch.Tensor
    hook_handle: object = None


@dataclass
class _RouterZState:
    """Per-router eval accumulator for log-sum-exp squared and logit scale."""

    layer_number: int | str
    z_sum: torch.Tensor
    z_n: torch.Tensor
    logit_sum_sq: torch.Tensor
    logit_n: torch.Tensor
    logit_absmax: torch.Tensor
    hook_handle: object = None


@dataclass
class _LogitState:
    """Eval accumulator for final LM logits."""

    name: str
    sum_sq: torch.Tensor
    n: torch.Tensor
    absmax: torch.Tensor
    hook_handle: object = None


@dataclass
class _ParamProbeEntry:
    name: str
    bucket: str
    optimizer_name: str
    group_index: int
    param: torch.nn.Parameter


@dataclass
class _SystemProbeState:
    started: bool = False
    paused: bool = False
    network_discovery_attempted: bool = False
    network_sources: tuple = ()
    network_source_names: tuple[str, ...] = ()
    wall_time: float = 0.0
    cpu_time: float = 0.0
    minor_faults: int = 0
    major_faults: int = 0
    network_tx_bytes: int = 0
    network_rx_bytes: int = 0
    cuda_alloc_retries: int = 0
    cuda_ooms: int = 0
    accumulated_wall_time: float = 0.0
    accumulated_cpu_time: float = 0.0
    accumulated_minor_faults: int = 0
    accumulated_major_faults: int = 0
    accumulated_network_tx_bytes: int = 0
    accumulated_network_rx_bytes: int = 0
    accumulated_cuda_alloc_retries: int = 0
    accumulated_cuda_ooms: int = 0


@dataclass
class _Probes:
    enabled: bool = False
    # Routers: one entry per MoE layer.
    routers: list = field(default_factory=list)  # list[_RouterState]
    router_layer_indices: list = field(default_factory=list)  # global layer_number per router
    num_experts: int = 0
    # RMS: one entry per TransformerLayer.
    layers: list = field(default_factory=list)  # list[_RMSState]
    branches: list = field(default_factory=list)  # list[_BranchState]
    router_z: list = field(default_factory=list)  # list[_RouterZState]
    logits: list = field(default_factory=list)  # list[_LogitState]
    param_probe_enabled: bool = False
    param_entries: list = field(default_factory=list)  # list[_ParamProbeEntry]
    optimizer_groups: list = field(default_factory=list)
    optimizer_inventory_logged: bool = False
    trainable_param_numel: int = 0
    trainable_param_bytes: int = 0
    system: _SystemProbeState = field(default_factory=_SystemProbeState)


_PROBES = _Probes()


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------


def _iter_modules(model):
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        for m in chunk.modules():
            yield m


def _iter_named_modules(model):
    chunks = model if isinstance(model, list) else [model]
    for chunk_i, chunk in enumerate(chunks):
        prefix = f"model_chunk_{chunk_i}"
        for name, module in chunk.named_modules():
            yield f"{prefix}.{name}" if name else prefix, module


def enable_probes(model) -> None:
    """Install metric-1 router hooks. Idempotent; safe to call once at startup.

    ``model`` is the list of model chunks returned by ``get_model``.
    """
    from megatron.core.transformer.moe.router import TopKRouter

    if _PROBES.enabled:
        return

    for module in _iter_modules(model):
        if isinstance(module, TopKRouter):
            num_experts = module.config.num_moe_experts
            counts = torch.zeros(num_experts, dtype=torch.int64, device="cuda")
            state = _RouterState(counts=counts, router=module)

            def _make_hook(s: _RouterState):
                def _hook(_mod, _inp, out):
                    # out = (probs, routing_map). routing_map is [tokens, num_experts] bool/int.
                    routing_map = out[1]
                    s.counts.add_(routing_map.sum(dim=0).to(torch.int64))

                return _hook

            state.hook_handle = module.register_forward_hook(_make_hook(state))
            _PROBES.routers.append(state)
            _PROBES.router_layer_indices.append(
                getattr(module, "layer_number", len(_PROBES.routers))
            )
            _PROBES.num_experts = num_experts

    _PROBES.enabled = True


def _optimizer_name_from_group(group: dict) -> str:
    name = group.get("optimizer")
    if name is not None:
        return str(name)
    return "default"


def _sanitize_tag_part(value: object) -> str:
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    tag = "".join(out).strip("_")
    return tag or "unknown"


def _param_bucket(name: str, param: torch.nn.Parameter) -> str:
    lower = name.lower()
    if getattr(param, "is_embedding_or_output_parameter", False) or "output_layer" in lower:
        if "embedding" in lower:
            return "embedding"
        return "lm_head"
    if "embedding" in lower:
        return "embedding"
    if "router" in lower:
        return "moe_router"
    if "shared_expert" in lower:
        return "moe_shared_expert"
    if "expert" in lower:
        return "moe_expert"
    if "fc1_latent" in lower or "fc2_latent" in lower or "latent" in lower:
        return "moe_latent"
    if "linear_qkv" in lower or "query_key_value" in lower:
        return "attn_qkv"
    if "linear_proj" in lower:
        return "attn_proj"
    if "q_layernorm" in lower or "k_layernorm" in lower or "qk" in lower:
        return "attn_qk_norm"
    if "self_attention" in lower or "attention" in lower:
        return "attention_other"
    if "linear_fc" in lower or ".mlp." in lower or "mlp" in lower:
        return "mlp"
    if "norm" in lower or "layernorm" in lower:
        return "norm"
    if param.dim() <= 1:
        return "vector"
    return "other_matrix" if param.dim() == 2 else "other"


def enable_optimizer_probes(model, optimizer) -> None:
    """Collect parameter/group metadata used by cheap log-cadence probes."""
    if optimizer is None or getattr(optimizer, "is_stub_optimizer", False):
        return
    if _PROBES.param_probe_enabled:
        return

    param_to_group: dict[int, tuple[int, str]] = {}
    _PROBES.optimizer_groups.clear()
    for group_i, group in enumerate(getattr(optimizer, "param_groups", []) or []):
        optimizer_name = _optimizer_name_from_group(group)
        numel = 0
        for param in group.get("params", []):
            param_to_group[id(param)] = (group_i, optimizer_name)
            numel += param.numel()
        _PROBES.optimizer_groups.append((group_i, optimizer_name, group, numel))

    for chunk_i, chunk in enumerate(model if isinstance(model, list) else [model]):
        for name, param in chunk.named_parameters():
            if not param.requires_grad:
                continue
            group_i, optimizer_name = param_to_group.get(id(param), (-1, "unassigned"))
            full_name = f"model_chunk_{chunk_i}.{name}"
            _PROBES.param_entries.append(
                _ParamProbeEntry(
                    name=full_name,
                    bucket=_param_bucket(full_name, param),
                    optimizer_name=optimizer_name,
                    group_index=group_i,
                    param=param,
                )
            )
            _PROBES.trainable_param_numel += param.numel()
            _PROBES.trainable_param_bytes += param.numel() * param.element_size()

    _PROBES.param_probe_enabled = bool(_PROBES.param_entries or _PROBES.optimizer_groups)


def _to_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_payload(writer, wandb_writer, payload: dict[str, float], iteration: int) -> None:
    if not payload:
        return
    if writer is not None:
        for tag, value in payload.items():
            writer.add_scalar(tag, value, iteration)
    if wandb_writer is not None:
        log_namespaced_metrics(wandb_writer, payload, iteration)


def _param_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    grad = getattr(param, "main_grad", None)
    if grad is None:
        grad = param.grad
    return grad


def log_training_step_metrics(
    iteration: int,
    writer,
    wandb_writer,
    *,
    grad_norm,
    skipped_iter,
    clip_grad: float | None,
    loss_scale: float | None,
) -> None:
    """Log cheap optimizer health and static parameter inventory."""
    if writer is None and wandb_writer is None:
        return
    if not _PROBES.param_probe_enabled:
        return

    payload: dict[str, float] = {}
    grad_norm_value = _to_float(grad_norm)
    clip_grad_value = _to_float(clip_grad)
    if grad_norm_value is not None:
        payload["optimizer/grad_norm_pre_clip"] = grad_norm_value
        if clip_grad_value is not None and clip_grad_value > 0:
            payload["optimizer/clip_coeff"] = min(1.0, clip_grad_value / max(grad_norm_value, 1e-12))
    if loss_scale is not None:
        payload["optimizer/loss_scale"] = float(loss_scale)
    payload["optimizer/skipped_iter"] = float(skipped_iter)

    lr_values = []
    wd_values = []
    eps_values = []
    beta1_values = []
    beta2_values = []
    for group_i, optimizer_name, group, numel in _PROBES.optimizer_groups:
        opt_tag = _sanitize_tag_part(optimizer_name)
        group_tag = f"optimizer/group_{group_i}_{opt_tag}"
        lr = _to_float(group.get("lr", group.get("max_lr")))
        wd = _to_float(group.get("weight_decay", group.get("wd", group.get("weight_decay_rate"))))
        eps = _to_float(group.get("eps"))
        betas = group.get("betas")
        beta1 = beta2 = None
        if isinstance(betas, (tuple, list)) and len(betas) >= 2:
            beta1 = _to_float(betas[0])
            beta2 = _to_float(betas[1])
        beta1 = _to_float(group.get("beta1", beta1))
        beta2 = _to_float(group.get("beta2", beta2))
        if lr is not None:
            payload[f"{group_tag}/lr"] = lr
            lr_values.append(lr)
        if wd is not None:
            payload[f"{group_tag}/weight_decay"] = wd
            wd_values.append(wd)
        if eps is not None:
            payload[f"{group_tag}/eps"] = eps
            eps_values.append(eps)
        if beta1 is not None:
            payload[f"{group_tag}/beta1"] = beta1
            beta1_values.append(beta1)
        if beta2 is not None:
            payload[f"{group_tag}/beta2"] = beta2
            beta2_values.append(beta2)
        if not _PROBES.optimizer_inventory_logged:
            payload[f"{group_tag}/param_count"] = float(numel)

    for tag, values in (
        ("optimizer/lr", lr_values),
        ("optimizer/weight_decay", wd_values),
        ("optimizer/eps", eps_values),
        ("optimizer/beta1", beta1_values),
        ("optimizer/beta2", beta2_values),
    ):
        if values:
            payload[f"{tag}_min"] = min(values)
            payload[f"{tag}_max"] = max(values)
            payload[f"{tag}_mean"] = sum(values) / len(values)

    inventory_by_bucket: dict[str, float] = defaultdict(float)
    inventory_by_optimizer: dict[str, float] = defaultdict(float)
    if not _PROBES.optimizer_inventory_logged:
        for entry in _PROBES.param_entries:
            param = entry.param
            bucket = _sanitize_tag_part(entry.bucket)
            inventory_by_bucket[bucket] += float(param.numel())
            inventory_by_optimizer[_sanitize_tag_part(entry.optimizer_name)] += float(param.numel())

    if not _PROBES.optimizer_inventory_logged:
        for bucket, numel in inventory_by_bucket.items():
            payload[f"optimizer_inventory/params_by_module/{bucket}"] = numel
        for optimizer_name, numel in inventory_by_optimizer.items():
            payload[f"optimizer_inventory/params_by_optimizer/{optimizer_name}"] = numel
        _PROBES.optimizer_inventory_logged = True

    _log_payload(writer, wandb_writer, payload, iteration)


def log_parameter_metrics(iteration: int, writer, wandb_writer) -> None:
    """Run the O(parameters) parameter/gradient scan at eval cadence."""

    if (writer is None and wandb_writer is None) or not _PROBES.param_probe_enabled:
        return
    by_bucket: dict[str, dict[str, float]] = defaultdict(
        lambda: {"param_sq": 0.0, "param_n": 0.0, "grad_sq": 0.0, "grad_n": 0.0}
    )
    for entry in _PROBES.param_entries:
        param = entry.param
        bucket = _sanitize_tag_part(entry.bucket)
        stats = by_bucket[bucket]
        pdata = param.detach().float()
        stats["param_sq"] += float(torch.sum(pdata * pdata).item())
        stats["param_n"] += float(param.numel())
        grad = _param_grad(param)
        if grad is not None:
            gdata = grad.detach().float()
            stats["grad_sq"] += float(torch.sum(gdata * gdata).item())
            stats["grad_n"] += float(grad.numel())

    payload: dict[str, float] = {}
    for bucket, stats in by_bucket.items():
        if stats["param_n"] > 0:
            param_norm = stats["param_sq"] ** 0.5
            param_rms = (stats["param_sq"] / stats["param_n"]) ** 0.5
            payload[f"mup/{bucket}/param_norm"] = param_norm
            payload[f"mup/{bucket}/param_rms"] = param_rms
        if stats["grad_n"] > 0:
            grad_norm_bucket = stats["grad_sq"] ** 0.5
            grad_rms = (stats["grad_sq"] / stats["grad_n"]) ** 0.5
            payload[f"mup/{bucket}/grad_norm"] = grad_norm_bucket
            payload[f"mup/{bucket}/grad_rms"] = grad_rms
            if stats["param_sq"] > 0:
                payload[f"mup/{bucket}/grad_to_param_norm"] = grad_norm_bucket / (
                    stats["param_sq"] ** 0.5
                )
                payload[f"mup/{bucket}/grad_to_param_rms"] = grad_rms / (
                    (stats["param_sq"] / max(stats["param_n"], 1.0)) ** 0.5
                )
    _log_payload(writer, wandb_writer, payload, iteration)


# ---------------------------------------------------------------------------
# Log-cadence system and communication probes
# ---------------------------------------------------------------------------


_COMMUNICATION_FIELDS = (
    "dp_grad_reduce_calls",
    "dp_grad_reduce_buffer_bytes",
    "dp_grad_reduce_exposed_seconds",
    "dp_grad_reduce_lifetime_seconds",
    "dp_grad_reduce_scatter_calls",
    "dp_grad_reduce_scatter_buffer_bytes",
    "dp_grad_reduce_scatter_group_size_sum",
    "dp_grad_allreduce_calls",
    "dp_grad_allreduce_buffer_bytes",
    "dp_grad_allreduce_group_size_sum",
    "dp_grad_inter_instance_allreduce_calls",
    "dp_grad_inter_instance_allreduce_buffer_bytes",
    "dp_grad_inter_instance_allreduce_group_size_sum",
    "dp_param_gather_calls",
    "dp_param_gather_buffer_bytes",
    "dp_param_gather_exposed_seconds",
    "dp_param_gather_lifetime_seconds",
    "router_counts_allreduce_calls",
    "router_counts_allreduce_buffer_bytes",
    "router_counts_allreduce_group_size_sum",
    "router_counts_allreduce_exposed_seconds",
    "router_qb_range_allreduce_calls",
    "router_qb_range_allreduce_buffer_bytes",
    "router_qb_range_allreduce_group_size_sum",
    "router_qb_range_allreduce_exposed_seconds",
    "router_qb_hist_allreduce_calls",
    "router_qb_hist_allreduce_buffer_bytes",
    "router_qb_hist_allreduce_group_size_sum",
    "router_qb_hist_allreduce_exposed_seconds",
    "muloco_aux_allreduce_calls",
    "muloco_aux_allreduce_buffer_bytes",
    "muloco_aux_allreduce_exposed_seconds",
    "muloco_aux_allreduce_lifetime_seconds",
    "muloco_signature_allgather_calls",
    "muloco_signature_allgather_buffer_bytes",
    "muloco_outer_allreduce_calls",
    "muloco_outer_allreduce_buffer_bytes",
    "muloco_outer_allreduce_exposed_seconds",
    "muloco_outer_allreduce_lifetime_seconds",
    "muloco_delta_build_seconds",
    "muloco_outer_optimizer_seconds",
    "muloco_anchor_install_seconds",
    "muloco_model_refresh_seconds",
    "muloco_outer_sync_seconds",
    "muloco_outer_syncs",
    "muloco_outer_commits",
    "muloco_outer_commit_failures",
    "muloco_quorum_host_seconds",
    "muloco_quorum_calls",
    "muloco_commit_host_seconds",
    "muloco_commit_calls",
)

_SYSTEM_FIELDS = (
    "step_seconds",
    "local_step_seconds",
    "process_rss_bytes",
    "process_cpu_cores",
    "minor_faults_per_second",
    "major_faults_per_second",
    "gpu_allocated_bytes",
    "gpu_reserved_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_utilization_pct",
    "gpu_memory_utilization_pct",
    "gpu_power_watts",
    "gpu_temperature_c",
    "gpu_clock_mhz",
    "cuda_alloc_retries",
    "cuda_ooms",
    "network_tx_bytes_per_second",
    "network_rx_bytes_per_second",
)


def _read_counter(path: Path) -> int:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return 0


def _explicit_network_interfaces() -> list[str]:
    raw = (
        os.environ.get("SYSTEM_PROBE_NETWORK_INTERFACES")
        or os.environ.get("SYSTEM_PROBE_NETWORK_INTERFACE")
        or os.environ.get("MULOCO_NETWORK_INTERFACE")
        or ""
    )
    interfaces = []
    for value in raw.split(","):
        value = value.strip().removeprefix("=")
        if value and not value.startswith("^"):
            interfaces.append(value)
    return interfaces


def _active_ib_sources(device_names: set[str] | None = None) -> list[tuple]:
    sources = []
    for device in sorted(Path("/sys/class/infiniband").glob("*")):
        if device_names is not None and device.name not in device_names:
            continue
        for port in sorted((device / "ports").glob("*")):
            try:
                state = (port / "state").read_text(encoding="ascii")
            except OSError:
                continue
            if "ACTIVE" not in state.upper():
                continue
            counters = port / "counters"
            tx = counters / "port_xmit_data"
            rx = counters / "port_rcv_data"
            if tx.is_file() and rx.is_file():
                # InfiniBand port_{xmit,rcv}_data counters are in 4-byte words.
                sources.append((tx, rx, 4, f"ib:{device.name}/port{port.name}"))
    return sources


def _default_route_interface() -> str | None:
    try:
        lines = Path("/proc/net/route").read_text(encoding="ascii").splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "00000000":
            try:
                if int(fields[3], 16) & 0x2:
                    return fields[0]
            except ValueError:
                continue
    return None


def _discover_network_sources() -> tuple[tuple, tuple[str, ...]]:
    explicit = _explicit_network_interfaces()
    if explicit:
        ib_devices = set()
        for interface in explicit:
            ib_root = Path("/sys/class/net") / interface / "device" / "infiniband"
            ib_devices.update(path.name for path in ib_root.glob("*"))
        sources = _active_ib_sources(ib_devices) if ib_devices else []
    else:
        sources = _active_ib_sources()

    if not sources:
        interfaces = explicit
        if not interfaces:
            default_interface = _default_route_interface()
            interfaces = [default_interface] if default_interface else []
        for interface in interfaces:
            stats = Path("/sys/class/net") / interface / "statistics"
            tx = stats / "tx_bytes"
            rx = stats / "rx_bytes"
            if tx.is_file() and rx.is_file():
                sources.append((tx, rx, 1, f"net:{interface}"))

    names = tuple(source[3] for source in sources)
    return tuple(sources), names


def _read_network_totals(sources: tuple) -> tuple[int, int]:
    tx_bytes = 0
    rx_bytes = 0
    for tx_path, rx_path, multiplier, _name in sources:
        tx_bytes += _read_counter(tx_path) * multiplier
        rx_bytes += _read_counter(rx_path) * multiplier
    return tx_bytes, rx_bytes


def _read_process_stats() -> tuple[float, int, int, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = 0
    try:
        resident_pages = int(
            Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
        )
        rss_bytes = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    return (
        usage.ru_utime + usage.ru_stime,
        int(usage.ru_minflt),
        int(usage.ru_majflt),
        rss_bytes,
    )


def _read_cuda_stats() -> tuple[int, int, int, int, int]:
    stats = torch.cuda.memory_stats()
    return (
        torch.cuda.memory_allocated(),
        torch.cuda.memory_reserved(),
        torch.cuda.max_memory_allocated(),
        int(stats.get("num_alloc_retries", 0)),
        int(stats.get("num_ooms", 0)),
    )


def _read_gpu_device_stats() -> tuple[float, float, float, float, float]:
    def read(name: str, scale: float = 1.0) -> float:
        try:
            return float(getattr(torch.cuda, name)()) * scale
        except Exception:  # NVML support and individual sensors are optional.
            return float("nan")

    return (
        read("utilization"),
        read("memory_usage"),
        read("power_draw", 1.0 / 1000.0),
        read("temperature"),
        read("clock_rate"),
    )


def _is_local_rank_zero() -> bool:
    try:
        return int(os.environ.get("LOCAL_RANK", torch.cuda.current_device())) == 0
    except (TypeError, ValueError):
        return False


def start_system_probes() -> None:
    """Establish process, allocator, and node-network counter baselines."""

    state = _PROBES.system
    if _is_local_rank_zero() and not state.network_discovery_attempted:
        state.network_discovery_attempted = True
        state.network_sources, state.network_source_names = _discover_network_sources()
        if state.network_source_names:
            logger.info(
                "system probes reading network counters from %s",
                ", ".join(state.network_source_names),
            )
        else:
            logger.warning(
                "system probes found no network counters; set "
                "SYSTEM_PROBE_NETWORK_INTERFACES explicitly"
            )
    cpu_time, minor_faults, major_faults, _rss = _read_process_stats()
    _allocated, _reserved, _peak, alloc_retries, ooms = _read_cuda_stats()
    tx_bytes, rx_bytes = _read_network_totals(state.network_sources)
    state.wall_time = time.monotonic()
    state.cpu_time = cpu_time
    state.minor_faults = minor_faults
    state.major_faults = major_faults
    state.network_tx_bytes = tx_bytes
    state.network_rx_bytes = rx_bytes
    state.cuda_alloc_retries = alloc_retries
    state.cuda_ooms = ooms
    state.accumulated_wall_time = 0.0
    state.accumulated_cpu_time = 0.0
    state.accumulated_minor_faults = 0
    state.accumulated_major_faults = 0
    state.accumulated_network_tx_bytes = 0
    state.accumulated_network_rx_bytes = 0
    state.accumulated_cuda_alloc_retries = 0
    state.accumulated_cuda_ooms = 0
    state.paused = False
    state.started = True


def reset_system_probe_window(*, reset_communication: bool = True) -> None:
    """Drop eval/checkpoint activity from the next training probe window."""

    if _PROBES.enabled:
        start_system_probes()
    if reset_communication:
        communication_probes.reset()


def finish_system_probe_log() -> None:
    """Rebase counters after logging while preserving an unreset first window."""

    state = _PROBES.system
    accumulated = {
        name: getattr(state, name)
        for name in (
            "accumulated_wall_time",
            "accumulated_cpu_time",
            "accumulated_minor_faults",
            "accumulated_major_faults",
            "accumulated_network_tx_bytes",
            "accumulated_network_rx_bytes",
            "accumulated_cuda_alloc_retries",
            "accumulated_cuda_ooms",
        )
    }
    start_system_probes()
    for name, value in accumulated.items():
        setattr(state, name, value)


def current_system_probe_window_seconds() -> float:
    """Return this rank's training-window wall time without a rank barrier."""

    state = _PROBES.system
    if not state.started:
        start_system_probes()
    active_wall = 0.0 if state.paused else max(0.0, time.monotonic() - state.wall_time)
    return max(state.accumulated_wall_time + active_wall, 0.0)


def pause_system_probe_window() -> None:
    """Carry the current training segment across an eval/checkpoint pause."""

    state = _PROBES.system
    if not state.started or state.paused:
        return
    now = time.monotonic()
    cpu_time, minor_faults, major_faults, _rss = _read_process_stats()
    _allocated, _reserved, _peak, alloc_retries, ooms = _read_cuda_stats()
    tx_bytes, rx_bytes = _read_network_totals(state.network_sources)
    state.accumulated_wall_time += max(0.0, now - state.wall_time)
    state.accumulated_cpu_time += max(0.0, cpu_time - state.cpu_time)
    state.accumulated_minor_faults += max(0, minor_faults - state.minor_faults)
    state.accumulated_major_faults += max(0, major_faults - state.major_faults)
    state.accumulated_network_tx_bytes += max(0, tx_bytes - state.network_tx_bytes)
    state.accumulated_network_rx_bytes += max(0, rx_bytes - state.network_rx_bytes)
    state.accumulated_cuda_alloc_retries += max(
        0, alloc_retries - state.cuda_alloc_retries
    )
    state.accumulated_cuda_ooms += max(0, ooms - state.cuda_ooms)
    state.paused = True
    communication_probes.carry_active_window()


def resume_system_probe_window() -> None:
    """Discard non-training activity and resume the current training window."""

    state = _PROBES.system
    if not state.started:
        start_system_probes()
        return
    cpu_time, minor_faults, major_faults, _rss = _read_process_stats()
    _allocated, _reserved, _peak, alloc_retries, ooms = _read_cuda_stats()
    tx_bytes, rx_bytes = _read_network_totals(state.network_sources)
    state.wall_time = time.monotonic()
    state.cpu_time = cpu_time
    state.minor_faults = minor_faults
    state.major_faults = major_faults
    state.network_tx_bytes = tx_bytes
    state.network_rx_bytes = rx_bytes
    state.cuda_alloc_retries = alloc_retries
    state.cuda_ooms = ooms
    state.paused = False
    communication_probes.discard_active_window()


def _sample_local_system(
    reset_window: bool, *, carried_wall_seconds: float | None = None
) -> dict[str, float]:
    state = _PROBES.system
    if not state.started:
        start_system_probes()

    now = time.monotonic()
    cpu_time, minor_faults, major_faults, rss_bytes = _read_process_stats()
    allocated, reserved, peak, alloc_retries, ooms = _read_cuda_stats()
    gpu_util, gpu_memory_util, gpu_power, gpu_temperature, gpu_clock = (
        _read_gpu_device_stats()
    )
    tx_bytes, rx_bytes = _read_network_totals(state.network_sources)
    active_wall = 0.0 if state.paused else max(0.0, now - state.wall_time)
    elapsed = max(state.accumulated_wall_time + active_wall, 1e-12)
    cpu_delta = state.accumulated_cpu_time
    minor_fault_delta = state.accumulated_minor_faults
    major_fault_delta = state.accumulated_major_faults
    tx_delta = state.accumulated_network_tx_bytes
    rx_delta = state.accumulated_network_rx_bytes
    retry_delta = state.accumulated_cuda_alloc_retries
    oom_delta = state.accumulated_cuda_ooms
    if not state.paused:
        cpu_delta += max(0.0, cpu_time - state.cpu_time)
        minor_fault_delta += max(0, minor_faults - state.minor_faults)
        major_fault_delta += max(0, major_faults - state.major_faults)
        tx_delta += max(0, tx_bytes - state.network_tx_bytes)
        rx_delta += max(0, rx_bytes - state.network_rx_bytes)
        retry_delta += max(0, alloc_retries - state.cuda_alloc_retries)
        oom_delta += max(0, ooms - state.cuda_ooms)
    node_counter = _is_local_rank_zero() and bool(state.network_sources)
    sample = {
        "local_window_seconds": elapsed,
        "process_rss_bytes": float(rss_bytes),
        "process_cpu_cores": cpu_delta / elapsed,
        "minor_faults_per_second": minor_fault_delta / elapsed,
        "major_faults_per_second": major_fault_delta / elapsed,
        "gpu_allocated_bytes": float(allocated),
        "gpu_reserved_bytes": float(reserved),
        "gpu_peak_allocated_bytes": float(peak),
        "gpu_utilization_pct": gpu_util,
        "gpu_memory_utilization_pct": gpu_memory_util,
        "gpu_power_watts": gpu_power,
        "gpu_temperature_c": gpu_temperature,
        "gpu_clock_mhz": gpu_clock,
        "cuda_alloc_retries": float(retry_delta),
        "cuda_ooms": float(oom_delta),
        "network_tx_bytes_per_second": (
            tx_delta / elapsed if node_counter else -1.0
        ),
        "network_rx_bytes_per_second": (
            rx_delta / elapsed if node_counter else -1.0
        ),
    }
    state.wall_time = now
    state.cpu_time = cpu_time
    state.minor_faults = minor_faults
    state.major_faults = major_faults
    state.network_tx_bytes = tx_bytes
    state.network_rx_bytes = rx_bytes
    state.cuda_alloc_retries = alloc_retries
    state.cuda_ooms = ooms
    if reset_window:
        state.accumulated_wall_time = 0.0
        state.accumulated_cpu_time = 0.0
        state.accumulated_minor_faults = 0
        state.accumulated_major_faults = 0
        state.accumulated_network_tx_bytes = 0
        state.accumulated_network_rx_bytes = 0
        state.accumulated_cuda_alloc_retries = 0
        state.accumulated_cuda_ooms = 0
    else:
        state.accumulated_wall_time = (
            elapsed if carried_wall_seconds is None else carried_wall_seconds
        )
        state.accumulated_cpu_time = cpu_delta
        state.accumulated_minor_faults = minor_fault_delta
        state.accumulated_major_faults = major_fault_delta
        state.accumulated_network_tx_bytes = tx_delta
        state.accumulated_network_rx_bytes = rx_delta
        state.accumulated_cuda_alloc_retries = retry_delta
        state.accumulated_cuda_ooms = oom_delta
    return sample


def _finite_values(rows: torch.Tensor, index: int, *, nonnegative: bool = False) -> list[float]:
    values = [float(value) for value in rows[:, index].tolist()]
    values = [value for value in values if math.isfinite(value)]
    if nonnegative:
        values = [value for value in values if value >= 0.0]
    return values


def log_system_communication_metrics(
    iteration: int,
    writer,
    wandb_writer,
    *,
    step_seconds: float,
    window_iterations: int,
    logical_batch_size: int,
    sequence_length: int,
    tflops_per_gpu: float,
    logical_worker_count: int,
    data_parallel_size: int,
    muloco_group_size: int,
    muloco_sync_steps: int,
    local_window_seconds: float,
    reset_window: bool,
) -> None:
    """Gather one compact rank summary and log comparable system/comm metrics."""

    steps = max(int(window_iterations), 1)
    system_values = _sample_local_system(
        reset_window, carried_wall_seconds=float(local_window_seconds)
    )
    system_values["step_seconds"] = float(step_seconds)
    system_values["local_step_seconds"] = float(local_window_seconds) / steps
    communication_values = communication_probes.snapshot(reset_window=reset_window)
    fields = _SYSTEM_FIELDS + _COMMUNICATION_FIELDS
    local_values = {
        **{field: float(system_values.get(field, 0.0)) for field in _SYSTEM_FIELDS},
        **{
            field: float(communication_values.get(field, 0.0))
            for field in _COMMUNICATION_FIELDS
        },
    }
    local = torch.tensor(
        [local_values[field] for field in fields], dtype=torch.float64, device="cuda"
    )
    world_size = torch.distributed.get_world_size()
    gathered = torch.empty(world_size * len(fields), dtype=torch.float64, device="cuda")
    aggregation_started = time.perf_counter()
    torch.distributed.all_gather_into_tensor(gathered, local)
    rows = gathered.view(world_size, len(fields)).cpu()
    aggregation_ms = (time.perf_counter() - aggregation_started) * 1000.0
    field_index = {field: index for index, field in enumerate(fields)}

    def values(field: str, *, nonnegative: bool = False) -> list[float]:
        return _finite_values(rows, field_index[field], nonnegative=nonnegative)

    def total(field: str) -> float:
        return sum(values(field))

    def mean(field: str) -> float:
        field_values = values(field)
        return sum(field_values) / max(len(field_values), 1)

    def maximum(field: str) -> float:
        field_values = values(field)
        return max(field_values) if field_values else 0.0

    def minimum(field: str) -> float:
        field_values = values(field)
        return min(field_values) if field_values else 0.0

    payload: dict[str, float] = {}
    step_max = maximum("step_seconds")
    local_step_min = minimum("local_step_seconds")
    local_step_mean = mean("local_step_seconds")
    local_step_max = maximum("local_step_seconds")
    payload["system/step_time_ms"] = step_max * 1000.0
    payload["system/local_step_time_ms_rank_min"] = local_step_min * 1000.0
    payload["system/local_step_time_ms_rank_mean"] = local_step_mean * 1000.0
    payload["system/local_step_time_ms_rank_max"] = local_step_max * 1000.0
    payload["system/rank_step_skew_pct"] = (
        (local_step_max / max(local_step_min, 1e-12) - 1.0) * 100.0
    )
    logical_tokens_per_step = float(logical_batch_size * sequence_length)
    payload["system/logical_tokens_per_second"] = logical_tokens_per_step / max(
        step_max, 1e-12
    )
    payload["system/tflops_per_gpu"] = float(tflops_per_gpu)
    payload["system/probe_aggregation_ms_writer"] = aggregation_ms

    gib = float(1024**3)
    for metric_field, tag in (
        ("process_rss_bytes", "system/process_rss_gib"),
        ("gpu_allocated_bytes", "system/gpu_memory_allocated_gib"),
        ("gpu_reserved_bytes", "system/gpu_memory_reserved_gib"),
        ("gpu_peak_allocated_bytes", "system/gpu_memory_peak_allocated_gib"),
    ):
        payload[f"{tag}_rank_mean"] = mean(metric_field) / gib
        payload[f"{tag}_rank_max"] = maximum(metric_field) / gib
    for metric_field, tag in (
        ("process_cpu_cores", "system/process_cpu_cores"),
        ("minor_faults_per_second", "system/minor_faults_per_second"),
        ("major_faults_per_second", "system/major_faults_per_second"),
    ):
        payload[f"{tag}_rank_mean"] = mean(metric_field)
        payload[f"{tag}_rank_max"] = maximum(metric_field)
    for metric_field, tag in (
        ("gpu_utilization_pct", "system/gpu_utilization_pct"),
        ("gpu_memory_utilization_pct", "system/gpu_memory_utilization_pct"),
        ("gpu_power_watts", "system/gpu_power_watts"),
        ("gpu_temperature_c", "system/gpu_temperature_c"),
        ("gpu_clock_mhz", "system/gpu_clock_mhz"),
    ):
        metric_values = values(metric_field)
        if metric_values:
            payload[f"{tag}_rank_mean"] = sum(metric_values) / len(metric_values)
            payload[f"{tag}_rank_max"] = max(metric_values)
    payload["system/cuda_alloc_retries_per_step_rank_max"] = maximum(
        "cuda_alloc_retries"
    ) / steps
    payload["system/cuda_ooms_window"] = total("cuda_ooms")

    network_tx = values("network_tx_bytes_per_second", nonnegative=True)
    network_rx = values("network_rx_bytes_per_second", nonnegative=True)
    if network_tx and network_rx:
        scale = max(int(logical_worker_count), 1)
        observed_nodes = len(network_tx)
        tx_observed = sum(network_tx)
        rx_observed = sum(network_rx)
        tx_cluster = tx_observed * scale
        rx_cluster = rx_observed * scale
        payload["communication/nic_observed_nodes"] = float(observed_nodes)
        payload["communication/nic_logical_cluster_scale"] = float(scale)
        payload["communication/nic_tx_gbps_per_node_mean"] = (
            tx_observed / observed_nodes * 8.0 / 1e9
        )
        payload["communication/nic_tx_gbps_per_node_max"] = max(network_tx) * 8.0 / 1e9
        payload["communication/nic_rx_gbps_per_node_mean"] = (
            rx_observed / observed_nodes * 8.0 / 1e9
        )
        payload["communication/nic_rx_gbps_per_node_max"] = max(network_rx) * 8.0 / 1e9
        payload["communication/nic_tx_gbps_logical_cluster_estimate"] = (
            tx_cluster * 8.0 / 1e9
        )
        payload["communication/nic_rx_gbps_logical_cluster_estimate"] = (
            rx_cluster * 8.0 / 1e9
        )
        logical_token_rate = payload["system/logical_tokens_per_second"]
        payload["communication/nic_tx_bytes_per_logical_token"] = tx_cluster / max(
            logical_token_rate, 1e-12
        )
        payload["communication/nic_rx_bytes_per_logical_token"] = rx_cluster / max(
            logical_token_rate, 1e-12
        )

    for prefix, counter_prefix in (
        ("dp_grad_reduce", "communication/dp_grad_reduce"),
        ("dp_param_gather", "communication/dp_param_gather"),
    ):
        payload[f"{counter_prefix}/collectives_per_step_rank_mean"] = mean(
            f"{prefix}_calls"
        ) / steps
        payload[f"{counter_prefix}/buffer_gib_per_step_rank_mean"] = mean(
            f"{prefix}_buffer_bytes"
        ) / steps / gib
        payload[f"{counter_prefix}/exposed_ms_per_step_rank_max"] = maximum(
            f"{prefix}_exposed_seconds"
        ) * 1000.0 / steps
        payload[f"{counter_prefix}/lifetime_ms_per_step_rank_max"] = maximum(
            f"{prefix}_lifetime_seconds"
        ) * 1000.0 / steps

    for prefix, counter_prefix in (
        ("router_counts_allreduce", "communication/router_counts_allreduce"),
        ("router_qb_range_allreduce", "communication/router_qb_range_allreduce"),
        ("router_qb_hist_allreduce", "communication/router_qb_hist_allreduce"),
        ("muloco_aux_allreduce", "communication/muloco_aux_allreduce"),
    ):
        payload[f"{counter_prefix}/collectives_per_step_rank_mean"] = mean(
            f"{prefix}_calls"
        ) / steps
        payload[f"{counter_prefix}/buffer_bytes_per_step_rank_mean"] = mean(
            f"{prefix}_buffer_bytes"
        ) / steps
        payload[f"{counter_prefix}/exposed_ms_per_step_rank_max"] = maximum(
            f"{prefix}_exposed_seconds"
        ) * 1000.0 / steps
    payload["communication/muloco_aux_allreduce/lifetime_ms_per_step_rank_max"] = (
        maximum("muloco_aux_allreduce_lifetime_seconds") * 1000.0 / steps
    )

    outer_syncs = maximum("muloco_outer_syncs")
    if outer_syncs > 0:
        outer_payload = mean("muloco_outer_allreduce_buffer_bytes") / outer_syncs
        outer_seconds = maximum("muloco_outer_sync_seconds") / outer_syncs
        outer_allreduce_lifetime_seconds = maximum(
            "muloco_outer_allreduce_lifetime_seconds"
        ) / outer_syncs
        outer_allreduce_exposed_seconds = maximum(
            "muloco_outer_allreduce_exposed_seconds"
        ) / outer_syncs
        payload["communication/muloco/outer_syncs_window"] = outer_syncs
        payload["communication/muloco/outer_commits_window"] = maximum(
            "muloco_outer_commits"
        )
        payload["communication/muloco/outer_commit_failures_window"] = maximum(
            "muloco_outer_commit_failures"
        )
        payload["communication/muloco/payload_gib_per_sync_rank_mean"] = (
            outer_payload / gib
        )
        payload["communication/muloco/collectives_per_sync_rank_mean"] = mean(
            "muloco_outer_allreduce_calls"
        ) / outer_syncs
        payload["communication/muloco/outer_total_ms_per_sync_rank_max"] = (
            outer_seconds * 1000.0
        )
        payload["communication/muloco/outer_amortized_ms_per_inner_step"] = (
            outer_seconds * 1000.0 / max(muloco_sync_steps, 1)
        )
        payload["communication/muloco/allreduce_exposed_ms_per_sync_rank_max"] = (
            outer_allreduce_exposed_seconds * 1000.0
        )
        payload["communication/muloco/allreduce_lifetime_ms_per_sync_rank_max"] = (
            outer_allreduce_lifetime_seconds * 1000.0
        )
        ring_factor = 2.0 * (max(muloco_group_size, 1) - 1) / max(
            muloco_group_size, 1
        )
        payload["communication/muloco/outer_bus_gbps_estimate"] = (
            outer_payload
            * ring_factor
            / max(outer_allreduce_lifetime_seconds, 1e-12)
            / 1e9
        )
        for metric_field, tag in (
            ("muloco_delta_build_seconds", "delta_build_ms_per_sync_rank_max"),
            ("muloco_outer_optimizer_seconds", "outer_optimizer_ms_per_sync_rank_max"),
            ("muloco_anchor_install_seconds", "anchor_install_ms_per_sync_rank_max"),
            ("muloco_model_refresh_seconds", "model_refresh_ms_per_sync_rank_max"),
        ):
            payload[f"communication/muloco/{tag}"] = (
                maximum(metric_field) * 1000.0 / outer_syncs
            )
        payload["communication/muloco/commit_host_ms_per_sync_rank_max"] = (
            maximum("muloco_commit_host_seconds") * 1000.0 / outer_syncs
        )
        payload["communication/muloco/quorum_host_ms_per_sync_rank_max"] = (
            maximum("muloco_quorum_host_seconds") * 1000.0 / outer_syncs
        )

    def collective_group_size(prefix: str, fallback: int) -> float:
        calls = total(f"{prefix}_calls")
        if calls <= 0:
            return float(max(fallback, 1))
        return max(total(f"{prefix}_group_size_sum") / calls, 1.0)

    def allgather_reduce_scatter_factor(group_size: float) -> float:
        return (group_size - 1.0) / group_size

    def allreduce_factor(group_size: float) -> float:
        return 2.0 * allgather_reduce_scatter_factor(group_size)

    logical_scale = max(int(logical_worker_count), 1)
    dp_group_size = float(max(data_parallel_size, 1))
    reduce_scatter_group_size = collective_group_size(
        "dp_grad_reduce_scatter", data_parallel_size
    )
    grad_allreduce_group_size = collective_group_size(
        "dp_grad_allreduce", data_parallel_size
    )
    inter_instance_group_size = collective_group_size(
        "dp_grad_inter_instance_allreduce", data_parallel_size
    )
    router_group_size = collective_group_size(
        "router_counts_allreduce", data_parallel_size
    )
    router_qb_range_group_size = collective_group_size(
        "router_qb_range_allreduce", data_parallel_size
    )
    router_qb_hist_group_size = collective_group_size(
        "router_qb_hist_allreduce", data_parallel_size
    )
    dp_reduce_scatter_tx = total(
        "dp_grad_reduce_scatter_buffer_bytes"
    ) * allgather_reduce_scatter_factor(reduce_scatter_group_size)
    dp_grad_allreduce_tx = total(
        "dp_grad_allreduce_buffer_bytes"
    ) * allreduce_factor(grad_allreduce_group_size)
    dp_inter_instance_tx = total(
        "dp_grad_inter_instance_allreduce_buffer_bytes"
    ) * allreduce_factor(inter_instance_group_size)
    dp_param_gather_tx = total(
        "dp_param_gather_buffer_bytes"
    ) * allgather_reduce_scatter_factor(dp_group_size)
    router_tx = total("router_counts_allreduce_buffer_bytes") * allreduce_factor(
        router_group_size
    )
    router_tx += total("router_qb_range_allreduce_buffer_bytes") * allreduce_factor(
        router_qb_range_group_size
    )
    router_tx += total("router_qb_hist_allreduce_buffer_bytes") * allreduce_factor(
        router_qb_hist_group_size
    )
    dp_tx_bytes = (
        dp_reduce_scatter_tx
        + dp_grad_allreduce_tx
        + dp_inter_instance_tx
        + dp_param_gather_tx
        + router_tx
    ) * logical_scale
    algorithmic_tx_bytes = dp_tx_bytes
    muloco_tx_bytes = 0.0
    if muloco_group_size > 1:
        outer_factor = 2.0 * (muloco_group_size - 1) / muloco_group_size
        muloco_tx_bytes = (
            total("muloco_aux_allreduce_buffer_bytes")
            + total("muloco_outer_allreduce_buffer_bytes")
        ) * outer_factor * logical_scale
        muloco_tx_bytes += (
            total("muloco_signature_allgather_buffer_bytes")
            * (muloco_group_size - 1)
            * logical_scale
        )
        algorithmic_tx_bytes += muloco_tx_bytes
    logical_tokens_window = logical_tokens_per_step * steps
    payload["communication/dp_tx_bytes_per_logical_token_estimate"] = (
        dp_tx_bytes / max(logical_tokens_window, 1.0)
    )
    payload["communication/muloco_tx_bytes_per_logical_token_estimate"] = (
        muloco_tx_bytes / max(logical_tokens_window, 1.0)
    )
    payload["communication/algorithmic_tx_bytes_per_logical_token_estimate"] = (
        algorithmic_tx_bytes / max(logical_tokens_window, 1.0)
    )

    exposed_seconds = (
        maximum("dp_grad_reduce_exposed_seconds")
        + maximum("dp_param_gather_exposed_seconds")
        + maximum("router_counts_allreduce_exposed_seconds")
        + maximum("muloco_aux_allreduce_exposed_seconds")
        + maximum("muloco_outer_allreduce_exposed_seconds")
        + maximum("muloco_quorum_host_seconds")
        + maximum("muloco_commit_host_seconds")
    )
    payload["communication/exposed_ms_per_step_upper_bound"] = (
        exposed_seconds * 1000.0 / steps
    )
    payload["communication/exposed_step_fraction_upper_bound"] = exposed_seconds / max(
        step_max * steps, 1e-12
    )
    payload["communication/aggregation_world_size"] = float(world_size)
    payload["communication/logical_worker_count"] = float(logical_worker_count)
    payload["communication/trainable_param_numel"] = float(_PROBES.trainable_param_numel)
    payload["communication/trainable_param_bytes"] = float(_PROBES.trainable_param_bytes)
    _log_payload(writer, wandb_writer, payload, iteration)


# ---------------------------------------------------------------------------
# Metric 1: per-step expert max violation
# ---------------------------------------------------------------------------


def _log_scalar(
    writer, wandb_payload: dict[str, float] | None, tag: str, value: float, iteration: int
) -> None:
    if writer is not None:
        writer.add_scalar(tag, value, iteration)
    if wandb_payload is not None:
        wandb_payload[tag] = value


def log_expert_histogram_step(iteration: int, writer, wandb_writer, log_image: bool) -> None:
    """Aggregate and log expert load, bias, and Quantile-Balancing diagnostics.

    Always drains the accumulators so they restart from zero for the next
    window. ``log_image`` is kept in the signature for the training-loop call
    site, but this probe intentionally emits only scalar maxvio metrics.
    """
    if not _PROBES.enabled or not _PROBES.routers:
        return

    # Stack all router counts: [num_moe_layers, num_experts]
    stacked = torch.stack([s.counts for s in _PROBES.routers], dim=0)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.SUM)

    counts = stacked.float()
    totals = counts.sum(dim=1)
    means = counts.mean(dim=1).clamp_min(1e-12)
    max_counts = counts.max(dim=1).values
    max_violation = max_counts / means - 1.0

    if writer is not None or wandb_writer is not None:
        wandb_payload = {} if wandb_writer is not None else None
        totals_cpu = totals.detach().cpu().tolist()
        maxvio_cpu = max_violation.detach().cpu().tolist()

        active_rows = []
        bias_absmax_values = []
        qb_margin_spans = []
        for i, layer_num in enumerate(_PROBES.router_layer_indices):
            if totals_cpu[i] <= 0:
                continue
            active_rows.append(i)
            _log_scalar(
                writer, wandb_payload, f"moe_probe/layer_{layer_num}/maxvio",
                maxvio_cpu[i], iteration,
            )
            router = _PROBES.routers[i].router
            expert_bias = getattr(router, "expert_bias", None)
            if expert_bias is not None:
                bias_fp32 = expert_bias.detach().float()
                bias_absmax = bias_fp32.abs().max().item()
                bias_rms = bias_fp32.square().mean().sqrt().item()
                bias_absmax_values.append(bias_absmax)
                _log_scalar(
                    writer,
                    wandb_payload,
                    f"moe_probe/layer_{layer_num}/expert_bias_absmax",
                    bias_absmax,
                    iteration,
                )
                _log_scalar(
                    writer,
                    wandb_payload,
                    f"moe_probe/layer_{layer_num}/expert_bias_rms",
                    bias_rms,
                    iteration,
                )
            qb_margin_min = getattr(router, "qb_last_margin_min", None)
            qb_margin_max = getattr(router, "qb_last_margin_max", None)
            if qb_margin_min is not None and qb_margin_max is not None:
                margin_min = qb_margin_min.item()
                margin_max = qb_margin_max.item()
                qb_margin_spans.append(margin_max - margin_min)
                _log_scalar(
                    writer,
                    wandb_payload,
                    f"moe_probe/layer_{layer_num}/qb_margin_min",
                    margin_min,
                    iteration,
                )
                _log_scalar(
                    writer,
                    wandb_payload,
                    f"moe_probe/layer_{layer_num}/qb_margin_max",
                    margin_max,
                    iteration,
                )

        if active_rows:
            _log_scalar(
                writer, wandb_payload, "moe_probe/summary/maxvio_mean",
                sum(maxvio_cpu[i] for i in active_rows) / len(active_rows), iteration,
            )
        if bias_absmax_values:
            _log_scalar(
                writer,
                wandb_payload,
                "moe_probe/summary/expert_bias_absmax",
                max(bias_absmax_values),
                iteration,
            )
        if qb_margin_spans:
            _log_scalar(
                writer,
                wandb_payload,
                "moe_probe/summary/qb_margin_span_mean",
                sum(qb_margin_spans) / len(qb_margin_spans),
                iteration,
            )

        if wandb_payload:
            log_namespaced_metrics(wandb_writer, wandb_payload, iteration)

    # Reset accumulators for the next window.
    for s in _PROBES.routers:
        s.counts.zero_()


# ---------------------------------------------------------------------------
# Metrics 2 & 3: per-eval activation RMS and adjacent-layer delta
# ---------------------------------------------------------------------------


def begin_eval_rms(model) -> None:
    """Register forward hooks for activation and stability probes during eval."""
    from megatron.core.transformer.attention import Attention
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.transformer_layer import TransformerLayer

    _clear_eval_probe_handles()
    _PROBES.layers.clear()
    _PROBES.branches.clear()
    _PROBES.router_z.clear()
    _PROBES.logits.clear()
    for module in _iter_modules(model):
        if isinstance(module, TransformerLayer):
            state = _RMSState(
                sum_sq=torch.zeros((), dtype=torch.float64, device="cuda"),
                n=torch.zeros((), dtype=torch.int64, device="cuda"),
            )

            def _make_hook(s: _RMSState):
                def _hook(_mod, _inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if h is None:
                        return
                    h = h.detach()
                    s.sum_sq.add_(h.float().pow(2).sum().to(torch.float64))
                    s.n.add_(h.numel())

                return _hook

            state.hook_handle = module.register_forward_hook(_make_hook(state))
            _PROBES.layers.append(state)

    for name, module in _iter_named_modules(model):
        if isinstance(module, Attention):
            layer_number = getattr(module, "layer_number", len(_PROBES.branches) + 1)
            state = _BranchState(
                kind="attention",
                layer_number=layer_number,
                sum_sq=torch.zeros((), dtype=torch.float64, device="cuda"),
                n=torch.zeros((), dtype=torch.int64, device="cuda"),
                outliers=torch.zeros((), dtype=torch.int64, device="cuda"),
                outlier_n=torch.zeros((), dtype=torch.int64, device="cuda"),
            )
            state.hook_handle = module.register_forward_hook(_make_branch_hook(state))
            _PROBES.branches.append(state)
        elif isinstance(module, MoELayer):
            layer_number = getattr(module, "layer_number", len(_PROBES.branches) + 1)
            state = _BranchState(
                kind="moe",
                layer_number=layer_number,
                sum_sq=torch.zeros((), dtype=torch.float64, device="cuda"),
                n=torch.zeros((), dtype=torch.int64, device="cuda"),
                outliers=torch.zeros((), dtype=torch.int64, device="cuda"),
                outlier_n=torch.zeros((), dtype=torch.int64, device="cuda"),
            )
            state.hook_handle = module.register_forward_hook(_make_branch_hook(state))
            _PROBES.branches.append(state)
        elif isinstance(module, TopKRouter):
            layer_number = getattr(module, "layer_number", len(_PROBES.router_z) + 1)
            state = _RouterZState(
                layer_number=layer_number,
                z_sum=torch.zeros((), dtype=torch.float64, device="cuda"),
                z_n=torch.zeros((), dtype=torch.int64, device="cuda"),
                logit_sum_sq=torch.zeros((), dtype=torch.float64, device="cuda"),
                logit_n=torch.zeros((), dtype=torch.int64, device="cuda"),
                logit_absmax=torch.zeros((), dtype=torch.float32, device="cuda"),
            )
            state.hook_handle = module.register_forward_hook(_make_router_z_hook(state))
            _PROBES.router_z.append(state)
        elif name.endswith(".output_layer") or name.endswith("output_layer"):
            state = _LogitState(
                name="lm_logits",
                sum_sq=torch.zeros((), dtype=torch.float64, device="cuda"),
                n=torch.zeros((), dtype=torch.int64, device="cuda"),
                absmax=torch.zeros((), dtype=torch.float32, device="cuda"),
            )
            state.hook_handle = module.register_forward_hook(_make_logit_hook(state))
            _PROBES.logits.append(state)


def end_eval_rms(iteration: int, writer, wandb_writer) -> None:
    """Reduce and log activation RMS and adjacent-layer deltas, then remove hooks."""
    if not _PROBES.layers and not _PROBES.branches and not _PROBES.router_z and not _PROBES.logits:
        return

    if _PROBES.layers:
        # Stack to [num_layers]
        sum_sqs = torch.stack([s.sum_sq for s in _PROBES.layers], dim=0)
        ns = torch.stack([s.n for s in _PROBES.layers], dim=0)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(sum_sqs, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(ns, op=torch.distributed.ReduceOp.SUM)

        rms = (sum_sqs / ns.clamp_min(1)).sqrt()

        # Writers live on the last rank (world_size - 1), not rank 0.
        if writer is not None or wandb_writer is not None:
            wandb_payload = {} if wandb_writer is not None else None
            rms_cpu = rms.detach().cpu()
            for layer_i, value in enumerate(rms_cpu.tolist()):
                tag = f"act_rms/layer_{layer_i}"
                _log_scalar(writer, wandb_payload, tag, value, iteration)

            deltas = (rms_cpu[1:] - rms_cpu[:-1]).abs()
            ratios = rms_cpu[1:] / rms_cpu[:-1].clamp_min(1e-12)
            for i, (delta, ratio) in enumerate(zip(deltas.tolist(), ratios.tolist())):
                tag_abs = f"act_rms_delta/abs_l{i}_to_l{i+1}"
                tag_ratio = f"act_rms_delta/ratio_l{i}_to_l{i+1}"
                _log_scalar(writer, wandb_payload, tag_abs, delta, iteration)
                _log_scalar(writer, wandb_payload, tag_ratio, ratio, iteration)
            if wandb_payload:
                log_namespaced_metrics(wandb_writer, wandb_payload, iteration)

    _log_eval_stability(iteration, writer, wandb_writer)

    # Teardown.
    _clear_eval_probe_handles()
    _PROBES.layers.clear()
    _PROBES.branches.clear()
    _PROBES.router_z.clear()
    _PROBES.logits.clear()


def _extract_tensor(out) -> torch.Tensor | None:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _accumulate_branch_stats(state: _BranchState, tensor: torch.Tensor) -> None:
    h = tensor.detach().float()
    state.sum_sq.add_(torch.sum(h * h).to(torch.float64))
    state.n.add_(h.numel())
    if h.numel() == 0 or h.dim() == 0:
        return
    flat = h.reshape(-1, h.shape[-1])
    mean = flat.mean(dim=-1, keepdim=True)
    std = flat.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    state.outliers.add_(((flat - mean).abs() > (5.0 * std)).sum().to(torch.int64))
    state.outlier_n.add_(flat.numel())


def _make_branch_hook(state: _BranchState):
    def _hook(_mod, _inp, out):
        tensor = _extract_tensor(out)
        if tensor is not None:
            _accumulate_branch_stats(state, tensor)

    return _hook


def _make_router_z_hook(state: _RouterZState):
    def _hook(mod, inp, _out):
        if not inp:
            return
        with torch.no_grad():
            logits = mod.gating(inp[0].detach())
            logits = logits.reshape(-1, logits.shape[-1]).float()
            if logits.numel() == 0:
                return
            lse = torch.logsumexp(logits, dim=-1)
            state.z_sum.add_(torch.sum(lse * lse).to(torch.float64))
            state.z_n.add_(lse.numel())
            state.logit_sum_sq.add_(torch.sum(logits * logits).to(torch.float64))
            state.logit_n.add_(logits.numel())
            state.logit_absmax.copy_(torch.maximum(state.logit_absmax, logits.abs().max()))

    return _hook


def _make_logit_hook(state: _LogitState):
    def _hook(_mod, _inp, out):
        tensor = _extract_tensor(out)
        if tensor is None:
            return
        logits = tensor.detach().float()
        if logits.numel() == 0:
            return
        state.sum_sq.add_(torch.sum(logits * logits).to(torch.float64))
        state.n.add_(logits.numel())
        state.absmax.copy_(torch.maximum(state.absmax, logits.abs().max()))

    return _hook


def _clear_eval_probe_handles() -> None:
    for states in (_PROBES.layers, _PROBES.branches, _PROBES.router_z, _PROBES.logits):
        for state in states:
            if state.hook_handle is not None:
                state.hook_handle.remove()
                state.hook_handle = None


def _reduce_sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _reduce_max_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor


def _log_eval_stability(iteration: int, writer, wandb_writer) -> None:
    payload: dict[str, float] = {}

    branch_summary: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for state in _PROBES.branches:
        sum_sq = _reduce_sum_tensor(state.sum_sq.clone())
        n = _reduce_sum_tensor(state.n.clone())
        outliers = _reduce_sum_tensor(state.outliers.clone())
        outlier_n = _reduce_sum_tensor(state.outlier_n.clone())
        if n.item() <= 0:
            continue
        rms = float((sum_sq / n.clamp_min(1)).sqrt().item())
        outlier_pct = float((outliers.float() / outlier_n.clamp_min(1).float() * 100.0).item())
        layer = state.layer_number
        payload[f"stability/{state.kind}_output_rms/layer_{layer}"] = rms
        payload[f"stability/{state.kind}_outlier_pct/layer_{layer}"] = outlier_pct
        branch_summary[state.kind].append((rms, outlier_pct))

    for kind, values in branch_summary.items():
        if values:
            payload[f"stability/{kind}_output_rms_mean"] = sum(v[0] for v in values) / len(values)
            payload[f"stability/{kind}_outlier_pct_mean"] = sum(v[1] for v in values) / len(values)

    router_z_values = []
    router_logit_rms_values = []
    router_logit_absmax_values = []
    for state in _PROBES.router_z:
        z_sum = _reduce_sum_tensor(state.z_sum.clone())
        z_n = _reduce_sum_tensor(state.z_n.clone())
        logit_sum_sq = _reduce_sum_tensor(state.logit_sum_sq.clone())
        logit_n = _reduce_sum_tensor(state.logit_n.clone())
        logit_absmax = _reduce_max_tensor(state.logit_absmax.clone())
        if z_n.item() <= 0:
            continue
        z_value = float((z_sum / z_n.clamp_min(1)).item())
        logit_rms = float((logit_sum_sq / logit_n.clamp_min(1)).sqrt().item())
        logit_max = float(logit_absmax.item())
        layer = state.layer_number
        payload[f"stability/router_z_lse2/layer_{layer}"] = z_value
        payload[f"stability/router_logit_rms/layer_{layer}"] = logit_rms
        payload[f"stability/router_logit_absmax/layer_{layer}"] = logit_max
        router_z_values.append(z_value)
        router_logit_rms_values.append(logit_rms)
        router_logit_absmax_values.append(logit_max)

    if router_z_values:
        payload["stability/router_z_lse2_mean"] = sum(router_z_values) / len(router_z_values)
        payload["stability/router_logit_rms_mean"] = (
            sum(router_logit_rms_values) / len(router_logit_rms_values)
        )
        payload["stability/router_logit_absmax_max"] = max(router_logit_absmax_values)

    for state in _PROBES.logits:
        sum_sq = _reduce_sum_tensor(state.sum_sq.clone())
        n = _reduce_sum_tensor(state.n.clone())
        absmax = _reduce_max_tensor(state.absmax.clone())
        if n.item() <= 0:
            continue
        payload[f"stability/{state.name}_rms"] = float((sum_sq / n.clamp_min(1)).sqrt().item())
        payload[f"stability/{state.name}_absmax"] = float(absmax.item())

    _log_payload(writer, wandb_writer, payload, iteration)
