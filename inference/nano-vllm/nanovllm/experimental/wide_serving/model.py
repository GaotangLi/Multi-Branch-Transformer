# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Wide (multi-branch) Standard and Concat models with an injected paged-attention backend.

Checkpoint tensors retain native grouped Q/gate/K/V packing. Physical layer
IDs are stable across serial, CUDA-stream, and TP1/2/4/8 branch-parallel execution.
"""

from datetime import timedelta
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from .config import ParallelContext, WideConfig


def validate_elementwise_backend(backend):
    if backend not in ("torch", "triton"):
        raise ValueError("elementwise_backend must be torch or triton")
    return backend


def validate_router_backend(backend):
    if backend not in ("original", "cached", "fused", "fused_post8"):
        raise ValueError("router_backend must be original, cached, fused, or fused_post8")
    return backend


def validate_qk_backend(backend):
    if backend not in ("separate", "packed"):
        raise ValueError("qk_backend must be separate or packed")
    return backend


def validate_greedy_head_backend(backend):
    if backend not in ("replicated", "vocab_sharded"):
        raise ValueError("greedy_head_backend must be replicated or vocab_sharded")
    return backend


def select_global_greedy(gathered, vocab_size):
    """Reduce [rank, token, (value, global ID)] with lowest-ID tie breaking.

    Payloads use FP64: BF16/FP32 logits and vocabulary IDs are represented
    exactly. NaNs follow torch.argmax's first-NaN rule, without host reads.
    """
    values, ids = gathered[..., 0], gathered[..., 1].to(torch.int64)
    maximum = values.max(dim=0).values
    matches = (values == maximum) | (values.isnan() & maximum.isnan())
    winner = torch.where(matches, ids, vocab_size).min(dim=0).values
    return maximum, winner


class RMSNorm(nn.Module):
    def __init__(self, width, eps, elementwise_backend="torch"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.elementwise_backend = validate_elementwise_backend(elementwise_backend)

    def forward(self, x):
        if self.elementwise_backend == "triton" and x.is_cuda:
            from .ops import fused_rmsnorm
            return fused_rmsnorm(x, self.weight, self.eps)
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(x.dtype)


def qb_route(x, weight, expert_bias, topk, renorm_sum):
    """Native QB inference: saved bias selects; unbiased sigmoid weights combine."""
    logits = F.linear(x.float(), weight.float())
    indices = torch.topk(logits + expert_bias.float(), topk + 1, dim=-1, sorted=True).indices[..., :topk]
    probabilities = torch.sigmoid(logits.gather(-1, indices))
    probabilities = probabilities * (renorm_sum / (probabilities.sum(-1, keepdim=True) + 1e-9))
    return indices.contiguous(), probabilities.contiguous()


def apply_rope(x, positions, inv_freq):
    phase = positions.float().unsqueeze(-1) * inv_freq.float()
    cosine, sine = phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(x.dtype)


def _sum_parallel(x, parallel):
    if parallel.world_size > 1:
        dist.all_reduce(x, group=parallel.group)
    return x


def create_branch_context(parallel):
    """Create both Concat subgroups in the same order on every WORLD rank.

    No process groups are constructed for metadata-only CPU/meta model inspection.
    A distributed outer context must span WORLD; subgroup layer contexts are
    created here, never recursively by each layer or only by group members.
    """
    local = parallel.branch_context()
    if parallel.world_size <= 2 or not dist.is_initialized():
        return local, ()
    if dist.get_world_size() != parallel.world_size:
        raise ValueError("Concat outer context must span the complete distributed WORLD")
    if dist.get_world_size(parallel.group) != parallel.world_size or dist.get_rank(parallel.group) != parallel.rank:
        raise ValueError("ParallelContext does not match its outer process group")
    parent = parallel.group if parallel.group is not None else dist.group.WORLD
    global_ranks = dist.get_process_group_ranks(parent)
    groups = tuple(dist.new_group(ranks=[global_ranks[rank] for rank in ranks],
                                  backend=dist.get_backend(parent), timeout=timedelta(minutes=5),
                                  use_local_synchronization=False)
                   for ranks in parallel.branch_rank_groups())
    return parallel.branch_context(groups[parallel.branch_owner]), groups


class WideAttention(nn.Module):
    def __init__(self, config, layer_id, parallel, backend):
        super().__init__()
        self.layer_id, self.parallel, self.backend = layer_id, parallel, backend
        self.elementwise_backend = "torch"
        self.qk_backend = "separate"
        self.head_dim = config.head_dim
        self.kv_start, self.kv_end = parallel.bounds(config.num_key_value_heads)
        self.num_kv_heads = self.kv_end - self.kv_start
        if self.num_kv_heads < 1:
            raise ValueError("this layout requires at least one KV group per attention rank")
        self.num_heads = 2 * self.num_kv_heads
        self.qkv = nn.Linear(config.hidden_size, self.num_kv_heads * 6 * config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.out = nn.Linear(self.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta ** (
            torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)), persistent=False)

    def _apply(self, fn, recurse=True):
        saved = self.inv_freq
        super()._apply(fn, recurse)
        self.inv_freq = saved.to(device=self.qkv.weight.device, dtype=torch.float32)
        return self

    def forward(self, x, positions, context):
        shape = (x.shape[0], self.num_kv_heads, 6 * self.head_dim)
        packed = self.qkv(x).view(shape)
        q, gate, k, v = packed.split((2*self.head_dim, 2*self.head_dim, self.head_dim, self.head_dim), -1)
        use_fused = self.elementwise_backend == "triton" and x.is_cuda
        if use_fused:
            from .ops import fused_gate, fused_rope
        if self.qk_backend == "packed":
            from .ops_qk import qk_fused
            q, k = qk_fused(packed, self.q_norm.weight, self.k_norm.weight,
                            positions, self.inv_freq, self.q_norm.eps)
        else:
            q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim))
            k = self.k_norm(k)
        if self.qk_backend == "separate" and use_fused:
            q, k = fused_rope(q, positions, self.inv_freq), fused_rope(k, positions, self.inv_freq)
        elif self.qk_backend == "separate":
            q, k = apply_rope(q, positions, self.inv_freq), apply_rope(k, positions, self.inv_freq)
        out = self.backend(q, k, v.contiguous(), self.layer_id, context)
        if use_fused:
            out = fused_gate(out, gate)
        else:
            out = (out.float() * torch.sigmoid(gate.reshape_as(out).float())).to(x.dtype)
        return _sum_parallel(self.out(out.flatten(1)), self.parallel)


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))
        self.register_buffer("weight_fp32", None, persistent=False)
        self.topk, self.renorm_sum = config.num_experts_per_tok, config.router_renorm_sum
        self.backend = "original"
        self.cache_revision = 0
        self._cache_source = None
        self._cache_tracks_version = False

    def invalidate_cache(self):
        """Call BEFORE manual weight mutation; graphs using this cache are stale.

        Inference tensors have no version counter, so unsupported .data writes
        cannot be detected automatically. Loading and _apply invoke this method.
        """
        self.weight_fp32 = None
        self._cache_source = None
        self._cache_tracks_version = False
        self.cache_revision += 1

    def _source_signature(self):
        return (id(self.weight), self.weight.data_ptr(), self.weight.device, self.weight.dtype,
                self.weight._version if self._cache_tracks_version else None)

    @torch.no_grad()
    def prepare_cache(self):
        """Explicit load/move boundary, never called by forward or graph capture."""
        if self.backend == "original":
            return
        if self.weight.is_meta:
            raise RuntimeError("router cache requires materialized loaded weights")
        if self.weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare router caches before CUDA graph capture")
        from .ops_router import prepare_router_weight
        self.invalidate_cache()
        self.weight_fp32 = prepare_router_weight(self.weight)
        try:
            self.weight._version
            self._cache_tracks_version = True
        except RuntimeError:
            # Parameters created inside inference_mode have no version counter.
            self._cache_tracks_version = False
        self._cache_source = self._source_signature()

    def validate_cache(self):
        """Host-metadata check at runner boundaries, never a hot-path rebuild."""
        if self.backend != "original":
            if self.weight_fp32 is None or self._cache_source != self._source_signature():
                raise RuntimeError("router cache is missing or stale; invalidate and prepare it before serving")

    def set_backend(self, backend, *, prepare_cache=True):
        backend = validate_router_backend(backend)
        self.validate_backend_shape(backend)
        self.invalidate_cache()
        self.backend = backend
        if prepare_cache:
            self.prepare_cache()

    def validate_backend_shape(self, backend):
        if backend == 'fused_post8' and (self.weight.shape[0] != 64 or self.topk != 8 or self.renorm_sum != 2.5):
            raise ValueError('fused_post8 requires E64/K8 and renorm_sum=2.5')

    def _apply(self, fn, recurse=True):
        # Module.to(bfloat16) must never quantize persistent QB bias.
        saved = self.expert_bias
        was_prepared = self.weight_fp32 is not None
        self.invalidate_cache()  # Exclude the FP32 cache from blanket dtype casts.
        super()._apply(fn, recurse)
        self.expert_bias = saved.to(device=self.weight.device, dtype=torch.float32)
        if was_prepared and not self.weight.is_meta:
            self.prepare_cache()  # Rebuild from the actual post-cast parameter.
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        self.invalidate_cache()
        super()._load_from_state_dict(*args, **kwargs)
        self.prepare_cache()

    def forward(self, x):
        if self.backend == "original":
            return qb_route(x, self.weight, self.expert_bias, self.topk, self.renorm_sum)
        if self.weight_fp32 is None:
            raise RuntimeError("router cache is not prepared; no forward-time cache allocation is allowed")
        from .ops_router import cached_qb_route
        return cached_qb_route(x, self.weight_fp32, self.expert_bias, self.topk, self.renorm_sum,
                               post_backend={"cached": "torch", "fused": "triton",
                                             "fused_post8": "triton_post8"}[self.backend])


class WideMoE(nn.Module):
    def __init__(self, config, parallel):
        super().__init__()
        self.config, self.parallel = config, parallel
        if config.intermediate_size % parallel.world_size:
            raise ValueError("expert FFN size must divide TP size")
        self.ffn_start, self.ffn_end = parallel.bounds(config.intermediate_size)
        width = self.ffn_end - self.ffn_start
        self.router = Router(config)
        self.down = nn.Linear(config.hidden_size, config.latent_size, bias=False)
        self.up = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.w1 = nn.Parameter(torch.empty(config.num_experts, 2 * width, config.latent_size))
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.latent_size, width))

    def forward(self, x):
        from .moe import optimized_moe
        ids, probabilities = self.router(x)
        latent = self.down(x)
        combined = optimized_moe(latent, ids, probabilities, self.w1, self.w2,
                                 backend=self.config.moe_backend)
        return self.up(_sum_parallel(combined, self.parallel))


class WideLayer(nn.Module):
    def __init__(self, config, layer_id, parallel, backend):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = WideAttention(config, layer_id, parallel, backend)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.moe = WideMoE(config, parallel)

    def forward(self, x, positions, context):
        x = x + self.attention(self.input_norm(x), positions, context)
        return x + self.moe(self.post_attention_norm(x))


class Fusion(nn.Module):
    def __init__(self, config, parallel):
        super().__init__()
        self.parallel = parallel
        self.branch_owner = parallel.branch_owner
        self.input_start, self.input_end = parallel.branch_context().bounds(config.hidden_size)
        if parallel.world_size == 1 or self.branch_owner == 0:
            self.norm_a = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if parallel.world_size == 1 or self.branch_owner == 1:
            self.norm_b = RMSNorm(config.hidden_size, config.rms_norm_eps)
        columns = 2*config.hidden_size if parallel.world_size == 1 else self.input_end-self.input_start
        self.projection = nn.Linear(columns, config.hidden_size, bias=False)

    def forward(self, a, b=None):
        if self.parallel.world_size == 1:
            return self.projection(torch.cat((self.norm_a(a), self.norm_b(b)), -1))
        norm = self.norm_a if self.branch_owner == 0 else self.norm_b
        # Branch outputs are subgroup-replicated. Each rank contributes disjoint
        # Wf input columns, so the WORLD sum includes each branch exactly once.
        normalized_slice = norm(a)[:, self.input_start:self.input_end]
        return _sum_parallel(self.projection(normalized_slice), self.parallel)


class WideForCausalLM(nn.Module):
    def __init__(self, config: WideConfig, parallel=None, attention_backend=None, *, elementwise_backend="torch",
                 router_backend="original", greedy_head_backend="replicated", qk_backend="separate"):
        super().__init__()
        validate_elementwise_backend(elementwise_backend)
        validate_router_backend(router_backend)
        validate_greedy_head_backend(greedy_head_backend)
        validate_qk_backend(qk_backend)
        if attention_backend is None:
            raise ValueError("an explicit attention backend is required")
        self.config = config
        self.parallel = parallel or ParallelContext()
        self.branch_parallel = self.parallel.branch_context()
        self._branch_groups = ()
        self._parallel_closed = False
        if config.num_key_value_heads < self.parallel.world_size:
            raise ValueError("KV replication is not implemented; each outer rank requires a KV group")
        if config.intermediate_size % self.parallel.world_size:
            raise ValueError("expert FFN width must divide the outer world size")
        if config.branch_execution == "streams" and self.parallel.world_size != 1:
            raise ValueError("CUDA streams apply to the one-GPU Concat control only")
        if config.variant == "concat":
            self.branch_parallel, self._branch_groups = create_branch_context(self.parallel)
        serialize_branches = (config.variant == "concat"
                              and config.branch_execution == "distributed_serial"
                              and self.parallel.world_size > 1)
        # Runtime-only ordering token, never a checkpoint tensor or model value.
        # A zero SUM stays zero across every forward and CUDA graph replay, so
        # no reset/allocation is needed in the captured path. Integer dtype also
        # survives model-wide floating dtype casts without changing this token.
        self.register_buffer("_branch_serial_fence",
                             torch.zeros(1, dtype=torch.int32) if serialize_branches else None,
                             persistent=False)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleDict()
        for index in range(config.num_hidden_layers):
            branch = self._branch_owner(index)
            if branch is None or self.parallel.world_size == 1 or branch == self.parallel.branch_owner:
                layout = self.parallel if branch is None else self.branch_parallel
                self.layers[str(index)] = WideLayer(config, index, layout, attention_backend)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.variant == "concat":
            self.fusion = Fusion(config, self.parallel)
        self._streams = None
        self.set_elementwise_backend(elementwise_backend)
        self.set_qk_backend(qk_backend)
        # Weights are not loaded yet. The strict loader prepares opt-in caches.
        self.set_router_backend(router_backend, prepare_cache=False)
        self.set_greedy_head_backend(greedy_head_backend)

    def set_greedy_head_backend(self, backend):
        """Decode-only runtime option; tied embedding/full-logit API stay intact.

        Change before CUDA graph capture. Each outer rank owns a contiguous
        vocabulary slice for greedy projection, including in a TP2 replica.
        This changes no checkpoint tensor, weight storage, or model arithmetic
        before the head. A different GEMM shape requires empirical BF16 parity.
        """
        backend = validate_greedy_head_backend(backend)
        if backend == "vocab_sharded" and self.config.vocab_size < self.parallel.world_size:
            raise ValueError("vocab_sharded head requires nonempty vocabulary slices")
        if self.config.vocab_size >= 2**53:
            raise ValueError("vocabulary IDs must be exactly representable in the FP64 gather payload")
        self.greedy_head_backend = backend

    def set_router_backend(self, backend, *, prepare_cache=True):
        """Runtime-only choice; changing it invalidates existing CUDA graphs."""
        backend = validate_router_backend(backend)
        # Validate every layer before changing any runtime state or cache.
        for layer in self.layers.values():
            layer.moe.router.validate_backend_shape(backend)
        self.router_backend = backend
        for layer in self.layers.values():
            layer.moe.router.set_backend(backend, prepare_cache=prepare_cache)

    def invalidate_router_caches(self):
        """Required before manual parameter updates, including inference tensors."""
        for layer in self.layers.values():
            layer.moe.router.invalidate_cache()

    def prepare_router_caches(self):
        for layer in self.layers.values():
            layer.moe.router.prepare_cache()

    def validate_router_caches(self):
        for layer in self.layers.values():
            layer.moe.router.validate_cache()

    def router_cache_signature(self):
        """Host-only graph-generation token; no CUDA tensor reads or allocation."""
        return tuple((layer.moe.router.backend, layer.moe.router.cache_revision)
                     for layer in self.layers.values())

    def set_qk_backend(self, backend):
        """Runtime-only opt-in; preserve native tensors and BF16 norm rounding.

        Release and recapture prefill/decode graphs after changing this flag.
        CUDA BF16/contiguity are checked by the kernel at the forward boundary.
        """
        backend = validate_qk_backend(backend)
        if backend == 'packed':
            if self.elementwise_backend != 'triton':
                raise ValueError('packed Q/K requires elementwise_backend=triton')
            for layer in self.layers.values():
                attn = layer.attention
                if attn.head_dim != 64 or attn.q_norm.eps != attn.k_norm.eps:
                    raise ValueError('packed Q/K requires head_dim=64 and equal Q/K norm epsilon')
        self.qk_backend = backend
        for layer in self.layers.values():
            layer.attention.qk_backend = backend

    def qk_signature(self):
        """Host-only graph token; includes effective per-layer operation flags."""
        return (self.qk_backend, tuple((layer.attention.qk_backend,
                layer.attention.elementwise_backend, layer.attention.q_norm.elementwise_backend,
                layer.attention.k_norm.elementwise_backend, layer.attention.q_norm.eps,
                layer.attention.k_norm.eps) for layer in self.layers.values()))

    def set_elementwise_backend(self, backend):
        """Change runtime operations without changing architecture or saved tensors.

        Call before CUDA graph capture. Captured graphs retain their original
        operations and must be recaptured after changing this choice.
        """
        backend = validate_elementwise_backend(backend)
        if backend != 'triton' and getattr(self, 'qk_backend', 'separate') == 'packed':
            raise ValueError('switch qk_backend to separate before disabling Triton elementwise operations')
        self.elementwise_backend = backend
        for module in self.modules():
            if isinstance(module, (RMSNorm, WideAttention)):
                module.elementwise_backend = backend

    def _branch_owner(self, index):
        cfg = self.config
        if cfg.variant != "concat" or not cfg.prelude <= index < cfg.prelude + 2*cfg.core:
            return None
        return int(index >= cfg.prelude + cfg.core)

    def cache_specs(self):
        return [dict(layer_id=int(key), num_kv_heads=layer.attention.num_kv_heads,
                     head_dim=self.config.head_dim) for key, layer in self.layers.items()]

    def parallel_layout(self):
        return dict(rank=self.parallel.rank, world_size=self.parallel.world_size,
                    qk_backend=self.qk_backend, router_backend=self.router_backend,
                    greedy_head=dict(backend=self.greedy_head_backend,
                        vocabulary_bounds=list(self.parallel.bounds(self.config.vocab_size))
                            if self.greedy_head_backend == "vocab_sharded" else [0, self.config.vocab_size],
                        embedding_storage="replicated and tied; no parameter-memory reduction",
                        collective="one outer-group FP64 all-gather of (maximum, global ID)"
                            if self.greedy_head_backend == "vocab_sharded" and self.parallel.world_size > 1 else "none",
                        scope="cached greedy decode only; full-logit/prefill APIs remain replicated"),
                    branch_owner=self.parallel.branch_owner if self.config.variant == "concat" and self.parallel.world_size > 1 else None,
                    branch_world_size=self.branch_parallel.world_size if self.config.variant == "concat" else None,
                    branch_local_rank=self.branch_parallel.rank if self.config.variant == "concat" else None,
                    serialization_control=dict(
                        implementation="preallocated int32 zero SUM all-reduce + block_current_stream",
                        extra_all_reduces_per_forward=2 if self._branch_serial_fence is not None else 0,
                        payload_bytes_per_collective=4 if self._branch_serial_fence is not None else 0,
                        group="outer",
                        timing="fence overhead included, not separately subtracted"),
                    layers=[dict(layer_id=int(key), tp_rank=layer.attention.parallel.rank,
                                 tp_size=layer.attention.parallel.world_size,
                                 kv_start=layer.attention.kv_start, kv_end=layer.attention.kv_end,
                                 ffn_start=layer.moe.ffn_start, ffn_end=layer.moe.ffn_end)
                            for key, layer in self.layers.items()])

    def close_parallel_groups(self):
        """After graph reset/synchronization, close owned subgroups before WORLD."""
        if self._parallel_closed:
            return
        groups, self._branch_groups = self._branch_groups, ()
        for group in groups:
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)
        self._parallel_closed = True

    def _run_layers(self, start, end, x, positions, context):
        for index in range(start, end):
            x = self.layers[str(index)](x, positions, context)
        return x

    def _serialize_branch_boundary(self):
        """Device-only ordering fence for the same-model serialized control.

        ProcessGroupNCCL's barrier blocks the CPU and fails CUDA graph capture.
        An ordinary asynchronous all-reduce instead waits for earlier work on
        this rank's current stream; block_current_stream inserts the return
        dependency without a CPU wait. Consequently no rank can start the next
        branch/fusion until every rank has completed the preceding branch.
        PyTorch 2.12 Work.block_current_stream is required, with no fallback.
        These two extra tiny collectives are part of this control's timing.
        """
        if self._branch_serial_fence is None:
            raise RuntimeError("distributed branch serialization fence is not configured")
        work = dist.all_reduce(self._branch_serial_fence, op=dist.ReduceOp.SUM,
                               group=self.parallel.group, async_op=True)
        work.block_current_stream()

    def _branches(self, x, positions, context):
        cfg = self.config
        start, middle, end = cfg.prelude, cfg.prelude + cfg.core, cfg.prelude + 2*cfg.core
        if self.parallel.world_size > 1:
            if cfg.branch_execution == "distributed_serial":
                # Same weights/partition/kernels as parallel branches; only
                # execution order changes. Fence overhead remains in timings.
                value = None
                for owner in (0, 1):
                    if self.parallel.branch_owner == owner:
                        value = self._run_layers(start + owner*cfg.core, middle + owner*cfg.core,
                                                 x, positions, context)
                    self._serialize_branch_boundary()
            else:
                owner = self.parallel.branch_owner
                value = self._run_layers(start + owner*cfg.core, middle + owner*cfg.core,
                                         x, positions, context)
            return self.fusion(value)
        if cfg.branch_execution == "streams":
            if not x.is_cuda:
                raise ValueError("CUDA-stream execution requires CUDA tensors")
            if self._streams is None:
                self._streams = (torch.cuda.Stream(device=x.device), torch.cuda.Stream(device=x.device))
            current = torch.cuda.current_stream(x.device)
            a_stream, b_stream = self._streams
            a_stream.wait_stream(current)
            b_stream.wait_stream(current)
            with torch.cuda.stream(a_stream):
                a = self._run_layers(start, middle, x, positions, context)
                x.record_stream(a_stream)
            with torch.cuda.stream(b_stream):
                b = self._run_layers(middle, end, x, positions, context)
                x.record_stream(b_stream)
            current.wait_stream(a_stream)
            current.wait_stream(b_stream)
            a.record_stream(current)
            b.record_stream(current)
        else:
            a = self._run_layers(start, middle, x, positions, context)
            b = self._run_layers(middle, end, x, positions, context)
        return self.fusion(a, b)

    def forward(self, input_ids, positions, context=None):
        if self._parallel_closed:
            raise RuntimeError("the model's parallel groups have been closed")
        x = self.embedding(input_ids)
        cfg = self.config
        if cfg.variant == "standard":
            x = self._run_layers(0, cfg.num_hidden_layers, x, positions, context)
        else:
            x = self._run_layers(0, cfg.prelude, x, positions, context)
            x = self._branches(x, positions, context)
            x = self._run_layers(cfg.prelude + 2*cfg.core, cfg.num_hidden_layers, x, positions, context)
        return self.norm(x)

    def compute_logits(self, hidden_states):
        return F.linear(hidden_states, self.embedding.weight)

    def compute_local_logits(self, hidden_states):
        """Debug/oracle API for this rank's vocabulary slice; no communication."""
        start, end = self.parallel.bounds(self.config.vocab_size)
        return F.linear(hidden_states, self.embedding.weight[start:end])

    def compute_greedy_max(self, hidden_states):
        """Return exact local/shard-global maximum and ID, without full gathering.

        The only collective uses the OUTER model group, never a Concat branch
        group or implicitly WORLD when an explicit replica group was supplied.
        All allocations/collectives are CUDA-graph capturable after warmup.
        """
        if self._parallel_closed:
            raise RuntimeError("the model's parallel groups have been closed")
        if self.greedy_head_backend == "replicated" or self.parallel.world_size == 1:
            return self.compute_logits(hidden_states).max(dim=-1)
        if hidden_states.ndim < 2 or hidden_states.numel() == 0:
            raise ValueError("sharded greedy head requires nonempty [..., hidden] input")
        if not dist.is_initialized():
            raise RuntimeError("sharded greedy head requires the initialized outer process group")
        group = self.parallel.group
        if (dist.get_world_size(group) != self.parallel.world_size
                or dist.get_rank(group) != self.parallel.rank):
            raise RuntimeError("sharded greedy head process group does not match ParallelContext")
        local_max, local_id = self.compute_local_logits(hidden_states).max(dim=-1)
        start, _ = self.parallel.bounds(self.config.vocab_size)
        payload = torch.stack((local_max.reshape(-1).double(),
                               (local_id.reshape(-1) + start).double()), dim=-1)
        gathered = torch.empty((self.parallel.world_size * payload.shape[0], 2),
                               dtype=payload.dtype, device=payload.device)
        dist.all_gather_into_tensor(gathered, payload, group=group)
        value, token = select_global_greedy(
            gathered.view(self.parallel.world_size, payload.shape[0], 2), self.config.vocab_size)
        return value.to(local_max.dtype).view_as(local_max), token.view_as(local_id)

    def compute_greedy(self, hidden_states):
        if self.greedy_head_backend == "replicated":
            # Preserve the original default's operations, not just its result.
            return self.compute_logits(hidden_states).argmax(dim=-1)
        return self.compute_greedy_max(hidden_states)[1]

    def shard_export_tensor(self, name, value):
        """Slice canonical full weights for this rank; called by the strict loader."""
        if name.startswith("layers."):
            _, layer_id, suffix = name.split(".", 2)
            layer = self.layers[layer_id]
            attn, moe = layer.attention, layer.moe
            if suffix == "attention.qkv.weight":
                start, end = attn.kv_start * 6*self.config.head_dim, attn.kv_end * 6*self.config.head_dim
                value = value[start:end]
            elif suffix == "attention.out.weight":
                value = value[:, attn.kv_start*2*self.config.head_dim:attn.kv_end*2*self.config.head_dim]
            elif suffix == "moe.w1":
                gate, linear = value.chunk(2, dim=1)
                value = torch.cat((gate[:, moe.ffn_start:moe.ffn_end], linear[:, moe.ffn_start:moe.ffn_end]), 1)
            elif suffix == "moe.w2":
                value = value[:, :, moe.ffn_start:moe.ffn_end]
        elif name == "fusion.projection.weight" and self.parallel.world_size > 1:
            branch_start = self.parallel.branch_owner * self.config.hidden_size
            value = value[:, branch_start+self.fusion.input_start:branch_start+self.fusion.input_end]
        return value
