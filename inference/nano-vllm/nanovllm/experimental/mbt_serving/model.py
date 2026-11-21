# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Single-GPU MBT with shared input bridge and independent per-layer or shared (bank) KV caches.

Reuse the established numerical attention/MoE blocks, not their v1 two-branch
layout. Multi-GPU execution is deliberately rejected until its heterogeneous
sharding/collective contract is qualified.

Shared KV (kv_mode != independent): branch layers keep Q, gate, O, Q-norm and MoE; a bank projects its
input to [K | V] per KV group, K gets the bank's key norm and RoPE once, and branch layers only read it.
Batched execution stacks each depth level's branches so every kernel serves all of them and folds their
Q heads into one GQA group per bank KV head, so one attention call reads each bank row once per level.
Arithmetic matches serial execution bitwise.
"""

import math

import torch
from torch import nn

from ..wide_serving.config import ParallelContext
from ..wide_serving.model import (
    RMSNorm, WideForCausalLM, WideLayer, validate_elementwise_backend,
    validate_router_backend, validate_greedy_head_backend, validate_qk_backend,
)
from ..wide_serving.moe import optimized_moe
from ..wide_serving.ops import fused_gate, fused_gate_unfold, fused_rmsnorm_stacked
from ..wide_serving.ops_qk import norm_rope_heads, norm_rope_store_kv
from ..wide_serving.ops_router import fused_postprocess8_stacked
from .config import MBTConfig

BANK_CACHE_BASE = 100000  # KV cache id of bank b is BANK_CACHE_BASE + b (never a layer id)


def _rope_frequencies(config):
    return 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim))


class KVBank(nn.Module):
    """Writes one shared [K (normed, rotated) | V] cache entry per token; branch layers only read it."""
    def __init__(self, config, bank, backend, *, input_norm):
        super().__init__()
        self.backend, self.cache_id = backend, BANK_CACHE_BASE + bank
        self.head_dim, self.num_kv_heads = config.head_dim, config.branch_hidden_size // (2 * config.head_dim)
        self.input_norm = RMSNorm(config.bank_input_width, config.rms_norm_eps) if input_norm else None
        self.kv_projection = nn.Linear(config.bank_input_width, config.branch_hidden_size, bias=False)
        self.key_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.register_buffer("inv_freq", _rope_frequencies(config), persistent=False)

    def _apply(self, fn, recurse=True):
        saved = self.inv_freq
        super()._apply(fn, recurse)
        self.inv_freq = saved.to(device=self.kv_projection.weight.device, dtype=torch.float32)
        return self

    def forward(self, x, positions, context):
        if self.input_norm is not None:
            x = self.input_norm(x)
        kv = self.kv_projection(x).view(x.shape[0], self.num_kv_heads, 2 * self.head_dim)
        kc, vc = self.backend.backend.cache.layers[self.cache_id]
        norm_rope_store_kv(kv, self.key_norm.weight, positions, self.inv_freq, self.key_norm.eps, kc, vc,
                           context.slot_mapping)


class SharedKVAttention(nn.Module):
    """Gated GQA2 attention with [Q(2D), gate(2D)] rows per KV group; K/V come from a bank."""
    elementwise_backend, qk_backend = "triton", "packed"  # always the Triton kernels

    def __init__(self, config, layer_id, bank, backend):
        super().__init__()
        self.layer_id, self.backend, self.cache_id = layer_id, backend, bank.cache_id
        self.head_dim, self.num_kv_heads = config.head_dim, config.num_key_value_heads
        self.num_heads = 2 * self.num_kv_heads
        self.qkv = nn.Linear(config.hidden_size, self.num_kv_heads * 4 * config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.__dict__["k_norm"] = bank.key_norm  # the bank's K norm, visible to Q/K checks but not a parameter here
        self.out = nn.Linear(self.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.register_buffer("inv_freq", _rope_frequencies(config), persistent=False)

    def _apply(self, fn, recurse=True):
        saved = self.inv_freq
        super()._apply(fn, recurse)
        self.inv_freq = saved.to(device=self.qkv.weight.device, dtype=torch.float32)
        return self

    def forward(self, x, positions, context):
        packed = self.qkv(x).view(x.shape[0], self.num_kv_heads, 4 * self.head_dim)
        q = norm_rope_heads(packed, self.q_norm.weight, positions, self.inv_freq, self.q_norm.eps, heads_per_group=2)
        out = self.backend.backend.attend(q, self.cache_id, context)
        return self.out(fused_gate(out, packed[..., 2 * self.head_dim:]).flatten(1))


def _joint_bank_input(normalized):
    """Native Joint-KV bank input: cat(normalized branch inputs) / sqrt(K), in the activation dtype."""
    return torch.cat(normalized, -1) * (1.0 / math.sqrt(len(normalized)))


class SharedInputBridge(nn.Module):
    """One normalization/projection before the fork, not K separate projections."""
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.projection = nn.Linear(config.hidden_size, config.branch_hidden_size, bias=False)

    def forward(self, x):
        return self.projection(self.norm(x))


class ConcatFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(config.branch_hidden_size, config.rms_norm_eps)
                                   for _ in config.branch_depths)
        self.projection = nn.Linear(len(config.branch_depths) * config.branch_hidden_size,
                                    config.hidden_size, bias=False)

    def forward(self, branches):
        if len(branches) != len(self.norms):
            raise ValueError("fusion received a different number of branches")
        # Pure concat/project: intentionally no additional p or shared-input residual.
        return self.projection(torch.cat([norm(value) for norm, value in zip(self.norms, branches)], -1))


class MBTForCausalLM(WideForCausalLM):
    """v2 layout; inherited helpers implement only layout-independent math/runtime."""
    def __init__(self, config: MBTConfig, parallel=None, attention_backend=None, *,
                 elementwise_backend="torch", router_backend="original",
                 greedy_head_backend="replicated", qk_backend="separate"):
        nn.Module.__init__(self)
        self.parallel = parallel or ParallelContext()
        if self.parallel.world_size != 1:
            raise ValueError("generic MBT v2 is single-GPU only; multi-GPU K-way sharding is not implemented")
        if attention_backend is None:
            raise ValueError("an explicit attention backend is required")
        validate_elementwise_backend(elementwise_backend)
        validate_router_backend(router_backend)
        validate_greedy_head_backend(greedy_head_backend)
        validate_qk_backend(qk_backend)
        self.config = config
        self.branch_parallel = self.parallel
        self._branch_groups = ()
        self._parallel_closed = False
        self.register_buffer("_branch_serial_fence", None, persistent=False)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleDict({str(index): WideLayer(config.layer_config(index), index,
                                                          self.parallel, attention_backend)
                                    for index in range(config.num_hidden_layers)})
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.uses_input_bridge:
            self.input_bridge = SharedInputBridge(config)
        if config.variant == "mbt":
            self.fusion = ConcatFusion(config)
        if config.kv_mode != "independent":
            self.kv_banks = nn.ModuleList(KVBank(config, bank, attention_backend, input_norm=config.kv_mode == "prelude-kv")
                                          for bank in range(config.kv_banks))
            for start, end in config.branch_ranges:
                for depth, index in enumerate(range(start, end)):
                    bank = self.kv_banks[depth if config.kv_mode == "joint-branch-kv" else 0]
                    self.layers[str(index)].attention = SharedKVAttention(config.layer_config(index), index, bank,
                                                                          attention_backend)
        self._streams = None
        self._levels = None
        self.set_elementwise_backend(elementwise_backend)
        self.set_qk_backend(qk_backend)
        self.set_router_backend(router_backend, prepare_cache=False)
        self.set_greedy_head_backend(greedy_head_backend)

    def _branches(self, x, positions, context):
        if self.config.uses_input_bridge:
            x = self.input_bridge(x)
        if self.config.branch_execution == "batched":
            return self.fusion(self._batched_levels(x, positions, context))
        if self.config.kv_mode == "joint-branch-kv":
            return self.fusion(self._joint_serial(x, positions, context))
        if self.config.branch_execution == "streams":
            if not x.is_cuda:
                raise ValueError("CUDA-stream execution requires CUDA tensors")
            if self._streams is None:
                self._streams = tuple(torch.cuda.Stream(device=x.device) for _ in self.config.branch_depths)
            current = torch.cuda.current_stream(x.device)
            outputs = []
            for stream, (start, end) in zip(self._streams, self.config.branch_ranges):
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    value = self._run_layers(start, end, x, positions, context)
                    x.record_stream(stream)
                outputs.append(value)
            for stream in self._streams:
                current.wait_stream(stream)
            for value in outputs:
                value.record_stream(current)
        else:
            # Blocks do not mutate their input; a physical clone is unnecessary
            # in inference. Every branch receives the identical tensor value.
            outputs = [self._run_layers(start, end, x, positions, context)
                       for start, end in self.config.branch_ranges]
        return self.fusion(outputs)

    def _joint_serial(self, x, positions, context):
        """Depth-major: each depth's bank needs every branch's normalized input before any branch reads it."""
        starts = [start for start, _ in self.config.branch_ranges]
        xs = [x] * len(starts)
        for depth in range(self.config.branch_depths[0]):
            layers = [self.layers[str(start + depth)] for start in starts]
            self.kv_banks[depth](_joint_bank_input([layer.input_norm(v) for layer, v in zip(layers, xs)]),
                                 positions, context)
            xs = [layer(v, positions, context) for layer, v in zip(layers, xs)]
        return xs

    @torch.no_grad()
    def _stacked_levels(self):
        """Each depth level's branch weights stacked once (rebuilt when backends or router caches change)."""
        signature = (self.router_cache_signature(), self.elementwise_backend, self.qk_backend, self.router_backend)
        if self._levels is not None and self._levels[0] == signature:
            return self._levels[1]
        if (self.elementwise_backend, self.qk_backend, self.router_backend) != ("triton", "packed", "fused_post8"):
            raise ValueError("batched execution requires elementwise=triton, qk=packed, router=fused_post8")
        levels = []
        for layer_ids in self.config.level_layers:
            group = [self.layers[str(index)] for index in layer_ids]
            for layer in group:
                layer.moe.router.validate_cache()
            def stack(get):
                return torch.stack([get(layer).detach() for layer in group]).contiguous()
            levels.append(dict(
                layer_ids=layer_ids, attention=group[0].attention, eps=group[0].input_norm.eps,
                in_norm=stack(lambda l: l.input_norm.weight), qkv=stack(lambda l: l.attention.qkv.weight),
                q_norm=stack(lambda l: l.attention.q_norm.weight), out=stack(lambda l: l.attention.out.weight),
                post_norm=stack(lambda l: l.post_attention_norm.weight),
                router=stack(lambda l: l.moe.router.weight_fp32), bias=stack(lambda l: l.moe.router.expert_bias),
                down=stack(lambda l: l.moe.down.weight), up=stack(lambda l: l.moe.up.weight),
                w1=torch.cat([l.moe.w1.detach() for l in group]).contiguous(),
                w2=torch.cat([l.moe.w2.detach() for l in group]).contiguous()))
        self._levels = (signature, levels)
        return levels

    def _batched_levels(self, x, positions, context):
        cfg = self.config
        order = sorted(range(len(cfg.branch_depths)), key=lambda b: -cfg.branch_depths[b])  # deepest first
        outputs = [None] * len(order)
        tokens = x.shape[0]
        xs = x.unsqueeze(0).expand(len(order), *x.shape).contiguous()
        repeated = {}
        for level, spec in enumerate(self._stacked_levels()):
            n, attn = len(spec["layer_ids"]), spec["attention"]
            xs = xs[:n]
            if n not in repeated:
                repeated[n] = positions.repeat(n)
            if cfg.kv_mode == "joint-branch-kv":  # the input norm also writes the bank input (= _joint_bank_input)
                hidden, bank_input = fused_rmsnorm_stacked(xs, spec["in_norm"], spec["eps"], 1.0 / math.sqrt(n))
                self.kv_banks[level](bank_input, positions, context)
            else:
                hidden = fused_rmsnorm_stacked(xs, spec["in_norm"], spec["eps"])
            packed = torch.bmm(hidden, spec["qkv"].transpose(1, 2)).view(n * tokens, attn.num_kv_heads, 4 * attn.head_dim)
            q = norm_rope_heads(packed, spec["q_norm"], repeated[n], attn.inv_freq, attn.q_norm.eps,
                                heads_per_group=2, branches=n)  # folded: one GQA group of 2n heads per bank KV head
            out = attn.backend.backend.attend(q, attn.cache_id, context)
            out = fused_gate_unfold(out, packed[..., 2 * attn.head_dim:], n)
            xs = xs + torch.bmm(out.flatten(1).view(n, tokens, -1), spec["out"].transpose(1, 2))
            xs = xs + self._stacked_moe(spec, fused_rmsnorm_stacked(xs, spec["post_norm"], spec["eps"]))
            for g in range(n):
                if cfg.branch_depths[order[g]] == level + 1:
                    outputs[order[g]] = xs[g]
        return outputs

    def _stacked_moe(self, spec, hidden):
        """Per-branch FP32 router GEMMs (one stacked GEMM is not bitwise F.linear) on forked streams so these
        latency-bound GEMVs overlap; then one top-k, one post8 and one MoE call for all branches."""
        n, tokens, _ = hidden.shape
        experts = self.config.num_experts
        h32 = hidden.float()
        logits = torch.empty((n, tokens, experts), device=hidden.device, dtype=torch.float32)
        current = torch.cuda.current_stream(hidden.device)
        streams = self.__dict__.setdefault("_router_streams", [])
        streams += [torch.cuda.Stream(device=hidden.device) for _ in range(n - 1 - len(streams))]
        for g in range(1, n):
            streams[g - 1].wait_stream(current)
            with torch.cuda.stream(streams[g - 1]):
                torch.mm(h32[g], spec["router"][g].t(), out=logits[g])
        torch.mm(h32[0], spec["router"][0].t(), out=logits[0])
        for g in range(1, n):
            current.wait_stream(streams[g - 1])
        selected = torch.topk(logits + spec["bias"].unsqueeze(1), self.config.num_experts_per_tok + 1, dim=-1,
                              sorted=True).indices
        ids, probabilities = fused_postprocess8_stacked(logits, selected.view(n * tokens, -1), tokens)
        for g, layer_id in enumerate(spec["layer_ids"]):  # keep Router forward hooks (parity tooling) observable
            router = self.layers[str(layer_id)].moe.router
            for hook in router._forward_hooks.values():
                hook(router, (hidden[g],), (ids.view(n, tokens, -1)[g] - g * experts,
                                            probabilities.view(n, tokens, -1)[g]))
        latent = torch.bmm(hidden, spec["down"].transpose(1, 2)).view(n * tokens, -1)
        combined = optimized_moe(latent, ids, probabilities, spec["w1"], spec["w2"],
                                 backend=self.config.moe_backend, tokens=tokens)
        return torch.bmm(combined.view(n, tokens, -1), spec["up"].transpose(1, 2))

    def cache_specs(self):
        if self.config.kv_mode == "independent":
            return super().cache_specs()
        # Branch layers own no KV; each bank is one cache shared by all of its readers.
        return ([dict(layer_id=int(key), num_kv_heads=layer.attention.num_kv_heads, head_dim=self.config.head_dim)
                 for key, layer in self.layers.items() if not self.config.layer_role(int(key)).startswith("branch_")]
                + [dict(layer_id=bank.cache_id, num_kv_heads=bank.num_kv_heads, head_dim=self.config.head_dim)
                   for bank in self.kv_banks])

    def forward(self, input_ids, positions, context=None):
        if self._parallel_closed:
            raise RuntimeError("the model has been closed")
        x = self.embedding(input_ids)
        if self.config.variant == "standard":
            x = self._run_layers(0, self.config.num_hidden_layers, x, positions, context)
        else:
            x = self._run_layers(0, self.config.prelude, x, positions, context)
            if self.config.kv_mode == "prelude-kv":
                self.kv_banks[0](x, positions, context)  # written before any branch layer reads it
            x = self._branches(x, positions, context)
            x = self._run_layers(self.config.prelude + sum(self.config.branch_depths),
                                 self.config.num_hidden_layers, x, positions, context)
        return self.norm(x)

    def parallel_layout(self):
        return dict(format_version=2, rank=0, world_size=1,
                    branch_execution=self.config.branch_execution,
                    branch_depths=list(self.config.branch_depths),
                    shared_input_bridge=self.config.uses_input_bridge,
                    attention="full causal GQA2 + RoPE + QK norm + sigmoid output gate",
                    fusion="per-branch RMSNorm -> concat -> projection; no fusion residual"
                           if self.config.variant == "mbt" else None,
                    layers=[dict(layer_id=int(key), role=self.config.layer_role(int(key)),
                                 hidden_size=layer.moe.config.hidden_size,
                                 latent_size=layer.moe.config.latent_size,
                                 intermediate_size=layer.moe.config.intermediate_size,
                                 num_kv_heads=layer.attention.num_kv_heads,
                                 tp_rank=0, tp_size=1)
                            for key, layer in self.layers.items()])

    def shard_export_tensor(self, name, value):
        # All canonical tensors are local in v2's single-GPU contract.
        return value
