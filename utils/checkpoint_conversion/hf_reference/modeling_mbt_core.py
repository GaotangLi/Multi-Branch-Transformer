"""PyTorch reference backbone for Multi-Branch Transformer checkpoints.

No Megatron, Transformer Engine, distributed initialization or custom CUDA
extensions are required. Parameter layout follows the released weights;
Hugging Face integration lives in ``modeling_mbt.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


KVPair = tuple[torch.Tensor, torch.Tensor]
LegacyCache = tuple[KVPair, ...]


@dataclass
class MBTCoreOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Optional[LegacyCache] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: None = None


class MBTRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(hidden.dtype)


class MBTRouter(nn.Module):
    """Quantile-balancing router with its learned, persistent FP32 bias.

    The bias selects experts; combination weights use unbiased sigmoid logits.
    """
    def __init__(self, width: int, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_experts, width))
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))
        self.topk = config.num_experts_per_tok
        self.renorm_sum = config.router_renorm_sum
        nn.init.normal_(self.weight, std=getattr(config, "initializer_range", 0.02))

    def _apply(self, fn, recurse=True):
        # Preserve the original values, not a BF16-roundtripped copy of them.
        original_bias = self.expert_bias
        result = super()._apply(fn, recurse=recurse)
        if original_bias.is_meta:
            # Materialization with to_empty has no source values to preserve.
            self.expert_bias = self.expert_bias.to(dtype=torch.float32)
        else:
            self.expert_bias = original_bias.to(device=self.weight.device, dtype=torch.float32)
        return result

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # This remains FP32 even inside an outer autocast context.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = F.linear(hidden.float(), self.weight.float())
            indices = torch.topk(
                logits + self.expert_bias.float(), self.topk + 1, dim=-1, sorted=True
            ).indices[..., :self.topk]
            probabilities = torch.sigmoid(logits.gather(-1, indices))
            probabilities = probabilities * (
                self.renorm_sum / (probabilities.sum(-1, keepdim=True) + 1e-9)
            )
        return indices, probabilities


class MBTMoE(nn.Module):
    """Latent MoE with native weighted-SwiGLU BF16 rounding order."""
    def __init__(self, width: int, latent: int, intermediate: int, config):
        super().__init__()
        self.router = MBTRouter(width, config)
        self.down = nn.Linear(width, latent, bias=False)
        self.up = nn.Linear(latent, width, bias=False)
        self.w1 = nn.Parameter(torch.empty(config.num_experts, 2 * intermediate, latent))
        self.w2 = nn.Parameter(torch.empty(config.num_experts, latent, intermediate))
        nn.init.normal_(self.w1, std=getattr(config, "initializer_range", 0.02))
        nn.init.normal_(self.w2, std=getattr(config, "initializer_range", 0.02))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        indices, probabilities = self.router(flat)
        latent = self.down(flat)
        routed = latent.new_empty((flat.shape[0], indices.shape[-1], latent.shape[-1]))
        # Reference implementation: each expert sees only its selected tokens.
        # No [tokens, experts, weights] expansion and no custom kernels.
        for expert in range(self.w1.shape[0]):
            token_ids, choices = torch.where(indices == expert)
            if token_ids.numel() == 0:
                continue
            projected = F.linear(latent[token_ids], self.w1[expert])
            gate, up = projected.float().chunk(2, dim=-1)
            activated = F.silu(gate) * up
            activated = (activated * probabilities[token_ids, choices, None]).to(projected.dtype)
            routed[token_ids, choices] = F.linear(activated, self.w2[expert])
        combined = routed.float().sum(dim=1).to(latent.dtype)
        return self.up(combined).reshape(shape)


def apply_rope(hidden: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Half-rotation RoPE for [batch, tokens, heads, head_dim]."""
    width = hidden.shape[-1]
    frequencies = 1.0 / (theta ** (
        torch.arange(0, width, 2, device=hidden.device, dtype=torch.float32) / width
    ))
    angles = positions.float().unsqueeze(-1) * frequencies
    cosine = angles.cos().unsqueeze(-2)
    sine = angles.sin().unsqueeze(-2)
    first, second = hidden.float().chunk(2, dim=-1)
    rotated = torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)
    return rotated.to(hidden.dtype)


def append_cache(key: torch.Tensor, value: torch.Tensor, past: Optional[KVPair]) -> KVPair:
    """Convert [B,T,H,D] projections to the [B,H,T,D] cache layout."""
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    if past is not None:
        old_key, old_value = past
        key = torch.cat((old_key, key), dim=-2)
        value = torch.cat((old_value, value), dim=-2)
    return key, value


class MBTAttention(nn.Module):
    def __init__(self, width: int, query_heads: int, kv_heads: int, config, *, joint: bool):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = config.head_dim
        self.queries_per_group = query_heads // kv_heads
        self.joint = joint
        self.rope_theta = config.rope_theta
        projection_heads = 2 * query_heads + (0 if joint else 2 * kv_heads)
        self.qkv = nn.Linear(width, projection_heads * self.head_dim, bias=False)
        self.q_norm = MBTRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = None if joint else MBTRMSNorm(self.head_dim, config.rms_norm_eps)
        self.out = nn.Linear(query_heads * self.head_dim, width, bias=False)

    def forward(self, normalized: torch.Tensor, positions: torch.Tensor,
                mask: torch.Tensor, *, past: Optional[KVPair] = None,
                shared_kv: Optional[KVPair] = None) -> tuple[torch.Tensor, KVPair]:
        batch, tokens, _ = normalized.shape
        packed = self.qkv(normalized).view(batch, tokens, self.kv_heads, -1)
        query_width = self.queries_per_group * self.head_dim
        if self.joint:
            if shared_kv is None or past is not None:
                raise ValueError("Joint-KV attention expects one preconstructed shared cache")
            query, gate = packed.split(query_width, dim=-1)
            kv = shared_kv
        else:
            if shared_kv is not None:
                raise ValueError("Independent attention cannot consume a shared cache")
            query, gate, key, value = packed.split(
                (query_width, query_width, self.head_dim, self.head_dim), dim=-1
            )
            key = apply_rope(self.k_norm(key), positions, self.rope_theta)
            kv = append_cache(key, value, past)
        query = query.reshape(batch, tokens, self.query_heads, self.head_dim)
        query = apply_rope(self.q_norm(query), positions, self.rope_theta).transpose(1, 2)
        key, value = kv
        # Repeating head views is the portable GQA reference. Efficient serving
        # engines can fold same-depth branch queries over the same KV tensors.
        key = key.repeat_interleave(self.queries_per_group, dim=1)
        value = value.repeat_interleave(self.queries_per_group, dim=1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)
        gate = gate.reshape(batch, tokens, self.query_heads, self.head_dim)
        attended = (attended.float() * torch.sigmoid(gate.float())).to(normalized.dtype)
        return self.out(attended.reshape(batch, tokens, -1)), kv


class MBTLayer(nn.Module):
    def __init__(self, width: int, latent: int, intermediate: int,
                 query_heads: int, kv_heads: int, config, *, joint: bool):
        super().__init__()
        self.input_norm = MBTRMSNorm(width, config.rms_norm_eps)
        self.attention = MBTAttention(width, query_heads, kv_heads, config, joint=joint)
        self.post_attention_norm = MBTRMSNorm(width, config.rms_norm_eps)
        self.moe = MBTMoE(width, latent, intermediate, config)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor,
                *, past: Optional[KVPair] = None, shared_kv: Optional[KVPair] = None,
                normalized: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, KVPair]:
        normalized = self.input_norm(hidden) if normalized is None else normalized
        attended, kv = self.attention(normalized, positions, mask, past=past, shared_kv=shared_kv)
        hidden = hidden + attended
        return hidden + self.moe(self.post_attention_norm(hidden)), kv


class MBTWidthBridge(nn.Module):
    def __init__(self, input_width: int, branch_width: int, eps: float):
        super().__init__()
        self.norm = MBTRMSNorm(input_width, eps)
        self.projection = nn.Linear(input_width, branch_width, bias=False)

    def forward(self, hidden):
        return self.projection(self.norm(hidden))


class MBTFusion(nn.Module):
    def __init__(self, branch_width: int, output_width: int, branches: int, eps: float):
        super().__init__()
        self.norms = nn.ModuleList(MBTRMSNorm(branch_width, eps) for _ in range(branches))
        self.projection = nn.Linear(branches * branch_width, output_width, bias=False)

    def forward(self, states):
        return self.projection(torch.cat([norm(state) for norm, state in zip(self.norms, states)], dim=-1))


class MBTJointKV(nn.Module):
    def __init__(self, width: int, branches: int, kv_heads: int, config):
        super().__init__()
        self.branches = branches
        self.kv_heads = kv_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.projection = nn.Linear(branches * width, 2 * kv_heads * self.head_dim, bias=False)
        self.key_norm = MBTRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, normalized_states, positions, past=None):
        batch, tokens, _ = normalized_states[0].shape
        combined = torch.cat(normalized_states, dim=-1) * (1.0 / math.sqrt(self.branches))
        packed = self.projection(combined).view(batch, tokens, self.kv_heads, 2 * self.head_dim)
        key, value = packed.split(self.head_dim, dim=-1)
        key = apply_rope(self.key_norm(key), positions, self.rope_theta)
        return append_cache(key, value, past)


class MBTCore(nn.Module):
    """Shared prelude, parallel branch graph, fusion, and shared coda.

    Physical blocks are stored once in prelude/branch-major/coda order. Cache
    order follows prelude, aligned Joint-KV depths (or physical independent
    branch blocks), and coda. Every shared KV is appended exactly once.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.branch_depths = tuple(config.branch_depths)
        self.branches = len(self.branch_depths)
        self.kv_mode = config.kv_mode
        self.prelude = config.prelude if self.branches else config.num_hidden_layers
        self.coda = config.coda if self.branches else 0
        self._validate_config()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = MBTRMSNorm(config.hidden_size, config.rms_norm_eps)
        layers = []
        self.layer_cache_specs = []
        core_start = self.prelude
        core_end = self.prelude + sum(self.branch_depths)
        for index in range(config.num_hidden_layers):
            is_branch = self.branches > 0 and core_start <= index < core_end
            width = config.branch_hidden_size if is_branch else config.hidden_size
            latent = config.branch_moe_latent_size if is_branch else config.moe_latent_size
            intermediate = config.branch_moe_intermediate_size if is_branch else config.moe_intermediate_size
            q_heads = config.branch_num_attention_heads if is_branch else config.num_attention_heads
            kv_heads = config.branch_num_key_value_heads if is_branch else config.num_key_value_heads
            layers.append(MBTLayer(width, latent, intermediate, q_heads, kv_heads, config,
                                   joint=is_branch and self.kv_mode == "joint"))
            self.layer_cache_specs.append((kv_heads, config.head_dim))
        self.layers = nn.ModuleList(layers)
        self.input_bridge = (
            MBTWidthBridge(config.hidden_size, config.branch_hidden_size, config.rms_norm_eps)
            if self.branches and config.use_input_bridge else None
        )
        self.fusion = (
            MBTFusion(config.branch_hidden_size, config.hidden_size, self.branches, config.rms_norm_eps)
            if self.branches else None
        )
        self.joint_kv = nn.ModuleList(
            MBTJointKV(config.branch_hidden_size, self.branches, config.branch_num_key_value_heads, config)
            for _ in range(self.branch_depths[0] if self.kv_mode == "joint" else 0)
        )
        start = self.prelude
        self.branch_starts = []
        for depth in self.branch_depths:
            self.branch_starts.append(start)
            start += depth
        self.coda_start = start
        if self.kv_mode == "joint":
            self.cache_specs = (
                self.layer_cache_specs[:self.prelude]
                + [(config.branch_num_key_value_heads, config.head_dim)] * self.branch_depths[0]
                + self.layer_cache_specs[self.coda_start:]
            )
        else:
            self.cache_specs = list(self.layer_cache_specs)

    def _validate_config(self):
        c = self.config
        if self.kv_mode not in ("standard", "independent", "joint"):
            raise ValueError("kv_mode must be standard, independent, or joint")
        if self.kv_mode == "standard" and self.branches:
            raise ValueError("Standard cannot have parallel branches")
        if self.kv_mode == "joint" and self.branches < 2:
            raise ValueError("Joint-KV requires at least two branches")
        if self.prelude < 0 or self.coda < 0 or any(d <= 0 for d in self.branch_depths):
            raise ValueError("Invalid block allocation")
        if self.prelude + sum(self.branch_depths) + self.coda != c.num_hidden_layers:
            raise ValueError("Block allocation differs from physical layer count")
        if self.kv_mode == "joint" and len(set(self.branch_depths)) != 1:
            raise ValueError("Joint-KV requires equal-depth branches")
        if self.branches and c.branch_hidden_size != c.hidden_size and not c.use_input_bridge:
            raise ValueError("Different branch width requires the shared input bridge")
        if c.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        head_specs = [(c.num_attention_heads, c.num_key_value_heads)]
        if self.branches:
            head_specs.append((c.branch_num_attention_heads, c.branch_num_key_value_heads))
        if any(q < 1 or kv < 1 or q % kv for q, kv in head_specs):
            raise ValueError("Query heads must be a positive multiple of KV heads")
        if not 0 < c.num_experts_per_tok < c.num_experts:
            raise ValueError("Quantile balancing requires 0 < top-k < expert count")

    def _past_length(self, past_key_values, batch: int) -> int:
        if past_key_values is None or len(past_key_values) == 0:
            return 0
        if len(past_key_values) != len(self.cache_specs):
            raise ValueError("Cache entry count differs from this model's KV layout")
        length = past_key_values[0][0].shape[-2]
        for pair, (heads, width) in zip(past_key_values, self.cache_specs):
            if len(pair) != 2 or pair[0].shape != pair[1].shape:
                raise ValueError("A cache entry must have equally shaped K and V tensors")
            if tuple(pair[0].shape) != (batch, heads, length, width):
                raise ValueError("Cache batch, head, sequence, or width shape mismatch")
        return length

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, inputs_embeds=None,
                output_hidden_states=False, output_attentions=False, **kwargs):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one of input_ids and inputs_embeds")
        if output_attentions:
            raise NotImplementedError("This reference uses SDPA and does not materialize attention matrices")
        hidden = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Expected [batch, tokens, hidden_size] inputs")
        batch, tokens, _ = hidden.shape
        if tokens == 0:
            raise ValueError("An empty token sequence cannot be evaluated")
        past_key_values = past_key_values or None
        past_length = self._past_length(past_key_values, batch)
        total_length = past_length + tokens
        if attention_mask is None:
            valid_keys = torch.ones(batch, total_length, dtype=torch.bool, device=hidden.device)
        else:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, total_length):
                raise ValueError("attention_mask must cover the full cached and current sequence")
            valid_keys = attention_mask.to(device=hidden.device, dtype=torch.bool)
        if position_ids is None:
            all_positions = valid_keys.long().cumsum(-1) - 1
            all_positions = all_positions.masked_fill(~valid_keys, 0)
            position_ids = all_positions[:, -tokens:]
        elif position_ids.ndim != 2 or position_ids.shape[-1] != tokens or position_ids.shape[0] not in (1, batch):
            raise ValueError("position_ids must have shape [batch, current_tokens] or [1, current_tokens]")
        position_ids = position_ids.to(device=hidden.device)
        causal = (
            torch.arange(total_length, device=hidden.device)[None, :]
            <= (past_length + torch.arange(tokens, device=hidden.device))[:, None]
        )
        mask = causal[None, None, :, :] & valid_keys[:, None, None, :]
        next_cache = []
        hidden_history = [hidden] if output_hidden_states else None
        cache_index = 0

        def run_layer(index, state):
            nonlocal cache_index
            previous = past_key_values[cache_index] if past_key_values is not None else None
            state, kv = self.layers[index](state, position_ids, mask, past=previous)
            if use_cache:
                next_cache.append(kv)
            cache_index += 1
            if hidden_history is not None:
                hidden_history.append(state)
            return state

        for index in range(self.prelude):
            hidden = run_layer(index, hidden)
        if self.branches:
            shared_input = self.input_bridge(hidden) if self.input_bridge is not None else hidden
            # Blocks are functional with respect to their inputs, so independent
            # branch references do not require copying their common input.
            states = [shared_input for _ in self.branch_depths]
            if self.kv_mode == "joint":
                for depth, projection in enumerate(self.joint_kv):
                    blocks = [self.layers[start + depth] for start in self.branch_starts]
                    normalized = [block.input_norm(state) for block, state in zip(blocks, states)]
                    previous = past_key_values[cache_index] if past_key_values is not None else None
                    kv = projection(normalized, position_ids, past=previous)
                    if use_cache:
                        next_cache.append(kv)
                    cache_index += 1
                    states = [
                        block(state, position_ids, mask, shared_kv=kv, normalized=norm)[0]
                        for block, state, norm in zip(blocks, states, normalized)
                    ]
                    if hidden_history is not None:
                        hidden_history.extend(states)
            else:
                for branch, (start, depth) in enumerate(zip(self.branch_starts, self.branch_depths)):
                    for index in range(start, start + depth):
                        states[branch] = run_layer(index, states[branch])
            hidden = self.fusion(states)
        for index in range(self.coda_start, self.config.num_hidden_layers):
            hidden = run_layer(index, hidden)
        hidden = self.norm(hidden)
        if hidden_history is not None:
            hidden_history.append(hidden)
        return MBTCoreOutput(
            last_hidden_state=hidden,
            past_key_values=tuple(next_cache) if use_cache else None,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
        )
