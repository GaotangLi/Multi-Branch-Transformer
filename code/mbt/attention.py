"""Explicit Joint-KV attention using native Transformer Engine attention ops."""
from __future__ import annotations

import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention, TELayerNormColumnParallelLinear, TENorm, TERowParallelLinear,
)
from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType


def attention_submodules():
    return SelfAttentionSubmodules(
        linear_qkv=TELayerNormColumnParallelLinear,
        core_attention=TEDotProductAttention,
        linear_proj=TERowParallelLinear,
        q_layernorm=TENorm,
        k_layernorm=TENorm,
    )


class JointKVAttention(Attention):
    """Branch-local queries/gates consuming an explicit shared K/V pair.

    ``linear_qkv`` retains the native checkpoint field name, but contains Q/G
    rows only. No unused branch-local K/V weights are constructed.
    """

    def __init__(self, config, layer_number: int, *, pg_collection):
        submodules = attention_submodules()
        super().__init__(config=config, submodules=submodules, layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal, attention_type="self", pg_collection=pg_collection)
        self.linear_qkv_out_dim = 2*config.hidden_size
        self.linear_qkv = TELayerNormColumnParallelLinear(
            config.hidden_size, self.linear_qkv_out_dim, config=config,
            init_method=config.init_method, gather_output=False, bias=False,
            skip_bias_add=False, is_expert=False, tp_comm_buffer_name="qkv",
            tp_group=pg_collection.tp,
        )
        self.q_layernorm = TENorm(config=config, hidden_size=64, eps=config.layernorm_epsilon)
        self.k_layernorm = None

    def normalized_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """Reuse this branch's learned attention prenorm; no additional norm."""
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon)
        return (value*self.linear_qkv.layer_norm_weight.float()).to(hidden.dtype)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None,
                                    output_gate=False, split_qkv=True):
        if not output_gate or not split_qkv or key_value_states is None:
            raise ValueError("Joint-KV attention requires gated queries and an explicit shared KV pair")
        if not isinstance(key_value_states, tuple) or len(key_value_states) != 2:
            raise TypeError("shared KV must be a (key, value) tuple")
        mixed, bias = self.linear_qkv(hidden_states)
        if bias is not None:
            raise RuntimeError("Joint-KV query/gate projection must be bias-free")
        groups = self.config.num_query_groups
        mixed = mixed.view(*hidden_states.shape[:2], groups, 4*64)
        query, gate = mixed.split(2*64, dim=-1)
        query = self.q_layernorm(query.reshape(*hidden_states.shape[:2], 2*groups, 64))
        gate = gate.reshape(*hidden_states.shape[:2], 2*groups, 64)
        key, value = key_value_states
        return query, key, value, gate

    def forward(self, hidden_states, attention_mask=None, *, shared_kv, rotary_pos_emb=None):
        return super().forward(hidden_states, attention_mask,
            key_value_states=shared_kv, rotary_pos_emb=rotary_pos_emb)
