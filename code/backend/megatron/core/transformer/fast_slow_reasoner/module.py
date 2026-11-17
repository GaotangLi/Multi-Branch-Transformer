# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""An isolated Fast--Slow Reasoner module.

Callers construct the physical fast Transformer layers and slow MoE with the
appropriate cloned Megatron configs and process groups. This file owns only
the architecture's tensor flow. The original narrow v1 uses adapters while the
full-width v2 removes them::

    a = Prelude(x)                         # [s, b, d]
    z = W_down(a) or clone(a)              # [s, b, d_fast]
    h = a.clone()                          # [s, b, d]
    repeat the configured visits (v1/v2: four; A: two):
        z = FastLayer_N(...FastLayer_1(z)) # shared physical layers
        u = W_up(z) or z                   # [s, b, d]
        h = SlowCrossAttentionMoE(h, u)    # shared slow block
    return h                               # caller runs Coda and LM head

Only the final fast state of a visit reaches the slow stream. The slow stream
never feeds the fast recurrence.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default


RotaryEmbedding = Union[Tensor, tuple[Tensor, Tensor]]


def _reduce_one_noninterleaved_rope_by_two(freqs: Tensor) -> Tensor:
    """Reduce full-head non-interleaved RoPE to half head dimension exactly.

    Megatron represents a 64-dimensional non-interleaved head as
    ``[f_0, ..., f_31, f_0, ..., f_31]``. A correctly reparameterized
    32-dimensional RoPE with the same base is therefore
    ``[f_0, f_2, ..., f_30, f_0, f_2, ..., f_30]``; taking the first 32
    coordinates would be wrong.
    """

    if not isinstance(freqs, Tensor):
        raise TypeError(f"RoPE frequencies must be a Tensor, got {type(freqs).__name__}.")
    rotary_dimension = freqs.shape[-1]
    if rotary_dimension < 4 or rotary_dimension % 4 != 0:
        raise ValueError(
            "Full non-interleaved RoPE dimension must be a positive multiple "
            f"of four, got {rotary_dimension}."
        )
    source_half = rotary_dimension // 2
    first = freqs[..., :source_half:2]
    second = freqs[..., source_half::2]
    if first.shape != second.shape:
        raise RuntimeError(
            "Non-interleaved RoPE halves produced inconsistent reduced shapes: "
            f"{tuple(first.shape)} and {tuple(second.shape)}."
        )
    return torch.cat((first, second), dim=-1)


def reduce_noninterleaved_rope_by_two(
    rotary_pos_emb: RotaryEmbedding,
) -> RotaryEmbedding:
    """Return exact half-head-dimension non-interleaved RoPE frequencies."""

    if isinstance(rotary_pos_emb, tuple):
        if len(rotary_pos_emb) != 2:
            raise ValueError("RoPE tuple must contain exactly query and key frequencies.")
        return (
            _reduce_one_noninterleaved_rope_by_two(rotary_pos_emb[0]),
            _reduce_one_noninterleaved_rope_by_two(rotary_pos_emb[1]),
        )
    return _reduce_one_noninterleaved_rope_by_two(rotary_pos_emb)


class _FP32StatsRMSNorm(nn.Module):
    """RMSNorm with FP32 statistics and an output in the input dtype."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype=hidden_states.dtype)


class FastSlowReasoner(nn.Module):
    """Shared fast recurrence with persistent full-width slow state.

    Optional config fields consumed here are ``fast_slow_fast_hidden_size``,
    ``fast_slow_fast_num_attention_heads``, ``fast_slow_fast_num_layers``, and
    ``fast_slow_num_visits``. When absent, this module derives the v1 ``d/4``
    width, half as many heads, the supplied physical-layer count, and four
    visits. A configured full-width Fast stream uses identity adapters and the
    unchanged full-head RoPE tensor.
    """

    def __init__(
        self,
        config: TransformerConfig,
        fast_layers: Sequence[nn.Module],
        slow_moe: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(config, TransformerConfig):
            # Lightweight config objects keep isolated CPU tests inexpensive.
            required = (
                "hidden_size",
                "num_attention_heads",
                "init_method",
                "output_layer_init_method",
            )
            missing = [name for name in required if not hasattr(config, name)]
            if missing:
                raise TypeError(
                    "config must be TransformerConfig-like; missing " + ", ".join(missing)
                )
        if not isinstance(slow_moe, nn.Module):
            raise TypeError(f"slow_moe must be an nn.Module, got {type(slow_moe).__name__}.")
        if isinstance(fast_layers, (str, bytes)):
            raise TypeError("fast_layers must be a sequence of nn.Module objects.")
        physical_fast_layers = tuple(fast_layers)
        if not physical_fast_layers:
            raise ValueError("FastSlowReasoner requires at least one physical fast layer.")
        if not all(isinstance(layer, nn.Module) for layer in physical_fast_layers):
            raise TypeError("Every fast layer must be an nn.Module.")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        if self.hidden_size < 4 or self.hidden_size % 4 != 0:
            raise ValueError(
                "FastSlowReasoner requires hidden_size divisible by four, got "
                f"{self.hidden_size}."
            )
        if (
            self.num_attention_heads < 2
            or self.num_attention_heads % 2 != 0
            or self.hidden_size % self.num_attention_heads != 0
        ):
            raise ValueError(
                "FastSlowReasoner requires an even number of full-width heads "
                "that divides hidden_size; got "
                f"d={self.hidden_size}, heads={self.num_attention_heads}."
            )

        configured_fast_hidden = getattr(config, "fast_slow_fast_hidden_size", None)
        self.fast_hidden_size = (
            self.hidden_size // 4
            if configured_fast_hidden is None
            else int(configured_fast_hidden)
        )
        configured_fast_heads = getattr(
            config, "fast_slow_fast_num_attention_heads", None
        )
        self.fast_num_attention_heads = (
            self.num_attention_heads // 2
            if configured_fast_heads is None
            else int(configured_fast_heads)
        )
        self.fast_router_topk = getattr(
            config,
            "fast_slow_fast_router_topk",
            getattr(config, "moe_router_topk", None),
        )
        self.fast_window_size = getattr(config, "fast_slow_fast_window_size", None)
        self.fast_attention_type = getattr(config, "fast_slow_fast_attention_type", "softmax")
        if self.fast_attention_type not in ("softmax", "kda"):
            raise ValueError("Fast mixer must be softmax or kda.")
        configured_fast_window_schedule = getattr(
            config, "fast_slow_fast_window_schedule", None
        )
        self.fast_is_full_width = self.fast_hidden_size == self.hidden_size
        expected_fast_hidden = (
            self.hidden_size if self.fast_is_full_width else self.hidden_size // 4
        )
        expected_fast_heads = (
            self.num_attention_heads
            if self.fast_is_full_width
            else self.num_attention_heads // 2
        )
        if self.fast_hidden_size != expected_fast_hidden:
            raise ValueError(
                "FastSlowReasoner fast width must be either d/4 (v1) or d (v2); "
                f"got d={self.hidden_size}, d_fast={self.fast_hidden_size}."
            )
        if self.fast_num_attention_heads != expected_fast_heads:
            raise ValueError(
                "FastSlowReasoner fast heads must be half the main heads for v1 or "
                "equal to the main heads for v2; got "
                f"heads={self.num_attention_heads}, heads_fast={self.fast_num_attention_heads}."
            )
        if self.fast_hidden_size % self.fast_num_attention_heads != 0:
            raise ValueError(
                "Fast width must divide into fast heads; got "
                f"d_fast={self.fast_hidden_size}, heads_fast={self.fast_num_attention_heads}."
            )
        self.main_head_dim = self.hidden_size // self.num_attention_heads
        self.fast_head_dim = self.fast_hidden_size // self.fast_num_attention_heads
        expected_fast_head_dim = (
            self.main_head_dim if self.fast_is_full_width else self.main_head_dim // 2
        )
        if self.main_head_dim % 2 != 0 or self.fast_head_dim != expected_fast_head_dim:
            raise ValueError(
                "Fast RoPE requires v1 fast heads at half the main head dimension or "
                "v2 fast heads at the full main head dimension; got "
                f"{self.fast_head_dim} and {self.main_head_dim}."
            )

        self.num_visits = int(getattr(config, "fast_slow_num_visits", 4))
        is_a = (
            self.fast_attention_type == "softmax"
            and
            self.fast_is_full_width
            and self.fast_router_topk == 4
            and self.num_visits == 2
            and configured_fast_window_schedule is not None
            and tuple(configured_fast_window_schedule) == (512, 512)
        )
        is_kda = (
            self.fast_attention_type == "kda"
            and self.fast_is_full_width
            and self.fast_router_topk == 4
            and self.num_visits == 2
            and getattr(config, "fast_slow_kda_head_dim", 128) == 128
            and configured_fast_window_schedule is None
            and self.fast_window_size is None
        )
        if self.fast_attention_type == "kda" and not is_kda:
            raise ValueError("KDA Fast requires full width, top4, R2, head_dim128 and no SWA window.")
        if self.num_visits != 4 and not (is_a or is_kda):
            raise ValueError(
                "The original FastSlowReasoner contract fixes exactly four visits; "
                "only full-width top-4 A with [512,512] permits two visits, "
                f"got {self.num_visits}."
            )
        self.fast_window_schedule = (
            None
            if configured_fast_window_schedule is None
            else tuple(configured_fast_window_schedule)
        )
        if self.fast_window_schedule is not None:
            allowed_schedules = {
                (512, 512, 512, 512),
                (128, 256, 512, 1024),
            }
            if is_a:
                allowed_schedules = {(512, 512)}
            if not self.fast_is_full_width:
                raise ValueError(
                    "Per-visit Fast windows are supported only by the full-width v2 stream."
                )
            if self.fast_window_schedule not in allowed_schedules:
                raise ValueError(
                    "FastSlowReasoner accepts only the fixed "
                    "[512, 512, 512, 512] or progressive "
                    "[128, 256, 512, 1024], or A [512,512] window schedule; got "
                    f"{list(self.fast_window_schedule)}."
                )
            maximum_window = max(self.fast_window_schedule)
            if self.fast_window_size not in (None, maximum_window):
                raise ValueError(
                    "fast_slow_fast_window_size must be unset or equal the schedule "
                    f"maximum {maximum_window}; got {self.fast_window_size}."
                )
        self.fast_construction_window_size = (
            max(self.fast_window_schedule)
            if self.fast_window_schedule is not None
            else self.fast_window_size
        )
        configured_fast_layers = getattr(config, "fast_slow_fast_num_layers", None)
        if configured_fast_layers is not None and int(configured_fast_layers) != len(
            physical_fast_layers
        ):
            raise ValueError(
                "Configured physical fast-layer count does not match supplied layers: "
                f"{configured_fast_layers} != {len(physical_fast_layers)}."
            )

        self._validate_clean_attention_contract(config)
        self._validate_external_modules(physical_fast_layers, slow_moe)

        self.fast_layers = nn.ModuleList(physical_fast_layers)
        self.slow_moe = slow_moe
        if self.fast_is_full_width:
            # Do not register same-width projection parameters: v2 starts both
            # streams from independent clones of the Prelude state and passes
            # terminal Fast memory directly to the Slow updater.
            self.down_projection = None
            self.up_projection = None
        else:
            self.down_projection = nn.Linear(
                self.hidden_size, self.fast_hidden_size, bias=False
            )
            self.up_projection = nn.Linear(
                self.fast_hidden_size, self.hidden_size, bias=False
            )
        self.slow_query_norm = _FP32StatsRMSNorm(
            self.hidden_size, float(config.layernorm_epsilon)
        )
        self.slow_memory_norm = _FP32StatsRMSNorm(
            self.hidden_size, float(config.layernorm_epsilon)
        )
        self.slow_moe_norm = _FP32StatsRMSNorm(
            self.hidden_size, float(config.layernorm_epsilon)
        )
        self.query_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attention_output_projection = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )

        input_projections = (
            ()
            if self.fast_is_full_width
            else (self.down_projection, self.up_projection)
        )
        for projection in (
            *input_projections,
            self.query_projection,
            self.key_projection,
            self.value_projection,
        ):
            config.init_method(projection.weight)
        config.output_layer_init_method(self.attention_output_projection.weight)

    @staticmethod
    def _validate_clean_attention_contract(config: TransformerConfig) -> None:
        effective_query_groups = getattr(config, "num_query_groups", None)
        if effective_query_groups not in (None, config.num_attention_heads):
            raise ValueError("FastSlowReasoner slow attention is MHA, not GQA.")
        for attribute in (
            "qk_layernorm",
            "qk_l2_norm",
            "qk_clip",
            "attention_output_norm",
            "attention_output_gate",
            "add_qkv_bias",
            "multi_latent_attention",
            "rotary_interleaved",
        ):
            if bool(getattr(config, attribute, False)):
                raise ValueError(f"FastSlowReasoner clean contract requires {attribute}=False.")
        if getattr(config, "softmax_scale", None) is not None:
            raise ValueError("FastSlowReasoner uses default attention scaling.")
        if getattr(config, "no_rope_freq", None) is not None:
            raise ValueError("FastSlowReasoner requires RoPE in every attention block.")
        if getattr(config, "experimental_attention_variant", None) is not None:
            raise ValueError("FastSlowReasoner forbids experimental attention variants.")
        if bool(getattr(config, "add_bias_linear", False)):
            raise ValueError("FastSlowReasoner projections and slow MoE must be bias-free.")
        if float(getattr(config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("FastSlowReasoner clean contract fixes attention dropout to zero.")
        if float(getattr(config, "hidden_dropout", 0.0)) != 0.0:
            raise ValueError("FastSlowReasoner clean contract fixes hidden dropout to zero.")
        if not callable(config.init_method) or not callable(config.output_layer_init_method):
            raise ValueError("Transformer initialization methods must be callable.")
        if float(getattr(config, "layernorm_epsilon", 0.0)) <= 0.0:
            raise ValueError("layernorm_epsilon must be positive.")

    def _validate_external_modules(
        self,
        fast_layers: tuple[nn.Module, ...],
        slow_moe: nn.Module,
    ) -> None:
        for index, layer in enumerate(fast_layers):
            if self.fast_attention_type == "kda":
                mixer = getattr(layer, "self_attention", None)
                if not getattr(mixer, "is_kda_attention", False):
                    raise ValueError(f"Fast layer {index} did not construct the required KDA mixer.")
                if (mixer.num_heads, mixer.head_dim) != (
                    self.fast_num_attention_heads, getattr(self.config, "fast_slow_kda_head_dim", 128)
                ):
                    raise ValueError("Effective KDA head shape differs from the declared configuration.")
            layer_config = getattr(layer, "config", None)
            if layer_config is not None and int(layer_config.hidden_size) != self.fast_hidden_size:
                raise ValueError(
                    f"fast_layers[{index}] uses hidden size {layer_config.hidden_size}, "
                    f"expected {self.fast_hidden_size}."
                )
            if layer_config is not None:
                expected_fast_window = (
                    None
                    if self.fast_construction_window_size is None
                    else (int(self.fast_construction_window_size), 0)
                )
                fast_contract = {
                    "num_attention_heads": self.fast_num_attention_heads,
                    "window_size": expected_fast_window,
                    "window_attn_skip_freq": None,
                }
                if self.fast_router_topk is not None:
                    fast_contract["moe_router_topk"] = int(self.fast_router_topk)
                for attribute, expected in fast_contract.items():
                    if hasattr(layer_config, attribute) and getattr(
                        layer_config, attribute
                    ) != expected:
                        raise ValueError(
                            f"fast_layers[{index}] uses {attribute}="
                            f"{getattr(layer_config, attribute)!r}, expected {expected!r}."
                        )
        slow_config = getattr(slow_moe, "config", None)
        if slow_config is not None and int(slow_config.hidden_size) != self.hidden_size:
            raise ValueError(
                f"slow_moe uses hidden size {slow_config.hidden_size}, expected {self.hidden_size}."
            )
        if slow_config is not None:
            slow_contract = {
                "window_size": getattr(self.config, "window_size", None),
                "window_attn_skip_freq": getattr(
                    self.config, "window_attn_skip_freq", None
                ),
            }
            slow_router_topk = getattr(self.config, "moe_router_topk", None)
            if slow_router_topk is not None:
                slow_contract["moe_router_topk"] = int(slow_router_topk)
            for attribute, expected in slow_contract.items():
                if hasattr(slow_config, attribute) and getattr(
                    slow_config, attribute
                ) != expected:
                    raise ValueError(
                        f"slow_moe uses {attribute}={getattr(slow_config, attribute)!r}, "
                        f"expected {expected!r}."
                    )

        layer_numbers = [
            getattr(module, "layer_number", None)
            for module in (*fast_layers, slow_moe)
        ]
        declared_numbers = [number for number in layer_numbers if number is not None]
        if len(declared_numbers) != len(set(declared_numbers)):
            raise ValueError(
                "Fast and slow modules must use unique layer_number values for "
                f"router/probe identity, got {layer_numbers}."
            )

    @staticmethod
    def _validate_state(state: Tensor, label: str, hidden_size: int) -> None:
        if state.ndim != 3:
            raise ValueError(
                f"{label} must have shape [sequence, batch, hidden], got {tuple(state.shape)}."
            )
        if state.shape[-1] != hidden_size:
            raise ValueError(
                f"{label} expected hidden size {hidden_size}, got {state.shape[-1]}."
            )

    def _validate_rope(self, rotary_pos_emb: RotaryEmbedding) -> None:
        tensors = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else (rotary_pos_emb,)
        if len(tensors) != 2 and isinstance(rotary_pos_emb, tuple):
            raise ValueError("RoPE tuple must contain exactly query and key frequencies.")
        for frequencies in tensors:
            if not isinstance(frequencies, Tensor):
                raise TypeError("RoPE entries must be tensors.")
            if frequencies.shape[-1] != self.main_head_dim:
                raise ValueError(
                    "FastSlowReasoner requires full main-head RoPE; got rotary "
                    f"dimension {frequencies.shape[-1]}, expected {self.main_head_dim}."
                )

    def _run_fast_layer(
        self,
        layer: nn.Module,
        fast_state: Tensor,
        fast_rotary_pos_emb: RotaryEmbedding,
        visit_index: int,
    ) -> Tensor:
        layer_kwargs = dict(
            hidden_states=fast_state,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=(fast_rotary_pos_emb if self.fast_attention_type == "softmax" else None),
            inference_context=None,
            packed_seq_params=None,
            loop_visit_index=visit_index,
        )
        if self.fast_window_schedule is not None:
            layer_kwargs["attention_window_size"] = (
                self.fast_window_schedule[visit_index - 1],
                0,
            )
        output = layer(**layer_kwargs)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise RuntimeError(
                    "A fast Transformer layer tuple output must be (hidden_states, context)."
                )
            output, output_context = output
            if output_context is not None:
                raise RuntimeError("Fast decoder layers must not create a cross-attention context.")
        if not isinstance(output, Tensor):
            raise RuntimeError(
                f"Fast layer returned {type(output).__name__}, expected a Tensor."
            )
        self._validate_state(output, "fast layer output", self.fast_hidden_size)
        return output

    def _update_slow(
        self,
        slow_state: Tensor,
        fast_memory: Tensor,
        rotary_pos_emb: RotaryEmbedding,
        visit_index: int,
    ) -> Tensor:
        sequence_length, batch_size, _ = slow_state.shape
        normalized_slow = self.slow_query_norm(slow_state)
        query = self.query_projection(normalized_slow).view(
            sequence_length, batch_size, self.num_attention_heads, self.main_head_dim
        )
        normalized_memory = self.slow_memory_norm(fast_memory)
        key = self.key_projection(normalized_memory).view_as(query)
        value = self.value_projection(normalized_memory).view_as(query)

        if isinstance(rotary_pos_emb, tuple):
            query_pos_emb, key_pos_emb = rotary_pos_emb
        else:
            query_pos_emb = key_pos_emb = rotary_pos_emb
        query = apply_rotary_pos_emb(query, query_pos_emb, config=self.config)
        key = apply_rotary_pos_emb(key, key_pos_emb, config=self.config)

        # Fail rather than silently falling back to a quadratic-memory math
        # kernel on the exact-shape GPU path. CPU unit tests intentionally use
        # the ordinary math backend.
        attention_backend = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION) if query.is_cuda else nullcontext()
        )
        with attention_backend:
            attention_output = F.scaled_dot_product_attention(
                query.permute(1, 2, 0, 3),
                key.permute(1, 2, 0, 3),
                value.permute(1, 2, 0, 3),
                dropout_p=0.0,
                is_causal=True,
                scale=None,
                enable_gqa=False,
            ).permute(2, 0, 1, 3)
        attention_output = self.attention_output_projection(
            attention_output.contiguous().view(
                sequence_length, batch_size, self.hidden_size
            )
        )
        slow_state = slow_state + attention_output

        moe_output = self.slow_moe(
            self.slow_moe_norm(slow_state), loop_visit_index=visit_index
        )
        if not isinstance(moe_output, tuple) or len(moe_output) != 2:
            raise RuntimeError("slow_moe must return (output, bias).")
        moe_hidden_states, moe_bias = moe_output
        if moe_bias is not None:
            raise RuntimeError("Bias-free FastSlowReasoner slow_moe returned a bias tensor.")
        self._validate_state(moe_hidden_states, "slow MoE output", self.hidden_size)
        return slow_state + moe_hidden_states

    def forward(
        self,
        prelude_state: Tensor,
        rotary_pos_emb: RotaryEmbedding,
        *,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        inference_context: object = None,
        packed_seq_params: object = None,
    ) -> Tensor:
        """Evolve the Fast--Slow streams and return the final slow state."""

        self._validate_state(prelude_state, "Prelude state", self.hidden_size)
        if attention_mask is not None:
            raise ValueError(
                "FastSlowReasoner expects fused causal attention with no materialized mask."
            )
        if padding_mask is not None:
            raise NotImplementedError(
                "Padding masks are not supported by the first FastSlowReasoner implementation."
            )
        if inference_context is not None:
            raise NotImplementedError(
                "FastSlowReasoner KV-cache inference requires an explicit fast/slow cache policy."
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Packed sequences are not supported by the first FastSlowReasoner implementation."
            )
        self._validate_rope(rotary_pos_emb)
        fast_rotary_pos_emb = (
            rotary_pos_emb
            if self.fast_is_full_width
            else reduce_noninterleaved_rope_by_two(rotary_pos_emb)
        )

        fast_state = (
            prelude_state.clone()
            if self.fast_is_full_width
            else self.down_projection(prelude_state)
        )
        slow_state = prelude_state.clone()
        for visit_index in range(1, self.num_visits + 1):
            for layer in self.fast_layers:
                fast_state = self._run_fast_layer(
                    layer, fast_state, fast_rotary_pos_emb, visit_index
                )
            slow_memory = (
                fast_state if self.fast_is_full_width else self.up_projection(fast_state)
            )
            slow_state = self._update_slow(
                slow_state, slow_memory, rotary_pos_emb, visit_index
            )
        return slow_state

    @staticmethod
    def _merge_sharded(
        destination: ShardedStateDict,
        source: ShardedStateDict,
        label: str,
    ) -> None:
        overlap = destination.keys() & source.keys()
        if overlap:
            raise RuntimeError(
                f"Duplicate sharded checkpoint keys while adding {label}: {sorted(overlap)}"
            )
        destination.update(source)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Return an EP-aware distributed checkpoint mapping.

        Fast layers and the slow MoE must expose their own checkpoint methods.
        Failing closed avoids silently marking nested EP expert tensors as
        replicated. The first integration is TP1/PP1, so experimental layers
        use independent key namespaces rather than the main block's PP axis.

        TODO: define explicit TP/PP placement and layer-axis mappings before
        accepting any non-empty enclosing ``sharded_offsets``.
        """

        if sharded_offsets:
            raise ValueError(
                "FastSlowReasoner sharded checkpoint currently requires TP1/PP1 "
                "with no enclosing sharded offsets."
            )

        # Validate the EP-owning children before emitting any replicated state.
        # This makes checkpoint support atomic: a partially populated mapping is
        # never produced when an externally supplied layer lacks sharding metadata.
        fast_sharded_methods = []
        for index, layer in enumerate(self.fast_layers):
            sharded_method = getattr(layer, "sharded_state_dict", None)
            if not callable(sharded_method):
                raise RuntimeError(
                    "Every production fast layer must implement sharded_state_dict; "
                    f"fast_layers[{index}] ({type(layer).__name__}) does not."
                )
            fast_sharded_methods.append(sharded_method)
        slow_moe_sharded = getattr(self.slow_moe, "sharded_state_dict", None)
        if not callable(slow_moe_sharded):
            raise RuntimeError(
                "Production slow_moe must implement sharded_state_dict so EP expert "
                f"tensors are not marked replicated; got {type(self.slow_moe).__name__}."
            )

        result: ShardedStateDict = {}
        adapter_children = (
            ()
            if self.fast_is_full_width
            else (
                ("down_projection", self.down_projection),
                ("up_projection", self.up_projection),
            )
        )
        replicated_children = (
            *adapter_children,
            ("slow_query_norm", self.slow_query_norm),
            ("slow_memory_norm", self.slow_memory_norm),
            ("slow_moe_norm", self.slow_moe_norm),
            ("query_projection", self.query_projection),
            ("key_projection", self.key_projection),
            ("value_projection", self.value_projection),
            ("attention_output_projection", self.attention_output_projection),
        )
        for name, child in replicated_children:
            self._merge_sharded(
                result,
                sharded_state_dict_default(
                    child,
                    prefix=f"{prefix}{name}.",
                    sharded_offsets=(),
                    metadata=metadata,
                ),
                name,
            )

        for index, sharded_method in enumerate(fast_sharded_methods):
            self._merge_sharded(
                result,
                sharded_method(
                    prefix=f"{prefix}fast_layers.{index}.",
                    sharded_offsets=(),
                    metadata=metadata,
                ),
                f"fast_layers[{index}]",
            )

        self._merge_sharded(
            result,
            slow_moe_sharded(
                prefix=f"{prefix}slow_moe.",
                sharded_offsets=(),
                metadata=metadata,
            ),
            "slow_moe",
        )
        return result

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, fast_hidden_size={self.fast_hidden_size}, "
            f"main_heads={self.num_attention_heads}, fast_heads={self.fast_num_attention_heads}, "
            f"physical_fast_layers={len(self.fast_layers)}, visits={self.num_visits}, "
            f"fast_adapters={'identity' if self.fast_is_full_width else 'learned'}, "
            "fast_layers_shared=True, slow_block_shared=True, slow_to_fast_feedback=False"
        )
