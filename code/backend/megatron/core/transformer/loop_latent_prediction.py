# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dual-stream latent recurrence with sparse accumulated prediction updates."""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.transformer_config import TransformerConfig


def prediction_update_offsets(
    loop_width: int,
    updates_per_visit: int,
) -> tuple[int, ...]:
    """Return evenly spaced one-based layer offsets within each loop visit.

    The final physical loop layer is always an update point.  Ceiling division
    puts the first of two updates at the middle layer for odd-width loops; for
    example, a seven-layer loop yields offsets ``(4, 7)``.
    """

    if loop_width < 1:
        raise ValueError(f"loop_width must be positive, got {loop_width}.")
    if updates_per_visit < 1:
        raise ValueError(
            f"updates_per_visit must be positive, got {updates_per_visit}."
        )
    if updates_per_visit > loop_width:
        raise ValueError(
            "updates_per_visit cannot exceed the number of physical loop "
            f"layers; got {updates_per_visit} updates for width {loop_width}."
        )

    offsets = tuple(
        (update_index * loop_width + updates_per_visit - 1) // updates_per_visit
        for update_index in range(1, updates_per_visit + 1)
    )
    if len(set(offsets)) != updates_per_visit or offsets[-1] != loop_width:
        raise RuntimeError(
            "Prediction-update offset construction did not produce distinct "
            f"checkpoints ending at the loop boundary: {offsets}."
        )
    return offsets


class _SharedPredictionMLP(nn.Module):
    """Bias-free SiLU MLP used for every latent-to-prediction correction."""

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        init_method: Callable[[Tensor], object],
        output_layer_init_method: Callable[[Tensor], object],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.output_projection = nn.Linear(mlp_hidden_size, hidden_size, bias=False)
        init_method(self.input_projection.weight)
        output_layer_init_method(self.output_projection.weight)

    def forward(self, latent_state: Tensor) -> Tensor:
        return self.output_projection(F.silu(self.input_projection(latent_state)))


class LatentPredictionAccumulator(nn.Module):
    """Initialize and evolve separate latent and prediction residual streams.

    ``W_split`` is a single bias-free ``d -> 2d`` projection initialized as two
    stacked identities, so both streams initially equal the Prelude output.  A
    single shared prediction MLP maps selected latent states to additive
    corrections of the prediction stream.  Prediction never feeds back into
    latent recurrence.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        init_method: Callable[[Tensor], object],
        output_layer_init_method: Callable[[Tensor], object],
        direct_identity_copy: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if mlp_hidden_size < 1:
            raise ValueError(
                f"mlp_hidden_size must be positive, got {mlp_hidden_size}."
            )

        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.direct_identity_copy = direct_identity_copy
        if direct_identity_copy:
            self.split_projection = None
        else:
            self.split_projection = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
            nn.init.eye_(self.split_projection.weight[:hidden_size])
            nn.init.eye_(self.split_projection.weight[hidden_size:])
            # Preserve the identity origin under AdamW, matching the existing
            # identity state/prediction projections in this project.
            self.split_projection.weight.is_loop_state_prediction_identity_parameter = True

        self.prediction_mlp = _SharedPredictionMLP(
            hidden_size,
            mlp_hidden_size,
            init_method,
            output_layer_init_method,
        )

    def initialize(self, prelude_state: Tensor) -> tuple[Tensor, Tensor]:
        """Map the Prelude output to exact identity-initialized ``(z_0, p_0)``."""

        self._validate_state(prelude_state, "Prelude state")
        if self.direct_identity_copy:
            # Clones make the two conceptual streams independent tensors while
            # retaining an exact gradient path from both back to Prelude.
            return prelude_state.clone(), prelude_state.clone()
        assert self.split_projection is not None
        latent_state, prediction_state = self.split_projection(prelude_state).chunk(
            2, dim=-1
        )
        return latent_state, prediction_state

    def update_prediction(
        self,
        latent_state: Tensor,
        prediction_state: Tensor,
    ) -> Tensor:
        """Accumulate one shared latent-to-prediction correction."""

        self._validate_state(latent_state, "latent state")
        self._validate_state(prediction_state, "prediction state")
        if latent_state.shape != prediction_state.shape:
            raise ValueError(
                "latent and prediction states must have identical shapes, got "
                f"{tuple(latent_state.shape)} and {tuple(prediction_state.shape)}."
            )
        return prediction_state + self.prediction_mlp(latent_state)

    def _validate_state(self, state: Tensor, label: str) -> None:
        if state.ndim != 3:
            raise ValueError(
                f"{label} must have shape [sequence, batch, hidden], got "
                f"{tuple(state.shape)}."
            )
        if state.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{label} expected hidden size {self.hidden_size}, got "
                f"{state.shape[-1]}."
            )

    def extra_repr(self) -> str:
        split = (
            "parameter_free_direct_identity_copy"
            if self.direct_identity_copy
            else "d_to_2d_stacked_identity"
        )
        return (
            f"hidden_size={self.hidden_size}, mlp_hidden_size={self.mlp_hidden_size}, "
            f"split={split}, activation=silu, bias=False, "
            "prediction_mlp_shared=True"
        )


class _FP32StatsRMSNorm(nn.Module):
    """RMSNorm with FP32 statistics and an output matching the input dtype."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(Tensor(hidden_size))
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(dtype=hidden_states.dtype)


class LatestLatentCrossAttentionPredictionAccumulator(nn.Module):
    """Update prediction state by causally cross-attending to the latest latent state.

    Query vectors come from the persistent prediction stream, while keys and
    values come only from the latest full-sequence latent state at the selected
    physical loop checkpoint.  There is no prediction self-attention and the
    prediction stream never feeds the latent recurrence.  The same complete
    cross-attention + SwiGLU block is shared across checkpoints and visits.

    This initial implementation deliberately supports full-sequence training
    and validation only.  Autoregressive generation needs an explicit policy
    for retaining, caching, or recomputing the visit-specific latent memory.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ffn_hidden_size: int,
    ) -> None:
        super().__init__()
        if config.tensor_model_parallel_size != 1:
            raise ValueError(
                "Latest-z prediction cross-attention currently requires tensor "
                "model parallel size one."
            )
        if config.context_parallel_size != 1:
            raise ValueError(
                "Latest-z prediction cross-attention currently requires context "
                "parallel size one."
            )
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads, got "
                f"{config.hidden_size} and {config.num_attention_heads}."
            )
        if ffn_hidden_size < 1:
            raise ValueError(
                f"ffn_hidden_size must be positive, got {ffn_hidden_size}."
            )
        if config.init_method is None or config.output_layer_init_method is None:
            raise ValueError("Transformer initialization methods must be configured.")

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.attention_dropout = config.attention_dropout
        self.hidden_dropout = config.hidden_dropout
        self.direct_identity_copy = (
            config.loop_latent_prediction_direct_identity_copy
        )

        if self.direct_identity_copy:
            self.split_projection = None
        else:
            self.split_projection = nn.Linear(
                self.hidden_size,
                2 * self.hidden_size,
                bias=False,
            )
            nn.init.eye_(self.split_projection.weight[: self.hidden_size])
            nn.init.eye_(self.split_projection.weight[self.hidden_size :])
            self.split_projection.weight.is_loop_state_prediction_identity_parameter = True

        self.prediction_norm = _FP32StatsRMSNorm(
            self.hidden_size,
            config.layernorm_epsilon,
        )
        self.latent_norm = _FP32StatsRMSNorm(
            self.hidden_size,
            config.layernorm_epsilon,
        )
        self.ffn_norm = _FP32StatsRMSNorm(
            self.hidden_size,
            config.layernorm_epsilon,
        )

        self.query_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attention_output_projection = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )
        self.ffn_input_projection = nn.Linear(
            self.hidden_size,
            2 * self.ffn_hidden_size,
            bias=False,
        )
        self.ffn_output_projection = nn.Linear(
            self.ffn_hidden_size,
            self.hidden_size,
            bias=False,
        )

        for projection in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.ffn_input_projection,
        ):
            config.init_method(projection.weight)
        for projection in (
            self.attention_output_projection,
            self.ffn_output_projection,
        ):
            config.output_layer_init_method(projection.weight)

    def initialize(self, prelude_state: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_state(prelude_state, "Prelude state")
        if self.direct_identity_copy:
            return prelude_state.clone(), prelude_state.clone()
        assert self.split_projection is not None
        return self.split_projection(prelude_state).chunk(2, dim=-1)

    def update_prediction(
        self,
        latent_state: Tensor,
        prediction_state: Tensor,
        *,
        rotary_pos_emb: Optional[Union[Tensor, tuple[Tensor, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        inference_context: object = None,
        packed_seq_params: object = None,
    ) -> Tensor:
        """Run one shared causal cross-attention + SwiGLU prediction update."""

        self._validate_state(latent_state, "latent state")
        self._validate_state(prediction_state, "prediction state")
        if latent_state.shape != prediction_state.shape:
            raise ValueError(
                "latent and prediction states must have identical shapes, got "
                f"{tuple(latent_state.shape)} and {tuple(prediction_state.shape)}."
            )
        if attention_mask is not None:
            raise ValueError(
                "Latest-z prediction cross-attention currently expects the fused "
                "causal-mask path with no materialized attention mask."
            )
        if inference_context is not None:
            raise NotImplementedError(
                "Latest-z prediction cross-attention generation requires an "
                "explicit latent-memory cache policy and is not enabled yet."
            )
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Packed sequences are not yet supported by latest-z prediction "
                "cross-attention."
            )

        sequence_length, batch_size, _ = prediction_state.shape
        query = self.query_projection(self.prediction_norm(prediction_state)).view(
            sequence_length,
            batch_size,
            self.num_attention_heads,
            self.head_dim,
        )
        normalized_latent = self.latent_norm(latent_state)
        key = self.key_projection(normalized_latent).view(
            sequence_length,
            batch_size,
            self.num_attention_heads,
            self.head_dim,
        )
        value = self.value_projection(normalized_latent).view(
            sequence_length,
            batch_size,
            self.num_attention_heads,
            self.head_dim,
        )

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                query_pos_emb, key_pos_emb = rotary_pos_emb
            else:
                query_pos_emb = key_pos_emb = rotary_pos_emb
            if query_pos_emb is not None:
                query = apply_rotary_pos_emb(query, query_pos_emb, config=self.config)
            if key_pos_emb is not None:
                key = apply_rotary_pos_emb(key, key_pos_emb, config=self.config)

        query = query.permute(1, 2, 0, 3)
        key = key.permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        attention_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attention_output = (
            attention_output.permute(2, 0, 1, 3)
            .contiguous()
            .view(sequence_length, batch_size, self.hidden_size)
        )
        attention_output = self.attention_output_projection(attention_output)
        prediction_state = prediction_state + F.dropout(
            attention_output,
            p=self.hidden_dropout,
            training=self.training,
        )

        gate, value_branch = self.ffn_input_projection(
            self.ffn_norm(prediction_state)
        ).chunk(2, dim=-1)
        ffn_output = self.ffn_output_projection(F.silu(gate) * value_branch)
        prediction_state = prediction_state + F.dropout(
            ffn_output,
            p=self.hidden_dropout,
            training=self.training,
        )
        return prediction_state

    def _validate_state(self, state: Tensor, label: str) -> None:
        if state.ndim != 3:
            raise ValueError(
                f"{label} must have shape [sequence, batch, hidden], got "
                f"{tuple(state.shape)}."
            )
        if state.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{label} expected hidden size {self.hidden_size}, got "
                f"{state.shape[-1]}."
            )

    def extra_repr(self) -> str:
        split = (
            "parameter_free_direct_identity_copy"
            if self.direct_identity_copy
            else "d_to_2d_stacked_identity"
        )
        return (
            f"hidden_size={self.hidden_size}, heads={self.num_attention_heads}, "
            f"ffn_hidden_size={self.ffn_hidden_size}, causal=True, "
            "memory=latest_latent_sequence, prediction_self_attention=False, "
            f"split={split}"
        )


def _scaleless_rms_norm(hidden_states: Tensor, eps: float) -> Tensor:
    """Apply the same mean-square normalization used by MCore's scaleless L2Norm."""

    normalized = hidden_states.float()
    normalized = normalized * torch.rsqrt(
        normalized.square().mean(dim=-1, keepdim=True) + eps
    )
    return normalized.to(dtype=hidden_states.dtype)


class _LatestLatentCrossAttentionMoEPredictionBlock(nn.Module):
    """One causal MHA/GQA cross-attention + LatentMoE prediction block."""

    def __init__(self, config: TransformerConfig, moe_layer: nn.Module) -> None:
        super().__init__()
        if config.num_query_groups is None:
            raise ValueError("PredictionBlock attention requires num_query_groups.")
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("PredictionBlock hidden size must divide into query heads.")
        if config.num_attention_heads % config.num_query_groups != 0:
            raise ValueError("PredictionBlock query heads must divide into KV heads.")
        if config.init_method is None or config.output_layer_init_method is None:
            raise ValueError("Transformer initialization methods must be configured.")

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_projection_size = self.num_query_groups * self.head_dim
        self.attention_dropout = config.attention_dropout
        self.hidden_dropout = config.hidden_dropout
        self.norm_eps = config.layernorm_epsilon

        self.prediction_norm = _FP32StatsRMSNorm(self.hidden_size, self.norm_eps)
        self.latent_norm = _FP32StatsRMSNorm(self.hidden_size, self.norm_eps)
        self.moe_norm = _FP32StatsRMSNorm(self.hidden_size, self.norm_eps)
        self.query_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_projection = nn.Linear(
            self.hidden_size, self.kv_projection_size, bias=False
        )
        self.value_projection = nn.Linear(
            self.hidden_size, self.kv_projection_size, bias=False
        )
        self.attention_gate_projection = (
            None
            if config.loop_prediction_disable_attention_output_gate
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )
        self.attention_output_projection = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )
        self.moe = moe_layer

        input_projections = [
            self.query_projection,
            self.key_projection,
            self.value_projection,
        ]
        if self.attention_gate_projection is not None:
            input_projections.append(self.attention_gate_projection)
        for projection in input_projections:
            config.init_method(projection.weight)
        config.output_layer_init_method(self.attention_output_projection.weight)

    def forward(
        self,
        latent_state: Tensor,
        prediction_state: Tensor,
        *,
        rotary_pos_emb: Optional[Union[Tensor, tuple[Tensor, Tensor]]] = None,
    ) -> Tensor:
        sequence_length, batch_size, _ = prediction_state.shape
        normalized_prediction = self.prediction_norm(prediction_state)
        query = self.query_projection(normalized_prediction).view(
            sequence_length,
            batch_size,
            self.num_attention_heads,
            self.head_dim,
        )
        gate = (
            self.attention_gate_projection(normalized_prediction).view_as(query)
            if self.attention_gate_projection is not None
            else None
        )
        normalized_latent = self.latent_norm(latent_state)
        key = self.key_projection(normalized_latent).view(
            sequence_length,
            batch_size,
            self.num_query_groups,
            self.head_dim,
        )
        value = self.value_projection(normalized_latent).view_as(key)

        if self.config.qk_l2_norm:
            query = _scaleless_rms_norm(query, self.norm_eps)
            key = _scaleless_rms_norm(key, self.norm_eps)
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                query_pos_emb, key_pos_emb = rotary_pos_emb
            else:
                query_pos_emb = key_pos_emb = rotary_pos_emb
            if query_pos_emb is not None:
                query = apply_rotary_pos_emb(query, query_pos_emb, config=self.config)
            if key_pos_emb is not None:
                key = apply_rotary_pos_emb(key, key_pos_emb, config=self.config)

        attention_output = F.scaled_dot_product_attention(
            query.permute(1, 2, 0, 3),
            key.permute(1, 2, 0, 3),
            value.permute(1, 2, 0, 3),
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.config.softmax_scale,
            enable_gqa=self.num_query_groups != self.num_attention_heads,
        ).permute(2, 0, 1, 3)
        if self.config.attention_output_norm:
            attention_output = _scaleless_rms_norm(attention_output, self.norm_eps)
        if gate is not None:
            attention_output = attention_output * torch.sigmoid(gate.float()).to(
                dtype=attention_output.dtype
            )
        attention_output = self.attention_output_projection(
            attention_output.contiguous().view(
                sequence_length, batch_size, self.hidden_size
            )
        )
        prediction_state = prediction_state + F.dropout(
            attention_output,
            p=self.hidden_dropout,
            training=self.training,
        )

        moe_output = self.moe(self.moe_norm(prediction_state))
        if not isinstance(moe_output, tuple) or len(moe_output) != 2:
            raise RuntimeError("PredictionBlock MoE must return (output, bias).")
        moe_hidden_states, moe_bias = moe_output
        if moe_bias is not None:
            raise RuntimeError("Bias-free PredictionBlock MoE returned a bias tensor.")
        return prediction_state + F.dropout(
            moe_hidden_states,
            p=self.hidden_dropout,
            training=self.training,
        )


class LatestLatentCrossAttentionMoEPredictionAccumulator(nn.Module):
    """Persistent stream updated by two distinct recurrently shared MoE blocks."""

    def __init__(
        self,
        config: TransformerConfig,
        prediction_moe_layers: tuple[nn.Module, nn.Module],
    ) -> None:
        super().__init__()
        if config.tensor_model_parallel_size != 1:
            raise ValueError("MoE PredictionBlocks currently require tensor parallel size one.")
        if config.context_parallel_size != 1:
            raise ValueError("MoE PredictionBlocks currently require context parallel size one.")
        if len(prediction_moe_layers) != 2:
            raise ValueError("Dual-stream 2Loop requires exactly two PredictionBlocks.")

        self.tp_group = getattr(prediction_moe_layers[0], "tp_group", None)
        self.config = config
        self.hidden_size = config.hidden_size
        self.direct_identity_copy = config.loop_latent_prediction_direct_identity_copy
        if self.direct_identity_copy:
            self.split_projection = None
        else:
            self.split_projection = nn.Linear(
                self.hidden_size, 2 * self.hidden_size, bias=False
            )
            nn.init.eye_(self.split_projection.weight[: self.hidden_size])
            nn.init.eye_(self.split_projection.weight[self.hidden_size :])
            self.split_projection.weight.is_loop_state_prediction_identity_parameter = True

        self.prediction_blocks = nn.ModuleList(
            [
                _LatestLatentCrossAttentionMoEPredictionBlock(config, moe_layer)
                for moe_layer in prediction_moe_layers
            ]
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Preserve global expert identities through the plain prediction containers."""
        from megatron.core.transformer.utils import sharded_state_dict_recursive

        return sharded_state_dict_recursive(
            self, prefix, sharded_offsets, metadata, self.tp_group
        )

    def initialize(self, prelude_state: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_state(prelude_state, "Prelude state")
        if self.direct_identity_copy:
            return prelude_state.clone(), prelude_state.clone()
        assert self.split_projection is not None
        return self.split_projection(prelude_state).chunk(2, dim=-1)

    def update_prediction(
        self,
        latent_state: Tensor,
        prediction_state: Tensor,
        *,
        prediction_block_index: int,
        rotary_pos_emb: Optional[Union[Tensor, tuple[Tensor, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        inference_context: object = None,
        packed_seq_params: object = None,
    ) -> Tensor:
        self._validate_state(latent_state, "latent state")
        self._validate_state(prediction_state, "prediction state")
        if latent_state.shape != prediction_state.shape:
            raise ValueError("latent and prediction states must have identical shapes.")
        if prediction_block_index not in (0, 1):
            raise ValueError(
                f"prediction_block_index must be zero or one, got {prediction_block_index}."
            )
        if attention_mask is not None:
            raise ValueError(
                "MoE PredictionBlocks expect the fused causal-mask path with no materialized mask."
            )
        if inference_context is not None:
            raise NotImplementedError(
                "MoE PredictionBlock generation requires an explicit latent-memory cache policy."
            )
        if packed_seq_params is not None:
            raise NotImplementedError("Packed sequences are not supported by MoE PredictionBlocks.")
        return self.prediction_blocks[prediction_block_index](
            latent_state,
            prediction_state,
            rotary_pos_emb=rotary_pos_emb,
        )

    def _validate_state(self, state: Tensor, label: str) -> None:
        if state.ndim != 3:
            raise ValueError(
                f"{label} must have shape [sequence, batch, hidden], got {tuple(state.shape)}."
            )
        if state.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{label} expected hidden size {self.hidden_size}, got {state.shape[-1]}."
            )

    def extra_repr(self) -> str:
        attention_type = (
            "mha"
            if self.config.num_query_groups == self.config.num_attention_heads
            else "gqa"
        )
        output_gate = not self.config.loop_prediction_disable_attention_output_gate
        return (
            f"hidden_size={self.hidden_size}, physical_blocks=2, "
            f"sharing=per_block_across_visits, causal_{attention_type}=True, "
            f"attention_output_gate={output_gate}, latent_moe=True, "
            f"direct_identity_copy={self.direct_identity_copy}"
        )
