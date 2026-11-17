# Native embedding, attention, MoE and head composition follows NVIDIA Megatron.
# See code/backend/LICENSE for their attribution and license terms.
"""Explicit Multi-branch Transformer language model with native GPU kernels."""
from __future__ import annotations

import dataclasses
from functools import partial

import torch

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import (
    TEActivationOp, TEColumnParallelGroupedLinear, TENorm, TERowParallelGroupedLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.utils import make_viewless_tensor

from .attention import JointKVAttention, attention_submodules
from .config import ModelConfig
from .decoder import MBTDecoder
from .modules import CheckpointModule, JointKVProjection


class MBTBlock(CheckpointModule):
    """One native gated attention + latent MoE block, with explicit residuals."""

    def __init__(self, config, layer_number: int, *, shared_kv: bool, pg_collection):
        super().__init__()
        self.config = config
        self.tp_group = pg_collection.tp
        self.layer_number = layer_number
        self.is_moe_layer = True
        self.uses_shared_kv = shared_kv
        self.self_attention = (JointKVAttention(config, layer_number, pg_collection=pg_collection)
            if shared_kv else SelfAttention(config=config, submodules=attention_submodules(),
                layer_number=layer_number, attn_mask_type=AttnMaskType.causal, pg_collection=pg_collection))
        self.pre_mlp_layernorm = TENorm(config=config, hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon, has_residual=True)
        experts = partial(TEGroupedMLP, submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear,
            linear_fc2=TERowParallelGroupedLinear,
            activation_func=TEActivationOp,
        ))
        self.mlp = MoELayer(config=config, submodules=MoESubmodules(experts=experts),
            layer_number=layer_number, pg_collection=pg_collection)
        self.mlp.set_layer_number(layer_number)

    def normalized_attention_input(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.uses_shared_kv:
            raise ValueError("Only Joint-KV blocks expose a shared-KV input")
        return self.self_attention.normalized_input(hidden)

    def forward(self, hidden, attention_mask=None, rotary_pos_emb=None, shared_kv=None):
        if self.uses_shared_kv:
            if shared_kv is None:
                raise ValueError("Joint-KV block requires shared_kv")
            attention_output = self.self_attention(hidden, attention_mask,
                rotary_pos_emb=rotary_pos_emb, shared_kv=shared_kv)
        else:
            if shared_kv is not None:
                raise ValueError("Independent-KV block cannot consume shared_kv")
            attention_output = self.self_attention(hidden, attention_mask, rotary_pos_emb=rotary_pos_emb)
        residual_add = get_bias_dropout_add(self.training, self.config.bias_dropout_fusion)
        hidden = residual_add(attention_output, hidden, self.config.hidden_dropout)
        normalized = self.pre_mlp_layernorm(hidden)
        if isinstance(normalized, tuple):
            normalized, residual = normalized
        else:
            residual = hidden
        moe_output = self.mlp(normalized)
        hidden = residual_add(moe_output, residual, self.config.hidden_dropout)
        return make_viewless_tensor(inp=hidden, requires_grad=hidden.requires_grad, keep_graph=True)


class MBTModel(LanguageModule):
    """Embedding → MBT decoder → tied vocabulary head.

    The released training graph is BF16, full-sequence causal attention with
    TP=PP=CP=1 and EP=1/2/4/8. It does not implement a paged decoding engine.
    """

    def __init__(self, model_config: ModelConfig, native_config, *, vocab_size=128256,
                 max_sequence_length=4096, pre_process=True, post_process=True,
                 parallel_output=True, pg_collection=None):
        if not pre_process or not post_process:
            raise ValueError("The MBT release uses a complete model on each EP/DP rank; PP=1")
        super().__init__(config=native_config, pg_collection=pg_collection)
        self.model_config = model_config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.max_position_embeddings = max_sequence_length
        self.pre_process = self.post_process = True
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = True
        self.fp16_lm_cross_entropy = False
        self.disable_param_offloading = True
        self.model_type = ModelType.encoder_or_decoder
        self.position_embedding_type = "rope"
        self.embedding_activation_buffer = self.grad_output_buffer = None
        self.embedding = LanguageModelEmbedding(config=native_config, vocab_size=vocab_size,
            max_sequence_length=max_sequence_length, position_embedding_type="rope",
            scatter_to_sequence_parallel=False, tp_group=self.pg_collection.tp)
        self.rotary_pos_emb = RotaryEmbedding(kv_channels=64, rotary_percent=1.,
            rotary_interleaved=native_config.rotary_interleaved, rotary_base=500000,
            use_cpu_initialization=native_config.use_cpu_initialization, cp_group=self.pg_collection.cp)

        def layer_factory(index, shape, shared_kv):
            width, latent_width, expert_width = shape
            layer_config = dataclasses.replace(native_config, hidden_size=width,
                num_attention_heads=width//64, num_query_groups=width//128,
                kv_channels=64, ffn_hidden_size=4*width,
                moe_latent_size=latent_width, moe_ffn_hidden_size=expert_width)
            return MBTBlock(layer_config, index+1, shared_kv=shared_kv,
                pg_collection=self.pg_collection)

        device = "cpu" if native_config.use_cpu_initialization else torch.cuda.current_device()
        dtype = native_config.params_dtype
        branch_width = model_config.branch_shape[0]

        def joint_kv_factory(depth):
            key_norm = TENorm(config=native_config, hidden_size=64, eps=native_config.layernorm_epsilon)
            return JointKVProjection(branch_width, len(model_config.branches),
                initializer=native_config.init_method, key_norm=key_norm,
                device=device, dtype=dtype, tp_group=self.pg_collection.tp)

        final_norm = TENorm(config=native_config, hidden_size=native_config.hidden_size,
            eps=native_config.layernorm_epsilon)
        self.decoder = MBTDecoder(model_config, layer_factory, final_norm,
            config=native_config, device=device, dtype=dtype,
            joint_kv_factory=joint_kv_factory, tp_group=self.pg_collection.tp)
        self.output_layer = tensor_parallel.ColumnParallelLinear(native_config.hidden_size,
            vocab_size, config=native_config, init_method=native_config.init_method,
            bias=False, skip_bias_add=False, gather_output=not parallel_output,
            skip_weight_param_allocation=True, tp_group=self.pg_collection.tp)
        self.setup_embeddings_and_output_layer()
        self._audit_parameter_counts()

    def _audit_parameter_counts(self):
        expert_parallel_size = self.config.expert_model_parallel_size
        expected = self.model_config.counts(expert_parallel_size)
        resident = sum(parameter.numel() for parameter in self.parameters())
        physical = sum(parameter.numel()*(1 if getattr(parameter, "allreduce", True)
            else expert_parallel_size) for parameter in self.parameters())
        if (resident, physical) != (expected["resident"], expected["physical"]):
            raise RuntimeError(f"native model parameter count mismatch: {(resident, physical)} vs "
                               f"{(expected['resident'], expected['physical'])}")
        self.decoder.mbt_receipt = dict(self.model_config.architecture(),
            resident_parameters=resident, global_parameters=physical,
            implementation="explicit_mbt", checkpoint_format="mbt-hierarchical-v2")

    def set_input_tensor(self, input_tensor):
        values = input_tensor if isinstance(input_tensor, list) else [input_tensor]
        if len(values) != 1 or values[0] is not None:
            raise ValueError("MBT uses PP=1 and accepts no external pipeline activation")

    def forward(self, input_ids, position_ids, attention_mask=None, labels=None,
                decoder_input=None, loss_mask=None, runtime_gather_output=None):
        # The trainer applies loss_mask to per-token CE, as in native Megatron.
        del loss_mask
        if attention_mask is not None:
            raise ValueError("MBT training uses causal attention without an explicit mask tensor")
        hidden = decoder_input if decoder_input is not None else self.embedding(
            input_ids=input_ids, position_ids=position_ids)
        if hidden.ndim != 3:
            raise ValueError("decoder input must have shape [sequence, batch, width]")
        rotary = self.rotary_pos_emb(hidden.shape[0])
        hidden = self.decoder(hidden, rotary_pos_emb=rotary)
        logits, _ = self.output_layer(hidden, weight=self.shared_embedding_or_output_weight(),
            runtime_gather_output=runtime_gather_output)
        if labels is None:
            return logits.transpose(0, 1).contiguous()
        return self.compute_language_model_loss(labels, logits)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        result = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        extra = result.pop(prefix+"output_layer._extra_state", None)
        if extra is not None and extra.data is not None:
            raise RuntimeError("unexpected output-layer extra checkpoint state")
        return result
