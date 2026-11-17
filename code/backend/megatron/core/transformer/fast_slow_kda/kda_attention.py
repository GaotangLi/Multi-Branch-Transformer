"""Opt-in, full-sequence KDA self-attention adapter for Idea2 A.

This replaces an entire fast ``self_attention`` module, not ``core_attention``.
The enclosing TransformerLayer must have IdentityOp as input_layernorm: the
single pre-attention RMSNorm lives here, replacing TE's fused QKV input norm.
The enclosing layer still owns the attention residual and the unmodified MoE.

Reviewed FLA API: fla-org/flash-linear-attention
0267589b1593c1b4c6752bbf0f0b5a8528e8fb9e (0.6.0), fla/layers/kda.py and
fla/ops/kda/chunk.py. FLA is imported only when this adapter is constructed;
ordinary model imports never need this optional dependency. FLA remains a
separately pinned dependency, under its own MIT license.

Scope: TP=PP=CP=1, ordinary replicated dense parameters, full unpadded
sequences, BF16/FP32 training and validation. EP belongs to the surrounding
native MoE and does not shard these parameters. There is deliberately no
incremental inference cache. Each call, including each shared-weight visit,
starts fresh token-axis matrix and convolution states, without detaching the
incoming hidden-state graph. Token-axis state is not loop-depth state.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import nn

from megatron.core.transformer.module import MegatronModule


FLA_REVIEWED_REVISION = "0267589b1593c1b4c6752bbf0f0b5a8528e8fb9e"


class KDAAttention(MegatronModule):
    """Megatron [S,B,d] -> [S,B,d] KDA, returning ``(output, None)``.

    Formal M1 uses d=896, H=14, K=V=128, inner dimension 1792, and both
    native low-rank gates have rank 128. This is expansion of the mixer width,
    not a hidden-width reduction. Smaller dimensions are accepted for isolated
    correctness tests; the formal launcher must seal the intended dimensions.
    """

    def __init__(
        self,
        config,
        layer_number: int | None = None,
        pg_collection=None,
        *,
        num_heads: int = 14,
        head_dim: int = 128,
        gate_low_rank: int = 128,
        conv_size: int = 4,
        physical_depth: int = 21,
        submodules=None,
        attn_mask_type=None,
        pp_layer_offset: int | None = None,
        cp_comm_type=None,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = getattr(pg_collection, "tp", None)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.gate_low_rank = int(gate_low_rank)
        self.conv_size = int(conv_size)
        self.physical_depth = int(physical_depth)
        self.key_dim = self.value_dim = self.num_heads * self.head_dim
        self.head_k_dim = self.head_v_dim = self.head_dim
        self.expand_v = 1.0
        self.mode = "chunk"
        self.is_kda_attention = True
        self.scale = self.head_dim ** -0.5 if self.head_dim > 0 else 0.0
        self._validate_layout()

        if min(self.hidden_size, self.num_heads, self.head_dim,
               self.gate_low_rank, self.physical_depth) < 1:
            raise ValueError("KDA dimensions and physical_depth must be positive.")
        if self.head_dim not in (16, 32, 64, 128):
            raise ValueError("KDA supports tested head dimensions 16/32/64/128 only.")
        if self.conv_size != 4:
            raise ValueError("This experiment fixes native causal short-convolution size to 4.")
        if submodules is not None:
            raise ValueError("KDA owns its projections; do not supply an old QKV submodule spec.")
        if pp_layer_offset not in (None, 0) or cp_comm_type is not None:
            raise ValueError("KDA adapter does not support PP offsets or CP communication.")
        # attn_mask_type is only a construction annotation. Runtime materialized
        # masks are rejected below; the chunk kernel itself is always causal.
        if attn_mask_type is not None and getattr(attn_mask_type, "name", "causal") != "causal":
            raise ValueError("KDA is strictly causal; non-causal attention types are unsupported.")
        if getattr(config, "normalization", "RMSNorm") != "RMSNorm":
            raise ValueError("The KDA fast-only recipe requires the original RMSNorm backbone.")
        if getattr(config, "layernorm_zero_centered_gamma", False):
            raise ValueError("Zero-centered gamma is not supported by this KDA adapter.")
        if os.environ.get("FLA_CONV_BACKEND", "triton") != "triton":
            raise ValueError("This recipe pins FLA_CONV_BACKEND=triton; CUDA extension is not used.")

        # No eager import at module scope: loading a legacy model must not import
        # FLA, Triton kernels, Hugging Face model layers, or optional conv wheels.
        try:
            from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
            from fla.ops.kda import chunk_kda
        except ImportError as error:
            raise ImportError(
                "Idea2 KDA requires its isolated pinned FLA dependency "
                f"({FLA_REVIEWED_REVISION}); do not modify the shared training environment."
            ) from error
        self._chunk_kda = chunk_kda

        dtype = getattr(config, "params_dtype", torch.float32)
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Initial KDA adapter qualifies only FP32/BF16 parameters.")
        device = (
            torch.device("cpu")
            if getattr(config, "use_cpu_initialization", False)
            else torch.device("cuda", torch.cuda.current_device())
        )
        factory = {"device": device, "dtype": dtype}
        eps = float(config.layernorm_epsilon)
        self.pre_norm = RMSNorm(self.hidden_size, eps=eps, **factory)
        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False, **factory)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False, **factory)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False, **factory)
        self.q_conv1d = ShortConvolution(
            self.key_dim, self.conv_size, bias=False, activation="silu",
            backend="triton", **factory,
        )
        self.k_conv1d = ShortConvolution(
            self.key_dim, self.conv_size, bias=False, activation="silu",
            backend="triton", **factory,
        )
        self.v_conv1d = ShortConvolution(
            self.value_dim, self.conv_size, bias=False, activation="silu",
            backend="triton", **factory,
        )
        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank, bias=False, **factory),
            nn.Linear(self.gate_low_rank, self.key_dim, bias=False, **factory),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False, **factory)
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank, bias=False, **factory),
            nn.Linear(self.gate_low_rank, self.value_dim, bias=True, **factory),
        )
        self.o_norm = FusedRMSNormGated(
            self.head_dim, activation="sigmoid", eps=eps, **factory,
        )
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False, **factory)

        # Native KDA stable gate initialization. These are real optimizer-owned
        # FP32 parameters, not buffers or detached copies of reduced-precision
        # parameters. _apply below protects them during Float16Module.bfloat16().
        self.A_log = nn.Parameter(torch.empty(self.num_heads, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, device=device, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        # Opt in only these native FP32 parameters to the mixed-dtype optimizer
        # checkpoint mapping; legacy models keep their existing save path.
        self.A_log._is_fast_slow_kda_fp32_gate = True
        self.dt_bias._is_fast_slow_kda_fp32_gate = True
        self.reset_parameters()
        for parameter in self.parameters():
            # These dense parameters reduce over ordinary DP, not expert-DP.
            parameter.allreduce = True
            parameter.sequence_parallel = False
            parameter.tensor_model_parallel = False

    def _validate_layout(self):
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size",
                     "context_parallel_size"):
            if int(getattr(self.config, name, 1)) != 1:
                raise ValueError(f"KDA adapter requires {name}=1.")
        if getattr(self.config, "sequence_parallel", False):
            raise ValueError("KDA adapter does not support sequence parallelism.")
        if getattr(self.config, "fp8", None) or getattr(self.config, "fp4", None):
            raise ValueError("KDA adapter does not support FP8/FP4.")
        if getattr(self.config, "fp16", False):
            raise ValueError("KDA adapter has not qualified FP16; use BF16 or FP32.")
        if self.pg_collection is not None:
            for name in ("tp", "pp", "cp"):
                group = getattr(self.pg_collection, name, None)
                if group is not None and group.size() != 1:
                    raise ValueError(f"KDA adapter received {name} process group larger than one.")

    @torch.no_grad()
    def reset_parameters(self):
        """Preserve the M1 matrix-init convention and KDA's native gate/conv init."""
        # nn.Linear constructors initialize even for a subsequent strict load;
        # match native initialization control for our deliberate custom reset.
        if not getattr(self.config, "perform_initialization", True):
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = 0.02 / math.sqrt(2 * self.physical_depth) if module is self.o_proj else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.ones_(self.pre_norm.weight)
        nn.init.ones_(self.o_norm.weight)
        # ShortConvolution retains its native nn.Conv1d Kaiming-uniform init;
        # it is not one of the Transformer projection matrices above.
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1.0, 16.0).log_())
        dt = torch.empty_like(self.dt_bias).uniform_(math.log(0.001), math.log(0.1)).exp_()
        dt.clamp_(min=1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _apply(self, fn, recurse=True):
        """Keep gate Parameters FP32 without replacing their Python identities.

        In particular, converting the full model to BF16 must not first round
        these values to BF16 and then cast back. Only use fn on an empty probe
        to discover the destination device. Device/dtype moves remain a setup
        operation; moving optimizer/DDP-bound gate storage is rejected.
        """
        gate_parameters = {name: self._parameters[name] for name in ("A_log", "dt_bias")
                           if name in self._parameters}
        destinations = {}
        for name, parameter in gate_parameters.items():
            probe = fn(torch.empty(0, dtype=torch.float32, device=parameter.device))
            if probe.is_meta or not probe.is_floating_point():
                raise ValueError("KDA FP32 gates do not support meta/non-floating conversion.")
            destinations[name] = probe.device
            if hasattr(parameter, "main_grad") and probe.device != parameter.device:
                raise RuntimeError("Cannot move KDA gate storage after DDP/optimizer binding.")
        for name in gate_parameters:
            self._parameters.pop(name)
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._parameters.update(gate_parameters)
        with torch.no_grad():
            for name, parameter in gate_parameters.items():
                if parameter.device != destinations[name] or parameter.dtype != torch.float32:
                    parameter.data = parameter.data.to(device=destinations[name], dtype=torch.float32)
                if parameter.grad is not None and (
                    parameter.grad.device != destinations[name] or parameter.grad.dtype != torch.float32
                ):
                    parameter.grad.data = parameter.grad.data.to(
                        device=destinations[name], dtype=torch.float32,
                    )
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        window_size=None,
        inference_params=None,
        key_value_states=None,
        **kwargs: Any,
    ):
        unsupported = {
            "attention_mask": attention_mask, "inference_context": inference_context,
            "inference_params": inference_params, "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos, "rotary_pos_sin": rotary_pos_sin,
            "rotary_pos_cos_sin": rotary_pos_cos_sin, "attention_bias": attention_bias,
            "packed_seq_params": packed_seq_params, "sequence_len_offset": sequence_len_offset,
            "window_size": window_size, "key_value_states": key_value_states,
        }
        bad = [name for name, value in unsupported.items() if value is not None]
        if bad or kwargs:
            raise NotImplementedError(
                "KDA accepts full unpadded causal sequences only; no cache/RoPE/SWA/packed "
                f"arguments may be silently ignored: {bad + list(kwargs)}"
            )
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"KDA requires [S,B,{self.hidden_size}] hidden states.")
        if hidden_states.shape[0] < 2 or hidden_states.shape[1] < 1:
            raise ValueError("KDA full-sequence adapter requires S>=2 and B>=1; decode is unsupported.")
        if self.A_log.dtype != torch.float32 or self.dt_bias.dtype != torch.float32:
            raise RuntimeError("KDA gate parameters must remain FP32 before optimizer construction.")

        x = self.pre_norm(hidden_states).transpose(0, 1).contiguous()
        batch, length, _ = x.shape
        # Fresh state for every invocation and every shared-weight visit. Never
        # reuse the previous visit's end-of-sequence state at its first token.
        q, _ = self.q_conv1d(self.q_proj(x), cache=None, output_final_state=False, cu_seqlens=None)
        k, _ = self.k_conv1d(self.k_proj(x), cache=None, output_final_state=False, cu_seqlens=None)
        v, _ = self.v_conv1d(self.v_proj(x), cache=None, output_final_state=False, cu_seqlens=None)
        shape = (batch, length, self.num_heads, self.head_dim)
        q, k, v = (value.reshape(shape).contiguous() for value in (q, k, v))
        g = self.f_proj(x).reshape(shape).contiguous()
        beta = self.b_proj(x).contiguous()
        output, final_state = self._chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            A_log=self.A_log, dt_bias=self.dt_bias, scale=self.scale,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False,
            safe_gate=False, lower_bound=None, state_v_first=True,
            disable_recompute=False, cu_seqlens=None,
        )
        if final_state is not None:
            raise RuntimeError("FLA returned unexpected persistent KDA state with use_cache=False.")
        output_gate = self.g_proj(x).reshape(shape).contiguous()
        output = self.o_norm(output, output_gate).reshape(batch, length, self.value_dim)
        output = self.o_proj(output).transpose(0, 1).contiguous()
        return output, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Native replicated-dense sharding, retaining optimizer Parameter identity.

        The outer native TransformerLayer and MoELayer continue to own their
        own sharding. There is no expert- or visit-axis in this module's keys.
        """
        self._validate_layout()
        if sharded_offsets:
            raise ValueError("KDA adapter does not support enclosing PP/layer-axis sharding.")
        return super().sharded_state_dict(prefix, sharded_offsets, metadata)

    def extra_repr(self):
        return (
            f"d={self.hidden_size}, heads={self.num_heads}, head_dim={self.head_dim}, "
            f"inner_dim={self.key_dim}, gate_rank={self.gate_low_rank}, conv={self.conv_size}, "
            "mode=chunk, per_call_state=fresh, pre_norm=RMSNorm, incremental_decode=False"
        )
