# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import cat_with_oom_fallback
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class LinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class LinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        stride: int = 1,
    ) -> LinearFc1Interface:
        """Builds a linear_fc1 layer for MLP."""
        ...


class TEActivationFunctionInterface(Protocol):
    """Interface for activation_function module in MLP."""

    def forward(self, input_: torch.Tensor, /) -> torch.Tensor:
        """Forward method for activation_function module."""
        ...


class TEActivationFunctionBuilder(Protocol):
    """Protocol for activation_function module in MLP."""

    def __call__(self, *, config: TransformerConfig) -> TEActivationFunctionInterface:
        """Builds an activation function module for MLP."""
        ...


class LinearFc2Interface(Protocol):
    """Interface for linear_fc2 module in MLP."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        /,
        *,
        freeze_weight_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class LinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> LinearFc2Interface:
        """Builds a linear_fc2 layer for MLP."""
        ...


@dataclass
class MLPSubmodules:
    """The module specifications used to construct an MLP."""

    linear_fc1: LinearFc1Builder

    linear_fc2: LinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


def get_slow_fast_route(
    loop_visit_index: int,
    slow_end_visit: int,
    fast_start_visit: int,
) -> tuple[bool, bool]:
    """Return whether a loop visit trains the slow and fast weights."""
    if loop_visit_index < 1:
        raise ValueError(
            f"slow/fast routing requires a positive loop visit, got {loop_visit_index}."
        )
    return loop_visit_index <= slow_end_visit, loop_visit_index >= fast_start_visit


class SlowFastDownProjectionAdapter(MegatronModule):
    """Layer-local LoRA branch for a looped FFN down projection.

    The adapter follows standard LoRA initialization: A uses the default
    Kaiming-uniform initialization of ``nn.Linear`` and B is exactly zero, so
    the initial function is identical to the unmodified loop model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        input_size: int,
        output_size: int,
        rank: int,
        alpha: float,
        layer_number: int,
        tp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super().__init__(config=config)
        if config.tensor_model_parallel_size != 1:
            raise ValueError("SlowFastDownProjectionAdapter currently requires TP=1.")

        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.layer_number = layer_number
        self.tp_group = tp_group
        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        factory_kwargs = {"device": device, "dtype": config.params_dtype}
        self.lora_a = torch.nn.Parameter(torch.empty(rank, input_size, **factory_kwargs))
        self.lora_b = torch.nn.Parameter(torch.empty(output_size, rank, **factory_kwargs))

        if config.perform_initialization:
            torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_b)

        for parameter in (self.lora_a, self.lora_b):
            setattr(parameter, "allreduce", True)
            setattr(parameter, "is_slow_fast_adapter_parameter", True)

        # Audit snapshots are intentionally not buffers or checkpoint state. A
        # fresh run therefore measures the first update relative to exact LoRA
        # initialization, while a resumed run establishes a new baseline at its
        # first audit event.
        self._audit_previous_a: torch.Tensor | None = self.lora_a.detach().clone()
        self._audit_previous_b: torch.Tensor | None = self.lora_b.detach().clone()
        self._audit_previous_iteration: int = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        low_rank = F.linear(hidden_states, self.lora_a)
        return F.linear(low_rank, self.lora_b) * self.scaling

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Use one checkpoint key per physical layer instead of a sparse layer axis.

        Only loop layers own adapters, so treating them as a dense 16-layer
        tensor leaves invalid holes. State-dict keys retain their local module
        paths for ``load_state_dict``; only distributed-checkpoint tensor keys
        gain the explicit physical-layer component.
        """
        del sharded_offsets
        sharded_state_dict = super().sharded_state_dict(prefix, (), metadata)
        replace_prefix_for_sharding(
            sharded_state_dict,
            prefix,
            f"{prefix}physical_layer_{self.layer_number:02d}.",
        )
        return sharded_state_dict


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.layer_number: int | None = None
        self.slow_fast_adapter: SlowFastDownProjectionAdapter | None = None
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = not_none(self.config.ffn_hidden_size)

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        # For GLU/SwiGLU, use stride=2 because each TP rank stores interleaved [gate, up] portions.
        # This is critical for correct weight resharding across different TP sizes.
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
            fc1_stride = 2
            if self.config.use_kitchen:
                # Kitchen Linear doesn't support stride != 1.
                # Weight resharding across TP sizes will have aforementioned problems.
                fc1_stride = 1
        else:
            fc1_stride = 1

        # Use moe_latent_size only for routed experts. 'is_expert' is false for
        # shared_experts.
        use_latent_size = (self.config.moe_latent_size is not None) and is_expert

        self.linear_fc1 = submodules.linear_fc1(
            self.input_size if not use_latent_size else not_none(self.config.moe_latent_size),
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
            stride=fc1_stride,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = submodules.linear_fc2(
            not_none(self.config.ffn_hidden_size),
            not_none(
                self.config.hidden_size if not use_latent_size else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def set_layer_number(self, layer_number: int) -> None:
        """Attach one independent adapter to each configured physical loop layer."""
        self.layer_number = layer_number
        if not self.config.slow_fast_ffn:
            return
        if not_none(self.config.loop_layer_start) <= layer_number <= not_none(
            self.config.loop_layer_end
        ):
            self.slow_fast_adapter = SlowFastDownProjectionAdapter(
                config=self.config,
                input_size=not_none(self.config.ffn_hidden_size),
                output_size=not_none(self.config.hidden_size),
                rank=self.config.slow_fast_ffn_rank,
                alpha=self.config.slow_fast_ffn_alpha,
                layer_number=layer_number,
                tp_group=self.tp_group,
            )

    def forward(
        self, hidden_states: torch.Tensor, per_token_scale: torch.Tensor | None = None, **kwargs
    ):
        """Perform the forward pass through the MLP block."""
        loop_visit_index = int(kwargs.pop("loop_visit_index", 0))
        slow_active = True
        fast_active = False
        if self.slow_fast_adapter is not None:
            slow_active, fast_active = get_slow_fast_route(
                loop_visit_index,
                self.config.slow_fast_ffn_slow_end_visit,
                self.config.slow_fast_ffn_fast_start_visit,
            )
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")

        if self.slow_fast_adapter is None:
            # Preserve the exact baseline call signature for alternate MLP
            # implementations when the experiment is disabled.
            output, output_bias = apply_module(self.linear_fc2)(
                cast(torch.Tensor, intermediate_parallel)
            )
        else:
            output, output_bias = apply_module(self.linear_fc2)(
                cast(torch.Tensor, intermediate_parallel),
                freeze_weight_gradient=not slow_active,
            )
        if fast_active:
            output = output + not_none(self.slow_fast_adapter)(
                cast(torch.Tensor, intermediate_parallel)
            )
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()

    @classmethod
    def as_mlp_submodule(
        cls,
        submodules: MLPSubmodules,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        is_mtp_layer: bool,
        is_expert: bool = False,
        input_size: int | None = None,
        ffn_hidden_size: int | None = None,
    ) -> MLP:
        """Helper function to build an MLP as a TransformerLayer's mlp submodule."""
        del is_mtp_layer
        assert hasattr(
            pg_collection, 'tp'
        ), 'TP process group is required for MLP in TransformerLayer'
        return cls(
            config=config,
            submodules=submodules,
            tp_group=pg_collection.tp,
            is_expert=is_expert,
            input_size=input_size,
            ffn_hidden_size=ffn_hidden_size,
        )


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        if singleton_local_shards:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            w_key = f'{key}_w'
            v_key = f'{key}_v'
        else:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
            offset_v = (
                swiglu_shard_axis + prepend_axis_num,
                rank_offset + axis_frag,
                axis_frag * 2,
            )
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *sharded_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *sharded_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        cat_with_oom_fallback,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )
