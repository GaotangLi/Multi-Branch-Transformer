# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Set, Union, cast

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.recompute import checkpointed_forward
from megatron.core.transformer.enums import InferenceCudaGraphScope, LayerType
from megatron.core.transformer.loop_depth import active_loop_depth_indices
from megatron.core.transformer.loop_latent_prediction import (
    LatentPredictionAccumulator,
    LatestLatentCrossAttentionMoEPredictionAccumulator,
    LatestLatentCrossAttentionPredictionAccumulator,
    prediction_update_offsets,
)
from megatron.core.transformer.loop_state_prediction import (
    DirectLinearStatePredictionSplit,
    DirectMLPStatePredictionSplit,
    IdentityLinearStatePredictionSplit,
    PerVisitIdentityLinearStatePredictionSplit,
    ProjectedResidualMLPStatePredictionSplit,
)
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.temporal_jacobi import TemporalJacobiMixer
from megatron.core.transformer.torch_norm import LayerNormBuilder
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_pg_rank,
    make_viewless_tensor,
)

try:
    import apex  # pylint: disable=unused-import

    HAVE_APEX = True
except ImportError:
    HAVE_APEX = False

get_cpu_offload_context = None
te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TENorm, get_cpu_offload_context

    LayerNormImpl = TENorm

elif HAVE_APEX:
    LayerNormImpl = FusedLayerNorm

else:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    LayerNormImpl = WrappedTorchNorm


logger = logging.getLogger(__name__)


def get_layer_execution_order(
    num_layers: int,
    loop_layer_start: Optional[int] = None,
    loop_layer_end: Optional[int] = None,
    loop_repeats: int = 1,
) -> tuple[int, ...]:
    """Return zero-based physical layer indices in execution order.

    Loop bounds are one-based and inclusive at the configuration boundary. Repeating
    an index reuses the existing module and therefore does not create parameters.
    """
    if loop_repeats < 1:
        raise ValueError(f"loop_repeats must be at least 1, got {loop_repeats}.")
    if loop_repeats == 1:
        return tuple(range(num_layers))
    if loop_layer_start is None or loop_layer_end is None:
        raise ValueError("Loop bounds are required when loop_repeats > 1.")
    if not 1 <= loop_layer_start <= loop_layer_end <= num_layers:
        raise ValueError(
            "Loop bounds must satisfy 1 <= start <= end <= num_layers; "
            f"got start={loop_layer_start}, end={loop_layer_end}, num_layers={num_layers}."
        )

    start = loop_layer_start - 1
    end = loop_layer_end
    return (
        tuple(range(start))
        + tuple(range(start, end)) * loop_repeats
        + tuple(range(end, num_layers))
    )


def get_layer_execution_visits(
    layer_execution_order: tuple[int, ...],
    loop_layer_start: Optional[int] = None,
    loop_layer_end: Optional[int] = None,
    loop_repeats: int = 1,
) -> tuple[int, ...]:
    """Return the one-based loop visit for each execution position.

    Prelude and coda positions use zero. Loop bounds remain one-based and
    inclusive at the configuration boundary, while execution-order entries are
    zero-based physical layer indices.
    """
    if loop_repeats == 1:
        return (0,) * len(layer_execution_order)
    if loop_layer_start is None or loop_layer_end is None:
        raise ValueError("Loop bounds are required when loop_repeats > 1.")

    start_index = loop_layer_start - 1
    end_index = loop_layer_end
    loop_width = end_index - start_index
    loop_execution_start = start_index
    loop_execution_end = loop_execution_start + loop_width * loop_repeats
    visits = []
    for execution_position, layer_index in enumerate(layer_execution_order):
        if (
            loop_execution_start <= execution_position < loop_execution_end
            and start_index <= layer_index < end_index
        ):
            visits.append((execution_position - loop_execution_start) // loop_width + 1)
        else:
            visits.append(0)
    return tuple(visits)


def get_runtime_layer_execution_positions(
    layer_execution_visits: tuple[int, ...],
    runtime_loop_repeats: int,
    configured_loop_repeats: int,
) -> tuple[int, ...]:
    """Select Prelude, the requested loop visits, and Coda from a max-depth schedule."""

    if not 1 <= runtime_loop_repeats <= configured_loop_repeats:
        raise ValueError(
            "runtime_loop_repeats must lie in [1, configured_loop_repeats]; "
            f"got {runtime_loop_repeats} and {configured_loop_repeats}."
        )
    return tuple(
        execution_position
        for execution_position, visit in enumerate(layer_execution_visits)
        if visit == 0 or visit <= runtime_loop_repeats
    )


def get_num_layers_to_build(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):

        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert (
                layers_to_distribute % pipeline_stages_left == 0
            ), "With uneven pipelineing the left over layers must be divisible by left over stages"
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert (
            num_layers % config.pipeline_model_parallel_size == 0
        ), f"{num_layers=} should be divisible by {config.pipeline_model_parallel_size=}"
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert (
            num_layers_per_pipeline_rank % vp_size == 0
        ), f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert (
                num_layers_to_build >= 0
            ), f"Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, f"Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


@dataclass
class TransformerBlockSubmodules:
    """
    Dataclass for specifying the submodules of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (List[ModuleSpec], optional): A list of module specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (Optional[Union[ModuleSpec, torch.nn.Module]], optional): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: Optional[List[ModuleSpec]] = None
    layer_norm: LayerNormBuilder | None = None


def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(GraphableMegatronModule, MegatronModule):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        pp_rank = get_pg_rank(pp_group)

        self.submodules = _get_block_submodules(config, spec, vp_stage, pp_rank)
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
            and "core_attn" in self.config.recompute_modules
        )

        if get_cpu_offload_context is not None:
            (self.offload_context, self.group_prefetch_offload_commit_async) = (
                get_cpu_offload_context(
                    self.config.cpu_offloading,
                    self.config.cpu_offloading_num_layers,
                    self.config.num_layers,
                    self.config.cpu_offloading_activations,
                    self.config.cpu_offloading_weights,
                    self.config.cpu_offloading_double_buffering,
                    self.config.cpu_offloading_retain_pinned_cpu_buffers,
                )
            )
            self.config._cpu_offloading_context = (
                self.offload_context if self.config.cpu_offloading else None
            )
        else:
            assert (
                self.config.cpu_offloading is False
            ), "CPU Offloading is enabled when TE is not present"

            self.offload_context, self.group_prefetch_offload_commit_async = nullcontext(), None
            self.config._cpu_offloading_context = None

        self._build_layers()
        if self.config.fast_slow_reasoner:
            self.fast_slow_reasoner_core = self._build_fast_slow_reasoner()
        else:
            self.fast_slow_reasoner_core = None
        if self.config.temporal_jacobi_passes > 1:
            self.temporal_jacobi_mixer = TemporalJacobiMixer(
                hidden_size=self.config.hidden_size,
                eps=self.config.temporal_jacobi_mixer_eps,
            )
        else:
            self.temporal_jacobi_mixer = None
        if self.config.loop_latent_prediction:
            if (
                self.config.init_method is None
                or self.config.output_layer_init_method is None
            ):
                raise RuntimeError(
                    "TransformerConfig initialization methods were not initialized."
                )
            if self.config.loop_prediction_update_type == "mlp":
                self.latent_prediction_accumulator = LatentPredictionAccumulator(
                    hidden_size=self.config.hidden_size,
                    mlp_hidden_size=(
                        self.config.loop_prediction_mlp_hidden_size
                        or self.config.hidden_size
                    ),
                    init_method=self.config.init_method,
                    output_layer_init_method=self.config.output_layer_init_method,
                    direct_identity_copy=(
                        self.config.loop_latent_prediction_direct_identity_copy
                    ),
                )
            elif (
                self.config.loop_prediction_update_type
                == "latest_z_cross_attention"
            ):
                self.latent_prediction_accumulator = (
                    LatestLatentCrossAttentionPredictionAccumulator(
                        config=self.config,
                        ffn_hidden_size=(
                            self.config.loop_prediction_cross_attention_ffn_hidden_size
                            or self.config.ffn_hidden_size
                        ),
                    )
                )
            elif (
                self.config.loop_prediction_update_type
                == "latest_z_cross_attention_moe"
            ):
                from megatron.core.transformer.moe.moe_layer import MoELayer

                reference_moe = next(
                    (
                        layer.mlp
                        for layer in self.layers
                        if isinstance(layer.mlp, MoELayer)
                    ),
                    None,
                )
                if reference_moe is None:
                    raise RuntimeError(
                        "MoE PredictionBlocks require at least one physical MoE layer."
                    )
                predictor_config = copy.copy(self.config)
                predictor_config.moe_latent_size = self.config.hidden_size // 2
                predictor_config.moe_ffn_hidden_size = (
                    self.config.loop_prediction_cross_attention_ffn_hidden_size
                    or self.config.hidden_size // 4
                )
                prediction_moe_layers = []
                for prediction_block_index in range(2):
                    prediction_moe = MoELayer(
                        config=predictor_config,
                        submodules=reference_moe.submodules,
                        layer_number=self.config.num_layers + prediction_block_index + 1,
                        pg_collection=self.pg_collection,
                    )
                    prediction_moe.set_layer_number(
                        self.config.num_layers + prediction_block_index + 1
                    )
                    prediction_moe_layers.append(prediction_moe)
                self.latent_prediction_accumulator = (
                    LatestLatentCrossAttentionMoEPredictionAccumulator(
                        config=self.config,
                        prediction_moe_layers=(
                            prediction_moe_layers[0],
                            prediction_moe_layers[1],
                        ),
                    )
                )
            else:
                raise RuntimeError(
                    "Unsupported loop prediction update type: "
                    f"{self.config.loop_prediction_update_type!r}."
                )
        else:
            self.latent_prediction_accumulator = None
        if self.config.loop_state_prediction_split:
            if self.config.loop_state_prediction_split_type == "linear":
                if self.config.init_method is None:
                    raise RuntimeError("TransformerConfig.init_method was not initialized.")
                self.loop_state_prediction_splitter = DirectLinearStatePredictionSplit(
                    hidden_size=self.config.hidden_size,
                    init_method=self.config.init_method,
                    eps=self.config.loop_state_prediction_split_eps,
                )
            elif self.config.loop_state_prediction_split_type == "mlp":
                self.loop_state_prediction_splitter = DirectMLPStatePredictionSplit(
                    hidden_size=self.config.hidden_size,
                    eps=self.config.loop_state_prediction_split_eps,
                )
            elif self.config.loop_state_prediction_split_type in (
                "prediction_only_identity",
                "dual_identity",
            ):
                self.loop_state_prediction_splitter = IdentityLinearStatePredictionSplit(
                    hidden_size=self.config.hidden_size,
                    prediction_only=(
                        self.config.loop_state_prediction_split_type
                        == "prediction_only_identity"
                    ),
                )
            elif self.config.loop_state_prediction_split_type in (
                "dual_identity_per_visit",
                "dual_identity_shared_state_per_visit_prediction",
            ):
                self.loop_state_prediction_splitter = (
                    PerVisitIdentityLinearStatePredictionSplit(
                        hidden_size=self.config.hidden_size,
                        num_visits=self.config.loop_repeats,
                        share_state_projection=(
                            self.config.loop_state_prediction_split_type
                            == "dual_identity_shared_state_per_visit_prediction"
                        ),
                    )
                )
            elif (
                self.config.loop_state_prediction_split_type
                == "identity_projected_residual_mlp_shared_state_per_visit_prediction"
            ):
                if (
                    self.config.init_method is None
                    or self.config.output_layer_init_method is None
                ):
                    raise RuntimeError(
                        "TransformerConfig initialization methods were not initialized."
                    )
                self.loop_state_prediction_splitter = (
                    ProjectedResidualMLPStatePredictionSplit(
                        hidden_size=self.config.hidden_size,
                        num_visits=self.config.loop_repeats,
                        init_method=self.config.init_method,
                        output_layer_init_method=self.config.output_layer_init_method,
                    )
                )
            else:
                raise RuntimeError(
                    "Unsupported loop state/prediction split type: "
                    f"{self.config.loop_state_prediction_split_type!r}."
                )
        else:
            self.loop_state_prediction_splitter = None
        self.num_layers_per_pipeline_rank = len(self.layers)
        self.layer_execution_order = get_layer_execution_order(
            num_layers=self.num_layers_per_pipeline_rank,
            loop_layer_start=self.config.loop_layer_start,
            loop_layer_end=self.config.loop_layer_end,
            loop_repeats=self.config.loop_repeats,
        )
        self.layer_execution_visits = get_layer_execution_visits(
            self.layer_execution_order,
            loop_layer_start=self.config.loop_layer_start,
            loop_layer_end=self.config.loop_layer_end,
            loop_repeats=self.config.loop_repeats,
        )
        self.num_executed_layers_per_pipeline_rank = len(self.layer_execution_order)
        self.last_layer_execution_position = {
            layer_index: execution_position
            for execution_position, layer_index in enumerate(self.layer_execution_order)
        }
        self.active_loop_depth_indices = (
            active_loop_depth_indices(
                self.config.loop_repeats,
                self.config.loop_depth_aux_loss_weight,
                self.config.loop_depth_loss_weights,
            )
            if self.config.loop_depth_supervision
            else ()
        )

        if self.config.loop_repeats > 1:
            logger.info(
                "Weight-shared transformer loop enabled: physical_layers=%d "
                "loop=[%d,%d] repeats=%d executed_layers=%d",
                self.num_layers_per_pipeline_rank,
                self.config.loop_layer_start,
                self.config.loop_layer_end,
                self.config.loop_repeats,
                self.num_executed_layers_per_pipeline_rank,
            )
        if self.config.loop_random_depth_training:
            logger.info(
                "Uniform random loop-depth training enabled: min_repeats=1 max_repeats=%d "
                "sampling_unit=optimizer_update ntp_exits_per_update=1",
                self.config.loop_repeats,
            )
        if self.config.temporal_jacobi_passes > 1:
            active_passes = [
                index + 1
                for index, weight in enumerate(self.config.temporal_jacobi_loss_weights or [])
                if weight > 0.0
            ]
            logger.info(
                "Temporal Jacobi recurrence enabled: passes=%d active_ntp_passes=%s "
                "carrier_after_layer=%d mixer=mandatory_state_causal_shift",
                self.config.temporal_jacobi_passes,
                active_passes,
                self.config.loop_layer_end,
            )
        if self.config.loop_depth_supervision:
            logger.info(
                "Multi-exit loop NTP supervision enabled: repeats=%d active_ntp_exits=%s "
                "zero_weight_codas_skipped=true",
                self.config.loop_repeats,
                [index + 1 for index in self.active_loop_depth_indices],
            )
        if self.config.loop_latent_prediction:
            loop_width = (
                cast(int, self.config.loop_layer_end)
                - cast(int, self.config.loop_layer_start)
                + 1
            )
            update_offsets = prediction_update_offsets(
                loop_width,
                self.config.loop_prediction_updates_per_visit,
            )
            update_layers = [
                cast(int, self.config.loop_layer_start) + offset - 1
                for offset in update_offsets
            ]
            prediction_update_type = self.config.loop_prediction_update_type
            split_parameter_count = (
                0
                if self.config.loop_latent_prediction_direct_identity_copy
                else 2 * self.config.hidden_size * self.config.hidden_size
            )
            if prediction_update_type == "mlp":
                prediction_hidden_size = (
                    self.config.loop_prediction_mlp_hidden_size
                    or self.config.hidden_size
                )
                parameter_count = (
                    split_parameter_count
                    + 2 * self.config.hidden_size * prediction_hidden_size
                )
            elif prediction_update_type == "latest_z_cross_attention":
                prediction_hidden_size = (
                    self.config.loop_prediction_cross_attention_ffn_hidden_size
                    or self.config.ffn_hidden_size
                )
                parameter_count = (
                    split_parameter_count
                    + 4 * self.config.hidden_size * self.config.hidden_size
                    + 3 * self.config.hidden_size * prediction_hidden_size
                    + 3 * self.config.hidden_size
                )
            else:
                prediction_hidden_size = (
                    self.config.loop_prediction_cross_attention_ffn_hidden_size
                    or self.config.hidden_size // 4
                )
                query_groups = cast(int, self.config.num_query_groups)
                gate_projection_count = int(
                    not self.config.loop_prediction_disable_attention_output_gate
                )
                attention_parameters = (
                    (2 + gate_projection_count)
                    * self.config.hidden_size
                    * self.config.hidden_size
                    + 2
                    * self.config.hidden_size
                    * query_groups
                    * (self.config.hidden_size // self.config.num_attention_heads)
                )
                latent_parameters = self.config.hidden_size * self.config.hidden_size
                router_and_norm_parameters = (
                    cast(int, self.config.num_moe_experts) + 3
                ) * self.config.hidden_size
                expert_parameters = (
                    3
                    * cast(int, self.config.num_moe_experts)
                    * (self.config.hidden_size // 2)
                    * prediction_hidden_size
                )
                parameter_count = split_parameter_count + 2 * (
                    attention_parameters
                    + latent_parameters
                    + router_and_norm_parameters
                    + expert_parameters
                )
            logger.info(
                "Dual-stream latent/prediction accumulation enabled: "
                "loop_visits=%d prediction_updates_per_visit=%d "
                "prediction_update_layers=%s total_prediction_updates=%d "
                "prediction_update_type=%s prediction_hidden_size=%d "
                "prediction_block_sharing=%s direct_identity_copy=%s "
                "prediction_reset_each_visit=%s exit0=false params=%d",
                self.config.loop_repeats,
                self.config.loop_prediction_updates_per_visit,
                update_layers,
                self.config.loop_repeats * len(update_offsets),
                prediction_update_type,
                prediction_hidden_size,
                (
                    "two_distinct_each_shared_across_visits"
                    if prediction_update_type == "latest_z_cross_attention_moe"
                    else "one_shared_across_updates_and_visits"
                ),
                self.config.loop_latent_prediction_direct_identity_copy,
                self.config.loop_prediction_reset_each_visit,
                parameter_count,
            )
        if self.config.loop_state_prediction_split:
            split_type = self.config.loop_state_prediction_split_type
            prediction_only = split_type == "prediction_only_identity"
            projected_residual_mlp = (
                split_type
                == "identity_projected_residual_mlp_shared_state_per_visit_prediction"
            )
            per_visit_prediction = split_type in (
                "dual_identity_per_visit",
                "dual_identity_shared_state_per_visit_prediction",
                "identity_projected_residual_mlp_shared_state_per_visit_prediction",
            )
            per_visit_state = split_type == "dual_identity_per_visit"
            identity_initialized = split_type in (
                "prediction_only_identity",
                "dual_identity",
                "dual_identity_per_visit",
                "dual_identity_shared_state_per_visit_prediction",
                "identity_projected_residual_mlp_shared_state_per_visit_prediction",
            )
            if per_visit_prediction:
                state_parameter_sets = (
                    self.config.loop_repeats - 1 if per_visit_state else 1
                )
                prediction_parameter_sets = self.config.loop_repeats
            else:
                state_parameter_sets = 0 if prediction_only else 1
                prediction_parameter_sets = 1
            split_parameters = (
                (state_parameter_sets + prediction_parameter_sets)
                * self.config.hidden_size
                * self.config.hidden_size
            )
            if projected_residual_mlp:
                split_parameters += (
                    (state_parameter_sets + prediction_parameter_sets)
                    * 2
                    * self.config.hidden_size
                    * (self.config.hidden_size // 4)
                )
            logger.info(
                "Direct loop state/prediction split enabled: type=%s hidden_size=%d "
                "bottleneck_size=%s params=%d normalization=%s residual=%s "
                "activation=%s initialization=%s state_mapping=%s "
                "state_shared_across_visits=%s prediction_shared_across_visits=%s "
                "state_parameter_sets=%d prediction_parameter_sets=%d",
                split_type,
                self.config.hidden_size,
                (
                    self.config.hidden_size // 4
                    if projected_residual_mlp
                    else self.config.hidden_size // 2
                    if split_type == "mlp"
                    else "none"
                ),
                split_parameters,
                "none" if identity_initialized else "rms_fp32_stats",
                "projected_branch" if projected_residual_mlp else "false",
                "silu" if split_type == "mlp" or projected_residual_mlp else "none",
                (
                    "identity_projection_plus_model_default_residual_mlp"
                    if projected_residual_mlp
                    else "identity"
                    if identity_initialized
                    else "model_default"
                ),
                (
                    "identity_passthrough"
                    if prediction_only
                    else "identity_projection_then_residual_mlp"
                    if projected_residual_mlp
                    else "learned_projection"
                ),
                not per_visit_state,
                not per_visit_prediction,
                state_parameter_sets,
                prediction_parameter_sets,
            )

    def get_layer_execution_order(self) -> tuple[int, ...]:
        """Return the zero-based physical layer indices used by this forward pass."""
        return self.layer_execution_order

    def get_layer_execution_visits(self) -> tuple[int, ...]:
        """Return the one-based loop visit for each execution position."""
        return self.layer_execution_visits

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(global_layer_number)
            else:
                layer_config = self.config

            # Get appropriate quantization context (FP8 and FP4 are mutually exclusive)
            if layer_config.fp8:
                quantization_context = get_fp8_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            elif layer_config.fp4:
                quantization_context = get_fp4_context(
                    layer_config, global_layer_number - 1, is_init=True
                )
            else:
                quantization_context = nullcontext()

            with quantization_context:
                module = build_module(
                    layer_spec,
                    config=layer_config,
                    layer_number=layer_number,
                    pg_collection=self.pg_collection,
                    vp_stage=self.vp_stage,
                )
            return module

        # offset is implicit in TransformerLayer
        self.layers = torch.nn.ModuleList(
            [
                build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

        # @TODO: add back account_for_embedding_in_pipeline_split (see issue #293)
        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if self.has_final_layernorm_in_this_stage():
            self.final_layernorm = not_none(self.submodules.layer_norm)(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

        if self.config.inference_fuse_tp_communication:
            self._setup_fused_tp_communication()

    def _build_fast_slow_reasoner(self):
        """Build the opt-in M1 Fast--Slow core without changing ordinary layers."""

        from megatron.core.transformer.fast_slow_reasoner import FastSlowReasoner
        from megatron.core.transformer.moe.moe_layer import MoELayer

        reference_pair = next(
            (
                (layer_index, layer.mlp)
                for layer_index, layer in enumerate(self.layers)
                if isinstance(layer.mlp, MoELayer)
            ),
            None,
        )
        if reference_pair is None:
            raise RuntimeError("Fast--Slow Reasoner requires a physical MoE backbone layer.")
        if not self.submodules.layer_specs:
            raise RuntimeError("Fast--Slow Reasoner requires a TransformerLayer spec.")
        reference_layer_index, reference_moe = reference_pair

        # Main layers use virtual numbers 1..14.  The six Fast blocks and
        # shared slow updater use 15..21 so router probes and checkpoint keys
        # cannot collide with the ordinary Prelude/Coda modules.
        virtual_total_layers = (
            self.config.num_layers + self.config.fast_slow_fast_num_layers + 1
        )
        fast_config = copy.copy(self.config)
        fast_config.fast_slow_reasoner = False
        fast_config.num_layers = virtual_total_layers
        fast_config.hidden_size = self.config.fast_slow_fast_hidden_size
        fast_config.num_attention_heads = (
            self.config.fast_slow_fast_num_attention_heads
        )
        fast_config.num_query_groups = self.config.fast_slow_fast_num_attention_heads
        fast_config.kv_channels = (
            self.config.fast_slow_fast_hidden_size
            // self.config.fast_slow_fast_num_attention_heads
        )
        fast_config.moe_latent_size = self.config.fast_slow_fast_moe_latent_size
        fast_config.moe_ffn_hidden_size = (
            self.config.fast_slow_fast_moe_ffn_hidden_size
        )
        fast_config.moe_router_topk = self.config.fast_slow_fast_router_topk
        fast_window_schedule = getattr(
            self.config, "fast_slow_fast_window_schedule", None
        )
        fast_window_size = (
            max(fast_window_schedule)
            if fast_window_schedule is not None
            else self.config.fast_slow_fast_window_size
        )
        fast_config.window_size = (
            None if fast_window_size is None else (fast_window_size, 0)
        )
        # A missing skip pattern means every layer constructed from this Fast-only
        # clone uses SWA. The parent config remains full attention.
        fast_config.window_attn_skip_freq = None

        fast_layers = []
        fast_layer_spec = self.submodules.layer_specs[reference_layer_index]
        if getattr(self.config, "fast_slow_fast_attention_type", "softmax") == "kda":
            # Only this cloned Fast spec changes. KDA owns the pre-RMSNorm
            # previously fused into TE's QKV projection, exactly once.
            from megatron.core.transformer.fast_slow_kda.kda_attention import KDAAttention
            from megatron.core.transformer.identity_op import IdentityOp

            fast_layer_spec = copy.deepcopy(fast_layer_spec)
            fast_layer_spec.submodules.self_attention = ModuleSpec(
                module=KDAAttention,
                params={
                    "num_heads": self.config.fast_slow_fast_num_attention_heads,
                    "head_dim": self.config.fast_slow_kda_head_dim,
                },
            )
            fast_layer_spec.submodules.input_layernorm = IdentityOp
        for fast_index in range(self.config.fast_slow_fast_num_layers):
            fast_layer = build_module(
                fast_layer_spec,
                config=fast_config,
                layer_number=self.config.num_layers + fast_index + 1,
                pg_collection=self.pg_collection,
                vp_stage=self.vp_stage,
            )
            if not isinstance(getattr(fast_layer, "mlp", None), MoELayer):
                raise RuntimeError(
                    "Fast--Slow accounting requires every Fast block to use MoE; "
                    f"built {type(getattr(fast_layer, 'mlp', None)).__name__}."
                )
            fast_layers.append(fast_layer)

        if (getattr(self.config, "fast_slow_fast_attention_type", "softmax") == "kda"
                and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
            # This is the launch contract, not a debug log: INFO is suppressed
            # by the default training logger and cannot serve as runtime evidence.
            print(
                f"Fast KDA effective config: heads={self.config.fast_slow_fast_num_attention_heads} "
                f"head_dim={self.config.fast_slow_kda_head_dim} "
                f"qkv_width={self.config.fast_slow_fast_num_attention_heads * self.config.fast_slow_kda_head_dim} "
                f"residual_width={self.config.fast_slow_fast_hidden_size} "
                "state_reset=every_complete_sequence_call fast_rope=false outer_slow_rope=true",
                flush=True,
            )

        slow_config = copy.copy(self.config)
        slow_config.fast_slow_reasoner = False
        slow_config.num_layers = virtual_total_layers
        slow_config.window_size = None
        slow_config.window_attn_skip_freq = None
        slow_config.moe_router_topk = self.config.moe_router_topk
        slow_layer_number = virtual_total_layers
        slow_moe = MoELayer(
            config=slow_config,
            submodules=reference_moe.submodules,
            layer_number=slow_layer_number,
            pg_collection=self.pg_collection,
        )
        slow_moe.set_layer_number(slow_layer_number)

        logger.info(
            "Fast--Slow Reasoner enabled: full_layers=%d prelude=%d coda=%d "
            "fast_width=%d fast_layers=%d fast_heads=%d fast_head_dim=%d "
            "fast_topk=%d fast_window=%s fast_window_schedule=%s "
            "visits=%d slow_blocks=1 "
            "sharing=fast_stack_and_slow_across_visits "
            "reference_init_depth=%d final_loss_only=true",
            self.config.num_layers,
            self.config.fast_slow_prelude_layers,
            self.config.fast_slow_coda_layers,
            self.config.fast_slow_fast_hidden_size,
            self.config.fast_slow_fast_num_layers,
            self.config.fast_slow_fast_num_attention_heads,
            fast_config.kv_channels,
            self.config.fast_slow_fast_router_topk,
            str(fast_config.window_size),
            str(fast_window_schedule),
            self.config.fast_slow_num_visits,
            self.config.fast_slow_reference_depth,
        )
        return FastSlowReasoner(
            config=self.config,
            fast_layers=fast_layers,
            slow_moe=slow_moe,
        )

    def has_final_layernorm_in_this_stage(self):
        """
        Check if this vpp stage contains the final layernorm.

        Note:
            Final layernorm now has been moved from the post-process stage to the last decoder
            layer by using this function.
            There will be a small numeric difference because of grad norm reduction when final
            layernorm is placed in different pipeline stages in deterministic mode. It can still
            be bitwise aligned by disabling grad norm clipping.
        """
        if self.config.mtp_num_layers is None:
            # for model without MTPLayer, the final layernorm is set in the stage which does
            # post_process
            return self.submodules.layer_norm and self.post_process and self.post_layer_norm
        else:
            # for model with MTPLayer, the final layernorm is set in the stage which has the
            # last layer of the decoder
            has_final_layernorm_in_this_stage = False
            for layer in self.layers:
                if layer.layer_number == self.config.num_layers:
                    has_final_layernorm_in_this_stage = True
                    break
            return (
                self.submodules.layer_norm
                and has_final_layernorm_in_this_stage
                and self.post_layer_norm
            )

    def _setup_fused_tp_communication(self):
        """Setup fused TP communication for all layers.
        We have a fused reduce-scatter + add + layer-norm + all-gather operation.
        We call this kernel from within row parallel linear layers.
        But layer-norm needs the layer norm weights from the
        successive column parallel linear layer.
        This function is used to pass those weights to the respective layers.
        """

        for i in range(len(self.layers)):
            current_layer = self.layers[i]

            # Get next layer's QKV norm weights (None for last layer)
            if i < len(self.layers) - 1:
                next_qkv_norm_weights = self.layers[i + 1].get_qkv_layer_norm_weights()
            else:
                next_qkv_norm_weights = None

            # Configure all fused TP communication settings in one call
            current_layer.configure_fused_tp_inference(
                skip_qkv_norm_and_all_gather=(i > 0),
                fc2_next_layer_norm_weights=next_qkv_norm_weights,
            )

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        if (
            InferenceMode.is_active()
            and hasattr(self, 'cudagraph_manager')
            and kwargs['attention_mask'] is None
            and (
                kwargs.get('inference_context') is not None
                or kwargs.get('inference_params') is not None
            )
            and self.config.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()

            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            kwargs['hidden_states'] = (
                kwargs['hidden_states'].unwrap()
                if isinstance(kwargs['hidden_states'], WrappedTensor)
                else kwargs['hidden_states']
            )
            return super().__call__(*args, **kwargs)[0]
        return super().__call__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        extract_layer_indices: Optional[Set[int]] = None,
        runtime_loop_repeats: Optional[int] = None,
        runtime_loop_exit_indices: Optional[tuple[int, ...]] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.
            extract_layer_indices (Set[int], optional): A set of global
                layer indices (0-based across all pipeline stages) from
                which to extract intermediate hidden states. If
                non-empty, the forward pass will collect hidden_states
                after each specified layer.
            runtime_loop_repeats (int, optional): Training-only sampled loop
                depth for ``loop_random_depth_training``. Prelude and Coda are
                always retained and exactly one ordinary output is returned.
            runtime_loop_exit_indices (tuple[int, ...], optional): Training-only
                active exits for the two-forward loop-gradient safeguard.  The
                recurrent trunk is truncated after the deepest requested exit.
            dynamic_inference_decode_only: Optional[bool]: If true, indicates that the current
                inference context is for decode-only. This args is only used to uniquely
                identify decode and non-decode cuda graph runners in the cuda graph manager.

        Returns:
            Union[Tensor, Tuple[Tensor, List[Tensor]]]:
                - If extract_layer_indices is None or empty: Returns the output hidden states tensor
                  of shape [s, b, h].
                - If extract_layer_indices is non-empty: Returns a tuple
                  of (hidden_states, intermediate_hidden_states) where
                  intermediate_hidden_states is a list of tensors
                  corresponding to hidden states after each layer in
                  extract_layer_indices.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if self.config.loop_repeats > 1 and inference_context is not None:
            raise NotImplementedError(
                "KV-cache inference for weight-shared loop layers is not implemented. "
                "Full-sequence training and validation are supported."
            )
        if self.config.fast_slow_reasoner and inference_context is not None:
            raise NotImplementedError(
                "Fast--Slow Reasoner KV-cache inference needs an explicit recurrent "
                "fast-memory policy and is not enabled yet."
            )
        if self.config.fast_slow_reasoner:
            if attention_mask is not None:
                raise ValueError(
                    "Fast--Slow Reasoner requires fused causal attention without a "
                    "materialized attention mask."
                )
            if padding_mask is not None:
                raise NotImplementedError(
                    "Fast--Slow Reasoner does not yet support padding masks."
                )
            unsupported_inputs = {
                "context": context,
                "context_mask": context_mask,
                "attention_bias": attention_bias,
                "rotary_pos_cos": rotary_pos_cos,
                "rotary_pos_sin": rotary_pos_sin,
                "rotary_pos_cos_sin": rotary_pos_cos_sin,
                "sequence_len_offset": sequence_len_offset,
            }
            present_unsupported_inputs = [
                name for name, value in unsupported_inputs.items() if value is not None
            ]
            if present_unsupported_inputs:
                raise NotImplementedError(
                    "The first Fast--Slow Reasoner path does not support alternate "
                    "decoder context/bias/RoPE inputs: "
                    + ", ".join(present_unsupported_inputs)
                )
            if rotary_pos_emb is None:
                raise ValueError(
                    "Fast--Slow Reasoner requires the ordinary non-interleaved RoPE tensor."
                )
        selected_loop_exit_indices = self.active_loop_depth_indices
        runtime_active_loop_repeats = self.config.loop_repeats
        if runtime_loop_exit_indices is not None:
            runtime_exit_allowed = (
                self.config.loop_exit_gradient_safeguard
                or self.config.loop_dual_r1_fraction > 0.0
            )
            if not runtime_exit_allowed:
                raise ValueError(
                    "runtime_loop_exit_indices requires loop_exit_gradient_safeguard "
                    "or loop_dual_r1_fraction."
                )
            if not self.config.loop_depth_supervision or not self.training:
                raise ValueError(
                    "runtime_loop_exit_indices is training-only and requires "
                    "loop_depth_supervision."
                )
            if runtime_loop_repeats is not None:
                raise ValueError(
                    "runtime_loop_exit_indices and runtime_loop_repeats are mutually exclusive."
                )
            selected_loop_exit_indices = tuple(runtime_loop_exit_indices)
            if (
                not selected_loop_exit_indices
                or tuple(sorted(set(selected_loop_exit_indices)))
                != selected_loop_exit_indices
                or any(
                    index not in self.active_loop_depth_indices
                    for index in selected_loop_exit_indices
                )
            ):
                raise ValueError(
                    "runtime_loop_exit_indices must be a non-empty sorted subset of "
                    f"active exits {self.active_loop_depth_indices}, got "
                    f"{selected_loop_exit_indices}."
                )
            if (
                self.config.loop_dual_r1_fraction > 0.0
                and not self.config.loop_exit_gradient_safeguard
                and selected_loop_exit_indices != (0,)
            ):
                raise ValueError(
                    "The Dual R1 curriculum runtime override must select only exit (0,)."
                )
            runtime_execution_positions = get_runtime_layer_execution_positions(
                self.layer_execution_visits,
                max(selected_loop_exit_indices) + 1,
                self.config.loop_repeats,
            )
            runtime_active_loop_repeats = max(selected_loop_exit_indices) + 1
        elif runtime_loop_repeats is not None:
            if not self.config.loop_random_depth_training:
                raise ValueError(
                    "runtime_loop_repeats requires loop_random_depth_training."
                )
            if not self.training:
                raise ValueError(
                    "runtime_loop_repeats is training-only; validation must select a fixed "
                    "loop_repeats configuration."
                )
            runtime_execution_positions = get_runtime_layer_execution_positions(
                self.layer_execution_visits,
                runtime_loop_repeats,
                self.config.loop_repeats,
            )
        else:
            if self.config.loop_random_depth_training and self.training:
                raise RuntimeError(
                    "loop_random_depth_training requires one runtime_loop_repeats sample "
                    "for every training forward."
                )
            runtime_execution_positions = tuple(range(len(self.layer_execution_order)))
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager

        # Initialize feature collection (consistent with FastGen's Wan implementation)
        if extract_layer_indices is None:
            extract_layer_indices = set()
        if (
            self.config.loop_depth_supervision
            or self.config.temporal_jacobi_passes > 1
            or self.config.fast_slow_reasoner
        ) and extract_layer_indices:
            raise NotImplementedError(
                "Feature extraction is not supported with multi-exit, temporal Jacobi, "
                "or Fast--Slow execution."
            )
        if self.config.temporal_jacobi_passes > 1 and packed_seq_params is not None:
            raise NotImplementedError(
                "Temporal Jacobi recurrence does not yet support packed sequences because "
                "every packed-document boundary needs an explicit state-reset mask."
            )
        if self.config.fast_slow_reasoner and packed_seq_params is not None:
            raise NotImplementedError(
                "Fast--Slow Reasoner does not yet support packed sequences."
            )
        intermediate_hidden_states: List[Tensor] = []
        loop_exit_states: List[Tensor] = []
        loop_exit_contexts: List[Optional[Tensor]] = []
        loop_coda_states: List[Tensor] = []
        temporal_coda_states: List[Tensor] = []

        # Calculate the global layer offset for this pipeline stage
        # This is needed to convert local layer indices to global indices for feature extraction
        pp_group = self.pg_collection.pp if hasattr(self.pg_collection, 'pp') else None
        layer_offset = get_transformer_layer_offset(
            self.config, self.vp_stage, get_pg_rank(pp_group)
        )

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        # For FP4: NVFP4BlockScaling doesn't have delayed scaling, always uses inner context
        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext()
            )
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            # No quantization
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                checkpointed_result = checkpointed_forward(
                    self,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    extract_layer_indices=extract_layer_indices,
                    layer_offset=layer_offset,
                )
                # Handle return value from _checkpointed_forward
                if len(extract_layer_indices) > 0:
                    # (hidden_states, intermediate_hidden_states) tuple
                    hidden_states, intermediate_hidden_states = checkpointed_result
                else:
                    # No intermediate_hidden_states requested: just hidden_states
                    hidden_states = checkpointed_result
            else:
                def run_layer_positions(
                    current_hidden_states: Tensor,
                    current_context: Optional[Tensor],
                    execution_positions: tuple[int, ...],
                    *,
                    collect_loop_exit_indices: Optional[Set[int]] = None,
                    collect_features: bool = False,
                    feature_last_positions: Optional[dict[int, int]] = None,
                ) -> tuple[Tensor, Optional[Tensor]]:
                    """Run selected positions from the shared physical-layer schedule."""

                    for execution_position in execution_positions:
                        layer_index = self.layer_execution_order[execution_position]
                        layer = self.layers[layer_index]
                        # Get appropriate inner quantization context
                        if use_inner_quantization_context:
                            if self.config.fp8:
                                inner_quantization_context = get_fp8_context(
                                    self.config, layer.layer_number - 1
                                )
                            elif self.config.fp4:
                                inner_quantization_context = get_fp4_context(
                                    self.config, layer.layer_number - 1
                                )
                            else:
                                inner_quantization_context = nullcontext()
                        else:
                            inner_quantization_context = nullcontext()

                        layer_kwargs = dict(
                            hidden_states=current_hidden_states,
                            attention_mask=attention_mask,
                            context=current_context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                            padding_mask=padding_mask,
                        )
                        if isinstance(layer, TransformerLayer):
                            layer_kwargs["loop_visit_index"] = self.layer_execution_visits[
                                execution_position
                            ]
                        with self.offload_context, inner_quantization_context:
                            current_hidden_states, current_context = layer(**layer_kwargs)

                        loop_visit = self.layer_execution_visits[execution_position]
                        is_loop_boundary = (
                            loop_visit > 0
                            and layer_index == self.config.loop_layer_end - 1
                        )
                        is_active_exit = (
                            collect_loop_exit_indices is not None
                            and loop_visit - 1 in collect_loop_exit_indices
                        )
                        if (
                            is_loop_boundary
                            and collect_loop_exit_indices is not None
                            and self.loop_state_prediction_splitter is not None
                        ):
                            # The last visit has no state consumer.  A zero-weight
                            # exit has no prediction consumer.  Selective routing
                            # therefore skips each unused d-by-d projection.
                            needs_state = loop_visit < self.config.loop_repeats
                            if needs_state or is_active_exit:
                                if isinstance(
                                    self.loop_state_prediction_splitter,
                                    (
                                        PerVisitIdentityLinearStatePredictionSplit,
                                        ProjectedResidualMLPStatePredictionSplit,
                                    ),
                                ):
                                    state, prediction = self.loop_state_prediction_splitter(
                                        current_hidden_states,
                                        loop_visit=loop_visit,
                                        project_state=needs_state,
                                        project_prediction=is_active_exit,
                                    )
                                else:
                                    state, prediction = self.loop_state_prediction_splitter(
                                        current_hidden_states,
                                        project_state=needs_state,
                                        project_prediction=is_active_exit,
                                    )
                                if is_active_exit:
                                    if prediction is None:
                                        raise RuntimeError(
                                            "Active loop exit produced no prediction state."
                                        )
                                    loop_exit_states.append(prediction)
                                    loop_exit_contexts.append(current_context)
                                if needs_state:
                                    if state is None:
                                        raise RuntimeError(
                                            "Non-final loop visit produced no recurrent state."
                                        )
                                    current_hidden_states = state
                        elif is_loop_boundary and is_active_exit:
                            # Legacy multi-exit behavior: Coda reads the same
                            # state that recurrence consumes on the next visit.
                            loop_exit_states.append(current_hidden_states)
                            loop_exit_contexts.append(current_context)

                        if (
                            torch.is_grad_enabled()
                            and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None
                        ):
                            current_hidden_states = self.group_prefetch_offload_commit_async(
                                current_hidden_states
                            )

                        if (
                            collect_features
                            and execution_position
                            == (feature_last_positions or self.last_layer_execution_position)[
                                layer_index
                            ]
                            and (layer_index + layer_offset) in extract_layer_indices
                        ):
                            intermediate_hidden_states.append(current_hidden_states)

                    return current_hidden_states, current_context

                if self.config.fast_slow_reasoner:
                    prelude_end = self.config.fast_slow_prelude_layers
                    prelude_positions = tuple(range(prelude_end))
                    coda_positions = tuple(range(prelude_end, self.config.num_layers))
                    if not prelude_positions or not coda_positions:
                        raise RuntimeError(
                            "Fast--Slow Reasoner requires non-empty Prelude and Coda."
                        )
                    hidden_states, context = run_layer_positions(
                        hidden_states,
                        context,
                        prelude_positions,
                    )
                    reasoner = cast(torch.nn.Module, self.fast_slow_reasoner_core)
                    hidden_states = reasoner(
                        prelude_state=hidden_states,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        inference_context=inference_context,
                        packed_seq_params=packed_seq_params,
                    )
                    hidden_states, context = run_layer_positions(
                        hidden_states,
                        context,
                        coda_positions,
                    )
                elif self.config.temporal_jacobi_passes > 1:
                    coda_start_index = cast(int, self.config.loop_layer_end)
                    trunk_positions = tuple(
                        execution_position
                        for execution_position, layer_index in enumerate(
                            self.layer_execution_order
                        )
                        if layer_index < coda_start_index
                    )
                    coda_positions = tuple(
                        execution_position
                        for execution_position, layer_index in enumerate(
                            self.layer_execution_order
                        )
                        if layer_index >= coda_start_index
                    )
                    if not trunk_positions or not coda_positions:
                        raise RuntimeError(
                            "Temporal Jacobi execution requires non-empty pre-coda trunk and coda."
                        )

                    token_embeddings = hidden_states
                    initial_context = context
                    previous_carrier: Optional[Tensor] = None
                    active_loss_weights = cast(
                        list[float], self.config.temporal_jacobi_loss_weights
                    )
                    mixer = cast(TemporalJacobiMixer, self.temporal_jacobi_mixer)

                    for temporal_pass_index in range(self.config.temporal_jacobi_passes):
                        if temporal_pass_index == 0:
                            pass_input = token_embeddings
                        else:
                            pass_input = mixer(
                                token_embeddings,
                                cast(Tensor, previous_carrier),
                            )

                        # Every pass evolves the shared prelude/loop trunk and
                        # exposes its pre-coda output as the next-pass carrier.
                        previous_carrier, pass_context = run_layer_positions(
                            pass_input,
                            initial_context,
                            trunk_positions,
                        )

                        # A zero-weight pass need not execute the coda/head.  Its
                        # carrier remains fully differentiable through later
                        # passes, so final-only training is exactly preserved.
                        if active_loss_weights[temporal_pass_index] > 0.0:
                            coda_hidden_states, coda_context = run_layer_positions(
                                previous_carrier,
                                pass_context,
                                coda_positions,
                            )
                            temporal_coda_states.append(coda_hidden_states)
                            if temporal_pass_index == self.config.temporal_jacobi_passes - 1:
                                context = coda_context

                    if not temporal_coda_states:
                        raise RuntimeError("Temporal Jacobi execution produced no active NTP exit.")
                    hidden_states = temporal_coda_states[-1]
                elif self.config.loop_depth_supervision:
                    coda_start_index = cast(int, self.config.loop_layer_end)
                    trunk_positions = tuple(
                        execution_position
                        for execution_position in runtime_execution_positions
                        if self.layer_execution_order[execution_position] < coda_start_index
                    )
                    coda_positions = tuple(
                        execution_position
                        for execution_position in runtime_execution_positions
                        if self.layer_execution_order[execution_position] >= coda_start_index
                    )
                    if not trunk_positions or not coda_positions:
                        raise RuntimeError(
                            "Multi-exit loop supervision requires non-empty recurrent trunk and Coda."
                        )

                    if self.config.loop_latent_prediction:
                        loop_start_index = cast(int, self.config.loop_layer_start) - 1
                        prelude_positions = tuple(
                            execution_position
                            for execution_position in trunk_positions
                            if self.layer_execution_order[execution_position]
                            < loop_start_index
                        )
                        if not prelude_positions:
                            raise RuntimeError(
                                "Dual-stream latent/prediction execution requires a "
                                "non-empty Prelude."
                            )

                        prelude_state, context = run_layer_positions(
                            hidden_states,
                            context,
                            prelude_positions,
                        )
                        accumulator = cast(
                            Union[
                                LatentPredictionAccumulator,
                                LatestLatentCrossAttentionMoEPredictionAccumulator,
                                LatestLatentCrossAttentionPredictionAccumulator,
                            ],
                            self.latent_prediction_accumulator,
                        )
                        latent_state, prediction_initial_state = accumulator.initialize(
                            prelude_state
                        )
                        prediction_state = prediction_initial_state
                        loop_width = (
                            cast(int, self.config.loop_layer_end)
                            - cast(int, self.config.loop_layer_start)
                            + 1
                        )
                        update_offsets = prediction_update_offsets(
                            loop_width,
                            self.config.loop_prediction_updates_per_visit,
                        )
                        active_exit_set = set(selected_loop_exit_indices)

                        for loop_visit in range(1, runtime_active_loop_repeats + 1):
                            if self.config.loop_prediction_reset_each_visit:
                                prediction_state = prediction_initial_state
                            visit_positions = tuple(
                                execution_position
                                for execution_position in trunk_positions
                                if self.layer_execution_visits[execution_position]
                                == loop_visit
                            )
                            if not visit_positions:
                                raise RuntimeError(
                                    "Dual-stream latent recurrence found no physical "
                                    f"loop layers for visit {loop_visit}."
                                )
                            if len(visit_positions) != loop_width:
                                raise RuntimeError(
                                    "Dual-stream latent recurrence expected "
                                    f"{loop_width} physical loop layers in visit "
                                    f"{loop_visit}, got {len(visit_positions)}."
                                )

                            segment_start = 0
                            for prediction_block_index, update_offset in enumerate(
                                update_offsets
                            ):
                                segment_positions = visit_positions[
                                    segment_start:update_offset
                                ]
                                latent_state, context = run_layer_positions(
                                    latent_state,
                                    context,
                                    segment_positions,
                                )
                                if isinstance(
                                    accumulator,
                                    LatestLatentCrossAttentionMoEPredictionAccumulator,
                                ):
                                    prediction_state = accumulator.update_prediction(
                                        latent_state,
                                        prediction_state,
                                        prediction_block_index=prediction_block_index,
                                        rotary_pos_emb=rotary_pos_emb,
                                        attention_mask=attention_mask,
                                        inference_context=inference_context,
                                        packed_seq_params=packed_seq_params,
                                    )
                                elif isinstance(
                                    accumulator,
                                    LatestLatentCrossAttentionPredictionAccumulator,
                                ):
                                    prediction_state = accumulator.update_prediction(
                                        latent_state,
                                        prediction_state,
                                        rotary_pos_emb=rotary_pos_emb,
                                        attention_mask=attention_mask,
                                        inference_context=inference_context,
                                        packed_seq_params=packed_seq_params,
                                    )
                                else:
                                    prediction_state = accumulator.update_prediction(
                                        latent_state,
                                        prediction_state,
                                    )
                                segment_start = update_offset

                            if segment_start != len(visit_positions):
                                raise RuntimeError(
                                    "Final latent-to-prediction update did not occur at "
                                    f"the loop boundary for visit {loop_visit}."
                                )
                            if loop_visit - 1 in active_exit_set:
                                loop_exit_states.append(prediction_state)
                                loop_exit_contexts.append(context)
                        hidden_states = latent_state
                    else:
                        hidden_states, context = run_layer_positions(
                            hidden_states,
                            context,
                            trunk_positions,
                            collect_loop_exit_indices=set(selected_loop_exit_indices),
                        )
                    if len(loop_exit_states) != len(selected_loop_exit_indices):
                        raise RuntimeError(
                            "Expected one saved state per active loop exit, got "
                            f"{len(loop_exit_states)} for active depths "
                            f"{tuple(index + 1 for index in selected_loop_exit_indices)}."
                        )

                    # Coda is a readout branch.  It never feeds its result back
                    # into the loop recurrence, and a zero-weight visit has no
                    # branch at all.
                    for loop_state, loop_context in zip(
                        loop_exit_states, loop_exit_contexts
                    ):
                        branch_hidden_states, branch_context = run_layer_positions(
                            loop_state,
                            loop_context,
                            coda_positions,
                        )
                        loop_coda_states.append(branch_hidden_states)
                        context = branch_context
                    if not loop_coda_states:
                        raise RuntimeError("Multi-exit loop execution produced no active NTP exit.")
                    hidden_states = loop_coda_states[-1]
                else:
                    runtime_last_layer_execution_position = {
                        self.layer_execution_order[execution_position]: execution_position
                        for execution_position in runtime_execution_positions
                    }
                    hidden_states, context = run_layer_positions(
                        hidden_states,
                        context,
                        runtime_execution_positions,
                        collect_features=True,
                        feature_last_positions=runtime_last_layer_execution_position,
                    )

        # Final layer norm.
        if self.final_layernorm is not None:
            active_coda_states = temporal_coda_states or loop_coda_states
            if active_coda_states:
                for index, pass_hidden_states in enumerate(active_coda_states):
                    pass_hidden_states = apply_module(self.final_layernorm)(
                        cast(Tensor, pass_hidden_states)
                    )
                    active_coda_states[index] = make_viewless_tensor(
                        inp=pass_hidden_states, requires_grad=True, keep_graph=True
                    )
                hidden_states = active_coda_states[-1]
            else:
                hidden_states = apply_module(self.final_layernorm)(cast(Tensor, hidden_states))
                # TENorm produces a "viewed" tensor. This will result in schedule.py's
                # deallocate_output_tensor() throwing an error, so a viewless tensor is
                # created to prevent this.
                hidden_states = make_viewless_tensor(
                    inp=hidden_states, requires_grad=True, keep_graph=True
                )

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        if len(extract_layer_indices) > 0:
            return hidden_states, intermediate_hidden_states

        if self.config.loop_depth_supervision:
            return loop_coda_states

        if self.config.temporal_jacobi_passes > 1:
            return temporal_coda_states

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        # TODO: remove multiple non_homogeneous_layers=True assignments
        #  once non_homogeneous_layers=False support is dropped
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if self.config.hetereogenous_dist_checkpoint:
            non_homogeneous_layers = True

        if isinstance(self.config.moe_layer_freq, int):
            if self.config.moe_layer_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.moe_layer_freq, list):
            non_homogeneous_layers = True

        if isinstance(self.config.linear_attention_freq, int):
            if self.config.linear_attention_freq > 1:
                non_homogeneous_layers = True
        elif isinstance(self.config.linear_attention_freq, list):
            non_homogeneous_layers = True

        if self.config.heterogeneous_block_specs:
            non_homogeneous_layers = True

        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        if singleton_local_shards:
            if metadata is not None and metadata.get('non_homogeneous_layers') is False:
                # non_homogeneous_layers=False was set explicitly - emit an override warning
                logger.warning(
                    'non_homogeneous_layers=False is deprecated.'
                    ' Setting non_homogeneous_layers=True.'
                )
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        f'{prefix}{name}.',
                        sharded_offsets,
                        metadata,
                        tp_group=self.tp_group,
                    )
                )

        return sharded_state_dict
