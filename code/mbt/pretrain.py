"""MBT-only next-token pretraining through the native Megatron training loop.

Model construction and lifecycle callbacks are passed explicitly. This module
does not import another experiment's entry point or modify global functions.
"""
from __future__ import annotations

from functools import partial
import sys
import time

import torch

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import RerunDataIterator, get_rerun_state_machine
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, get_timers, print_rank_0, set_startup_timestamps
from megatron.training import training
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.utils import get_batch_on_this_tp_rank, get_blend_and_blend_per_split

from .config import ModelConfig
from .model import build_model
from .runtime import TrainingLifecycle
from .training_config import require


def loss_function(loss_mask, per_token_losses):
    """One final next-token CE, using native token-weighted DP reporting."""
    losses = per_token_losses.reshape(-1).float()
    mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * mask)
    num_tokens = mask.sum().detach().to(torch.int)
    args = get_args()
    if args.check_for_nan_in_loss_and_grad:
        state = get_rerun_state_machine()
        state.validate_result(result=loss, rejection_func=lambda value: ~torch.isfinite(value),
                              message="Non-finite MBT next-token loss", tolerance=0.0, fatal=True)
    report = {"lm loss": torch.cat((loss.detach().view(1), num_tokens.view(1)))}
    return loss, num_tokens, report


def forward_step(data_iterator, model, return_schedule_plan=False):
    """Full-sequence causal language-model training/evaluation, TP=PP=CP=1."""
    require(not return_schedule_plan, "MBT does not expose a pipeline schedule plan")
    timers = get_timers()
    timers("batch-generator", log_level=2).start()
    batch = get_batch_on_this_tp_rank(data_iterator)
    timers("batch-generator").stop()
    losses = model(batch["tokens"], batch["position_ids"], batch["attention_mask"],
                   labels=batch["labels"], loss_mask=batch["loss_mask"])
    return losses, partial(loss_function, batch["loss_mask"])


def dataset_config(args):
    """Ordered indexed-dataset blend and ordinary causal next-token samples."""
    blend, blend_per_split = get_blend_and_blend_per_split(args)
    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=build_tokenizer(args),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        context_parallel_size=1,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=0,
        hybrid_context_parallel=False,
    )


def train_valid_test_datasets_provider(sample_counts, vp_stage=None):
    require(vp_stage is None, "Virtual pipeline stages are unsupported")
    args = get_args()
    dataset_type = MockGPTDataset if args.mock_data else GPTDataset
    print_rank_0("> building MBT train/validation datasets")
    return BlendedMegatronDatasetBuilder(
        dataset_type, sample_counts,
        lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
        dataset_config(args),
    ).build()


# The core dataset builder coordinates rank-local construction itself.
train_valid_test_datasets_provider.is_distributed = True


def make_model_provider(model_config, backend_root):
    def provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
        require(vp_stage is None, "Virtual pipeline stages are unsupported")
        args = get_args()
        native_config = config if config is not None else core_transformer_config_from_args(args)
        return build_model(
            model_config, native_config=native_config,
            max_sequence_length=args.max_position_embeddings,
            vocab_size=args.padded_vocab_size,
            pre_process=pre_process, post_process=post_process,
            parallel_output=True, pg_collection=pg_collection,
            backend_root=backend_root,
        )
    return provider


def run(manifest, *, lifecycle_factory=None):
    """Run one explicit model/data/optimizer contract in a torchrun worker."""
    started = time.time()
    previous_argv = sys.argv
    sys.argv = ["mbt.pretrain", *manifest["native_arguments"]]
    try:
        args = parse_and_validate_args(args_defaults={"tokenizer_type": "TikTokenizer"})
        require(args.tensor_model_parallel_size == args.pipeline_model_parallel_size ==
                args.context_parallel_size == args.expert_tensor_parallel_size == 1,
                "MBT pretraining supports TP=PP=CP=expert-TP=1")
        require(args.expert_model_parallel_size in (1, 2, 4, 8), "Unsupported expert-parallel layout")
        require(not args.reset_attention_mask and not args.reset_position_ids,
                "The released recipes use full-sequence causal samples")
        set_startup_timestamps(program_start=started, main_entry=started)
        model_config = ModelConfig.from_dict(manifest["recipe"]["model"])
        lifecycle = (lifecycle_factory or TrainingLifecycle)(manifest, training, RerunDataIterator)
        training.pretrain(
            pretrain_cfg_container_from_args(args),
            train_valid_test_datasets_provider,
            make_model_provider(model_config, manifest["backend_root"]),
            ModelType.encoder_or_decoder,
            forward_step,
            lifecycle=lifecycle,
        )
    finally:
        sys.argv = previous_argv
