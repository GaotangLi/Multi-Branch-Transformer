# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Cached single-GPU inference for v2 Standard/Wide–Shallow/Narrow–Deep/Hybrid.

The proven paged-cache, prefill and CUDA-graph lifecycle methods are shared with
wide_serving. Only strict architecture loading differs. This is a fixed-batch
research interface, not Hugging Face AutoModel or a production HTTP server.
"""

from dataclasses import replace
import json
from pathlib import Path

import torch

from ..wide_serving.config import ParallelContext
from ..wide_serving.model import (validate_elementwise_backend, validate_router_backend,
                                  validate_greedy_head_backend, validate_qk_backend)
from ..wide_serving.runner import AttentionHandle, ServingRunner
from .config import MBTConfig
from .model import MBTForCausalLM


class MBTServingRunner(ServingRunner):
    def __init__(self, export_dir, *, parallel=None, branch_execution="serial", moe_backend="auto",
                 elementwise_backend="torch", router_backend="original",
                 greedy_head_backend="replicated", qk_backend="separate", device="cuda",
                 verify_hashes=True):
        self.parallel = parallel or ParallelContext()
        if self.parallel.world_size != 1:
            raise ValueError("generic MBT v2 runner supports one GPU only")
        self.elementwise_backend = validate_elementwise_backend(elementwise_backend)
        self.router_backend = validate_router_backend(router_backend)
        self.greedy_head_backend = validate_greedy_head_backend(greedy_head_backend)
        self.qk_backend = validate_qk_backend(qk_backend)
        if self.qk_backend == "packed" and self.elementwise_backend != "triton":
            raise ValueError("packed Q/K requires elementwise_backend=triton")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("paged v2 runner requires CUDA; CPU tests use MBTForCausalLM with an explicit oracle")
        config = MBTConfig.from_dict(json.loads((Path(export_dir) / "config.json").read_text()))
        if config.dtype != "bfloat16":
            raise ValueError("paged GPU runtime requires a bfloat16 export")
        self.config = replace(config, branch_execution=branch_execution, moe_backend=moe_backend)
        self.handle = AttentionHandle()
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(self.device):
                self.model = MBTForCausalLM(self.config, self.parallel, self.handle,
                                            elementwise_backend=self.elementwise_backend,
                                            router_backend=self.router_backend,
                                            greedy_head_backend=self.greedy_head_backend,
                                            qk_backend=self.qk_backend).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        from .export import load_export
        self.load_report = load_export(self.model, export_dir, verify_hashes=verify_hashes)
        self.cache = None
        self.graph = None
        self.prefill_graph = None
        self._graph_router_signature = None
        self._graph_head_backend = None
        self._graph_execution_signature = None
        self._warmup_stream = None
        self._decode_ready = False

    @torch.inference_mode()
    def generate_ids(self, tokens, max_new_tokens, *, use_graph=True, chunk_size=8192,
                     attention="fa4", eos_token_id=None):
        """Return [batch,generated_length<=max_new_tokens] IDs using KV decode.

        Input cohorts must have equal unpadded lengths. For an EOS-aware demo,
        terminate only when all clients have reached EOS and pad finished rows
        with EOS; the lower-level benchmark methods always use fixed lengths.
        """
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
            raise ValueError("tokens must be a nonempty equal-length [batch,prompt] tensor")
        if tokens.device != self.model.embedding.weight.device:
            raise ValueError("tokens must already be on the runner CUDA device")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("tokens must have integer dtype")
        if tokens.shape[1] + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError("generation exceeds trained context; use explicitly labeled low-level systems benchmarks instead")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")
        self.allocate(tokens.shape[0], tokens.shape[1] + max_new_tokens + 4, attention=attention)
        _, first = self.prefill(tokens, chunk_size=chunk_size)
        outputs = [first.clone()]
        finished = torch.zeros_like(first, dtype=torch.bool)
        if eos_token_id is not None:
            finished = first == eos_token_id
        if max_new_tokens > 1:
            self.prepare_decode(first, tokens.shape[1], use_graph=use_graph)
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and bool(finished.all()):
                break
            value = self.decode(1).clone()
            if eos_token_id is not None:
                value = torch.where(finished, eos_token_id, value)
                finished |= value == eos_token_id
            outputs.append(value)
        return torch.stack(outputs, dim=1)
