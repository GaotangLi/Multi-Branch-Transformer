# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Fixed-batch, chunked-prefill and cached-decode runner (not a web server)."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from .attention import PagedAttentionBackend, PagedKVCache
from .config import ParallelContext, WideConfig
from .export import load_export
from .model import (WideForCausalLM, validate_elementwise_backend, validate_router_backend,
                    validate_greedy_head_backend, validate_qk_backend)


class AttentionHandle:
    def __init__(self):
        self.backend = None

    def __call__(self, *args):
        if self.backend is None:
            raise RuntimeError('KV cache must be allocated before forward')
        return self.backend(*args)


class ServingRunner:
    """One rank owns its local layers and KV. No mutable global batch context."""
    _sampling = None  # greedy unless set_sampling() configures stochastic sampling
    def __init__(self, export_dir, *, parallel=None, branch_execution='serial',
                 moe_backend='auto', elementwise_backend='torch', router_backend='original',
                 greedy_head_backend='replicated', qk_backend='separate', device='cuda', verify_hashes=True):
        self.elementwise_backend = validate_elementwise_backend(elementwise_backend)
        self.router_backend = validate_router_backend(router_backend)
        self.greedy_head_backend = validate_greedy_head_backend(greedy_head_backend)
        self.qk_backend = validate_qk_backend(qk_backend)
        if self.qk_backend == 'packed' and self.elementwise_backend != 'triton':
            raise ValueError('packed Q/K requires elementwise_backend=triton')
        self.device = torch.device(device)
        self.parallel = parallel or ParallelContext()
        config = WideConfig.from_dict(json.loads((Path(export_dir)/'config.json').read_text()))
        self.config = replace(config, branch_execution=branch_execution, moe_backend=moe_backend)
        self.handle = AttentionHandle()
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(self.device):
                self.model = WideForCausalLM(self.config, self.parallel, self.handle,
                                            elementwise_backend=self.elementwise_backend,
                                            router_backend=self.router_backend,
                                            qk_backend=self.qk_backend,
                                            greedy_head_backend=self.greedy_head_backend).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        self.load_report = load_export(self.model, export_dir, verify_hashes=verify_hashes)
        self.cache = None
        self.graph = None
        self.prefill_graph = None
        self._graph_router_signature = None
        self._graph_head_backend = None
        self._graph_execution_signature = None
        self._warmup_stream = None
        self._decode_ready = False

    def set_sampling(self, kind='greedy', *, temperature=1.0, top_p=1.0, top_k=0, seed=0):
        """Call before allocate(). With sampling, the decode step (and its CUDA graph) ends at the logits and
        FlashInfer's deterministic kernels draw each token from a dedicated generator; reseeding identically
        per request makes every branch mode draw identical tokens from bitwise-identical logits."""
        self.release_cache()
        self._sampling = None
        if kind == 'sample':
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            self._sampling = dict(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k),
                                  generator=generator)

    def reseed(self, seed):
        if self._sampling is not None:
            self._sampling['generator'].manual_seed(int(seed))

    def _next_tokens(self, logits):
        if self._sampling is None:
            return logits.argmax(-1)
        import flashinfer.sampling as sampling
        config = self._sampling
        values = logits.float()
        if config['temperature'] != 1.0:
            values = values / config['temperature']
        if config['top_k'] > 0:
            ids = sampling.top_k_top_p_sampling_from_logits(values, config['top_k'], config['top_p'],
                                                            deterministic=True, generator=config['generator'])
        else:
            ids = sampling.top_p_sampling_from_probs(torch.softmax(values, -1), config['top_p'],
                                                     deterministic=True, generator=config['generator'])
        return ids.to(torch.long)

    def allocate(self, batch_size, capacity_tokens, *, attention='fa4', decode_splits=0, page_size=256):
        if page_size not in (128, 256):
            raise ValueError('page_size must be 128 or 256 for the bounded cache ablation')
        self.release_cache()
        self.cache = PagedKVCache(self.model.cache_specs(),batch_size,capacity_tokens,device=self.device,
                                 block_size=page_size)
        self.handle.backend = PagedAttentionBackend(self.cache,implementation=attention,decode_splits=decode_splits)
        return self.cache.nbytes

    def release_cache(self):
        if self.device.type == 'cuda':
            # Finish replay/branch-stream work before releasing graph-owned
            # NCCL resources. Merely dropping the Python reference can leave
            # communicator teardown waiting on graph-retained allocations.
            torch.cuda.synchronize(self.device)
        if self.graph is not None:
            self.graph.reset()
        if getattr(self, 'prefill_graph', None) is not None:
            self.prefill_graph.reset()
        self.graph = None
        self._clear_prefill_buffers()
        self._graph_router_signature = None
        self._graph_head_backend = None
        self._graph_execution_signature = None
        self._decode_ready = False
        self.decode_context = None
        self.decode_positions = None
        self.decode_input = None
        self.decode_output = None
        self.decode_logits = None
        self.cache = None
        self.handle.backend = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _clear_prefill_buffers(self):
        # Both graphs must already be reset before these graph-owned tensors die.
        self.prefill_graph = None
        self.prefill_input = None
        self.prefill_positions = None
        self.prefill_context = None
        self.prefill_logits = None
        self.prefill_first = None
        self._prefill_signature = None
        self._prefill_chunk_size = None

    def _execution_signature(self):
        return (self.model.router_cache_signature(), self.model.elementwise_backend,
                self.model.greedy_head_backend, self.model.qk_signature(),
                tuple(layer.moe.config.moe_backend for layer in self.model.layers.values()),
                self.model.config.branch_execution, id(self.handle.backend),
                self.handle.backend.implementation, self.handle.backend.decode_splits, self._sampling is None)

    def _get_warmup_stream(self):
        """Reuse one stream across captures to bound per-stream cuBLAS state.

        release_cache keeps it alive; close drops it only after graph teardown.
        Never clear global cuBLAS workspaces while any live graph can use them.
        The existing current/warmup dependencies are still set at each capture.
        """
        if self.device.type != 'cuda':
            raise RuntimeError('CUDA warmup stream requires a CUDA runner')
        if getattr(self, '_warmup_stream', None) is None:
            self._warmup_stream = torch.cuda.Stream(device=self.device)
        return self._warmup_stream

    def _validate_prefill_graph_tokens(self, tokens, chunk_size):
        if self.cache is None:
            raise RuntimeError('allocate KV cache before preparing prefill')
        if tokens.ndim != 2 or tokens.shape[0] != self.cache.batch_size:
            raise ValueError('tokens must be [allocated batch, prompt length]')
        length = tokens.shape[1]
        if not 0 < length <= min(8192, chunk_size):
            raise ValueError('captured prefill requires a nonempty full prompt <=8192 and <=chunk_size')
        if length >= self.cache.capacity_tokens:
            raise ValueError('reserve cache space for generated tokens')
        if tokens.device != self.cache.page_table.device or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError('prefill IDs must be integer tensors on the runner device')

    def _finish_prefill(self, first, prompt_length):
        # Reusing a prepared decode graph is safe only after resetting every live
        # field. Old suffix KV is masked by lengths and overwritten during decode.
        if getattr(self, 'decode_input', None) is not None:
            self.reset_decode(first, prompt_length)
        else:
            self._decode_ready = False

    def _prefill_step(self):
        hidden = self.model(self.prefill_input.flatten(), self.prefill_positions, self.prefill_context)
        last = hidden.view(*self.prefill_input.shape, self.config.hidden_size)[:, -1]
        self.prefill_logits = self.model.compute_logits(last)
        self.prefill_first = self.prefill_logits.argmax(-1)

    @torch.inference_mode()
    def prepare_prefill_graph(self, tokens, *, chunk_size=8192):
        """Capture one fixed full-prompt shape; no chunked/long-context fallback.

        Call on every model rank in identical order, outside timed work. Captured
        outputs remain graph-owned and callers must clone retained observations.
        Decode has separate buffers/graph and is not destroyed by this operation.
        """
        self._validate_prefill_graph_tokens(tokens, chunk_size)
        self.model.validate_router_caches()
        if self.device.type != 'cuda':
            raise RuntimeError('prefill graph capture requires CUDA')
        torch.cuda.synchronize(self.device)
        if self.prefill_graph is not None:
            self.prefill_graph.reset()
        self._clear_prefill_buffers()
        self._decode_ready = False
        self.prefill_input = tokens.clone(memory_format=torch.contiguous_format)
        self.prefill_positions, self.prefill_context = self.cache.context(0, tokens.shape[1], prefill=True)
        self._prefill_chunk_size = chunk_size
        stream = self._get_warmup_stream()
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._prefill_step()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph):
            self._prefill_step()
        self._prefill_signature = self._execution_signature()

    @torch.inference_mode()
    def replay_prefill(self, tokens):
        """Copy new device IDs and replay; the copy is part of measured TTFT.

        All layers overwrite prompt KV before reading it. Cache zeroing is not
        required, and prompt context tensors are distinct from mutable decode
        tensors. Returned tensors alias static graph outputs across replays.
        """
        if self.prefill_graph is None:
            raise RuntimeError('prepare_prefill_graph must precede replay')
        self._validate_prefill_graph_tokens(tokens, self._prefill_chunk_size)
        if tokens.shape != self.prefill_input.shape or tokens.dtype != self.prefill_input.dtype:
            raise ValueError('prefill graph replay requires the captured shape and ID dtype')
        self.model.validate_router_caches()
        if self._prefill_signature != self._execution_signature():
            raise RuntimeError('prefill runtime/cache changed after capture; recapture the CUDA graph')
        self.prefill_input.copy_(tokens, non_blocking=True)
        self.prefill_graph.replay()
        self._finish_prefill(self.prefill_first, tokens.shape[1])
        return self.prefill_logits, self.prefill_first

    def close(self):
        """Collective lifecycle endpoint: graph reset, then branch subgroups.

        Every rank must call this after the final workload and before destroying
        WORLD. release_cache() alone intentionally keeps model groups reusable.
        """
        self.release_cache()
        if self.model is not None:
            self.model.close_parallel_groups()
            self.model = None
        self._warmup_stream = None

    @torch.inference_mode()
    def prefill(self, tokens, *, chunk_size=8192):
        self.model.validate_router_caches()
        if tokens.ndim != 2 or tokens.shape[0] != self.cache.batch_size:
            raise ValueError('tokens must be [batch, prompt length]')
        if tokens.shape[1] >= self.cache.capacity_tokens:
            raise ValueError('reserve cache space for generated tokens')
        self._decode_ready = False
        b,length = tokens.shape
        for start in range(0,length,chunk_size):
            chunk=tokens[:,start:start+chunk_size].contiguous()
            positions,context=self.cache.context(start,chunk.shape[1],prefill=True)
            hidden=self.model(chunk.flatten(),positions,context)
        last=hidden.view(b,-1,self.config.hidden_size)[:,-1]
        logits=self.model.compute_logits(last)
        first=self._next_tokens(logits)
        self._finish_prefill(first,length)
        return logits,first

    def _decode_step(self):
        hidden=self.model(self.decode_input,self.decode_positions,self.decode_context)
        if self._sampling is None:
            # Default keeps full replicated logits; opt-in exchanges only per-rank
            # maxima/IDs over the outer group. Both return identical-rank tokens.
            self.decode_output=self.model.compute_greedy(hidden)
            self.decode_input.copy_(self.decode_output)
        else:
            self.decode_logits=self.model.compute_logits(hidden)  # decode() draws the token after the step
        self.decode_positions.add_(1)
        self.decode_context.seqused_k.add_(1)
        self.decode_context.slot_mapping.add_(1)

    @torch.inference_mode()
    def prepare_decode(self, first_token, prompt_length, *, use_graph=True):
        self.model.validate_router_caches()
        if self.graph is not None:
            torch.cuda.synchronize(self.device)
            self.graph.reset()
        self.graph=None
        self._graph_router_signature=None
        self._graph_head_backend=None
        self._graph_execution_signature=None
        self.decode_cursor=prompt_length
        self.decode_positions,self.decode_context=self.cache.context(prompt_length,1,prefill=False)
        self.decode_input=first_token.clone()
        self._decode_ready=True
        if not use_graph:
            return
        # Warm all kernels/collectives and initialize branch streams before capture.
        stream=self._get_warmup_stream()
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._decode_step()
                self.reset_decode(first_token,prompt_length)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._decode_step()
        self._graph_router_signature=self.model.router_cache_signature()
        self._graph_head_backend=self.model.greedy_head_backend
        self._graph_execution_signature=self._execution_signature()
        self.reset_decode(first_token,prompt_length)

    @torch.inference_mode()
    def reset_decode(self, first_token, prompt_length):
        if not 0 <= prompt_length < self.cache.capacity_tokens:
            raise ValueError('decode reset exceeds cache capacity')
        self.decode_cursor=prompt_length
        self.decode_input.copy_(first_token)
        self.decode_positions.fill_(prompt_length)
        self.decode_context.seqused_k.fill_(prompt_length+1)
        offsets=torch.arange(self.cache.batch_size,device=self.device,dtype=torch.long)
        self.decode_context.slot_mapping.copy_(offsets*(self.cache.pages_per_seq*self.cache.block_size)+prompt_length)
        self._decode_ready=True

    @torch.inference_mode()
    def decode(self, steps):
        self.model.validate_router_caches()
        if self.graph is not None and self._graph_router_signature != self.model.router_cache_signature():
            raise RuntimeError('router weights/backend/cache changed after capture; recapture the CUDA graph')
        if self.graph is not None and getattr(self, '_graph_head_backend', 'replicated') != self.model.greedy_head_backend:
            raise RuntimeError('greedy head backend changed after capture; recapture the CUDA graph')
        if (self.graph is not None and getattr(self, '_graph_execution_signature', None) is not None
                and self._graph_execution_signature != self._execution_signature()):
            raise RuntimeError('execution/QK backend changed after capture; recapture the CUDA graph')
        if not getattr(self, '_decode_ready', True):
            raise RuntimeError('prepare/reset decode after prefill graph preparation before decoding')
        if steps < 1 or self.decode_cursor+steps > self.cache.capacity_tokens:
            raise ValueError('decode exceeds allocated KV capacity')
        for _ in range(steps):
            if self.graph is None:
                self._decode_step()
            else:
                self.graph.replay()
            if self._sampling is not None:
                self.decode_output=self._next_tokens(self.decode_logits)
                self.decode_input.copy_(self.decode_output)
        self.decode_cursor+=steps
        return self.decode_output
