# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Single-GPU fixed-slot adapter; existing wide_serving sources stay unchanged."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import math
import time

import torch

from .protocol import RoundResult, interval_summary


class FixedCohortBackend:
    def __init__(self, runner, prompt_bank, *, output_length, chunk_size, use_graph=True,
                 page_size=256, decode_splits=0, sampling_seed=0):
        if prompt_bank.ndim != 3 or min(prompt_bank.shape) < 1:
            raise ValueError('prompt bank must be [rounds, clients, prompt length]')
        if output_length < 2 or chunk_size < 1:
            raise ValueError('output length >=2 and positive chunk size required')
        if runner.parallel.world_size != 1:
            raise ValueError('this comparison requires one GPU, TP/EP/PP=1')
        self.runner, self.prompt_bank = runner, prompt_bank
        self.output_length, self.chunk_size = output_length, chunk_size
        self.use_graph, self.page_size, self.decode_splits = use_graph, page_size, decode_splits
        # With sampling, every request is reseeded from its prompt index so all modes draw alike.
        self.sampling_seed = sampling_seed
        self.concurrency, self.prompt_length = prompt_bank.shape[1:]
        self.mode = None
        self._observed_graph = None
        self._observed_decode_logits = None
        self._observed_routes = None

    def synchronize(self):
        if self.runner.device.type == 'cuda':
            torch.cuda.synchronize(self.runner.device)

    @torch.inference_mode()
    def prepare(self, mode):
        """Fresh cache and graph for each mode; all compilation is outside timing."""
        if mode not in ('serial', 'streams', 'batched'):
            raise ValueError('expected serial, streams or batched')
        runner = self.runner
        runner.release_cache()
        runner.config = replace(runner.config, branch_execution=mode)
        runner.model.config = replace(runner.model.config, branch_execution=mode)
        self.mode = mode
        self._observed_graph = None
        self._observed_decode_logits = None
        self._observed_routes = None
        runner.allocate(self.concurrency, self.prompt_length+self.output_length+4,
                        attention='fa3' if runner.device.type == 'cuda' else 'reference',
                        decode_splits=self.decode_splits, page_size=self.page_size)
        tokens = self.prompt_bank[0].to(runner.device)
        logits, first = runner.prefill(tokens, chunk_size=self.chunk_size)
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError('nonfinite warmup prefill logits')
        runner.prepare_decode(first, self.prompt_length, use_graph=self.use_graph)
        self.synchronize()

    def assert_execution(self):
        runner = self.runner
        if runner.model.config.branch_execution != self.mode or runner.config.branch_execution != self.mode:
            raise RuntimeError('requested branch mode changed')
        if self.use_graph != (runner.graph is not None):
            raise RuntimeError('decode graph policy changed; no fallback is allowed')
        if runner.prefill_graph is not None:
            raise RuntimeError('prefill policy changed; this workload specifies eager chunked prefill')

    @torch.inference_mode()
    def run_round(self, index):
        self.assert_execution()
        runner = self.runner
        # Selecting the next held-out prompt, the host-to-device copy, full
        # prefill and reset of every KV cursor are deliberately inside timing.
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if runner.device.type == 'cuda' else None
        if events:
            events[0].record()
        slot = index % len(self.prompt_bank)
        runner.reseed(self.sampling_seed + slot)
        tokens = self.prompt_bank[slot].to(runner.device)
        _, first = runner.prefill(tokens, chunk_size=self.chunk_size)
        if events:
            events[1].record()
        last = runner.decode(self.output_length-1)
        if events:
            events[2].record()
        if tuple(first.shape) != (self.concurrency,) or tuple(last.shape) != (self.concurrency,):
            raise RuntimeError('generated token batch no longer matches active clients')
        self.synchronize()
        split = {}
        if events:
            split = dict(prefill_seconds=events[0].elapsed_time(events[1]) / 1e3,
                         decode_seconds=events[1].elapsed_time(events[2]) / 1e3)
        return RoundResult(self.concurrency, self.concurrency*self.output_length,
                           decode_graph_hits=self.output_length-1 if self.use_graph else 0,
                           decode_eager_calls=0 if self.use_graph else self.output_length-1,
                           prefill_eager_calls=math.ceil(self.prompt_length/self.chunk_size),
                           prompt_tokens=self.concurrency*self.prompt_length, **split)

    @torch.inference_mode()
    def observe(self, index):
        """Return real generated IDs plus prefill/cached-decode logits, untimed.

        The selected decode logits are materialized by the same replicated
        vocabulary head that generates tokens. For graphs its live output buffer
        is exposed during a separate capture; no alternate head is substituted.
        """
        self.assert_execution()
        runner = self.runner
        runner.reseed(self.sampling_seed + index % len(self.prompt_bank))
        tokens = self.prompt_bank[index % len(self.prompt_bank)].to(runner.device)
        original = runner.model.compute_logits
        latest = {}
        latest_routes, prefill_routes, hooks = {}, {}, []
        observation_phase = {'name': 'prefill'}
        def routing_hook(layer_id):
            def hook(module, inputs, outputs):
                ids, probabilities = outputs
                latest_routes[layer_id] = (ids, probabilities)
                if observation_phase['name'] == 'prefill':
                    # Sample every client, both ends of every prefill chunk.
                    # Clone on the owning CUDA stream, never synchronizing a
                    # branch inside its hook. These samples are untimed only.
                    per_client = ids.shape[0] // self.concurrency
                    selected = [0] if per_client == 1 else [0, per_client-1]
                    sampled = tuple(value.reshape(self.concurrency, per_client, -1)[:, selected].clone()
                                    for value in (ids, probabilities))
                    prefill_routes.setdefault(layer_id, []).append(sampled)
            return hook
        for layer_id, layer in runner.model.layers.items():
            hooks.append(layer.moe.router.register_forward_hook(routing_hook(layer_id)))
        def observed_head(hidden):
            value = original(hidden)
            latest['logits'] = value
            return value
        runner.model.compute_logits = observed_head
        try:
            prefill_logits, first = runner.prefill(tokens, chunk_size=self.chunk_size)
            prefill_logits, first_copy = prefill_logits.float().cpu(), first.clone()
            prefill_routes = {layer_id: [tuple(value.cpu() for value in pair) for pair in chunks]
                              for layer_id, chunks in prefill_routes.items()}
            observation_phase['name'] = 'decode'
            # Capture once, then reuse EXACTLY that graph for A/B/A. Subsequent
            # prefill calls must reset the existing positions/lengths/slots via
            # _finish_prefill, just like production replacement requests.
            if self.use_graph and runner.graph is not self._observed_graph:
                runner.prepare_decode(first_copy, self.prompt_length, use_graph=True)
                self._observed_graph = runner.graph
                self._observed_decode_logits = latest['logits']
                self._observed_routes = dict(latest_routes)
            graph_before = runner.graph
            generated = [first_copy.cpu()]
            samples = sorted({0, (self.output_length-2)//2, self.output_length-2})
            logits, decode_routes = {}, {}
            for step in range(self.output_length-1):
                result = runner.decode(1)
                generated.append(result.clone().cpu())
                if step in samples:
                    current_logits = self._observed_decode_logits if self.use_graph else latest['logits']
                    logits[str(step+1)] = current_logits.float().cpu()
                    routes = self._observed_routes if self.use_graph else latest_routes
                    decode_routes[str(step+1)] = {layer_id: tuple(value.cpu().clone() for value in pair)
                                                 for layer_id, pair in routes.items()}
            if runner.graph is not graph_before:
                raise RuntimeError('request observation changed the captured graph')
            self.synchronize()
            return dict(tokens=torch.stack(generated, dim=1), prefill_logits=prefill_logits,
                        decode_logits=logits, decode_logit_steps=[i+1 for i in samples],
                        prefill_routes=prefill_routes, decode_routes=decode_routes)
        finally:
            runner.model.compute_logits = original
            for hook in hooks:
                hook.remove()


def compare_observations(reference, actual, *, atol=0., rtol=0.):
    """Fail closed on any token mismatch or predeclared logit tolerance failure."""
    if reference['decode_logit_steps'] != actual['decode_logit_steps']:
        raise AssertionError('decode logit observation steps differ')
    torch.testing.assert_close(actual['tokens'], reference['tokens'], atol=0, rtol=0)
    pairs = [('prefill', reference['prefill_logits'], actual['prefill_logits'])]
    pairs += [(f'decode_{key}', value, actual['decode_logits'][key])
              for key, value in reference['decode_logits'].items()]
    errors = {}
    for name, expected, observed in pairs:
        if not torch.isfinite(expected).all().item() or not torch.isfinite(observed).all().item():
            raise FloatingPointError(f'nonfinite {name} parity logits')
        torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)
        difference = (observed-expected).abs()
        errors[name] = dict(max_abs=difference.max().item(),
                           max_relative=(difference/expected.abs().clamp_min(1e-8)).max().item(),
                           compared_logits=expected.numel())
    routing_pairs = []
    if reference['prefill_routes'].keys() != actual['prefill_routes'].keys():
        raise AssertionError('prefill routing layer sets differ')
    for layer_id, chunks in reference['prefill_routes'].items():
        other_chunks = actual['prefill_routes'][layer_id]
        if len(chunks) != len(other_chunks):
            raise AssertionError('prefill routing chunk counts differ')
        routing_pairs.extend(zip(chunks, other_chunks))
    for step, layers in reference['decode_routes'].items():
        if layers.keys() != actual['decode_routes'][step].keys():
            raise AssertionError('decode routing layer sets differ')
        routing_pairs.extend((pair, actual['decode_routes'][step][layer_id]) for layer_id, pair in layers.items())
    route_count, probability_error = 0, 0.
    for (expected_ids, expected_weights), (observed_ids, observed_weights) in routing_pairs:
        # Expert identity must always match exactly, even if an explicitly
        # selected numerical tolerance allows probability/logit roundoff.
        torch.testing.assert_close(observed_ids, expected_ids, atol=0, rtol=0)
        if not torch.isfinite(expected_weights).all().item() or not torch.isfinite(observed_weights).all().item():
            raise FloatingPointError('nonfinite routing probabilities')
        torch.testing.assert_close(observed_weights, expected_weights, atol=atol, rtol=rtol)
        route_count += expected_ids.numel()
        probability_error = max(probability_error, (observed_weights-expected_weights).abs().max().item())
    return dict(status='PASS', atol=atol, rtol=rtol, generated_tokens_exact=True,
                compared_generated_tokens=reference['tokens'].numel(), logits=errors,
                routing=dict(expert_ids_exact=True, compared_expert_ids=route_count,
                             max_probability_abs_error=probability_error,
                             sampling='all layers/clients, first+last token per prefill chunk, all clients at reported decode steps'))


def router_state(model):
    """Read-only state identity, copied outside all throughput measurements."""
    state = {}
    for layer_id, layer in model.layers.items():
        router = layer.moe.router
        record = dict(backend=router.backend, cache_revision=router.cache_revision)
        for name in ('expert_bias', 'weight', 'weight_fp32'):
            value = getattr(router, name)
            record[name] = None if value is None else dict(dtype=str(value.dtype), shape=list(value.shape),
                sha256=hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest())
        state[layer_id] = record
    return state


@torch.inference_mode()
def qualify(backend, *, atol=0., rtol=0., on_record=None):
    """Compare both modes and overwritten-KV replacement against a fresh cache."""
    references, records = {}, []
    initial_router_state = router_state(backend.runner.model)
    for mode in ('serial', 'streams'):
        backend.prepare(mode)
        fresh_a = backend.observe(0)
        # B dirties all prompt+decode slots, then A must reproduce fresh A.
        backend.observe(1)
        reused_a = backend.observe(0)
        record = dict(check='kv_replacement', mode=mode,
                      graph_reused_without_recapture=backend.use_graph,
                      decode_context_reused=True,
                      **compare_observations(fresh_a, reused_a, atol=atol, rtol=rtol))
        records.append(record)
        if on_record is not None:
            on_record(record)
        references[mode] = fresh_a
        current_router_state = router_state(backend.runner.model)
        if current_router_state != initial_router_state:
            raise AssertionError(f'router bias/weights/cache/backend changed during {mode} qualification')
        state_record = dict(check='router_state_unchanged', mode=mode, status='PASS',
                            before=initial_router_state, after=current_router_state)
        records.append(state_record)
        if on_record is not None:
            on_record(state_record)
    record = dict(check='serial_vs_streams',
                  **compare_observations(references['serial'], references['streams'], atol=atol, rtol=rtol))
    records.append(record)
    if on_record is not None:
        on_record(record)
    return records


@contextmanager
def stage_events(model, events):
    """Instrument only this instance; events record on each actual branch stream."""
    original_layers, original_fusion = model._run_layers, model.fusion.forward
    cfg = model.config
    names = {(0, cfg.prelude): 'prelude',
             (cfg.prelude, cfg.prelude+cfg.core): 'core_a',
             (cfg.prelude+cfg.core, cfg.prelude+2*cfg.core): 'core_b',
             (cfg.prelude+2*cfg.core, cfg.num_hidden_layers): 'coda'}
    def layers(start, end, *args):
        name = names[(start, end)]
        events[name][0].record()
        result = original_layers(start, end, *args)
        events[name][1].record()
        return result
    def fusion(*args, **kwargs):
        events['fusion'][0].record()
        result = original_fusion(*args, **kwargs)
        events['fusion'][1].record()
        return result
    model._run_layers, model.fusion.forward = layers, fusion
    try:
        yield
    finally:
        model._run_layers, model.fusion.forward = original_layers, original_fusion


@torch.inference_mode()
def trace_decode(backend):
    """Separate diagnostic graph, with external CUDA timing events, after samples.

    No timing event is added to the throughput graph. The diagnostic uses the
    same mode/kernels/graph policy, but its extra event nodes can perturb overlap;
    the complete measured decode wall time and that limitation are both reported.
    No eager substitution occurs when decode graphs are requested.
    """
    runner = backend.runner
    backend.assert_execution()
    tokens = backend.prompt_bank[0].to(runner.device)
    _, first = runner.prefill(tokens, chunk_size=backend.chunk_size)
    first = first.clone()
    backend.synchronize()
    old_graph = runner.graph
    old_output = runner.decode_output
    diagnostic = None
    events = {name: (torch.cuda.Event(enable_timing=True, external=backend.use_graph),
                     torch.cuda.Event(enable_timing=True, external=backend.use_graph))
              for name in ('prelude', 'core_a', 'core_b', 'fusion', 'coda', 'decode_step')}
    try:
        runner.graph = None
        with stage_events(runner.model, events):
            def operation():
                events['decode_step'][0].record()
                runner._decode_step()
                events['decode_step'][1].record()
            if backend.use_graph:
                diagnostic = torch.cuda.CUDAGraph()
                with torch.cuda.graph(diagnostic):
                    operation()
                # Event-node initialization and first graph launches are not
                # the representative steady decode timeline.
                for _ in range(3):
                    runner.reset_decode(first, backend.prompt_length)
                    diagnostic.replay()
                runner.reset_decode(first, backend.prompt_length)
                backend.synchronize()
                start = time.perf_counter()
                diagnostic.replay()
            else:
                start = time.perf_counter()
                operation()
            backend.synchronize()
            wall = (time.perf_counter()-start)*1000
        origin = events['decode_step'][0]
        intervals = {name: (origin.elapsed_time(pair[0]), origin.elapsed_time(pair[1]))
                     for name, pair in events.items()}
        # Floating event-resolution noise can make the self-difference -0.0.
        intervals['decode_step'] = (0., intervals['decode_step'][1])
        return dict(interval_summary(intervals), mode=backend.mode,
                    decode_step_synchronized_wall_ms=wall,
                    execution='cuda_graph_with_external_events' if backend.use_graph else 'eager_with_events',
                    diagnostic_only=True, throughput_graph_instrumented=False,
                    diagnostic_graph_warmup_replays=3 if backend.use_graph else 0,
                    limitation='event nodes may perturb overlap; stage durations across streams are not additive')
    finally:
        backend.synchronize()
        if diagnostic is not None:
            diagnostic.reset()
        runner.graph = old_graph
        runner.decode_output = old_output
        runner.reset_decode(first, backend.prompt_length)
