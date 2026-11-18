# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Separate diagnostic instrumentation; never installed in throughput windows."""
from contextlib import contextmanager
import math
import statistics
import time


def sample_statistics(values):
    values = sorted(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError('statistics require nonempty finite samples')
    def percentile(q):
        position = (len(values)-1)*q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high]-values[low])*(position-low)
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                p50=percentile(.5), p95=percentile(.95), minimum=values[0], maximum=values[-1], n=len(values))


def decompose_request(*, total, prefill_inclusive, reset, decode, scheduler, transfer):
    """Reset lives inside prefill. Transfer is included in the requested residual."""
    values = (total, prefill_inclusive, reset, decode, scheduler, transfer)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('timings must be finite and nonnegative')
    prefill = prefill_inclusive-reset
    other = total-prefill-decode-reset-scheduler
    if prefill < -1e-6 or other < -1e-6:
        raise ValueError('nested timing decomposition has overlapping or invalid boundaries')
    return dict(request_total_wall_ms=total, prefill_wall_ms=max(0., prefill),
                decode_total_wall_ms=decode, kv_reset_wall_ms=reset,
                scheduler_wall_ms=scheduler, host_to_device_or_prompt_transfer_ms=transfer,
                other_wall_ms=max(0., other))


class ReplayEvents:
    """Wrap existing graph replay without changing graph nodes or per-step sync."""
    def __init__(self, graph, event_pairs, clock=time.perf_counter):
        self.graph, self.event_pairs, self.clock = graph, event_pairs, clock
        self.host_ms = []

    def replay(self):
        pair = self.event_pairs[len(self.host_ms)]
        pair[0].record()
        start = self.clock()
        self.graph.replay()
        self.host_ms.append((self.clock()-start)*1000)
        pair[1].record()


@contextmanager
def instrument_replays(runner, pairs):
    original_graph = runner.graph
    if original_graph is None:
        raise RuntimeError('diagnostics require the same captured decode graph as throughput')
    proxy = ReplayEvents(original_graph, pairs)
    runner.graph = proxy
    try:
        yield proxy
    finally:
        runner.graph = original_graph


def profile_request(backend, index):
    """One actual fixed-C cohort: shared client latency, one independent sample.

    Component wall times synchronize at component boundaries only. Decode events
    use the already-qualified graph and never synchronize between decode steps.
    Recording events and boundary synchronization can perturb this diagnostic.
    """
    import torch
    runner = backend.runner
    backend.assert_execution()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(backend.output_length-1)]
    reset_times = []
    original_reset = runner.reset_decode
    def timed_reset(first, length):
        # Drain prior prefill before starting the reset clock. That drain is
        # charged to prefill, not hidden inside the metadata-reset component.
        backend.synchronize()
        start = time.perf_counter()
        original_reset(first, length)
        backend.synchronize()
        reset_times.append((time.perf_counter()-start)*1000)
    runner.reset_decode = timed_reset
    try:
        with torch.inference_mode(), instrument_replays(runner, pairs) as proxy:
            backend.synchronize()
            start = time.perf_counter()
            scheduler_start = time.perf_counter()
            prompt = backend.prompt_bank[index % len(backend.prompt_bank)]
            admitted = tuple(range(backend.concurrency))
            scheduler = (time.perf_counter()-scheduler_start)*1000
            transfer_start = time.perf_counter()
            tokens = backend.stage_prompt(prompt) if hasattr(backend, 'stage_prompt') else prompt.to(runner.device)
            backend.synchronize()
            transfer = (time.perf_counter()-transfer_start)*1000
            prefill_start = time.perf_counter()
            _, first = runner.prefill(tokens, chunk_size=backend.chunk_size)
            backend.synchronize()
            prefill_inclusive = (time.perf_counter()-prefill_start)*1000
            decode_start = time.perf_counter()
            last = runner.decode(backend.output_length-1)
            backend.synchronize()
            decode = (time.perf_counter()-decode_start)*1000
            scheduler_start = time.perf_counter()
            if tuple(first.shape) != (len(admitted),) or tuple(last.shape) != (len(admitted),):
                raise RuntimeError('profiling request cohort changed size')
            completed = len(admitted)
            scheduler += (time.perf_counter()-scheduler_start)*1000
            total = (time.perf_counter()-start)*1000
        if len(reset_times) != 1 or len(proxy.host_ms) != len(pairs):
            raise RuntimeError('missing reset or decode replay instrumentation')
        steps = [pair[0].elapsed_time(pair[1]) for pair in pairs]
        stats = sample_statistics(steps)
        values = decompose_request(total=total, prefill_inclusive=prefill_inclusive,
                                   reset=sum(reset_times), decode=decode,
                                   scheduler=scheduler, transfer=transfer)
        values.update(decode_step_mean_ms=stats['mean'], decode_step_p50_ms=stats['p50'],
                      decode_step_p95_ms=stats['p95'])
        return dict(kind='profile_request', cohort_index=index, concurrency=backend.concurrency,
                    mode=backend.mode, completed_requests=completed,
                    output_tokens=completed*backend.output_length, timings=values,
                    decode_step_device_ms=steps, decode_step_host_enqueue_ms=proxy.host_ms,
                    decode_step_statistics=stats, diagnostic_only=True,
                    measurement_unit='one cohort; all C clients share latency; not C independent samples',
                    timing_semantics='component wall boundaries synchronize; decode steps use events without per-step synchronization; other_wall_ms includes measured transfer',
                    graph_hits=len(pairs), graph_fallbacks=0, kv_preemptions=0)
    finally:
        runner.reset_decode = original_reset


def profile_summary(rows):
    if not rows:
        raise ValueError('no profiling cohorts')
    metrics = rows[0]['timings'].keys()
    if any(row['timings'].keys() != metrics for row in rows):
        raise ValueError('inconsistent timing metrics')
    return dict(kind='profile_summary', mode=rows[0]['mode'], concurrency=rows[0]['concurrency'],
                per_request={key: sample_statistics([row['timings'][key] for row in rows]) for key in metrics},
                decode_step_device_ms=sample_statistics([value for row in rows for value in row['decode_step_device_ms']]),
                decode_step_host_enqueue_ms=sample_statistics([value for row in rows for value in row['decode_step_host_enqueue_ms']]),
                cohort_samples=len(rows), actual_completed_requests=sum(row['completed_requests'] for row in rows),
                diagnostic_only=True,
                hardware_counters={name: dict(status='unavailable', reason='requires separately collected hardware performance counters')
                                   for name in ('sm_active', 'tensor_core_utilization', 'hbm_bandwidth', 'l2_hit_rate',
                                                'l2_throughput', 'occupancy', 'achieved_flops')})


def timeline_statistics(rows):
    if not rows:
        raise ValueError('no stage traces')
    values = {name: [] for name in ('overlap_ratio', 'core_a_ms', 'core_b_ms',
                                    'branches_until_fusion_start_ms', 'decode_step_device_ms')}
    for row in rows:
        a = row['stages_ms']['core_a']['duration_ms']
        b = row['stages_ms']['core_b']['duration_ms']
        if min(a, b) <= 0:
            raise ValueError('nonpositive branch duration')
        values['overlap_ratio'].append(row['core_overlap_ms']/min(a, b))
        values['core_a_ms'].append(a)
        values['core_b_ms'].append(b)
        for name in ('branches_until_fusion_start_ms', 'decode_step_device_ms'):
            values[name].append(row[name])
    return {key: sample_statistics(value) for key, value in values.items()}


def compare_timelines(serial, streams):
    left, right = timeline_statistics(serial), timeline_statistics(streams)
    return dict(branch_speedup=left['branches_until_fusion_start_ms']['mean']/right['branches_until_fusion_start_ms']['mean'],
                core_a_slowdown=right['core_a_ms']['mean']/left['core_a_ms']['mean'],
                core_b_slowdown=right['core_b_ms']['mean']/left['core_b_ms']['mean'],
                serial=left, streams=right,
                semantics='ratios of independent sample means; stream durations are not added')


def export_torch_trace(backend, output, steps):
    """Optional bounded diagnostic trace, separate from all measured requests."""
    import json
    import torch
    from trace_analysis import analyze_trace
    if not 1 <= steps <= min(32, backend.output_length-1):
        raise ValueError('trace must contain 1..32 in-capacity decode steps')
    with torch.inference_mode():
        prompt = backend.prompt_bank[0].to(backend.runner.device)
        backend.runner.prefill(prompt, chunk_size=backend.chunk_size)
        backend.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            with torch.profiler.record_function('bounded_decode_graph_replays'):
                backend.runner.decode(steps)
                backend.synchronize()
        profiler.export_chrome_trace(str(output))
    analysis = analyze_trace(json.loads(output.read_text()))
    return dict(kind='torch_trace', path=str(output), decode_steps=steps, diagnostic_only=True,
                trace_analysis=analysis,
                hardware_counters_available=False,
                limitation='trace reveals launches and waits, not SM/tensor/L2/HBM utilization counters')
