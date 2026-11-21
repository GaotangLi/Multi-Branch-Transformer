# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""CPU-only scheduler and accounting contracts for the bounded load generator.

Equal-length clients arrive together, so each round is a complete cohort. This
is genuine zero-think-time closed-loop replacement, but is deliberately not a
variable-length, continuously batched, network-facing serving scheduler.
"""
from dataclasses import asdict, dataclass
import math
import time


@dataclass(frozen=True)
class RoundResult:
    completed_requests: int
    output_tokens: int
    decode_graph_hits: int = 0
    decode_eager_calls: int = 0
    prefill_graph_hits: int = 0
    prefill_eager_calls: int = 0
    graph_fallbacks: int = 0
    mode_fallbacks: int = 0
    kv_preemptions: int = 0
    # Per-round split from CUDA events (device time, no extra synchronization).
    prompt_tokens: int = 0
    prefill_seconds: float = 0.
    decode_seconds: float = 0.


class WindowFailure(RuntimeError):
    def __init__(self, cause, counters):
        super().__init__(str(cause))
        self.cause, self.counters = cause, counters


def run_window(backend, concurrency, output_length, seconds, *, clock=time.perf_counter):
    """Time complete request rounds, including refill, scheduling and GPU drain.

    A window closes only at a cohort completion after the minimum duration.
    Its actual duration, not the requested minimum, is the denominator. Model
    load/graph capture/warmup happen in backend.prepare, before this function.
    The backend must synchronize before returning each RoundResult, so requests
    are counted completed only after their final generated tokens exist.
    """
    if min(concurrency, output_length) < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive concurrency/output length and finite window seconds required')
    counters = dict(rounds=0, submitted_requests=0, completed_requests=0,
                    replacement_requests=0, output_tokens=0, decode_graph_hits=0,
                    decode_eager_calls=0, prefill_graph_hits=0, prefill_eager_calls=0,
                    graph_fallbacks=0, mode_fallbacks=0, kv_preemptions=0,
                    prompt_tokens=0, prefill_seconds=0., decode_seconds=0.)
    clients = tuple(range(concurrency))
    backend.synchronize()
    start = clock()
    try:
        while True:
            # Admission and client iteration are timed scheduler work. Each
            # client has exactly one outstanding request until this round ends.
            admitted = len(clients)
            counters['submitted_requests'] += admitted
            if counters['rounds']:
                counters['replacement_requests'] += admitted
            result = backend.run_round(counters['rounds'])
            if result.completed_requests != admitted or result.output_tokens != admitted * output_length:
                raise RuntimeError('backend returned an incomplete or miscounted request cohort')
            for name, value in asdict(result).items():
                if value < 0:
                    raise RuntimeError('negative backend counter')
                counters[name] += value
            counters['rounds'] += 1
            elapsed = clock() - start
            if elapsed >= seconds:
                break
    except Exception as error:
        counters.update(window_seconds=clock() - start,
                        incomplete_requests=counters['submitted_requests'] - counters['completed_requests'])
        raise WindowFailure(error, counters) from error
    throughput = counters['output_tokens'] / elapsed
    normal = not any(counters[name] for name in ('graph_fallbacks', 'mode_fallbacks', 'kv_preemptions'))
    split = {}
    if counters['decode_seconds'] > 0:
        # Prefill emits the first output token; decode emits the remaining output_length-1 per client.
        decode_tokens = counters['output_tokens'] - counters['completed_requests']
        split = dict(decode_tokens=decode_tokens,
                     decode_tokens_per_second=decode_tokens / counters['decode_seconds'],
                     decode_tokens_per_second_per_client=decode_tokens / counters['decode_seconds'] / concurrency,
                     prefill_seconds_per_round=counters['prefill_seconds'] / counters['rounds'],
                     decode_seconds_per_round=counters['decode_seconds'] / counters['rounds'],
                     total_tokens_per_second=(counters['prompt_tokens'] + counters['output_tokens']) / elapsed,
                     split_semantics='CUDA-event device intervals: prompt staging+prefill+first token, then all decode steps')
    return dict(counters, **split, window_seconds=elapsed, requested_window_seconds=seconds,
                incomplete_requests=0, completion_rate=1.0,
                actual_active_clients_min=concurrency, actual_active_clients_max=concurrency,
                actual_active_clients_mean=float(concurrency), active_client_samples=counters['rounds'],
                active_client_semantics='one admitted outstanding request per client during each cohort; boundary accounting included',
                output_tokens_per_second=throughput, output_tokens_per_second_per_client=throughput/concurrency,
                oom=False, normal=normal, status='PASS' if normal else 'STRESS')


def interval_summary(intervals):
    """Summarize same-device CUDA event timestamps without adding stream times."""
    required = {'prelude', 'core_a', 'core_b', 'fusion', 'coda', 'decode_step'}
    if set(intervals) != required:
        raise ValueError('stage interval set is incomplete')
    for name, (start, end) in intervals.items():
        if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end < start:
            raise ValueError(f'invalid CUDA event interval: {name}')
    a, b = intervals['core_a'], intervals['core_b']
    return dict(stages_ms={name: dict(start_ms=v[0], end_ms=v[1], duration_ms=v[1]-v[0])
                          for name, v in intervals.items()},
                core_overlap_ms=max(0., min(a[1], b[1])-max(a[0], b[0])),
                branches_until_fusion_start_ms=intervals['fusion'][0]-min(a[0], b[0]),
                decode_step_device_ms=intervals['decode_step'][1]-intervals['decode_step'][0])
