# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Bounded Chrome-trace accounting; annotations are never kernel time.

Kernel-duration sums are work counters, not wall time. This module does not
infer SM activity, tensor utilization, FLOP/s or bandwidth from busy intervals.
"""
from __future__ import annotations

import math
import statistics


def distribution(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, sample_sd=None, p50=None, p95=None, min=None, max=None)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError('Expected finite timing observations')
    ordered = sorted(values)
    def percentile(q):
        x = (len(ordered)-1)*q
        lo, hi = math.floor(x), math.ceil(x)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(x-lo)
    return dict(n=len(values), mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                p50=percentile(.5), p95=percentile(.95), min=ordered[0], max=ordered[-1])


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            raise ValueError('Invalid event interval')
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def analyze_trace(value):
    events = value.get('traceEvents', []) if isinstance(value, dict) else value
    kernels = []
    api = dict(graph_launch=[], stream_wait_event=[], event_synchronize=[], device_synchronize=[], launch=[])
    streams = set()
    for event in events:
        if event.get('ph') != 'X':
            continue
        category, name = event.get('cat'), event.get('name', '')
        if category not in ('kernel', 'cuda_runtime', 'cuda_driver'):
            continue
        start, duration = float(event['ts']), float(event['dur'])
        if not all(map(math.isfinite, (start, duration))) or duration < 0:
            raise ValueError('Malformed timed CUDA event')
        if category == 'kernel':
            kernels.append((start, start+duration))
            streams.add(str(event.get('args', {}).get('stream', event.get('tid', 'unknown'))))
            continue
        normalized = name.lower()
        if 'graphlaunch' in normalized:
            api['graph_launch'].append(duration)
        elif 'streamwaitevent' in normalized:
            api['stream_wait_event'].append(duration)
        elif 'eventsynchronize' in normalized:
            api['event_synchronize'].append(duration)
        elif 'devicesynchronize' in normalized or 'contextsynchronize' in normalized:
            api['device_synchronize'].append(duration)
        elif 'launchkernel' in normalized or 'launchcooperativekernel' in normalized:
            api['launch'].append(duration)
    union = merge_intervals(kernels)
    span = union[-1][1]-union[0][0] if union else 0.
    covered = sum(end-start for start, end in union)
    gaps = [b[0]-a[1] for a,b in zip(union,union[1:])]
    return dict(status='MEASURED' if kernels else 'NO_KERNEL_EVENTS',
        kernel_count=len(kernels), stream_count=len(streams),
        kernel_duration_sum_us=sum(end-start for start,end in kernels),
        first_to_last_kernel_span_us=span, kernel_interval_union_us=covered,
        kernel_interval_gap_us=distribution(gaps),
        kernel_interval_coverage=covered/span if span>0 else None,
        cuda_api_cpu_duration_us={name:distribution(samples) for name,samples in api.items()},
        unavailable_hardware_counters=['SM active','tensor core utilization','HBM bandwidth',
            'L2 hit rate','L2 throughput','occupancy','achieved FLOP/s'],
        limitations=[
            'Instrumented diagnostic only; not an uninstrumented throughput result.',
            'Kernel intervals may overlap. Duration sum is not GPU wall time.',
            'Interval coverage is not SM utilization or achieved compute throughput.',
            'Gaps may reflect dependencies, host scheduling or unrecorded work; not causal attribution.',
            'CUDA API durations are host-side calls, not device-side wait or graph execution durations.',
            'Only cat=kernel events count as kernels; annotation ranges, copies and memsets are excluded.'])
