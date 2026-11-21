# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""CPU contracts for timing semantics, instrumentation, and statistics."""
import unittest
from types import SimpleNamespace

from profiling import (ReplayEvents, compare_timelines, decompose_request,
                       instrument_replays, profile_summary, sample_statistics)


class StatisticsTests(unittest.TestCase):
    def test_sample_sd_linear_quantiles_and_singleton(self):
        result = sample_statistics([1, 2, 3, 4])
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['p50'], 2.5)
        self.assertAlmostEqual(result['p95'], 3.85)
        self.assertAlmostEqual(result['sample_sd'], 1.2909944487358056)
        self.assertIsNone(sample_statistics([1])['sample_sd'])

    def test_bad_samples_are_rejected(self):
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                sample_statistics(values)

    def test_nested_reset_is_not_double_counted_and_transfer_is_in_other(self):
        result = decompose_request(total=100, prefill_inclusive=40, reset=5,
                                   decode=45, scheduler=3, transfer=8)
        self.assertEqual(result['prefill_wall_ms'], 35)
        self.assertEqual(result['other_wall_ms'], 12)
        accounted = sum(result[key] for key in ('prefill_wall_ms', 'kv_reset_wall_ms',
                                                'decode_total_wall_ms', 'scheduler_wall_ms', 'other_wall_ms'))
        self.assertEqual(accounted, result['request_total_wall_ms'])
        with self.assertRaises(ValueError):
            decompose_request(total=5, prefill_inclusive=2, reset=3, decode=1, scheduler=1, transfer=1)

    def test_independent_sample_count_is_cohorts_not_clients(self):
        rows = [dict(mode='serial', concurrency=128, completed_requests=128,
                     timings={'request_total_wall_ms': value},
                     decode_step_device_ms=[1., 2.], decode_step_host_enqueue_ms=[.1, .2])
                for value in (10., 20., 30.)]
        result = profile_summary(rows)
        self.assertEqual(result['per_request']['request_total_wall_ms']['n'], 3)
        self.assertEqual(result['actual_completed_requests'], 384)
        self.assertEqual(result['decode_step_device_ms']['n'], 6)
        self.assertTrue(all(value['status']=='unavailable' for value in result['hardware_counters'].values()))

    def test_replay_events_never_synchronize_and_restore_graph_after_error(self):
        calls = []
        class Event:
            def __init__(self, label):
                self.label = label
            def record(self):
                calls.append(self.label)
        graph = SimpleNamespace(replay=lambda: calls.append('replay'))
        pairs = [(Event('start'), Event('end'))]
        times = iter((1., 1.002))
        proxy = ReplayEvents(graph, pairs, clock=lambda: next(times))
        proxy.replay()
        self.assertEqual(calls, ['start', 'replay', 'end'])
        self.assertAlmostEqual(proxy.host_ms[0], 2.)
        runner = SimpleNamespace(graph=graph)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with instrument_replays(runner, pairs):
                self.assertIsNot(runner.graph, graph)
                raise RuntimeError('sentinel')
        self.assertIs(runner.graph, graph)

    def test_timeline_overlap_and_slowdown_are_not_additive(self):
        def row(a, b, overlap, span, total):
            return dict(stages_ms={'core_a': {'duration_ms': a}, 'core_b': {'duration_ms': b}},
                        core_overlap_ms=overlap, branches_until_fusion_start_ms=span, decode_step_device_ms=total)
        result = compare_timelines([row(10, 10, 0, 20, 30)]*3,
                                   [row(15, 15, 15, 15, 25)]*3)
        self.assertEqual(result['streams']['overlap_ratio']['mean'], 1.)
        self.assertEqual(result['core_a_slowdown'], 1.5)
        self.assertAlmostEqual(result['branch_speedup'], 4/3)
        self.assertEqual(result['streams']['decode_step_device_ms']['mean'], 25)


if __name__ == '__main__':
    unittest.main()
