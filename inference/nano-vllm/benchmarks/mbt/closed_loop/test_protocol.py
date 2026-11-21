# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Run with python -m unittest closed_loop.test_protocol; no GPU dependencies."""
import math
import unittest

from .protocol import RoundResult, WindowFailure, interval_summary, run_window


class FakeClock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now


class FakeBackend:
    def __init__(self, clock, *, clients=4, output=8, duration=2., fail_after=None):
        self.clock, self.clients, self.output = clock, clients, output
        self.duration, self.fail_after = duration, fail_after
        self.indices, self.drains = [], 0
    def synchronize(self):
        self.drains += 1
    def run_round(self, index):
        self.indices.append(index)
        self.clock.now += self.duration
        if index == self.fail_after:
            raise MemoryError('synthetic OOM')
        return RoundResult(self.clients, self.clients*self.output, decode_graph_hits=self.output-1,
                           prefill_eager_calls=16)


class ProtocolTests(unittest.TestCase):
    def test_complete_cohorts_denominator_first_token_and_replacements(self):
        clock = FakeClock()
        backend = FakeBackend(clock)
        result = run_window(backend, 4, 8, 5., clock=clock)
        self.assertEqual(backend.indices, [0, 1, 2])
        self.assertEqual(result['window_seconds'], 6.)
        self.assertEqual(result['output_tokens'], 96)
        self.assertEqual(result['output_tokens_per_second'], 16.)
        self.assertEqual(result['output_tokens_per_second_per_client'], 4.)
        self.assertEqual(result['submitted_requests'], 12)
        self.assertEqual(result['completed_requests'], 12)
        self.assertEqual(result['replacement_requests'], 8)
        self.assertEqual(result['decode_graph_hits'], 21)
        self.assertEqual(result['prefill_eager_calls'], 48)
        self.assertEqual(result['actual_active_clients_min'], 4)
        self.assertEqual(result['actual_active_clients_mean'], 4.)
        self.assertTrue(result['normal'])
        self.assertEqual(backend.drains, 1)

    def test_failed_round_never_gets_completed_or_output_credit(self):
        clock = FakeClock()
        backend = FakeBackend(clock, fail_after=1)
        with self.assertRaises(WindowFailure) as raised:
            run_window(backend, 4, 8, 5., clock=clock)
        failure = raised.exception
        self.assertIsInstance(failure.cause, MemoryError)
        self.assertEqual(failure.counters['completed_requests'], 4)
        self.assertEqual(failure.counters['submitted_requests'], 8)
        self.assertEqual(failure.counters['incomplete_requests'], 4)
        self.assertEqual(failure.counters['output_tokens'], 32)

    def test_wrong_batch_or_token_semantics_fail_closed(self):
        clock = FakeClock()
        for clients, outputs in ((3, 8), (4, 7), (4, 2048+8)):
            with self.subTest(clients=clients, outputs=outputs):
                with self.assertRaises(WindowFailure):
                    run_window(FakeBackend(clock, clients=clients, output=outputs), 4, 8, 1., clock=clock)

    def test_fallback_is_retained_as_stress(self):
        clock = FakeClock()
        backend = FakeBackend(clock)
        def fallback(index):
            clock.now += 2
            return RoundResult(4, 32, graph_fallbacks=1)
        backend.run_round = fallback
        result = run_window(backend, 4, 8, 1., clock=clock)
        self.assertFalse(result['normal'])
        self.assertEqual(result['status'], 'STRESS')

    def test_invalid_protocol_inputs(self):
        for c, length, seconds in ((0, 4, 1), (1, 0, 1), (1, 4, 0), (1, 4, math.inf), (1, 4, math.nan)):
            with self.assertRaises(ValueError):
                run_window(None, c, length, seconds)

    def test_trace_overlap_and_wall_are_not_sum_of_stages(self):
        intervals = dict(prelude=(1., 2.), core_a=(2., 7.), core_b=(3., 8.),
                         fusion=(8.2, 9.), coda=(9., 12.), decode_step=(0., 14.))
        result = interval_summary(intervals)
        self.assertEqual(result['core_overlap_ms'], 4.)
        self.assertAlmostEqual(result['branches_until_fusion_start_ms'], 6.2)
        self.assertEqual(result['decode_step_device_ms'], 14.)
        intervals['core_b'] = (7., 8.)
        self.assertEqual(interval_summary(intervals)['core_overlap_ms'], 0.)
        with self.assertRaises(ValueError):
            interval_summary(dict(core_a=(0., 1.)))


if __name__ == '__main__':
    unittest.main()
