# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
import unittest

from trace_analysis import analyze_trace, distribution, merge_intervals


class TraceAnalysisTests(unittest.TestCase):
    def test_distribution(self):
        result = distribution([1,2,3])
        self.assertEqual(result['n'],3)
        self.assertEqual(result['mean'],2)
        self.assertEqual(result['sample_sd'],1)
        self.assertAlmostEqual(result['p95'],2.9)

    def test_empty_not_zero_measurement(self):
        self.assertIsNone(distribution([])['mean'])
        self.assertIsNone(distribution([2])['sample_sd'])
        self.assertEqual(analyze_trace([])['status'],'NO_KERNEL_EVENTS')

    def test_union_not_sum(self):
        self.assertEqual(merge_intervals([(0,4),(2,6),(8,9)]),[[0,6],[8,9]])
        result = analyze_trace([dict(ph='X',cat='kernel',ts=a,dur=b-a) for a,b in [(0,4),(2,6),(8,9)]])
        self.assertEqual(result['kernel_duration_sum_us'],9)
        self.assertEqual(result['kernel_interval_union_us'],7)
        self.assertEqual(result['kernel_interval_gap_us']['mean'],2)

    def test_annotations_not_kernels(self):
        rows = [dict(ph='X',cat=cat,ts=0,dur=100,name='serving::decode')
                for cat in ['user_annotation','gpu_user_annotation','cuda_runtime']]
        rows.append(dict(ph='X',cat='kernel',ts=10,dur=2,name='actual'))
        self.assertEqual(analyze_trace(rows)['kernel_duration_sum_us'],2)

    def test_cpu_wait_not_gpu_wait(self):
        result = analyze_trace([dict(ph='X',cat='cuda_runtime',name='cudaStreamWaitEvent',ts=0,dur=3)])
        self.assertEqual(result['cuda_api_cpu_duration_us']['stream_wait_event']['mean'],3)
        self.assertEqual(result['kernel_count'],0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            distribution([float('nan')])
        with self.assertRaises(ValueError):
            analyze_trace([dict(ph='X',cat='kernel',ts=0,dur=-1)])


if __name__ == '__main__':
    unittest.main()
