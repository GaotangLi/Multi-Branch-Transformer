# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""CLI and real tiny CPU-model parity; never requests a GPU or downloads weights."""
import unittest
from types import SimpleNamespace

from runtime import context_contract, make_backend, parser, preallocated_runner, validate_args


class ContractTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(['--phase', 'smoke', '--export', '/unused', '--data-prefix', '/unused',
                                   '--output-dir', '/unused', '--concurrency', '1', *extra])

    def test_defaults_preserve_shapes_and_exact_preset(self):
        args = self.args()
        validate_args(args)
        self.assertEqual(args.prefill_chunk_size, 128)
        self.assertEqual(args.variant, 'both')
        self.assertEqual((args.prompt_length, args.output_length), (4096, 1024))
        self.assertFalse(hasattr(args, 'parity_atol'))

    def test_extrapolation_requires_explicit_system_only_acknowledgement(self):
        with self.assertRaisesRegex(ValueError, 'explicit'):
            context_contract(4096, 1024, 4096, False)
        contract = context_contract(4096, 1024, 4096, True)
        self.assertTrue(contract['context_extrapolated'])
        self.assertEqual(contract['total_tokens'], 5120)
        self.assertFalse(contract['model_config_changed'])
        self.assertFalse(contract['rope_changed'])

    def test_repetition_window_and_profiler_are_bounded(self):
        for extra in (('--repeats', '2'), ('--profile-rounds', '2'), ('--timeline-samples', '2'),
                      ('--window-seconds', '59'), ('--window-seconds', 'nan'), ('--torch-profiler-steps', '33')):
            with self.assertRaises(ValueError):
                validate_args(self.args(*extra))


class PreallocatedCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_real_logits_routes_tokens_and_replacement_match_exactly(self):
        import torch
        from closed_loop.backend import FixedCohortBackend, compare_observations
        from closed_loop.test_backend_cpu import tiny_runner
        runner = tiny_runner()
        bank = torch.tensor([[[2, 4, 7, 1, 8, 3], [1, 5, 6, 3, 2, 9]],
                             [[9, 8, 7, 6, 5, 4], [8, 2, 9, 1, 4, 3]]])
        args = SimpleNamespace(output_length=5, prefill_chunk_size=2, prefill_token_budget=0, sampling_seed=0)
        backend = FixedCohortBackend(runner, bank, output_length=5, chunk_size=2, use_graph=False)
        backend.prepare('serial')
        reference_a, reference_b = backend.observe(0), backend.observe(1)
        original_prefill, original_reset, original_release = runner.prefill, runner.reset_decode, runner.release_cache
        with preallocated_runner(runner) as state:
            candidate = make_backend(runner, bank, args, 'preallocated')
            candidate.use_graph = False
            candidate.prepare('serial')
            first_context = state.context(0, 2)
            self.assertIs(first_context, state.context(0, 2))
            a, b, again = candidate.observe(0), candidate.observe(1), candidate.observe(0)
            for expected, actual in ((reference_a, a), (reference_b, b), (reference_a, again)):
                record = compare_observations(expected, actual, atol=0, rtol=0)
                self.assertTrue(record['generated_tokens_exact'])
                self.assertTrue(record['routing']['expert_ids_exact'])
            static_pointer = candidate.prompt_staging.data_ptr()
            result = candidate.run_round(1)
            self.assertEqual(result.output_tokens, 10)
            torch.testing.assert_close(runner.decode_output, reference_b['tokens'][:, -1], atol=0, rtol=0)
            self.assertEqual(candidate.prompt_staging.data_ptr(), static_pointer)
            torch.testing.assert_close(candidate.prompt_staging, bank[1], atol=0, rtol=0)
            candidate.prepare('serial')
            self.assertIsNot(first_context, state.context(0, 2))
            runner.release_cache()
            self.assertIsNone(state.cache)
            self.assertFalse(state.contexts)
            self.assertFalse(state.reset_slots)
        self.assertEqual(runner.prefill, original_prefill)
        self.assertEqual(runner.reset_decode, original_reset)
        self.assertEqual(runner.release_cache, original_release)
        runner.close()

    def test_context_seqused_is_not_mutated_by_cached_decode(self):
        import torch
        from closed_loop.test_backend_cpu import tiny_runner
        runner = tiny_runner()
        bank = torch.tensor([[[1, 2, 3, 4]], [[4, 3, 2, 1]]])
        args = SimpleNamespace(output_length=4, prefill_chunk_size=2, prefill_token_budget=0, sampling_seed=0)
        with preallocated_runner(runner) as state:
            backend = make_backend(runner, bank, args, 'preallocated')
            backend.use_graph = False
            backend.prepare('serial')
            snapshots = {key: (positions.clone(), context.seqused_k.clone(), context.slot_mapping.clone())
                         for key, (positions, context) in state.contexts.items()}
            backend.run_round(0)
            backend.run_round(1)
            for key, (positions, context) in state.contexts.items():
                for actual, expected in zip((positions, context.seqused_k, context.slot_mapping), snapshots[key]):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        runner.close()


if __name__ == '__main__':
    unittest.main()
