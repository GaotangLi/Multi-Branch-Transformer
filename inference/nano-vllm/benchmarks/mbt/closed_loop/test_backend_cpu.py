# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Real tiny-model paged-KV replacement checks. CPU only; no kernel compilation."""
from dataclasses import replace
import unittest

import torch

from nanovllm.experimental.wide_serving.config import WideConfig, ParallelContext
from nanovllm.experimental.wide_serving.attention import PagedKVCache, PagedAttentionBackend
from nanovllm.experimental.wide_serving.model import WideForCausalLM
from nanovllm.experimental.wide_serving.runner import AttentionHandle, ServingRunner
from .backend import FixedCohortBackend, compare_observations, router_state


def tiny_runner():
    torch.manual_seed(125)
    runner = ServingRunner.__new__(ServingRunner)
    runner.device = torch.device('cpu')
    runner.parallel = ParallelContext()
    runner.config = WideConfig(variant='concat', hidden_size=24, num_hidden_layers=4,
                              num_attention_heads=6, num_key_value_heads=3, head_dim=4,
                              latent_size=12, intermediate_size=12, num_experts=4,
                              num_experts_per_tok=2, vocab_size=31, prelude=1, core=1,
                              coda=1, dtype='float32', moe_backend='reference')
    runner.handle = AttentionHandle()
    runner.model = WideForCausalLM(runner.config, runner.parallel, runner.handle).eval()
    with torch.no_grad():
        for name, parameter in runner.model.named_parameters():
            if name.endswith('norm.weight') or 'norm_a.weight' in name or 'norm_b.weight' in name:
                parameter.uniform_(.8, 1.2)
            else:
                parameter.normal_(0, .15)
        for name, buffer in runner.model.named_buffers():
            if name.endswith('expert_bias'):
                buffer.copy_(torch.linspace(-.3, .2, buffer.numel()))
    runner.graph = runner.prefill_graph = runner.cache = None
    runner._graph_router_signature = runner._graph_head_backend = runner._graph_execution_signature = None
    runner._decode_ready = False
    # ServingRunner.allocate intentionally allocates production BF16 KV. The
    # tiny CPU reference is FP32, so preserve the CPU fixture's matching dtype.
    def allocate(batch_size, capacity_tokens, *, attention, decode_splits, page_size):
        runner.release_cache()
        runner.cache = PagedKVCache(runner.model.cache_specs(), batch_size, capacity_tokens,
                                    device='cpu', dtype=torch.float32, block_size=page_size)
        runner.handle.backend = PagedAttentionBackend(runner.cache, implementation=attention,
                                                       decode_splits=decode_splits)
        return runner.cache.nbytes
    runner.allocate = allocate
    return runner


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def backend(self):
        bank = torch.tensor([[[2, 4, 7, 1, 8, 3], [1, 5, 6, 3, 2, 9]],
                             [[9, 8, 7, 6, 5, 4], [8, 2, 9, 1, 4, 3]]])
        return FixedCohortBackend(tiny_runner(), bank, output_length=5, chunk_size=2, use_graph=False)

    @torch.inference_mode()
    def test_full_generated_sequence_and_cached_logits_survive_request_replacement(self):
        backend = self.backend()
        backend.prepare('serial')
        original_router_state = router_state(backend.runner.model)
        original_context = backend.runner.decode_context
        def recapture_forbidden(*args, **kwargs):
            raise AssertionError('replacement must reuse existing decode context')
        backend.runner.prepare_decode = recapture_forbidden
        original = backend.observe(0)
        backend.observe(1)
        replaced = backend.observe(0)
        report = compare_observations(original, replaced)
        self.assertEqual(original['tokens'].shape, (2, 5))
        self.assertEqual(report['compared_generated_tokens'], 10)
        self.assertTrue(report['generated_tokens_exact'])
        self.assertTrue(report['routing']['expert_ids_exact'])
        self.assertGreater(report['routing']['compared_expert_ids'], 0)
        self.assertEqual(report['routing']['max_probability_abs_error'], 0)
        self.assertEqual(router_state(backend.runner.model), original_router_state)
        self.assertTrue(all(value['max_abs'] == 0 for value in report['logits'].values()))
        self.assertEqual(backend.runner.decode_cursor, 10)
        self.assertEqual(backend.runner.decode_positions.tolist(), [10, 10])
        self.assertEqual(backend.runner.decode_context.seqused_k.tolist(), [11, 11])
        self.assertIs(backend.runner.decode_context, original_context)
        backend.runner.close()

    @torch.inference_mode()
    def test_round_counts_first_token_and_overwrites_existing_cache(self):
        backend = self.backend()
        backend.prepare('serial')
        for index in range(6):
            result = backend.run_round(index)
            self.assertEqual(result.output_tokens, 10)
            self.assertEqual(result.completed_requests, 2)
            self.assertEqual(result.decode_eager_calls, 4)
            self.assertEqual(result.prefill_eager_calls, 3)
            self.assertEqual(result.graph_fallbacks, 0)
            self.assertEqual(backend.runner.decode_cursor, 10)
        backend.runner.close()

    def test_execution_changes_fail_instead_of_falling_back(self):
        backend = self.backend()
        backend.prepare('serial')
        backend.runner.model.config = replace(backend.runner.model.config, branch_execution='streams')
        with self.assertRaisesRegex(RuntimeError, 'mode changed'):
            backend.run_round(0)
        backend.runner.model.config = replace(backend.runner.model.config, branch_execution='serial')
        backend.use_graph = True
        with self.assertRaisesRegex(RuntimeError, 'graph policy'):
            backend.run_round(0)
        backend.runner.close()

    @torch.inference_mode()
    def test_parity_requires_exact_tokens_even_if_logit_tolerance_relaxed(self):
        backend = self.backend()
        backend.prepare('serial')
        a, b = backend.observe(0), backend.observe(0)
        b['tokens'][0, -1] += 1
        with self.assertRaises(AssertionError):
            compare_observations(a, b, atol=100, rtol=100)
        backend.runner.close()

    @torch.inference_mode()
    def test_routing_ids_checked_independently_of_logits_and_tokens(self):
        backend = self.backend()
        backend.prepare('serial')
        a, b = backend.observe(0), backend.observe(0)
        b['prefill_routes']['0'][0][0][0, 0, 0] += 1
        with self.assertRaises(AssertionError):
            compare_observations(a, b, atol=100, rtol=100)
        backend.runner.close()

    @torch.inference_mode()
    def test_router_snapshot_detects_bias_mutation(self):
        backend = self.backend()
        original = router_state(backend.runner.model)
        backend.runner.model.layers['0'].moe.router.expert_bias[0] += .5
        self.assertNotEqual(router_state(backend.runner.model), original)
        backend.runner.close()


if __name__ == '__main__':
    unittest.main()
