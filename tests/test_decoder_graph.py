"""Numerical CPU tests of the production MBT routing graph and fusion modules.

Diagnostic blocks replace expensive attention/MoE kernels, not the decoder.
Real attention, router and checkpoint behavior are checked separately on GPU.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mbt.config import ModelConfig
from mbt.decoder import MBTDecoder
from mbt.modules import ConcatFusion, JointKVProjection, WidthBridge


def rms(hidden, weight, eps=1e-5):
    work = hidden.float()
    return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + eps) * weight.float()).to(hidden.dtype)


class DiagnosticBlock(nn.Module):
    def __init__(self, index, trace, shared=False):
        super().__init__()
        self.index, self.trace, self.shared = index, trace, shared
        self.scale = nn.Parameter(torch.tensor(1.0 + (index + 1) / 10))
        self.shift = nn.Parameter(torch.tensor((index + 1) / 20))
        self.normalization_gain = nn.Parameter(torch.tensor(1.0 + index / 5))

    def normalized_attention_input(self, hidden):
        self.trace.append(("norm", self.index, hidden.detach().clone()))
        return hidden * self.normalization_gain

    def forward(self, hidden, attention_mask=None, rotary_pos_emb=None, shared_kv=None):
        self.trace.append(("block", self.index, hidden.detach().clone()))
        result = hidden * self.scale + self.shift
        if self.shared:
            if shared_kv is None:
                raise AssertionError("Shared block did not receive KV")
            key, value = shared_kv
            result = result + key + 2 * value
        elif shared_kv is not None:
            raise AssertionError("Independent block received external KV")
        return result


class DiagnosticJointKV(nn.Module):
    def __init__(self, depth, trace):
        super().__init__()
        self.depth, self.trace = depth, trace

    def forward(self, *states):
        self.trace.append(("kv", self.depth, tuple(h.detach().clone() for h in states)))
        combined = torch.stack(states).sum(0)
        return combined, combined * 0.5


class FirstBranch(nn.Module):
    def forward(self, *states):
        return states[0]


def independent_reference(decoder, hidden):
    """Write the mathematical branch decomposition independently of forward."""
    prelude = hidden
    for block in decoder.prelude:
        prelude = prelude * block.scale + block.shift
    if decoder.branches:
        if decoder.width_in is not None:
            bridge = decoder.width_in
            prelude = F.linear(rms(prelude, bridge.norm.weight), bridge.projection.weight)
        outputs = []
        for branch in decoder.branches:
            value = prelude
            for block in branch.blocks:
                value = value * block.scale + block.shift
            outputs.append(value)
        normalized = [rms(value, norm.weight) for value, norm in zip(outputs, decoder.fusion.norms)]
        hidden = F.linear(torch.cat(normalized, -1), decoder.fusion.fusion_projection.weight)
    else:
        hidden = prelude
    for block in decoder.coda:
        hidden = hidden * block.scale + block.shift
    return hidden


class DecoderGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def build(self, model, joint=False):
        trace = []
        decoder = MBTDecoder(model,
            lambda index, shape, shared: DiagnosticBlock(index, trace, shared),
            nn.Identity(),
            joint_kv_factory=(lambda depth: DiagnosticJointKV(depth, trace)) if joint else None)
        return decoder, trace

    def test_independent_outputs_and_gradients_match_branch_decomposition(self):
        # Includes no outer stack, unequal branch depths, and a real width bridge.
        configs = [
            ModelConfig(scale="W1", hidden_size=128, latent_size=64, ffn_hidden_size=64,
                        family="standard", num_layers=4),
            ModelConfig(scale="W1", hidden_size=128, latent_size=64, ffn_hidden_size=64,
                        family="wide-shallow", prelude=0, branches=(1, 2), coda=0),
            ModelConfig(scale="W1", hidden_size=128, latent_size=64, ffn_hidden_size=64,
                        family="wide-shallow", prelude=1, branches=(1, 3, 2), coda=2),
            ModelConfig(scale="W1", hidden_size=256, latent_size=128, ffn_hidden_size=128,
                        family="balanced", prelude=1, branches=(2, 1, 2, 1), coda=1,
                        branch_hidden_size=128, branch_latent_size=64, branch_ffn_hidden_size=64),
        ]
        for config in configs:
            with self.subTest(family=config.family, branches=config.branches):
                decoder, trace = self.build(config)
                original = torch.randn(3, 2, config.outer_shape[0])
                actual_input = original.clone().requires_grad_()
                reference_input = original.clone().requires_grad_()
                actual = decoder(actual_input)
                reference = independent_reference(decoder, reference_input)
                torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(actual_input.detach(), original, rtol=0, atol=0)
                weights = torch.randn_like(actual)
                parameters = tuple(decoder.parameters())
                left = torch.autograd.grad((actual * weights).sum(), (actual_input, *parameters), allow_unused=True)
                right = torch.autograd.grad((reference * weights).sum(), (reference_input, *parameters), allow_unused=True)
                for a, b in zip(left, right):
                    self.assertEqual(a is None, b is None)
                    if a is not None:
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
                self.assertEqual([x[1] for x in trace if x[0] == "block"],
                                 list(range(config.physical_layers)))
                self.assertEqual(len(list(decoder.parameters())), len({id(x) for x in decoder.parameters()}))

    def test_independent_branch_inputs_are_isolated(self):
        config = ModelConfig(scale="W1", family="wide-shallow", hidden_size=128,
                             latent_size=64, ffn_hidden_size=64, prelude=0, branches=(1, 1), coda=0)
        trace = []

        class InPlaceBlock(nn.Module):
            def __init__(self, number):
                super().__init__()
                self.number = number

            def forward(self, hidden, **unused):
                trace.append(hidden.clone())
                return hidden.add_(self.number + 1)

        decoder = MBTDecoder(config, lambda index, shape, shared: InPlaceBlock(index), nn.Identity())
        decoder.fusion = FirstBranch()
        value = torch.randn(2, 1, 128)
        saved = value.clone()
        decoder(value)
        torch.testing.assert_close(trace[0], saved, rtol=0, atol=0)
        torch.testing.assert_close(trace[1], saved, rtol=0, atol=0)
        torch.testing.assert_close(value, saved, rtol=0, atol=0)

    def test_joint_kv_uses_all_inputs_before_any_branch_update(self):
        for count in (2, 3, 4):
            with self.subTest(branches=count):
                config = ModelConfig(scale="W1", family="wide-shallow", hidden_size=128,
                    latent_size=64, ffn_hidden_size=64, prelude=1, branches=(2,) * count,
                    coda=1, kv_mode="joint")
                decoder, trace = self.build(config, joint=True)
                decoder.fusion = FirstBranch()
                value = torch.randn(2, 1, 128, requires_grad=True)
                output = decoder(value)
                pre = value * decoder.prelude[0].scale + decoder.prelude[0].shift
                states = [pre] * count
                expected_trace = [0]
                for depth in range(2):
                    inputs = [state * branch.blocks[depth].normalization_gain
                              for state, branch in zip(states, decoder.branches)]
                    observed = [x for x in trace if x[0] == "kv" and x[1] == depth]
                    self.assertEqual(len(observed), 1)
                    for a, b in zip(observed[0][2], inputs):
                        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
                    combined = torch.stack(inputs).sum(0)
                    states = [state * branch.blocks[depth].scale + branch.blocks[depth].shift + 2 * combined
                              for state, branch in zip(states, decoder.branches)]
                    expected_trace.extend(1 + index * 2 + depth for index in range(count))
                expected = states[0] * decoder.coda[0].scale + decoder.coda[0].shift
                torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
                expected_trace.append(config.physical_layers - 1)
                self.assertEqual([x[1] for x in trace if x[0] == "block"], expected_trace)
                self.assertEqual(tuple(expected_trace), decoder.get_layer_execution_order())
                # Selecting only branch 0 must still backpropagate into the other
                # branches' normalization parameters through the shared KV.
                contributors = [branch.blocks[0].normalization_gain for branch in decoder.branches]
                gradients = torch.autograd.grad(output.square().mean(), contributors)
                for gradient in gradients:
                    self.assertTrue(bool(torch.isfinite(gradient)))
                    self.assertGreater(float(gradient.abs()), 0)

    def test_fusion_has_no_hidden_residual_and_all_branch_gradients(self):
        module = ConcatFusion(128, 256, 3, initializer=lambda weight: nn.init.normal_(weight, std=.02))
        values = [torch.randn(2, 1, 128, requires_grad=True) for _ in range(3)]
        expected = F.linear(torch.cat([rms(x, n.weight) for x, n in zip(values, module.norms)], -1),
                            module.fusion_projection.weight)
        torch.testing.assert_close(module(*values), expected, rtol=0, atol=0)
        gradients = torch.autograd.grad(module(*values).square().mean(), values)
        self.assertTrue(all(bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0) for g in gradients))
        with torch.no_grad():
            module.fusion_projection.weight.zero_()
        self.assertEqual(float(module(*values).detach().abs().max()), 0.0)
        with self.assertRaises(ValueError):
            module(*values[:2])

    def test_real_joint_projection_preserves_gradients_to_every_branch(self):
        for count in (2, 3, 4):
            module = JointKVProjection(128, count, initializer=lambda weight: nn.init.normal_(weight, std=.02))
            states = [torch.randn(3, 2, 128, requires_grad=True) for _ in range(count)]
            key, value = module(*states)
            self.assertEqual(tuple(key.shape), (3, 2, 1, 64))
            self.assertEqual(tuple(value.shape), (3, 2, 1, 64))
            grads = torch.autograd.grad((key.square() + value.square()).mean(), states)
            self.assertTrue(all(bool(torch.isfinite(g).all()) and bool(g.abs().sum() > 0) for g in grads))


if __name__ == "__main__":
    unittest.main()
