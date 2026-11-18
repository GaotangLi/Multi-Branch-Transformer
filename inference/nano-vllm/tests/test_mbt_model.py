# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Tiny v2 architecture/cache invariants; no real checkpoint or GPU is loaded."""

from dataclasses import replace
import os

import pytest
import torch
from torch.nn import functional as F

from nanovllm.experimental.wide_serving.config import ParallelContext
from nanovllm.experimental.mbt_serving.config import MBTConfig
from nanovllm.experimental.mbt_serving.model import MBTForCausalLM


def tiny(branches=(2, 1), bridge=False):
    return MBTConfig(variant="mbt" if branches else "standard", hidden_size=24,
                     num_hidden_layers=2 + sum(branches) if branches else 4,
                     num_attention_heads=6, num_key_value_heads=3, head_dim=4,
                     latent_size=12, intermediate_size=12, num_experts=4, num_experts_per_tok=2,
                     vocab_size=31, prelude=1 if branches else 0, coda=1 if branches else 0,
                     branch_depths=branches, input_bridge=bridge,
                     core_hidden_size=16 if bridge else None,
                     core_latent_size=8 if bridge else None,
                     core_intermediate_size=8 if bridge else None,
                     dtype="float32", moe_backend="reference")


class CausalAttention:
    def __init__(self, cached=False):
        self.cached, self.cache = cached, {}

    def __call__(self, q, k, v, layer_id, context):
        if self.cached:
            old_k, old_v = self.cache.get(layer_id, (k[:0], v[:0]))
            self.cache[layer_id] = (torch.cat((old_k, k)), torch.cat((old_v, v)))
            k, v = self.cache[layer_id]
        offset = k.shape[0] - q.shape[0]
        mask = torch.arange(k.shape[0], device=q.device)[None, :] <= torch.arange(
            offset, offset + q.shape[0], device=q.device)[:, None]
        output = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                               v.transpose(0, 1)[None], attn_mask=mask, enable_gqa=True)
        return output[0].transpose(0, 1)


def make_model(config, attention=None):
    torch.manual_seed(117)
    result = MBTForCausalLM(config, attention_backend=attention or CausalAttention())
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if "norm" in name:
                parameter.uniform_(.8, 1.2)
            else:
                parameter.normal_(0, .08)
        for name, buffer in result.named_buffers():
            if name.endswith("expert_bias"):
                buffer.copy_(torch.linspace(-.2, .2, buffer.numel()))
    return result.eval()


@pytest.mark.parametrize("branches,bridge", [((), False), ((2, 1), False), ((2, 1, 1), False),
                                            ((1, 1, 1, 1), False), ((2, 2), True),
                                            ((2, 2, 2), True), ((2, 2, 2, 2), True)])
def test_full_prefill_equals_chunked_cached_decode(branches, bridge):
    cfg = tiny(branches, bridge)
    full, cached = make_model(cfg), make_model(cfg, CausalAttention(cached=True))
    tokens = torch.tensor([2, 5, 3, 8, 1, 12])
    with torch.inference_mode():
        expected = full.compute_logits(full(tokens, torch.arange(6)))
        pieces = [cached(tokens[:3], torch.arange(3))]
        for index in range(3, 6):
            pieces.append(cached(tokens[index:index + 1], torch.tensor([index])))
        actual = cached.compute_logits(torch.cat(pieces))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("branches", [(2, 1), (2, 1, 1), (1, 1, 1, 1)])
@pytest.mark.parametrize("bridge", [False, True])
def test_one_shared_bridge_fork_and_pure_concat_fusion(branches, bridge):
    model = make_model(tiny(branches, bridge))
    inputs, outputs, bridge_calls = {}, {}, []
    hooks = []
    for branch, (start, end) in enumerate(model.config.branch_ranges):
        hooks.append(model.layers[str(start)].register_forward_pre_hook(
            lambda module, args, branch=branch: inputs.__setitem__(branch, args[0].clone())))
        hooks.append(model.layers[str(end - 1)].register_forward_hook(
            lambda module, args, output, branch=branch: outputs.__setitem__(branch, output.clone())))
    coda_start = model.config.prelude + sum(branches)
    hooks.append(model.layers[str(coda_start)].register_forward_pre_hook(
        lambda module, args: inputs.__setitem__("coda", args[0].clone())))
    if bridge:
        hooks.append(model.input_bridge.register_forward_hook(
            lambda module, args, output: bridge_calls.append(output.clone())))
    with torch.inference_mode():
        model(torch.tensor([2, 5, 3]), torch.arange(3))
        for branch in range(1, len(branches)):
            torch.testing.assert_close(inputs[0], inputs[branch], rtol=0, atol=0)
        if bridge:
            assert len(bridge_calls) == 1
            torch.testing.assert_close(inputs[0], bridge_calls[0], rtol=0, atol=0)
        expected = F.linear(torch.cat([norm(outputs[index]) for index, norm in enumerate(model.fusion.norms)], -1),
                            model.fusion.projection.weight)
        torch.testing.assert_close(inputs["coda"], expected, rtol=0, atol=0)
    for hook in hooks:
        hook.remove()


def test_explicit_equal_width_bridge_is_preserved():
    cfg = replace(tiny((2, 2)), input_bridge=True, core_hidden_size=24)
    model = make_model(cfg)
    assert model.input_bridge.projection.weight.shape == (24, 24)
    assert cfg.uses_input_bridge
    assert "input_bridge.norm.weight" in model.state_dict()


def test_config_serializes_exact_layer_roles_and_heterogeneous_shapes():
    cfg = tiny((2, 1, 1), True)
    assert MBTConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.branch_ranges == ((1, 3), (3, 4), (4, 5))
    assert cfg.physical_layer_count == cfg.num_hidden_layers == 6
    assert [cfg.layer_config(i).hidden_size for i in range(6)] == [24, 16, 16, 16, 16, 24]
    model = make_model(cfg)
    assert model.fusion.projection.weight.shape == (24, 48)
    assert len(model.cache_specs()) == 6
    assert [spec["num_kv_heads"] for spec in model.cache_specs()] == [3, 2, 2, 2, 2, 3]
    assert model.layers["1"].moe.w1.shape == (4, 16, 8)
    assert model.layers["0"].moe.w1.shape == (4, 24, 12)


@pytest.mark.parametrize("changes", [dict(format_version=1), dict(variant="concat"),
    dict(branch_depths=[1]), dict(branch_depths=[1, 0]), dict(branch_depths=[1, 1, 1, 1, 1]),
    dict(num_hidden_layers=100), dict(core_hidden_size=16, input_bridge=False),
    dict(core_hidden_size=18, input_bridge=True), dict(prelude=-1), dict(input_bridge=1),
    dict(core_latent_size=0), dict(branch_execution="distributed_serial"), dict(dtype="float16")])
def test_rejects_unsupported_or_ambiguous_architecture(changes):
    with pytest.raises(ValueError):
        replace(tiny(), **changes)


def test_unknown_config_keys_and_multigpu_fail_closed():
    with pytest.raises(TypeError):
        MBTConfig.from_dict(dict(tiny().to_dict(), surprise=1))
    with pytest.raises(ValueError, match="single-GPU"):
        MBTForCausalLM(tiny(), ParallelContext(0, 2), CausalAttention())
    from nanovllm.experimental.mbt_serving.runner import MBTServingRunner
    with pytest.raises(ValueError, match="one GPU"):
        MBTServingRunner("/not-read", parallel=ParallelContext(0, 2))


def test_streams_require_cuda_instead_of_silent_serial_fallback():
    model = make_model(replace(tiny(), branch_execution="streams"))
    with pytest.raises(ValueError, match="CUDA"):
        model(torch.tensor([2, 3]), torch.arange(2))


@pytest.mark.skipif(os.environ.get("MBT_CUDA_TESTS") != "1", reason="requires an explicitly allocated GPU")
@pytest.mark.parametrize("branches", [(2, 1), (2, 1, 1), (1, 1, 1, 1)])
def test_cuda_streams_equal_same_model_serial(branches):
    # This checks execution scheduling with an explicit attention oracle, not
    # real-weight FA4 or native Megatron equivalence.
    cfg = tiny(branches, True)
    serial = make_model(cfg).cuda()
    streams = make_model(replace(cfg, branch_execution="streams")).cuda()
    tokens, positions = torch.tensor([2, 5, 3], device="cuda"), torch.arange(3, device="cuda")
    with torch.inference_mode():
        expected = serial.compute_logits(serial(tokens, positions))
        actual = streams.compute_logits(streams(tokens, positions))
        torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
