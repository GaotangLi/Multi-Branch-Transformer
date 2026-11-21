# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Small numerical checks for native routing, graph boundaries and export loading."""

from dataclasses import replace
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from nanovllm.experimental.wide_serving.config import ParallelContext, WideConfig
from nanovllm.experimental.wide_serving.export import (
    canonical_shapes, convert_native_tensors, file_sha256, load_export, native_shapes, tensor_dtype,
)
from nanovllm.experimental.wide_serving.model import RMSNorm, WideForCausalLM, apply_rope, qb_route


def tiny(variant="concat"):
    return WideConfig(variant=variant, hidden_size=24, num_hidden_layers=4,
                      num_attention_heads=6, num_key_value_heads=3, head_dim=4,
                      latent_size=12, intermediate_size=12, num_experts=4,
                      num_experts_per_tok=2, vocab_size=31, prelude=1, core=1,
                      coda=1, dtype="float32", moe_backend="reference")


class CausalAttention:
    def __init__(self, cached=False):
        self.cached = cached
        self.cache = {}

    def __call__(self, q, k, v, layer_id, context):
        if self.cached:
            old_k, old_v = self.cache.get(layer_id, (k[:0], v[:0]))
            self.cache[layer_id] = (torch.cat((old_k, k)), torch.cat((old_v, v)))
            k, v = self.cache[layer_id]
        offset = k.shape[0] - q.shape[0]
        mask = torch.arange(k.shape[0])[None, :] <= torch.arange(offset, offset+q.shape[0])[:, None]
        result = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                                v.transpose(0, 1)[None], attn_mask=mask,
                                                enable_gqa=True)
        return result[0].transpose(0, 1)


def model(config, backend=None, parallel=None, elementwise_backend="torch"):
    torch.manual_seed(125)
    result = WideForCausalLM(config, parallel, backend or CausalAttention(), elementwise_backend=elementwise_backend)
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if name.endswith("norm.weight") or "norm_a.weight" in name or "norm_b.weight" in name:
                parameter.uniform_(0.8, 1.2)
            else:
                parameter.normal_(0, .15)
        for name, buffer in result.named_buffers():
            if name.endswith("expert_bias"):
                buffer.copy_(torch.linspace(-.3, .2, buffer.numel()))
    return result.eval()


def test_qb_saved_bias_only_changes_selection_and_weights_sum_2p5():
    x = torch.eye(4)
    weight = torch.tensor([[1., -1., .3, -.2], [-.5, 1., .1, .2],
                           [.7, -.3, -.7, .2], [-.4, .1, .6, -.3]])
    bias = torch.tensor([3., 2., -2., -3.])
    ids, probability = qb_route(x, weight, bias, 2, 2.5)
    assert ids.is_contiguous() and probability.is_contiguous()
    logits = x @ weight.T
    selected = torch.argsort(logits+bias, dim=-1, descending=True)[:, :2]
    torch.testing.assert_close(ids, selected)
    expected = torch.sigmoid(logits.gather(-1, selected))
    expected *= 2.5 / (expected.sum(-1, keepdim=True)+1e-9)
    torch.testing.assert_close(probability, expected)
    torch.testing.assert_close(probability.sum(-1), torch.full((4,), 2.5))


@pytest.mark.parametrize("variant", ["standard", "concat"])
def test_cached_decode_matches_full_causal_graph(variant):
    config = tiny(variant)
    full = model(config)
    cached = model(config, CausalAttention(cached=True))
    tokens = torch.tensor([2, 5, 3, 8, 1, 12])
    with torch.inference_mode():
        expected = full.compute_logits(full(tokens, torch.arange(6)))
        actual = [cached(tokens[:3], torch.arange(3))]
        for index in range(3, 6):
            actual.append(cached(tokens[index:index+1], torch.tensor([index])))
        actual = cached.compute_logits(torch.cat(actual))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_concat_forks_same_input_and_fuses_without_residual():
    result = model(tiny())
    captures = {}
    def capture(name):
        def hook(module, args):
            captures[name] = args[0].detach().clone()
        return hook
    handles = [result.layers[str(i)].register_forward_pre_hook(capture(str(i))) for i in (1, 2, 3)]
    outputs = {}
    for index in (1, 2):
        result.layers[str(index)].register_forward_hook(
            lambda module, args, output, key=index: outputs.__setitem__(key, output.detach().clone()))
    with torch.inference_mode():
        result(torch.tensor([2, 5, 3]), torch.arange(3))
        expected = F.linear(torch.cat((result.fusion.norm_a(outputs[1]), result.fusion.norm_b(outputs[2])), -1),
                            result.fusion.projection.weight)
    torch.testing.assert_close(captures["1"], captures["2"], rtol=0, atol=0)
    torch.testing.assert_close(captures["3"], expected, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


def test_grouped_q_gate_k_v_packing_and_gate_location():
    cfg = tiny("standard")
    saved = {}
    def backend(q, k, v, layer_id, context):
        saved.update(q=q, k=k, v=v)
        return torch.ones_like(q)
    attention = model(cfg, backend).layers["0"].attention
    x = torch.randn(2, cfg.hidden_size)
    packed = F.linear(x, attention.qkv.weight).reshape(2, 3, 24)
    q, gate, k, v = packed.split((8, 8, 4, 4), -1)
    with torch.inference_mode():
        out = attention(x, torch.zeros(2, dtype=torch.long), None)
        torch.testing.assert_close(saved["q"], attention.q_norm(q.reshape(2, 6, 4)))
        torch.testing.assert_close(saved["k"], attention.k_norm(k))
        torch.testing.assert_close(saved["v"], v)
        expected = F.linear(torch.sigmoid(gate.reshape(2, 24)), attention.out.weight)
        torch.testing.assert_close(out, expected)


def test_tp2_uneven_attention_and_branch_ownership():
    cfg = tiny()
    left = model(cfg, parallel=ParallelContext(0, 2))
    right = model(cfg, parallel=ParallelContext(1, 2))
    assert set(left.layers) == {"0", "1", "3"}
    assert set(right.layers) == {"0", "2", "3"}
    assert left.layers["0"].attention.num_kv_heads == 1
    assert right.layers["0"].attention.num_kv_heads == 2
    assert left.layers["1"].attention.num_kv_heads == 3
    assert right.layers["2"].attention.num_kv_heads == 3
    assert left.layers["0"].moe.w1.shape == (4, 12, 12)
    assert left.layers["1"].moe.w1.shape == (4, 24, 12)
    full = torch.arange(4*24*12).reshape(4, 24, 12)
    a = left.shard_export_tensor("layers.0.moe.w1", full)
    b = right.shard_export_tensor("layers.0.moe.w1", full)
    torch.testing.assert_close(torch.cat((a[:, :6], b[:, :6], a[:, 6:], b[:, 6:]), 1), full)


def test_to_bf16_preserves_router_bias_and_rope_fp32():
    result = model(tiny())
    original = {name: value.clone() for name, value in result.named_buffers()}
    result.to(torch.bfloat16)
    for name, value in result.named_buffers():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_elementwise_backend_rejects_invalid_choice_before_loading():
    from nanovllm.experimental.wide_serving.runner import ServingRunner
    with pytest.raises(ValueError, match="elementwise_backend"):
        model(tiny(), elementwise_backend="unknown")
    with pytest.raises(ValueError, match="elementwise_backend"):
        RMSNorm(4, 1e-5, elementwise_backend="unknown")
    with pytest.raises(ValueError, match="elementwise_backend"):
        ServingRunner("/nonexistent-export", elementwise_backend="unknown", device="cpu")


def test_optional_elementwise_flag_keeps_cpu_math_and_checkpoint_contract(monkeypatch):
    from nanovllm.experimental.wide_serving import ops
    from nanovllm.experimental.wide_serving.model import WideAttention
    def unexpected(*args, **kwargs):
        raise AssertionError("CUDA-only Triton operation invoked on CPU")
    for name in ("fused_rmsnorm", "fused_rope", "fused_gate"):
        monkeypatch.setattr(ops, name, unexpected)
    plain = model(tiny())
    optional = model(tiny(), elementwise_backend="triton")
    assert plain.config == optional.config
    assert plain.state_dict().keys() == optional.state_dict().keys()
    assert "elementwise_backend" not in optional.config.to_dict()
    assert all(module.elementwise_backend == "triton" for module in optional.modules()
               if isinstance(module, (RMSNorm, WideAttention)))
    with torch.inference_mode():
        tokens, positions = torch.tensor([2, 5, 1]), torch.arange(3)
        torch.testing.assert_close(plain(tokens, positions), optional(tokens, positions), rtol=0, atol=0)
    optional.set_elementwise_backend("torch")
    assert all(module.elementwise_backend == "torch" for module in optional.modules()
               if isinstance(module, (RMSNorm, WideAttention)))
    with pytest.raises(ValueError, match="elementwise_backend"):
        optional.set_elementwise_backend("unknown")
    assert optional.elementwise_backend == "torch"


def test_cache_release_synchronizes_then_resets_graph_before_freeing(monkeypatch):
    from nanovllm.experimental.wide_serving.runner import ServingRunner
    events = []
    runner = ServingRunner.__new__(ServingRunner)
    runner.device = torch.device("cuda")
    runner.graph = SimpleNamespace(reset=lambda: events.append("graph_reset"))
    runner.cache = object()
    runner.handle = SimpleNamespace(backend=object())
    for name in ("decode_context", "decode_positions", "decode_input", "decode_output"):
        setattr(runner, name, object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    runner.release_cache()
    assert events == ["synchronize", "graph_reset", "empty_cache"]
    assert runner.graph is None and runner.cache is None and runner.handle.backend is None
    assert all(getattr(runner, name) is None for name in ("decode_context", "decode_positions", "decode_input", "decode_output"))


def make_export(tmp_path, config):
    from safetensors.torch import save_file
    native = {name: torch.randn(shape, dtype=tensor_dtype(name, config)) for name, shape in native_shapes(config).items()}
    converted = convert_native_tensors(native, config)
    save_file(converted, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    manifest = {"format_version": 1, "config_sha256": file_sha256(tmp_path / "config.json"),
                "weights_sha256": file_sha256(tmp_path / "model.safetensors")}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return converted


def test_strict_export_loads_buffers_and_tp_slices(tmp_path):
    cfg = tiny()
    tensors = make_export(tmp_path, cfg)
    for rank in (0, 1):
        result = model(cfg, parallel=ParallelContext(rank, 2))
        report = load_export(result, tmp_path)
        assert report["canonical_tensors"] == len(canonical_shapes(cfg))
        for name, value in result.state_dict().items():
            expected = result.shard_export_tensor(name, tensors[name])
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_strict_loader_rejects_missing_router_state_even_with_hash_check_disabled(tmp_path):
    from safetensors.torch import save_file
    cfg = tiny()
    tensors = make_export(tmp_path, cfg)
    tensors.pop("layers.0.moe.router.expert_bias")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_export(model(cfg), tmp_path, verify_hashes=False)


def test_native_converter_rejects_unknown_or_missing_tensors():
    cfg = tiny()
    native = {name: torch.zeros(shape, dtype=tensor_dtype(name, cfg)) for name, shape in native_shapes(cfg).items()}
    native["decoder.layers.mlp.router.fake_bias"] = torch.zeros(1)
    with pytest.raises(ValueError, match="native key mismatch"):
        convert_native_tensors(native, cfg)


def _native_function(relative, name):
    """Run the frozen source's numerical function, without initializing Megatron."""
    source = os.environ.get("MBT_NATIVE_SOURCE")
    if not source:
        pytest.skip("set MBT_NATIVE_SOURCE to test against the frozen training implementation")
    path = Path(source) / relative
    tree = ast.parse(path.read_text())
    if "." in name:
        class_name, name = name.split(".", 1)
        tree = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    namespace = dict(torch=torch, F=F, SplitAlongDim=None, apply_module=lambda module: module)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_qb_matches_function_extracted_from_frozen_native_source():
    native = _native_function("megatron/core/transformer/moe/moe_utils.py", "quantile_balancing_topk")
    torch.manual_seed(125)
    x, weight, bias = torch.randn(19, 24), torch.randn(64, 24), torch.randn(64)
    actual_ids, actual_weights = qb_route(x, weight, bias, 8, 2.5)
    expected_weights, expected_ids, _ = native(F.linear(x, weight), 8, bias, 2.5)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)


def test_qkv_matches_method_extracted_from_frozen_native_source():
    native = _native_function("megatron/core/transformer/attention.py", "SelfAttention.get_query_key_value_tensors")
    attention = model(tiny("standard")).layers["0"].attention
    proxy = SimpleNamespace(linear_qkv=lambda x: (attention.qkv(x), None),
                            num_attention_heads_per_partition=6, num_query_groups_per_partition=3,
                            hidden_size_per_attention_head=4, world_size=1,
                            config=SimpleNamespace(num_query_groups=3, test_mode=False),
                            q_layernorm=attention.q_norm, k_layernorm=attention.k_norm)
    x = torch.randn(7, 1, 24)
    q, k, v, gate = native(proxy, x, output_gate=True)
    packed = attention.qkv(x[:, 0]).reshape(7, 3, 24)
    expected_q, expected_gate, expected_k, expected_v = packed.split((8, 8, 4, 4), -1)
    torch.testing.assert_close(q[:, 0], attention.q_norm(expected_q.reshape(7, 6, 4)), rtol=0, atol=0)
    torch.testing.assert_close(k[:, 0], attention.k_norm(expected_k), rtol=0, atol=0)
    torch.testing.assert_close(v[:, 0], expected_v, rtol=0, atol=0)
    torch.testing.assert_close(gate[:, 0], expected_gate.reshape(7, 6, 4), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native TE fidelity requires a GPU")
def test_bf16_norm_and_rope_match_native_transformer_engine():
    te = pytest.importorskip("transformer_engine.pytorch",
                             reason="install Transformer Engine to compare against its RMSNorm and RoPE")
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
    torch.manual_seed(125)
    x = torch.randn(17, 22, 64, device="cuda", dtype=torch.bfloat16)
    ours = RMSNorm(64, 1e-5).to(device="cuda", dtype=torch.bfloat16)
    native = te.RMSNorm(64, eps=1e-5, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        ours.weight.uniform_(.4, 1.6)
        native.weight.copy_(ours.weight)
    inv_freq = 1.0 / (500000.0 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32)/64))
    positions = torch.arange(17, device="cuda")
    phases = positions[:, None] * inv_freq[None, :]
    frequencies = torch.cat((phases, phases), -1)[:, None, None, :]
    with torch.inference_mode():
        actual_norm, expected_norm = ours(x), native(x)
        torch.testing.assert_close(actual_norm, expected_norm, atol=.015625, rtol=.008)
        actual_rope = apply_rope(expected_norm, positions, inv_freq)
        expected_rope = apply_rotary_pos_emb(expected_norm[:, None], frequencies, fused=True)[:, 0]
        torch.testing.assert_close(actual_rope, expected_rope, atol=.015625, rtol=.008)
