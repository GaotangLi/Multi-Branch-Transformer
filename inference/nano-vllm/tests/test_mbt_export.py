# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Small stdlib contract checks; no Torch import, checkpoint weights or GPU."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nanovllm.experimental.mbt_serving.config import MBTConfig
from nanovllm.experimental.mbt_serving.export import (
    canonical_shapes, direct_map, expected_dtype, inspect_checkpoint,
    native_shapes, scratch_destination, validate_metadata, verify_model_contract, verify_proof,
    convert_native_tensors, export_checkpoint, load_export,
)


def config(*, branches=(2, 1), heterogeneous=False):
    return MBTConfig(variant="mbt", hidden_size=256, latent_size=128, intermediate_size=128,
                     num_attention_heads=4, num_key_value_heads=2, prelude=1, coda=1,
                     branch_depths=branches, num_hidden_layers=2+sum(branches),
                     core_hidden_size=128 if heterogeneous else None,
                     core_latent_size=64 if heterogeneous else None,
                     core_intermediate_size=80 if heterogeneous else None,
                     input_bridge=heterogeneous, vocab_size=256)


def metadata(c, layout):
    return SimpleNamespace(state_dict_metadata={name: SimpleNamespace(
        size=shape, properties=SimpleNamespace(dtype="torch." + expected_dtype(name, c)))
        for name, shape in native_shapes(c, layout).items()})


def model_proof(c):
    result = dict(hidden_size=c.hidden_size, num_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads, num_query_groups=c.num_key_value_heads,
        kv_channels=c.head_dim, moe_latent_size=c.latent_size, moe_ffn_hidden_size=c.intermediate_size,
        num_moe_experts=c.num_experts, moe_router_topk=c.num_experts_per_tok,
        rotary_base=c.rope_theta, vocabulary=c.vocab_size, layernorm_epsilon=c.rms_norm_eps,
        normalization="RMSNorm", attention="full_causal_gqa2", attention_output_gate=True,
        attention_output_norm=False, qk_layernorm=True, qk_l2_norm=False,
        qk_norm_location="pre_rope", tied_embeddings=True, loop_repeats=1, loss_count=1,
        prefix=False, gate_type="query_dependent_channel_sigmoid", gate_multiplier=1.0,
        gate_location="post_sdpa_pre_output_projection", position_embedding_type="rope",
        parallel_streams=c.variant == "mbt")
    if c.variant == "mbt":
        result.update(prelude=c.prelude, coda=c.coda, fusion_residual=False, fusion_extra_moe=False,
            fusion_norm_epsilon=c.rms_norm_eps, fusion_norms_learned=True, fusion="normalized_concat",
            core_depths=list(c.branch_depths), branch_count=len(c.branch_depths),
            shared_input_bridge=c.uses_input_bridge, input_bridge_count=int(c.uses_input_bridge),
            fusion_projection_shape=[c.hidden_size, len(c.branch_depths)*c.branch_hidden_size])
    if c.uses_input_bridge:
        result.update(input_bridge_shape=[c.branch_hidden_size, c.hidden_size], bridge_residual=False)
    result["layer_shapes"] = [dict(d=c.layer_config(i).hidden_size,
        m=c.layer_config(i).latent_size, f=c.layer_config(i).intermediate_size,
        q_heads=c.layer_config(i).num_attention_heads, kv_heads=c.layer_config(i).num_key_value_heads)
        for i in range(c.num_hidden_layers)]
    return result


class SchemaTests(unittest.TestCase):
    def test_k2_k3_k4_stacked_and_explicit(self):
        for depths in ((2, 1), (2, 1, 1), (2, 1, 1, 1)):
            c = config(branches=depths)
            for layout in ("stacked", "explicit"):
                report = validate_metadata(metadata(c, layout), c)
                self.assertEqual(report["native_layout"], layout)
                self.assertEqual(report["canonical_tensor_count"], len(canonical_shapes(c)))
            self.assertEqual(canonical_shapes(c)["fusion.projection.weight"], (256, 256*len(depths)))
            self.assertNotIn("input_bridge.projection.weight", canonical_shapes(c))

    def test_hybrid_shapes_and_single_shared_bridge(self):
        c = config(branches=(2, 2, 2), heterogeneous=True)
        report = validate_metadata(metadata(c, "explicit"), c)
        self.assertEqual(report["native_layout"], "explicit")
        target = canonical_shapes(c)
        self.assertEqual(target["input_bridge.projection.weight"], (128, 256))
        self.assertEqual(target["layers.0.moe.w1"], (64, 256, 128))
        self.assertEqual(target["layers.1.moe.w1"], (64, 160, 64))
        self.assertEqual(target["fusion.projection.weight"], (256, 384))
        self.assertEqual(sum(key.startswith("input_bridge.") for key in target), 2)
        with self.assertRaises(ValueError):
            native_shapes(c, "stacked")

    def test_equal_width_bridge_is_explicit(self):
        c = replace(config(), input_bridge=True)
        self.assertEqual(canonical_shapes(c)["input_bridge.projection.weight"], (256, 256))
        self.assertIn("decoder.width_in.projection.weight", direct_map(c))

    def test_standard_has_no_fusion_or_bridge(self):
        c = MBTConfig(hidden_size=256, latent_size=128, intermediate_size=128,
                      num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=4, vocab_size=256)
        validate_metadata(metadata(c, "stacked"), c)
        self.assertFalse(any(key.startswith(("fusion.", "input_bridge.")) for key in canonical_shapes(c)))
        verify_model_contract(model_proof(c), c)

    def test_unknown_model_tensor_is_rejected_optimizer_is_ignored(self):
        c = config()
        m = metadata(c, "stacked")
        item = next(iter(m.state_dict_metadata.values()))
        m.state_dict_metadata["optimizer.state.fake"] = item
        m.state_dict_metadata["decoder.layers.extra_state"] = object()
        validate_metadata(m, c)
        m.state_dict_metadata["decoder.unexpected.weight"] = item
        with self.assertRaises(ValueError):
            validate_metadata(m, c)

    def test_qb_dtype_shape_and_missing_keys_fail(self):
        c = config()
        for mutate in ("dtype", "shape", "missing"):
            m = metadata(c, "stacked")
            name = "decoder.layers.mlp.router.expert_bias"
            if mutate == "dtype":
                m.state_dict_metadata[name].properties.dtype = "torch.bfloat16"
            elif mutate == "shape":
                m.state_dict_metadata[name].size = (1, 64)
            else:
                del m.state_dict_metadata[name]
            with self.assertRaises(ValueError):
                validate_metadata(m, c)

    def test_proof_rejects_wrong_width_depth_fusion_or_loop(self):
        c = config(branches=(2, 2, 1), heterogeneous=True)
        verify_model_contract(model_proof(c), c)
        for key, value in (("loop_repeats", 2), ("prefix", True), ("fusion_residual", True),
                           ("core_depths", [1, 2, 2]), ("shared_input_bridge", False),
                           ("fusion", "softmax"), ("moe_latent_size", 256)):
            proof = model_proof(c)
            proof[key] = value
            with self.assertRaises(ValueError, msg=key):
                verify_model_contract(proof, c)
        proof = model_proof(c)
        del proof["layer_shapes"]
        with self.assertRaises(ValueError):
            verify_model_contract(proof, c)

    def test_old_k2_proof_core_a_b_supported(self):
        c = config(branches=(2, 2))
        proof = model_proof(c)
        for key in ("layer_shapes", "core_depths", "branch_count", "shared_input_bridge", "input_bridge_count"):
            del proof[key]
        proof.update(core_a=2, core_b=2, fusion="concat")
        verify_model_contract(proof, c)

    def test_untrusted_metadata_never_unpickled(self):
        with self.assertRaises(ValueError):
            inspect_checkpoint("/does-not-exist", config())


class ProofBindingTests(unittest.TestCase):
    def test_inventory_hash_binding_and_public_provenance(self):
        c = config()
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / ".metadata").write_bytes(b"tiny metadata fixture; not pickle")
            (checkpoint / "metadata.json").write_text("{}")
            inventory = {path.name: dict(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                         for path in checkpoint.iterdir()}
            hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint.iterdir()}
            proof = dict(status="PASS", stage="formal", iteration=7, horizon=7, labels=["100"],
                         checkpoint=str(checkpoint), model=model_proof(c), checkpoint_inventory=inventory,
                         metadata_sha256=hashes, source_inventory_sha256="a"*64)
            proof_path = root / "proof.json"
            proof_path.write_text(json.dumps(proof))
            public = verify_proof(proof_path, checkpoint, c)
            self.assertEqual(set(public), {"proof_sha256", "iteration", "source_inventory_sha256"})
            self.assertNotIn(temporary, json.dumps(public))
            (checkpoint / "metadata.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_proof(proof_path, checkpoint, c)

    def test_no_overwrite_or_escape(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, USER="portable-test"), patch.object(Path, "home", return_value=Path("/nonexistent-home")):
                self.assertEqual(scratch_destination(root / "new", root), root / "new")
                for path in (root, root.parent / "escape"):
                    with self.assertRaises(ValueError):
                        scratch_destination(path, root)
                (root / "exists").mkdir()
                with self.assertRaises(FileExistsError):
                    scratch_destination(root / "exists", root)


@unittest.skipUnless(os.environ.get("MBT_TORCH_TESTS") == "1", "opt-in tiny synthetic CPU tensor tests")
class TensorConversionTests(unittest.TestCase):
    """Tiny synthetic DCP fixtures; TMPDIR must be on scratch, no trained weights."""

    def tiny_config(self, heterogeneous=False):
        return MBTConfig(variant="mbt", hidden_size=16, num_hidden_layers=6,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                         latent_size=8, intermediate_size=8, vocab_size=32,
                         num_experts=4, num_experts_per_tok=2,
                         prelude=1, coda=1, branch_depths=(2, 1, 1), dtype="float32",
                         core_hidden_size=8 if heterogeneous else None,
                         core_latent_size=4 if heterogeneous else None,
                         core_intermediate_size=6 if heterogeneous else None,
                         input_bridge=heterogeneous)

    def test_native_explicit_and_stacked_keep_exact_values(self):
        import torch
        for hetero in (False, True):
            c = self.tiny_config(hetero)
            for layout in (("explicit",) if hetero else ("stacked", "explicit")):
                native = {name: torch.full(shape, float(i), dtype=torch.float32)
                          for i, (name, shape) in enumerate(native_shapes(c, layout).items())}
                converted = convert_native_tensors(native, c, layout)
                self.assertEqual(set(converted), set(canonical_shapes(c)))
                self.assertTrue(torch.equal(converted["embedding.weight"], native["embedding.word_embeddings.weight"]))
                for native_key, canonical in direct_map(c).items():
                    self.assertTrue(torch.equal(converted[canonical], native[native_key]))
                for i in range(c.num_hidden_layers):
                    key = f"decoder.layers.{i}.mlp.router.weight" if layout == "explicit" else "decoder.layers.mlp.router.weight"
                    value = native[key] if layout == "explicit" else native[key][i]
                    self.assertTrue(torch.equal(converted[f"layers.{i}.moe.router.weight"], value))

    def test_dcp_to_public_safetensors_roundtrip(self):
        import torch
        from torch.distributed.checkpoint import save, FileSystemWriter
        from safetensors.torch import load_file
        c = self.tiny_config(True)
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = root / "native"
            native = {name: torch.full(shape, float(i)/100, dtype=torch.float32)
                      for i, (name, shape) in enumerate(native_shapes(c, "explicit").items())}
            save(native, storage_writer=FileSystemWriter(str(checkpoint)))
            inventory = {path.name: dict(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                         for path in checkpoint.iterdir() if path.is_file()}
            digest = hashlib.sha256((checkpoint / ".metadata").read_bytes()).hexdigest()
            proof = dict(status="PASS", stage="formal", iteration=7, horizon=7, labels=["100"],
                         checkpoint=str(checkpoint), model=model_proof(c), checkpoint_inventory=inventory,
                         metadata_sha256={".metadata": digest}, source_inventory_sha256="a"*64)
            proof_path = root / "proof.json"
            proof_path.write_text(json.dumps(proof))
            with patch.dict(os.environ, USER="portable-test"), patch.object(Path, "home", return_value=Path("/nonexistent-home")):
                report = export_checkpoint(checkpoint, proof_path, root / "export", c, root, trusted=True)
            actual = load_file(str(root / "export/model.safetensors"))
            expected = convert_native_tensors(native, c, "explicit")
            self.assertEqual(set(actual), set(expected))
            self.assertTrue(all(torch.equal(actual[key], expected[key]) for key in expected))
            self.assertNotIn(temporary, json.dumps(report))
            self.assertFalse(report["optimizer_exported"])
            self.assertFalse(report["runtime_parity_qualified"])
            from nanovllm.experimental.mbt_serving.model import MBTForCausalLM
            model = MBTForCausalLM(c, attention_backend=lambda q, k, v, layer, ctx: q)
            loaded = load_export(model, root / "export")
            self.assertEqual(loaded["loaded_local_tensors"], len(expected))
            self.assertTrue(all(torch.equal(model.state_dict()[key], expected[key]) for key in expected))
            self.assertFalse(loaded["native_parity_qualified"])


if __name__ == "__main__":
    unittest.main()
