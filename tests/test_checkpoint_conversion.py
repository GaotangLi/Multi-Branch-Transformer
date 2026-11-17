"""Small synthetic tensor checks; no released model weights or GPU are needed."""
import importlib.util
from pathlib import Path
import unittest

import torch


_PATH = Path(__file__).resolve().parents[1] / "utils/checkpoint_conversion/weight_mapping.py"
_SPEC = importlib.util.spec_from_file_location("mbt_release_weight_mapping", _PATH)
mapping = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mapping)


def config(joint=False, branches=2):
    return dict(vocab_size=17, hidden_size=8, num_hidden_layers=branches + 2 if joint else 2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=2,
        moe_latent_size=4, moe_intermediate_size=3, num_experts=3, num_experts_per_tok=1,
        prelude=1 if joint else 0, coda=1 if joint else 0,
        branch_depths=[1] * branches if joint else [], branch_hidden_size=4,
        branch_moe_latent_size=3, branch_moe_intermediate_size=2,
        branch_num_attention_heads=2, branch_num_key_value_heads=1,
        kv_mode="joint" if joint else "independent", use_input_bridge=joint)


def native_fixture(joint=False, branches=2):
    """Build native-format tensors directly, independently of canonical_shapes."""
    c = config(joint, branches)
    generator = torch.Generator().manual_seed(123)

    def tensor(shape, dtype=torch.bfloat16):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)

    values = {"embedding.word_embeddings.weight": tensor((17, 8)),
              "decoder.final_layernorm.weight": tensor((8,))}
    per_layer = []
    for index in range(c["num_hidden_layers"]):
        shared = joint and 1 <= index <= branches
        d, m, f, q, kv = (4, 3, 2, 2, 1) if shared else (8, 4, 3, 4, 2)
        row = {
            "self_attention.linear_qkv.layer_norm_weight": tensor((d,)),
            "self_attention.linear_qkv.weight": tensor((2 * q * 2 + (0 if shared else 2 * kv * 2), d)),
            "self_attention.q_layernorm.weight": tensor((2,)),
            "self_attention.linear_proj.weight": tensor((d, q * 2)),
            "pre_mlp_layernorm.weight": tensor((d,)),
            "mlp.router.weight": tensor((3, d)),
            "mlp.router.expert_bias": tensor((3,), torch.float32),
            "mlp.fc1_latent_proj.weight": tensor((m, d)),
            "mlp.fc2_latent_proj.weight": tensor((d, m)),
        }
        if not shared:
            row["self_attention.k_layernorm.weight"] = tensor((2,))
        if joint:
            for expert in range(3):
                row[f"mlp.experts.linear_fc1.weight{expert}"] = tensor((2 * f, m))
                row[f"mlp.experts.linear_fc2.weight{expert}"] = tensor((m, f))
            values.update({f"decoder.layers.{index}." + name: value for name, value in row.items()})
        else:
            row["mlp.experts.experts.linear_fc1.weight"] = tensor((3, 2 * f, m))
            row["mlp.experts.experts.linear_fc2.weight"] = tensor((3, m, f))
            per_layer.append(row)
    if joint:
        values.update({
            "decoder.width_in.norm.weight": tensor((8,)),
            "decoder.width_in.projection.weight": tensor((4, 8)),
            "decoder.stream_concat_fusion.fusion_projection.weight": tensor((8, 4 * branches)),
            "decoder.shared_kv_memory.banks.0.kv_projection.weight": tensor((4, 4 * branches)),
            "decoder.shared_kv_memory.banks.0.key_norm.weight": tensor((2,)),
        })
        for branch in range(branches):
            values[f"decoder.stream_concat_fusion.norm_{'abcd'[branch]}.weight"] = tensor((4,))
    else:
        for name in per_layer[0]:
            values["decoder.layers." + name] = torch.stack([row[name] for row in per_layer])
    return c, values, mapping.MBT_FORMAT if joint else mapping.STANDARD_FORMAT


class WeightMappingTests(unittest.TestCase):
    def test_standard_layer_and_expert_axes(self):
        c, source, format_name = native_fixture()
        target, report = mapping.convert_values(source, c, format_name)
        for layer in range(2):
            self.assertTrue(torch.equal(target[f"model.layers.{layer}.attention.qkv.weight"],
                source["decoder.layers.self_attention.linear_qkv.weight"][layer]))
            self.assertTrue(torch.equal(target[f"model.layers.{layer}.moe.w1"],
                source["decoder.layers.mlp.experts.experts.linear_fc1.weight"][layer]))
        self.assertEqual(report["consumed_source_tensor_count"], len(source))

    def test_three_and_four_branch_shared_projection_and_fusion(self):
        for branches in (3, 4):
            with self.subTest(branches=branches):
                c, source, format_name = native_fixture(joint=True, branches=branches)
                target, report = mapping.convert_values(source, c, format_name)
                self.assertEqual(target["model.joint_kv.0.projection.weight"].shape, (4, 4 * branches))
                self.assertTrue(torch.equal(target[f"model.fusion.norms.{branches - 1}.weight"],
                    source[f"decoder.stream_concat_fusion.norm_{'abcd'[branches - 1]}.weight"]))
                self.assertEqual(report["consumed_source_tensor_count"], len(source))
        self.assertEqual(report["excluded_nonalias_tensors"], [])

    def test_joint_bridges_sharing_and_expert_order(self):
        c, source, format_name = native_fixture(joint=True)
        target, report = mapping.convert_values(source, c, format_name)
        self.assertNotIn("model.layers.1.attention.k_norm.weight", target)
        self.assertIn("model.layers.0.attention.k_norm.weight", target)
        self.assertTrue(torch.equal(target["model.input_bridge.projection.weight"],
            source["decoder.width_in.projection.weight"]))
        self.assertTrue(torch.equal(target["model.joint_kv.0.projection.weight"],
            source["decoder.shared_kv_memory.banks.0.kv_projection.weight"]))
        self.assertTrue(torch.equal(target["model.fusion.norms.1.weight"],
            source["decoder.stream_concat_fusion.norm_b.weight"]))
        for layer in range(4):
            for expert in range(3):
                self.assertTrue(torch.equal(target[f"model.layers.{layer}.moe.w1"][expert],
                    source[f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert}"]))
        self.assertEqual(report["consumed_source_tensor_count"], len(source))

    def test_router_bias_stays_exact_float32(self):
        c, source, format_name = native_fixture(joint=True)
        original = torch.tensor([0.123456789, -0.987654321, 0.000031416], dtype=torch.float32)
        source["decoder.layers.1.mlp.router.expert_bias"] = original
        target, _ = mapping.convert_values(source, c, format_name)
        result = target["model.layers.1.moe.router.expert_bias"]
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, original))
        source["decoder.layers.1.mlp.router.expert_bias"] = original.bfloat16()
        with self.assertRaisesRegex(ValueError, "Shape/dtype mismatch"):
            mapping.convert_values(source, c, format_name)

    def test_unknown_and_missing_tensors_fail(self):
        c, source, format_name = native_fixture()
        source["unexpected.weight"] = torch.ones(1, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "Unmapped source"):
            mapping.convert_values(source, c, format_name)
        del source["unexpected.weight"]
        del source["decoder.final_layernorm.weight"]
        with self.assertRaisesRegex(ValueError, "Missing source"):
            mapping.convert_values(source, c, format_name)

    def test_tied_head_is_the_only_verified_alias(self):
        c, source, format_name = native_fixture()
        source["output_layer.weight"] = source["embedding.word_embeddings.weight"].clone()
        _, report = mapping.convert_values(source, c, format_name)
        self.assertEqual(len(report["exact_tied_aliases"]), 1)
        source["output_layer.weight"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "not exactly tied"):
            mapping.convert_values(source, c, format_name)

    def test_extra_layer_axis_and_nonfinite_weights_fail(self):
        c, source, format_name = native_fixture()
        name = "decoder.layers.self_attention.linear_qkv.weight"
        source[name] = torch.cat([source[name], source[name][:1]])
        with self.assertRaisesRegex(ValueError, "leading layer axis"):
            mapping.convert_values(source, c, format_name)
        c, source, format_name = native_fixture(joint=True)
        source["decoder.final_layernorm.weight"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            mapping.convert_values(source, c, format_name)


if __name__ == "__main__":
    unittest.main()
