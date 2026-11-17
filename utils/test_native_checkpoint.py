"""Small CPU-only contract tests; imports no Torch or GPU libraries."""
import copy
import sys
import unittest

from native_checkpoint import model_config_from_bundle, standard_source_shapes, weight_mapping


def fixture(branches=0):
    a = dict(rung="W1", hidden_size=256, moe_latent_size=128, moe_ffn_hidden_size=128,
             num_attention_heads=4, num_query_groups=2, kv_channels=64,
             num_moe_experts=64, moe_router_topk=8, layernorm_epsilon=1e-5,
             normalization="RMSNorm", qk_layernorm=True, qk_l2_norm=False,
             attention_output_gate=True, attention_output_norm=False,
             rotary_base=500000, rotary_percent=1.0, tied_embeddings=True, vocabulary=128256,
             num_layers=2, physical_blocks=2, parallel_streams=False)
    kind = "mbt-native-standard-global-v1"
    if branches:
        import math
        kind = "mbt-native-shared-kv-v1"
        outer = dict(d=256, m=128, f=128, q_heads=4, kv_heads=2)
        core = dict(d=128, m=64, f=80, q_heads=2, kv_heads=1)
        a.update(num_layers=2+2*branches, physical_blocks=2+2*branches,
                 prelude=1, coda=1, core_depths=[2]*branches, core_blocks=[2]*branches,
                 branch_count=branches, parallel_streams=True, shared_input_bridge=True,
                 kv_mode="joint-branch-kv", shared_kv_source="aligned_branch_block_inputs",
                 shared_kv_cross_depth=False, shared_kv_detached=False,
                 shared_kv_input_scale=1/math.sqrt(branches), fusion="normalized_concat",
                 layer_shapes=[outer]+[core]*(2*branches)+[outer])
    return dict(bundle_format=kind, format_version=1, dtype="bfloat16", architecture=a)


class NativeLoaderContractTests(unittest.TestCase):
    def test_config_families(self):
        for branches in (0, 2, 3, 4):
            config = model_config_from_bundle(fixture(branches))
            self.assertEqual(len(config.branches), branches)
            self.assertEqual(config.physical_layers, 2 + 2*branches)
            self.assertEqual(config.kv_mode, "joint" if branches else "independent")

    def test_reject_changed_geometry_and_depth(self):
        for field, value in (("kv_channels", 32), ("num_query_groups", 1),
                             ("moe_router_topk", 4), ("tied_embeddings", False)):
            document = fixture(2)
            document["architecture"][field] = value
            with self.assertRaises(ValueError):
                model_config_from_bundle(document)
        document = fixture(2)
        document["architecture"]["core_depths"] = [1, 3]
        with self.assertRaises(ValueError):
            model_config_from_bundle(document)

    def test_mbt_legacy_key_mapping(self):
        document = fixture(2)
        config = model_config_from_bundle(document)
        source = {"decoder.layers.0.self_attention.linear_qkv.weight",
                  "decoder.layers.1.mlp.router.expert_bias",
                  "decoder.layers.3.mlp.router.expert_bias",
                  "decoder.layers.5.mlp.router.expert_bias",
                  "decoder.stream_concat_fusion.norm_b.weight",
                  "decoder.shared_kv_memory.banks.1.kv_projection.weight"}
        target = {"decoder.prelude.0.self_attention.linear_qkv.weight",
                  "decoder.branches.0.blocks.0.mlp.router.expert_bias",
                  "decoder.branches.1.blocks.0.mlp.router.expert_bias",
                  "decoder.coda.0.mlp.router.expert_bias",
                  "decoder.fusion.norms.1.weight",
                  "decoder.joint_kv.1.kv_projection.weight"}
        mapping = weight_mapping(source, target, config, document["architecture"], document["bundle_format"])
        self.assertEqual(mapping["decoder.branches.1.blocks.0.mlp.router.expert_bias"],
                         ("decoder.layers.3.mlp.router.expert_bias", ()))
        with self.assertRaises(ValueError):
            weight_mapping(source | {"unknown.weight"}, target, config,
                           document["architecture"], document["bundle_format"])

    def test_standard_global_layer_and_expert_axes(self):
        document = fixture()
        config = model_config_from_bundle(document)
        source = standard_source_shapes(document["architecture"])
        target = set()
        for name in source:
            if not name.startswith("decoder.layers."):
                target.add(name)
                continue
            suffix = name.removeprefix("decoder.layers.")
            for layer in range(2):
                if ".experts.experts." in name:
                    fc = suffix.split(".")[-2]
                    target.update(f"decoder.prelude.{layer}.mlp.experts.{fc}.weight{expert}"
                                  for expert in range(64))
                else:
                    target.add(f"decoder.prelude.{layer}.{suffix}")
        mapping = weight_mapping(source, target, config, document["architecture"], document["bundle_format"])
        self.assertEqual(mapping["decoder.prelude.1.mlp.experts.linear_fc2.weight63"],
                         ("decoder.layers.mlp.experts.experts.linear_fc2.weight", (1, 63)))
        self.assertEqual(mapping["decoder.prelude.0.mlp.router.expert_bias"],
                         ("decoder.layers.mlp.router.expert_bias", (0,)))
        with self.assertRaises(ValueError):
            weight_mapping(source, target | {"random.weight"}, config,
                           document["architecture"], document["bundle_format"])

    def test_no_gpu_library_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformer_engine", sys.modules)
        self.assertNotIn("megatron", sys.modules)


if __name__ == "__main__":
    unittest.main()
