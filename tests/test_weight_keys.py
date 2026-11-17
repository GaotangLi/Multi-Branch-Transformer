"""Raw model-key conversion must be explicit, reversible and value-preserving."""
from collections import OrderedDict
import unittest

from mbt.config import ModelConfig
from mbt.model import convert_legacy_state_dict


class WeightKeyTests(unittest.TestCase):
    def test_standard_layer_mapping_and_extra_state_are_preserved(self):
        config = ModelConfig(scale="W1", family="standard", num_layers=4)
        values = [object() for _ in range(4)]
        state = OrderedDict((f"decoder.layers.{i}.self_attention.linear_qkv._extra_state", value)
                            for i, value in enumerate(values))
        state["output_layer._extra_state"] = None
        state["embedding.word_embeddings.weight"] = object()
        new = convert_legacy_state_dict(state, config)
        for i, value in enumerate(values):
            self.assertIs(new[f"decoder.prelude.{i}.self_attention.linear_qkv._extra_state"], value)
        self.assertIn("output_layer._extra_state", new)
        self.assertIsNone(new["output_layer._extra_state"])
        self.assertEqual(list(convert_legacy_state_dict(new, config, reverse=True)), list(state))

    def test_all_branch_keys_roundtrip_without_copying_values(self):
        for branches in (2, 3, 4):
            with self.subTest(branches=branches):
                config = ModelConfig(scale="W1", family="balanced", prelude=1,
                    branches=(2,) * branches, coda=1, branch_hidden_size=768,
                    branch_latent_size=368, branch_ffn_hidden_size=384, kv_mode="joint")
                state = OrderedDict()
                for layer in range(config.physical_layers):
                    state[f"decoder.layers.{layer}.self_attention.linear_qkv.weight"] = object()
                for letter in "abcd"[:branches]:
                    state[f"decoder.stream_concat_fusion.norm_{letter}.weight"] = object()
                state["decoder.stream_concat_fusion.fusion_projection.weight"] = object()
                for depth in range(2):
                    for child in ("kv_projection", "key_norm"):
                        state[f"decoder.shared_kv_memory.banks.{depth}.{child}.weight"] = object()
                for key in ("decoder.width_in.norm.weight", "decoder.width_in.projection.weight",
                            "decoder.final_layernorm.weight", "embedding.word_embeddings.weight"):
                    state[key] = object()
                old_keys = list(state)
                new = convert_legacy_state_dict(state, config)
                self.assertEqual(list(state), old_keys)
                for branch in range(branches):
                    for depth in range(2):
                        key = f"decoder.branches.{branch}.blocks.{depth}.self_attention.linear_qkv.weight"
                        self.assertIs(new[key], state[f"decoder.layers.{1 + branch * 2 + depth}.self_attention.linear_qkv.weight"])
                    self.assertIn(f"decoder.fusion.norms.{branch}.weight", new)
                self.assertIn("decoder.coda.0.self_attention.linear_qkv.weight", new)
                self.assertIn("decoder.joint_kv.1.key_norm.weight", new)
                roundtrip = convert_legacy_state_dict(new, config, reverse=True)
                self.assertEqual(list(roundtrip), old_keys)
                for key in state:
                    self.assertIs(roundtrip[key], state[key])

    def test_mixed_old_and_new_aliases_fail_on_collision(self):
        config = ModelConfig(scale="W1", family="standard", num_layers=4)
        state = {"decoder.layers.0.weight": object(), "decoder.prelude.0.weight": object()}
        with self.assertRaisesRegex(ValueError, "collision"):
            convert_legacy_state_dict(state, config)


if __name__ == "__main__":
    unittest.main()
