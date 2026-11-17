"""Tiny validation-accounting tests; no model or dataset downloads required."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("evaluate_native", Path(__file__).with_name("evaluate_native.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class ValidationAccountingTests(unittest.TestCase):
    def manifest(self):
        return {"schema_version": 1, "context_length": 3, "num_sequences": 2,
                "provenance": {"packing": "frozen test"},
                "tokens": {"file": "tokens.npy", "dtype": "int32", "shape": [2, 4], "sha256": "a" * 64}}

    def test_manifest_contract_rejects_unbound_arrays(self):
        manifest = self.manifest()
        self.assertEqual(evaluation.validate_manifest(manifest), (3, 2))
        for key, value in (("shape", [2, 3]), ("dtype", "float32"), ("file", "../tokens.npy"), ("sha256", "missing")):
            changed = copy.deepcopy(manifest)
            changed["tokens"][key] = value
            with self.assertRaises(ValueError):
                evaluation.validate_manifest(changed)
        manifest["context_length"] = 8192
        with self.assertRaises(ValueError):
            evaluation.validate_manifest(manifest)

    def test_prefix_includes_final_partial_batch(self):
        count, starts = evaluation.sample_ranges(9, 4, 7)
        self.assertEqual(count, 7)
        self.assertEqual([(s, min(s + 4, count)) for s in starts], [(0, 4), (4, 7)])
        with self.assertRaises(ValueError):
            evaluation.sample_ranges(9, 4, 10)

    def test_exactly_one_shift_and_target_aligned_mask(self):
        import numpy as np
        tokens = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool_)
        inputs, labels, selected = evaluation.split_batch(tokens, mask, context=3, vocabulary=9)
        np.testing.assert_array_equal(inputs, [[1, 2, 3], [5, 6, 7]])
        np.testing.assert_array_equal(labels, [[2, 3, 4], [6, 7, 8]])
        np.testing.assert_array_equal(selected, mask)
        # This target-weighted mean differs from an unweighted average of the two sequence means.
        losses = np.array([[1., 99., 3.], [99., 6., 99.]])
        result = evaluation.summarize(float((losses * selected).sum()), int(selected.sum()))
        self.assertAlmostEqual(result["validation_loss"], 10 / 3)
        self.assertAlmostEqual(result["perplexity"], math.exp(10 / 3))
        np.testing.assert_array_equal(tokens, [[1, 2, 3, 4], [5, 6, 7, 8]])
        with self.assertRaises(ValueError):
            evaluation.split_batch(tokens, mask, context=3, vocabulary=8)
        with self.assertRaises(ValueError):
            evaluation.split_batch(tokens, mask.astype(float) + 0.5, context=3, vocabulary=9)

    def test_ce_chunks_equal_dense_reference(self):
        import torch
        from torch.nn import functional as F
        # Deterministic small values, including unequal rows and a partial chunk.
        logits = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7).sin().bfloat16()
        labels = torch.tensor([[0, 6, 1], [3, 2, 5]])
        result = evaluation.token_cross_entropy(logits, labels, chunk_size=4)
        reference = F.cross_entropy(logits.float().reshape(-1, 7), labels.reshape(-1), reduction="none").reshape(2, 3)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_empty_effective_validation_fails(self):
        with self.assertRaises(ValueError):
            evaluation.summarize(0, 0)
        with self.assertRaises(ValueError):
            evaluation.summarize(float("nan"), 3)


if __name__ == "__main__":
    unittest.main()
