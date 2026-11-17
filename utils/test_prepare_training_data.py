"""Small CPU tests; set MBT_TEST_TOKENIZER for the full CLI integration test."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("prepare_training_data", Path(__file__).with_name("prepare_training_data.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def example_plan(tokenizer_sha256="a" * 64):
    names = [f"000_{index:05d}.parquet" for index in range(140)]
    return {"schema_version": 1, "dataset_repo": "test/offline", "dataset_revision": "fixed-test",
            "tokenizer_sha256": tokenizer_sha256,
            "training_sources": [names[index + 1::64] for index in range(64)],
            "validation_source": names[0]}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_original_batch_encoding_order_and_long_document_boundary(self):
        class Encoding:
            def __init__(self):
                self.long_lengths = []
            def encode_ordinary_batch(self, texts, num_threads):
                self.assert_threads = num_threads
                return [[] if text == "" else [len(text)] for text in texts]
            def encode_ordinary(self, text):
                self.long_lengths.append(len(text))
                return [len(text)]
        encoding = Encoding()
        result = prepare.encode_text_batch(encoding, ["short", "x" * 400003, "", "中文"])
        self.assertEqual(result, [[5], [400000, 3], [], [2]])
        self.assertEqual(encoding.long_lengths, [400000, 3])
        self.assertEqual(encoding.assert_threads, 1)

    def test_plan_requires_disjoint_exact_sources(self):
        plan = example_plan()
        groups, valid = prepare.validate_plan(plan)
        self.assertEqual(sum(map(len, groups)), 139)
        self.assertNotIn(valid, [name for group in groups for name in group])
        duplicate = copy.deepcopy(plan)
        duplicate["training_sources"][0][0] = valid
        with self.assertRaises(ValueError):
            prepare.validate_plan(duplicate)
        traversal = copy.deepcopy(plan)
        traversal["training_sources"][0][0] = "../other.parquet"
        with self.assertRaises(ValueError):
            prepare.validate_plan(traversal)
        truncated = copy.deepcopy(plan)
        truncated["training_sources"] = truncated["training_sources"][:-1]
        with self.assertRaises(ValueError):
            prepare.validate_plan(truncated)

    def test_source_preflight_rejects_missing_and_extra(self):
        plan = example_plan()
        raw = self.root / "raw"
        raw.mkdir()
        names = [name for group in plan["training_sources"] for name in group] + [plan["validation_source"]]
        for name in names:
            (raw / name).touch()
        groups, valid = prepare.source_paths(raw, plan)
        self.assertEqual(groups[0], [raw / name for name in plan["training_sources"][0]])
        self.assertEqual(valid, [raw / plan["validation_source"]])
        (raw / names[0]).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            prepare.source_paths(raw, plan)
        (raw / names[0]).touch()
        (raw / "extra.parquet").touch()
        with self.assertRaisesRegex(ValueError, "extra"):
            prepare.source_paths(raw, plan)

    def test_existing_output_and_wrong_tokenizer_fail_before_processing(self):
        plan = example_plan()
        raw = self.root / "raw"
        raw.mkdir()
        for name in [name for group in plan["training_sources"] for name in group] + [plan["validation_source"]]:
            (raw / name).touch()
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        tokenizer = self.root / "tokenizer.model"
        tokenizer.write_text("not the expected tokenizer")
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_text("original")
        with self.assertRaisesRegex(ValueError, "NEW output"):
            prepare.prepare(raw, tokenizer, output, plan_path, workers=1)
        self.assertEqual(sentinel.read_text(), "original")
        with self.assertRaisesRegex(ValueError, "Tokenizer SHA256"):
            prepare.prepare(raw, tokenizer, self.root / "new", plan_path, workers=1)
        self.assertFalse((self.root / "new").exists())

    @unittest.skipUnless(os.environ.get("MBT_TEST_TOKENIZER"), "Set MBT_TEST_TOKENIZER for native CLI integration")
    def test_main_cli_preserves_all_groups_document_tokens_and_empty_eod(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        tokenizer = Path(os.environ["MBT_TEST_TOKENIZER"]).resolve(strict=True)
        plan = example_plan(prepare.sha256(tokenizer))
        plan_path = self.root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        raw = self.root / "raw"
        raw.mkdir()
        text_by_name = {}
        special = ["Held-out text.", "Hello world!", "", " \n\t ", "中文 café مرحبا",
                   "line one\nline two", "<|end_of_text|>", 'print("hello")']
        for index in range(140):
            name = f"000_{index:05d}.parquet"
            text = special[index] if index < len(special) else f"Document {index}."
            text_by_name[name] = text
            pq.write_table(pa.table({"text": [text]}), raw / name)
        output = self.root / "indexed"
        subprocess.run([sys.executable, "-B", str(Path(prepare.__file__)), "--raw-dir", str(raw),
                        "--tokenizer-model", str(tokenizer), "--output-dir", str(output),
                        "--plan", str(plan_path), "--workers", "1"], check=True, timeout=240)
        data = json.loads((output / "data.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(data["train"]), 64)
        self.assertEqual(len(data["validation"]), 1)
        self.assertEqual(manifest["training_sources"], plan["training_sources"])
        encoding = prepare._encoding(str(tokenizer))
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        groups = plan["training_sources"] + [[plan["validation_source"]]]
        for index, (entry, names) in enumerate(zip(data["train"] + data["validation"], groups, strict=True)):
            self.assertEqual(entry["weight"], 1.0)
            expected_prefix = f"train/fineweb_edu_100bt.train.{index:03d}_text_document" if index < 64 else "valid/fineweb_edu_100bt.valid.000_text_document"
            self.assertEqual(entry["prefix"], str((output / expected_prefix).resolve()))
            dataset = IndexedDataset(entry["prefix"], multimodal=False, mmap=True, object_storage_config=None)
            self.assertEqual(len(dataset), len(names))
            for row, name in enumerate(names):
                expected = encoding.encode_ordinary(text_by_name[name]) + [128001]
                np.testing.assert_array_equal(dataset[row], expected)
            del dataset
        self.assertEqual(sum(record["documents"] for record in manifest["shards"]), 140)
        self.assertFalse(list(output.rglob("*.partial*")))


if __name__ == "__main__":
    unittest.main()
