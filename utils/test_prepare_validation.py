"""Small offline tests for strict Parquet-to-token validation preparation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("prepare_validation", Path(__file__).with_name("prepare_validation.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [[3, 1, 7, 4], [2, 6, 0, 5], [8, 4, 1, 3]]
        self.parquet = self.root / "fixture.parquet"
        self.write(self.rows)
        reference = self.root / "reference.npy"
        np.save(reference, np.asarray(self.rows, dtype="<i4"), allow_pickle=False)
        self.digest = prepare.sha256(reference)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, rows, value_type=None, name="input_ids"):
        column = pa.array(rows, type=pa.list_(value_type or pa.int32()))
        pq.write_table(pa.table({name: column}), self.parquet, row_group_size=2)

    def convert(self, output="prepared", **kwargs):
        return prepare._convert_parquet(self.parquet, self.root / output,
            expected_shape=(3, 4), expected_sha256=self.digest,
            vocabulary_size=9, batch_size=1, **kwargs)

    def test_exact_order_shape_checksum_and_evaluator_contract(self):
        manifest = self.convert(provenance={"source": "offline test"})
        np.testing.assert_array_equal(np.load(self.root / "prepared/tokens.npy", mmap_mode="r"), self.rows)
        self.assertEqual(manifest["tokens"], {"file": "tokens.npy", "shape": [3, 4], "dtype": "int32", "sha256": self.digest})
        self.assertEqual(manifest["context_length"], 3)
        self.assertEqual(manifest["num_sequences"], 3)
        self.assertEqual(json.loads((self.root / "prepared/manifest.json").read_text()), manifest)
        self.assertFalse((self.root / "prepared/loss_mask.npy").exists())
        self.assertEqual(sorted(p.name for p in (self.root / "prepared").iterdir()), ["manifest.json", "tokens.npy"])

    def test_reordered_rows_fail_pinned_checksum(self):
        self.write(list(reversed(self.rows)))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.convert()
        self.assertFalse((self.root / "prepared/manifest.json").exists())

    def test_wrong_width_nulls_and_token_range_fail(self):
        variants = ([self.rows[0][:-1], *self.rows[1:]],
                    [None, *self.rows[1:]],
                    [[3, None, 7, 4], *self.rows[1:]],
                    [[3, -1, 7, 4], *self.rows[1:]],
                    [[3, 9, 7, 4], *self.rows[1:]])
        for index, rows in enumerate(variants):
            with self.subTest(index=index):
                self.write(rows)
                with self.assertRaises(ValueError):
                    self.convert(output="bad-" + str(index))
                self.assertFalse((self.root / ("bad-" + str(index)) / "manifest.json").exists())

    def test_wrong_row_count_column_and_type_fail(self):
        self.write(self.rows[:2])
        with self.assertRaisesRegex(ValueError, "row count"):
            self.convert()
        self.write(self.rows, name="labels")
        with self.assertRaisesRegex(ValueError, "column"):
            self.convert()
        self.write(self.rows, value_type=pa.int64())
        with self.assertRaisesRegex(ValueError, "list<int32>"):
            self.convert()
        self.assertFalse((self.root / "prepared").exists())

    def test_existing_output_is_never_overwritten(self):
        destination = self.root / "prepared"
        destination.mkdir()
        sentinel = destination / "keep.txt"
        sentinel.write_text("untouched")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.convert()
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertEqual([p.name for p in destination.iterdir()], ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
