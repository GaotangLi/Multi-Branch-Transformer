# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""CPU tests for the strict prompt-manifest loader (unittest; no GPU, no pytest needed)."""
import json
import tempfile
import unittest
from pathlib import Path

from prompt_manifest import load_manifest

SHA = 'a' * 64


def write(rows):
    handle = tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False)
    for row in rows:
        handle.write(json.dumps(row) + '\n')
    handle.close()
    return Path(handle.name)


def row(index, length, source='fineweb', sha=SHA, **extra):
    return dict(id=f'{source}-{index}', source=source, length=length,
                token_ids=[(index * 7 + j) % 100 for j in range(length)], tokenizer_sha256=sha, **extra)


class ManifestTests(unittest.TestCase):
    def test_selects_first_matching_in_order(self):
        path = write([row(0, 8), row(1, 4), row(2, 8), row(3, 8)])
        tensor, evidence = load_manifest(path, length=8, count=2, vocab_size=100)
        self.assertEqual(tuple(tensor.shape), (2, 8))
        self.assertEqual(evidence['ids'], ['fineweb-0', 'fineweb-2'])
        self.assertEqual(tensor[1, 0].item(), 14)

    def test_insufficient_prompts_raise(self):
        path = write([row(0, 8), row(1, 8)])
        with self.assertRaisesRegex(ValueError, 'no silent reuse'):
            load_manifest(path, length=8, count=3, vocab_size=100)

    def test_declared_length_mismatch_raises(self):
        bad = row(0, 8)
        bad['token_ids'] = bad['token_ids'][:-1]
        with self.assertRaisesRegex(ValueError, 'declared length'):
            load_manifest(write([bad]), length=8, count=1, vocab_size=100)

    def test_tokenizer_mismatch_raises(self):
        path = write([row(0, 8)])
        with self.assertRaisesRegex(ValueError, 'expected'):
            load_manifest(path, length=8, count=1, vocab_size=100, expected_tokenizer_sha256='b' * 64)

    def test_mixed_tokenizers_raise(self):
        path = write([row(0, 8), row(1, 8, sha='c' * 64)])
        with self.assertRaisesRegex(ValueError, 'mixed tokenizers'):
            load_manifest(path, length=8, count=2, vocab_size=100)

    def test_vocab_overflow_raises(self):
        bad = row(0, 8)
        bad['token_ids'][3] = 100
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            load_manifest(write([bad]), length=8, count=1, vocab_size=100)

    def test_duplicate_ids_raise(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_manifest(write([row(0, 8), row(0, 8)]), length=8, count=1, vocab_size=100)

    def test_missing_key_raises(self):
        bad = row(0, 8)
        del bad['tokenizer_sha256']
        with self.assertRaisesRegex(ValueError, 'missing keys'):
            load_manifest(write([bad]), length=8, count=1, vocab_size=100)

    def test_source_filter(self):
        path = write([row(0, 8, source='ruler', task='qa_1'), row(1, 8, source='pg19'), row(2, 8, source='ruler', task='qa_2')])
        tensor, evidence = load_manifest(path, length=8, count=2, vocab_size=100, source='ruler')
        self.assertEqual(evidence['ids'], ['ruler-0', 'ruler-2'])
        self.assertEqual(evidence['tasks'], ['qa_1', 'qa_2'])


if __name__ == '__main__':
    unittest.main()
