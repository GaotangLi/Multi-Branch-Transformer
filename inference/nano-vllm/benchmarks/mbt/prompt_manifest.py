# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Strict prompt-manifest loader, the robust alternative to slicing one token stream.

A manifest is JSONL, one prompt per line, tokenized offline with a recorded tokenizer:
  {"id": str, "source": str, "length": int, "token_ids": [int, ...], "tokenizer_sha256": str,
   optional "task", "text", "orig_tokens", "trim_left", ...}
Selection is the first `count` matching records in file order. Every mismatch raises; nothing is
padded, truncated, reused or skipped silently. Tokenization never happens inside timed work.
"""
import hashlib
import json
from pathlib import Path

REQUIRED_KEYS = ('id', 'source', 'length', 'token_ids', 'tokenizer_sha256')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, *, length, count, vocab_size, expected_tokenizer_sha256=None, source=None):
    """Return ([count, length] int64 tensor, evidence dict) or raise ValueError."""
    import numpy as np
    path = Path(path)
    if count < 1 or length < 1:
        raise ValueError('positive prompt count and length required')
    records, seen = [], set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [key for key in REQUIRED_KEYS if key not in record]
            if missing:
                raise ValueError(f'{path}:{line_number} missing keys {missing}')
            if record['id'] in seen:
                raise ValueError(f'{path}:{line_number} duplicate prompt id {record["id"]!r}')
            seen.add(record['id'])
            if record['length'] != len(record['token_ids']):
                raise ValueError(f'{path}:{line_number} declared length {record["length"]} != {len(record["token_ids"])} tokens')
            if (source is None or record['source'] == source) and record['length'] == length:
                records.append(record)
    tokenizers = {record['tokenizer_sha256'] for record in records}
    if len(tokenizers) > 1:
        raise ValueError(f'mixed tokenizers among selected prompts: {sorted(tokenizers)}')
    if expected_tokenizer_sha256 is not None and tokenizers and tokenizers != {expected_tokenizer_sha256}:
        raise ValueError(f'tokenizer {sorted(tokenizers)} != expected {expected_tokenizer_sha256}')
    if len(records) < count:
        raise ValueError(f'{path} has {len(records)} prompts of length {length}'
                         f'{"" if source is None else " from " + source}; {count} required (no silent reuse)')
    chosen = records[:count]
    values = np.asarray([record['token_ids'] for record in chosen], dtype=np.int64)
    if values.shape != (count, length):
        raise ValueError('selected prompts are not one rectangular [count, length] array')
    if values.min() < 0 or values.max() >= vocab_size:
        raise ValueError('token IDs outside the checkpoint vocabulary')
    import torch
    evidence = dict(manifest=str(path.resolve()), manifest_sha256=file_sha256(path), source=source,
                    length=length, count=count, ids=[record['id'] for record in chosen],
                    tasks=sorted({record.get('task', record['source']) for record in chosen}),
                    tokenizer_sha256=next(iter(tokenizers)),
                    tokens_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    selection='first matching records in file order; no padding/truncation/reuse')
    return torch.from_numpy(values), evidence
