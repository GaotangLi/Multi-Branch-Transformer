# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Write a prompt manifest in the format `benchmarks/mbt/prompt_manifest.py` loads.

Two modes:

  Real text (what measurements should use). Tokenize a corpus with a recorded tokenizer and cut it
  into non-overlapping windows of exactly `--length` tokens:

      python tools/build_prompt_manifest.py --output prompts-L4096.jsonl --length 4096 --count 256 \
        --text-file corpus.txt --tokenizer /path/to/tokenizer --tokenizer-file /path/to/tokenizer/tokenizer.json

  Synthetic ids (for checking that the engine and the benchmark run). Token ids are drawn uniformly
  from the vocabulary with a fixed seed:

      python tools/build_prompt_manifest.py --output demo-L4096.jsonl --length 4096 --count 256 \
        --export /path/to/export

Prompt content does not change the work a dense-attention forward pass does at a fixed length, so
synthetic ids measure the same throughput as real prompts. They are useless for anything about
output quality, and `source` records `synthetic-random` so the two can never be confused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_records(args, vocab_size):
    generator = random.Random(args.seed)
    population = range(vocab_size)
    fingerprint = hashlib.sha256(f"synthetic-random:{args.seed}:{vocab_size}:{args.length}".encode()).hexdigest()
    for index in range(args.count):
        ids = generator.choices(population, k=args.length)
        yield dict(id=f"{args.source}-L{args.length}-{index}", source=args.source, length=args.length,
                   token_ids=ids, tokenizer_sha256=fingerprint, trained_tokenizer=False)


def tokenized_records(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    fingerprint = file_sha256(args.tokenizer_file)
    stream, index = [], 0
    for line in Path(args.text_file).open():
        stream.extend(tokenizer(line, add_special_tokens=False)["input_ids"])
        while len(stream) >= args.length and index < args.count:
            ids, stream = stream[:args.length], stream[args.length:]
            yield dict(id=f"{args.source}-L{args.length}-{index}", source=args.source,
                       length=args.length, token_ids=ids, tokenizer_sha256=fingerprint,
                       tokenizer=str(Path(args.tokenizer_file).name))
            index += 1
        if index >= args.count:
            return
    raise SystemExit(f"{args.text_file} yielded {index} windows of {args.length} tokens; "
                     f"{args.count} requested (the loader never pads, truncates or reuses a prompt)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, required=True, help="tokens per prompt, exactly")
    parser.add_argument("--count", type=int, required=True,
                        help="at least concurrency * --prompt-rounds of the benchmark run")
    parser.add_argument("--export", default=None, help="read the vocabulary size from this bundle's config.json")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--text-file", default=None, help="one document per line; enables real tokenization")
    parser.add_argument("--tokenizer", default=None, help="name or path accepted by AutoTokenizer")
    parser.add_argument("--tokenizer-file", default=None, help="the tokenizer file to record a SHA-256 of")
    parser.add_argument("--source", default=None, help="defaults to synthetic-random or the text file's stem")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.count < 1 or args.length < 1:
        parser.error("positive --count and --length required")
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} exists; manifests are never appended to or overwritten")
    if args.text_file:
        if not (args.tokenizer and args.tokenizer_file):
            parser.error("--text-file requires --tokenizer and --tokenizer-file")
        args.source = args.source or Path(args.text_file).stem
        records = tokenized_records(args)
    else:
        vocab_size = args.vocab_size
        if vocab_size is None:
            if not args.export:
                parser.error("synthetic prompts need --vocab-size or --export")
            vocab_size = json.loads((Path(args.export) / "config.json").read_text())["vocab_size"]
        args.source = args.source or "synthetic-random"
        records = synthetic_records(args, vocab_size)
    written, fingerprints = 0, set()
    with output.open("x") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
            fingerprints.add(record["tokenizer_sha256"])
            written += 1
    print(json.dumps(dict(output=str(output.resolve()), records=written, length=args.length,
                          source=args.source, tokenizer_sha256=sorted(fingerprints),
                          manifest_sha256=file_sha256(output)), indent=2))


if __name__ == "__main__":
    main()
