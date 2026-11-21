# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
"""Convert a published native model bundle into an export this engine can serve.

The released MBT and Standard checkpoints are Megatron-native model-only bundles (`config.json`,
`manifest.json`, `model.safetensors`) meant for full-sequence loading. This rewrites one into an
`mbt-serving-v2` export: tensor names are mapped, experts stacked or layers sliced, and nothing is
recomputed. Both published layouts are recognized from the bundle's own manifest:

  mbt-native-shared-kv-v1        the Prelude-KV and Joint-KV multi-branch models
  mbt-native-standard-global-v1  the Standard Transformer baselines

    pip install huggingface_hub
    hf download gaotang/MBT-4-branch-W1 --local-dir MBT-4-branch-W1
    python tools/convert_native_bundle.py --bundle MBT-4-branch-W1 --output mbt-4-branch-w1-serving

The export's `manifest.json` records the source bundle's name and the SHA-256 of the exact
`config.json` and `model.safetensors` it was built from, so a measurement can be traced back to a
published checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.experimental.mbt_serving.export import convert_native_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="a downloaded native bundle directory")
    parser.add_argument("--output", required=True, help="new directory for the mbt-serving-v2 export")
    parser.add_argument("--scratch-root", default=None, help="optional: confine the output to this root")
    args = parser.parse_args()
    manifest = convert_native_bundle(args.bundle, args.output, args.scratch_root)
    print(json.dumps(dict(output=str(Path(args.output).resolve()), **manifest), indent=2))


if __name__ == "__main__":
    main()
