#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Multi-branch Transformer contributors
set -euo pipefail
nv_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
nv_python=${1:-python3}
if [ -n "${TMPDIR:-}" ]; then
  # An explicit TMPDIR is honoured, but never one on a home filesystem: these tests write caches
  # there, and a full home quota fails mid-write instead of erroring.
  test -d "$TMPDIR" && test -w "$TMPDIR"
  nv_tmp=$(readlink -f "$TMPDIR")
  case "$nv_tmp" in "$(readlink -f "$HOME")"/*|/home/*) echo 'TMPDIR must not be on home' >&2; exit 2;; esac
else
  TMPDIR=$(mktemp -d); export TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$nv_root:$nv_root/benchmarks/mbt"
export XDG_CACHE_HOME="$TMPDIR/cache" TRITON_CACHE_DIR="$TMPDIR/triton" CUDA_CACHE_PATH="$TMPDIR/cuda"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TORCH_EXTENSIONS_DIR="$TMPDIR/extensions"
cd "$nv_root"
"$nv_python" -B -m unittest discover -s benchmarks/mbt -p 'test_*.py' -q
"$nv_python" -B -m pytest -c "$nv_root/pyproject.toml" -q -p no:cacheprovider tests/test_wide_model.py tests/test_wide_attention.py
