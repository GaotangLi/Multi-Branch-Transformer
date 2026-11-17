#!/usr/bin/env bash
# One-GPU mechanics: full native graph + save/resume. Never starts paper training.
set -euo pipefail
: "${MBT_TEST_ROOT:?Set a scratch directory for test artifacts}"
: "${MBT_TEMP_ROOT:?Set a short scratch temporary directory}"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
unset WANDB_SERVICE _WANDB_SERVICE
PYTHON_BIN=${PYTHON_BIN:-python}
config=${1:-"${ROOT}/tests/native_smoke.json"}
common=(--config "$config" --output-dir "$MBT_TEST_ROOT" --temp-root "$MBT_TEMP_ROOT"
        --expert-parallel-size 1 --smoke-steps 4 --min-free-gb 1 --execute)
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/tests/native_probe.py" "${common[@]}" --stop-after-step 2
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/tests/native_probe.py" "${common[@]}" --resume
