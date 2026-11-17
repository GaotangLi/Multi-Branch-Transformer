#!/usr/bin/env bash
# Portable launcher; no scheduler policy, node names, credentials, or downloads.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
unset WANDB_SERVICE _WANDB_SERVICE
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
if [[ "$NNODES" == 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m mbt.train --execute "$@"
else
  : "${NODE_RANK:?Set NODE_RANK for multi-node torchrun}"
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node torchrun}"
  : "${MASTER_PORT:?Set MASTER_PORT for multi-node torchrun}"
  exec "$PYTHON_BIN" -m torch.distributed.run --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m mbt.train --execute "$@"
fi
