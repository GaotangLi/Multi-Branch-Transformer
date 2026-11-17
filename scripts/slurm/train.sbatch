#!/usr/bin/env bash
# Example only. Set partition/account/time/GPU/CPU/memory and scratch log paths
# in your sbatch command; these choices belong to your cluster.
#SBATCH --job-name=mbt-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
set -euo pipefail
: "${MBT_REPO:?Set MBT_REPO to the repository path}"
: "${MBT_CONFIG:?Set MBT_CONFIG to a recipe JSON}"
: "${MBT_DATA_CONFIG:?Set MBT_DATA_CONFIG to your data JSON}"
: "${MBT_OUTPUT_DIR:?Set MBT_OUTPUT_DIR to a fresh scratch run directory}"
: "${MBT_TEMP_ROOT:?Set MBT_TEMP_ROOT to a short writable scratch directory}"
unset WANDB_SERVICE _WANDB_SERVICE
export NPROC_PER_NODE=8
cd "$MBT_REPO"
srun --ntasks=1 bash scripts/train.sh --config "$MBT_CONFIG" \
  --data-config "$MBT_DATA_CONFIG" --output-dir "$MBT_OUTPUT_DIR" \
  --temp-root "$MBT_TEMP_ROOT" --expert-parallel-size 8 "$@"
