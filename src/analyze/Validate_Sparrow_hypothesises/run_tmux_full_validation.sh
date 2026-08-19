#!/usr/bin/env bash
# Launch the complete local Sparrow verification inside a long-lived tmux job.
# The wrapper delegates the measured run/report generation to
# run_sparrow_validation_gpu.sh. All stages use the cached Qwen2-VL checkpoints.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-results/sparrow_validation_tmux_20260815_1643}"
source "$REPO_ROOT/src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/tmux.log") 2>&1

echo "== Sparrow VDC-50 tmux run: $(date -Is) =="
echo "Run directory: $RUN_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MSD_DEVICE_MAP=${MSD_DEVICE_MAP:-model_parallel}"
echo "MSD_MAX_MEMORY=${MSD_MAX_MEMORY:-0:22GiB,1:14GiB}"

nvidia-smi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MSD_DEVICE_MAP="${MSD_DEVICE_MAP:-model_parallel}"
export MSD_MAX_MEMORY="${MSD_MAX_MEMORY:-0:22GiB,1:14GiB}"
export OUTPUT_DIR="$RUN_DIR"
export ALLOW_OUT_OF_TOLERANCE="${ALLOW_OUT_OF_TOLERANCE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$REPO_ROOT/src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh"

echo "== Sparrow VDC-50 tmux run finished: $(date -Is) =="
echo "Statistics: $RUN_DIR/report/paper_statistics.json"
echo "Report: $RUN_DIR/report/REPORT.md"
