#!/usr/bin/env bash
# Run the complete Sparrow-insight validation with the official MSD model.
#
# Intended for a GPU host (T4/16GB or better) that already has:
#   - this repository checked out,
#   - the VDC-50 subset at dataset/VideoDetailCaption/,
#   - a Python 3.10/3.11 CUDA environment with the dependencies from
#     src/analyze/Validate_Sparrow_hypothesises/requirements.txt,
#   - Hugging Face models cached (see RUN_ON_GPU.md).
#
# It reproduces Figure 1(a)/1(b) with MSD, Figure 2 on both the target model
# and the MSD draft model, Figure 3/6 layer analyses on Qwen2.5-VL-7B, then
# audits everything and renders REPORT.md + figures + statistics.
#
# Usage:
#   ./run_sparrow_validation_gpu.sh [--limit N] [--skip-calibration] [--skip-msd] ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh"

OUTPUT_DIR="${OUTPUT_DIR:-results/sparrow_validation}"
LOG_PATH="${LOG_PATH:-$OUTPUT_DIR/gpu_run.log}"
# Strict local evidence defaults to calibrated points only.  Set
# ALLOW_OUT_OF_TOLERANCE=1 explicitly for a separately-labelled diagnostic.
ALLOW_OUT_OF_TOLERANCE="${ALLOW_OUT_OF_TOLERANCE:-0}"
EXTRA_ARGS=("$@")

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "== Sparrow validation run: $(date -Is) =="

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; run this script on the GPU host")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 1. Calibration is processor-only and needs no GPU model; run it first if the
#    file is missing (or force with RECALIBRATE=1).
if [[ ! -s "$OUTPUT_DIR/calibration.jsonl" || "${RECALIBRATE:-0}" == "1" ]]; then
    python -u -m src.analyze.Validate_Sparrow_hypothesises calibrate \
        --output "$OUTPUT_DIR/calibration.jsonl"
fi

# 2. GPU stages + audit + report.  Out-of-tolerance points are excluded from
#    the strict paper-shaped cohort by default.  Opt in explicitly when a
#    separately-labelled diagnostic is desired.
ALLOW_FLAG=()
if [[ "$ALLOW_OUT_OF_TOLERANCE" == "1" ]]; then
    ALLOW_FLAG+=(--allow-out-of-tolerance)
fi

python -u -m src.analyze.Validate_Sparrow_hypothesises all \
    --output-dir "$OUTPUT_DIR" \
    --quantized \
    "${ALLOW_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"

echo
echo "== Done. Report: $OUTPUT_DIR/report/REPORT.md =="
echo "   Figures:   $OUTPUT_DIR/report/figure*_insight_*.png"
echo "   Statistics: $OUTPUT_DIR/report/paper_statistics.json"
