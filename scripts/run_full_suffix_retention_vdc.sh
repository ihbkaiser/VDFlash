#!/usr/bin/env bash
# ==============================================================================
# Full Benchmark: Suffix Visual Cut on VDC (50 samples x 5 retention levels)
# Models:
#   1. DFlash Depth 3 (Qwen2.5-VL-3B-Instruct)
#   2. MSD Depth 3    (Qwen2.5-VL-3B-Instruct)
#   3. MSD 7B Gốc     (Qwen2-VL-7B-Instruct + MSD-Qwen2VL-7B-Instruct)
# Retention Levels: 100%, 50%, 25%, 5%, 0%
# Strategy: Suffix Retention (keep the last K% visual tokens)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_BASE="${ROOT_DIR}/results/full_suffix_retention"
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${OUTPUT_BASE}/dflash" "${OUTPUT_BASE}/msd_3b" "${OUTPUT_BASE}/msd_7b" "${LOG_DIR}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

echo "========================================================================"
echo "STARTING FULL BENCHMARK: SUFFIX VISUAL CUT ON VDC (50 SAMPLES x 5 LEVELS)"
echo "Output Directory: ${OUTPUT_BASE}"
echo "Time: $(date)"
echo "========================================================================"

# ------------------------------------------------------------------------------
# Stage 1: DFlash Depth 3 (Qwen2.5-VL-3B) - ALREADY COMPLETED (250/250)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 1/3] DFlash Depth 3: Already completed (250/250 rows). Skipping."

# ------------------------------------------------------------------------------
# Stage 2: MSD Depth 3 (Qwen2.5-VL-3B) - ALREADY COMPLETED (250/250)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 2/3] MSD Depth 3: Already completed (250/250 rows). Skipping."

# ------------------------------------------------------------------------------
# Stage 3: MSD 7B Gốc (Qwen2-VL-7B)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 3/3] Running MSD 7B Gốc on VDC (50 samples)..."
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/path/to/local-resource \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${PYTHON_BIN}" -u -m src.analyze.Validate_Sparrow_hypothesises.run_msd \
    --base-model /path/to/local-resource \
    --msd-model /path/to/local-resource \
    --manifest dataset/VideoDetailCaption/test.jsonl \
    --dataset-root dataset/VideoDetailCaption \
    --condition retention \
    --selection suffix \
    --retention-percentages 100 50 25 5 0 \
    --max-frames 16 \
    --max-new-tokens 128 \
    --device-map cuda \
    --output "${OUTPUT_BASE}/msd_7b/msd_retention_suffix_vdc.jsonl" \
    2>&1 | tee "${LOG_DIR}/msd7b_suffix_vdc.log"

echo ">>> [Stage 3/3] MSD 7B Gốc Completed!"

echo ""
echo "========================================================================"
echo "ALL 3 BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "Finished at: $(date)"
echo "Results:"
echo "  1. DFlash: ${OUTPUT_BASE}/dflash/vdc/dflash_retention_suffix_vdc.jsonl"
echo "  2. MSD 3B: ${OUTPUT_BASE}/msd_3b/depth3/vdc/msd_retention_suffix_vdc.jsonl"
echo "  3. MSD 7B: ${OUTPUT_BASE}/msd_7b/msd_retention_suffix_vdc.jsonl"
echo "========================================================================"
