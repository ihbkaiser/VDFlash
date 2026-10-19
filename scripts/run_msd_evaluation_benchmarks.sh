#!/usr/bin/env bash
# ==============================================================================
# Pipeline đánh giá MSD (Multi-modal Speculative Decoding) trên 2 bộ dữ liệu:
# 1. VidHalluc: 100 mẫu (ACH, TSH, STH)
# 2. VideoChair: 100 mẫu (20 categories MSR-VTT)
# Qua 4 mức visual retention: 100%, 50%, 25%, 0%
# Thu thập đầy đủ acceptance_by_position, timing, lossless trace
# ==============================================================================
set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_BASE="results/msd_eval_20261007"
mkdir -p "${OUTPUT_BASE}/vidhalluc" "${OUTPUT_BASE}/videochair" "${OUTPUT_BASE}/logs"

echo "=== [1/2] Đang chạy đánh giá MSD trên VidHalluc (100 mẫu x 4 mức retention) ==="
.venv/bin/python -m src.analyze.Validate_Sparrow_hypothesises.run_msd \
  --manifest dataset/VidHalluc/test.jsonl \
  --dataset-root dataset/VidHalluc \
  --output "${OUTPUT_BASE}/vidhalluc/msd_retention_vidhalluc.jsonl" \
  --condition retention \
  --selection uniform \
  --retention-percentages 100 50 25 0 \
  --max-frames 32 \
  --device-map cuda \
  2>&1 | tee "${OUTPUT_BASE}/logs/msd_vidhalluc.log"

echo "=== [2/2] Đang chạy đánh giá MSD trên VideoChair (100 mẫu x 4 mức retention) ==="
.venv/bin/python -m src.analyze.Validate_Sparrow_hypothesises.run_msd \
  --manifest dataset/VideoChair/test.jsonl \
  --dataset-root dataset/VideoChair \
  --output "${OUTPUT_BASE}/videochair/msd_retention_videochair.jsonl" \
  --condition retention \
  --selection uniform \
  --retention-percentages 100 50 25 0 \
  --max-frames 32 \
  --device-map cuda \
  2>&1 | tee "${OUTPUT_BASE}/logs/msd_videochair.log"

echo "=== Hoàn thành toàn bộ quy trình đánh giá MSD trên VidHalluc và VideoChair! ==="
