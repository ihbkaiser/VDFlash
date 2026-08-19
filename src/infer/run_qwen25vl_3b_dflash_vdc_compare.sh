#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-results/infer/qwen25vl_3b_dflash_vdc_sample0.json}"
NUM_FRAMES="${NUM_FRAMES:-8}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-50176}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

export PYTHONPATH="${REPO_ROOT}/src/train_Dflash_SpecForge${PYTHONPATH:+:${PYTHONPATH}}"

python -u -m src.infer.qwen25vl_dflash_compare \
  --target-model "$TARGET_MODEL" \
  --manifest dataset/VideoDetailCaption/test.jsonl \
  --video-root dataset/VideoDetailCaption \
  --sample-index 0 \
  --checkpoint dataset/qwen25vl-3b-dflash-llava68k-latest \
  --checkpoint dataset/qwen25vl-3b-dflash-sharegpt68k-latest \
  --draft-config src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --dtype auto \
  --target-attention sdpa \
  --num-frames "$NUM_FRAMES" \
  --video-min-pixels "$VIDEO_MIN_PIXELS" \
  --video-max-pixels "$VIDEO_MAX_PIXELS" \
  --max-new-tokens "$MAX_NEW_TOKENS"
