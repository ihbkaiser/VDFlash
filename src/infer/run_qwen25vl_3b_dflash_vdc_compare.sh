#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT="${OUTPUT:-results/infer/qwen25vl_3b_dflash_vdc_sample0.json}"
OUTPUT_DIR="${OUTPUT_DIR:-results/infer/qwen25vl_3b_dflash_vdc50}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
ALL_SAMPLES="${ALL_SAMPLES:-0}"
RESUME="${RESUME:-1}"
MANIFEST="${MANIFEST:-dataset/VideoDetailCaption/test.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-dataset/VideoDetailCaption}"
NUM_FRAMES="${NUM_FRAMES:-90}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-50176}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

export PYTHONPATH="${REPO_ROOT}/src/train_Dflash_SpecForge${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
  --target-model "$TARGET_MODEL" \
  --manifest "$MANIFEST" \
  --video-root "$VIDEO_ROOT" \
  --checkpoint dataset/qwen25vl-3b-dflash-llava68k-latest \
  --checkpoint dataset/qwen25vl-3b-dflash-sharegpt68k-latest \
  --draft-config src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json \
  --device "$DEVICE" \
  --dtype auto \
  --target-attention sdpa \
  --num-frames "$NUM_FRAMES" \
  --video-min-pixels "$VIDEO_MIN_PIXELS" \
  --video-max-pixels "$VIDEO_MAX_PIXELS" \
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ "$ALL_SAMPLES" == "1" ]]; then
  BATCH_ARGS=(--all-samples --output-dir "$OUTPUT_DIR")
  if [[ "$RESUME" == "1" ]]; then
    BATCH_ARGS+=(--resume)
  fi
  exec python -u -m src.infer.qwen25vl_dflash_compare \
    "${COMMON_ARGS[@]}" "${BATCH_ARGS[@]}"
fi

exec python -u -m src.infer.qwen25vl_dflash_compare \
  "${COMMON_ARGS[@]}" \
  --sample-index "$SAMPLE_INDEX" \
  --output "$OUTPUT"
