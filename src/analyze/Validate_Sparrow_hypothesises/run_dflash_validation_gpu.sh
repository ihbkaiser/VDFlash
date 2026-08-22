#!/usr/bin/env bash
set -euo pipefail

# DFlash-only Sparrow/MSD hypothesis validation.  The existing MSD launchers
# are intentionally not sourced or modified here.
PYTHON_BIN="${PYTHON_BIN:-.venv-msd/bin/python}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
CHECKPOINT="${CHECKPOINT:-dataset/qwen25vl-3b-dflash-llava68k-latest/training_state.pt}"
DRAFT_CONFIG="${DRAFT_CONFIG:-src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json}"
MANIFEST="${MANIFEST:-dataset/VideoDetailCaption/test.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-dataset/VideoDetailCaption}"
CALIBRATION_INPUT="${CALIBRATION_INPUT:-dataset/VideoDetailCaption/calibration.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sparrow_validation_dflash_qwen25vl3b_$(date +%F)}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"
# The sweep changes video-token shapes from row to row.  Expandable CUDA
# segments reduce allocator fragmentation across those heterogeneous jobs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

exec "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments all \
  --target-model "$TARGET_MODEL" \
  --checkpoint "$CHECKPOINT" \
  --draft-config "$DRAFT_CONFIG" \
  --manifest "$MANIFEST" \
  --video-root "$VIDEO_ROOT" \
  --calibration-input "$CALIBRATION_INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  "${LIMIT_ARGS[@]}" \
  "$@"
