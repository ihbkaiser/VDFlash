#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_PATH="${OUTPUT_PATH:-results/qwen35_dflash/vdc50_t4_16frames.jsonl}"
LOG_PATH="${LOG_PATH:-results/qwen35_dflash/vdc50_t4_16frames.log}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3.5-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}"
NUM_FRAMES="${NUM_FRAMES:-16}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-4194304}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$LOG_PATH")"

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; run this script on the T4 host")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"GPU: {name} (compute capability {major}.{minor})")
PY

# Reduce CUDA allocator fragmentation during repeated video samples.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Keep a persistent log while still showing progress inside tmux.
exec > >(tee -a "$LOG_PATH") 2>&1

python -u -m src.analyze.Whether_they_are_appliable_for_dDrafter.qwen35_dflash_benchmark \
  --manifest dataset/VideoDetailCaption/subset_manifest.jsonl \
  --video-root dataset/VideoDetailCaption \
  --output "$OUTPUT_PATH" \
  --limit 50 \
  --num-frames "$NUM_FRAMES" \
  --video-max-pixels "$VIDEO_MAX_PIXELS" \
  --target-model "$TARGET_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --visual-percentages 100,50,12.5,0 \
  --verify-mode exact \
  --block-size 16 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0 \
  --device cuda:0 \
  --resume
