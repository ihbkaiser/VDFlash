#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${DSPARK_PYTHON:-$REPO_ROOT/.venv/bin/python}"

HF_HOME="${HF_HOME:?Set HF_HOME to the shared Hugging Face cache}"
DSPARK_IMAGE="${DSPARK_IMAGE:?Set DSPARK_IMAGE to a local image file}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$DSPARK_IMAGE" ]]; then
  echo "Image does not exist: $DSPARK_IMAGE" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ "$DSPARK_IMAGE" != /* ]]; then
  DSPARK_IMAGE="$REPO_ROOT/$DSPARK_IMAGE"
fi
DSPARK_IMAGE="$(cd -- "$(dirname -- "$DSPARK_IMAGE")" && pwd)/$(basename -- "$DSPARK_IMAGE")"
DSPARK_OUTPUT_DIR="${DSPARK_OUTPUT_DIR:-$REPO_ROOT/results/dspark_inference/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$DSPARK_OUTPUT_DIR"
DSPARK_OUTPUT_DIR="$(cd -- "$DSPARK_OUTPUT_DIR" && pwd)"

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
DSPARK_SAMPLE_ID="${DSPARK_SAMPLE_ID:-$(basename -- "$DSPARK_IMAGE")}"
DSPARK_PROMPT="${DSPARK_PROMPT:-Describe this image.}"
DSPARK_MAX_NEW_TOKENS="${DSPARK_MAX_NEW_TOKENS:-64}"
DSPARK_SCHEDULER="${DSPARK_SCHEDULER:-fixed}"
DSPARK_CONFIDENCE_THRESHOLD="${DSPARK_CONFIDENCE_THRESHOLD:-0.5}"
DSPARK_CHECKPOINT="${DSPARK_CHECKPOINT:-}"
DSPARK_DRAFT_CONFIG="${DSPARK_DRAFT_CONFIG:-}"
DSPARK_THROUGHPUT_PROFILE="${DSPARK_THROUGHPUT_PROFILE:-}"

export HF_HOME REPO_ROOT DSPARK_IMAGE DSPARK_OUTPUT_DIR TARGET_MODEL_PATH
export DSPARK_SAMPLE_ID DSPARK_PROMPT DSPARK_MAX_NEW_TOKENS DSPARK_SCHEDULER
export DSPARK_CONFIDENCE_THRESHOLD DSPARK_CHECKPOINT DSPARK_DRAFT_CONFIG
export DSPARK_THROUGHPUT_PROFILE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

preflight_args=(
  --image "$DSPARK_IMAGE"
  --sample-id "$DSPARK_SAMPLE_ID"
  --prompt "$DSPARK_PROMPT"
  --target-model-path "$TARGET_MODEL_PATH"
  --max-new-tokens "$DSPARK_MAX_NEW_TOKENS"
  --scheduler "$DSPARK_SCHEDULER"
  --confidence-threshold "$DSPARK_CONFIDENCE_THRESHOLD"
  --preflight-only
)
if [[ -n "$DSPARK_CHECKPOINT" ]]; then
  preflight_args+=(--checkpoint "$DSPARK_CHECKPOINT")
fi
if [[ -n "$DSPARK_DRAFT_CONFIG" ]]; then
  preflight_args+=(--draft-config "$DSPARK_DRAFT_CONFIG")
fi
if [[ -n "$DSPARK_THROUGHPUT_PROFILE" ]]; then
  preflight_args+=(--throughput-profile "$DSPARK_THROUGHPUT_PROFILE")
fi

echo "[dspark-submit] running CPU preflight; output=$DSPARK_OUTPUT_DIR"
"$PYTHON_BIN" -u "$SCRIPT_DIR/infer_qwen25vl_dspark.py" \
  "${preflight_args[@]}" 2>&1 | tee "$DSPARK_OUTPUT_DIR/preflight.log"

variables="REPO_ROOT,HF_HOME,DSPARK_IMAGE,DSPARK_OUTPUT_DIR,TARGET_MODEL_PATH,DSPARK_SAMPLE_ID,DSPARK_PROMPT,DSPARK_MAX_NEW_TOKENS,DSPARK_SCHEDULER,DSPARK_CONFIDENCE_THRESHOLD,DSPARK_CHECKPOINT,DSPARK_DRAFT_CONFIG,DSPARK_THROUGHPUT_PROFILE"
qsub \
  -o "$DSPARK_OUTPUT_DIR/pbs.stdout" \
  -e "$DSPARK_OUTPUT_DIR/pbs.stderr" \
  -v "$variables" \
  "$SCRIPT_DIR/infer_qwen25vl_dspark_gadi.pbs"
