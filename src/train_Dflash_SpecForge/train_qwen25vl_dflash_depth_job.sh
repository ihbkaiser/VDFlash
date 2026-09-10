#!/usr/bin/env bash
set -euo pipefail

# Train one DFlash depth end-to-end: Phase 1 -> Phase 2.
# Existing hidden-state caches are intentionally reused; this script does not
# recapture them. Multiple depth jobs can share the feature roots safely.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
DEPTH=${SPECFORGE_DFLASH_DEPTH:-}
GPU_IDS=${SPECFORGE_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}
PHASE1_ARTIFACT_ROOT=${SPECFORGE_PHASE1_ARTIFACT_ROOT:-"$ROOT_DIR/artifacts/qwen25vl_dflash_sharegpt68k"}
PHASE2_ARTIFACT_ROOT=${SPECFORGE_PHASE2_ARTIFACT_ROOT:-"$ROOT_DIR/artifacts/qwen25vl_${MODEL_SIZE}_dflash_llava68k"}
BASE_OUTPUT_ROOT=${SPECFORGE_OUTPUT_ROOT:-"$ROOT_DIR/outputs/dflash_depth_jobs"}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-}
RESUME=0

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dflash_depth_job.sh --depth N --gpu-ids IDS [--resume]

Train exactly one DFlash depth through Phase 1 and then Phase 2.
Hidden-state caches are reused and must already exist.

Options:
  --depth N             DFlash decoder depth, e.g. 1 or 2.
  --gpu-ids IDS        Physical GPU IDs, e.g. 0,1 or 2,3.
  --resume             Resume an existing depth checkpoint.

Environment:
  SPECFORGE_MODEL_SIZE=3b|7b (default: 3b)
  TARGET_MODEL_PATH=/models/qwen25-vl-3b (required)
  SPECFORGE_PHASE1_ARTIFACT_ROOT=old Phase 1 artifact root
  SPECFORGE_PHASE2_ARTIFACT_ROOT=old Phase 2 artifact root
  SPECFORGE_OUTPUT_ROOT=base output root; depthN is appended
  SPECFORGE_GPU_IDS=IDS (alternative to --gpu-ids)
  SPECFORGE_DFLASH_DEPTH=N (alternative to --depth)

The script sets CUDA_VISIBLE_DEVICES=IDS and SPECFORGE_GPUS to the number of
IDs. Each depth gets a unique run/config suffix and a private data cache.
EOF
}

while (($#)); do
  case "$1" in
    --depth)
      DEPTH=${2:?--depth requires a value}
      shift 2
      ;;
    --gpu-ids|--gpus)
      GPU_IDS=${2:?--gpu-ids requires a value}
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODEL_SIZE" in
  3b|7b) ;;
  *) echo "SPECFORGE_MODEL_SIZE must be 3b or 7b" >&2; exit 2 ;;
esac
if [[ ! "$DEPTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "--depth must be a positive integer" >&2
  exit 2
fi
GPU_IDS=${GPU_IDS//[[:space:]]/}
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpu-ids must be comma-separated physical GPU IDs, got: $GPU_IDS" >&2
  exit 2
fi
if [[ -z "$TARGET_MODEL_PATH" ]]; then
  echo "TARGET_MODEL_PATH must be set" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ID_ARRAY[@]}
RUN_SUFFIX=${SPECFORGE_RUN_SUFFIX:-"-h32-depth${DEPTH}"}
DRAFT_CONFIG_SUFFIX=${SPECFORGE_DRAFT_CONFIG_SUFFIX:-"-h32-depth${DEPTH}"}
JOB_OUTPUT_ROOT="$BASE_OUTPUT_ROOT/depth${DEPTH}"
PHASE1_OUTPUT_ROOT="$JOB_OUTPUT_ROOT/phase1"
PHASE2_OUTPUT_ROOT="$JOB_OUTPUT_ROOT/phase2"
JOB_CACHE_ROOT="$JOB_OUTPUT_ROOT/data_cache"

case "$MODEL_SIZE" in
  3b) export MODEL_3B="${MODEL_3B:-$TARGET_MODEL_PATH}" ;;
  7b) export MODEL_7B="${MODEL_7B:-$TARGET_MODEL_PATH}" ;;
esac
export PYTHON_BIN SPECFORGE_MODEL_SIZE TARGET_MODEL_PATH
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export SPECFORGE_GPUS="$GPU_COUNT"
export SPECFORGE_DFLASH_DEPTH="$DEPTH"
export SPECFORGE_RUN_SUFFIX="$RUN_SUFFIX"
export SPECFORGE_DRAFT_CONFIG_SUFFIX="$DRAFT_CONFIG_SUFFIX"

RESUME_ARGS=()
if ((RESUME)); then
  RESUME_ARGS+=(--resume)
fi

phase1_run_id="qwen25vl-${MODEL_SIZE}-dflash-sharegpt68k${RUN_SUFFIX}"
phase1_checkpoint="$PHASE1_OUTPUT_ROOT/$phase1_run_id/$phase1_run_id-latest"

echo "[depth-job] depth=$DEPTH GPUs=$GPU_IDS (count=$GPU_COUNT)"
echo "[depth-job] Phase 1: reuse $PHASE1_ARTIFACT_ROOT"
ARTIFACT_ROOT="$PHASE1_ARTIFACT_ROOT" \
OUTPUT_ROOT="$PHASE1_OUTPUT_ROOT" \
SPECFORGE_DATA_CACHE_ROOT="$JOB_CACHE_ROOT/phase1" \
  bash "$ROOT_DIR/train_qwen25vl_dflash_sharegpt_68k.sh" \
  --models "$MODEL_SIZE" --phase train "${RESUME_ARGS[@]}"

if [[ ! -e "$phase1_checkpoint" ]]; then
  echo "Phase 1 checkpoint not found after training: $phase1_checkpoint" >&2
  exit 1
fi

echo "[depth-job] Phase 2: reuse $PHASE2_ARTIFACT_ROOT"
ARTIFACT_ROOT="$PHASE2_ARTIFACT_ROOT" \
OUTPUT_ROOT="$PHASE2_OUTPUT_ROOT" \
SPECFORGE_DATA_CACHE_ROOT="$JOB_CACHE_ROOT/phase2" \
PHASE1_CHECKPOINT="$phase1_checkpoint" \
  bash "$ROOT_DIR/train_qwen25vl_dflash_llava_68k.sh" \
  --phase train "${RESUME_ARGS[@]}"

echo "[depth-job] completed depth=$DEPTH; Phase 1 checkpoint=$phase1_checkpoint"
