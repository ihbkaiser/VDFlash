#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Load a trusted shell-style environment file before applying defaults below.
# Explicit command-line options are parsed afterwards and take precedence.
ENV_FILE=${SPECFORGE_ENV_FILE:-}
ARGS=("$@")
for ((arg_index = 0; arg_index < ${#ARGS[@]}; arg_index++)); do
  case "${ARGS[arg_index]}" in
    --env-file)
      arg_index=$((arg_index + 1))
      if ((arg_index >= ${#ARGS[@]})); then
        echo "--env-file requires a path" >&2
        exit 2
      fi
      ENV_FILE=${ARGS[arg_index]}
      ;;
    --env-file=*)
      ENV_FILE=${ARGS[arg_index]#--env-file=}
      ;;
  esac
done
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "environment file not found: $ENV_FILE" >&2
    exit 2
  fi
  # The env file is intentionally shell syntax so paths may be quoted.
  # Only source files the user explicitly selected.
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
# Keep the local SpecForge package importable when this launcher is invoked
# from the repository root or from a scheduler working directory.
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_JSONL=${SOURCE_JSONL:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
IMAGE_ARCHIVE=${IMAGE_ARCHIVE:-}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-}
PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$ROOT_DIR/artifacts/qwen25vl_dflash_llava68k"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs"}
GPU_COUNT=${SPECFORGE_GPUS:-4}
EXPECTED_RECORDS=${SPECFORGE_NUM_SAMPLES:-68000}
MAX_LENGTH=${SPECFORGE_MAX_LENGTH:-3072}
GLOBAL_BATCH_SIZE=${SPECFORGE_GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${SPECFORGE_MICRO_BATCH_SIZE:-1}
PHASE=${PHASE:-all}
RESUME=0
COMPRESS=${SPECFORGE_COMPRESS:-0}

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dflash_llava_68k.sh [options]

Required environment:
  SOURCE_JSONL       complete flat LLaVA caption JSONL on the Phase 2 server
  TARGET_MODEL_PATH  same Qwen2.5-VL-3B target used by Phase 1
  PHASE1_CHECKPOINT  Phase 1 DFlash draft checkpoint for a new Phase 2 run
  IMAGE_ROOT         extracted LLaVA image hierarchy, or set IMAGE_ARCHIVE

Options:
  --env-file FILE
  --env-file=FILE
  --phase data|capture|train|infer|all
  --gpus N
  --resume
  --help
EOF
}

while (($#)); do
  case "$1" in
    --env-file) shift 2 ;;
    --env-file=*) shift ;;
    --phase) PHASE=${2:?--phase requires a value}; shift 2 ;;
    --gpus) GPU_COUNT=${2:?--gpus requires a value}; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PHASE" in
  data|capture|train|infer|all) ;;
  *) echo "invalid phase: $PHASE" >&2; exit 2 ;;
esac
if (( GPU_COUNT < 1 || EXPECTED_RECORDS < 1 || MAX_LENGTH < 32 || GLOBAL_BATCH_SIZE < 1 || MICRO_BATCH_SIZE < 1 )); then
  echo "invalid numeric configuration" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (GPU_COUNT * MICRO_BATCH_SIZE) != 0 )); then
  echo "global batch must be divisible by GPUs * micro batch" >&2
  exit 2
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (GPU_COUNT * MICRO_BATCH_SIZE)))

MANIFEST="$ARTIFACT_ROOT/manifest.jsonl"
FEATURE_ROOT="$ARTIFACT_ROOT/hidden_states"
IMAGE_STAGE_ROOT="$ARTIFACT_ROOT/images"
CONFIG="$ROOT_DIR/examples/configs/qwen2.5-vl-3b-dflash-llava68k-offline.yaml"
RUN_OUTPUT="$OUTPUT_ROOT/qwen25vl-3b-dflash-llava68k"

require_value() {
  local name=$1 value=${!1:-}
  if [[ -z "$value" ]]; then
    echo "$name must be set for phase $PHASE" >&2
    exit 2
  fi
}

if [[ "$PHASE" == data || "$PHASE" == all ]]; then
  require_value SOURCE_JSONL
  if [[ -n "$IMAGE_ROOT" && -n "$IMAGE_ARCHIVE" ]]; then
    echo "set only one of IMAGE_ROOT or IMAGE_ARCHIVE" >&2
    exit 2
  fi
  if [[ -z "$IMAGE_ROOT" && -z "$IMAGE_ARCHIVE" ]]; then
    echo "set IMAGE_ROOT or IMAGE_ARCHIVE" >&2
    exit 2
  fi
  mkdir -p "$ARTIFACT_ROOT"
  image_args=(--expected-records "$EXPECTED_RECORDS")
  if [[ -n "$IMAGE_ROOT" ]]; then
    image_args+=(--image-root "$IMAGE_ROOT")
  else
    image_args+=(--image-archive "$IMAGE_ARCHIVE")
  fi
  "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_llava_caption_manifest.py" \
    --input "$SOURCE_JSONL" --output "$MANIFEST" "${image_args[@]}"
  if [[ -n "$IMAGE_ARCHIVE" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/materialize_llava_images.py" \
      --manifest "$MANIFEST" --archive "$IMAGE_ARCHIVE" --output-root "$IMAGE_STAGE_ROOT"
    IMAGE_ROOT="$IMAGE_STAGE_ROOT"
  fi
fi

if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
  require_value TARGET_MODEL_PATH
  [[ -f "$MANIFEST" ]] || { echo "manifest missing: $MANIFEST" >&2; exit 1; }
  if [[ -z "$IMAGE_ROOT" && -n "$IMAGE_ARCHIVE" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/materialize_llava_images.py" \
      --manifest "$MANIFEST" --archive "$IMAGE_ARCHIVE" --output-root "$IMAGE_STAGE_ROOT"
    IMAGE_ROOT="$IMAGE_STAGE_ROOT"
  fi
  require_value IMAGE_ROOT
  mkdir -p "$FEATURE_ROOT"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/preflight_llava_caption.py" \
    --manifest "$MANIFEST" --image-root "$IMAGE_ROOT" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$ROOT_DIR/configs/qwen2.5-vl-3b-dflash.json" \
    --output-path "$FEATURE_ROOT" --max-length "$MAX_LENGTH" \
    --expected-records "$EXPECTED_RECORDS"
  compress_args=()
  if [[ "$COMPRESS" == 1 ]]; then
    compress_args+=(--compress)
  fi
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    "$ROOT_DIR/scripts/prepare_llava_caption_hidden_states.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$ROOT_DIR/configs/qwen2.5-vl-3b-dflash.json" \
    --manifest "$MANIFEST" --image-root "$IMAGE_ROOT" \
    --output-path "$FEATURE_ROOT" --max-length "$MAX_LENGTH" \
    --expected-records "$EXPECTED_RECORDS" --tp-size 1 "${compress_args[@]}"
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  require_value TARGET_MODEL_PATH
  [[ -d "$FEATURE_ROOT" ]] || { echo "feature directory missing: $FEATURE_ROOT" >&2; exit 1; }
  train_args=(
    "model.target_model_path=$TARGET_MODEL_PATH"
    "model.draft_model_config=$ROOT_DIR/configs/qwen2.5-vl-3b-dflash.json"
    "model.input_modality=qwen2_5_vl"
    "data.hidden_states_path=$FEATURE_ROOT"
    "data.max_length=$MAX_LENGTH"
    "training.batch_size=$MICRO_BATCH_SIZE"
    "training.accumulation_steps=$ACCUMULATION_STEPS"
    "deployment.trainer.nproc_per_node=$GPU_COUNT"
    "output_dir=$RUN_OUTPUT"
    "run_id=qwen25vl-3b-dflash-llava68k"
  )
  latest="$RUN_OUTPUT/qwen25vl-3b-dflash-llava68k-latest"
  if (( RESUME == 1 )) && [[ -e "$latest" ]]; then
    train_args+=("training.resume_from=$RUN_OUTPUT")
  else
    require_value PHASE1_CHECKPOINT
    train_args+=("model.draft_checkpoint_path=$PHASE1_CHECKPOINT")
  fi
  (cd "$ROOT_DIR" && "$PYTHON_BIN" -m specforge.cli train --config "$CONFIG" "${train_args[@]}")
fi

if [[ "$PHASE" == infer ]]; then
  require_value TARGET_MODEL_PATH
  [[ -f "$MANIFEST" ]] || { echo "manifest missing: $MANIFEST" >&2; exit 1; }
  if [[ -z "$IMAGE_ROOT" && -n "$IMAGE_ARCHIVE" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/materialize_llava_images.py" \
      --manifest "$MANIFEST" --archive "$IMAGE_ARCHIVE" --output-root "$IMAGE_STAGE_ROOT"
    IMAGE_ROOT="$IMAGE_STAGE_ROOT"
  fi
  require_value IMAGE_ROOT
  "$PYTHON_BIN" "$ROOT_DIR/scripts/infer_qwen25vl_dflash.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$ROOT_DIR/configs/qwen2.5-vl-3b-dflash.json" \
    --checkpoint "$RUN_OUTPUT" \
    --manifest "$MANIFEST" --image-root "$IMAGE_ROOT"
fi
