#!/usr/bin/env bash
set -euo pipefail

SPECFORGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
SOURCE_JSONL=${SOURCE_JSONL:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
IMAGE_ARCHIVE=${IMAGE_ARCHIVE:-}
MATERIALIZED_IMAGE_ROOT=${MATERIALIZED_IMAGE_ROOT:-}
PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-}
VOCAB_MAPPING_PATH=${VOCAB_MAPPING_PATH:-}

case "$MODEL_SIZE" in
  3b)
    DEFAULT_TARGET_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
    DEFAULT_ARTIFACT_ROOT="$SPECFORGE_DIR/artifacts/qwen25vl_eagle3_phase2_3b"
    DEFAULT_RUN_CONFIG="$SPECFORGE_DIR/examples/configs/qwen2.5-vl-3b-eagle3-caption-offline.yaml"
    DEFAULT_DRAFT_CONFIG="$SPECFORGE_DIR/configs/qwen2.5-vl-3b-eagle3.json"
    DEFAULT_RUN_ID=qwen25vl-3b-eagle3-caption-offline
    DEFAULT_PHASE1_ROOT="$SPECFORGE_DIR/artifacts/qwen25vl_eagle3_text"
    ;;
  7b)
    DEFAULT_TARGET_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
    DEFAULT_ARTIFACT_ROOT="$SPECFORGE_DIR/artifacts/qwen25vl_eagle3_phase2_7b"
    DEFAULT_RUN_CONFIG="$SPECFORGE_DIR/examples/configs/qwen2.5-vl-7b-eagle3-caption-offline.yaml"
    DEFAULT_DRAFT_CONFIG="$SPECFORGE_DIR/configs/qwen2.5-vl-7b-eagle3.json"
    DEFAULT_RUN_ID=qwen25vl-7b-eagle3-caption-offline
    DEFAULT_PHASE1_ROOT="$SPECFORGE_DIR/artifacts/qwen25vl_eagle3_text_7b"
    ;;
  *)
    echo 'SPECFORGE_MODEL_SIZE must be 3b or 7b' >&2
    exit 2
    ;;
esac

TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-$DEFAULT_TARGET_MODEL_PATH}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$DEFAULT_ARTIFACT_ROOT}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ARTIFACT_ROOT/outputs"}
RUN_CONFIG=${SPECFORGE_CONFIG:-$DEFAULT_RUN_CONFIG}

GPU_COUNT=${SPECFORGE_GPUS:-}
PHASE=all
RESUME=0
OVERWRITE_DATA=0
EXPECTED_RECORDS=${SPECFORGE_NUM_SAMPLES:-68000}
CAPTURE_BATCH_SIZE=${SPECFORGE_CAPTURE_BATCH_SIZE:-1}
CAPTURE_WORKERS=${SPECFORGE_CAPTURE_WORKERS:-4}
CAPTURE_QUEUE=${SPECFORGE_CAPTURE_QUEUE:-32}
CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-4}
CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-64}

DRAFT_CONFIG=${DRAFT_CONFIG:-$DEFAULT_DRAFT_CONFIG}
MANIFEST_PATH="$ARTIFACT_ROOT/shared/qwen25vl_caption_manifest.jsonl"
FEATURE_ROOT="$ARTIFACT_ROOT/hidden_states"
RUN_ID=${SPECFORGE_RUN_ID:-$DEFAULT_RUN_ID}
RUN_OUTPUT="$OUTPUT_ROOT/$RUN_ID"

usage() {
  printf '%s\n' \
    "Train standalone Qwen2.5-VL $MODEL_SIZE EAGLE3 Phase 2 on image captions." \
    '' \
    'Usage: bash train_qwen25vl_eagle3_captioning.sh [options]' \
    '' \
    'Options:' \
    '  --config FILE        YAML recipe.' \
    '  --gpus N             Capture/trainer process count.' \
    '  --phase VALUE        data|capture|train|all.' \
    '  --phase1-checkpoint  Phase 1 EAGLE3 checkpoint for fresh training.' \
    '  --source-jsonl FILE  Flat image-caption JSONL source.' \
    '  --image-root DIR     Image directory.' \
    '  --image-archive FILE Optional zip/tar image archive.' \
    '  --resume             Resume existing capture/checkpoint state.' \
    '  --overwrite-data     Rebuild the normalized caption manifest.' \
    '  -h, --help           Show this help.' \
    '' \
    'Environment: SPECFORGE_MODEL_SIZE SOURCE_JSONL IMAGE_ROOT IMAGE_ARCHIVE TARGET_MODEL_PATH' \
    '  PHASE1_CHECKPOINT VOCAB_MAPPING_PATH ARTIFACT_ROOT OUTPUT_ROOT' \
    '  SPECFORGE_CONFIG SPECFORGE_GPUS SPECFORGE_NUM_SAMPLES SPECFORGE_RUN_ID' \
    '  SPECFORGE_CAPTURE_BATCH_SIZE SPECFORGE_CAPTURE_WORKERS' \
    '  SPECFORGE_CAPTURE_QUEUE SPECFORGE_CAPTURE_IO_THREADS' \
    '  SPECFORGE_CAPTURE_IO_QUEUE'
}

while (($#)); do
  case "$1" in
    --config)
      RUN_CONFIG=${2:?'--config requires a file'}
      shift 2
      ;;
    --gpus)
      GPU_COUNT=${2:?'--gpus requires a value'}
      shift 2
      ;;
    --phase)
      PHASE=${2:?'--phase requires a value'}
      shift 2
      ;;
    --phase1-checkpoint)
      PHASE1_CHECKPOINT=${2:?'--phase1-checkpoint requires a path'}
      shift 2
      ;;
    --source-jsonl)
      SOURCE_JSONL=${2:?'--source-jsonl requires a file'}
      shift 2
      ;;
    --image-root)
      IMAGE_ROOT=${2:?'--image-root requires a directory'}
      shift 2
      ;;
    --image-archive)
      IMAGE_ARCHIVE=${2:?'--image-archive requires a file'}
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --overwrite-data)
      OVERWRITE_DATA=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$PHASE" in
  data|capture|train|all) ;;
  *) echo '--phase must be data, capture, train, or all' >&2; exit 2 ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$RUN_CONFIG" != /* ]]; then
  RUN_CONFIG="$(cd -- "$(dirname -- "$RUN_CONFIG")" && pwd)/$(basename -- "$RUN_CONFIG")"
fi

config_value() {
  "$PYTHON_BIN" - "$RUN_CONFIG" "$1" <<'PY'
import sys
import yaml

path, dotted_key = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for key in dotted_key.split("."):
    value = value[key]
if value is None:
    raise SystemExit(f"config value is null: {dotted_key}")
print(value)
PY
}

CONFIG_GPU_COUNT=$(config_value deployment.trainer.nproc_per_node)
GPU_COUNT=${GPU_COUNT:-$CONFIG_GPU_COUNT}
MAX_LENGTH=$(config_value data.max_length)
MICRO_BATCH_SIZE=$(config_value training.batch_size)
ACCUMULATION_STEPS=$(config_value training.accumulation_steps)
SGLANG_MEM_FRACTION_STATIC=$(config_value model.sglang_mem_fraction_static)
SGLANG_ATTENTION_BACKEND=$(config_value model.sglang_attention_backend)

for value_name in GPU_COUNT EXPECTED_RECORDS MICRO_BATCH_SIZE ACCUMULATION_STEPS CAPTURE_BATCH_SIZE MAX_LENGTH CAPTURE_WORKERS CAPTURE_QUEUE CAPTURE_IO_THREADS CAPTURE_IO_QUEUE; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if ! "$PYTHON_BIN" -c 'import sys; value=float(sys.argv[1]); raise SystemExit(0 if 0 < value <= 1 else 1)' "$SGLANG_MEM_FRACTION_STATIC"; then
  echo 'model.sglang_mem_fraction_static must be in (0, 1]' >&2
  exit 2
fi

if [[ -z "$MATERIALIZED_IMAGE_ROOT" ]]; then
  MATERIALIZED_IMAGE_ROOT="$ARTIFACT_ROOT/images"
fi
if [[ -n "$IMAGE_ARCHIVE" && -z "$IMAGE_ROOT" ]]; then
  IMAGE_ROOT="$MATERIALIZED_IMAGE_ROOT"
fi
if [[ -z "$VOCAB_MAPPING_PATH" ]]; then
  VOCAB_MAPPING_PATH="$DEFAULT_PHASE1_ROOT/hidden_states/vocab_mapping/vocab_mapping.pt"
fi
export PYTHONPATH="$SPECFORGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT_ROOT/shared" "$OUTPUT_ROOT"

jsonl_count() {
  awk 'NF { count += 1 } END { print count + 0 }' "$1"
}

require_manifest_count() {
  local count
  count=$(jsonl_count "$MANIFEST_PATH")
  if ((count != EXPECTED_RECORDS)); then
    echo "Caption manifest has $count records; expected $EXPECTED_RECORDS: $MANIFEST_PATH" >&2
    exit 1
  fi
}

prepare_data() {
  if [[ -s "$MANIFEST_PATH" ]] && ((!OVERWRITE_DATA)); then
    echo "[data] reusing $MANIFEST_PATH"
    require_manifest_count
    return
  fi
  if [[ -z "$SOURCE_JSONL" || ! -f "$SOURCE_JSONL" ]]; then
    echo "SOURCE_JSONL must point to an existing image-caption JSONL file: $SOURCE_JSONL" >&2
    exit 1
  fi
  if [[ -z "$IMAGE_ROOT" && -z "$IMAGE_ARCHIVE" ]]; then
    echo 'set IMAGE_ROOT or IMAGE_ARCHIVE for image-caption data' >&2
    exit 1
  fi
  local image_args=()
  if [[ -n "$IMAGE_ARCHIVE" ]]; then
    IMAGE_ROOT="$MATERIALIZED_IMAGE_ROOT"
    image_args+=(--image-archive "$IMAGE_ARCHIVE" --materialized-image-root "$IMAGE_ROOT")
  else
    image_args+=(--image-root "$IMAGE_ROOT")
  fi
  "$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_qwen25vl_caption_manifest.py" \
    --input "$SOURCE_JSONL" \
    --output "$MANIFEST_PATH" \
    --expected-records "$EXPECTED_RECORDS" \
    "${image_args[@]}"
  echo "[data] wrote normalized manifest to $MANIFEST_PATH"
}

feature_count() {
  if [[ ! -d "$FEATURE_ROOT" ]]; then
    echo 0
    return
  fi
  find "$FEATURE_ROOT" -type f \( -name '*.ckpt' -o -name '*.ckpt.gz' \) -print | wc -l
}

check_target_path() {
  if [[ ! -e "$TARGET_MODEL_PATH" && ( "$TARGET_MODEL_PATH" == /* || "$TARGET_MODEL_PATH" == ./* || "$TARGET_MODEL_PATH" == ../* ) ]]; then
    echo "Target model path not found: $TARGET_MODEL_PATH" >&2
    exit 1
  fi
}

capture_features() {
  check_target_path
  if [[ -z "$IMAGE_ROOT" || ! -d "$IMAGE_ROOT" ]]; then
    echo "IMAGE_ROOT must point to a materialized image directory: $IMAGE_ROOT" >&2
    exit 1
  fi
  local existing_features
  existing_features=$(feature_count)
  if ((existing_features > 0 && !RESUME)); then
    echo "Offline features already exist at $FEATURE_ROOT; pass --resume or use a new ARTIFACT_ROOT" >&2
    exit 1
  fi
  mkdir -p "$FEATURE_ROOT"
  local compress_args=()
  if [[ ${SPECFORGE_COMPRESS:-0} == 1 ]]; then
    compress_args+=(--compress)
  fi
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$GPU_COUNT" \
    "$SPECFORGE_DIR/scripts/prepare_qwen25vl_caption_hidden_states.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$DRAFT_CONFIG" \
    --manifest "$MANIFEST_PATH" \
    --image-root "$IMAGE_ROOT" \
    --output-path "$FEATURE_ROOT" \
    --max-length "$MAX_LENGTH" \
    --expected-records "$EXPECTED_RECORDS" \
    --tp-size 1 \
    --batch-size "$CAPTURE_BATCH_SIZE" \
    --num-preprocess-workers "$CAPTURE_WORKERS" \
    --preprocess-queue-size "$CAPTURE_QUEUE" \
    --num-io-threads "$CAPTURE_IO_THREADS" \
    --io-queue-size "$CAPTURE_IO_QUEUE" \
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    "${compress_args[@]}"
}

train_draft() {
  check_target_path
  local count latest_link resume_args=()
  count=$(feature_count)
  if ((count == 0)); then
    echo "No offline features found at $FEATURE_ROOT; run --phase capture first" >&2
    exit 1
  fi
  latest_link="$RUN_OUTPUT/$RUN_ID-latest"
  if ((RESUME)); then
    if [[ -e "$latest_link" ]]; then
      resume_args+=("training.resume_from=$RUN_OUTPUT")
    elif [[ -z "$PHASE1_CHECKPOINT" ]]; then
      echo 'PHASE1_CHECKPOINT is required when --resume has no existing Phase 2 checkpoint' >&2
      exit 1
    fi
  elif [[ -e "$latest_link" ]]; then
    echo "Checkpoint already exists at $latest_link; pass --resume or use a new ARTIFACT_ROOT" >&2
    exit 1
  fi
  if ((${#resume_args[@]} == 0)) && [[ -z "$PHASE1_CHECKPOINT" ]]; then
    echo 'PHASE1_CHECKPOINT is required for a fresh Phase 2 training run' >&2
    exit 1
  fi
  local warm_start_args=()
  if [[ -n "$PHASE1_CHECKPOINT" && ${#resume_args[@]} -eq 0 ]]; then
    warm_start_args+=("model.draft_checkpoint_path=$PHASE1_CHECKPOINT")
  fi
  (
    cd "$SPECFORGE_DIR"
    "$PYTHON_BIN" -m specforge.cli train \
      --config "$RUN_CONFIG" \
      "model.target_model_path=$TARGET_MODEL_PATH" \
      "model.vocab_mapping_path=$VOCAB_MAPPING_PATH" \
      "data.hidden_states_path=$FEATURE_ROOT" \
      "data.cache_dir=$ARTIFACT_ROOT/cache" \
      "deployment.trainer.nproc_per_node=$GPU_COUNT" \
      "run_id=$RUN_ID" \
      "output_dir=$RUN_OUTPUT" \
      "${warm_start_args[@]}" \
      "${resume_args[@]}"
  )
}

if [[ "$PHASE" == data || "$PHASE" == all ]]; then
  prepare_data
elif [[ ! -s "$MANIFEST_PATH" ]]; then
  echo "Caption manifest not found: $MANIFEST_PATH; run --phase data first" >&2
  exit 1
else
  require_manifest_count
fi

if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
  capture_features
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  train_draft
fi
