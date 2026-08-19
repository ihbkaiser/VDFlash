#!/usr/bin/env bash
set -euo pipefail

SPECFORGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
SOURCE_DATA=${SOURCE_DATA:-}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$SPECFORGE_DIR/artifacts/qwen25vl_eagle3_text"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ARTIFACT_ROOT/outputs"}
RUN_CONFIG=${SPECFORGE_CONFIG:-"$SPECFORGE_DIR/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml"}

GPU_COUNT=${SPECFORGE_GPUS:-}
PHASE=all
RESUME=0
OVERWRITE_DATA=0
EXPECTED_RECORDS=${SPECFORGE_NUM_SAMPLES:-68000}
CAPTURE_BATCH_SIZE=${SPECFORGE_CAPTURE_BATCH_SIZE:-32}
CAPTURE_WORKERS=${SPECFORGE_CAPTURE_WORKERS:-4}
CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-4}
CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-50}

DRAFT_CONFIG="$SPECFORGE_DIR/configs/qwen2.5-vl-3b-eagle3.json"
SPECFORGE_DATA="$ARTIFACT_ROOT/shared/sharegpt_train.jsonl"
FEATURE_ROOT="$ARTIFACT_ROOT/hidden_states"
RUN_ID=qwen25vl-3b-eagle3-text-offline
RUN_OUTPUT="$OUTPUT_ROOT/$RUN_ID"

usage() {
  printf '%s\n' \
    'Train the Phase 1 text-only Qwen2.5-VL 3B EAGLE3 draft with SpecForge.' \
    '' \
    'Usage:' \
    '  bash train_qwen25vl_eagle3_text.sh [options]' \
    '' \
    'Options:' \
    '  --config FILE        YAML recipe (default: SPECFORGE_CONFIG or the Phase 1 recipe).' \
    '  --gpus N             Capture/trainer process count (default: recipe value).' \
    '  --phase VALUE        data, capture, train, or all (default: all).' \
    '  --resume             Reuse existing capture files/checkpoints.' \
    '  --overwrite-data     Rebuild the converted ShareGPT JSONL.' \
    '  -h, --help           Show this help.' \
    '' \
    'Environment:' \
    '  PYTHON_BIN, SOURCE_DATA, TARGET_MODEL_PATH, ARTIFACT_ROOT, OUTPUT_ROOT' \
    '  SPECFORGE_CONFIG, SPECFORGE_GPUS, SPECFORGE_NUM_SAMPLES' \
    '  SPECFORGE_CAPTURE_BATCH_SIZE, SPECFORGE_CAPTURE_WORKERS' \
    '  SPECFORGE_CAPTURE_IO_THREADS, SPECFORGE_CAPTURE_IO_QUEUE' \
    '' \
    '  Training hyperparameters are controlled by the YAML recipe.'
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
DATASET_NUM_PROC=$(config_value data.build_dataset_num_proc)
DATALOADER_WORKERS=$(config_value data.dataloader_num_workers)
MICRO_BATCH_SIZE=$(config_value training.batch_size)
ACCUMULATION_STEPS=$(config_value training.accumulation_steps)
NUM_EPOCHS=$(config_value training.num_epochs)
SAVE_INTERVAL=$(config_value training.save_interval)
LOG_INTERVAL=$(config_value training.log_interval)
SGLANG_MEM_FRACTION_STATIC=$(config_value model.sglang_mem_fraction_static)
SGLANG_ATTENTION_BACKEND=$(config_value model.sglang_attention_backend)
SGLANG_DISABLE_RADIX_CACHE=$(config_value model.sglang_disable_radix_cache)
FSDP_SHARDING=$(config_value training.fsdp_sharding)

for value_name in GPU_COUNT EXPECTED_RECORDS MICRO_BATCH_SIZE ACCUMULATION_STEPS CAPTURE_BATCH_SIZE MAX_LENGTH NUM_EPOCHS DATASET_NUM_PROC CAPTURE_WORKERS CAPTURE_IO_THREADS CAPTURE_IO_QUEUE DATALOADER_WORKERS LOG_INTERVAL; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ ! "$SAVE_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo 'SPECFORGE_SAVE_INTERVAL must be a non-negative integer' >&2
  exit 2
fi
case "$FSDP_SHARDING" in
  NO_SHARD|SHARD_GRAD_OP|FULL_SHARD) ;;
  *) echo 'SPECFORGE_FSDP_SHARDING must be NO_SHARD, SHARD_GRAD_OP, or FULL_SHARD' >&2; exit 2 ;;
esac
if ! "$PYTHON_BIN" -c 'import sys; value=float(sys.argv[1]); raise SystemExit(0 if 0 < value <= 1 else 1)' "$SGLANG_MEM_FRACTION_STATIC"; then
  echo 'SPECFORGE_SGLANG_MEM_FRACTION_STATIC must be in (0, 1]' >&2
  exit 2
fi
if [[ -z "$SGLANG_ATTENTION_BACKEND" ]]; then
  echo 'model.sglang_attention_backend must be non-empty' >&2
  exit 2
fi
case "$SGLANG_DISABLE_RADIX_CACHE" in
  True|true|1|False|false|0) ;;
  *) echo 'model.sglang_disable_radix_cache must be boolean' >&2; exit 2 ;;
esac

GLOBAL_BATCH_SIZE=$((GPU_COUNT * MICRO_BATCH_SIZE * ACCUMULATION_STEPS))

export PYTHONPATH="$SPECFORGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT_ROOT/shared" "$OUTPUT_ROOT"
ARTIFACT_ROOT=$(cd "$ARTIFACT_ROOT" && pwd)
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
SPECFORGE_DATA="$ARTIFACT_ROOT/shared/sharegpt_train.jsonl"
FEATURE_ROOT="$ARTIFACT_ROOT/hidden_states"
RUN_OUTPUT="$OUTPUT_ROOT/$RUN_ID"

jsonl_count() {
  awk 'NF { count += 1 } END { print count + 0 }' "$1"
}

require_data_count() {
  local count
  count=$(jsonl_count "$SPECFORGE_DATA")
  if ((count != EXPECTED_RECORDS)); then
    echo "Converted dataset has $count records; expected $EXPECTED_RECORDS: $SPECFORGE_DATA" >&2
    echo 'Run the data phase with --overwrite-data.' >&2
    exit 1
  fi
}

prepare_data() {
  if [[ -s "$SPECFORGE_DATA" ]] && ((!OVERWRITE_DATA)); then
    echo "[data] reusing $SPECFORGE_DATA"
    require_data_count
    return
  fi
  if [[ -z "$SOURCE_DATA" || ! -f "$SOURCE_DATA" ]]; then
    echo "SOURCE_DATA must point to an existing ShareGPT JSON/JSONL file: $SOURCE_DATA" >&2
    exit 1
  fi

  local temporary_dir temporary_data count
  temporary_dir=$(mktemp -d "$ARTIFACT_ROOT/shared/.prepare-sharegpt.XXXXXX")
  temporary_data="$temporary_dir/sharegpt_train.jsonl"
  if ! "$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_data.py" \
      --dataset sharegpt \
      --data-path "$SOURCE_DATA" \
      --sample-size "$EXPECTED_RECORDS" \
      --output-path "$temporary_dir"; then
    rm -rf -- "$temporary_dir"
    return 1
  fi
  count=$(jsonl_count "$temporary_data")
  if ((count != EXPECTED_RECORDS)); then
    rm -rf -- "$temporary_dir"
    echo "ShareGPT conversion produced $count records; expected $EXPECTED_RECORDS" >&2
    exit 1
  fi
  mv -f -- "$temporary_data" "$SPECFORGE_DATA"
  rm -rf -- "$temporary_dir"
  echo "[data] wrote $count records to $SPECFORGE_DATA"
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
  local sglang_args=(
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND"
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
  )
  case "$SGLANG_DISABLE_RADIX_CACHE" in
    True|true|1) sglang_args+=(--sglang-disable-radix-cache) ;;
  esac
  echo "[capture] GPUs=$GPU_COUNT TP=1 DP=$GPU_COUNT batch/rank=$CAPTURE_BATCH_SIZE"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$GPU_COUNT" \
    "$SPECFORGE_DIR/scripts/prepare_hidden_states.py" \
    --strategy eagle3 \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$DRAFT_CONFIG" \
    --data-path "$SPECFORGE_DATA" \
    --output-path "$FEATURE_ROOT" \
    --cache-dir "$ARTIFACT_ROOT/cache" \
    --chat-template qwen \
    --max-length "$MAX_LENGTH" \
    --num-samples "$EXPECTED_RECORDS" \
    --build-dataset-num-proc "$DATASET_NUM_PROC" \
    --tp-size 1 \
    --batch-size "$CAPTURE_BATCH_SIZE" \
    --num-workers "$CAPTURE_WORKERS" \
    --num-io-threads "$CAPTURE_IO_THREADS" \
    --io-queue-size "$CAPTURE_IO_QUEUE" \
    --file-group-size 4096 \
    "${sglang_args[@]}" \
    "${compress_args[@]}"
  echo "[capture] feature records=$(feature_count)"
}

train_draft() {
  check_target_path
  local count latest_link resume_args=()
  count=$(feature_count)
  if ((count == 0)); then
    echo "No offline features found at $FEATURE_ROOT; run --phase capture first" >&2
    exit 1
  fi
  if ((count < GPU_COUNT * MICRO_BATCH_SIZE)); then
    echo "Feature set has $count records; need at least $((GPU_COUNT * MICRO_BATCH_SIZE)) for one training batch" >&2
    exit 1
  fi

  latest_link="$RUN_OUTPUT/$RUN_ID-latest"
  if ((RESUME)); then
    if [[ -e "$latest_link" ]]; then
      resume_args+=("training.resume_from=$RUN_OUTPUT")
    else
      echo '[train] no checkpoint exists yet; starting a new training run'
    fi
  elif [[ -e "$latest_link" ]]; then
    echo "Checkpoint already exists at $latest_link; pass --resume or use a new ARTIFACT_ROOT" >&2
    exit 1
  fi

  echo "[train] GPUs=$GPU_COUNT micro/rank=$MICRO_BATCH_SIZE accumulation=$ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE features=$count"
  (
    cd "$SPECFORGE_DIR"
    "$PYTHON_BIN" -m specforge.cli train \
      --config "$RUN_CONFIG" \
      "model.target_model_path=$TARGET_MODEL_PATH" \
      "model.vocab_mapping_path=$FEATURE_ROOT/vocab_mapping/vocab_mapping.pt" \
      "data.hidden_states_path=$FEATURE_ROOT" \
      "data.cache_dir=$ARTIFACT_ROOT/cache" \
      "deployment.trainer.nproc_per_node=$GPU_COUNT" \
      "run_id=$RUN_ID" \
      "output_dir=$RUN_OUTPUT" \
      "${resume_args[@]}"
  )
}

if [[ "$PHASE" == data || "$PHASE" == all ]]; then
  prepare_data
elif [[ ! -s "$SPECFORGE_DATA" ]]; then
  echo "Converted dataset not found: $SPECFORGE_DATA; run --phase data first" >&2
  exit 1
else
  require_data_count
fi

if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
  capture_features
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  train_draft
fi
