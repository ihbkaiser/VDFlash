#!/usr/bin/env bash
set -euo pipefail

SPECFORGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
SOURCE_DATA=${SOURCE_DATA:-"/workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"}
MODEL_3B=${MODEL_3B:-"/workspace/storage-shared/nlp/tungdd11/tungdecoder/models/qwen25-vl-3b"}
MODEL_7B=${MODEL_7B:-"/workspace/storage-shared/nlp/tungdd11/tungdecoder/models/qwen25-vl-7b"}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$SPECFORGE_DIR/artifacts/qwen25vl_dflash_sharegpt68k"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$SPECFORGE_DIR/outputs"}

GPU_COUNT=${SPECFORGE_GPUS:-4}
MODELS=both
PHASE=all
RESUME=0
OVERWRITE_DATA=0
EXPECTED_RECORDS=${SPECFORGE_NUM_SAMPLES:-68000}
GLOBAL_BATCH_SIZE=${SPECFORGE_GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${SPECFORGE_MICRO_BATCH_SIZE:-16}
CAPTURE_BATCH_SIZE=${SPECFORGE_CAPTURE_BATCH_SIZE:-64}
MAX_LENGTH=${SPECFORGE_MAX_LENGTH:-2048}
NUM_EPOCHS=${SPECFORGE_NUM_EPOCHS:-6}
SGLANG_MEM_FRACTION_STATIC=${SPECFORGE_SGLANG_MEM_FRACTION_STATIC:-0.4}
OPTIMIZER_CPU_OFFLOAD=${SPECFORGE_OPTIMIZER_CPU_OFFLOAD:-0}
EMBEDDING_KEY=${SPECFORGE_EMBEDDING_KEY:-model.language_model.embed_tokens.weight}
FSDP_SHARDING=${SPECFORGE_FSDP_SHARDING:-NO_SHARD}
OBJECTIVE_CHUNK_BLOCKS=${SPECFORGE_OBJECTIVE_CHUNK_BLOCKS:-256}
DATASET_NUM_PROC=${SPECFORGE_DATASET_NUM_PROC:-32}
CAPTURE_WORKERS=${SPECFORGE_CAPTURE_WORKERS:-8}
CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-8}
CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-128}
DATALOADER_WORKERS=${SPECFORGE_DATALOADER_WORKERS:-12}
SAVE_INTERVAL=${SPECFORGE_SAVE_INTERVAL:-1000}
LOG_INTERVAL=${SPECFORGE_LOG_INTERVAL:-100}
USE_LIGER=${SPECFORGE_USE_LIGER:-auto}
CAPTURE_TORCH_COMPILE=${SPECFORGE_CAPTURE_TORCH_COMPILE:-0}

usage() {
  printf '%s\n' \
    'Train text-only Qwen2.5-VL DFlash drafts with this SpecForge checkout.' \
    '' \
    'Usage:' \
    '  bash train_qwen25vl_dflash_sharegpt_68k.sh [options]' \
    '' \
    'Options:' \
    '  --gpus N             Capture/trainer process count (default: 4).' \
    '  --models VALUE       3b, 7b, or both (default: both).' \
    '  --phase VALUE        data, capture, train, or all (default: all).' \
    '  --resume             Continue capture files and the latest checkpoint.' \
    '  --overwrite-data     Rebuild the converted 68k ShareGPT JSONL.' \
    '  -h, --help           Show this help.' \
    '' \
    'Environment:' \
    '  PYTHON_BIN, SOURCE_DATA, MODEL_3B, MODEL_7B, ARTIFACT_ROOT, OUTPUT_ROOT' \
    '  SPECFORGE_GPUS, SPECFORGE_NUM_SAMPLES, SPECFORGE_GLOBAL_BATCH_SIZE' \
    '  SPECFORGE_MICRO_BATCH_SIZE, SPECFORGE_CAPTURE_BATCH_SIZE' \
    '  SPECFORGE_MAX_LENGTH, SPECFORGE_NUM_EPOCHS' \
    '  SPECFORGE_SGLANG_MEM_FRACTION_STATIC, SPECFORGE_OPTIMIZER_CPU_OFFLOAD' \
    '  SPECFORGE_FSDP_SHARDING, SPECFORGE_OBJECTIVE_CHUNK_BLOCKS' \
    '  SPECFORGE_DATASET_NUM_PROC, SPECFORGE_CAPTURE_WORKERS' \
    '  SPECFORGE_CAPTURE_IO_THREADS, SPECFORGE_CAPTURE_IO_QUEUE' \
    '  SPECFORGE_DATALOADER_WORKERS, SPECFORGE_SAVE_INTERVAL' \
    '  SPECFORGE_LOG_INTERVAL, SPECFORGE_USE_LIGER=auto|0|1' \
    '  SPECFORGE_CAPTURE_TORCH_COMPILE=0|1, SPECFORGE_EMBEDDING_KEY' \
    '  SPECFORGE_COMPRESS=1'
}

while (($#)); do
  case "$1" in
    --gpus)
      GPU_COUNT=${2:?"--gpus requires a value"}
      shift 2
      ;;
    --models)
      MODELS=${2:?"--models requires a value"}
      shift 2
      ;;
    --phase)
      PHASE=${2:?"--phase requires a value"}
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

case "$MODELS" in
  3b|7b|both) ;;
  *) echo "--models must be 3b, 7b, or both" >&2; exit 2 ;;
esac
case "$PHASE" in
  data|capture|train|all) ;;
  *) echo "--phase must be data, capture, train, or all" >&2; exit 2 ;;
esac
if [[ "$OPTIMIZER_CPU_OFFLOAD" != 0 && "$OPTIMIZER_CPU_OFFLOAD" != 1 ]]; then
  echo "SPECFORGE_OPTIMIZER_CPU_OFFLOAD must be 0 or 1" >&2
  exit 2
fi
case "$FSDP_SHARDING" in
  NO_SHARD|SHARD_GRAD_OP|FULL_SHARD) ;;
  *) echo "SPECFORGE_FSDP_SHARDING must be NO_SHARD, SHARD_GRAD_OP, or FULL_SHARD" >&2; exit 2 ;;
esac
case "$USE_LIGER" in
  auto|0|1) ;;
  *) echo "SPECFORGE_USE_LIGER must be auto, 0, or 1" >&2; exit 2 ;;
esac
if [[ "$CAPTURE_TORCH_COMPILE" != 0 && "$CAPTURE_TORCH_COMPILE" != 1 ]]; then
  echo "SPECFORGE_CAPTURE_TORCH_COMPILE must be 0 or 1" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
for value_name in GPU_COUNT EXPECTED_RECORDS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE CAPTURE_BATCH_SIZE MAX_LENGTH NUM_EPOCHS DATASET_NUM_PROC CAPTURE_WORKERS CAPTURE_IO_THREADS CAPTURE_IO_QUEUE DATALOADER_WORKERS LOG_INTERVAL; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
if [[ ! "$OBJECTIVE_CHUNK_BLOCKS" =~ ^[0-9]+$ ]]; then
  echo "SPECFORGE_OBJECTIVE_CHUNK_BLOCKS must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$SAVE_INTERVAL" =~ ^[0-9]+$ ]]; then
  echo "SPECFORGE_SAVE_INTERVAL must be a non-negative integer" >&2
  exit 2
fi
if ! "$PYTHON_BIN" -c 'import sys; value=float(sys.argv[1]); raise SystemExit(0 if 0 < value <= 1 else 1)' "$SGLANG_MEM_FRACTION_STATIC"; then
  echo "SPECFORGE_SGLANG_MEM_FRACTION_STATIC must be in (0, 1]" >&2
  exit 2
fi

GLOBAL_MICRO_BATCH=$((GPU_COUNT * MICRO_BATCH_SIZE))
if ((GLOBAL_BATCH_SIZE % GLOBAL_MICRO_BATCH != 0)); then
  echo "Global batch $GLOBAL_BATCH_SIZE must be divisible by GPUs x micro batch ($GLOBAL_MICRO_BATCH)" >&2
  exit 2
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / GLOBAL_MICRO_BATCH))

if [[ "$USE_LIGER" == auto ]]; then
  if "$PYTHON_BIN" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("liger_kernel") else 1)'; then
    USE_LIGER=true
  else
    USE_LIGER=false
  fi
elif [[ "$USE_LIGER" == 1 ]]; then
  USE_LIGER=true
else
  USE_LIGER=false
fi

export PYTHONPATH="$SPECFORGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ARTIFACT_ROOT/shared"
ARTIFACT_ROOT=$(cd "$ARTIFACT_ROOT" && pwd)
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
SPECFORGE_DATA="$ARTIFACT_ROOT/shared/sharegpt_train.jsonl"

jsonl_count() {
  awk 'NF { count += 1 } END { print count + 0 }' "$1"
}

require_data_count() {
  local count
  count=$(jsonl_count "$SPECFORGE_DATA")
  if ((count != EXPECTED_RECORDS)); then
    echo "Converted dataset has $count records; expected $EXPECTED_RECORDS: $SPECFORGE_DATA" >&2
    echo "Run the data phase with --overwrite-data." >&2
    exit 1
  fi
}

prepare_data() {
  if [[ -s "$SPECFORGE_DATA" ]] && ((!OVERWRITE_DATA)); then
    echo "[data] reusing $SPECFORGE_DATA"
    require_data_count
    return
  fi
  if [[ ! -f "$SOURCE_DATA" ]]; then
    echo "ShareGPT source file not found: $SOURCE_DATA" >&2
    exit 1
  fi

  local temporary_dir temporary_data
  temporary_dir=$(mktemp -d "$ARTIFACT_ROOT/shared/.prepare-sharegpt.XXXXXX")
  temporary_data="$temporary_dir/sharegpt_train.jsonl"
  trap 'rm -rf -- "$temporary_dir"' RETURN
  echo "[data] converting the first $EXPECTED_RECORDS ShareGPT rows"
  (
    cd "$SPECFORGE_DIR"
    "$PYTHON_BIN" scripts/prepare_data.py \
      --dataset sharegpt \
      --data-path "$SOURCE_DATA" \
      --sample-size "$EXPECTED_RECORDS" \
      --output-path "$temporary_dir"
  )
  local count
  count=$(jsonl_count "$temporary_data")
  if ((count != EXPECTED_RECORDS)); then
    echo "ShareGPT conversion produced $count records; expected $EXPECTED_RECORDS" >&2
    exit 1
  fi
  mv -f -- "$temporary_data" "$SPECFORGE_DATA"
  trap - RETURN
  rmdir "$temporary_dir"
  echo "[data] wrote $count records to $SPECFORGE_DATA"
}

feature_count() {
  local feature_dir=$1
  if [[ ! -d "$feature_dir" ]]; then
    echo 0
    return
  fi
  find "$feature_dir" -type f \( -name '*.ckpt' -o -name '*.ckpt.gz' \) -print | wc -l
}

run_model() {
  local size=$1
  local target_model draft_config run_config slug run_id
  case "$size" in
    3b)
      target_model=$MODEL_3B
      draft_config="$SPECFORGE_DIR/configs/qwen2.5-vl-3b-dflash.json"
      run_config="$SPECFORGE_DIR/examples/configs/qwen2.5-vl-3b-dflash-offline-b200.yaml"
      slug=qwen25vl_3b
      run_id=qwen25vl-3b-dflash-sharegpt68k
      ;;
    7b)
      target_model=$MODEL_7B
      draft_config="$SPECFORGE_DIR/configs/qwen2.5-vl-7b-dflash.json"
      run_config="$SPECFORGE_DIR/examples/configs/qwen2.5-vl-7b-dflash-offline-b200.yaml"
      slug=qwen25vl_7b
      run_id=qwen25vl-7b-dflash-sharegpt68k
      ;;
  esac
  local feature_dir="$ARTIFACT_ROOT/$slug/hidden_states"
  local output_dir="$OUTPUT_ROOT/$run_id"

  if [[ ! -e "$target_model" && ( "$target_model" == /* || "$target_model" == ./* || "$target_model" == ../* ) ]]; then
    echo "Target model path not found: $target_model" >&2
    exit 1
  fi

  if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
    local existing_features
    existing_features=$(feature_count "$feature_dir")
    if ((existing_features > 0 && !RESUME)); then
      echo "Offline features already exist at $feature_dir; pass --resume or use a new ARTIFACT_ROOT" >&2
      exit 1
    fi
    mkdir -p "$feature_dir"
    local compress_args=()
    local compile_args=()
    if [[ ${SPECFORGE_COMPRESS:-0} == 1 ]]; then
      compress_args+=(--compress)
    fi
    if ((CAPTURE_TORCH_COMPILE)); then
      compile_args+=(--sglang-enable-torch-compile)
    fi
    echo "[capture:$size] GPUs=$GPU_COUNT TP=1 DP=$GPU_COUNT batch/rank=$CAPTURE_BATCH_SIZE"
    (
      cd "$SPECFORGE_DIR"
      "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="$GPU_COUNT" \
        scripts/prepare_hidden_states.py \
        --strategy dflash \
        --target-model-path "$target_model" \
        --draft-model-config "$draft_config" \
        --data-path "$SPECFORGE_DATA" \
        --output-path "$feature_dir" \
        --cache-dir "$ARTIFACT_ROOT/cache" \
        --chat-template qwen \
        --train-only-last-turn \
        --max-length "$MAX_LENGTH" \
        --num-samples "$EXPECTED_RECORDS" \
        --build-dataset-num-proc "$DATASET_NUM_PROC" \
        --tp-size 1 \
        --batch-size "$CAPTURE_BATCH_SIZE" \
        --num-workers "$CAPTURE_WORKERS" \
        --num-io-threads "$CAPTURE_IO_THREADS" \
        --io-queue-size "$CAPTURE_IO_QUEUE" \
        --file-group-size 4096 \
        --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
        --sglang-disable-radix-cache \
        "${compile_args[@]}" \
        "${compress_args[@]}"
    )
    echo "[capture:$size] feature records=$(feature_count "$feature_dir")"
  fi

  if [[ "$PHASE" == train || "$PHASE" == all ]]; then
    local count per_rank_samples micro_batches_per_epoch max_steps
    count=$(feature_count "$feature_dir")
    if ((count == 0)); then
      echo "No offline features found at $feature_dir; run --phase capture first" >&2
      exit 1
    fi
    per_rank_samples=$(((count + GPU_COUNT - 1) / GPU_COUNT))
    micro_batches_per_epoch=$((per_rank_samples / MICRO_BATCH_SIZE))
    max_steps=$((micro_batches_per_epoch * NUM_EPOCHS / ACCUMULATION_STEPS))
    if ((max_steps == 0)); then
      echo "Feature set is too small to produce one optimizer step for $size" >&2
      exit 1
    fi

    local latest_link="$output_dir/$run_id-latest"
    local resume_args=()
    local optimizer_cpu_offload=false
    if ((OPTIMIZER_CPU_OFFLOAD)); then
      optimizer_cpu_offload=true
    fi
    if ((RESUME)) && [[ -e "$latest_link" ]]; then
      resume_args+=("training.resume_from=$output_dir")
    elif ((RESUME)); then
      echo "[train:$size] no checkpoint exists yet; starting a new training run"
    elif [[ -e "$latest_link" ]]; then
      echo "Checkpoint already exists at $latest_link; pass --resume or use a new ARTIFACT_ROOT" >&2
      exit 1
    fi

    echo "[train:$size] GPUs=$GPU_COUNT micro/rank=$MICRO_BATCH_SIZE accumulation=$ACCUMULATION_STEPS global_batch=$GLOBAL_BATCH_SIZE sharding=$FSDP_SHARDING objective_chunk=$OBJECTIVE_CHUNK_BLOCKS liger=$USE_LIGER features=$count max_steps=$max_steps"
    (
      cd "$SPECFORGE_DIR"
      "$PYTHON_BIN" -m specforge.cli train \
        --config "$run_config" \
        "model.target_model_path=$target_model" \
        "model.draft_model_config=$draft_config" \
        "model.embedding_key=$EMBEDDING_KEY" \
        "model.use_liger_kernel=$USE_LIGER" \
        "data.hidden_states_path=$feature_dir" \
        "data.cache_dir=$ARTIFACT_ROOT/cache" \
        "data.max_length=$MAX_LENGTH" \
        "data.dataloader_num_workers=$DATALOADER_WORKERS" \
        "training.num_epochs=$NUM_EPOCHS" \
        "training.fsdp_sharding=$FSDP_SHARDING" \
        "training.optimizer_cpu_offload=$optimizer_cpu_offload" \
        "training.batch_size=$MICRO_BATCH_SIZE" \
        "training.accumulation_steps=$ACCUMULATION_STEPS" \
        "training.objective_chunk_blocks=$OBJECTIVE_CHUNK_BLOCKS" \
        "training.save_interval=$SAVE_INTERVAL" \
        "training.log_interval=$LOG_INTERVAL" \
        "training.max_steps=$max_steps" \
        "deployment.trainer.nproc_per_node=$GPU_COUNT" \
        "run_id=$run_id" \
        "output_dir=$output_dir" \
        "${resume_args[@]}"
    )
  fi
}

if [[ "$PHASE" == data || "$PHASE" == all ]]; then
  prepare_data
elif [[ ! -s "$SPECFORGE_DATA" ]]; then
  echo "Converted dataset not found: $SPECFORGE_DATA; run --phase data first" >&2
  exit 1
else
  require_data_count
fi

if [[ "$PHASE" != data ]]; then
  if [[ "$MODELS" == 3b || "$MODELS" == both ]]; then
    run_model 3b
  fi
  if [[ "$MODELS" == 7b || "$MODELS" == both ]]; then
    run_model 7b
  fi
fi
