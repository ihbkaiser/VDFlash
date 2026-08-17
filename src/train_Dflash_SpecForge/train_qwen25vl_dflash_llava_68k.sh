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
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs"}
GPU_COUNT=${SPECFORGE_GPUS:-4}
EXPECTED_RECORDS=${SPECFORGE_NUM_SAMPLES:-68000}
MAX_LENGTH=${SPECFORGE_MAX_LENGTH:-3072}
GLOBAL_BATCH_SIZE=${SPECFORGE_GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${SPECFORGE_MICRO_BATCH_SIZE:-1}
NUM_EPOCHS=${SPECFORGE_NUM_EPOCHS:-6}
DATALOADER_WORKERS=${SPECFORGE_DATALOADER_WORKERS:-12}
FSDP_SHARDING=${SPECFORGE_FSDP_SHARDING:-NO_SHARD}
OBJECTIVE_CHUNK_BLOCKS=${SPECFORGE_OBJECTIVE_CHUNK_BLOCKS:-256}
OPTIMIZER_CPU_OFFLOAD=${SPECFORGE_OPTIMIZER_CPU_OFFLOAD:-0}
ATTENTION_BACKEND=${SPECFORGE_ATTENTION_BACKEND:-flex_attention}
USE_LIGER=${SPECFORGE_USE_LIGER:-auto}
SAVE_INTERVAL=${SPECFORGE_SAVE_INTERVAL:-1000}
LOG_INTERVAL=${SPECFORGE_LOG_INTERVAL:-100}
CAPTURE_BATCH_SIZE=${SPECFORGE_CAPTURE_BATCH_SIZE:-16}
CAPTURE_PREPROCESS_WORKERS=${SPECFORGE_CAPTURE_PREPROCESS_WORKERS:-8}
CAPTURE_PREPROCESS_QUEUE=${SPECFORGE_CAPTURE_PREPROCESS_QUEUE:-32}
CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-8}
CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-64}
PHASE=${PHASE:-all}
RESUME=0
COMPRESS=${SPECFORGE_COMPRESS:-0}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
SGLANG_MEM_FRACTION_STATIC=${SPECFORGE_SGLANG_MEM_FRACTION_STATIC:-0.4}

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dflash_llava_68k.sh [options]

Required environment:
  SOURCE_JSONL       complete flat LLaVA caption JSONL on the Phase 2 server
  TARGET_MODEL_PATH  Qwen2.5-VL target matching SPECFORGE_MODEL_SIZE
  PHASE1_CHECKPOINT  same-size Phase 1 DFlash checkpoint for a new Phase 2 run
  IMAGE_ROOT         extracted LLaVA image hierarchy, or set IMAGE_ARCHIVE

Optional environment:
  SPECFORGE_MODEL_SIZE=3b|7b (default: 3b)
  SKIP_PREFLIGHT=1   skip the full LLaVA validation pass before capture
  SPECFORGE_SGLANG_MEM_FRACTION_STATIC
                     SGLang weights/KV memory fraction (default: 0.4)
  SPECFORGE_CAPTURE_BATCH_SIZE
                     requests per GPU capture batch (default: 16)
  SPECFORGE_CAPTURE_PREPROCESS_WORKERS
                     image preprocessing threads per GPU (default: 8)
  SPECFORGE_CAPTURE_IO_THREADS
                     asynchronous checkpoint writers per GPU (default: 8)
  SPECFORGE_NUM_EPOCHS, SPECFORGE_DATALOADER_WORKERS
  SPECFORGE_FSDP_SHARDING=NO_SHARD|SHARD_GRAD_OP|FULL_SHARD
  SPECFORGE_OBJECTIVE_CHUNK_BLOCKS, SPECFORGE_OPTIMIZER_CPU_OFFLOAD=0|1
  SPECFORGE_ATTENTION_BACKEND=flex_attention|sdpa|eager
  SPECFORGE_USE_LIGER=auto|0|1, SPECFORGE_SAVE_INTERVAL,
  SPECFORGE_LOG_INTERVAL

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
case "$SKIP_PREFLIGHT" in
  0|1) ;;
  *) echo "SKIP_PREFLIGHT must be 0 or 1" >&2; exit 2 ;;
esac
if (( GPU_COUNT < 1 || EXPECTED_RECORDS < 1 || MAX_LENGTH < 32 || GLOBAL_BATCH_SIZE < 1 || MICRO_BATCH_SIZE < 1 || NUM_EPOCHS < 1 || DATALOADER_WORKERS < 1 || LOG_INTERVAL < 1 )); then
  echo "invalid numeric configuration" >&2
  exit 2
fi
if (( OBJECTIVE_CHUNK_BLOCKS < 0 || SAVE_INTERVAL < 0 )); then
  echo "invalid numeric configuration" >&2
  exit 2
fi
if (( CAPTURE_BATCH_SIZE < 1 || CAPTURE_PREPROCESS_WORKERS < 0 || CAPTURE_PREPROCESS_QUEUE < 1 || CAPTURE_IO_THREADS < 1 || CAPTURE_IO_QUEUE < 1 )); then
  echo "invalid numeric configuration" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (GPU_COUNT * MICRO_BATCH_SIZE) != 0 )); then
  echo "global batch must be divisible by GPUs * micro batch" >&2
  exit 2
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (GPU_COUNT * MICRO_BATCH_SIZE)))
case "$FSDP_SHARDING" in
  NO_SHARD|SHARD_GRAD_OP|FULL_SHARD) ;;
  *) echo "SPECFORGE_FSDP_SHARDING must be NO_SHARD, SHARD_GRAD_OP, or FULL_SHARD" >&2; exit 2 ;;
esac
case "$OPTIMIZER_CPU_OFFLOAD" in
  0) OPTIMIZER_CPU_OFFLOAD=false ;;
  1) OPTIMIZER_CPU_OFFLOAD=true ;;
  *) echo "SPECFORGE_OPTIMIZER_CPU_OFFLOAD must be 0 or 1" >&2; exit 2 ;;
esac
case "$ATTENTION_BACKEND" in
  flex_attention|sdpa|eager) ;;
  *) echo "SPECFORGE_ATTENTION_BACKEND must be flex_attention, sdpa, or eager" >&2; exit 2 ;;
esac
case "$USE_LIGER" in
  auto)
    if "$PYTHON_BIN" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("liger_kernel") else 1)'; then
      USE_LIGER=true
    else
      USE_LIGER=false
    fi
    ;;
  0) USE_LIGER=false ;;
  1) USE_LIGER=true ;;
  *) echo "SPECFORGE_USE_LIGER must be auto, 0, or 1" >&2; exit 2 ;;
esac

case "$MODEL_SIZE" in
  3b)
    DRAFT_CONFIG="$ROOT_DIR/configs/qwen2.5-vl-3b-dflash.json"
    CONFIG="$ROOT_DIR/examples/configs/qwen2.5-vl-3b-dflash-llava68k-offline.yaml"
    RUN_ID=qwen25vl-3b-dflash-llava68k
    ;;
  7b)
    DRAFT_CONFIG="$ROOT_DIR/configs/qwen2.5-vl-7b-dflash.json"
    CONFIG="$ROOT_DIR/examples/configs/qwen2.5-vl-7b-dflash-offline-b200.yaml"
    RUN_ID=qwen25vl-7b-dflash-llava68k
    ;;
  *)
    echo "SPECFORGE_MODEL_SIZE must be 3b or 7b" >&2
    exit 2
    ;;
esac
if [[ -z "$ARTIFACT_ROOT" ]]; then
  ARTIFACT_ROOT="$ROOT_DIR/artifacts/qwen25vl_${MODEL_SIZE}_dflash_llava68k"
fi

MANIFEST="$ARTIFACT_ROOT/manifest.jsonl"
FEATURE_ROOT="$ARTIFACT_ROOT/hidden_states"
IMAGE_STAGE_ROOT="$ARTIFACT_ROOT/images"
RUN_OUTPUT="$OUTPUT_ROOT/$RUN_ID"

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
  if [[ "$SKIP_PREFLIGHT" == 1 ]]; then
    echo "Skipping LLaVA preflight (SKIP_PREFLIGHT=1)"
  else
    "$PYTHON_BIN" "$ROOT_DIR/scripts/preflight_llava_caption.py" \
      --manifest "$MANIFEST" --image-root "$IMAGE_ROOT" \
      --target-model-path "$TARGET_MODEL_PATH" \
      --draft-model-config "$DRAFT_CONFIG" \
      --output-path "$FEATURE_ROOT" --max-length "$MAX_LENGTH" \
      --expected-records "$EXPECTED_RECORDS"
  fi
  compress_args=()
  if [[ "$COMPRESS" == 1 ]]; then
    compress_args+=(--compress)
  fi
  printf '[capture] GPUs=%s TP=1 DP=%s batch/rank=%s preprocess_workers/rank=%s io_threads/rank=%s\n' \
    "$GPU_COUNT" "$GPU_COUNT" "$CAPTURE_BATCH_SIZE" \
    "$CAPTURE_PREPROCESS_WORKERS" "$CAPTURE_IO_THREADS"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    "$ROOT_DIR/scripts/prepare_llava_caption_hidden_states.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$DRAFT_CONFIG" \
    --manifest "$MANIFEST" --image-root "$IMAGE_ROOT" \
    --output-path "$FEATURE_ROOT" --max-length "$MAX_LENGTH" \
    --expected-records "$EXPECTED_RECORDS" --tp-size 1 \
    --batch-size "$CAPTURE_BATCH_SIZE" \
    --num-preprocess-workers "$CAPTURE_PREPROCESS_WORKERS" \
    --preprocess-queue-size "$CAPTURE_PREPROCESS_QUEUE" \
    --num-io-threads "$CAPTURE_IO_THREADS" \
    --io-queue-size "$CAPTURE_IO_QUEUE" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    "${compress_args[@]}"
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  require_value TARGET_MODEL_PATH
  [[ -d "$FEATURE_ROOT" ]] || { echo "feature directory missing: $FEATURE_ROOT" >&2; exit 1; }
  train_args=(
    "model.target_model_path=$TARGET_MODEL_PATH"
    "model.draft_model_config=$DRAFT_CONFIG"
    "model.input_modality=qwen2_5_vl"
    "model.use_liger_kernel=$USE_LIGER"
    "data.hidden_states_path=$FEATURE_ROOT"
    "data.max_length=$MAX_LENGTH"
    "data.dataloader_num_workers=$DATALOADER_WORKERS"
    "training.num_epochs=$NUM_EPOCHS"
    "training.batch_size=$MICRO_BATCH_SIZE"
    "training.accumulation_steps=$ACCUMULATION_STEPS"
    "training.fsdp_sharding=$FSDP_SHARDING"
    "training.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD"
    "training.attention_backend=$ATTENTION_BACKEND"
    "training.objective_chunk_blocks=$OBJECTIVE_CHUNK_BLOCKS"
    "training.save_interval=$SAVE_INTERVAL"
    "training.log_interval=$LOG_INTERVAL"
    "deployment.trainer.nproc_per_node=$GPU_COUNT"
    "output_dir=$RUN_OUTPUT"
    "run_id=$RUN_ID"
  )
  latest="$RUN_OUTPUT/$RUN_ID-latest"
  if (( RESUME == 1 )) && [[ -e "$latest" ]]; then
    train_args+=("training.resume_from=$RUN_OUTPUT")
  else
    require_value PHASE1_CHECKPOINT
    train_args+=("model.draft_checkpoint_path=$PHASE1_CHECKPOINT")
  fi
  printf '[train:%s] GPUs=%s micro/rank=%s accumulation=%s global_batch=%s epochs=%s sharding=%s objective_chunk=%s loader_workers/rank=%s liger=%s\n' \
    "$MODEL_SIZE" "$GPU_COUNT" "$MICRO_BATCH_SIZE" "$ACCUMULATION_STEPS" \
    "$GLOBAL_BATCH_SIZE" "$NUM_EPOCHS" "$FSDP_SHARDING" \
    "$OBJECTIVE_CHUNK_BLOCKS" "$DATALOADER_WORKERS" "$USE_LIGER"
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
    --draft-model-config "$DRAFT_CONFIG" \
    --checkpoint "$RUN_OUTPUT" \
    --manifest "$MANIFEST" --image-root "$IMAGE_ROOT"
fi
