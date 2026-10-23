#!/usr/bin/env bash
set -euo pipefail

# Run the two-stage Qwen2.5-VL-3B DFlash recipe for one five-layer target set.
# Layer-specific artifact roots ensure hidden-state caches cannot be reused
# accidentally across different target-layer selections.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PHASE1_LAUNCHER="$ROOT_DIR/train_qwen25vl_dflash_sharegpt_68k.sh"
PHASE2_LAUNCHER="$ROOT_DIR/train_qwen25vl_dflash_llava_68k.sh"

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
    --env-file=*) ENV_FILE=${ARGS[arg_index]#--env-file=} ;;
  esac
done
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "environment file not found: $ENV_FILE" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
WRAPPER_PHASE=${SPECFORGE_WRAPPER_PHASE:-all}
INNER_PHASE=${SPECFORGE_INNER_PHASE:-all}
GPU_IDS=${SPECFORGE_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}
SKIP_GPU_CHECK=${SPECFORGE_SKIP_GPU_CHECK:-0}
SET_NAME=${SPECFORGE_LAYER_SET:-}
TARGET_LAYER_IDS=${SPECFORGE_TARGET_LAYER_IDS:-}
RESUME=0
PRINT_CONFIG=0
CLI_SELECTION=

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dflash_3layer.sh (--set NAME | --layer IDS) [options]

Run the Qwen2.5-VL-3B DFlash Phase 1 + Phase 2 pipeline with a 3-layer draft.
Each layer selection gets a separate artifact and output root.

Layer sets:
  high_only       26,28,30,32,34
  middle_high     17,19,25,33,34
  middle_high_2   17,28,30,32,34
  shallow_high    1,28,30,32,34

Options:
  --set NAME              Select one named layer set.
  --layer IDS             Custom ordered list of five target layer IDs.
  --gpu-ids IDS           Exactly two distinct physical GPU IDs (default: 0,1).
  --phase all|phase1|phase2
                          Run both phases, Phase 1 only, or Phase 2 only.
  --inner-phase data|capture|train|all
                          Operation forwarded to each selected phase.
  --env-file FILE         Source a trusted shell-style environment file.
  --resume                Resume existing capture files/checkpoints.
  --print-config          Print resolved IDs, paths, and phase commands.
  -h, --help              Show this help.

Required environment for actual runs:
  SOURCE_DATA, MODEL_3B or TARGET_MODEL_PATH,
  SOURCE_JSONL, IMAGE_ROOT or IMAGE_ARCHIVE.
  Phase 2 training also needs PHASE1_CHECKPOINT when run independently.

Optional:
  ARTIFACT_ROOT, OUTPUT_ROOT, SPECFORGE_GPU_IDS, SPECFORGE_LAYER_SET,
  SPECFORGE_TARGET_LAYER_IDS, SPECFORGE_WRAPPER_PHASE,
  SPECFORGE_INNER_PHASE, SPECFORGE_SKIP_GPU_CHECK.
EOF
}

while (($#)); do
  case "$1" in
    --env-file) shift 2 ;;
    --env-file=*) shift ;;
    --set)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "--set requires a value" >&2
        exit 2
      fi
      if [[ -n "$CLI_SELECTION" ]]; then
        echo "choose exactly one of --set or --layer" >&2
        exit 2
      fi
      CLI_SELECTION=set
      SET_NAME=${2:?--set requires a value}
      TARGET_LAYER_IDS=
      shift 2
      ;;
    --layer)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "--layer requires a value" >&2
        exit 2
      fi
      if [[ -n "$CLI_SELECTION" ]]; then
        echo "choose exactly one of --set or --layer" >&2
        exit 2
      fi
      CLI_SELECTION=layer
      TARGET_LAYER_IDS=${2:?--layer requires a value}
      SET_NAME=
      shift 2
      ;;
    --gpu-ids|--gpus)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      GPU_IDS=$2
      shift 2
      ;;
    --phase|--inner-phase)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "$1 requires a value" >&2
        exit 2
      fi
      if [[ "$1" == --phase ]]; then WRAPPER_PHASE=$2; else INNER_PHASE=$2; fi
      shift 2
      ;;
    --resume) RESUME=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$SET_NAME" && -n "$TARGET_LAYER_IDS" ]]; then
  echo "choose exactly one of --set or --layer" >&2
  exit 2
fi
case "$SET_NAME" in
  high_only) TARGET_LAYER_IDS=26,28,30,32,34 ;;
  middle_high) TARGET_LAYER_IDS=17,19,25,33,34 ;;
  middle_high_2) TARGET_LAYER_IDS=17,28,30,32,34 ;;
  shallow_high) TARGET_LAYER_IDS=1,28,30,32,34 ;;
  "") ;;
  *) echo "unknown layer set: $SET_NAME" >&2; exit 2 ;;
esac
if [[ -z "$TARGET_LAYER_IDS" ]]; then
  echo "select a layer set with --set NAME or provide --layer IDS" >&2
  exit 2
fi

case "$MODEL_SIZE" in
  3b) ;;
  *) echo "this launcher supports only SPECFORGE_MODEL_SIZE=3b" >&2; exit 2 ;;
esac
if [[ -n "${SPECFORGE_DFLASH_DEPTH:-}" && "$SPECFORGE_DFLASH_DEPTH" != 3 ]]; then
  echo "this launcher fixes SPECFORGE_DFLASH_DEPTH=3" >&2
  exit 2
fi
case "$WRAPPER_PHASE" in
  all|phase1|phase2) ;;
  *) echo "--phase must be all, phase1, or phase2" >&2; exit 2 ;;
esac
case "$INNER_PHASE" in
  data|capture|train|all) ;;
  *) echo "--inner-phase must be data, capture, train, or all" >&2; exit 2 ;;
esac
case "$SKIP_GPU_CHECK" in
  0|1) ;;
  *) echo "SPECFORGE_SKIP_GPU_CHECK must be 0 or 1" >&2; exit 2 ;;
esac

TARGET_LAYER_IDS=${TARGET_LAYER_IDS//[[:space:]]/}
IFS=',' read -r -a RAW_LAYER_IDS <<< "$TARGET_LAYER_IDS"
if ((${#RAW_LAYER_IDS[@]} != 5)); then
  echo "layer selection must contain exactly five comma-separated IDs" >&2
  exit 2
fi
NORMALIZED_LAYER_IDS=()
for layer_id in "${RAW_LAYER_IDS[@]}"; do
  if [[ ! "$layer_id" =~ ^[0-9]+$ ]]; then
    echo "layer IDs must be non-negative integers" >&2
    exit 2
  fi
  numeric_layer_id=$((10#$layer_id))
  if ((numeric_layer_id > 34)); then
    echo "layer IDs must leave a following decoder boundary available (Qwen2.5-VL-3B valid range: 0..34; got $numeric_layer_id)" >&2
    exit 2
  fi
  for previous_id in "${NORMALIZED_LAYER_IDS[@]}"; do
    if ((numeric_layer_id == previous_id)); then
      echo "layer IDs must be unique" >&2
      exit 2
    fi
  done
  NORMALIZED_LAYER_IDS+=("$numeric_layer_id")
done
TARGET_LAYER_IDS=$(IFS=,; printf '%s' "${NORMALIZED_LAYER_IDS[*]}")
LAYER_SLUG=$(IFS=_; printf '%s' "${NORMALIZED_LAYER_IDS[*]}")

GPU_IDS=${GPU_IDS//[[:space:]]/}
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpu-ids must be comma-separated non-negative integers" >&2
  exit 2
fi
IFS=',' read -r -a RAW_GPU_IDS <<< "$GPU_IDS"
if ((${#RAW_GPU_IDS[@]} != 2)); then
  echo "exactly two GPU IDs are required" >&2
  exit 2
fi
NORMALIZED_GPU_IDS=()
for gpu_id in "${RAW_GPU_IDS[@]}"; do
  numeric_gpu_id=$((10#$gpu_id))
  for previous_id in "${NORMALIZED_GPU_IDS[@]}"; do
    if ((numeric_gpu_id == previous_id)); then
      echo "two distinct GPU IDs are required" >&2
      exit 2
    fi
  done
  NORMALIZED_GPU_IDS+=("$numeric_gpu_id")
done
GPU_IDS=$(IFS=,; printf '%s' "${NORMALIZED_GPU_IDS[*]}")

ARTIFACT_JOB_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/artifacts}/dflash_3layer_${LAYER_SLUG}"
OUTPUT_JOB_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}/dflash_3layer_${LAYER_SLUG}"
PHASE1_ARTIFACT_ROOT="$ARTIFACT_JOB_ROOT/phase1"
PHASE2_ARTIFACT_ROOT="$ARTIFACT_JOB_ROOT/phase2"
PHASE1_OUTPUT_ROOT="$OUTPUT_JOB_ROOT/phase1"
PHASE2_OUTPUT_ROOT="$OUTPUT_JOB_ROOT/phase2"
RUN_SUFFIX="-dflash_3layer_${LAYER_SLUG}"
DRAFT_CONFIG_SUFFIX="$RUN_SUFFIX"
PHASE1_RUN_ID="qwen25vl-3b-dflash-sharegpt68k${RUN_SUFFIX}"
PHASE2_RUN_ID="qwen25vl-3b-dflash-llava68k${RUN_SUFFIX}"
PHASE1_FEATURE_ROOT="$PHASE1_ARTIFACT_ROOT/qwen25vl_3b/hidden_states"
PHASE1_DATA_CACHE_ROOT="$PHASE1_ARTIFACT_ROOT/cache"
PHASE2_DATA_CACHE_ROOT="$PHASE2_ARTIFACT_ROOT/cache"
PHASE1_MAX_LENGTH=${SPECFORGE_PHASE1_MAX_LENGTH:-2048}
PHASE2_MAX_LENGTH=${SPECFORGE_PHASE2_MAX_LENGTH:-3072}
DEFAULT_PHASE1_CHECKPOINT="$PHASE1_OUTPUT_ROOT/$PHASE1_RUN_ID/$PHASE1_RUN_ID-latest"
if [[ "$WRAPPER_PHASE" == all && -n "${PHASE1_CHECKPOINT:-}" && \
      "$PHASE1_CHECKPOINT" != "$DEFAULT_PHASE1_CHECKPOINT" ]]; then
  echo "Phase 1 checkpoint is fixed to this run's output; use --phase phase2 to select an existing checkpoint" >&2
  exit 2
fi
if [[ "$WRAPPER_PHASE" == all ]]; then
  PHASE1_CHECKPOINT=$DEFAULT_PHASE1_CHECKPOINT
else
  PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-$DEFAULT_PHASE1_CHECKPOINT}
fi

if [[ -n "${MODEL_3B:-}" && -n "${TARGET_MODEL_PATH:-}" && "$MODEL_3B" != "$TARGET_MODEL_PATH" ]]; then
  echo "MODEL_3B and TARGET_MODEL_PATH must identify the same Qwen2.5-VL-3B model" >&2
  exit 2
fi
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${MODEL_3B:-}}
MODEL_3B=$TARGET_MODEL_PATH

# Match the repository's existing two-GPU run defaults while allowing every
# value to be overridden from the selected environment file.
export SPECFORGE_NUM_SAMPLES=${SPECFORGE_NUM_SAMPLES:-68000}
export SPECFORGE_GLOBAL_BATCH_SIZE=${SPECFORGE_GLOBAL_BATCH_SIZE:-64}
export SPECFORGE_MICRO_BATCH_SIZE=${SPECFORGE_MICRO_BATCH_SIZE:-16}
export SPECFORGE_NUM_EPOCHS=${SPECFORGE_NUM_EPOCHS:-6}
export SPECFORGE_DATALOADER_WORKERS=${SPECFORGE_DATALOADER_WORKERS:-12}
export SPECFORGE_FSDP_SHARDING=${SPECFORGE_FSDP_SHARDING:-NO_SHARD}
export SPECFORGE_OBJECTIVE_CHUNK_BLOCKS=${SPECFORGE_OBJECTIVE_CHUNK_BLOCKS:-256}
export SPECFORGE_OPTIMIZER_CPU_OFFLOAD=${SPECFORGE_OPTIMIZER_CPU_OFFLOAD:-0}
export SPECFORGE_ATTENTION_BACKEND=${SPECFORGE_ATTENTION_BACKEND:-flex_attention}
export SPECFORGE_USE_LIGER=${SPECFORGE_USE_LIGER:-auto}
export SPECFORGE_SAVE_INTERVAL=${SPECFORGE_SAVE_INTERVAL:-1000}
export SPECFORGE_LOG_INTERVAL=${SPECFORGE_LOG_INTERVAL:-100}
export SPECFORGE_COMPRESS=${SPECFORGE_COMPRESS:-0}
export SPECFORGE_CAPTURE_BATCH_SIZE=${SPECFORGE_CAPTURE_BATCH_SIZE:-16}
export SPECFORGE_CAPTURE_PREPROCESS_WORKERS=${SPECFORGE_CAPTURE_PREPROCESS_WORKERS:-8}
export SPECFORGE_CAPTURE_PREPROCESS_QUEUE=${SPECFORGE_CAPTURE_PREPROCESS_QUEUE:-32}
export SPECFORGE_CAPTURE_WORKERS=${SPECFORGE_CAPTURE_WORKERS:-8}
export SPECFORGE_CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-8}
export SPECFORGE_CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-64}
export SPECFORGE_SGLANG_MEM_FRACTION_STATIC=${SPECFORGE_SGLANG_MEM_FRACTION_STATIC:-0.4}
export SPECFORGE_PHASE1_MAX_LENGTH="$PHASE1_MAX_LENGTH"
export SPECFORGE_PHASE2_MAX_LENGTH="$PHASE2_MAX_LENGTH"
export SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-1}

export SPECFORGE_GPUS=2
export SPECFORGE_DFLASH_DEPTH=3
export SPECFORGE_PHASE1_TARGET_LAYER_IDS="$TARGET_LAYER_IDS"
export SPECFORGE_PHASE2_TARGET_LAYER_IDS="$TARGET_LAYER_IDS"
export SPECFORGE_TARGET_LAYER_IDS="$TARGET_LAYER_IDS"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export SPECFORGE_RUN_SUFFIX="$RUN_SUFFIX"
export SPECFORGE_DRAFT_CONFIG_SUFFIX="$DRAFT_CONFIG_SUFFIX"
export MODEL_3B TARGET_MODEL_PATH PHASE1_CHECKPOINT

print_config() {
  printf 'LAYER_SET=%s\n' "${SET_NAME:-custom}"
  printf 'TARGET_LAYER_IDS=%s\n' "$TARGET_LAYER_IDS"
  printf 'SPECFORGE_PHASE1_TARGET_LAYER_IDS=%s\n' "$SPECFORGE_PHASE1_TARGET_LAYER_IDS"
  printf 'SPECFORGE_PHASE2_TARGET_LAYER_IDS=%s\n' "$SPECFORGE_PHASE2_TARGET_LAYER_IDS"
  printf 'SPECFORGE_DFLASH_DEPTH=%s\n' "$SPECFORGE_DFLASH_DEPTH"
  printf 'SPECFORGE_GPUS=%s\n' "$SPECFORGE_GPUS"
  printf 'SPECFORGE_GLOBAL_BATCH_SIZE=%s\n' "$SPECFORGE_GLOBAL_BATCH_SIZE"
  printf 'SPECFORGE_MICRO_BATCH_SIZE=%s\n' "$SPECFORGE_MICRO_BATCH_SIZE"
  printf 'SPECFORGE_PHASE1_MAX_LENGTH=%s\n' "$SPECFORGE_PHASE1_MAX_LENGTH"
  printf 'SPECFORGE_PHASE2_MAX_LENGTH=%s\n' "$SPECFORGE_PHASE2_MAX_LENGTH"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'WRAPPER_PHASE=%s\n' "$WRAPPER_PHASE"
  printf 'INNER_PHASE=%s\n' "$INNER_PHASE"
  printf 'PHASE1_ARTIFACT_ROOT=%s\n' "$PHASE1_ARTIFACT_ROOT"
  printf 'PHASE2_ARTIFACT_ROOT=%s\n' "$PHASE2_ARTIFACT_ROOT"
  printf 'PHASE1_OUTPUT_ROOT=%s\n' "$PHASE1_OUTPUT_ROOT"
  printf 'PHASE2_OUTPUT_ROOT=%s\n' "$PHASE2_OUTPUT_ROOT"
  printf 'PHASE1_FEATURE_ROOT=%s\n' "$PHASE1_FEATURE_ROOT"
  printf 'PHASE1_DATA_CACHE_ROOT=%s\n' "$PHASE1_DATA_CACHE_ROOT"
  printf 'PHASE2_DATA_CACHE_ROOT=%s\n' "$PHASE2_DATA_CACHE_ROOT"
  printf 'PHASE1_CHECKPOINT=%s\n' "$PHASE1_CHECKPOINT"
  if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase1 ]]; then
    printf 'PHASE1_COMMAND=bash %s --models 3b --phase %s --gpus 2\n' \
      "$PHASE1_LAUNCHER" "$INNER_PHASE"
  fi
  if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase2 ]]; then
    printf 'PHASE2_COMMAND=bash %s --phase %s --gpus 2\n' \
      "$PHASE2_LAUNCHER" "$INNER_PHASE"
  fi
}

if ((PRINT_CONFIG)); then
  print_config
  exit 0
fi

if [[ -z "$TARGET_MODEL_PATH" ]]; then
  echo "set MODEL_3B or TARGET_MODEL_PATH to the matching Qwen2.5-VL-3B target model" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

verify_two_gpus() {
  if [[ "$SKIP_GPU_CHECK" == 1 ]]; then
    echo "[3layer-run] skipping two-GPU check (SPECFORGE_SKIP_GPU_CHECK=1)"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"PyTorch is not installed in PYTHON_BIN: {exc}")

count = torch.cuda.device_count()
if count != 2:
    raise SystemExit(f"this run requires exactly two visible GPUs, found {count}")
names = [torch.cuda.get_device_name(index) for index in range(count)]
print(f"[3layer-run] torch={torch.__version__} cuda={torch.version.cuda} GPUs={names}")
PY
}

run_phase1() {
  local resume_args=()
  if ((RESUME)); then resume_args+=(--resume); fi
  echo "[3layer-run] Phase 1 layers=$TARGET_LAYER_IDS depth=3 artifacts=$PHASE1_ARTIFACT_ROOT output=$PHASE1_OUTPUT_ROOT"
  env \
    ARTIFACT_ROOT="$PHASE1_ARTIFACT_ROOT" \
    OUTPUT_ROOT="$PHASE1_OUTPUT_ROOT" \
    MODEL_3B="$TARGET_MODEL_PATH" \
    TARGET_MODEL_PATH="$TARGET_MODEL_PATH" \
    SPECFORGE_PHASE1_FEATURE_ROOT="$PHASE1_FEATURE_ROOT" \
    SPECFORGE_DATA_CACHE_ROOT="$PHASE1_DATA_CACHE_ROOT" \
    SPECFORGE_ALLOW_FEATURE_ONLY_TRAIN=0 \
    SPECFORGE_MIGRATE_LEGACY_HIDDEN_STATE_METADATA=0 \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    SPECFORGE_GPUS=2 \
    SPECFORGE_DFLASH_DEPTH=3 \
    SPECFORGE_MAX_LENGTH="$PHASE1_MAX_LENGTH" \
    SPECFORGE_PHASE1_TARGET_LAYER_IDS="$TARGET_LAYER_IDS" \
    SPECFORGE_PHASE2_TARGET_LAYER_IDS="$TARGET_LAYER_IDS" \
    SPECFORGE_RUN_SUFFIX="$RUN_SUFFIX" \
    SPECFORGE_DRAFT_CONFIG_SUFFIX="$DRAFT_CONFIG_SUFFIX" \
    SPECFORGE_ENV_FILE= \
    bash "$PHASE1_LAUNCHER" --models 3b --phase "$INNER_PHASE" --gpus 2 "${resume_args[@]}"
}

run_phase2() {
  local resume_args=()
  if ((RESUME)); then resume_args+=(--resume); fi
  if [[ "$INNER_PHASE" == train || "$INNER_PHASE" == all ]]; then
    if [[ ! -e "$PHASE1_CHECKPOINT" ]]; then
      echo "Phase 1 checkpoint not found: $PHASE1_CHECKPOINT" >&2
      exit 1
    fi
  fi
  echo "[3layer-run] Phase 2 layers=$TARGET_LAYER_IDS depth=3 artifacts=$PHASE2_ARTIFACT_ROOT output=$PHASE2_OUTPUT_ROOT"
  env \
    ARTIFACT_ROOT="$PHASE2_ARTIFACT_ROOT" \
    OUTPUT_ROOT="$PHASE2_OUTPUT_ROOT" \
    SPECFORGE_DATA_CACHE_ROOT="$PHASE2_DATA_CACHE_ROOT" \
    SPECFORGE_MIGRATE_LEGACY_HIDDEN_STATE_METADATA=0 \
    TARGET_MODEL_PATH="$TARGET_MODEL_PATH" \
    PHASE1_CHECKPOINT="$PHASE1_CHECKPOINT" \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    SPECFORGE_GPUS=2 \
    SPECFORGE_DFLASH_DEPTH=3 \
    SPECFORGE_MAX_LENGTH="$PHASE2_MAX_LENGTH" \
    SPECFORGE_PHASE1_TARGET_LAYER_IDS="$TARGET_LAYER_IDS" \
    SPECFORGE_PHASE2_TARGET_LAYER_IDS="$TARGET_LAYER_IDS" \
    SPECFORGE_RUN_SUFFIX="$RUN_SUFFIX" \
    SPECFORGE_DRAFT_CONFIG_SUFFIX="$DRAFT_CONFIG_SUFFIX" \
    SPECFORGE_ENV_FILE= \
    bash "$PHASE2_LAUNCHER" --phase "$INNER_PHASE" --gpus 2 "${resume_args[@]}"
}

verify_two_gpus
echo "[3layer-run] phase=$WRAPPER_PHASE inner_phase=$INNER_PHASE GPUs=2 layers=$TARGET_LAYER_IDS"
if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase1 ]]; then
  run_phase1
fi
if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase2 ]]; then
  run_phase2
fi
