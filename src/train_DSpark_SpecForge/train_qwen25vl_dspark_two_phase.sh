#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$ROOT_DIR/../.." && pwd)

# Load a trusted shell-style environment file before resolving defaults.
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
  [[ -f "$ENV_FILE" ]] || { echo "environment file not found: $ENV_FILE" >&2; exit 2; }
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${MODEL_3B:-}}
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
SET_NAME=${SPECFORGE_LAYER_SET:-}
TARGET_LAYER_IDS=${SPECFORGE_TARGET_LAYER_IDS:-}
WRAPPER_PHASE=${SPECFORGE_WRAPPER_PHASE:-all}
GPU_IDS=${SPECFORGE_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}
GPU_COUNT=${SPECFORGE_GPUS:-}
RESUME=0
PRINT_CONFIG=0
CLI_SELECTION=

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dspark_two_phase.sh (--set NAME | --layer IDS) [options]

Train DSpark on existing Qwen2.5-VL-3B Phase 1 and Phase 2 feature caches.
The default run trains Phase 1, then warm-starts Phase 2 from its latest draft.
No data preparation or hidden-state capture is run by this launcher.

Teacher-layer sets (each set selects five target hidden-state taps):
  spread          1,9,17,25,33
  high_only       26,28,30,32,34
  middle_high     17,19,25,33,34
  middle_high_2   17,28,30,32,34
  shallow_high    1,28,30,32,34

Options:
  --set NAME              Select one named cached target-layer set.
  --layer IDS             Custom ordered list of five cached layer IDs.
  --phase all|phase1|phase2
                          Run both phases, Phase 1 only, or Phase 2 only.
  --gpu-ids IDS           Comma-separated visible GPU IDs (default: 0,1).
  --env-file FILE         Source a trusted shell-style environment file.
  --resume                Resume the selected DSpark output checkpoint.
  --print-config          Print resolved paths and training commands.
  -h, --help              Show this help.

Required environment for training:
  TARGET_MODEL_PATH or MODEL_3B, plus existing Phase 1 and Phase 2 feature roots.
  The feature roots default to the matching dflash_3layer_<IDs> cache paths under
  SPECFORGE_DFLASH_ARTIFACT_ROOT; override them when caches live elsewhere.

Useful settings:
  SPECFORGE_DFLASH_ARTIFACT_ROOT, SPECFORGE_PHASE1_FEATURE_ROOT,
  SPECFORGE_PHASE2_FEATURE_ROOT, SPECFORGE_OUTPUT_ROOT,
  SPECFORGE_ARTIFACT_ROOT, SPECFORGE_NUM_SAMPLES,
  SPECFORGE_GLOBAL_BATCH_SIZE, SPECFORGE_MICRO_BATCH_SIZE,
  SPECFORGE_NUM_EPOCHS, SPECFORGE_DSPARK_DEPTH,
  SPECFORGE_DSPARK_BLOCK_SIZE, SPECFORGE_CACHE_VALIDATE_SAMPLE_LIMIT,
  SPECFORGE_ATTENTION_BACKEND, SPECFORGE_FSDP_SHARDING.
EOF
}

while (($#)); do
  case "$1" in
    --env-file) shift 2 ;;
    --env-file=*) shift ;;
    --set)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "--set requires a value" >&2; exit 2
      fi
      [[ -z "$CLI_SELECTION" ]] || { echo "choose exactly one of --set or --layer" >&2; exit 2; }
      CLI_SELECTION=set
      SET_NAME=$2
      TARGET_LAYER_IDS=
      shift 2
      ;;
    --layer)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "--layer requires a value" >&2; exit 2
      fi
      [[ -z "$CLI_SELECTION" ]] || { echo "choose exactly one of --set or --layer" >&2; exit 2; }
      CLI_SELECTION=layer
      TARGET_LAYER_IDS=$2
      SET_NAME=
      shift 2
      ;;
    --phase)
      WRAPPER_PHASE=${2:?--phase requires a value}
      shift 2
      ;;
    --gpu-ids|--gpus)
      GPU_IDS=${2:?$1 requires a value}
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
  spread) TARGET_LAYER_IDS=1,9,17,25,33 ;;
  high_only) TARGET_LAYER_IDS=26,28,30,32,34 ;;
  middle_high) TARGET_LAYER_IDS=17,19,25,33,34 ;;
  middle_high_2) TARGET_LAYER_IDS=17,28,30,32,34 ;;
  shallow_high) TARGET_LAYER_IDS=1,28,30,32,34 ;;
  "") ;;
  *) echo "unknown layer set: $SET_NAME" >&2; exit 2 ;;
esac
[[ -n "$TARGET_LAYER_IDS" ]] || { echo "select a layer set with --set NAME or provide --layer IDS" >&2; exit 2; }
[[ "$MODEL_SIZE" == 3b ]] || { echo "this launcher supports only SPECFORGE_MODEL_SIZE=3b" >&2; exit 2; }
case "$WRAPPER_PHASE" in
  all|phase1|phase2) ;;
  *) echo "--phase must be all, phase1, or phase2" >&2; exit 2 ;;
esac

TARGET_LAYER_IDS=${TARGET_LAYER_IDS//[[:space:]]/}
IFS=',' read -r -a RAW_LAYER_IDS <<< "$TARGET_LAYER_IDS"
if ((${#RAW_LAYER_IDS[@]} != 5)); then
  echo "target layer selection must contain exactly five comma-separated IDs" >&2
  exit 2
fi
NORMALIZED_LAYER_IDS=()
for layer_id in "${RAW_LAYER_IDS[@]}"; do
  if [[ ! "$layer_id" =~ ^[0-9]+$ ]]; then
    echo "target layer IDs must be non-negative integers" >&2
    exit 2
  fi
  numeric_layer_id=$((10#$layer_id))
  if ((numeric_layer_id > 34)); then
    echo "Qwen2.5-VL-3B target layer IDs must be in 0..34; got $numeric_layer_id" >&2
    exit 2
  fi
  for previous_id in "${NORMALIZED_LAYER_IDS[@]}"; do
    ((numeric_layer_id != previous_id)) || { echo "target layer IDs must be unique" >&2; exit 2; }
  done
  NORMALIZED_LAYER_IDS+=("$numeric_layer_id")
done
TARGET_LAYER_IDS=$(IFS=,; printf '%s' "${NORMALIZED_LAYER_IDS[*]}")
LAYER_SLUG=$(IFS=_; printf '%s' "${NORMALIZED_LAYER_IDS[*]}")

GPU_IDS=${GPU_IDS//[[:space:]]/}
[[ -n "$GPU_IDS" ]] || { echo "at least one GPU ID is required" >&2; exit 2; }
IFS=',' read -r -a RAW_GPU_IDS <<< "$GPU_IDS"
if [[ -z "$GPU_COUNT" ]]; then GPU_COUNT=${#RAW_GPU_IDS[@]}; fi
[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "SPECFORGE_GPUS must be a positive integer" >&2; exit 2; }
((GPU_COUNT == ${#RAW_GPU_IDS[@]})) || {
  echo "SPECFORGE_GPUS ($GPU_COUNT) must match GPU IDs ($GPU_IDS)" >&2; exit 2;
}
NORMALIZED_GPU_IDS=()
for gpu_id in "${RAW_GPU_IDS[@]}"; do
  [[ -n "$gpu_id" ]] || { echo "GPU IDs must be comma-separated non-empty values" >&2; exit 2; }
  for previous_id in "${NORMALIZED_GPU_IDS[@]}"; do
    [[ "$gpu_id" != "$previous_id" ]] || { echo "GPU IDs must be unique" >&2; exit 2; }
  done
  NORMALIZED_GPU_IDS+=("$gpu_id")
done

EXPECTED_PHASE1_SAMPLES=${SPECFORGE_PHASE1_NUM_SAMPLES:-${SPECFORGE_NUM_SAMPLES:-68000}}
EXPECTED_PHASE2_SAMPLES=${SPECFORGE_PHASE2_NUM_SAMPLES:-${SPECFORGE_NUM_SAMPLES:-68000}}
PHASE1_MAX_LENGTH=${SPECFORGE_PHASE1_MAX_LENGTH:-2048}
PHASE2_MAX_LENGTH=${SPECFORGE_PHASE2_MAX_LENGTH:-3072}
NUM_EPOCHS=${SPECFORGE_NUM_EPOCHS:-6}
PHASE1_EPOCHS=${SPECFORGE_PHASE1_NUM_EPOCHS:-$NUM_EPOCHS}
PHASE2_EPOCHS=${SPECFORGE_PHASE2_NUM_EPOCHS:-$NUM_EPOCHS}
GLOBAL_BATCH_SIZE=${SPECFORGE_GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${SPECFORGE_MICRO_BATCH_SIZE:-1}
DSPARK_DEPTH=${SPECFORGE_DSPARK_DEPTH:-3}
DSPARK_BLOCK_SIZE=${SPECFORGE_DSPARK_BLOCK_SIZE:-7}
NUM_ANCHORS=${SPECFORGE_NUM_ANCHORS:-512}
OBJECTIVE_CHUNK_BLOCKS=${SPECFORGE_OBJECTIVE_CHUNK_BLOCKS:-32}
DATALOADER_WORKERS=${SPECFORGE_DATALOADER_WORKERS:-8}
ATTENTION_BACKEND=${SPECFORGE_ATTENTION_BACKEND:-flex_attention}
FSDP_SHARDING=${SPECFORGE_FSDP_SHARDING:-NO_SHARD}
OPTIMIZER_CPU_OFFLOAD=${SPECFORGE_OPTIMIZER_CPU_OFFLOAD:-0}
LEARNING_RATE=${SPECFORGE_LEARNING_RATE:-6e-4}
WARMUP_RATIO=${SPECFORGE_WARMUP_RATIO:-0.04}
LOSS_DECAY_GAMMA=${SPECFORGE_LOSS_DECAY_GAMMA:-4.0}
CE_LOSS_ALPHA=${SPECFORGE_DSPARK_CE_LOSS_ALPHA:-0.1}
L1_LOSS_ALPHA=${SPECFORGE_DSPARK_L1_LOSS_ALPHA:-0.9}
CONFIDENCE_LOSS_ALPHA=${SPECFORGE_DSPARK_CONFIDENCE_HEAD_ALPHA:-1.0}
SAVE_INTERVAL=${SPECFORGE_SAVE_INTERVAL:-1000}
MAX_CHECKPOINTS=${SPECFORGE_MAX_CHECKPOINTS:-3}
LOG_INTERVAL=${SPECFORGE_LOG_INTERVAL:-100}
VALIDATE_SAMPLE_LIMIT=${SPECFORGE_CACHE_VALIDATE_SAMPLE_LIMIT:-32}
SKIP_GPU_CHECK=${SPECFORGE_SKIP_GPU_CHECK:-0}
PHASE1_FEATURE_ROOT=${SPECFORGE_PHASE1_FEATURE_ROOT:-}
PHASE2_FEATURE_ROOT=${SPECFORGE_PHASE2_FEATURE_ROOT:-}

for name in EXPECTED_PHASE1_SAMPLES EXPECTED_PHASE2_SAMPLES PHASE1_MAX_LENGTH PHASE2_MAX_LENGTH PHASE1_EPOCHS PHASE2_EPOCHS GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE DSPARK_DEPTH NUM_ANCHORS DATALOADER_WORKERS SAVE_INTERVAL MAX_CHECKPOINTS LOG_INTERVAL; do
  value=${!name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer, got $value" >&2; exit 2; }
done
[[ "$DSPARK_BLOCK_SIZE" =~ ^[0-9]+$ ]] && ((DSPARK_BLOCK_SIZE >= 2)) || { echo "SPECFORGE_DSPARK_BLOCK_SIZE must be an integer >= 2" >&2; exit 2; }
[[ "$OBJECTIVE_CHUNK_BLOCKS" =~ ^[0-9]+$ ]] || { echo "SPECFORGE_OBJECTIVE_CHUNK_BLOCKS must be a non-negative integer" >&2; exit 2; }
[[ "$VALIDATE_SAMPLE_LIMIT" =~ ^[0-9]+$ ]] || { echo "SPECFORGE_CACHE_VALIDATE_SAMPLE_LIMIT must be non-negative" >&2; exit 2; }
case "$ATTENTION_BACKEND" in flex_attention|sdpa|eager) ;; *) echo "SPECFORGE_ATTENTION_BACKEND must be flex_attention, sdpa, or eager" >&2; exit 2 ;; esac
case "$FSDP_SHARDING" in NO_SHARD|SHARD_GRAD_OP|FULL_SHARD) ;; *) echo "SPECFORGE_FSDP_SHARDING must be NO_SHARD, SHARD_GRAD_OP, or FULL_SHARD" >&2; exit 2 ;; esac
case "$OPTIMIZER_CPU_OFFLOAD" in 0|1) ;; *) echo "SPECFORGE_OPTIMIZER_CPU_OFFLOAD must be 0 or 1" >&2; exit 2 ;; esac
if [[ "$OPTIMIZER_CPU_OFFLOAD" == 1 ]]; then
  OPTIMIZER_CPU_OFFLOAD=true
else
  OPTIMIZER_CPU_OFFLOAD=false
fi
case "$SKIP_GPU_CHECK" in 0|1) ;; *) echo "SPECFORGE_SKIP_GPU_CHECK must be 0 or 1" >&2; exit 2 ;; esac
if ((GLOBAL_BATCH_SIZE % (GPU_COUNT * MICRO_BATCH_SIZE) != 0)); then
  echo "global batch must be divisible by GPUs x micro batch" >&2
  exit 2
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (GPU_COUNT * MICRO_BATCH_SIZE)))

SOURCE_CACHE_ROOT=${SPECFORGE_DFLASH_ARTIFACT_ROOT:-${DFLASH_ARTIFACT_ROOT:-${ARTIFACT_ROOT:-"$REPO_ROOT/src/train_Dflash_SpecForge/artifacts"}}}
CACHE_JOB_ROOT="$SOURCE_CACHE_ROOT/dflash_3layer_${LAYER_SLUG}"
PHASE1_FEATURE_ROOT=${PHASE1_FEATURE_ROOT:-"$CACHE_JOB_ROOT/phase1/qwen25vl_3b/hidden_states"}
PHASE2_FEATURE_ROOT=${PHASE2_FEATURE_ROOT:-"$CACHE_JOB_ROOT/phase2/hidden_states"}
ARTIFACT_ROOT=${SPECFORGE_ARTIFACT_ROOT:-"$ROOT_DIR/artifacts"}
OUTPUT_ROOT=${SPECFORGE_OUTPUT_ROOT:-"$ROOT_DIR/outputs"}
DS_PERSIST_ROOT="$ARTIFACT_ROOT/dspark_targets_${LAYER_SLUG}"
DS_OUTPUT_ROOT="$OUTPUT_ROOT/dspark_targets_${LAYER_SLUG}"
PHASE1_RUN_ID="qwen25vl-3b-dspark-sharegpt68k-dspark_targets_${LAYER_SLUG}"
PHASE2_RUN_ID="qwen25vl-3b-dspark-llava68k-dspark_targets_${LAYER_SLUG}"
PHASE1_RUN_OUTPUT="$DS_OUTPUT_ROOT/phase1/$PHASE1_RUN_ID"
PHASE2_RUN_OUTPUT="$DS_OUTPUT_ROOT/phase2/$PHASE2_RUN_ID"
DEFAULT_PHASE1_CHECKPOINT="$PHASE1_RUN_OUTPUT/$PHASE1_RUN_ID-latest"
EXPLICIT_PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-}
if [[ "$WRAPPER_PHASE" == all && -n "$EXPLICIT_PHASE1_CHECKPOINT" && \
      "$EXPLICIT_PHASE1_CHECKPOINT" != "$DEFAULT_PHASE1_CHECKPOINT" ]]; then
  echo "--phase all uses this run's Phase 1 checkpoint; use --phase phase2 to select another checkpoint" >&2
  exit 2
fi
PHASE1_CHECKPOINT=${EXPLICIT_PHASE1_CHECKPOINT:-$DEFAULT_PHASE1_CHECKPOINT}
PHASE1_DATA_CACHE="$DS_PERSIST_ROOT/phase1/cache"
PHASE2_DATA_CACHE="$DS_PERSIST_ROOT/phase2/cache"

print_config() {
  local selection_args
  if [[ -n "$SET_NAME" ]]; then
    selection_args="--set $SET_NAME"
  else
    selection_args="--layer $TARGET_LAYER_IDS"
  fi
  printf 'LAYER_SET=%s\n' "${SET_NAME:-custom}"
  printf 'TARGET_LAYER_IDS=%s\n' "$TARGET_LAYER_IDS"
  printf 'DSPARK_DEPTH=%s\n' "$DSPARK_DEPTH"
  printf 'DSPARK_BLOCK_SIZE=%s\n' "$DSPARK_BLOCK_SIZE"
  printf 'WRAPPER_PHASE=%s\n' "$WRAPPER_PHASE"
  printf 'GPU_IDS=%s\nGPU_COUNT=%s\n' "$GPU_IDS" "$GPU_COUNT"
  printf 'PHASE1_FEATURE_ROOT=%s\n' "$PHASE1_FEATURE_ROOT"
  printf 'PHASE2_FEATURE_ROOT=%s\n' "$PHASE2_FEATURE_ROOT"
  printf 'PHASE1_RUN_OUTPUT=%s\n' "$PHASE1_RUN_OUTPUT"
  printf 'PHASE2_RUN_OUTPUT=%s\n' "$PHASE2_RUN_OUTPUT"
  printf 'PHASE1_CHECKPOINT=%s\n' "$PHASE1_CHECKPOINT"
  printf 'EXPECTED_PHASE1_SAMPLES=%s\n' "$EXPECTED_PHASE1_SAMPLES"
  printf 'EXPECTED_PHASE2_SAMPLES=%s\n' "$EXPECTED_PHASE2_SAMPLES"
  printf 'GLOBAL_BATCH_SIZE=%s\nMICRO_BATCH_SIZE=%s\nACCUMULATION_STEPS=%s\n' \
    "$GLOBAL_BATCH_SIZE" "$MICRO_BATCH_SIZE" "$ACCUMULATION_STEPS"
  if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase1 ]]; then
    printf 'PHASE1_COMMAND=bash %s --phase phase1 %s --gpu-ids %s\n' \
      "$ROOT_DIR/train_qwen25vl_dspark_two_phase.sh" "$selection_args" "$GPU_IDS"
  fi
  if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase2 ]]; then
    printf 'PHASE2_COMMAND=bash %s --phase phase2 %s --gpu-ids %s\n' \
      "$ROOT_DIR/train_qwen25vl_dspark_two_phase.sh" "$selection_args" "$GPU_IDS"
  fi
}

if ((PRINT_CONFIG)); then
  print_config
  exit 0
fi

[[ -n "$TARGET_MODEL_PATH" ]] || { echo "set TARGET_MODEL_PATH or MODEL_3B" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

if [[ "$WRAPPER_PHASE" == phase2 && ! -e "$PHASE1_CHECKPOINT" ]]; then
  echo "Phase 1 DSpark checkpoint not found: $PHASE1_CHECKPOINT" >&2
  exit 1
fi

verify_gpus() {
  if ((SKIP_GPU_CHECK)); then
    echo "[dspark] skipping GPU count check (SPECFORGE_SKIP_GPU_CHECK=1)"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
import torch

count = torch.cuda.device_count()
if count < 1:
    raise SystemExit("no CUDA GPUs are visible to the selected Python environment")
print(f"[dspark] torch={torch.__version__} cuda={torch.version.cuda} visible_gpus={count}")
PY
  visible_count=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
  [[ "$visible_count" == "$GPU_COUNT" ]] || {
    echo "expected $GPU_COUNT visible GPUs from --gpu-ids $GPU_IDS, found $visible_count" >&2
    exit 1
  }
}

resolve_config() {
  local phase=$1 output=$2
  mkdir -p "$(dirname "$output")"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/resolve_dspark_config.py" \
    --input "$ROOT_DIR/configs/qwen2.5-vl-3b-dspark.json" \
    --output "$output" \
    --phase "$phase" \
    --target-layer-ids "$TARGET_LAYER_IDS" \
    --draft-num-hidden-layers "$DSPARK_DEPTH" \
    --block-size "$DSPARK_BLOCK_SIZE"
}

validate_cache() {
  local phase=$1 feature_root=$2 config_path=$3 expected=$4
  [[ -d "$feature_root" ]] || { echo "$phase feature cache not found: $feature_root" >&2; exit 1; }
  "$PYTHON_BIN" "$ROOT_DIR/scripts/validate_dspark_cache.py" \
    --feature-root "$feature_root" \
    --draft-model-config "$config_path" \
    --phase "$phase" \
    --expected-samples "$expected" \
    --sample-limit "$VALIDATE_SAMPLE_LIMIT"
}

phase_max_steps() {
  local sample_count=$1 epochs=$2
  local samples_per_rank micro_batches steps
  samples_per_rank=$(((sample_count + GPU_COUNT - 1) / GPU_COUNT))
  micro_batches=$((samples_per_rank / MICRO_BATCH_SIZE))
  steps=$((micro_batches * epochs / ACCUMULATION_STEPS))
  ((steps > 0)) || { echo "cache is too small to produce one optimizer step" >&2; exit 1; }
  printf '%s' "$steps"
}

preflight_phase() {
  local phase=$1 feature_root=$2 expected=$3
  local phase_config="$DS_PERSIST_ROOT/$phase/draft_config.json"
  resolve_config "$phase" "$phase_config"
  validate_cache "$phase" "$feature_root" "$phase_config" "$expected"
}

train_phase() {
  local phase=$1 feature_root=$2 expected=$3 max_length=$4 epochs=$5
  local run_id=$6 run_output=$7 data_cache=$8
  local phase_config="$DS_PERSIST_ROOT/$phase/draft_config.json"
  local max_steps latest
  max_steps=$(phase_max_steps "$expected" "$epochs")
  mkdir -p "$data_cache" "$(dirname "$run_output")"

  latest="$run_output/$run_id-latest"
  local train_args=(
    --config "$ROOT_DIR/examples/configs/qwen2.5-vl-3b-dspark-offline.yaml"
    "model.target_model_path=$TARGET_MODEL_PATH"
    "model.draft_model_config=$phase_config"
    "model.input_modality=qwen2_5_vl"
    "model.draft_warm_start_mode=auto"
    "data.hidden_states_path=$feature_root"
    "data.hidden_state_phase=$phase"
    "data.cache_dir=$data_cache"
    "data.max_length=$max_length"
    "data.dataloader_num_workers=$DATALOADER_WORKERS"
    "training.num_epochs=$epochs"
    "training.max_steps=$max_steps"
    "training.batch_size=$MICRO_BATCH_SIZE"
    "training.accumulation_steps=$ACCUMULATION_STEPS"
    "training.learning_rate=$LEARNING_RATE"
    "training.warmup_ratio=$WARMUP_RATIO"
    "training.loss_decay_gamma=$LOSS_DECAY_GAMMA"
    "training.dspark_ce_loss_alpha=$CE_LOSS_ALPHA"
    "training.dspark_l1_loss_alpha=$L1_LOSS_ALPHA"
    "training.dspark_confidence_head_alpha=$CONFIDENCE_LOSS_ALPHA"
    "training.num_anchors=$NUM_ANCHORS"
    "training.objective_chunk_blocks=$OBJECTIVE_CHUNK_BLOCKS"
    "training.attention_backend=$ATTENTION_BACKEND"
    "training.fsdp_sharding=$FSDP_SHARDING"
    "training.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD"
    "training.save_interval=$SAVE_INTERVAL"
    "training.max_checkpoints=$MAX_CHECKPOINTS"
    "training.log_interval=$LOG_INTERVAL"
    "deployment.trainer.nproc_per_node=$GPU_COUNT"
    "run_id=$run_id"
    "output_dir=$run_output"
  )

  if [[ "$phase" == phase2 ]]; then
    [[ -e "$PHASE1_CHECKPOINT" ]] || {
      echo "Phase 1 DSpark checkpoint not found: $PHASE1_CHECKPOINT" >&2
      exit 1
    }
  fi
  if ((RESUME)); then
    if [[ -e "$latest" ]]; then
      train_args+=("training.resume_from=$run_output")
    else
      echo "[dspark:$phase] no checkpoint exists; starting a new run"
      if [[ "$phase" == phase2 ]]; then
        train_args+=("model.draft_checkpoint_path=$PHASE1_CHECKPOINT")
      fi
    fi
  elif [[ -e "$latest" ]]; then
    echo "checkpoint already exists at $latest; pass --resume or choose another SPECFORGE_OUTPUT_ROOT" >&2
    exit 1
  elif [[ "$phase" == phase2 ]]; then
    train_args+=("model.draft_checkpoint_path=$PHASE1_CHECKPOINT")
  fi

  echo "[dspark:$phase] layers=$TARGET_LAYER_IDS depth=$DSPARK_DEPTH block=$DSPARK_BLOCK_SIZE samples=$expected max_steps=$max_steps GPUs=$GPU_COUNT micro=$MICRO_BATCH_SIZE accumulation=$ACCUMULATION_STEPS output=$run_output"
  (cd "$ROOT_DIR" && "$PYTHON_BIN" -m specforge.cli train "${train_args[@]}")
}

if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase1 ]]; then
  preflight_phase phase1 "$PHASE1_FEATURE_ROOT" "$EXPECTED_PHASE1_SAMPLES"
fi
if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase2 ]]; then
  preflight_phase phase2 "$PHASE2_FEATURE_ROOT" "$EXPECTED_PHASE2_SAMPLES"
fi

verify_gpus
if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase1 ]]; then
  train_phase phase1 "$PHASE1_FEATURE_ROOT" "$EXPECTED_PHASE1_SAMPLES" \
    "$PHASE1_MAX_LENGTH" "$PHASE1_EPOCHS" "$PHASE1_RUN_ID" \
    "$PHASE1_RUN_OUTPUT" "$PHASE1_DATA_CACHE"
fi
if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase2 ]]; then
  train_phase phase2 "$PHASE2_FEATURE_ROOT" "$EXPECTED_PHASE2_SAMPLES" \
    "$PHASE2_MAX_LENGTH" "$PHASE2_EPOCHS" "$PHASE2_RUN_ID" \
    "$PHASE2_RUN_OUTPUT" "$PHASE2_DATA_CACHE"
fi
