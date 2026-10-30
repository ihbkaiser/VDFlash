#!/usr/bin/env bash
set -euo pipefail

# End-to-end two-B200 DSpark run: prepare ShareGPT, capture DSpark features for
# both phases, then train Phase 1 and warm-start Phase 2 from its checkpoint.
# Machine-specific paths are loaded from the ignored DFlash environment file.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$ROOT_DIR/../.." && pwd)
ENV_FILE=${SPECFORGE_ENV_FILE:-"$REPO_ROOT/src/train_Dflash_SpecForge/train_qwen25vl_dflash_3layer.env"}
ARGS=("$@")

for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --env-file)
      ((i + 1 < ${#ARGS[@]})) || { echo "--env-file requires a path" >&2; exit 2; }
      ENV_FILE=${ARGS[i + 1]}
      ;;
    --env-file=*) ENV_FILE=${ARGS[i]#--env-file=} ;;
  esac
done

[[ -f "$ENV_FILE" ]] || {
  echo "environment file not found: $ENV_FILE" >&2
  echo "set SPECFORGE_ENV_FILE or place the ignored DFlash .env at its default path" >&2
  exit 2
}
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

RESUME=0
PRINT_CONFIG=0
CLI_LAYER_SET=
CLI_TEACHER_LAYERS=
while (($#)); do
  case "$1" in
    --env-file) shift 2 ;;
    --env-file=*) shift ;;
    --resume) RESUME=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    --layer-set|--set)
      (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
      CLI_LAYER_SET=$2
      shift 2
      ;;
    --layer-set=*|--set=*) CLI_LAYER_SET=${1#*=}; shift ;;
    --teacher-layers|--layer)
      (($# >= 2)) || { echo "$1 requires a value" >&2; exit 2; }
      CLI_TEACHER_LAYERS=$2
      shift 2
      ;;
    --teacher-layers=*|--layer=*) CLI_TEACHER_LAYERS=${1#*=}; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: bash src/train_DSpark_SpecForge/train_qwen25vl_dspark_3layer_b200.sh [options]

Runs data preparation, DSpark feature capture for both phases, then two-phase
training. The DSpark draft depth is 3. Phase 2 warm-starts from Phase 1.

Teacher-layer sets (five target hidden-state taps each):
  spread       [1,9,17,25,33]
  high_only    [26,28,30,32,34] (default)
  middle_high  [17,19,25,33,34]

Options:
  --env-file FILE   Source the local machine environment (default: the ignored
                    train_qwen25vl_dflash_3layer.env beside the DFlash scripts).
  --resume          Resume training checkpoints and continue partial captures.
  --layer-set NAME  Select spread, high_only, or middle_high.
  --teacher-layers IDS
                   Custom ordered five layer IDs, e.g. 1,9,17,25,33.
  --print-config    Print resolved paths without starting data/capture/training.

Useful overrides: SPECFORGE_GPU_IDS, SPECFORGE_LAYER_SET,
SPECFORGE_TARGET_LAYER_IDS, SPECFORGE_NUM_SAMPLES,
SPECFORGE_CAPTURE_BATCH_SIZE,
SPECFORGE_GLOBAL_BATCH_SIZE, SPECFORGE_NUM_EPOCHS, and PYTHON_BIN.
EOF
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PYTHON_BIN=${PYTHON_BIN:-python3}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-${MODEL_3B:-}}
export TARGET_MODEL_PATH
[[ ${SPECFORGE_MODEL_SIZE:-3b} == 3b ]] || {
  echo "this launcher is for Qwen2.5-VL-3B; set SPECFORGE_MODEL_SIZE=3b" >&2
  exit 2
}
GPU_IDS=${SPECFORGE_GPU_IDS:-0,1}
GPU_IDS=${GPU_IDS//[[:space:]]/}
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
[[ ${#GPU_ID_ARRAY[@]} -eq 2 && -n ${GPU_ID_ARRAY[0]} && -n ${GPU_ID_ARRAY[1]} && ${GPU_ID_ARRAY[0]} != "${GPU_ID_ARRAY[1]}" ]] || {
  echo "this launcher requires two distinct GPU IDs, got: $GPU_IDS" >&2
  exit 2
}
GPU_COUNT=2
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export SPECFORGE_GPU_IDS="$GPU_IDS"
export SPECFORGE_GPUS=2
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_DATA=${SOURCE_DATA:-}
SOURCE_JSONL=${SOURCE_JSONL:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
BASE_ARTIFACT_ROOT=${ARTIFACT_ROOT:-${DFLASH_ARTIFACT_ROOT:-}}
BASE_OUTPUT_ROOT=${OUTPUT_ROOT:-}
[[ -n "$TARGET_MODEL_PATH" ]] || { echo "MODEL_3B or TARGET_MODEL_PATH must be set in $ENV_FILE" >&2; exit 2; }
[[ -n "$SOURCE_DATA" ]] || { echo "SOURCE_DATA must be set in $ENV_FILE" >&2; exit 2; }
[[ -n "$SOURCE_JSONL" ]] || { echo "SOURCE_JSONL must be set in $ENV_FILE" >&2; exit 2; }
[[ -n "$IMAGE_ROOT" ]] || { echo "IMAGE_ROOT must be set in $ENV_FILE" >&2; exit 2; }
[[ -n "$BASE_ARTIFACT_ROOT" ]] || { echo "ARTIFACT_ROOT must be set in $ENV_FILE" >&2; exit 2; }
[[ -n "$BASE_OUTPUT_ROOT" ]] || BASE_OUTPUT_ROOT="$BASE_ARTIFACT_ROOT/outputs"

if [[ -n "$CLI_LAYER_SET" && -n "$CLI_TEACHER_LAYERS" ]]; then
  echo "choose either --layer-set or --teacher-layers" >&2
  exit 2
fi
if [[ -n "$CLI_LAYER_SET" ]]; then
  LAYER_SET=$CLI_LAYER_SET
  TARGET_LAYER_IDS=
  LAYER_SELECTION=set
elif [[ -n "$CLI_TEACHER_LAYERS" ]]; then
  LAYER_SET=custom
  TARGET_LAYER_IDS=$CLI_TEACHER_LAYERS
  LAYER_SELECTION=layer
else
  LAYER_SET=${SPECFORGE_LAYER_SET:-high_only}
  TARGET_LAYER_IDS=${SPECFORGE_TARGET_LAYER_IDS:-}
  if [[ -n "$TARGET_LAYER_IDS" ]]; then
    LAYER_SET=custom
    LAYER_SELECTION=layer
  else
    LAYER_SELECTION=set
  fi
fi
if [[ "$LAYER_SELECTION" == set ]]; then
  case "$LAYER_SET" in
    spread) TARGET_LAYER_IDS=1,9,17,25,33 ;;
    high_only) TARGET_LAYER_IDS=26,28,30,32,34 ;;
    middle_high) TARGET_LAYER_IDS=17,19,25,33,34 ;;
    *) echo "unknown teacher layer set: $LAYER_SET" >&2; exit 2 ;;
  esac
fi
TARGET_LAYER_IDS=$(printf '%s' "$TARGET_LAYER_IDS" | tr -d '[][:space:]')
IFS=',' read -r -a LAYER_ARRAY <<< "$TARGET_LAYER_IDS"
[[ ${#LAYER_ARRAY[@]} -eq 5 ]] || { echo "DSpark requires five target layer IDs, got: $TARGET_LAYER_IDS" >&2; exit 2; }
declare -A SEEN_LAYERS=()
NORMALIZED_LAYER_IDS=()
for layer_id in "${LAYER_ARRAY[@]}"; do
  [[ "$layer_id" =~ ^[0-9]+$ ]] || { echo "invalid target layer ID: $layer_id" >&2; exit 2; }
  numeric_layer_id=$((10#$layer_id))
  ((numeric_layer_id <= 34)) || { echo "Qwen2.5-VL-3B target layer IDs must be in 0..34" >&2; exit 2; }
  [[ -z ${SEEN_LAYERS[$numeric_layer_id]+x} ]] || { echo "target layer IDs must be unique" >&2; exit 2; }
  SEEN_LAYERS[$numeric_layer_id]=1
  NORMALIZED_LAYER_IDS+=("$numeric_layer_id")
done
TARGET_LAYER_IDS=$(IFS=,; printf '%s' "${NORMALIZED_LAYER_IDS[*]}")
LAYER_SLUG=$(IFS=_; printf '%s' "${NORMALIZED_LAYER_IDS[*]}")

NUM_SAMPLES=${SPECFORGE_NUM_SAMPLES:-68000}
PHASE1_MAX_LENGTH=${SPECFORGE_PHASE1_MAX_LENGTH:-2048}
PHASE2_MAX_LENGTH=${SPECFORGE_PHASE2_MAX_LENGTH:-3072}
[[ ${SPECFORGE_DSPARK_DEPTH:-3} == 3 ]] || {
  echo "this launcher fixes the DSpark draft depth at 3 transformer layers" >&2
  exit 2
}
DSPARK_DEPTH=3
DSPARK_BLOCK_SIZE=${SPECFORGE_DSPARK_BLOCK_SIZE:-7}
CAPTURE_BATCH_SIZE=${SPECFORGE_CAPTURE_BATCH_SIZE:-8}
CAPTURE_WORKERS=${SPECFORGE_CAPTURE_PREPROCESS_WORKERS:-8}
CAPTURE_QUEUE=${SPECFORGE_CAPTURE_PREPROCESS_QUEUE:-32}
CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-8}
CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-64}
DATASET_NUM_PROC=${SPECFORGE_DATASET_NUM_PROC:-8}
SGLANG_MEM_FRACTION=${SPECFORGE_SGLANG_MEM_FRACTION_STATIC:-0.4}
DS_ARTIFACT_ROOT=${SPECFORGE_DSPARK_ARTIFACT_ROOT:-"$BASE_ARTIFACT_ROOT/dspark_depth${DSPARK_DEPTH}_layers_${LAYER_SLUG}"}
DS_OUTPUT_ROOT=${SPECFORGE_DSPARK_OUTPUT_ROOT:-"$BASE_OUTPUT_ROOT/dspark_depth${DSPARK_DEPTH}_layers_${LAYER_SLUG}"}
SHAREGPT_DATA="$BASE_ARTIFACT_ROOT/shared/sharegpt_train.jsonl"
PHASE1_FEATURE_ROOT="$DS_ARTIFACT_ROOT/phase1/hidden_states"
PHASE2_FEATURE_ROOT="$DS_ARTIFACT_ROOT/phase2/hidden_states"
PHASE1_CONFIG="$DS_ARTIFACT_ROOT/configs/draft_config_phase1.json"
PHASE2_CONFIG="$DS_ARTIFACT_ROOT/configs/draft_config_phase2.json"
MANIFEST="$DS_ARTIFACT_ROOT/phase2/manifest.jsonl"

if ((PRINT_CONFIG)); then
  printf 'ENV_FILE=%s\nTARGET_MODEL_PATH=%s\nGPU_IDS=%s\n' "$ENV_FILE" "$TARGET_MODEL_PATH" "$GPU_IDS"
  printf 'LAYER_SET=%s\nTARGET_LAYER_IDS=%s\nDSPARK_DEPTH=%s\nDSPARK_BLOCK_SIZE=%s\n' "$LAYER_SET" "$TARGET_LAYER_IDS" "$DSPARK_DEPTH" "$DSPARK_BLOCK_SIZE"
  printf 'SHAREGPT_DATA=%s\nPHASE1_FEATURE_ROOT=%s\nPHASE2_FEATURE_ROOT=%s\n' "$SHAREGPT_DATA" "$PHASE1_FEATURE_ROOT" "$PHASE2_FEATURE_ROOT"
  printf 'PHASE1_CONFIG=%s\nPHASE2_CONFIG=%s\nMANIFEST=%s\n' "$PHASE1_CONFIG" "$PHASE2_CONFIG" "$MANIFEST"
  printf 'OUTPUT_ROOT=%s\nNUM_SAMPLES=%s\n' "$DS_OUTPUT_ROOT" "$NUM_SAMPLES"
  exit 0
fi

[[ -f "$SOURCE_DATA" ]] || { echo "ShareGPT source file not found: $SOURCE_DATA" >&2; exit 1; }
[[ -f "$SOURCE_JSONL" ]] || { echo "LLaVA source JSONL not found: $SOURCE_JSONL" >&2; exit 1; }
[[ -d "$IMAGE_ROOT" ]] || { echo "LLaVA image root not found: $IMAGE_ROOT" >&2; exit 1; }

command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
visible_gpus=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
[[ "$visible_gpus" == "$GPU_COUNT" ]] || {
  echo "expected two visible GPUs after CUDA_VISIBLE_DEVICES=$GPU_IDS, found $visible_gpus" >&2
  exit 1
}
mkdir -p "$DS_ARTIFACT_ROOT/configs" "$DS_OUTPUT_ROOT"

resolve_config() {
  local phase=$1 destination=$2
  "$PYTHON_BIN" "$ROOT_DIR/scripts/resolve_dspark_config.py" \
    --input "$ROOT_DIR/configs/qwen2.5-vl-3b-dspark.json" \
    --output "$destination" \
    --phase "$phase" \
    --target-layer-ids "$TARGET_LAYER_IDS" \
    --draft-num-hidden-layers "$DSPARK_DEPTH" \
    --block-size "$DSPARK_BLOCK_SIZE"
}

feature_count() {
  if [[ ! -d "$1" ]]; then
    printf '0'
  else
    find "$1" -type f \( -name '*.ckpt' -o -name '*.ckpt.gz' \) -print | wc -l | tr -d '[:space:]'
  fi
}

resolve_config phase1 "$PHASE1_CONFIG"
resolve_config phase2 "$PHASE2_CONFIG"

echo "[dspark] preparing ShareGPT records"
bash "$REPO_ROOT/src/train_Dflash_SpecForge/train_qwen25vl_dflash_sharegpt_68k.sh" \
  --env-file "$ENV_FILE" --models 3b --phase data
[[ -s "$SHAREGPT_DATA" ]] || { echo "converted ShareGPT file missing: $SHAREGPT_DATA" >&2; exit 1; }

mkdir -p "$PHASE1_FEATURE_ROOT" "$DS_ARTIFACT_ROOT/phase1/cache"
echo "[dspark:phase1-capture] GPUs=$GPU_COUNT batch/rank=$CAPTURE_BATCH_SIZE max_length=$PHASE1_MAX_LENGTH"
(
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    "$ROOT_DIR/scripts/prepare_hidden_states.py" \
    --strategy dspark \
    --input-modality qwen2_5_vl \
    --phase phase1 \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$PHASE1_CONFIG" \
    --data-path "$SHAREGPT_DATA" \
    --output-path "$PHASE1_FEATURE_ROOT" \
    --cache-dir "$DS_ARTIFACT_ROOT/phase1/cache" \
    --chat-template qwen \
    --train-only-last-turn \
    --max-length "$PHASE1_MAX_LENGTH" \
    --num-samples "$NUM_SAMPLES" \
    --build-dataset-num-proc "$DATASET_NUM_PROC" \
    --tp-size 1 \
    --batch-size "$CAPTURE_BATCH_SIZE" \
    --num-workers "$CAPTURE_WORKERS" \
    --num-io-threads "$CAPTURE_IO_THREADS" \
    --io-queue-size "$CAPTURE_IO_QUEUE" \
    --file-group-size 2000 \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION" \
    --sglang-disable-radix-cache
)
PHASE1_COUNT=$(feature_count "$PHASE1_FEATURE_ROOT")
((PHASE1_COUNT > 0)) || { echo "Phase 1 DSpark capture produced no feature records" >&2; exit 1; }

echo "[dspark] preparing LLaVA manifest"
mkdir -p "$(dirname "$MANIFEST")" "$PHASE2_FEATURE_ROOT"
"$PYTHON_BIN" "$ROOT_DIR/../train_Dflash_SpecForge/scripts/prepare_llava_caption_manifest.py" \
  --input "$SOURCE_JSONL" \
  --output "$MANIFEST" \
  --image-root "$IMAGE_ROOT" \
  --expected-records "$NUM_SAMPLES"

echo "[dspark:phase2-capture] GPUs=$GPU_COUNT batch/rank=$CAPTURE_BATCH_SIZE max_length=$PHASE2_MAX_LENGTH"
(
  cd "$ROOT_DIR"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    "$ROOT_DIR/scripts/prepare_llava_caption_hidden_states.py" \
    --strategy dspark \
    --phase phase2 \
    --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$PHASE2_CONFIG" \
    --manifest "$MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --output-path "$PHASE2_FEATURE_ROOT" \
    --max-length "$PHASE2_MAX_LENGTH" \
    --expected-records "$NUM_SAMPLES" \
    --tp-size 1 \
    --batch-size "$CAPTURE_BATCH_SIZE" \
    --num-preprocess-workers "$CAPTURE_WORKERS" \
    --preprocess-queue-size "$CAPTURE_QUEUE" \
    --num-io-threads "$CAPTURE_IO_THREADS" \
    --io-queue-size "$CAPTURE_IO_QUEUE" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION"
)
PHASE2_COUNT=$(feature_count "$PHASE2_FEATURE_ROOT")
((PHASE2_COUNT > 0)) || { echo "Phase 2 DSpark capture produced no feature records" >&2; exit 1; }

export SPECFORGE_DSPARK_DEPTH="$DSPARK_DEPTH"
export SPECFORGE_DSPARK_BLOCK_SIZE="$DSPARK_BLOCK_SIZE"
export SPECFORGE_PHASE1_FEATURE_ROOT="$PHASE1_FEATURE_ROOT"
export SPECFORGE_PHASE2_FEATURE_ROOT="$PHASE2_FEATURE_ROOT"
export SPECFORGE_PHASE1_NUM_SAMPLES="$PHASE1_COUNT"
export SPECFORGE_PHASE2_NUM_SAMPLES="$PHASE2_COUNT"
export SPECFORGE_PHASE1_MAX_LENGTH="$PHASE1_MAX_LENGTH"
export SPECFORGE_PHASE2_MAX_LENGTH="$PHASE2_MAX_LENGTH"
export SPECFORGE_ARTIFACT_ROOT="$DS_ARTIFACT_ROOT/training"
export SPECFORGE_OUTPUT_ROOT="$DS_OUTPUT_ROOT"

if [[ "$LAYER_SELECTION" == set ]]; then
  train_args=(--set "$LAYER_SET" --gpu-ids "$GPU_IDS" --env-file "$ENV_FILE")
else
  train_args=(--layer "$TARGET_LAYER_IDS" --gpu-ids "$GPU_IDS" --env-file "$ENV_FILE")
fi
if ((RESUME)); then
  train_args+=(--resume)
fi
echo "[dspark] training Phase 1 samples=$PHASE1_COUNT, then warm-starting Phase 2 samples=$PHASE2_COUNT; draft depth=$DSPARK_DEPTH on GPUs=$GPU_IDS"
bash "$ROOT_DIR/train_qwen25vl_dspark_two_phase.sh" "${train_args[@]}"
