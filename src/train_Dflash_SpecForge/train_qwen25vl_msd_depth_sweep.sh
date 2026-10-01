#!/usr/bin/env bash
set -euo pipefail

SPECFORGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

DEPTHS=1,3,5
DEPTHS=${SPECFORGE_MSD_DEPTHS:-$DEPTHS}
TOTAL_EPOCHS=40
GLOBAL_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=${SPECFORGE_GLOBAL_BATCH_SIZE:-$GLOBAL_BATCH_SIZE}
LEARNING_RATE=5e-5
LEARNING_RATE=${SPECFORGE_LEARNING_RATE:-$LEARNING_RATE}
GPU_COUNT=${SPECFORGE_GPUS:-4}
MICRO_BATCH_SIZE=${SPECFORGE_MICRO_BATCH_SIZE:-1}
EXPECTED_RECORDS=${SPECFORGE_NUM_SAMPLES:-68000}
MAX_LENGTH=${SPECFORGE_MAX_LENGTH:-2048}
PHASE=all
RESUME=0
PRINT_CONFIG=0

TARGET_MODEL_PATH=${TARGET_MODEL_PATH-/models/qwen25-vl-3b}
SHAREGPT_SOURCE=${SHAREGPT_SOURCE-/data/sharegpt68k.json}
SHAREGPT_JSONL=${SHAREGPT_JSONL-/data/msd/manifests/sharegpt_train.jsonl}
LLAVA_MANIFEST=${LLAVA_MANIFEST-/data/msd/manifests/llava68k.jsonl}
IMAGE_ROOT=${IMAGE_ROOT-/data/images}
TEXT_FEATURE_ROOT=${TEXT_FEATURE_ROOT-/data/msd/features/sharegpt68k}
VISUAL_FEATURE_ROOT=${VISUAL_FEATURE_ROOT-/data/msd/features/llava68k}
OUTPUT_ROOT=${OUTPUT_ROOT-"$SPECFORGE_DIR/outputs/qwen25vl-3b-msd"}
GENERATED_ROOT=${GENERATED_ROOT-"$OUTPUT_ROOT/generated"}

usage() {
  printf '%s\n' \
    'Qwen2.5-VL-3B original-MSD 1/3/5-layer static sweep launcher.' \
    'Usage: bash train_qwen25vl_msd_depth_sweep.sh [options]' \
    '  --phase data|capture|train|all' \
    '  --resume' \
    '  --print-config' \
    '  -h, --help'
}

while (($#)); do
  case "$1" in
    --phase) PHASE=${2:?"--phase requires a value"}; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PHASE" in data|capture|train|all) ;; *) echo "invalid --phase: $PHASE" >&2; exit 2 ;; esac
if [[ "${SPECFORGE_TOTAL_EPOCHS:-40}" != 40 ]]; then
  echo "MSD replication fixes TOTAL_EPOCHS=40" >&2
  exit 2
fi
for name in TARGET_MODEL_PATH TEXT_FEATURE_ROOT VISUAL_FEATURE_ROOT OUTPUT_ROOT; do
  if [[ -z "${!name}" ]]; then echo "$name must not be empty" >&2; exit 2; fi
done

IFS=',' read -r -a DEPTH_ARRAY <<< "$DEPTHS"
declare -A SEEN_DEPTHS=()
for depth in "${DEPTH_ARRAY[@]}"; do
  case "$depth" in 1|3|5) ;; *) echo "unsupported MSD depth: $depth" >&2; exit 2 ;; esac
  if [[ -n "${SEEN_DEPTHS[$depth]:-}" ]]; then
    echo "duplicate MSD depth: $depth" >&2
    exit 2
  fi
  SEEN_DEPTHS[$depth]=1
done
if ((${#DEPTH_ARRAY[@]} == 0)); then echo "DEPTHS must not be empty" >&2; exit 2; fi

GLOBAL_MICRO_BATCH=$((GPU_COUNT * MICRO_BATCH_SIZE))
if ((GLOBAL_BATCH_SIZE % GLOBAL_MICRO_BATCH != 0)); then
  echo "GLOBAL_BATCH_SIZE must divide GPUs x micro batch" >&2
  exit 2
fi
ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / GLOBAL_MICRO_BATCH))

print_config() {
  echo "TARGET_MODEL_PATH=$TARGET_MODEL_PATH"
  echo "DEPTHS=$DEPTHS"
  echo "TOTAL_EPOCHS=$TOTAL_EPOCHS"
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "TEXT_FEATURE_ROOT=$TEXT_FEATURE_ROOT"
  echo "VISUAL_FEATURE_ROOT=$VISUAL_FEATURE_ROOT"
  for depth in "${DEPTH_ARRAY[@]}"; do
    echo "DEPTH_${depth}_OUTPUT=$OUTPUT_ROOT/depth${depth}/output"
  done
}
if ((PRINT_CONFIG)); then print_config; exit 0; fi

for command in "$PYTHON_BIN"; do
  command -v "$command" >/dev/null 2>&1 || { echo "missing command: $command" >&2; exit 2; }
done

if [[ "$PHASE" == data || "$PHASE" == all ]]; then
  [[ -f "$SHAREGPT_SOURCE" ]] || { echo "missing SHAREGPT_SOURCE: $SHAREGPT_SOURCE" >&2; exit 2; }
  "$PYTHON_BIN" scripts/prepare_data.py --dataset sharegpt \
    --data-path "$SHAREGPT_SOURCE" --output-path "$(dirname "$SHAREGPT_JSONL")"
fi

if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
  [[ -f "$SHAREGPT_JSONL" ]] || { echo "missing SHAREGPT_JSONL: $SHAREGPT_JSONL" >&2; exit 2; }
  [[ -f "$LLAVA_MANIFEST" ]] || { echo "missing LLAVA_MANIFEST: $LLAVA_MANIFEST" >&2; exit 2; }
  [[ -d "$IMAGE_ROOT" ]] || { echo "missing IMAGE_ROOT: $IMAGE_ROOT" >&2; exit 2; }
  torchrun --nproc_per_node="$GPU_COUNT" scripts/prepare_hidden_states.py \
    --strategy msd --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config configs/qwen2.5-vl-3b-msd.json \
    --data-path "$SHAREGPT_JSONL" --output-path "$TEXT_FEATURE_ROOT" \
    --chat-template qwen --max-length "$MAX_LENGTH" --num-samples "$EXPECTED_RECORDS"
  torchrun --nproc_per_node="$GPU_COUNT" scripts/prepare_llava_caption_hidden_states.py \
    --strategy msd --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config configs/qwen2.5-vl-3b-msd.json \
    --manifest "$LLAVA_MANIFEST" --image-root "$IMAGE_ROOT" \
    --output-path "$VISUAL_FEATURE_ROOT" --max-length "$MAX_LENGTH" \
    --expected-records "$EXPECTED_RECORDS"
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  [[ -d "$TEXT_FEATURE_ROOT" ]] || { echo "missing text feature root" >&2; exit 2; }
  [[ -d "$VISUAL_FEATURE_ROOT" ]] || { echo "missing visual feature root" >&2; exit 2; }
  for depth in "${DEPTH_ARRAY[@]}"; do
    "$PYTHON_BIN" scripts/materialize_msd_sweep.py \
      --base-draft configs/qwen2.5-vl-3b-msd.json \
      --base-recipe examples/configs/qwen2.5-vl-3b-msd-68k-offline.yaml \
      --generated-root "$GENERATED_ROOT" --output-root "$OUTPUT_ROOT" \
      --depth "$depth" --target-model-path "$TARGET_MODEL_PATH" \
      --text-feature-root "$TEXT_FEATURE_ROOT" \
      --visual-feature-root "$VISUAL_FEATURE_ROOT" \
      --learning-rate "$LEARNING_RATE" --micro-batch-size "$MICRO_BATCH_SIZE" \
      --accumulation-steps "$ACCUMULATION_STEPS" --gpu-count "$GPU_COUNT"
    config="$GENERATED_ROOT/depth${depth}/train.yaml"
    output="$OUTPUT_ROOT/depth${depth}/output"
    resume_args=()
    if ((RESUME)); then resume_args+=("training.resume_from=$output"); fi
    "$PYTHON_BIN" -m specforge.cli train --config "$config" "${resume_args[@]}"
  done
fi
