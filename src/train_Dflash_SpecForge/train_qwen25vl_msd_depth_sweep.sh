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

SHARED_STORAGE_ROOT=${SPECFORGE_SHARED_STORAGE_ROOT:-/workspace/storage-shared/nlp/tungdd11/tungdecoder}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$SHARED_STORAGE_ROOT/artifacts/qwen25vl_msd_68k"}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-"$SHARED_STORAGE_ROOT/models/qwen25-vl-3b"}
SHAREGPT_SOURCE=${SHAREGPT_SOURCE:-"$SHARED_STORAGE_ROOT/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"}
LLAVA_SOURCE_JSONL=${LLAVA_SOURCE_JSONL:-"$SHARED_STORAGE_ROOT/data/llava_dflash_qwen25vl3b_68k/llava_dflash_68k_clean_3b.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-"$SHARED_STORAGE_ROOT/LlaVA-Pretrain"}
SHAREGPT_JSONL=${SHAREGPT_JSONL:-"$ARTIFACT_ROOT/manifests/sharegpt_train.jsonl"}
LLAVA_MANIFEST=${LLAVA_MANIFEST:-"$ARTIFACT_ROOT/manifests/llava68k.jsonl"}
TEXT_FEATURE_ROOT=${TEXT_FEATURE_ROOT:-"$ARTIFACT_ROOT/features/sharegpt68k"}
VISUAL_FEATURE_ROOT=${VISUAL_FEATURE_ROOT:-"$ARTIFACT_ROOT/features/llava68k"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ARTIFACT_ROOT/outputs"}
GENERATED_ROOT=${GENERATED_ROOT:-"$ARTIFACT_ROOT/generated"}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}

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
if ((GPU_COUNT < 1 || MICRO_BATCH_SIZE < 1 || GLOBAL_BATCH_SIZE < 1 || EXPECTED_RECORDS < 1 || MAX_LENGTH < 1)); then
  echo "GPU, batch, sample, and sequence sizes must be positive" >&2
  exit 2
fi

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
  echo "SHARED_STORAGE_ROOT=$SHARED_STORAGE_ROOT"
  echo "ARTIFACT_ROOT=$ARTIFACT_ROOT"
  echo "TARGET_MODEL_PATH=$TARGET_MODEL_PATH"
  echo "SHAREGPT_SOURCE=$SHAREGPT_SOURCE"
  echo "SHAREGPT_JSONL=$SHAREGPT_JSONL"
  echo "LLAVA_SOURCE_JSONL=$LLAVA_SOURCE_JSONL"
  echo "LLAVA_MANIFEST=$LLAVA_MANIFEST"
  echo "IMAGE_ROOT=$IMAGE_ROOT"
  echo "DEPTHS=$DEPTHS"
  echo "TOTAL_EPOCHS=$TOTAL_EPOCHS"
  echo "GPU_COUNT=$GPU_COUNT"
  echo "MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE"
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
  echo "ACCUMULATION_STEPS=$ACCUMULATION_STEPS"
  echo "LEARNING_RATE=$LEARNING_RATE"
  echo "TEXT_FEATURE_ROOT=$TEXT_FEATURE_ROOT"
  echo "VISUAL_FEATURE_ROOT=$VISUAL_FEATURE_ROOT"
  echo "OUTPUT_ROOT=$OUTPUT_ROOT"
  echo "GENERATED_ROOT=$GENERATED_ROOT"
  for depth in "${DEPTH_ARRAY[@]}"; do
    echo "DEPTH_${depth}_OUTPUT=$OUTPUT_ROOT/depth${depth}/output"
  done
}
if ((PRINT_CONFIG)); then print_config; exit 0; fi

for command in "$PYTHON_BIN"; do
  command -v "$command" >/dev/null 2>&1 || { echo "missing command: $command" >&2; exit 2; }
done

require_jsonl_count() {
  local label=$1 path=$2 expected=$3 count
  [[ -f "$path" ]] || { echo "missing $label output: $path" >&2; exit 1; }
  count=$(awk 'NF { count += 1 } END { print count + 0 }' "$path")
  if ((count != expected)); then
    echo "$label produced $count records; expected $expected" >&2
    exit 1
  fi
}

feature_count() {
  local root=$1
  [[ -d "$root" ]] || { echo 0; return; }
  find "$root" -type f \( -name '*.ckpt' -o -name '*.ckpt.gz' \) -print 2>/dev/null | wc -l
}

guard_capture_root() {
  local label=$1 root=$2 count
  count=$(feature_count "$root")
  if ((count > 0 && !RESUME)); then
    echo "$label feature cache already contains $count records at $root; pass --resume" >&2
    exit 1
  fi
}

if [[ "$PHASE" == data || "$PHASE" == all ]]; then
  [[ -f "$SHAREGPT_SOURCE" ]] || { echo "missing SHAREGPT_SOURCE: $SHAREGPT_SOURCE" >&2; exit 2; }
  [[ -f "$LLAVA_SOURCE_JSONL" ]] || { echo "missing LLAVA_SOURCE_JSONL: $LLAVA_SOURCE_JSONL" >&2; exit 2; }
  [[ -d "$IMAGE_ROOT" ]] || { echo "missing IMAGE_ROOT: $IMAGE_ROOT" >&2; exit 2; }
  mkdir -p "$(dirname "$SHAREGPT_JSONL")" "$(dirname "$LLAVA_MANIFEST")"
  "$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_data.py" --dataset sharegpt \
    --data-path "$SHAREGPT_SOURCE" --output-path "$(dirname "$SHAREGPT_JSONL")" \
    --sample-size "$EXPECTED_RECORDS"
  "$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_llava_caption_manifest.py" \
    --input "$LLAVA_SOURCE_JSONL" --output "$LLAVA_MANIFEST" \
    --image-root "$IMAGE_ROOT" --expected-records "$EXPECTED_RECORDS"
  require_jsonl_count ShareGPT "$SHAREGPT_JSONL" "$EXPECTED_RECORDS"
  require_jsonl_count LLaVA "$LLAVA_MANIFEST" "$EXPECTED_RECORDS"
fi

if [[ "$PHASE" == capture || "$PHASE" == all ]]; then
  command -v "$TORCHRUN_BIN" >/dev/null 2>&1 || { echo "missing command: $TORCHRUN_BIN" >&2; exit 2; }
  [[ -d "$TARGET_MODEL_PATH" ]] || { echo "missing TARGET_MODEL_PATH: $TARGET_MODEL_PATH" >&2; exit 2; }
  [[ -f "$SHAREGPT_JSONL" ]] || { echo "missing SHAREGPT_JSONL: $SHAREGPT_JSONL" >&2; exit 2; }
  [[ -f "$LLAVA_MANIFEST" ]] || { echo "missing LLAVA_MANIFEST: $LLAVA_MANIFEST" >&2; exit 2; }
  [[ -d "$IMAGE_ROOT" ]] || { echo "missing IMAGE_ROOT: $IMAGE_ROOT" >&2; exit 2; }
  guard_capture_root ShareGPT "$TEXT_FEATURE_ROOT"
  guard_capture_root LLaVA "$VISUAL_FEATURE_ROOT"
  mkdir -p "$TEXT_FEATURE_ROOT" "$VISUAL_FEATURE_ROOT"
  "$TORCHRUN_BIN" --nproc_per_node="$GPU_COUNT" "$SPECFORGE_DIR/scripts/prepare_hidden_states.py" \
    --strategy msd --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$SPECFORGE_DIR/configs/qwen2.5-vl-3b-msd.json" \
    --data-path "$SHAREGPT_JSONL" --output-path "$TEXT_FEATURE_ROOT" \
    --chat-template qwen --max-length "$MAX_LENGTH" --num-samples "$EXPECTED_RECORDS"
  "$TORCHRUN_BIN" --nproc_per_node="$GPU_COUNT" "$SPECFORGE_DIR/scripts/prepare_llava_caption_hidden_states.py" \
    --strategy msd --target-model-path "$TARGET_MODEL_PATH" \
    --draft-model-config "$SPECFORGE_DIR/configs/qwen2.5-vl-3b-msd.json" \
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
