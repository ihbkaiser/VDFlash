#!/usr/bin/env bash
set -euo pipefail

# Short, real-data end-to-end smoke for the production two-phase B200 launcher.
# The target model, sample data, capture settings, BF16 path, objective, and
# warm-start remain production code; only the sample count, batch, and epochs
# are reduced. All generated files are isolated under SMOKE_ROOT.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$ROOT_DIR/../.." && pwd)
ENV_FILE=${SPECFORGE_ENV_FILE:-"$REPO_ROOT/src/train_Dflash_SpecForge/train_qwen25vl_dflash_3layer.env"}
LAYER_SELECTION=middle_high
SAMPLE_COUNT=${SPECFORGE_SMOKE_NUM_SAMPLES:-8}
CLI_SMOKE_ROOT=
PRINT_CONFIG=0

while (($#)); do
  case "$1" in
    --env-file)
      (($# >= 2)) || { echo "--env-file requires a path" >&2; exit 2; }
      ENV_FILE=$2
      shift 2
      ;;
    --env-file=*) ENV_FILE=${1#--env-file=}; shift ;;
    --layer-set)
      (($# >= 2)) || { echo "--layer-set requires a value" >&2; exit 2; }
      LAYER_SELECTION=$2
      shift 2
      ;;
    --layer-set=*) LAYER_SELECTION=${1#*=}; shift ;;
    --samples)
      (($# >= 2)) || { echo "--samples requires a value" >&2; exit 2; }
      SAMPLE_COUNT=$2
      shift 2
      ;;
    --samples=*) SAMPLE_COUNT=${1#*=}; shift ;;
    --smoke-root)
      (($# >= 2)) || { echo "--smoke-root requires a path" >&2; exit 2; }
      CLI_SMOKE_ROOT=$2
      shift 2
      ;;
    --smoke-root=*) CLI_SMOKE_ROOT=${1#*=}; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: bash src/train_DSpark_SpecForge/train_qwen25vl_dspark_3layer_b200_smoke.sh [options]

Run the real ShareGPT and LLaVA capture pipelines, then train Phase 1 and
warm-start Phase 2 on exactly two B200 GPUs. The default smoke uses 8 records,
one epoch, and the middle_high teacher-layer set. Production model, capture
lengths, BF16, DSpark depth/block, loss, and attention settings are preserved.

Options:
  --layer-set NAME   middle_high, spread, high_only, or all (default: middle_high)
  --samples N        Records per phase (default: 8; minimum: 2)
  --smoke-root PATH  Output root (default: a new timestamped directory beside
                     the DFlash artifacts)
  --env-file FILE    Machine environment file (default: tracked Qwen2.5-VL .env)
  --print-config     Print the resolved smoke settings without starting work
  -h, --help         Show this help

GPU selection uses SPECFORGE_GPU_IDS, then CUDA_VISIBLE_DEVICES, then 0,1.
The script refuses to run unless both visible GPUs identify as B200.
For example:
  CUDA_VISIBLE_DEVICES=2,3 SPECFORGE_GPU_IDS=2,3 bash \
    src/train_DSpark_SpecForge/train_qwen25vl_dspark_3layer_b200_smoke.sh
EOF
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$LAYER_SELECTION" in
  spread|high_only|middle_high|all) ;;
  *) echo "--layer-set must be spread, high_only, middle_high, or all" >&2; exit 2 ;;
esac
[[ "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "--samples must be a positive integer" >&2
  exit 2
}
((SAMPLE_COUNT >= 2)) || {
  echo "--samples must be at least 2 for two-GPU training" >&2
  exit 2
}
[[ -f "$ENV_FILE" ]] || {
  echo "environment file not found: $ENV_FILE" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_IDS=${SPECFORGE_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}
GPU_IDS=${GPU_IDS//[[:space:]]/}
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
[[ ${#GPU_ID_ARRAY[@]} -eq 2 && -n ${GPU_ID_ARRAY[0]} && -n ${GPU_ID_ARRAY[1]} && ${GPU_ID_ARRAY[0]} != "${GPU_ID_ARRAY[1]}" ]] || {
  echo "smoke requires exactly two distinct GPU IDs, got: $GPU_IDS" >&2
  exit 2
}
[[ ${SPECFORGE_MODEL_SIZE:-3b} == 3b ]] || {
  echo "this smoke supports SPECFORGE_MODEL_SIZE=3b only" >&2
  exit 2
}

SOURCE_DATA=${SOURCE_DATA:-}
MODEL_3B=${MODEL_3B:-}
SOURCE_JSONL=${SOURCE_JSONL:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
BASE_ARTIFACT_ROOT=${ARTIFACT_ROOT:-${DFLASH_ARTIFACT_ROOT:-}}
[[ -n "$BASE_ARTIFACT_ROOT" ]] || { echo "ARTIFACT_ROOT must be set in $ENV_FILE" >&2; exit 2; }

UTC_STAMP=$(date -u +%Y%m%d_%H%M%S)
SMOKE_ROOT=${CLI_SMOKE_ROOT:-${SPECFORGE_DSPARK_SMOKE_ROOT:-"$(dirname "$BASE_ARTIFACT_ROOT")/qwen25_dspark_3layer_smoke/$UTC_STAMP"}}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export SPECFORGE_GPU_IDS="$GPU_IDS"
export SPECFORGE_GPUS=2
export SPECFORGE_NUM_SAMPLES="$SAMPLE_COUNT"
export SPECFORGE_GLOBAL_BATCH_SIZE=2
export SPECFORGE_MICRO_BATCH_SIZE=1
export SPECFORGE_NUM_EPOCHS=1
export SPECFORGE_PHASE1_NUM_EPOCHS=1
export SPECFORGE_PHASE2_NUM_EPOCHS=1
export SPECFORGE_DSPARK_DEPTH=3
export SPECFORGE_DSPARK_BLOCK_SIZE=${SPECFORGE_DSPARK_BLOCK_SIZE:-7}
export SPECFORGE_CAPTURE_BATCH_SIZE=4
export SPECFORGE_CAPTURE_PREPROCESS_WORKERS=2
export SPECFORGE_CAPTURE_PREPROCESS_QUEUE=8
export SPECFORGE_CAPTURE_IO_THREADS=2
export SPECFORGE_CAPTURE_IO_QUEUE=8
export SPECFORGE_DATASET_NUM_PROC=2
export SPECFORGE_DSPARK_ROOT="$SMOKE_ROOT/dspark"

case "$LAYER_SELECTION" in
  all) LAYER_SETS=(spread high_only middle_high) ;;
  *) LAYER_SETS=("$LAYER_SELECTION") ;;
esac

if ((PRINT_CONFIG)); then
  printf 'ENV_FILE=%s\nGPU_IDS=%s\nSAMPLE_COUNT=%s\n' "$ENV_FILE" "$GPU_IDS" "$SAMPLE_COUNT"
  printf 'LAYER_SELECTION=%s\nSMOKE_ROOT=%s\n' "$LAYER_SELECTION" "$SMOKE_ROOT"
  printf 'GLOBAL_BATCH_SIZE=%s\nMICRO_BATCH_SIZE=%s\nEPOCHS=1\n' \
    "$SPECFORGE_GLOBAL_BATCH_SIZE" "$SPECFORGE_MICRO_BATCH_SIZE"
  printf 'LAYER_SETS=%s\n' "${LAYER_SETS[*]}"
  printf 'SOURCE_DATA=%s\nMODEL_3B=%s\nSOURCE_JSONL=%s\nIMAGE_ROOT=%s\n' \
    "$SOURCE_DATA" "$MODEL_3B" "$SOURCE_JSONL" "$IMAGE_ROOT"
  exit 0
fi

[[ -n "$SOURCE_DATA" && -f "$SOURCE_DATA" ]] || { echo "ShareGPT source not found: $SOURCE_DATA" >&2; exit 1; }
[[ -n "$MODEL_3B" && -e "$MODEL_3B" ]] || { echo "Qwen2.5-VL-3B model not found: $MODEL_3B" >&2; exit 1; }
[[ -n "$SOURCE_JSONL" && -f "$SOURCE_JSONL" ]] || { echo "LLaVA JSONL not found: $SOURCE_JSONL" >&2; exit 1; }
[[ -n "$IMAGE_ROOT" && -d "$IMAGE_ROOT" ]] || { echo "LLaVA image root not found: $IMAGE_ROOT" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 1; }
[[ ! -e "$SMOKE_ROOT" ]] || {
  echo "smoke root already exists; choose a fresh path: $SMOKE_ROOT" >&2
  exit 2
}
mkdir -p "$SMOKE_ROOT/input"
LOG_FILE="$SMOKE_ROOT/smoke.log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; echo "[smoke] exit_code=$status"' EXIT

"$PYTHON_BIN" - "$GPU_IDS" <<'PY'
import sys
import torch

expected_ids = sys.argv[1]
expected_count = len(expected_ids.split(","))
count = torch.cuda.device_count()
if count != expected_count:
    raise SystemExit(
        f"expected {expected_count} visible GPUs from CUDA_VISIBLE_DEVICES="
        f"{expected_ids}, found {count}"
    )
names = [torch.cuda.get_device_name(index) for index in range(count)]
not_b200 = [name for name in names if "B200" not in name.upper()]
if not_b200:
    raise SystemExit(f"this smoke only runs on B200 GPUs; detected: {names}")
for index, name in enumerate(names):
    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(f"GPU {index} ({name}) does not support BF16")
print(
    f"[smoke] torch={torch.__version__} cuda={torch.version.cuda} "
    f"visible_gpus={count} names={names} dtype=bfloat16"
)
PY

SMOKE_JSONL="$SMOKE_ROOT/input/llava_smoke_${SAMPLE_COUNT}.jsonl"
head -n "$SAMPLE_COUNT" "$SOURCE_JSONL" > "$SMOKE_JSONL"
JSONL_COUNT=$(awk 'END { print NR + 0 }' "$SMOKE_JSONL")
[[ "$JSONL_COUNT" == "$SAMPLE_COUNT" ]] || {
  echo "could not select $SAMPLE_COUNT LLaVA rows; got $JSONL_COUNT" >&2
  exit 1
}

# A smoke-specific env isolates the small ShareGPT conversion from the normal
# 68k DFlash artifact and points phase-2 capture at the selected LLaVA rows.
SMOKE_ENV_FILE="$SMOKE_ROOT/input/smoke.env"
{
  printf 'SOURCE_DATA=%q\n' "$SOURCE_DATA"
  printf 'MODEL_3B=%q\n' "$MODEL_3B"
  printf 'SOURCE_JSONL=%q\n' "$SMOKE_JSONL"
  printf 'IMAGE_ROOT=%q\n' "$IMAGE_ROOT"
  printf 'ARTIFACT_ROOT=%q\n' "$SMOKE_ROOT/dflash_artifacts"
  printf 'OUTPUT_ROOT=%q\n' "$SMOKE_ROOT/dflash_outputs"
  printf 'SPECFORGE_MODEL_SIZE=3b\n'
} > "$SMOKE_ENV_FILE"

echo "[smoke] root=$SMOKE_ROOT samples_per_phase=$SAMPLE_COUNT GPUs=$GPU_IDS"
echo "[smoke] training: global_batch=2 micro_batch=1 accumulation=1 epochs=1"
echo "[smoke] capture lengths remain Phase1=2048, Phase2=3072; DSpark depth=3"
echo "[smoke] selected teacher-layer sets: ${LAYER_SETS[*]}"

for layer_set in "${LAYER_SETS[@]}"; do
  case "$layer_set" in
    spread) layer_slug=1_9_17_25_33 ;;
    high_only) layer_slug=26_28_30_32_34 ;;
    middle_high) layer_slug=17_19_25_33_34 ;;
    *) echo "invalid internal layer set: $layer_set" >&2; exit 2 ;;
  esac
  export SPECFORGE_DSPARK_ARTIFACT_ROOT="$SMOKE_ROOT/dspark/dspark_depth3_layers_$layer_slug"
  export SPECFORGE_DSPARK_OUTPUT_ROOT="$SMOKE_ROOT/dspark/outputs/dspark_depth3_layers_$layer_slug"
  echo "[smoke] starting complete Phase1+Phase2 run for $layer_set"
  bash "$ROOT_DIR/train_qwen25vl_dspark_3layer_b200.sh" \
    --env-file "$SMOKE_ENV_FILE" \
    --layer-set "$layer_set"
  echo "[smoke] completed Phase1+Phase2 run for $layer_set"
done

echo "[smoke] all requested B200 phase runs completed; log=$LOG_FILE"
