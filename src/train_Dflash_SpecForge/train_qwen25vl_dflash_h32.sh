#!/usr/bin/env bash
set -euo pipefail

# Run the H3.2 depth ablation with one shared five-layer target feature cache.
# The first depth captures the target hidden states; every depth gets its own
# Phase 1/Phase 2 checkpoint and run ID.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
DEPTH_SPEC=${SPECFORGE_DFLASH_DEPTHS:-1,3,5}
H32_ROOT=${SPECFORGE_H32_ROOT:-"$ROOT_DIR/artifacts/qwen25vl_${MODEL_SIZE}_dflash_h32"}
PHASE1_ARTIFACT_ROOT=${SPECFORGE_H32_PHASE1_ARTIFACT_ROOT:-"$H32_ROOT/phase1"}
PHASE2_ARTIFACT_ROOT=${SPECFORGE_H32_PHASE2_ARTIFACT_ROOT:-"$H32_ROOT/phase2"}
OUTPUT_ROOT=${SPECFORGE_H32_OUTPUT_ROOT:-"$H32_ROOT/outputs"}
CAPTURE_FIRST=${SPECFORGE_H32_CAPTURE_FIRST:-1}
RESUME=0

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dflash_h32.sh [--resume]

Runs DFlash depth 1, 3, and 5 (or SPECFORGE_DFLASH_DEPTHS) through:
  Phase 1: ShareGPT text-only training
  Phase 2: LLaVA multimodal training

Required for Phase 2:
  SOURCE_JSONL, TARGET_MODEL_PATH, and IMAGE_ROOT or IMAGE_ARCHIVE

Environment:
  SPECFORGE_MODEL_SIZE=3b|7b (default: 3b)
  SPECFORGE_DFLASH_DEPTHS=1,3,5 (default: 1,3,5)
  SPECFORGE_H32_ROOT (default: train_Dflash_SpecForge/artifacts/..._h32)
  SPECFORGE_H32_CAPTURE_FIRST=1|0 (capture the shared cache on the first depth)
  SPECFORGE_H32_PHASE1_ARTIFACT_ROOT, SPECFORGE_H32_PHASE2_ARTIFACT_ROOT
  SPECFORGE_H32_OUTPUT_ROOT

All other SPECFORGE_* settings are forwarded to the existing phase launchers.
Use --resume when reusing an interrupted capture or checkpoint.
EOF
}

while (($#)); do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$MODEL_SIZE" in
  3b|7b) ;;
  *) echo "SPECFORGE_MODEL_SIZE must be 3b or 7b" >&2; exit 2 ;;
esac
case "$CAPTURE_FIRST" in
  0|1) ;;
  *) echo "SPECFORGE_H32_CAPTURE_FIRST must be 0 or 1" >&2; exit 2 ;;
esac

DEPTH_SPEC=${DEPTH_SPEC//,/ }
# shellcheck disable=SC2206
DEPTHS=($DEPTH_SPEC)
if ((${#DEPTHS[@]} == 0)); then
  echo "SPECFORGE_DFLASH_DEPTHS must contain at least one depth" >&2
  exit 2
fi
declare -A seen_depths=()
for depth in "${DEPTHS[@]}"; do
  case "$depth" in
    1|3|5) ;;
    *) echo "H3.2 depths must be chosen from 1, 3, or 5; got: $depth" >&2; exit 2 ;;
  esac
  if [[ -n "${seen_depths[$depth]:-}" ]]; then
    echo "duplicate H3.2 depth: $depth" >&2
    exit 2
  fi
  seen_depths[$depth]=1
done

PYTHON_BIN=${PYTHON_BIN:-python3}
SOURCE_DATA=${SOURCE_DATA:-}
SOURCE_JSONL=${SOURCE_JSONL:-}
IMAGE_ROOT=${IMAGE_ROOT:-}
IMAGE_ARCHIVE=${IMAGE_ARCHIVE:-}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-}

if [[ -z "$SOURCE_JSONL" || -z "$TARGET_MODEL_PATH" || ( -z "$IMAGE_ROOT" && -z "$IMAGE_ARCHIVE" ) ]]; then
  echo "Phase 2 requires SOURCE_JSONL, TARGET_MODEL_PATH, and IMAGE_ROOT or IMAGE_ARCHIVE" >&2
  echo "Set those variables before launching H3.2." >&2
  exit 2
fi

mkdir -p "$PHASE1_ARTIFACT_ROOT" "$PHASE2_ARTIFACT_ROOT" "$OUTPUT_ROOT"
PHASE1_ARTIFACT_ROOT=$(cd "$PHASE1_ARTIFACT_ROOT" && pwd)
PHASE2_ARTIFACT_ROOT=$(cd "$PHASE2_ARTIFACT_ROOT" && pwd)
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)

RESUME_ARGS=()
if ((RESUME)); then
  RESUME_ARGS+=(--resume)
fi

export PYTHON_BIN MODEL_SIZE SOURCE_DATA SOURCE_JSONL IMAGE_ROOT IMAGE_ARCHIVE TARGET_MODEL_PATH
export SPECFORGE_MODEL_SIZE="$MODEL_SIZE"

first_depth=${DEPTHS[0]}
SPECFORGE_DFLASH_DEPTH="$first_depth"
SPECFORGE_RUN_SUFFIX="-h32-depth${first_depth}"
export SPECFORGE_DFLASH_DEPTH SPECFORGE_RUN_SUFFIX

echo "[h3.2] Phase 1 data"
ARTIFACT_ROOT="$PHASE1_ARTIFACT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT/phase1" \
  bash "$ROOT_DIR/train_qwen25vl_dflash_sharegpt_68k.sh" \
  --models "$MODEL_SIZE" --phase data

if ((CAPTURE_FIRST)); then
  echo "[h3.2] Phase 1 capture (shared target features; first depth=$first_depth)"
  ARTIFACT_ROOT="$PHASE1_ARTIFACT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT/phase1" \
    bash "$ROOT_DIR/train_qwen25vl_dflash_sharegpt_68k.sh" \
    --models "$MODEL_SIZE" --phase capture "${RESUME_ARGS[@]}"
fi

for depth in "${DEPTHS[@]}"; do
  suffix="-h32-depth${depth}"
  SPECFORGE_DFLASH_DEPTH="$depth"
  SPECFORGE_RUN_SUFFIX="$suffix"
  export SPECFORGE_DFLASH_DEPTH SPECFORGE_RUN_SUFFIX
  echo "[h3.2] Phase 1 train depth=$depth"
  ARTIFACT_ROOT="$PHASE1_ARTIFACT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT/phase1" \
    bash "$ROOT_DIR/train_qwen25vl_dflash_sharegpt_68k.sh" \
    --models "$MODEL_SIZE" --phase train "${RESUME_ARGS[@]}"
done

SPECFORGE_DFLASH_DEPTH="$first_depth"
SPECFORGE_RUN_SUFFIX="-h32-depth${first_depth}"
export SPECFORGE_DFLASH_DEPTH SPECFORGE_RUN_SUFFIX

echo "[h3.2] Phase 2 data"
ARTIFACT_ROOT="$PHASE2_ARTIFACT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT/phase2" \
  bash "$ROOT_DIR/train_qwen25vl_dflash_llava_68k.sh" --phase data

if ((CAPTURE_FIRST)); then
  echo "[h3.2] Phase 2 capture (shared target+visual features; first depth=$first_depth)"
  ARTIFACT_ROOT="$PHASE2_ARTIFACT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT/phase2" \
    bash "$ROOT_DIR/train_qwen25vl_dflash_llava_68k.sh" \
    --phase capture "${RESUME_ARGS[@]}"
fi

for depth in "${DEPTHS[@]}"; do
  suffix="-h32-depth${depth}"
  phase1_run_id="qwen25vl-${MODEL_SIZE}-dflash-sharegpt68k${suffix}"
  phase1_checkpoint="$OUTPUT_ROOT/phase1/$phase1_run_id/$phase1_run_id-latest"
  SPECFORGE_DFLASH_DEPTH="$depth"
  SPECFORGE_RUN_SUFFIX="$suffix"
  PHASE1_CHECKPOINT="$phase1_checkpoint"
  export SPECFORGE_DFLASH_DEPTH SPECFORGE_RUN_SUFFIX PHASE1_CHECKPOINT
  echo "[h3.2] Phase 2 train depth=$depth"
  ARTIFACT_ROOT="$PHASE2_ARTIFACT_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT/phase2" \
    bash "$ROOT_DIR/train_qwen25vl_dflash_llava_68k.sh" \
    --phase train "${RESUME_ARGS[@]}"
done

echo "[h3.2] completed depths: ${DEPTHS[*]}"
