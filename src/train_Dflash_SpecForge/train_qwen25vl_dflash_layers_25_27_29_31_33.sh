#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PHASE1_LAUNCHER="$ROOT_DIR/train_qwen25vl_dflash_sharegpt_68k.sh"
PHASE2_LAUNCHER="$ROOT_DIR/train_qwen25vl_dflash_llava_68k.sh"

# Load a trusted shell-style environment file before applying this wrapper's
# defaults.  The same file may contain the ShareGPT and LLaVA paths.
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
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
WRAPPER_PHASE=${SPECFORGE_WRAPPER_PHASE:-all}
INNER_PHASE=${SPECFORGE_INNER_PHASE:-all}
GPU_COUNT=${SPECFORGE_GPUS:-2}
MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-3b}
RESUME=0
PRINT_CONFIG=0
SKIP_GPU_CHECK=${SPECFORGE_SKIP_GPU_CHECK:-0}

TARGET_LAYER_IDS=${SPECFORGE_TARGET_LAYER_IDS:-25,27,29,31,33}

usage() {
  cat <<'EOF'
Usage: train_qwen25vl_dflash_layers_25_27_29_31_33.sh [options]

Runs the Qwen2.5-VL-3B DFlash Phase 1 and Phase 2 pipeline with one shared
five-layer hidden-state selection.  The default is the current two-B200
profile and target layers 25,27,29,31,33.

Options:
  --env-file FILE       Source a private shell-style environment file.
  --phase all|phase1|phase2
                        Run both phases, only Phase 1, or only Phase 2.
  --inner-phase data|capture|train|all
                        Phase operation forwarded to the selected launcher.
  --gpus 2              Require the two-GPU B200 profile (default: 2).
  --resume              Resume existing capture files/checkpoints.
  --print-config        Print the resolved profile and child launchers.
  -h, --help            Show this help.

Environment:
  SPECFORGE_TARGET_LAYER_IDS=comma-separated five layer IDs
  SOURCE_DATA, MODEL_3B, SOURCE_JSONL, TARGET_MODEL_PATH,
  IMAGE_ROOT or IMAGE_ARCHIVE, PHASE1_CHECKPOINT
  ARTIFACT_ROOT and OUTPUT_ROOT are base roots; phase1/phase2 subdirectories
  are created automatically.  The 2x-B200 defaults match run.sh.
EOF
}

while (($#)); do
  case "$1" in
    --env-file)
      shift 2
      ;;
    --env-file=*)
      shift
      ;;
    --phase)
      WRAPPER_PHASE=${2:?--phase requires a value}
      shift 2
      ;;
    --inner-phase)
      INNER_PHASE=${2:?--inner-phase requires a value}
      shift 2
      ;;
    --gpus)
      GPU_COUNT=${2:?--gpus requires a value}
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --print-config)
      PRINT_CONFIG=1
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

case "$WRAPPER_PHASE" in
  all|phase1|phase2) ;;
  *)
    echo "--phase must be all, phase1, or phase2" >&2
    exit 2
    ;;
esac
case "$INNER_PHASE" in
  data|capture|train|all) ;;
  *)
    echo "--inner-phase must be data, capture, train, or all" >&2
    exit 2
    ;;
esac
if [[ "$GPU_COUNT" != 2 ]]; then
  echo "this launcher is fixed to the two-GPU B200 profile; use --gpus 2" >&2
  exit 2
fi
if [[ "$MODEL_SIZE" != 3b ]]; then
  echo "layers 25,27,29,31,33 are supported by this launcher only for MODEL_SIZE=3b" >&2
  exit 2
fi
case "$SKIP_GPU_CHECK" in
  0|1) ;;
  *) echo "SPECFORGE_SKIP_GPU_CHECK must be 0 or 1" >&2; exit 2 ;;
esac

IFS=',' read -r -a TARGET_LAYER_ARRAY <<< "$TARGET_LAYER_IDS"
if ((${#TARGET_LAYER_ARRAY[@]} != 5)); then
  echo "SPECFORGE_TARGET_LAYER_IDS must contain exactly five comma-separated IDs" >&2
  exit 2
fi
for layer_id in "${TARGET_LAYER_ARRAY[@]}"; do
  if [[ ! "$layer_id" =~ ^[0-9]+$ ]]; then
    echo "SPECFORGE_TARGET_LAYER_IDS must contain non-negative integers" >&2
    exit 2
  fi
done
for ((left = 0; left < ${#TARGET_LAYER_ARRAY[@]}; left++)); do
  for ((right = left + 1; right < ${#TARGET_LAYER_ARRAY[@]}; right++)); do
    if [[ "${TARGET_LAYER_ARRAY[left]}" == "${TARGET_LAYER_ARRAY[right]}" ]]; then
      echo "SPECFORGE_TARGET_LAYER_IDS must contain unique IDs" >&2
      exit 2
    fi
  done
done

BASE_ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$ROOT_DIR/artifacts/qwen25vl_dflash_layers_25_27_29_31_33"}
BASE_OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_DIR/outputs/qwen25vl_dflash_layers_25_27_29_31_33"}
PHASE1_ARTIFACT_ROOT=${PHASE1_ARTIFACT_ROOT:-"$BASE_ARTIFACT_ROOT/phase1"}
PHASE2_ARTIFACT_ROOT=${PHASE2_ARTIFACT_ROOT:-"$BASE_ARTIFACT_ROOT/phase2"}
PHASE1_OUTPUT_ROOT=${PHASE1_OUTPUT_ROOT:-"$BASE_OUTPUT_ROOT/phase1"}
PHASE2_OUTPUT_ROOT=${PHASE2_OUTPUT_ROOT:-"$BASE_OUTPUT_ROOT/phase2"}
PHASE1_RUN_ID=qwen25vl-3b-dflash-sharegpt68k
PHASE2_RUN_ID=qwen25vl-3b-dflash-llava68k
PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-"$PHASE1_OUTPUT_ROOT/$PHASE1_RUN_ID-latest"}

# Match the repository's current two-B200 wrapper.  Per-phase max lengths are
# supplied at invocation time because text and multimodal recipes differ.
export SPECFORGE_GPUS=2
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
export SKIP_PREFLIGHT=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export SPECFORGE_PHASE1_TARGET_LAYER_IDS="$TARGET_LAYER_IDS"
export SPECFORGE_PHASE2_TARGET_LAYER_IDS="$TARGET_LAYER_IDS"

MODEL_3B=${MODEL_3B:-${TARGET_MODEL_PATH:-}}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-$MODEL_3B}
export MODEL_3B TARGET_MODEL_PATH PHASE1_CHECKPOINT

print_config() {
  printf 'WRAPPER_PHASE=%s\n' "$WRAPPER_PHASE"
  printf 'INNER_PHASE=%s\n' "$INNER_PHASE"
  printf 'MODEL_SIZE=%s\n' "$MODEL_SIZE"
  printf 'TARGET_LAYER_IDS=%s\n' "$TARGET_LAYER_IDS"
  printf 'SPECFORGE_PHASE1_TARGET_LAYER_IDS=%s\n' "$SPECFORGE_PHASE1_TARGET_LAYER_IDS"
  printf 'SPECFORGE_PHASE2_TARGET_LAYER_IDS=%s\n' "$SPECFORGE_PHASE2_TARGET_LAYER_IDS"
  printf 'SPECFORGE_GPUS=%s\n' "$SPECFORGE_GPUS"
  printf 'SPECFORGE_GLOBAL_BATCH_SIZE=%s\n' "$SPECFORGE_GLOBAL_BATCH_SIZE"
  printf 'SPECFORGE_MICRO_BATCH_SIZE=%s\n' "$SPECFORGE_MICRO_BATCH_SIZE"
  printf 'SPECFORGE_FSDP_SHARDING=%s\n' "$SPECFORGE_FSDP_SHARDING"
  printf 'SPECFORGE_ATTENTION_BACKEND=%s\n' "$SPECFORGE_ATTENTION_BACKEND"
  printf 'SPECFORGE_OBJECTIVE_CHUNK_BLOCKS=%s\n' "$SPECFORGE_OBJECTIVE_CHUNK_BLOCKS"
  printf 'SPECFORGE_CAPTURE_BATCH_SIZE=%s\n' "$SPECFORGE_CAPTURE_BATCH_SIZE"
  printf 'SPECFORGE_SGLANG_MEM_FRACTION_STATIC=%s\n' "$SPECFORGE_SGLANG_MEM_FRACTION_STATIC"
  printf 'PHASE1_ARTIFACT_ROOT=%s\n' "$PHASE1_ARTIFACT_ROOT"
  printf 'PHASE2_ARTIFACT_ROOT=%s\n' "$PHASE2_ARTIFACT_ROOT"
  printf 'PHASE1_OUTPUT_ROOT=%s\n' "$PHASE1_OUTPUT_ROOT"
  printf 'PHASE2_OUTPUT_ROOT=%s\n' "$PHASE2_OUTPUT_ROOT"
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

configure_nvrtc() {
  local nvrtc_root=${SPECFORGE_NVRTC_ROOT:-}
  if [[ -z "$nvrtc_root" && -n "${CUDA_HOME:-}" && -f "$CUDA_HOME/include/nvrtc.h" ]]; then
    nvrtc_root=$CUDA_HOME
  fi
  if [[ -z "$nvrtc_root" && -f /usr/local/cuda/include/nvrtc.h ]]; then
    nvrtc_root=/usr/local/cuda
  fi
  if [[ -z "$nvrtc_root" ]]; then
    nvrtc_root=$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import sys

candidates = []
for entry in sys.path:
    if not entry:
        continue
    root = Path(entry)
    candidates.extend(root.glob("nvidia/cu*/include/nvrtc.h"))
    candidates.extend(root.glob("nvidia/cuda_nvrtc/include/nvrtc.h"))
if candidates:
    print(sorted(candidates)[-1].parent.parent)
PY
    )
  fi
  if [[ -n "$nvrtc_root" ]]; then
    if [[ ! -f "$nvrtc_root/include/nvrtc.h" ]]; then
      echo "SPECFORGE_NVRTC_ROOT does not contain include/nvrtc.h: $nvrtc_root" >&2
      exit 1
    fi
    local nvrtc_lib
    if [[ -d "$nvrtc_root/lib" ]]; then
      nvrtc_lib="$nvrtc_root/lib"
    elif [[ -d "$nvrtc_root/lib64" ]]; then
      nvrtc_lib="$nvrtc_root/lib64"
    else
      echo "NVRTC library directory not found under: $nvrtc_root" >&2
      exit 1
    fi
    export CPATH="$nvrtc_root/include${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="$nvrtc_root/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export LIBRARY_PATH="$nvrtc_lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$nvrtc_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[layers-run] NVRTC include=$nvrtc_root/include lib=$nvrtc_lib"
  else
    echo "[layers-run] warning: nvrtc.h was not found; B200 FlashInfer JIT may fail" >&2
  fi
}

verify_b200_gpus() {
  if [[ "$SKIP_GPU_CHECK" == 1 ]]; then
    echo "[layers-run] skipping two-GPU B200 check (SPECFORGE_SKIP_GPU_CHECK=1)"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"PyTorch is not installed in PYTHON_BIN: {exc}")

count = torch.cuda.device_count()
if count < 2:
    raise SystemExit(f"two-GPU B200 profile requires two visible CUDA GPUs, found {count}")
names = [torch.cuda.get_device_name(index) for index in range(2)]
print(f"[layers-run] torch={torch.__version__} cuda={torch.version.cuda} GPUs={names}")
if any("B200" not in name.upper() for name in names):
    print("[layers-run] warning: selected GPUs are not named B200", file=sys.stderr)
PY
}

run_phase1() {
  local resume_args=()
  if ((RESUME)); then
    resume_args+=(--resume)
  fi
  echo "[layers-run] Phase 1 layers=$TARGET_LAYER_IDS artifacts=$PHASE1_ARTIFACT_ROOT output=$PHASE1_OUTPUT_ROOT"
  env \
    ARTIFACT_ROOT="$PHASE1_ARTIFACT_ROOT" \
    OUTPUT_ROOT="$PHASE1_OUTPUT_ROOT" \
    MODEL_3B="$MODEL_3B" \
    SPECFORGE_MAX_LENGTH=2048 \
    SPECFORGE_ENV_FILE= \
    bash "$PHASE1_LAUNCHER" --models 3b --phase "$INNER_PHASE" --gpus 2 "${resume_args[@]}"
}

run_phase2() {
  local resume_args=()
  if ((RESUME)); then
    resume_args+=(--resume)
  fi
  if [[ "$INNER_PHASE" == train || "$INNER_PHASE" == all ]]; then
    if [[ ! -e "$PHASE1_CHECKPOINT" ]]; then
      echo "Phase 1 checkpoint not found: $PHASE1_CHECKPOINT" >&2
      exit 1
    fi
  fi
  echo "[layers-run] Phase 2 layers=$TARGET_LAYER_IDS artifacts=$PHASE2_ARTIFACT_ROOT output=$PHASE2_OUTPUT_ROOT"
  env \
    ARTIFACT_ROOT="$PHASE2_ARTIFACT_ROOT" \
    OUTPUT_ROOT="$PHASE2_OUTPUT_ROOT" \
    PHASE1_CHECKPOINT="$PHASE1_CHECKPOINT" \
    SPECFORGE_MAX_LENGTH=3072 \
    SPECFORGE_ENV_FILE= \
    bash "$PHASE2_LAUNCHER" --phase "$INNER_PHASE" --gpus 2 "${resume_args[@]}"
}

configure_nvrtc
verify_b200_gpus
echo "[layers-run] phase=$WRAPPER_PHASE inner_phase=$INNER_PHASE GPUs=2 layers=$TARGET_LAYER_IDS"

if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase1 ]]; then
  run_phase1
fi
if [[ "$WRAPPER_PHASE" == all || "$WRAPPER_PHASE" == phase2 ]]; then
  run_phase2
fi
