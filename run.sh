#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAUNCHER="$REPO_ROOT/src/train_Dflash_SpecForge/train_qwen25vl_dflash_llava_68k.sh"
EXAMPLE_ENV="$REPO_ROOT/src/train_Dflash_SpecForge/train_qwen25vl_dflash_llava_68k.env.example"
ENV_FILE=${SPECFORGE_ENV_FILE:-}
PASSTHROUGH=()

usage() {
  cat <<EOF
Usage: ./run.sh [--env-file FILE] [launcher options]

Runs the complete LLaVA Phase 2 pipeline on two GPUs. Existing hidden-state
files are reused, LLaVA preflight is skipped, and training resumes from the
latest Phase 2 checkpoint or starts from PHASE1_CHECKPOINT.

If --env-file is omitted, exported variables are used. See:
  $EXAMPLE_ENV

Examples:
  cp "$EXAMPLE_ENV" qwen25vl_llava_phase2.env
  # edit qwen25vl_llava_phase2.env
  ./run.sh --env-file qwen25vl_llava_phase2.env

  # Reuse the same profile but run only training:
  ./run.sh --env-file qwen25vl_llava_phase2.env --phase train
EOF
}

while (($#)); do
  case "$1" in
    --env-file)
      ENV_FILE=${2:?--env-file requires a path}
      shift 2
      ;;
    --env-file=*)
      ENV_FILE=${1#--env-file=}
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
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

# Two-B200 throughput profile. Values explicitly set in the private env file
# remain authoritative, except GPU count and preflight behavior promised by
# this wrapper.
export SKIP_PREFLIGHT=1
export SPECFORGE_GPUS=2
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
export SPECFORGE_CAPTURE_IO_THREADS=${SPECFORGE_CAPTURE_IO_THREADS:-8}
export SPECFORGE_CAPTURE_IO_QUEUE=${SPECFORGE_CAPTURE_IO_QUEUE:-64}
export SPECFORGE_SGLANG_MEM_FRACTION_STATIC=${SPECFORGE_SGLANG_MEM_FRACTION_STATIC:-0.4}

# Fail stalled collectives promptly and reduce allocator fragmentation during
# variable-length multimodal batches. Keep a user-selected GPU mapping intact.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

# FlashInfer's fastest Blackwell backend JIT-compiles a small TRT-LLM MHA
# launcher. CUDA runtime images often have nvcc but omit nvrtc.h from
# /usr/local/cuda; PyTorch's CUDA wheel carries the matching header and library
# under site-packages/nvidia. Discover that location from the selected Python
# instead of hard-coding one machine's virtualenv path.
NVRTC_ROOT=${SPECFORGE_NVRTC_ROOT:-}
if [[ -z "$NVRTC_ROOT" && -n "${CUDA_HOME:-}" && -f "$CUDA_HOME/include/nvrtc.h" ]]; then
  NVRTC_ROOT=$CUDA_HOME
fi
if [[ -z "$NVRTC_ROOT" && -f /usr/local/cuda/include/nvrtc.h ]]; then
  NVRTC_ROOT=/usr/local/cuda
fi
if [[ -z "$NVRTC_ROOT" ]]; then
  NVRTC_ROOT=$("$PYTHON_BIN" - <<'PY'
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

if [[ -n "$NVRTC_ROOT" ]]; then
  if [[ ! -f "$NVRTC_ROOT/include/nvrtc.h" ]]; then
    echo "SPECFORGE_NVRTC_ROOT does not contain include/nvrtc.h: $NVRTC_ROOT" >&2
    exit 1
  fi
  if [[ -d "$NVRTC_ROOT/lib" ]]; then
    NVRTC_LIB="$NVRTC_ROOT/lib"
  elif [[ -d "$NVRTC_ROOT/lib64" ]]; then
    NVRTC_LIB="$NVRTC_ROOT/lib64"
  else
    echo "NVRTC library directory not found under: $NVRTC_ROOT" >&2
    exit 1
  fi
  export CPATH="$NVRTC_ROOT/include${CPATH:+:$CPATH}"
  export CPLUS_INCLUDE_PATH="$NVRTC_ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export LIBRARY_PATH="$NVRTC_LIB${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$NVRTC_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "[run] NVRTC include=$NVRTC_ROOT/include lib=$NVRTC_LIB"
else
  echo "[run] warning: nvrtc.h was not found; Blackwell FlashInfer JIT may fail" >&2
fi

"$PYTHON_BIN" - <<'PY'
import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"PyTorch is not installed in PYTHON_BIN: {exc}")

count = torch.cuda.device_count()
if count < 2:
    raise SystemExit(f"run.sh requires two visible CUDA GPUs, found {count}")
names = [torch.cuda.get_device_name(index) for index in range(2)]
print(f"[run] torch={torch.__version__} cuda={torch.version.cuda} GPUs={names}")
if any("B200" not in name.upper() for name in names):
    print("[run] warning: the selected profile is tuned for B200 GPUs", file=sys.stderr)
PY

echo "[run] phase=all GPUs=2 skip_preflight=1 resume=auto"
echo "[run] micro/rank=$SPECFORGE_MICRO_BATCH_SIZE global_batch=$SPECFORGE_GLOBAL_BATCH_SIZE attention=$SPECFORGE_ATTENTION_BACKEND"

# The wrapper has already sourced the trusted file. Prevent the child launcher
# from sourcing it again after the B200 profile has been applied.
unset SPECFORGE_ENV_FILE
exec bash "$LAUNCHER" --phase all --gpus 2 --resume "${PASSTHROUGH[@]}"
