#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/resolve_workspace.sh"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Define parameters with comments
MODEL_TYPE="qwen2_5_vl"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen2.5-VL-32B-Instruct}"  # Path to base model
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"  # Path to draft model

TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dataset/VideoDetailCaption}"  # Path to dataset

EVAL_NUM="${EVAL_NUM:-1}"        # Number of evaluation samples
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"  # Number of new tokens to generate
DATA_NUM="${DATA_NUM:-100}"      # Number of data samples to load
DROP_RATE="${DROP_RATE:-0.9}"    # Pruning ratio
GPU_IDS="${GPU_IDS:-0,1}"       # GPU IDs to use

# Qwen2.5-VL currently does not support specifying input length directly. To control the input length, we adjust the fps accordingly.
FRAME_NUM="${FRAME_NUM:-128}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/results/specvlm/${MODEL_TYPE}_${TASK}_drop_rate_${DROP_RATE}}"

# Run evaluation
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -u inference.py \
    --model_type "${MODEL_TYPE}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --task "${TASK}" \
    --frame_num "${FRAME_NUM}" \
    --evaluation_num "${EVAL_NUM}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --drop_rate "${DROP_RATE}" \
    --data_num "${DATA_NUM}" \
    --gpu_ids "${GPU_IDS}" \
    --save_path "${SAVE_PATH}"
