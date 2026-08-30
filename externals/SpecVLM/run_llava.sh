#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/resolve_workspace.sh"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Define parameters with comments
MODEL_TYPE="llava_ov"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="${BASE_MODEL_PATH:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"  # Path to base model
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"  # Path to draft model

TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dataset/VideoDetailCaption}"  # Path to dataset

EVAL_NUM="${EVAL_NUM:-2}"        # Number of evaluation samples
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"  # Number of new tokens to generate
DATA_NUM="${DATA_NUM:-100}"      # Number of data samples to load
DROP_RATE="${DROP_RATE:-0.9}"    # Pruning ratio
GPU_IDS="${GPU_IDS:-0,1}"       # GPU IDs to use

# A larger number of frames is generally recommended, as permitted by the model capacity, your GPU memory capacity and bandwidth. Memory bottlenecks are typically triggered by long visual sequence. 
FRAME_NUM="${FRAME_NUM:-64}"
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
