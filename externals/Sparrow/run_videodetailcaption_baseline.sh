#!/usr/bin/env bash
# Portable one-sample/full runner for Sparrow's dense baseline.
# This is a local adapter around the vendored Sparrow evaluator; model paths
# remain explicit because they are machine-specific and must not be committed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/resolve_workspace.sh"

SPARROW_ROOT="${REPO_ROOT}/externals/Sparrow"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
SPEC_MODEL_PATH="${SPEC_MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/dataset/VideoDetailCaption}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/sparrow_baseline}"
MODEL_ID="${MODEL_ID:-baseline}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5_vl}"
GPU_IDS="${GPU_IDS:-0}"
FRAME_NUM="${FRAME_NUM:-16}"
DATA_NUM="${DATA_NUM:-1}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
ANSWER_FILE="${ANSWER_FILE:-}"

usage() {
    sed -n '2,7p' "${BASH_SOURCE[0]}"
    cat >&2 <<'EOF'

Required model arguments can be supplied as environment variables or flags:
  --base-model-path MODEL  --spec-model-path MODEL

Useful flags:
  --data-path PATH --output-dir PATH --answer-file PATH --gpu-ids IDS
  --model-id ID --model-type {qwen2_5_vl,llava_ov}
  --frame-num N --data-num N --max-new-token N --temperature FLOAT
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-model-path) BASE_MODEL_PATH="$2"; shift 2 ;;
        --spec-model-path) SPEC_MODEL_PATH="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --answer-file) ANSWER_FILE="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --model-type) MODEL_TYPE="$2"; shift 2 ;;
        --frame-num) FRAME_NUM="$2"; shift 2 ;;
        --data-num) DATA_NUM="$2"; shift 2 ;;
        --max-new-token) MAX_NEW_TOKEN="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${BASE_MODEL_PATH}" || -z "${SPEC_MODEL_PATH}" ]]; then
    echo "Set BASE_MODEL_PATH and SPEC_MODEL_PATH (or pass both model flags)." >&2
    exit 2
fi

if [[ "${DATA_PATH}" != /* ]]; then
    DATA_PATH="${REPO_ROOT}/${DATA_PATH}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi
if [[ -n "${ANSWER_FILE}" && "${ANSWER_FILE}" != /* ]]; then
    ANSWER_FILE="${REPO_ROOT}/${ANSWER_FILE}"
fi

if [[ -z "${ANSWER_FILE}" ]]; then
    ANSWER_FILE="${OUTPUT_DIR}/${MODEL_ID}.jsonl"
fi

mkdir -p "$(dirname -- "${ANSWER_FILE}")"
cd "${REPO_ROOT}"
export PYTHONPATH="${SPARROW_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${PYTHON_BIN}" -u -m sparrow.evaluation.gen_baseline_answer_video \
    --spec-model-path "${SPEC_MODEL_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --answer-file "${ANSWER_FILE}" \
    --model-id "${MODEL_ID}" \
    --temperature "${TEMPERATURE}" \
    --data_path "${DATA_PATH}" \
    --task VideoDetailCaption \
    --model_type "${MODEL_TYPE}" \
    --frame_num "${FRAME_NUM}" \
    --data_num "${DATA_NUM}" \
    --max-new-token "${MAX_NEW_TOKEN}"
