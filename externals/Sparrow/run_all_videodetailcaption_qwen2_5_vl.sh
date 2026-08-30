#!/usr/bin/env bash
# Portable dense Qwen2.5-VL baseline adapter.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TYPE="qwen2_5_vl"
FRAME_NUM="${FRAME_NUM:-16}"
exec "${SCRIPT_DIR}/run_videodetailcaption_baseline.sh" \
    --model-type "${MODEL_TYPE}" \
    --frame-num "${FRAME_NUM}" \
    "$@"
