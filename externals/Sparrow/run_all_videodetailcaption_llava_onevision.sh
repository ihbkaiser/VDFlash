#!/usr/bin/env bash
# Portable dense LLaVA-OneVision baseline adapter.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TYPE="llava_ov"
FRAME_NUM="${FRAME_NUM:-32}"
exec "${SCRIPT_DIR}/run_videodetailcaption_baseline.sh" \
    --model-type "${MODEL_TYPE}" \
    --frame-num "${FRAME_NUM}" \
    "$@"
