#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SPECFORGE_MODEL_SIZE=${SPECFORGE_MODEL_SIZE:-7b}
exec bash "$SCRIPT_DIR/train_qwen25vl_eagle3_captioning.sh" "$@"
