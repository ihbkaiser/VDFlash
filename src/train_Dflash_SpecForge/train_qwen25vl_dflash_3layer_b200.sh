#!/usr/bin/env bash
set -euo pipefail

# One-command two-B200 entry point for the local Qwen2.5-VL-3B recipe.
# Private machine paths stay in the ignored sibling .env file.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${SPECFORGE_ENV_FILE:-"$ROOT_DIR/train_qwen25vl_dflash_3layer.env"}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "environment file not found: $ENV_FILE" >&2
  echo "set SPECFORGE_ENV_FILE or place train_qwen25vl_dflash_3layer.env beside this script" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

GPU_IDS=${SPECFORGE_GPU_IDS:-0,1}
LAYER_SET=${SPECFORGE_LAYER_SET:-high_only}
export SPECFORGE_GPU_IDS="$GPU_IDS"

exec bash "$ROOT_DIR/train_qwen25vl_dflash_3layer.sh" \
  --set "$LAYER_SET" \
  --gpu-ids "$GPU_IDS" \
  --env-file "$ENV_FILE" \
  "$@"
