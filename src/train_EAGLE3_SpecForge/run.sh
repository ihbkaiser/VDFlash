#!/usr/bin/env bash
set -euo pipefail

# Keep every Hugging Face-related component strictly offline.  In particular,
# the data phase must not issue the metadata HEAD request that older versions
# of `datasets` make for `load_dataset("json", ...)`.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SOURCE_DATA=/workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
TARGET_MODEL_PATH=/workspace/storage-shared/nlp/tungdd11/tungdecoder/models/qwen25-vl-3b \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
bash train_qwen25vl_eagle3_text.sh \
  --config examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase all
