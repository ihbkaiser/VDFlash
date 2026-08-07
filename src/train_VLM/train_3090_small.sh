#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${VIDEO_DFLASH_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
TEXT_CONFIG="${SCRIPT_DIR}/config_3090_small_text.json"
MM_CONFIG="${SCRIPT_DIR}/config_3090_small_multimodal.json"
FORCE_OVERWRITE="${VIDEO_DFLASH_OVERWRITE:-0}"
OVERWRITE_ARGS=()
if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

cd "${REPO_ROOT}"

if [[ "${FORCE_OVERWRITE}" == "1" || ! -f artifacts/video_dflash_3090_small_v2/stage1_sharegpt/manifest.jsonl ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.prepare_data --config "${TEXT_CONFIG}" "${OVERWRITE_ARGS[@]}"
fi
if [[ "${FORCE_OVERWRITE}" == "1" || ! -f artifacts/video_dflash_3090_small_v2/stage1_sharegpt/teacher_cache/metadata.json ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.cache_teacher_features --config "${TEXT_CONFIG}" "${OVERWRITE_ARGS[@]}"
fi
if [[ "${FORCE_OVERWRITE}" == "1" || ! -f artifacts/video_dflash_3090_small_v2/stage1_sharegpt/checkpoint/model.safetensors ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.train_draft --config "${TEXT_CONFIG}" "${OVERWRITE_ARGS[@]}"
fi

if [[ "${FORCE_OVERWRITE}" == "1" || ! -f artifacts/video_dflash_3090_small_v2/stage2_llava/manifest.jsonl ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.prepare_data --config "${MM_CONFIG}" "${OVERWRITE_ARGS[@]}"
fi
if [[ "${FORCE_OVERWRITE}" == "1" || ! -f artifacts/video_dflash_3090_small_v2/stage2_llava/teacher_cache/metadata.json ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.cache_teacher_features --config "${MM_CONFIG}" "${OVERWRITE_ARGS[@]}"
fi
if [[ "${FORCE_OVERWRITE}" == "1" || ! -f artifacts/video_dflash_3090_small_v2/stage2_llava/checkpoint/model.safetensors ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.train_draft --config "${MM_CONFIG}" "${OVERWRITE_ARGS[@]}"
fi
