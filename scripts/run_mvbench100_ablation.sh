#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/resolve_workspace.sh"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/infer/mvbench100_manifest_20260823/selected.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/infer}"
LIMIT="${LIMIT:-}"

cd "${REPO_ROOT}"

COMMON_ARGS=(
  --target-model Qwen/Qwen2.5-VL-3B-Instruct
  --manifest "${MANIFEST}"
  --video-root dataset/MVBench
  --checkpoint dataset/qwen25vl-3b-dflash-llava68k-latest
  --checkpoint dataset/qwen25vl-3b-dflash-sharegpt68k-latest
  --draft-config src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json
  --device "${GPU_DEVICE}"
  --dtype auto
  --target-attention sdpa
  --dataset-format mvbench
  --num-frames 8
  --video-min-pixels 50176
  --video-max-pixels 50176
  --max-new-tokens 16
  --all-samples
  --resume
)

if [[ -n "${LIMIT}" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi

run_condition() {
  local name="$1"
  shift
  HF_HOME="${HF_HOME}" \
  HF_HUB_OFFLINE=1 \
  PYTHONPATH="${REPO_ROOT}/src/train_Dflash_SpecForge" \
  "${PYTHON_BIN}" -u -m src.infer.qwen25vl_dflash_compare \
    "${COMMON_ARGS[@]}" "$@" \
    --output-dir "${OUTPUT_ROOT}/${name}"
}

run_condition mvbench100_full_20260823
run_condition mvbench100_exp2_zero_20260823 --visual-ablation-layers 25 33 --visual-ablation-mode zero
run_condition mvbench100_exp1_zero_20260823 --visual-ablation-layers 1 9 17 25 33 --visual-ablation-mode zero
run_condition mvbench100_exp1_cut_20260823 --visual-ablation-layers 1 9 17 25 33 --visual-ablation-mode cut
