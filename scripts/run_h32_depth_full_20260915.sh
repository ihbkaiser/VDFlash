#!/usr/bin/env bash
set -euo pipefail

# H3.2: compare independently trained DFlash depth-1 and depth-3 checkpoints.
# Jobs are intentionally sequential because all eight runs share the same two
# GPUs. Each runner appends one row after each condition, so --resume is safe.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export HF_HOME="${repo_root}/.hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONPATH="src/train_Dflash_SpecForge:."
export MPLCONFIGDIR="/tmp/vdflash-mpl"

runner=(
  .venv-msd/bin/python -m
  src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31
  h3_2
  --device cuda:0
  --draft-device cuda:1
  --device-map model_parallel
  --max-memory 0:22GiB,1:14GiB
  --dtype auto
  --max-new-tokens 64
)

vdc_calibration="results/sparrow_validation_dflash_qwen25vl3b_2026-08-23/calibration_3000.jsonl"
vdc_common=(
  --dataset vdc
  --manifest dataset/VideoDetailCaption/test.jsonl
  --video-root dataset/VideoDetailCaption
  --calibration "${vdc_calibration}"
  --target-visual-tokens 3000
  --allow-out-of-tolerance
  --limit 50
)

mvbench_common=(
  --dataset mvbench
  --manifest dataset/MVBench/classified/selected.jsonl
  --video-root dataset/MVBench
  --calibration /dev/null
  --mvbench-num-frames 8
  --mvbench-min-pixels 200704
  --mvbench-max-pixels 200704
  --max-new-tokens 16
  --limit 1000
)

run_vdc() {
  local corpus="$1"
  local depth="$2"
  "${runner[@]}" \
    --training-corpus "${corpus}" \
    --draft-depth "${depth}" \
    --checkpoint "dataset/qwen25vl-3b-depth1-depth3/qwen25vl-3b-dflash-${corpus}-h32-depth${depth}-latest/training_state.pt" \
    --draft-config "src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash-h32-depth${depth}.json" \
    "${vdc_common[@]}" \
    --output-dir "results/h32_depth_20260915/vdc/${corpus}_depth${depth}" \
    --resume
}

run_mvbench() {
  local corpus="$1"
  local depth="$2"
  "${runner[@]}" \
    --training-corpus "${corpus}" \
    --draft-depth "${depth}" \
    --checkpoint "dataset/qwen25vl-3b-depth1-depth3/qwen25vl-3b-dflash-${corpus}-h32-depth${depth}-latest/training_state.pt" \
    --draft-config "src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash-h32-depth${depth}.json" \
    "${mvbench_common[@]}" \
    --output-dir "results/h32_depth_20260915/mvbench/${corpus}_depth${depth}" \
    --resume
}

for corpus in llava68k sharegpt68k; do
  for depth in 1 3; do
    echo "[H3.2] VDC ${corpus} depth${depth}" >&2
    run_vdc "${corpus}" "${depth}"
  done
done

for corpus in llava68k sharegpt68k; do
  for depth in 1 3; do
    echo "[H3.2] MVBench ${corpus} depth${depth}" >&2
    run_mvbench "${corpus}" "${depth}"
  done
done

echo "[H3.2] all depth runs completed" >&2
