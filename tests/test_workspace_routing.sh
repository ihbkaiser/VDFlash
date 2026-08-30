#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOLVER="${ROOT_DIR}/scripts/resolve_workspace.sh"

test -f "${RESOLVER}"

for launcher in \
  "${ROOT_DIR}/scripts/run_mvbench100_ablation.sh" \
  "${ROOT_DIR}/src/infer/run_qwen25vl_3b_dflash_vdc_compare.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_dflash_validation_gpu.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_msd_2gpu.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_msd_memory_aware_2gpu.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_msd_model_parallel_2gpu.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_dflash_model_parallel_2gpu.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_sparrow_stages_t4.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh" \
  "${ROOT_DIR}/src/analyze/Validate_Sparrow_hypothesises/run_tmux_full_validation.sh" \
  "${ROOT_DIR}/src/analyze/Whether_they_are_appliable_for_dDrafter/run_qwen35_dflash_t4.sh" \
  "${ROOT_DIR}/externals/Sparrow/run_videodetailcaption_baseline.sh"; do
  ! grep -qF '.venv-msd' "${launcher}"
  grep -qF 'resolve_workspace.sh' "${launcher}"
done

grep -qF 'resolve_workspace.sh' "${ROOT_DIR}/src/train_Dflash_SpecForge/run_specforge.sh"
grep -qF 'resolve_workspace.sh' "${ROOT_DIR}/externals/ViSpec/baseline.sh"
grep -qF 'resolve_workspace.sh' "${ROOT_DIR}/externals/SpecVLM/run_qwenvl.sh"
grep -qF 'resolve_workspace.sh' "${ROOT_DIR}/externals/SpecVLM/run_llava.sh"
! grep -qF '/ycji/' "${ROOT_DIR}/externals/SpecVLM/utils/utils.py"

! grep -qF '/home/hust/Phuc/VDFlash' "${ROOT_DIR}/scripts/run_mvbench100_ablation.sh"
! grep -qF 'exec python ' "${ROOT_DIR}/src/infer/run_qwen25vl_3b_dflash_vdc_compare.sh"
! grep -qF '/root/autodl-tmp' "${ROOT_DIR}/externals/Sparrow/run_videodetailcaption_baseline.sh"
grep -qF 'run_videodetailcaption_baseline.sh' "${ROOT_DIR}/externals/Sparrow/run_all_videodetailcaption_qwen2_5_vl.sh"
grep -qF 'run_videodetailcaption_baseline.sh' "${ROOT_DIR}/externals/Sparrow/run_all_videodetailcaption_llava_onevision.sh"
