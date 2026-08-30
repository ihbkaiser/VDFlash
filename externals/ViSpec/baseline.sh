#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../scripts/resolve_workspace.sh"
cd "${REPO_ROOT}"
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 1048576

spec_dir=""
bench_dir="${BENCH_DIR:-${REPO_ROOT}/externals/ViSpec/vispec_data/bench_data/}"
result_dir="${RESULT_DIR:-${REPO_ROOT}/results/vispec/}"
result_name=""
base_model=""
temperature="1.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec_dir)
      spec_dir="$2"
      shift 2
      ;;
    --bench_dir)
      bench_dir="$2"
      shift 2
      ;;
    --result_dir)
      result_dir="$2"
      shift 2
      ;;
    --result_name)
      result_name="$2"
      shift 2
      ;;
    --base_model)
      base_model="$2"
      shift 2
      ;;
    --temperature)
        temperature="$2"
        shift 2
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "${bench_dir}" != /* ]]; then
  bench_dir="${REPO_ROOT}/${bench_dir}"
fi
if [[ "${result_dir}" != /* ]]; then
  result_dir="${REPO_ROOT}/${result_dir}"
fi

if [[ -z "$spec_dir" || -z "$result_name" || -z "$base_model" ]]; then
  echo "Error: Missing required parameter."
  exit 1
fi

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_sqa \
--model-id test \
--test_split=test \
--test_number=-1 \
--shot_number=0 \
--prompt_format=QCM-ALE \
--bench-name="${result_dir}/sqa_test/${result_name}/" \
--base-model-path="$base_model" \
--spec-model-path="$spec_dir"  \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_coco_caption \
--base-model-path="$base_model" \
--model-id test \
--bench-name="${result_dir}/coco_caption_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_gqa \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/gqa/" \
--bench-name="${result_dir}/gqa_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_mme \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/MME/" \
--bench-name="${result_dir}/mme_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_mmvet \
--base-model-path="$base_model" \
--model-id test \
--bench-name="${result_dir}/mmvet_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_seed_bench \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/seed_bench/" \
--bench-name="${result_dir}/seed_bench_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_textvqa \
--base-model-path="$base_model" \
--model-id test \
--bench-name="${result_dir}/textvqa_test/${result_name}/" \
--data-folder="${bench_dir}/textvqa" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"

"${PYTHON_BIN}" -u -m vispec.evaluation.gen_baseline_answer_vizwiz \
--base-model-path="$base_model" \
--model-id test \
--data-folder="${bench_dir}/vizwiz" \
--bench-name="${result_dir}/vizwiz_test/${result_name}/" \
--spec-model-path="$spec_dir" \
--temperature="$temperature"
