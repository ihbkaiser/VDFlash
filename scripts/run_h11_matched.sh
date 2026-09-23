#!/usr/bin/env bash
# Equal-budget visual/text control on the exact samples from a completed H1.1 run.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (($# != 1)); then
  echo "Usage: bash scripts/run_h11_matched.sh results/h11_<timestamp>" >&2
  exit 2
fi
SOURCE_RUN=$1
if [[ "$SOURCE_RUN" != /* ]]; then
  SOURCE_RUN="$REPO_ROOT/$SOURCE_RUN"
fi
if [[ ! -f "$SOURCE_RUN/train_full.npz" || ! -f "$SOURCE_RUN/test_full_cut.npz" || ! -f "$SOURCE_RUN/test_full_cut.jsonl" ]]; then
  echo "Pass a completed H1.1 run directory with training reference and paired reports: $SOURCE_RUN" >&2
  exit 2
fi
PROFILE="$REPO_ROOT/scripts/h11_3b_20e_cache.env"
if [[ ! -f "$PROFILE" ]]; then
  echo "Missing 3B Phase 2 profile: $PROFILE" >&2
  exit 2
fi
# Same checkpoint, target, feature root, two-GPU placement, and truncation as H1.1.
# shellcheck disable=SC1090
source "$PROFILE"
PYTHON_BIN=${PYTHON_BIN:-python3}
H11_MATCHED_BUDGET=${H11_MATCHED_BUDGET:-8}
if [[ ! "$H11_MATCHED_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "H11_MATCHED_BUDGET must be a positive integer" >&2
  exit 2
fi
OUTPUT_DIR="$SOURCE_RUN/matched_k${H11_MATCHED_BUDGET}_$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output already exists: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_study() {
  echo "Equal-budget H1.1 control: remove $H11_MATCHED_BUDGET visual versus text tokens"
  echo "Using original Full/Cut samples from: $SOURCE_RUN"
  "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_h11_matched \
    --source-run "$SOURCE_RUN" --output-dir "$OUTPUT_DIR" \
    --checkpoint "$H11_CHECKPOINT" --draft-config "$H11_DRAFT_CONFIG" \
    --feature-root "$H11_FEATURE_ROOT" --target-model "$H11_TARGET_MODEL" \
    --budget "$H11_MATCHED_BUDGET" --train-max-length "${H11_TRAIN_MAX_LENGTH:-3072}" \
    --device "${H11_DEVICE:-cuda:0}" --draft-device "${H11_DRAFT_DEVICE:-cuda:1}" \
    --device-map "${H11_DEVICE_MAP:-cuda}" --max-memory "${H11_MAX_MEMORY:-0:22GiB,1:14GiB}" || return $?
  "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11_matched \
    --source-run "$SOURCE_RUN" --matched-run "$OUTPUT_DIR" || return $?
}

if run_study 2>&1 | tee "$OUTPUT_DIR/run.log"; then
  echo "Completed. Read: $OUTPUT_DIR/summary.txt"
else
  echo "Matched control stopped. Read: $OUTPUT_DIR/run.log" >&2
  exit 1
fi
