#!/usr/bin/env bash
# Run the existing-checkpoint H1.1 reference, paired Full/Cut probe, and report.
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=

usage() {
  cat <<'EOF'
Usage: bash scripts/run_dflash_h11.sh --env-file /path/to/h11.env

The env file defines H11_CHECKPOINT, H11_DRAFT_CONFIG, H11_FEATURE_ROOT,
and H11_TARGET_MODEL. H11_EVAL_MODE=cache (default) needs no videos.
H11_EVAL_MODE=video additionally needs H11_MANIFEST, H11_VIDEO_ROOT,
and H11_CALIBRATION.
Copy scripts/h11.env.example and fill in your existing GPU-machine paths.

Optional values: H11_OUTPUT_DIR, H11_REFERENCE_SAMPLES (200), H11_LIMIT (2),
H11_TRAIN_MAX_LENGTH (3072), H11_MAX_NEW_TOKENS (32), H11_DEVICE (cuda:0),
H11_DRAFT_DEVICE (cuda:1), H11_DEVICE_MAP (model_parallel), H11_MAX_MEMORY.
EOF
}

while (($#)); do
  case "$1" in
    --env-file) ENV_FILE=${2:?--env-file requires a path}; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z "$ENV_FILE" || ! -f "$ENV_FILE" ]]; then
  echo "Pass an existing --env-file (see scripts/h11.env.example)" >&2
  exit 2
fi
# Explicit user-selected, shell-format config; same convention as the Phase 2 launcher.
# shellcheck disable=SC1090
source "$ENV_FILE"

PYTHON_BIN=${PYTHON_BIN:-python3}
H11_REFERENCE_SAMPLES=${H11_REFERENCE_SAMPLES:-200}
H11_EVAL_MODE=${H11_EVAL_MODE:-cache}
H11_TRAIN_MAX_LENGTH=${H11_TRAIN_MAX_LENGTH:-3072}
H11_LIMIT=${H11_LIMIT:-2}
H11_MAX_NEW_TOKENS=${H11_MAX_NEW_TOKENS:-32}
H11_TARGET_VISUAL_TOKENS=${H11_TARGET_VISUAL_TOKENS:-3000}
H11_DEVICE=${H11_DEVICE:-cuda:0}
H11_DRAFT_DEVICE=${H11_DRAFT_DEVICE:-cuda:1}
H11_DEVICE_MAP=${H11_DEVICE_MAP:-model_parallel}
H11_MAX_MEMORY=${H11_MAX_MEMORY:-0:22GiB,1:14GiB}
H11_OUTPUT_DIR=${H11_OUTPUT_DIR:-"$REPO_ROOT/results/h11_$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ "$H11_OUTPUT_DIR" != /* ]]; then
  H11_OUTPUT_DIR="$REPO_ROOT/$H11_OUTPUT_DIR"
fi

case "$H11_EVAL_MODE" in
  cache|video) ;;
  *) echo "H11_EVAL_MODE must be cache or video" >&2; exit 2 ;;
esac
required=(H11_CHECKPOINT H11_DRAFT_CONFIG H11_FEATURE_ROOT H11_TARGET_MODEL)
if [[ "$H11_EVAL_MODE" == video ]]; then
  required+=(H11_MANIFEST H11_VIDEO_ROOT H11_CALIBRATION)
fi
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing $name in $ENV_FILE" >&2
    exit 2
  fi
done
for name in H11_REFERENCE_SAMPLES H11_TRAIN_MAX_LENGTH H11_LIMIT H11_MAX_NEW_TOKENS H11_TARGET_VISUAL_TOKENS; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer" >&2
    exit 2
  fi
done
local_paths=(H11_CHECKPOINT H11_DRAFT_CONFIG H11_FEATURE_ROOT)
if [[ "$H11_EVAL_MODE" == video ]]; then
  local_paths+=(H11_MANIFEST H11_VIDEO_ROOT H11_CALIBRATION)
fi
for name in "${local_paths[@]}"; do
  if [[ "${!name}" != /* ]]; then
    echo "$name must be an absolute path: ${!name}" >&2
    exit 2
  fi
  if [[ ! -e "${!name}" ]]; then
    echo "$name does not exist: ${!name}" >&2
    exit 2
  fi
done
if (( H11_REFERENCE_SAMPLES < 40 )); then
  echo "H11_REFERENCE_SAMPLES must be at least 40 for fit/calibration" >&2
  exit 2
fi
if [[ "$H11_TARGET_MODEL" == /* && ! -d "$H11_TARGET_MODEL" ]]; then
  echo "H11_TARGET_MODEL does not exist: $H11_TARGET_MODEL" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$H11_OUTPUT_DIR" ]]; then
  echo "Output already exists; choose a new H11_OUTPUT_DIR: $H11_OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$H11_OUTPUT_DIR"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

common=(
  --checkpoint "$H11_CHECKPOINT"
  --draft-config "$H11_DRAFT_CONFIG"
  --target-model "$H11_TARGET_MODEL"
  --device "$H11_DEVICE"
  --draft-device "$H11_DRAFT_DEVICE"
  --device-map "$H11_DEVICE_MAP"
  --max-memory "$H11_MAX_MEMORY"
  --output-dir "$H11_OUTPUT_DIR"
)

run_pipeline() {
  echo "H1.1 output: $H11_OUTPUT_DIR"
  echo "Checkpoint: $H11_CHECKPOINT"
  echo "Mode: $H11_EVAL_MODE; reference samples: $H11_REFERENCE_SAMPLES; evaluation samples: $H11_LIMIT"
  echo "[1/3] Capture Full representations on the Phase 2 training features"
  "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_h11 reference \
    "${common[@]}" \
    --feature-root "$H11_FEATURE_ROOT" \
    --reference-samples "$H11_REFERENCE_SAMPLES" \
    --train-max-length "$H11_TRAIN_MAX_LENGTH" || return $?

  if [[ "$H11_EVAL_MODE" == cache ]]; then
    echo "[2/3] Paired Full/Cut on disjoint cached LLaVA images; no video files"
    "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_h11 evaluate-cache \
      "${common[@]}" \
      --feature-root "$H11_FEATURE_ROOT" \
      --train-max-length "$H11_TRAIN_MAX_LENGTH" \
      --limit "$H11_LIMIT" || return $?
  else
    echo "[2/3] Paired Full/Cut evaluation; target receives the full video"
    "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_h11 evaluate \
      "${common[@]}" \
      --manifest "$H11_MANIFEST" \
      --video-root "$H11_VIDEO_ROOT" \
      --calibration "$H11_CALIBRATION" \
      --target-visual-tokens "$H11_TARGET_VISUAL_TOKENS" \
      --limit "$H11_LIMIT" --max-new-tokens "$H11_MAX_NEW_TOKENS" || return $?
  fi

  echo "[3/3] Paired representation/acceptance analysis"
  "$PYTHON_BIN" -m src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11 \
    --reference "$H11_OUTPUT_DIR/train_full.npz" \
    --evaluation "$H11_OUTPUT_DIR/test_full_cut.npz" \
    --reports "$H11_OUTPUT_DIR/test_full_cut.jsonl" \
    --output-dir "$H11_OUTPUT_DIR/analysis" || return $?

  "$PYTHON_BIN" - "$H11_OUTPUT_DIR/analysis/summary.json" <<'PY' | tee "$H11_OUTPUT_DIR/summary.txt" || return $?
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
print("\nH1.1 paired results (Cut minus Full)")
mode = report["layers"][0].get("verification_mode")
print(f"Verification: {mode}")
print("Distance > 0: Cut farther from training reference")
print("Layer  Site       N    Distance delta [bootstrap 95% CI]   OOD Full/Cut   First-block Δ")
for row in report["layers"]:
    low, high = row["paired_delta_ci95"]
    first_block = row.get("mean_first_block_acceptance_delta", float("nan"))
    print(
        f'{row["layer"]:>5}  {row["site"]:<9} {row["n_test"]:>3}  '
        f'{row["mean_paired_delta"]:>9.3f} [{low:>8.3f}, {high:>8.3f}]  '
        f'{row["full_ood_rate"]:.1%}/{row["cut_ood_rate"]:.1%}       '
        f'{first_block:+.2f}'
    )
attention_rows = [row for row in report["layers"] if "mean_full_text_mass" in row]
if attention_rows:
    print("\nDraft attention at the first-block query (Full -> Cut; Cut minus Full):")
    print("Layer   Text mass Full/Cut (delta)     Top-5 text share Full/Cut (delta)")
    for row in attention_rows:
        print(
            f'{row["layer"]:>5}   '
            f'{row["mean_full_text_mass"]:.3f}/{row["mean_cut_text_mass"]:.3f} '
            f'({row["mean_paired_text_mass_delta"]:+.3f})'
            f'              {row["mean_full_top5_text_mass_conditional"]:.3f}/'
            f'{row["mean_cut_top5_text_mass_conditional"]:.3f} '
            f'({row["mean_paired_top5_text_mass_conditional_delta"]:+.3f})'
        )
if mode == "cached_teacher_tokens":
    print("First-block match is against cached teacher tokens; no fresh target decode or accuracy score.")
else:
    print("Acceptance reflects agreement with the target, not caption accuracy.")
PY
}

if run_pipeline 2>&1 | tee "$H11_OUTPUT_DIR/run.log"; then
  echo "Completed. Read: $H11_OUTPUT_DIR/summary.txt"
  echo "Full log: $H11_OUTPUT_DIR/run.log"
else
  echo "H1.1 stopped with an error. Read: $H11_OUTPUT_DIR/run.log" >&2
  exit 1
fi
