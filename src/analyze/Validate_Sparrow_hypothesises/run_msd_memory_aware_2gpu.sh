#!/usr/bin/env bash
# Memory-aware data-parallel MSD runner for a heterogeneous 3090 + A4000 host.
#
# Each GPU owns one full model instance.  Jobs are split at the calibrated
# (sample, visual-token milestone) level, not at the sample level.  The
# smaller GPU receives only jobs below its configured visual-token budget.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh"

PYTHON="${PYTHON:-$REPO_ROOT/.venv-msd/bin/python}"
GPUS="${GPUS:-0,1}"
MANIFEST="dataset/VideoDetailCaption/subset_manifest.jsonl"
DATASET_ROOT="dataset/VideoDetailCaption"
OUTPUT_DIR="results/sparrow_validation_memory_aware_2gpu"
CALIBRATION=""
LIMIT=""
CONDITION="both"
LENGTH_SERIES="keep_visual"
RETENTION_PERCENTAGES=""
VISUAL_TARGETS="400,3000,13000,25000"
A4000_MAX_VISUAL_TOKENS="${A4000_MAX_VISUAL_TOKENS:-5500}"
STRONG_MAX_VISUAL_TOKENS="${STRONG_MAX_VISUAL_TOKENS:-11000}"
STRONG_GPU_SPEED_WEIGHT="${STRONG_GPU_SPEED_WEIGHT:-1.6}"
ALLOW_OUT_OF_TOLERANCE=0
SKIP_CALIBRATION=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_msd_memory_aware_2gpu.sh [options]

The first GPU is treated as the larger/faster GPU (normally RTX 3090); the
second is treated as the limited GPU (normally RTX A4000). Jobs above the
A4000 cap go to the first GPU; jobs above the first-GPU cap are recorded as
unsupported instead of being launched.

Options:
  --gpus 0,1                 GPU IDs (default: 0,1)
  --manifest PATH            input VideoDetailCaption manifest
  --dataset-root PATH        root containing the videos
  --output-dir PATH          output directory
  --calibration PATH         existing measured calibration JSONL
  --limit N                  use the first N samples
  --condition full|retention|both
  --length-series keep_visual|remove_all
  --retention-percentages LIST comma-separated retention values
  --visual-targets LIST      comma-separated targets, e.g. 400,3000,13000,25000
  --a4000-max-visual-tokens N maximum measured tokens allowed on GPU 2 (default: 5500)
  --strong-max-visual-tokens N maximum measured tokens allowed on GPU 1 (default: 11000)
  --strong-gpu-speed-weight X estimated 3090/A4000 throughput ratio (default: 1.6)
  --skip-calibration         require --calibration to already exist
  --allow-out-of-tolerance   run measured points outside the 10% calibration tolerance
  --no-allow-out-of-tolerance
  --dry-run                  require calibration, print assignments, do not run models
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --calibration) CALIBRATION="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --condition) CONDITION="$2"; shift 2 ;;
        --length-series) LENGTH_SERIES="$2"; shift 2 ;;
        --retention-percentages) RETENTION_PERCENTAGES="$2"; shift 2 ;;
        --visual-targets) VISUAL_TARGETS="$2"; shift 2 ;;
        --a4000-max-visual-tokens) A4000_MAX_VISUAL_TOKENS="$2"; shift 2 ;;
        --strong-max-visual-tokens) STRONG_MAX_VISUAL_TOKENS="$2"; shift 2 ;;
        --strong-gpu-speed-weight) STRONG_GPU_SPEED_WEIGHT="$2"; shift 2 ;;
        --allow-out-of-tolerance) ALLOW_OUT_OF_TOLERANCE=1; shift ;;
        --skip-calibration) SKIP_CALIBRATION=1; shift ;;
        --no-allow-out-of-tolerance) ALLOW_OUT_OF_TOLERANCE=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -ne 2 || -z "${GPU_IDS[0]}" || -z "${GPU_IDS[1]}" || "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "--gpus must contain exactly two different IDs, for example --gpus 0,1" >&2
    exit 2
fi
if [[ "$CONDITION" != "full" && "$CONDITION" != "retention" && "$CONDITION" != "both" ]]; then
    echo "--condition must be full, retention, or both" >&2
    exit 2
fi
if [[ "$LENGTH_SERIES" != "keep_visual" && "$LENGTH_SERIES" != "remove_all" ]]; then
    echo "--length-series must be keep_visual or remove_all" >&2
    exit 2
fi
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
if [[ -z "$CALIBRATION" ]]; then
    CALIBRATION="$OUTPUT_DIR/calibration.jsonl"
fi

if [[ "$SKIP_CALIBRATION" == "1" ]]; then
    [[ -s "$CALIBRATION" ]] || { echo "Calibration missing: $CALIBRATION" >&2; exit 1; }
elif [[ ! -s "$CALIBRATION" ]]; then
    CALIBRATION_LIMIT=()
    [[ -n "$LIMIT" ]] && CALIBRATION_LIMIT+=(--limit "$LIMIT")
    "$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises calibrate \
        --manifest "$MANIFEST" --dataset-root "$DATASET_ROOT" \
        --output "$CALIBRATION" "${CALIBRATION_LIMIT[@]}"
fi

WORKLIST_STRONG="$OUTPUT_DIR/worklist_gpu${GPU_IDS[0]}.jsonl"
WORKLIST_LIMITED="$OUTPUT_DIR/worklist_gpu${GPU_IDS[1]}.jsonl"
UNSUPPORTED="$OUTPUT_DIR/unsupported_jobs.jsonl"
IFS=',' read -r -a TARGET_VALUES <<< "$VISUAL_TARGETS"
TARGET_ARGS=()
for target in "${TARGET_VALUES[@]}"; do TARGET_ARGS+=("$target"); done

"$PYTHON" - "$MANIFEST" "$CALIBRATION" "$WORKLIST_STRONG" "$WORKLIST_LIMITED" \
    "$UNSUPPORTED" "$LIMIT" "$A4000_MAX_VISUAL_TOKENS" "$STRONG_MAX_VISUAL_TOKENS" \
    "$STRONG_GPU_SPEED_WEIGHT" "${TARGET_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, calibration_path, strong_path, limited_path, unsupported_path = sys.argv[1:6]
limit = int(sys.argv[6]) if sys.argv[6] else None
limited_cap = int(sys.argv[7])
strong_cap = int(sys.argv[8])
strong_weight = float(sys.argv[9])
targets = {int(value) for value in sys.argv[10:]}

manifest = [json.loads(line) for line in Path(manifest_path).read_text().splitlines() if line.strip()]
if limit is not None:
    manifest = manifest[:limit]
sample_ids = {str(row["video_name"]) for row in manifest}

points = []
for line in Path(calibration_path).read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if str(row.get("sample_id")) not in sample_ids:
        continue
    target = int(row.get("target_visual_tokens", -1))
    actual = row.get("actual_visual_tokens")
    if target in targets and actual is not None:
        points.append((str(row["sample_id"]), target, int(actual)))

expected = len(sample_ids) * len(targets)
if len(points) != expected:
    raise SystemExit(
        f"Calibration has {len(points)} usable sample/target points; expected {expected}. "
        "Run calibration for the same manifest and --visual-targets."
    )

# Largest jobs first.  The load is normalized by the measured speed ratio,
# while the limited GPU is never eligible for a job over its VRAM cap.
points.sort(key=lambda item: (-item[2], item[0], item[1]))
loads = [0.0, 0.0]
assigned = [[], []]
unsupported = []
for sample_id, target, actual in points:
    if actual > strong_cap:
        unsupported.append({
            "sample_id": sample_id,
            "target_visual_tokens": target,
            "measured_visual_tokens": actual,
            "reason": "exceeds_single_gpu_limit",
            "strong_gpu_limit": strong_cap,
        })
        continue
    eligible = [0] if actual > limited_cap else [0, 1]
    gpu = min(eligible, key=lambda index: loads[index])
    assigned[gpu].append({
        "sample_id": sample_id,
        "target_visual_tokens": target,
        "measured_visual_tokens": actual,
    })
    loads[gpu] += actual / (strong_weight if gpu == 0 else 1.0)

for path, rows in zip((strong_path, limited_path), assigned):
    Path(path).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
Path(unsupported_path).write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in unsupported),
    encoding="utf-8",
)
print(f"GPU strong: {len(assigned[0])} jobs, normalized load={loads[0]:.1f}")
print(f"GPU limited: {len(assigned[1])} jobs, normalized load={loads[1]:.1f}, cap={limited_cap}")
print(f"Unsupported single-GPU jobs: {len(unsupported)}, strong cap={strong_cap}")
for index, rows in enumerate(assigned):
    if rows:
        values = [row["measured_visual_tokens"] for row in rows]
        print(f"GPU {index}: measured token range {min(values)}..{max(values)}")
PY

echo "Worklist strong GPU:  $WORKLIST_STRONG"
echo "Worklist limited GPU: $WORKLIST_LIMITED"
echo "Unsupported jobs:     $UNSUPPORTED"
if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

COMMON_ARGS=(
    --dataset-root "$DATASET_ROOT"
    --calibration "$CALIBRATION"
    --visual-targets "${TARGET_ARGS[@]}"
    --condition "$CONDITION"
    --length-series "$LENGTH_SERIES"
)
if [[ -n "$RETENTION_PERCENTAGES" ]]; then
    IFS=',' read -r -a RETENTION_VALUES <<< "$RETENTION_PERCENTAGES"
    COMMON_ARGS+=(--retention-percentages "${RETENTION_VALUES[@]}")
fi
[[ "$ALLOW_OUT_OF_TOLERANCE" == "1" ]] && COMMON_ARGS+=(--allow-out-of-tolerance)

OUTPUT_STRONG="$OUTPUT_DIR/msd_gpu${GPU_IDS[0]}.jsonl"
OUTPUT_LIMITED="$OUTPUT_DIR/msd_gpu${GPU_IDS[1]}.jsonl"
LOG_STRONG="$OUTPUT_DIR/msd_gpu${GPU_IDS[0]}.log"
LOG_LIMITED="$OUTPUT_DIR/msd_gpu${GPU_IDS[1]}.log"

CMD0=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises msd
    --manifest "$MANIFEST" --worklist "$WORKLIST_STRONG" "${COMMON_ARGS[@]}" --output "$OUTPUT_STRONG")
CMD1=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises msd
    --manifest "$MANIFEST" --worklist "$WORKLIST_LIMITED" "${COMMON_ARGS[@]}" --output "$OUTPUT_LIMITED")

echo "GPU ${GPU_IDS[0]}: ${CMD0[*]}"
echo "GPU ${GPU_IDS[1]}: ${CMD1[*]}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${CMD0[@]}" >"$LOG_STRONG" 2>&1 & PID0=$!
if [[ -s "$WORKLIST_LIMITED" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" "${CMD1[@]}" >"$LOG_LIMITED" 2>&1 & PID1=$!
else
    : >"$OUTPUT_LIMITED"
    : >"$LOG_LIMITED"
    PID1=""
    echo "GPU ${GPU_IDS[1]}: no jobs below the A4000 cap; worker not started"
fi

STATUS=0
wait "$PID0" || STATUS=$?
if [[ -n "$PID1" ]]; then
    wait "$PID1" || STATUS=$?
fi
if [[ "$STATUS" -ne 0 ]]; then
    echo "At least one MSD worker failed." >&2
    tail -40 "$LOG_STRONG" >&2 || true
    tail -40 "$LOG_LIMITED" >&2 || true
    exit "$STATUS"
fi

MERGED="$OUTPUT_DIR/msd.jsonl"
"$PYTHON" - "$OUTPUT_STRONG" "$OUTPUT_LIMITED" "$MERGED" <<'PY'
import json
import sys
from pathlib import Path

rows, seen = [], set()
for name in sys.argv[1:3]:
    for line in Path(name).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("row_id") in seen:
            raise SystemExit(f"Duplicate row_id while merging: {row.get('row_id')}")
        seen.add(row.get("row_id"))
        rows.append(row)
Path(sys.argv[3]).write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Merged {len(rows)} rows into {sys.argv[3]}")
PY

AUDIT_STATUS=0
"$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises audit \
    --input "$MERGED" --output "$OUTPUT_DIR/audit.json" || AUDIT_STATUS=$?
"$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises report \
    --input "$MERGED" --output-dir "$OUTPUT_DIR/report" || true
echo "Report: $OUTPUT_DIR/report/REPORT.md"
exit "$AUDIT_STATUS"
