#!/usr/bin/env bash
# Run the MSD Figure 1 experiments concurrently on two GPUs.
#
# This is data-parallel execution: each process owns one GPU and receives a
# disjoint subset of the VideoDetailCaption manifest. It intentionally does
# not use torchrun because the MSD runner is batch-size-one and has no DDP
# synchronization.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv-msd/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv-msd/bin/python"
    else
        PYTHON="python"
    fi
fi

GPUS="${GPUS:-0,1}"
MANIFEST="dataset/VideoDetailCaption/subset_manifest.jsonl"
DATASET_ROOT="dataset/VideoDetailCaption"
OUTPUT_DIR="results/sparrow_validation_2gpu"
CALIBRATION=""
LIMIT=""
CONDITION="both"
LENGTH_SERIES="keep_visual"
RETENTION_PERCENTAGES=""
VISUAL_TARGETS="400,3000,13000,25000"
ALLOW_OUT_OF_TOLERANCE=0
SKIP_CALIBRATION=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_msd_2gpu.sh [options]

Runs two independent MSD workers concurrently, one per GPU, then merges,
audits, and reports the combined JSONL output.

Options:
  --gpus 0,1                 GPU IDs (default: 0,1)
  --manifest PATH            input VideoDetailCaption manifest
  --dataset-root PATH        root containing the videos
  --output-dir PATH          output directory
  --calibration PATH         existing calibration JSONL
  --limit N                  use the first N samples (minimum 2 for two GPUs)
  --condition full|retention|both
  --length-series keep_visual|remove_all
  --retention-percentages LIST comma-separated retention values
  --visual-targets LIST      comma-separated targets, e.g. 400,3000
  --allow-out-of-tolerance   run measured points outside the 10% calibration tolerance
  --skip-calibration         require --calibration to already exist
  --no-allow-out-of-tolerance
  --dry-run                  print worker commands without running models
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
    echo "--gpus must contain exactly two IDs, for example --gpus 0,1" >&2
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
if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [[ -z "$CALIBRATION" ]]; then
    CALIBRATION="$OUTPUT_DIR/calibration.jsonl"
fi

MANIFEST_GPU0="$OUTPUT_DIR/manifest_gpu${GPU_IDS[0]}.jsonl"
MANIFEST_GPU1="$OUTPUT_DIR/manifest_gpu${GPU_IDS[1]}.jsonl"

"$PYTHON" - "$MANIFEST" "$MANIFEST_GPU0" "$MANIFEST_GPU1" "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

source, output0, output1, limit = sys.argv[1:]
rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
if limit:
    rows = rows[: int(limit)]
if len(rows) < 2:
    raise SystemExit("At least two manifest rows are required to use two GPUs")

# Round-robin keeps the two workers balanced even when the manifest is ordered.
parts = (rows[::2], rows[1::2])
for path, part in zip((output0, output1), parts):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in part),
        encoding="utf-8",
    )
    print(f"{path}: {len(part)} samples")
PY

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ "$SKIP_CALIBRATION" == "1" ]]; then
        [[ -s "$CALIBRATION" ]] || {
            echo "Calibration file is missing or empty: $CALIBRATION" >&2
            exit 1
        }
    elif [[ ! -s "$CALIBRATION" ]]; then
        CALIBRATION_LIMIT=()
        if [[ -n "$LIMIT" ]]; then
            CALIBRATION_LIMIT+=(--limit "$LIMIT")
        fi
        "$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises calibrate \
            --manifest "$MANIFEST" \
            --dataset-root "$DATASET_ROOT" \
            --output "$CALIBRATION" \
            "${CALIBRATION_LIMIT[@]}"
    fi
fi

TARGET_ARGS=()
IFS=',' read -r -a TARGET_VALUES <<< "$VISUAL_TARGETS"
for target in "${TARGET_VALUES[@]}"; do
    TARGET_ARGS+=("$target")
done

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
if [[ "$ALLOW_OUT_OF_TOLERANCE" == "1" ]]; then
    COMMON_ARGS+=(--allow-out-of-tolerance)
fi

OUTPUT_GPU0="$OUTPUT_DIR/msd_gpu${GPU_IDS[0]}.jsonl"
OUTPUT_GPU1="$OUTPUT_DIR/msd_gpu${GPU_IDS[1]}.jsonl"
LOG_GPU0="$OUTPUT_DIR/msd_gpu${GPU_IDS[0]}.log"
LOG_GPU1="$OUTPUT_DIR/msd_gpu${GPU_IDS[1]}.log"

CMD0=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises msd
    --manifest "$MANIFEST_GPU0" "${COMMON_ARGS[@]}" --output "$OUTPUT_GPU0")
CMD1=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises msd
    --manifest "$MANIFEST_GPU1" "${COMMON_ARGS[@]}" --output "$OUTPUT_GPU1")

echo "GPU ${GPU_IDS[0]}: ${CMD0[*]}"
echo "GPU ${GPU_IDS[1]}: ${CMD1[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
    exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${CMD0[@]}" >"$LOG_GPU0" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" "${CMD1[@]}" >"$LOG_GPU1" 2>&1 &
PID1=$!

STATUS=0
wait "$PID0" || STATUS=$?
wait "$PID1" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
    echo "At least one MSD worker failed." >&2
    tail -40 "$LOG_GPU0" >&2 || true
    tail -40 "$LOG_GPU1" >&2 || true
    exit "$STATUS"
fi

MERGED="$OUTPUT_DIR/msd.jsonl"
"$PYTHON" - "$OUTPUT_GPU0" "$OUTPUT_GPU1" "$MERGED" <<'PY'
import json
import sys
from pathlib import Path

rows = []
seen = set()
for name in sys.argv[1:3]:
    for line in Path(name).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = row.get("row_id")
        if row_id in seen:
            raise SystemExit(f"Duplicate row_id while merging: {row_id}")
        seen.add(row_id)
        rows.append(row)
Path(sys.argv[3]).write_text(
    "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"Merged {len(rows)} rows into {sys.argv[3]}")
PY

AUDIT_STATUS=0
"$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises audit \
    --input "$MERGED" \
    --output "$OUTPUT_DIR/audit.json" || AUDIT_STATUS=$?
"$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises report \
    --input "$MERGED" \
    --output-dir "$OUTPUT_DIR/report" || true

echo "Report: $OUTPUT_DIR/report/REPORT.md"
exit "$AUDIT_STATUS"
