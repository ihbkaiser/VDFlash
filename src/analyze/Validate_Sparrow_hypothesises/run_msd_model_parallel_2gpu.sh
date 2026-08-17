#!/usr/bin/env bash
# Run one MSD model sharded across RTX 3090 + RTX A4000.
# This is the 25K-capable path; it is intentionally one process using both
# GPUs, not two independent full-model workers.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh"

PYTHON="${PYTHON:-$REPO_ROOT/.venv-msd/bin/python}"
GPUS="${GPUS:-0,1}"
MANIFEST="dataset/VideoDetailCaption/subset_manifest.jsonl"
DATASET_ROOT="dataset/VideoDetailCaption"
OUTPUT="results/sparrow_validation_model_parallel_2gpu/msd.jsonl"
CALIBRATION=""
LIMIT=""
CONDITION="both"
LENGTH_SERIES="keep_visual"
RETENTION_PERCENTAGES=""
VISUAL_TARGETS="400,3000,13000,25000"
MAX_NEW_TOKENS="512"
MAX_MEMORY="0:22GiB,1:14GiB"
ALLOW_OUT_OF_TOLERANCE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_msd_model_parallel_2gpu.sh [options]

Runs ONE MSD process whose base model is explicitly sharded across both GPUs.
The 3090 owns vision, first/last decoder layers, lm_head and EAGLE draft;
the A4000 owns middle decoder layers and their KV cache.

Options:
  --gpus 0,1                 physical GPU IDs (default: 0,1)
  --manifest PATH            input VDC manifest
  --dataset-root PATH        root containing the videos
  --output PATH              output JSONL
  --calibration PATH         existing measured calibration JSONL
  --limit N                  use the first N samples
  --condition full|retention|both
  --length-series keep_visual|remove_all
  --retention-percentages LIST comma-separated retention values
  --visual-targets LIST      comma-separated targets
  --max-new-tokens N         generation length (default: 512)
  --max-memory SPEC          per-visible-GPU budget (default: 0:22GiB,1:14GiB)
  --allow-out-of-tolerance   run nearest measured points outside 10% tolerance
  --no-allow-out-of-tolerance
  --dry-run                  print the command without running
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --calibration) CALIBRATION="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --condition) CONDITION="$2"; shift 2 ;;
        --length-series) LENGTH_SERIES="$2"; shift 2 ;;
        --retention-percentages) RETENTION_PERCENTAGES="$2"; shift 2 ;;
        --visual-targets) VISUAL_TARGETS="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --max-memory) MAX_MEMORY="$2"; shift 2 ;;
        --allow-out-of-tolerance) ALLOW_OUT_OF_TOLERANCE=1; shift ;;
        --no-allow-out-of-tolerance) ALLOW_OUT_OF_TOLERANCE=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ "${#GPU_IDS[@]}" -ne 2 || "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
    echo "--gpus must contain two different IDs, for example --gpus 0,1" >&2
    exit 2
fi
if [[ "$LENGTH_SERIES" != "keep_visual" && "$LENGTH_SERIES" != "remove_all" ]]; then
    echo "--length-series must be keep_visual or remove_all" >&2
    exit 2
fi
[[ -f "$MANIFEST" ]] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }

OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"
if [[ -z "$CALIBRATION" ]]; then
    CALIBRATION="$OUTPUT_DIR/calibration.jsonl"
fi
if [[ ! -s "$CALIBRATION" ]]; then
    CALIBRATION_LIMIT=()
    [[ -n "$LIMIT" ]] && CALIBRATION_LIMIT+=(--limit "$LIMIT")
    "$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises calibrate \
        --manifest "$MANIFEST" --dataset-root "$DATASET_ROOT" \
        --output "$CALIBRATION" "${CALIBRATION_LIMIT[@]}"
fi

IFS=',' read -r -a TARGET_VALUES <<< "$VISUAL_TARGETS"
TARGET_ARGS=()
for target in "${TARGET_VALUES[@]}"; do TARGET_ARGS+=("$target"); done
LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS+=(--limit "$LIMIT")

CMD=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises msd
    --manifest "$MANIFEST"
    --dataset-root "$DATASET_ROOT"
    --calibration "$CALIBRATION"
    --visual-targets "${TARGET_ARGS[@]}"
    --condition "$CONDITION"
    --length-series "$LENGTH_SERIES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --device-map model_parallel
    --max-memory "$MAX_MEMORY"
    --output "$OUTPUT"
    "${LIMIT_ARGS[@]}")
if [[ -n "$RETENTION_PERCENTAGES" ]]; then
    IFS=',' read -r -a RETENTION_VALUES <<< "$RETENTION_PERCENTAGES"
    CMD+=(--retention-percentages "${RETENTION_VALUES[@]}")
fi
if [[ "$ALLOW_OUT_OF_TOLERANCE" == "1" ]]; then
    CMD+=(--allow-out-of-tolerance)
fi

echo "CUDA_VISIBLE_DEVICES=${GPU_IDS[*]} MSD_VISION_CHUNK_FRAMES=${MSD_VISION_CHUNK_FRAMES:-8}"
echo "MSD_MODEL_PARALLEL_PREFIX_LAYERS=${MSD_MODEL_PARALLEL_PREFIX_LAYERS:-3}"
echo "MSD_MODEL_PARALLEL_SUFFIX_LAYERS=${MSD_MODEL_PARALLEL_SUFFIX_LAYERS:-2}"
echo "Command: ${CMD[*]}"
[[ "$DRY_RUN" == "1" ]] && exit 0

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MSD_VISION_CHUNK_FRAMES="${MSD_VISION_CHUNK_FRAMES:-8}"
export MSD_MODEL_PARALLEL_PREFIX_LAYERS="${MSD_MODEL_PARALLEL_PREFIX_LAYERS:-3}"
export MSD_MODEL_PARALLEL_SUFFIX_LAYERS="${MSD_MODEL_PARALLEL_SUFFIX_LAYERS:-2}"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]},${GPU_IDS[1]}" "${CMD[@]}" \
    >"$OUTPUT_DIR/msd_model_parallel.log" 2>&1

echo "Output: $OUTPUT"
