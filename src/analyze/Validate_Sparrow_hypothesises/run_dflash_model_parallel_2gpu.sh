#!/usr/bin/env bash
# Run one Qwen2.5-VL DFlash target sharded across two visible GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../scripts/resolve_workspace.sh"
cd "$REPO_ROOT"

PYTHON="$PYTHON_BIN"
GPUS="${GPUS:-0,1}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
CHECKPOINT="${CHECKPOINT:-dataset/qwen25vl-3b-dflash-llava68k-latest/training_state.pt}"
DRAFT_CONFIG="${DRAFT_CONFIG:-src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json}"
MANIFEST="${MANIFEST:-dataset/VideoDetailCaption/test.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-dataset/VideoDetailCaption}"
CALIBRATION_INPUT="${CALIBRATION_INPUT:-dataset/VideoDetailCaption/calibration_complete_20260823_currentenv.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sparrow_validation_dflash_qwen25vl3b_model_parallel_2gpu_2026-08-24}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"
TARGET_ATTENTION="${TARGET_ATTENTION:-flash_attention_2}"
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_MEMORY="${MAX_MEMORY:-0:22GiB,1:14GiB}"
LIMIT="${LIMIT:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: run_dflash_model_parallel_2gpu.sh [options]

Runs one DFlash target model with Qwen2.5-VL decoder layers sharded over two
GPUs. GPU 0 owns vision, boundary layers, and the final path; GPU 1 owns the
middle language-model layers and the DFlash draft.

Options:
  --gpus 0,1                 physical GPU IDs (default: 0,1)
  --manifest PATH            input VDC manifest
  --video-root PATH          root containing the videos
  --calibration PATH         measured calibration JSONL
  --output-dir PATH          dedicated result directory
  --limit N                  use the first N samples
  --sample-id ID             use one manifest row by sample ID
  --max-new-tokens N         generation length (default: 256)
  --target-attention NAME    target attention backend (default: flash_attention_2)
  --draft-device DEVICE      DFlash draft device (default: cuda:1)
  --max-memory SPEC          per-visible-GPU budget (default: 0:22GiB,1:14GiB)
  --dry-run                  print commands without loading the model
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus) GPUS="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --video-root) VIDEO_ROOT="$2"; shift 2 ;;
        --calibration) CALIBRATION_INPUT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --sample-id) SAMPLE_ID="$2"; shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --target-attention) TARGET_ATTENTION="$2"; shift 2 ;;
        --draft-device) DRAFT_DEVICE="$2"; shift 2 ;;
        --max-memory) MAX_MEMORY="$2"; shift 2 ;;
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

mkdir -p "$OUTPUT_DIR"
if [[ -n "$SAMPLE_ID" ]]; then
    SELECTED_MANIFEST="$OUTPUT_DIR/selected_manifest.jsonl"
    "$PYTHON" - "$MANIFEST" "$SAMPLE_ID" "$SELECTED_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

source, sample_id, destination = sys.argv[1:]
matches = []
with Path(source).open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("video_name") or row.get("sample_id") or row.get("id")) == sample_id:
            matches.append(row)
if len(matches) != 1:
    raise SystemExit(f"expected exactly one sample_id={sample_id}, found {len(matches)}")
Path(destination).write_text(
    json.dumps(matches[0], ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
    MANIFEST="$SELECTED_MANIFEST"
    LIMIT=1
fi
LIMIT_ARGS=()
[[ -n "$LIMIT" ]] && LIMIT_ARGS+=(--limit "$LIMIT")

LENGTH_CMD=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments length
    --target-model "$TARGET_MODEL"
    --checkpoint "$CHECKPOINT"
    --draft-config "$DRAFT_CONFIG"
    --manifest "$MANIFEST"
    --video-root "$VIDEO_ROOT"
    --calibration-input "$CALIBRATION_INPUT"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --draft-device "$DRAFT_DEVICE"
    --dtype "$DTYPE"
    --target-attention "$TARGET_ATTENTION"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --device-map model_parallel
    --max-memory "$MAX_MEMORY"
    --allow-out-of-tolerance
    "${LIMIT_ARGS[@]}")
REPORT_CMD=("$PYTHON" -u -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments report
    --target-model "$TARGET_MODEL"
    --checkpoint "$CHECKPOINT"
    --draft-config "$DRAFT_CONFIG"
    --manifest "$MANIFEST"
    --video-root "$VIDEO_ROOT"
    --calibration-input "$CALIBRATION_INPUT"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --draft-device "$DRAFT_DEVICE"
    --dtype "$DTYPE"
    --target-attention "$TARGET_ATTENTION"
    --device-map model_parallel
    --max-memory "$MAX_MEMORY"
    --allow-out-of-tolerance)

echo "CUDA_VISIBLE_DEVICES=${GPU_IDS[0]},${GPU_IDS[1]}"
echo "DFlash model placement: device-map=model_parallel max-memory=$MAX_MEMORY"
echo "Target attention: $TARGET_ATTENTION"
echo "DFlash draft device: $DRAFT_DEVICE"
echo "Selected sample: ${SAMPLE_ID:-all}"
echo "Length command: ${LENGTH_CMD[*]}"
echo "Report command: ${REPORT_CMD[*]}"
[[ "$DRY_RUN" == "1" ]] && exit 0

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]},${GPU_IDS[1]}" "${LENGTH_CMD[@]}" \
    >"$OUTPUT_DIR/length.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]},${GPU_IDS[1]}" "${REPORT_CMD[@]}" \
    >"$OUTPUT_DIR/report.log" 2>&1

echo "Output directory: $OUTPUT_DIR"
echo "Report: $OUTPUT_DIR/REPORT.md"
