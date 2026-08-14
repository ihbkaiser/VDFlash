#!/usr/bin/env bash
# Sequential T4 GPU run for the Sparrow-insight validation with MSD.
#
# Stages run one after another (single 16 GB T4), each in its own process so
# VRAM is released between stages.  Completed stages are skipped on re-run
# (resume).  The script waits for the fast calibration to finish and for the
# Qwen2.5-VL-7B download (needed by the layer analysis) to complete.
#
# Usage: ./run_sparrow_stages_t4.sh [--limit N] [--skip-msd] ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-results/sparrow_validation}"
LOG_DIR="$OUTPUT_DIR/stage_logs"
mkdir -p "$LOG_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CALIBRATION="$OUTPUT_DIR/calibration.jsonl"
MSD_OUT="$OUTPUT_DIR/msd.jsonl"
ATTENTION_OUT="$OUTPUT_DIR/figure2_attention.jsonl"
DRAFT_ATTENTION_OUT="$OUTPUT_DIR/figure2_draft_attention.jsonl"
LAYERS_OUT="$OUTPUT_DIR/layer_analysis.jsonl"

EXTRA_ARGS=("$@")

echo "== Sparrow T4 stages: $(date -Is) =="

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

# 1. Wait for the fast calibration (2,4,8,16,32,64 frames x 200704,401408 px).
#    The calibrate process writes incrementally; we wait for 200 rows
#    (50 samples x 4 milestones) or for the process to exit.
for _ in $(seq 1 400); do
    if [[ -f "$CALIBRATION" ]]; then
        ROWS=$(wc -l < "$CALIBRATION" 2>/dev/null || echo 0)
        if [[ "$ROWS" -ge 200 ]]; then
            echo "calibration complete: $ROWS rows"
            break
        fi
    fi
    if ! pgrep -f "Validate_Sparrow_hypoth[e]sises calibrate" >/dev/null; then
        echo "calibration process exited; rows=$(wc -l < "$CALIBRATION" 2>/dev/null || echo 0)"
        break
    fi
    sleep 15
done

CAL_ARGS=(--calibration "$CALIBRATION" --allow-out-of-tolerance --visual-targets 400 3000)

echo "== stage msd (Figure 1a+1b, milestones 400/3000) $(date -Is) =="
# The msd runner is idempotent: it resumes completed jobs from the output
# file (incremental write), so re-running after a pause is cheap.
python -u -m src.analyze.Validate_Sparrow_hypothesises msd \
    "${CAL_ARGS[@]}" --condition both \
    --output "$MSD_OUT" "${EXTRA_ARGS[@]}" \
    > "$LOG_DIR/msd.log" 2>&1 || { echo "msd FAILED"; tail -20 "$LOG_DIR/msd.log"; exit 1; }

if [[ -s "$ATTENTION_OUT" ]]; then
    echo "skip attention (exists)"
else
    echo "== stage attention (Figure 2 target proxy) $(date -Is) =="
    python -u -m src.analyze.Validate_Sparrow_hypothesises attention \
        "${CAL_ARGS[@]}" --quantized \
        --output "$ATTENTION_OUT" "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/attention.log" 2>&1 || { echo "attention FAILED"; tail -20 "$LOG_DIR/attention.log"; exit 1; }
fi

if [[ -s "$DRAFT_ATTENTION_OUT" ]]; then
    echo "skip draft_attention (exists)"
else
    echo "== stage draft_attention (Figure 2 MSD draft) $(date -Is) =="
    python -u -m src.analyze.Validate_Sparrow_hypothesises draft_attention \
        "${CAL_ARGS[@]}" \
        --output "$DRAFT_ATTENTION_OUT" "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/draft_attention.log" 2>&1 || { echo "draft_attention FAILED"; tail -20 "$LOG_DIR/draft_attention.log"; exit 1; }
fi

# 2. Wait for the Qwen2.5-VL-7B download before the layer analysis.
for _ in $(seq 1 400); do
    if [[ -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct" ]] \
        && [[ -f "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/refs/main" ]]; then
        break
    fi
    sleep 15
done

if [[ -s "$LAYERS_OUT" ]]; then
    echo "skip layers (exists)"
else
    echo "== stage layers (Figure 3+6, Qwen2.5-VL-7B) $(date -Is) =="
    python -u -m src.analyze.Validate_Sparrow_hypothesises layers \
        "${CAL_ARGS[@]}" --experiments both --quantized \
        --output "$LAYERS_OUT" "${EXTRA_ARGS[@]}" \
        > "$LOG_DIR/layers.log" 2>&1 || { echo "layers FAILED"; tail -20 "$LOG_DIR/layers.log"; exit 1; }
fi

# 3. Merge + audit + report.
echo "== merge + audit + report $(date -Is) =="
python -u - <<PY
import json
from pathlib import Path
paths = [Path("$MSD_OUT"), Path("$ATTENTION_OUT"), Path("$DRAFT_ATTENTION_OUT"), Path("$LAYERS_OUT")]
rows = []
for path in paths:
    if path.exists():
        rows.extend(json.loads(line) for line in path.open() if line.strip())
Path("$OUTPUT_DIR/results.jsonl").write_text(
    "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
)
print(f"merged {len(rows)} rows")
PY
python -u -m src.analyze.Validate_Sparrow_hypothesises audit \
    --input "$OUTPUT_DIR/results.jsonl" --output "$OUTPUT_DIR/audit.json" || true
python -u -m src.analyze.Validate_Sparrow_hypothesises report \
    --input "$OUTPUT_DIR/results.jsonl" --output-dir "$OUTPUT_DIR/report" || true

echo "== Done: $OUTPUT_DIR/report/REPORT.md =="
