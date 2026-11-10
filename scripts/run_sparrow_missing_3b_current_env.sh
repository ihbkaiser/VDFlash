#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tuantb/VDFlash"
PYTHON="${PYTHON:-/home/tuantb/miniconda3/bin/python}"
MODEL="${MODEL:-/home/tuantb/models/Qwen2.5-VL-3B-Instruct}"
OUT="${OUT:-${ROOT}/results/sparrow_validation_dflash_qwen25vl3b_2026-08-23}"
BASE="${ROOT}/results/sparrow_validation/calibration.jsonl"
MISSING_IDS=(
  v_QN-4aLiDhiQ
  v_aLb79wtvxoQ
  v_tXUssW4vT3o
  v_zxr6UZKPDh4
)

export HF_HOME="${HF_HOME:-/home/tuantb/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord

mkdir -p "${OUT}"
test -x "${PYTHON}"
test -f "${BASE}"
test -d "${MODEL}"

"${PYTHON}" - "${ROOT}" "${OUT}" "${BASE}" <<'PY'
import json
import sys
from pathlib import Path

root, out, base_path = map(Path, sys.argv[1:])
ids = {
    "v_QN-4aLiDhiQ",
    "v_aLb79wtvxoQ",
    "v_tXUssW4vT3o",
    "v_zxr6UZKPDh4",
}

manifest = root / "dataset/VideoDetailCaption/subset_manifest.jsonl"
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
selected = [row for row in rows if row.get("video_name") in ids]
if {row.get("video_name") for row in selected} != ids or len(selected) != 4:
    raise SystemExit(f"Expected exactly 4 selected videos, found {len(selected)}")
(out / "calibration_25000_missing_manifest_currentenv.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
)

base = [json.loads(line) for line in base_path.read_text().splitlines() if line.strip()]
if len(base) != 200:
    raise SystemExit(f"Expected 200 base rows, found {len(base)}")
non_25k = [row for row in base if row.get("target_visual_tokens") != 25000]
remaining = [
    row for row in base
    if row.get("target_visual_tokens") == 25000
    and row.get("sample_id") not in ids
]
if len(non_25k) != 150 or len(remaining) != 46:
    raise SystemExit(f"Unexpected shards: non_25k={len(non_25k)}, remaining={len(remaining)}")
(out / "calibration_qwen25vl3b_currentenv.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in non_25k)
)
(out / "calibration_25000_remaining_currentenv.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in remaining)
)
print(f"Prepared manifest=4, base_non_25k=150, base_remaining_25k=46")
PY

"${PYTHON}" -u -m src.analyze.Validate_Sparrow_hypothesises \
  --contract "${ROOT}/src/analyze/Validate_Sparrow_hypothesises/configs/local_insight_vdc50.yaml" \
  calibrate \
  --manifest "${OUT}/calibration_25000_missing_manifest_currentenv.jsonl" \
  --dataset-root "${ROOT}/dataset/VideoDetailCaption" \
  --model "${MODEL}" \
  --min-pixels 100352 \
  --max-pixels 602112 \
  --targets 25000 \
  --grid-frames 1,2,3,4,6,8,12,16,24,32,48,64,96,128,160,192 \
  --grid-pixels 100352,125440,150528,163072,175616,188160,200704,225792,250880,275968,301056,351232,401408,501760,602112 \
  --output "${OUT}/calibration_25000_missing_3b_currentenv.jsonl" \
  2>&1 | tee "${OUT}/calibration_25000_missing_3b_currentenv.log"

"${PYTHON}" - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
paths = [
    out / "calibration_qwen25vl3b_currentenv.jsonl",
    out / "calibration_25000_remaining_currentenv.jsonl",
    out / "calibration_25000_missing_3b_currentenv.jsonl",
]
rows = []
for path in paths:
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())

normalized = []
for row in rows:
    row = dict(row)
    row["row_id"] = row.get("row_id") or f'{row["sample_id"]}:{int(row["target_visual_tokens"])}'
    normalized.append(row)

by_id = {}
for row in normalized:
    by_id[row["row_id"]] = row
if len(by_id) != 200:
    raise SystemExit(f"Expected 200 unique rows after merge, found {len(by_id)}")

merged = sorted(by_id.values(), key=lambda row: (row["sample_id"], int(row["target_visual_tokens"])))
final = out / "calibration_complete_20260823_currentenv.jsonl"
final.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged))
print(f"Wrote {len(merged)} unique rows to {final}")
PY
