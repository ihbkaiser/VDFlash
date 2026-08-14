"""Mechanically wrap each runner's per-sample loop body in try/except.

The GPU runners previously aborted the whole stage on a single transient
video-read failure.  This script re-indents the loop body and appends an
audit-compatible error row (paper_figure="Figure 1(a)" + condition="error")
before `continue`.
"""
from __future__ import annotations

import sys
from pathlib import Path

LOOP = "    for index, (sample, point) in enumerate(jobs, start=1):\n"

ERROR_ROW = (
    "        except Exception as exc:  # noqa: BLE001 - transient video/OOM errors\n"
    "            print(f\"  ERROR {sample.sample_id}: {exc}\", flush=True)\n"
    "            rows.append({\n"
    "                \"row_id\": f\"{sample.sample_id}:error\",\n"
    "                \"paper_figure\": \"Figure 1(a)\",\n"
    "                \"sample_id\": sample.sample_id,\n"
    "                \"target_model\": {target_model},\n"
    "                \"temperature\": 0.0,\n"
    "                \"target_visual_tokens\": point.get(\"target_visual_tokens\") if point else None,\n"
    "                \"actual_visual_tokens\": None,\n"
    "                \"target_input_fingerprint\": \"unavailable\",\n"
    "                \"draft_input_fingerprint\": \"unavailable\",\n"
    "                \"condition\": \"error\",\n"
    "                \"status\": \"error\",\n"
    "                \"error\": str(exc),\n"
    "            })\n"
    "            continue\n"
)


def wrap(path: Path, target_model_expr: str) -> None:
    text = path.read_text()
    if LOOP not in text:
        raise SystemExit(f"loop not found in {path}")
    head, _, tail = text.partition(LOOP)
    body_lines: list[str] = []
    rest_lines: list[str] = []
    for line in tail.splitlines(keepends=True):
        if not body_lines and line.strip() and not line.startswith("        "):
            rest_lines.append(line)
            continue
        if body_lines:
            if line.strip() and not line.startswith("        "):
                rest_lines.append(line)
                continue
        body_lines.append(line)
    if not body_lines:
        raise SystemExit(f"empty body in {path}")
    new_body = "".join(("    " + line) if line.strip() else line for line in body_lines)
    error_row = ERROR_ROW.replace("{target_model}", target_model_expr)
    result = head + LOOP + "        try:\n" + new_body + error_row + "".join(rest_lines)
    path.write_text(result)
    print(f"wrapped {path.name}: {len(body_lines)} body lines")


if __name__ == "__main__":
    base = Path("src/analyze/Validate_Sparrow_hypothesises")
    wrap(base / "run_attention.py", 'args.model')
    wrap(base / "run_draft_attention.py", 'args.base_model')
    wrap(base / "run_layer_analysis.py", 'args.model')
