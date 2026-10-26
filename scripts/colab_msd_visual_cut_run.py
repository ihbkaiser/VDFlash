#!/usr/bin/env python3
"""Launch the corrected runner inside the prepared Colab workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/content/vdflash")
TARGET_MODEL_PATH = (
    "/content/hf/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/"
    "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
)
MSD_MODEL_PATH = (
    "/content/hf/models--lucylyn--MSD-Qwen2VL-7B-Instruct/snapshots/"
    "c764d2ebd9b45d86dbf7960bffc918806c015234"
)


def main() -> None:
    limit = int(os.environ.get("MSD_RUN_LIMIT", "1"))
    repeats = int(os.environ.get("MSD_RUN_REPEATS", "1"))
    output = os.environ.get(
        "MSD_RUN_OUTPUT",
        "results/22_10_26/msd_qwen2vl7b_visual_cut_colab_pilot.jsonl",
    )
    summary = os.environ.get(
        "MSD_RUN_SUMMARY",
        "results/22_10_26/msd_qwen2vl7b_visual_cut_colab_pilot_summary.json",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["HF_HOME"] = "/content/hf"
    env["MSD_VISION_CHUNK_FRAMES"] = "8"
    command = [
        sys.executable,
        "scripts/run_msd_visual_verification_7b.py",
        "--manifest", "dataset/VideoDetailCaption/subset_manifest.jsonl",
        "--dataset-root", "dataset/VideoDetailCaption",
        "--base-model", TARGET_MODEL_PATH,
        "--msd-model", MSD_MODEL_PATH,
        "--target-model-id", "Qwen/Qwen2-VL-7B-Instruct",
        "--target-model-revision", "eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        "--msd-model-id", "lucylyn/MSD-Qwen2VL-7B-Instruct",
        "--msd-model-revision", "c764d2ebd9b45d86dbf7960bffc918806c015234",
        "--output", output,
        "--summary", summary,
        "--limit", str(limit),
        "--max-new-tokens", "48",
        "--fps", "8",
        "--max-frames", "16",
        "--max-pixels", "100352",
        "--device-map", "auto",
        "--max-memory", "0:20GiB",
        "--timing-repeats", str(repeats),
        "--fresh",
    ]
    output_path = ROOT / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(output_path.suffix + ".log")
    print(
        json.dumps(
            {"cwd": str(ROOT), "command": command, "log": str(log_path)},
            indent=2,
        ),
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            time.sleep(30)
            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            row_count = len(output_path.read_text(encoding="utf-8").splitlines()) if output_path.exists() else 0
            print(
                json.dumps(
                    {
                        "status": "running",
                        "pid": process.pid,
                        "completed_rows": row_count,
                        "last_log_line": log_lines[-1] if log_lines else "",
                    }
                ),
                flush=True,
            )
        return_code = process.wait()
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    print("\n".join(tail), flush=True)
    if return_code:
        raise SystemExit(return_code)


main()
