#!/usr/bin/env python3
"""Run raw download, integrity checks and cataloging as one resumable job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from download_posttraining_raw import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dflash_posttraining/raw/llava_video_178k"))
    parser.add_argument("--exclude-vdc-metadata", type=Path, default=Path("dataset/VideoDetailCaption/source_metadata.parquet"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--range-workers", type=int, default=16)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Prevent two managed jobs from writing the same raw transfer files.
    lock = (root / "collection.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A collection job already holds this output directory") from None
    scripts = Path(__file__).resolve().parent
    status_path = root / "pipeline_status.json"
    status = {"status": "running", "stage": "download", "started_utc": datetime.now(timezone.utc).isoformat(),
              "output_dir": str(root), "target_responses_generated": False, "features_cached": False}
    write_json(status_path, status)
    commands = [
        ("download", [sys.executable, "-u", str(scripts / "download_posttraining_raw.py"),
                      "--output-dir", str(root), "--workers", str(args.workers),
                      "--range-workers", str(args.range_workers)]),
        ("catalog", [sys.executable, "-u", str(scripts / "catalog_posttraining_raw.py"),
                     "--raw-dir", str(root), "--exclude-vdc-metadata", str(args.exclude_vdc_metadata.resolve())]),
    ]
    for stage, command in commands:
        status["stage"] = stage
        write_json(status_path, status)
        print(f"[pipeline] starting {stage}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            status.update(status="incomplete", failed_stage=stage, exit_code=result.returncode,
                          updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(status_path, status)
            raise SystemExit(result.returncode)
    downloads = json.loads((root / "download_status.json").read_text())
    checksum_file = root / "SHA256SUMS"
    checksum_file.write_text("".join(f"{row['sha256']}  {row['path']}\n"
                                    for row in sorted(downloads["files"], key=lambda item: item["path"])))
    summary = json.loads((root / "collection_summary.json").read_text())
    status.update(status="complete", stage="raw_collection_verified",
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  unique_videos=summary["unique_video_keys"],
                  eligible_annotated_videos=summary["annotated_unique_videos_after_vdc_exclusion"])
    write_json(status_path, status)
    print("[pipeline] complete: raw source files verified and cataloged", flush=True)


if __name__ == "__main__":
    main()
