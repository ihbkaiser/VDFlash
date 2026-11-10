#!/usr/bin/env python3
"""Download a pinned, bounded LLaVA-Video raw pool; never run model inference."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import threading
import time

import requests


REPO = "lmms-lab/LLaVA-Video-178K"
REVISION = "6d8c562dc26d70042a0d9704d1cae58c94b89098"
GROUPS = {
    "0_30_s_youtube_v0_1": 1,
    "30_60_s_youtube_v0_1": 1,
    "1_2_m_youtube_v0_1": 2,
    "2_3_m_youtube_v0_1": 2,
    "1_2_m_academic_v0_1": 1,
    "2_3_m_academic_v0_1": 1,
}


def download_segment(url: str, start: int, end: int, total: int, dest: Path) -> Path:
    """Fetch one bounded byte range; publish only a complete segment."""
    size = end - start + 1
    if dest.exists() and dest.stat().st_size == size:
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(8):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            part.replace(dest)
            return dest
        if offset > size:
            raise ValueError("Oversized download segment")
        try:
            range_start = start + offset
            with requests.get(
                url + f"&segment={start}-{end}&resume={offset}",
                headers={"Range": f"bytes={range_start}-{end}", "Accept-Encoding": "identity"},
                stream=True, timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                expected_range = f"bytes {range_start}-{end}/{total}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Unexpected bounded Content-Range")
                with part.open("ab") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if offset + len(chunk) > size:
                            raise ValueError("Segment exceeded requested size")
                        output.write(chunk)
                        offset += len(chunk)
            if offset == size:
                part.replace(dest)
                return dest
        except requests.RequestException:
            if attempt == 7:
                raise RuntimeError("Segment download failed after retries") from None
        time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("Segment incomplete after retries")


def download_parallel(root: Path, row: dict, workers: int) -> dict:
    """Download immutable ranges in parallel, retaining an ordered resume prefix."""
    dest = local_path(root, row["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return download_file(root, row)
    partial = dest.with_name(dest.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    total = row["size"]
    if offset > total:
        raise ValueError("Oversized partial")
    segments = dest.parent / (dest.name + ".segments")
    segments.mkdir(exist_ok=True)
    segment_size = 64 * 1024 * 1024
    ranges = [(start, min(start + segment_size, total) - 1)
              for start in range(offset, total, segment_size)]
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}?download=true"
    print(f"[parallel] {row['path']}: {len(ranges)} ranges, {workers} connections", flush=True)
    stop_progress = threading.Event()

    def report_progress():
        while not stop_progress.wait(30):
            transferred = offset
            for start, end in ranges:
                segment = segments / f"{start}-{end}"
                try:
                    transferred += segment.stat().st_size
                except FileNotFoundError:
                    try:
                        transferred += segment.with_name(segment.name + ".part").stat().st_size
                    except FileNotFoundError:
                        pass
            print(f"[progress] {row['path']}: {transferred / total:.1%} ({transferred / 1e9:.3f}/{total / 1e9:.3f} GB)", flush=True)

    progress_thread = threading.Thread(target=report_progress, daemon=True)
    progress_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(download_segment, url, start, end, total,
                                   segments / f"{start}-{end}") for start, end in ranges]
            for number, future in enumerate(as_completed(futures), 1):
                future.result()
                if number % 8 == 0 or number == len(ranges):
                    print(f"[ranges] {row['path']}: {number}/{len(ranges)}", flush=True)
    finally:
        stop_progress.set()
        progress_thread.join()
    # Ranges are committed in source order. If interrupted during consolidation,
    # the .part file remains a valid source prefix for the next invocation.
    with partial.open("ab") as output:
        for start, end in ranges:
            segment = segments / f"{start}-{end}"
            with segment.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            output.flush()
    result = download_file(root, row)
    # Only task-owned transfer chunks are removed after full source validation.
    for segment in segments.iterdir():
        segment.unlink()
    segments.rmdir()
    return result


def write_json(path: Path, value: object) -> None:
    """Publish metadata only after the complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root: Path, relative: str) -> Path:
    """Reject source paths escaping the designated raw-data directory."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe source path: {relative!r}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source path resolves outside output directory")
    return result


def remote_json(url: str) -> object:
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(20, 90))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise RuntimeError("Could not retrieve Hugging Face metadata") from None
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def build_plan(root: Path, seed: int) -> dict:
    """Select archive shards with a fixed seed, preserving original annotations."""
    plan_path = root / "download_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["revision"] != REVISION or plan["seed"] != seed:
            raise ValueError("Existing plan has a different revision/seed")
        return plan
    files = []
    inventories = {}
    rng = random.Random(seed)
    for group, count in GROUPS.items():
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{group}"
        entries = remote_json(url + "?recursive=false&limit=1000")
        inventories[group] = entries
        annotations = [
            row for row in entries
            if row["type"] == "file" and row["path"].endswith(".json")
            and ("_cap_" in row["path"] or "_oe_" in row["path"])
        ]
        if len(annotations) != 2:
            raise ValueError(f"Expected caption and open-ended annotations in {group}")
        archives = sorted(
            (row for row in entries if row["path"].endswith(".tar.gz")),
            key=lambda row: int(re.search(r"_(\d+)\.tar\.gz$", row["path"])[1]),
        )
        selected = rng.sample(archives, count)
        for row in annotations + selected:
            files.append({
                "path": row["path"], "size": row["size"],
                "source_sha256": row.get("lfs", {}).get("oid"),
                "kind": "archive" if row["path"].endswith(".tar.gz") else "annotation",
                "group": group,
            })
        print(f"[plan] {group}: {count} archive(s), 2 original annotation files", flush=True)
    files.append({"path": "README.md", "size": 7165, "source_sha256": None,
                  "kind": "source_card", "group": None})
    plan = {"repo_id": REPO, "revision": REVISION, "seed": seed,
            "stage": "raw_collection_only", "files": files,
            "total_bytes": sum(row["size"] for row in files),
            "features_cached": False, "target_responses_generated": False,
            "visual_budgets_calibrated": False,
            "selection_note": "Random archive pool; not a globally uniform video sample or final split."}
    write_json(root / "source_inventory.json", inventories)
    write_json(plan_path, plan)
    return plan


def download_file(root: Path, row: dict) -> dict:
    """Resume an interrupted raw download and validate against its source hash."""
    dest = local_path(root, row["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = row["size"]
    source_hash = row.get("source_sha256")
    if dest.exists():
        digest = sha256_file(dest)
        if dest.stat().st_size != expected or (source_hash and digest != source_hash):
            raise ValueError(f"Existing file failed validation: {row['path']}")
        print(f"[verified-existing] {row['path']}", flush=True)
        return {**row, "sha256": digest, "status": "verified"}
    partial = dest.with_name(dest.name + ".part")
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError(f"Oversized partial: {row['path']}")
        if offset == expected:
            break
        # A unique query avoids a proxy reusing a previous response to a
        # different Range request on the immutable revision URL.
        url = (f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
               f"{row['path']}?download=true&offset={offset}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                append = response.status_code == 206
                if append:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != expected:
                        raise ValueError("Unexpected Content-Range")
                elif response.status_code != 200:
                    raise ValueError("Unexpected HTTP status")
                written = offset if append else 0
                start = time.monotonic()
                last_log = start
                print(f"[download] {row['path']} from {written / 1e9:.3f}/{expected / 1e9:.3f} GB", flush=True)
                with partial.open("ab" if append else "wb") as output:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if not chunk:
                            continue
                        if written + len(chunk) > expected:
                            raise ValueError("Response exceeded pinned source size")
                        output.write(chunk)
                        written += len(chunk)
                        now = time.monotonic()
                        if now - last_log >= 20:
                            print(f"[progress] {row['path']}: {written / expected:.1%} ({written / 1e9:.3f} GB)", flush=True)
                            last_log = now
            if partial.stat().st_size == expected:
                break
        except requests.RequestException as exc:
            # Do not log redirect URLs, which can contain signed query strings.
            print(f"[retry {attempt + 1}/8] {row['path']}: {type(exc).__name__}", flush=True)
        if attempt == 7:
            raise RuntimeError(f"Download incomplete after retries: {row['path']}")
        time.sleep(min(2 ** attempt, 15))
    if partial.stat().st_size != expected:
        raise ValueError(f"Incorrect downloaded size: {row['path']}")
    digest = sha256_file(partial)
    if source_hash and digest != source_hash:
        raise ValueError(f"SHA256 mismatch; retaining partial for audit: {row['path']}")
    if row["kind"] == "annotation":
        with partial.open() as source:
            annotations = json.load(source)
        if not isinstance(annotations, list) or not annotations:
            raise ValueError("Expected a non-empty source annotation array")
    partial.replace(dest)
    print(f"[verified] {row['path']} sha256={digest}", flush=True)
    return {**row, "sha256": digest, "status": "verified"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/dflash_posttraining/raw/llava_video_178k"))
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--range-workers", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.range_workers < 1:
        parser.error("Worker counts must be positive")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(root, args.seed)
    print(f"[plan] {len(plan['files'])} files, {plan['total_bytes'] / 1e9:.2f} GB", flush=True)
    if args.plan_only:
        return
    required = sum(row["size"] for row in plan["files"] if not local_path(root, row["path"]).exists())
    if shutil.disk_usage(root).free < 2 * required + 10 * 1024 ** 3:
        raise RuntimeError("Insufficient free disk for bounded raw pool plus 10 GiB reserve")
    status = {"repo_id": REPO, "revision": REVISION, "status": "running", "files": [], "errors": []}
    write_json(root / "download_status.json", status)
    ordered = sorted(plan["files"], key=lambda row: row["kind"] == "archive")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for row in ordered:
            if args.range_workers > 1 and row["size"] >= 64 * 1024 * 1024:
                future = executor.submit(download_parallel, root, row, args.range_workers)
            else:
                future = executor.submit(download_file, root, row)
            futures[future] = row
        for future in as_completed(futures):
            row = futures[future]
            try:
                status["files"].append(future.result())
            except Exception as exc:
                status["errors"].append({"path": row["path"], "error_type": type(exc).__name__})
                print(f"[error] {row['path']}: {type(exc).__name__}", flush=True)
            write_json(root / "download_status.json", status)
    status["status"] = "complete" if not status["errors"] else "incomplete"
    write_json(root / "download_status.json", status)
    if status["errors"]:
        raise SystemExit("Some downloads failed; rerun the same command to resume")
    print(f"[complete] verified {len(status['files'])} raw files", flush=True)


if __name__ == "__main__":
    main()
