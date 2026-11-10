#!/usr/bin/env python3
"""Catalog downloaded raw archives and their original annotation references."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import tarfile

from download_posttraining_raw import local_path, write_json


def video_key(value: str) -> str:
    """Join the source's ytb_/v_ naming conventions without changing raw files."""
    stem = PurePosixPath(value).stem
    for prefix in ("ytb_", "v_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def catalog(root: Path, exclusion_metadata: Path | None = None) -> dict:
    plan = json.loads((root / "download_plan.json").read_text())
    status = json.loads((root / "download_status.json").read_text())
    if status["status"] != "complete" or status["errors"]:
        raise RuntimeError("Complete and verify all planned raw downloads first")
    expected = {row["path"] for row in plan["files"]}
    verified = {row["path"] for row in status["files"] if row["status"] == "verified"}
    if expected != verified:
        raise RuntimeError("Verified file set differs from download plan")
    excluded = set()
    if exclusion_metadata is not None:
        import pyarrow.parquet as pq

        excluded = {video_key(value) for value in
                    pq.read_table(exclusion_metadata, columns=["video_name"])["video_name"].to_pylist()}
    videos = []
    archives = [row for row in plan["files"] if row["kind"] == "archive"]
    group_counts = Counter()
    for number, row in enumerate(archives, 1):
        count = 0
        with tarfile.open(local_path(root, row["path"]), "r|gz") as archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("Unsafe archive member path")
                if not member.isfile() or member_path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
                    continue
                key = video_key(member.name)
                videos.append({"video_key": key, "group": row["group"],
                               "archive": row["path"], "member": member.name,
                               "bytes": member.size, "annotation_refs": [],
                               "excluded_vdc": key in excluded})
                count += 1
                group_counts[row["group"]] += 1
        print(f"[catalog] archive {number}/{len(archives)}: {count} videos ({row['group']})", flush=True)
    lookup = defaultdict(list)
    for video in videos:
        lookup[(video["group"], video["video_key"])].append(video)
    annotation_counts = {}
    for row in plan["files"]:
        if row["kind"] != "annotation":
            continue
        task = "caption" if "_cap_" in row["path"] else "open_ended"
        with local_path(root, row["path"]).open() as source:
            annotations = json.load(source)
        annotation_counts[row["path"]] = len(annotations)
        joined = 0
        for index, annotation in enumerate(annotations):
            key = video_key(annotation.get("video", str(annotation.get("id", ""))))
            for video in lookup.get((row["group"], key), []):
                video["annotation_refs"].append({"file": row["path"], "row_index": index,
                                                 "id": annotation.get("id"), "task": task})
                joined += 1
        print(f"[annotations] {task} {row['group']}: {joined} matching source rows", flush=True)
    index_path = root / "raw_video_inventory.jsonl"
    tmp = index_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as output:
        for video in videos:
            output.write(json.dumps(video, ensure_ascii=False) + "\n")
    tmp.replace(index_path)
    unique = {video["video_key"] for video in videos}
    eligible = {video["video_key"] for video in videos
                if not video["excluded_vdc"] and video["annotation_refs"]}
    tasks = Counter(ref["task"] for video in videos for ref in video["annotation_refs"])
    summary = {
        "stage": "raw_collection_only", "repo_id": plan["repo_id"],
        "revision": plan["revision"], "verified_source_files": len(verified),
        "download_bytes": plan["total_bytes"], "video_archive_entries": len(videos),
        "unique_video_keys": len(unique), "duplicate_video_keys": len(videos) - len(unique),
        "videos_by_duration_source_group": dict(group_counts),
        "matched_original_annotation_rows": dict(tasks),
        "entries_missing_annotations": sum(not video["annotation_refs"] for video in videos),
        "vdc_exclusion_metadata": str(exclusion_metadata) if exclusion_metadata else None,
        "vdc_exclusion_keys": len(excluded),
        "vdc_overlap_entries": sum(video["excluded_vdc"] for video in videos),
        "annotated_unique_videos_after_vdc_exclusion": len(eligible),
        "source_annotation_row_counts": annotation_counts,
        "videos_extracted": False, "target_responses_generated": False,
        "features_cached": False, "visual_budgets_calibrated": False,
        "train_validation_test_split_created": False,
        "checks": "Pinned source SHA256 and gzip/tar scan; no full video decode check.",
        "sampling_limit": "Random archive pool; duration buckets are source labels, not measured visual-token budgets.",
    }
    write_json(root / "collection_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/dflash_posttraining/raw/llava_video_178k"))
    parser.add_argument("--exclude-vdc-metadata", type=Path)
    args = parser.parse_args()
    catalog(args.raw_dir.resolve(), args.exclude_vdc_metadata)


if __name__ == "__main__":
    main()
