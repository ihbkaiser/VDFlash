#!/usr/bin/env python3
"""Upload the verified raw DFlash post-training pool to a private HF dataset repo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import time
from typing import Any

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "Tphuc15/dflash-posttraining-raw"
DEFAULT_ROOT = Path("data/dflash_posttraining/raw/llava_video_178k")
SOURCE_DATASET = "lmms-lab/LLaVA-Video-178K"
SOURCE_REVISION = "6d8c562dc26d70042a0d9704d1cae58c94b89098"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_collection(root: Path, should_wait: bool, poll_seconds: int) -> dict[str, Any]:
    status_path = root / "pipeline_status.json"
    while True:
        if not status_path.exists():
            raise FileNotFoundError(f"Missing collection status: {status_path}")
        status = json.loads(status_path.read_text())
        if status.get("status") == "complete" and status.get("stage") == "raw_collection_verified":
            return status
        if status.get("status") in {"incomplete", "failed"}:
            raise RuntimeError("Raw collection stopped before verification; resolve it and rerun upload")
        if not should_wait:
            raise RuntimeError("Raw collection is not complete; pass --wait to upload after completion")
        print(f"[wait] collection status={status.get('status')} stage={status.get('stage')}; checking again in {poll_seconds}s",
              flush=True)
        time.sleep(poll_seconds)


def validate_pool(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    plan = json.loads((root / "download_plan.json").read_text())
    downloads = json.loads((root / "download_status.json").read_text())
    summary = json.loads((root / "collection_summary.json").read_text())
    if plan.get("repo_id") != SOURCE_DATASET or plan.get("revision") != SOURCE_REVISION:
        raise ValueError("Download plan is not the pinned LLaVA-Video-178K source")
    if downloads.get("status") != "complete" or downloads.get("errors"):
        raise ValueError("Download status is not complete and error-free")

    planned = {row["path"]: row for row in plan["files"]}
    verified = {row["path"]: row for row in downloads["files"] if row.get("status") == "verified"}
    if set(planned) != set(verified):
        raise ValueError("Verified source file set differs from the download plan")

    payload_paths: list[str] = []
    for relative, source_row in planned.items():
        if source_row["kind"] not in {"archive", "annotation"}:
            continue
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe planned path: {relative}")
        local = root.joinpath(*relative_path.parts)
        if not local.is_file() or local.stat().st_size != source_row["size"]:
            raise ValueError(f"Missing or wrong-sized source file: {relative}")
        recorded_hash = verified[relative].get("sha256")
        if not recorded_hash or sha256_file(local) != recorded_hash:
            raise ValueError(f"Checksum changed after download verification: {relative}")
        source_hash = source_row.get("source_sha256")
        if source_hash and source_hash.removeprefix("sha256:") != recorded_hash:
            raise ValueError(f"Checksum differs from the pinned upstream file: {relative}")
        payload_paths.append(relative)

    required_metadata = (
        "README.md", "source_inventory.json", "download_plan.json", "download_status.json",
        "SHA256SUMS", "raw_video_inventory.jsonl", "collection_summary.json",
    )
    for relative in required_metadata:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"Missing catalog/provenance file: {relative}")
    if summary.get("verified_source_files") != len(planned):
        raise ValueError("Collection summary does not cover every planned source file")
    if summary.get("unique_video_keys", 0) <= 0:
        raise ValueError("Collection summary contains no video entries")
    return plan, downloads, summary, sorted(payload_paths)


def dataset_card(plan: dict[str, Any], summary: dict[str, Any]) -> str:
    total_gb = plan["total_bytes"] / 1_000_000_000
    group_rows = "\n".join(
        f"| `{group}` | {count} |"
        for group, count in sorted(summary.get("videos_by_duration_source_group", {}).items())
    )
    return f"""---
pretty_name: DFlash post-training raw video subset
language:
- en
task_categories:
- video-text-to-text
tags:
- video
- dflash
---

# DFlash post-training raw video subset

This is a bounded raw-data subset prepared for academic experiments on DFlash post-training. It contains the selected original video archives and original caption/open-ended QA annotation files from [{SOURCE_DATASET}](https://huggingface.co/datasets/{SOURCE_DATASET}), pinned to revision `{SOURCE_REVISION}`.

## Contents

- {len([row for row in plan['files'] if row['kind'] == 'archive'])} original video archives and {len([row for row in plan['files'] if row['kind'] == 'annotation'])} original annotation JSON files; total source payload is about {total_gb:.2f} GB.
- {summary['video_archive_entries']} video archive entries, {summary['unique_video_keys']} unique video IDs, and {summary['annotated_unique_videos_after_vdc_exclusion']} annotated unique IDs after excluding overlap with the local VDC test metadata.
- `raw_video_inventory.jsonl` maps each source archive member to its original annotation row references. `metadata/collection_summary.json` records the catalog counts.
- `metadata/SHA256SUMS` records the checksums of the downloaded source files. `source/README.md` preserves the upstream dataset card.

| Upstream duration group | Archive entries |
|---|---:|
{group_rows}

## Use and attribution

The upstream dataset card states that use is limited to academic research and education. This private copy does not broaden those terms. Review `source/README.md` and the [upstream dataset card](https://huggingface.co/datasets/{SOURCE_DATASET}) before use. Please cite the upstream dataset and its paper, [Video Instruction Tuning With Synthetic Data](https://arxiv.org/abs/2410.02713).

This repository contains raw source archives and annotations only. It does not contain generated target-model responses, model logits or hidden states, feature caches, decoded-video caches, calibrated visual-token budgets, or train/validation/test splits. Duration groups are source labels and are not calibrated visual-token budgets.

## Download from the training server

Grant the server's Hugging Face token read access to this private dataset, then use:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="{DEFAULT_REPO_ID}",
    repo_type="dataset",
    token=True,
)
```
"""


def upload(root: Path, repo_id: str) -> None:
    plan, downloads, summary, payload_paths = validate_pool(root)
    owner = repo_id.split("/", 1)[0]
    api = HfApi()
    identity = api.whoami()
    if identity.get("name", "").casefold() != owner.casefold():
        raise PermissionError("The active Hugging Face login does not match the requested account owner")

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not info.private:
        raise PermissionError("Destination dataset repo is public; refusing to upload the raw video pool")

    archive_count = sum(row["kind"] == "archive" for row in plan["files"])
    annotation_count = sum(row["kind"] == "annotation" for row in plan["files"])
    state: dict[str, Any] = {
        "status": "running", "stage": "uploading", "repo_id": repo_id,
        "source_repo_id": SOURCE_DATASET, "source_revision": SOURCE_REVISION,
        "source_files": len(plan["files"]), "uploaded_data_files_expected": len(payload_paths),
        "archive_files": archive_count, "annotation_files": annotation_count,
        "source_bytes": plan["total_bytes"], "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_path = root / "upload_status.json"
    write_json(state_path, state)
    try:
        print(f"[upload] {len(payload_paths)} checksum-verified files to private {repo_id}", flush=True)
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=root,
            allow_patterns=payload_paths,
            num_workers=4,
            print_report=True,
            print_report_every=60,
        )

        metadata = {
            "source/README.md": root / "README.md",
            "metadata/source_inventory.json": root / "source_inventory.json",
            "metadata/download_plan.json": root / "download_plan.json",
            "metadata/download_status.json": root / "download_status.json",
            "metadata/raw_video_inventory.jsonl": root / "raw_video_inventory.jsonl",
            "metadata/SHA256SUMS": root / "SHA256SUMS",
        }
        for path_in_repo, local_path in metadata.items():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {PurePosixPath(path_in_repo).name}",
            )

        sanitized_summary = dict(summary)
        sanitized_summary["vdc_exclusion_metadata"] = "local VDC test metadata (path omitted)"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as temp_summary:
            json.dump(sanitized_summary, temp_summary, indent=2, ensure_ascii=False)
            temp_summary.write("\n")
            temp_summary.flush()
            api.upload_file(
                path_or_fileobj=temp_summary.name,
                path_in_repo="metadata/collection_summary.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add raw collection summary",
            )

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md") as temp_card:
            temp_card.write(dataset_card(plan, summary))
            temp_card.flush()
            api.upload_file(
                path_or_fileobj=temp_card.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Document private DFlash raw subset",
            )

        expected_remote = set(payload_paths) | set(metadata) | {
            "metadata/collection_summary.json", "README.md",
        }
        remote_files = {
            entry.path for entry in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
            if getattr(entry, "path", None)
        }
        missing = expected_remote - remote_files
        if missing:
            raise RuntimeError(f"Upload returned without these remote files: {len(missing)} missing")

        state.update(
            status="complete", stage="uploaded_and_remote_files_verified",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            remote_files_verified=len(expected_remote),
            unique_video_keys=summary["unique_video_keys"],
            annotated_unique_videos_after_vdc_exclusion=summary["annotated_unique_videos_after_vdc_exclusion"],
        )
        write_json(state_path, state)
        api.upload_file(
            path_or_fileobj=str(state_path),
            path_in_repo="metadata/upload_status.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Record verified raw data upload",
        )
        print(f"[complete] verified {len(expected_remote)} remote files in https://huggingface.co/datasets/{repo_id}",
              flush=True)
    except Exception as error:
        state.update(
            status="incomplete", stage="upload_interrupted",
            updated_utc=datetime.now(timezone.utc).isoformat(),
            error_type=type(error).__name__,
        )
        write_json(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--wait", action="store_true", help="wait for the raw collection job to complete")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    root = args.raw_dir.resolve()
    wait_for_collection(root, args.wait, args.poll_seconds)
    upload(root, args.repo_id)


if __name__ == "__main__":
    main()
