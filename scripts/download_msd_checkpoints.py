#!/usr/bin/env python3
"""Download and verify MSD checkpoints from Hugging Face."""

import argparse
import hashlib
import os
import zipfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "ihbkaiser/qwen25vl-3b-msd-checkpoints"
REPO_TYPE = "dataset"
CHECKPOINTS = [
    {
        "filename": "qwen25vl-3b-msd-depth3-latest.zip",
        "expected_sha256": "f14da72e8e650732f4d1bb42b412612bcae637b593539a503a1a6ac9bf96b8ac",
        "extract_to": "qwen25vl-3b-msd-depth3-latest",
    },
    {
        "filename": "qwen25vl-3b-msd-depth5-latest.zip",
        "expected_sha256": "24ac7b4de1d0330007e4a4d662aebc026a546d384de4f811a151c208eb2a02fc",
        "extract_to": "qwen25vl-3b-msd-depth5-latest",
    },
]


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Download and verify MSD checkpoints.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/qwen25vl-3b-msd-checkpoints"),
        help="Target directory for downloaded checkpoints.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=True,
        help="Extract zip files after downloading and verifying.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    print(f"Target directory: {args.output_dir.resolve()}")
    print(f"Source repository: {REPO_ID} ({REPO_TYPE})")

    for item in CHECKPOINTS:
        fname = item["filename"]
        target_zip = args.output_dir / fname
        expected_sha = item["expected_sha256"]
        extract_dir = args.output_dir / item["extract_to"]

        print(f"\n==========================================")
        print(f"Processing: {fname}")
        print(f"Expected SHA256: {expected_sha}")
        print(f"==========================================")

        # Download if not exists or hash mismatch
        need_download = True
        if target_zip.exists():
            print(f"File {fname} already exists locally. Verifying SHA256...")
            current_sha = compute_sha256(target_zip)
            if current_sha == expected_sha:
                print(f"SHA256 matched: {current_sha}")
                need_download = False
            else:
                print(f"SHA256 mismatch ({current_sha} != {expected_sha}). Re-downloading...")

        if need_download:
            print(f"Downloading {fname} from Hugging Face...")
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=fname,
                local_dir=args.output_dir,
            )
            print(f"Downloaded to {downloaded_path}")

            print(f"Verifying SHA256 of downloaded file...")
            current_sha = compute_sha256(target_zip)
            if current_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for {fname}!\nExpected: {expected_sha}\nGot: {current_sha}"
                )
            print(f"SHA256 verified successfully: {current_sha}")

        if args.extract:
            print(f"Extracting {fname} to {extract_dir}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target_zip, "r") as zf:
                zf.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")
            extracted_files = list(extract_dir.rglob("*"))
            print(f"Total extracted items: {len(extracted_files)}")
            for p in extracted_files[:10]:
                if p.is_file():
                    print(f" - {p.relative_to(extract_dir)} ({p.stat().st_size:,} bytes)")
            if len(extracted_files) > 10:
                print(f" ... and {len(extracted_files) - 10} more items.")

    print("\nAll checkpoints downloaded, verified, and extracted successfully!")


if __name__ == "__main__":
    main()
