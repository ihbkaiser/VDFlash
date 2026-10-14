"""Download VideoDetailCaption once and extract a deterministic small subset."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def prepare(output_dir: str, subset_size: int, seed: int = 42) -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = hf_hub_download(
        repo_id="lmms-lab/VideoDetailCaption",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir="/root/.cache/huggingface",
    )
    archive = hf_hub_download(
        repo_id="lmms-lab/VideoDetailCaption",
        filename="videos.zip",
        repo_type="dataset",
        cache_dir="/root/.cache/huggingface",
    )
    frame = pd.read_parquet(metadata)
    rows = frame.to_dict("records")
    if not rows:
        raise RuntimeError("VDC metadata is empty")
    count = min(int(subset_size), len(rows))
    # Deterministic index-spread selection.  The full duration-quantile
    # preparer remains available locally; this fast path avoids probing all
    # 499 videos before extracting a small remote subset.
    indices = sorted({round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)})
    selected = [dict(rows[index]) for index in indices]
    archive_index: dict[str, str] = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".mp4", ".mkv", ".mov", ".avi"}:
                continue
            key = Path(info.filename).stem.lower()
            if key not in archive_index or "test_videos" in info.filename.lower():
                archive_index[key] = info.filename
        video_dir = root / "Test_Videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        kept = []
        for position, row in enumerate(selected, start=1):
            video_name = str(row["video_name"])
            member = archive_index.get(Path(video_name).stem.lower())
            if member is None:
                print(f"[skip] video not found: {video_name}", flush=True)
                continue
            filename = Path(video_name).name
            if not Path(filename).suffix:
                filename += Path(member).suffix
            destination = video_dir / filename
            with zf.open(member, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            row["local_video_path"] = str(Path("Test_Videos") / filename)
            row["subset_rank"] = len(kept)
            kept.append(row)
            print(f"[extract] {position}/{len(selected)} {destination}", flush=True)
    if not kept:
        raise RuntimeError("none of the selected VDC videos were found in videos.zip")
    for filename in ("subset_manifest.jsonl", "test.jsonl"):
        with (root / filename).open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    (root / "selection_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": "lmms-lab/VideoDetailCaption",
                "subset_size_requested": int(subset_size),
                "subset_size_selected": len(kept),
                "selection": "deterministic index-spread subset",
                "seed": int(seed),
                "selected_video_names": [row["video_name"] for row in kept],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"VDC_SUBSET": {"count": len(kept), "output": str(root)}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.output_dir, args.subset_size, args.seed)


if __name__ == "__main__":
    main()
