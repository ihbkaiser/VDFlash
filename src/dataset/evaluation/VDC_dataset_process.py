"""Download, analyze, and select a representative VideoDetailCaption subset.

The source dataset contains metadata for 499 videos and a separate ``videos.zip``
archive.  This script downloads both through the Hugging Face cache, probes all
videos for duration, selects approximately 50 samples at evenly spaced duration
quantiles, and extracts only those selected videos into::

    dataset/VideoDetailCaption/
        Test_Videos/
        test.jsonl
        subset_manifest.jsonl
        duration_analysis.csv
        duration_analysis.json
        selection_summary.json

The selected directory layout is compatible with the existing benchmark runner
which looks for ``<video-root>/Test_Videos/<video_name>.mp4``.

python src/dataset/evaluation/VDC_dataset_process.py \
  --subset-size 50 \
  --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET_ID = "lmms-lab/VideoDetailCaption"
DEFAULT_METADATA_FILE = "data/test-00000-of-00001.parquet"
DEFAULT_VIDEO_ARCHIVE = "videos.zip"
DEFAULT_OUTPUT_DIR = Path("dataset/VideoDetailCaption")


def _download_source_file(
    dataset_id: str,
    filename: str,
    *,
    cache_dir: str | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return Path(path)


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    required = {"video_name", "question", "answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(frame.to_dict("records")):
        record = dict(row)
        record["source_row_index"] = index
        record["video_name"] = str(record["video_name"])
        rows.append(record)
    return rows


def _archive_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Index video members by stem, preferring Test_Videos entries."""

    index: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        suffix = Path(info.filename).suffix.lower()
        if suffix not in {".mp4", ".mkv", ".mov", ".avi"}:
            continue
        key = Path(info.filename).stem.lower()
        previous = index.get(key)
        if previous is None or "test_videos" in info.filename.lower():
            index[key] = info
    return index


def _probe_video(path: Path) -> dict[str, Any]:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / 1_000_000.0)

        total_frames = int(stream.frames or 0)
        if total_frames <= 0 and duration > 0:
            total_frames = int(round(duration * fps))
        if duration <= 0 and total_frames > 0 and fps > 0:
            duration = total_frames / fps

        return {
            "duration_sec": round(float(duration), 6),
            "fps": round(float(fps), 6),
            "total_frames": total_frames,
            "width": int(stream.width),
            "height": int(stream.height),
            "codec": str(stream.codec_context.name),
        }
    finally:
        container.close()


def _probe_archive(
    archive: zipfile.ZipFile,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Probe every metadata row through a temporary extracted video file."""

    index = _archive_index(archive)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vdc_probe_") as temp_dir:
        temp_root = Path(temp_dir)
        for position, row in enumerate(rows, start=1):
            video_name = str(row["video_name"])
            key = Path(video_name).stem.lower()
            info = index.get(key)
            base = {
                "source_row_index": row["source_row_index"],
                "video_name": video_name,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "archive_member": info.filename if info else None,
            }
            if info is None:
                failure = {**base, "reason": "video not found in archive"}
                failures.append(failure)
                print(f"[probe] {position}/{len(rows)} missing {video_name}")
                continue

            suffix = Path(info.filename).suffix or ".mp4"
            temp_path = temp_root / f"{position:05d}{suffix}"
            try:
                with archive.open(info, "r") as source, temp_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                probe = _probe_video(temp_path)
                if probe["duration_sec"] <= 0:
                    raise ValueError("non-positive duration")
                records.append({**base, **probe})
                print(
                    f"[probe] {position}/{len(rows)} {video_name} "
                    f"{probe['duration_sec']:.2f}s"
                )
            except Exception as exc:
                failures.append({**base, "reason": f"{type(exc).__name__}: {exc}"})
                print(f"[probe] {position}/{len(rows)} failed {video_name}: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)
    return records, failures


def _select_quantiles(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select rows at evenly spaced duration quantiles.

    Quantile positions make the subset cover the complete short-to-long range
    while preserving the empirical duration distribution.  Ties are resolved
    deterministically using ``seed``.
    """

    if not records:
        return []
    if count < 1:
        raise ValueError("subset size must be positive")
    ordered = sorted(records, key=lambda row: (float(row["duration_sec"]), row["video_name"]))
    target = min(count, len(ordered))
    if target == 1:
        selected = [ordered[len(ordered) // 2]]
    else:
        rng = random.Random(seed)
        selected_indices: set[int] = set()
        for quantile in np.linspace(0.0, 1.0, target):
            center = quantile * (len(ordered) - 1)
            candidates = [
                index
                for index in range(len(ordered))
                if index not in selected_indices
                and abs(index - center)
                == min(
                    abs(candidate - center)
                    for candidate in range(len(ordered))
                    if candidate not in selected_indices
                )
            ]
            selected_indices.add(rng.choice(candidates))
        selected = [ordered[index] for index in sorted(selected_indices)]

    for index, row in enumerate(selected):
        row["subset_rank"] = index
        row["duration_percentile"] = round(
            100.0 * ordered.index(row) / max(1, len(ordered) - 1), 4
        )
    return selected


def _summary(records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    durations = np.asarray([float(row["duration_sec"]) for row in records], dtype=float)
    if durations.size == 0:
        return {
            "count": 0,
            "failed_count": len(failures),
            "failure_reasons": failures,
        }
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "count": int(durations.size),
        "failed_count": len(failures),
        "min_sec": float(durations.min()),
        "max_sec": float(durations.max()),
        "mean_sec": float(durations.mean()),
        "median_sec": float(np.median(durations)),
        "std_sec": float(durations.std()),
        "percentiles_sec": {
            str(percentile): float(np.percentile(durations, percentile))
            for percentile in percentiles
        },
        "failure_reasons": failures,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _write_analysis_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("")
        return
    fields = [
        "source_row_index", "video_name", "duration_sec", "fps", "total_frames",
        "width", "height", "codec", "archive_member",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_duration_plot(
    path: Path,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_durations = [float(row["duration_sec"]) for row in records]
    selected_durations = [float(row["duration_sec"]) for row in selected]
    if not all_durations:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(all_durations, bins=20, alpha=0.7, label="all valid videos")
    axis.scatter(
        selected_durations,
        [0.0] * len(selected_durations),
        marker="|",
        s=220,
        linewidths=1.5,
        label="selected subset",
    )
    axis.set_xlabel("Video duration (seconds)")
    axis.set_ylabel("Number of videos")
    axis.set_title("VideoDetailCaption duration distribution and selected subset")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _extract_selected(
    archive: zipfile.ZipFile,
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    video_dir = output_dir / "Test_Videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for position, row in enumerate(selected, start=1):
        member = row["archive_member"]
        suffix = Path(member).suffix or ".mp4"
        filename = Path(row["video_name"]).name
        if not Path(filename).suffix:
            filename += suffix
        destination = video_dir / filename
        with archive.open(member, "r") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        row["local_video_path"] = str(Path("Test_Videos") / filename)
        print(f"[extract] {position}/{len(selected)} {destination}")


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] metadata: {args.dataset_id}/{args.metadata_file}")
    metadata_path = _download_source_file(
        args.dataset_id, args.metadata_file, cache_dir=args.cache_dir
    )
    print(f"[download] videos: {args.dataset_id}/{args.video_archive}")
    archive_path = _download_source_file(
        args.dataset_id, args.video_archive, cache_dir=args.cache_dir
    )
    shutil.copy2(metadata_path, output_dir / "source_metadata.parquet")

    rows = _load_metadata(metadata_path)
    print(f"[prepare] metadata rows: {len(rows)}")
    with zipfile.ZipFile(archive_path) as archive:
        records, failures = _probe_archive(archive, rows)
        selected = _select_quantiles(records, count=args.subset_size, seed=args.seed)
        _extract_selected(archive, selected, output_dir)

    _write_analysis_csv(output_dir / "duration_analysis.csv", records)
    _write_duration_plot(output_dir / "duration_distribution.png", records, selected)
    (output_dir / "duration_analysis.json").write_text(
        json.dumps(_summary(records, failures), indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "probe_failures.jsonl", failures)

    # Keep the original question/answer fields in the subset manifest.
    _write_jsonl(output_dir / "subset_manifest.jsonl", selected)
    _write_jsonl(output_dir / "test.jsonl", selected)
    (output_dir / "selection_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "subset_size_requested": args.subset_size,
                "subset_size_selected": len(selected),
                "seed": args.seed,
                "selection": "evenly spaced empirical duration quantiles",
                "selected_duration_sec": [row["duration_sec"] for row in selected],
                "selected_video_names": [row["video_name"] for row in selected],
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] selected {len(selected)} videos into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and select a duration-representative VideoDetailCaption subset"
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA_FILE)
    parser.add_argument("--video-archive", default=DEFAULT_VIDEO_ARCHIVE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=prepare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
