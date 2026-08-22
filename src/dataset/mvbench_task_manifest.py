"""Build deterministic task-specific manifests for the MVBench video release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REQUESTED_TASKS = (
    "action_prediction",
    "action_sequence",
    "moving_attribute",
    "moving_direction",
    "object_interaction",
)

TASK_VIDEO_ROOTS = {
    "action_prediction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "moving_attribute": "clevrer/video_validation",
    "moving_direction": "clevrer/video_validation",
    "object_interaction": "star/Charades_segment",
}


def _safe_video_name(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"annotation has invalid video value: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"annotation has unsafe video path: {value!r}")
    return path


def build_task_records(
    task: str,
    annotations: Iterable[Mapping[str, object]],
    *,
    dataset_root: str | Path,
) -> list[dict[str, object]]:
    """Add stable task and local video-path fields to source annotations."""

    if task not in TASK_VIDEO_ROOTS:
        raise ValueError(f"unsupported MVBench task: {task}")

    root = Path(dataset_root)
    video_root = Path(TASK_VIDEO_ROOTS[task])
    records: list[dict[str, object]] = []
    for index, annotation in enumerate(annotations):
        video = _safe_video_name(annotation.get("video"))
        relative_path = video_root / video
        record = dict(annotation)
        record.update(
            {
                "task": task,
                "sample_id": f"{task}:{index:06d}",
                "video_root": video_root.as_posix(),
                "video_relpath": relative_path.as_posix(),
                "video_path": str((root / relative_path).resolve()),
                "video_exists": (root / relative_path).is_file(),
            }
        )
        records.append(record)
    return records


def _read_annotations(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_task_manifests(
    *,
    annotation_dir: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path,
    tasks: Sequence[str] = REQUESTED_TASKS,
    require_videos: bool = False,
) -> dict[str, int]:
    """Write one JSONL manifest per task and a deterministic combined manifest."""

    annotation_root = Path(annotation_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    combined: list[dict[str, object]] = []
    missing: list[str] = []
    for task in tasks:
        if task not in TASK_VIDEO_ROOTS:
            raise ValueError(f"unsupported MVBench task: {task}")
        records = build_task_records(
            task,
            _read_annotations(annotation_root / f"{task}.json"),
            dataset_root=dataset_root,
        )
        counts[task] = len(records)
        _write_jsonl(output_root / f"{task}.jsonl", records)
        combined.extend(records)
        missing.extend(
            str(record["video_path"])
            for record in records
            if not record["video_exists"]
        )

    if require_videos and missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} MVBench videos are missing; examples: {examples}"
        )

    _write_jsonl(output_root / "selected.jsonl", combined)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "tasks": list(tasks),
                "counts": counts,
                "total_records": len(combined),
                "missing_videos": len(missing),
                "dataset_root": str(Path(dataset_root).resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", default="dataset/MVBench/json")
    parser.add_argument("--dataset-root", default="dataset/MVBench")
    parser.add_argument("--output-dir", default="dataset/MVBench/classified")
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=REQUESTED_TASKS, default=REQUESTED_TASKS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    counts = write_task_manifests(
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        tasks=args.tasks,
        require_videos=args.require_videos,
    )
    for task, count in counts.items():
        print(f"{task}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
