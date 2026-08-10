"""Dataset and visual-token calibration utilities.

Calibration never fabricates a model token count. A planned point is marked
``estimated`` until the Qwen processor has reported ``video_grid_thw``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class VideoSample:
    video_name: str
    question: str
    answer: str
    local_video_path: str
    duration_sec: float | None = None
    total_frames: int | None = None
    width: int | None = None
    height: int | None = None

    @property
    def sample_id(self) -> str:
        return self.video_name

    def resolved_path(self, dataset_root: str | Path) -> Path:
        path = Path(self.local_video_path)
        return path if path.is_absolute() else Path(dataset_root) / path

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class CalibrationPoint:
    sample_id: str
    target_visual_tokens: int
    actual_visual_tokens: int | None
    candidate_id: str | None
    status: str
    relative_error: float | None
    source: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(value)
    return rows


def load_vdc_manifest(path: str | Path, dataset_root: str | Path | None = None) -> list[VideoSample]:
    path = Path(path)
    root = Path(dataset_root) if dataset_root is not None else path.parent
    samples: list[VideoSample] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        required = ("video_name", "question", "answer", "local_video_path")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        sample = VideoSample(
            video_name=str(row["video_name"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            local_video_path=str(row["local_video_path"]),
            duration_sec=float(row["duration_sec"]) if row.get("duration_sec") is not None else None,
            total_frames=int(row["total_frames"]) if row.get("total_frames") is not None else None,
            width=int(row["width"]) if row.get("width") is not None else None,
            height=int(row["height"]) if row.get("height") is not None else None,
        )
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate video_name in manifest: {sample.sample_id}")
        if not sample.resolved_path(root).exists():
            raise FileNotFoundError(sample.resolved_path(root))
        seen.add(sample.sample_id)
        samples.append(sample)
    if not samples:
        raise ValueError(f"Manifest is empty: {path}")
    return samples


def choose_nearest_calibration(
    sample_id: str,
    target_visual_tokens: int,
    candidates: Sequence[tuple[str, int]],
    tolerance: float = 0.10,
) -> CalibrationPoint:
    """Choose the nearest *measured* processor candidate."""

    if target_visual_tokens <= 0:
        raise ValueError("target_visual_tokens must be positive")
    if not candidates:
        return CalibrationPoint(sample_id, target_visual_tokens, None, None, "missing", None, "processor")
    candidate_id, actual = min(candidates, key=lambda item: (abs(item[1] - target_visual_tokens), item[0]))
    relative_error = abs(actual - target_visual_tokens) / target_visual_tokens
    status = "ok" if relative_error <= tolerance else "out_of_tolerance"
    return CalibrationPoint(
        sample_id=sample_id,
        target_visual_tokens=target_visual_tokens,
        actual_visual_tokens=actual,
        candidate_id=candidate_id,
        status=status,
        relative_error=relative_error,
        source="processor",
    )


def planned_calibration(sample_id: str, targets: Iterable[int]) -> list[CalibrationPoint]:
    """Create explicit placeholders before model-backed calibration runs."""

    return [
        CalibrationPoint(sample_id, int(target), None, None, "pending", None, "not_measured")
        for target in targets
    ]


def qwen2vl_video_token_count(video_grid_thw: Sequence[Sequence[int]], merge_size: int = 2) -> int:
    """Count Qwen2-VL video placeholder tokens from processor grid metadata."""

    if merge_size <= 0:
        raise ValueError("merge_size must be positive")
    total = 0
    for row in video_grid_thw:
        if len(row) != 3:
            raise ValueError(f"video_grid_thw row must have three values: {row}")
        temporal, height, width = (int(value) for value in row)
        if min(temporal, height, width) <= 0:
            raise ValueError(f"video_grid_thw values must be positive: {row}")
        if height % merge_size or width % merge_size:
            raise ValueError(f"grid dimensions must be divisible by merge_size={merge_size}: {row}")
        total += temporal * (height // merge_size) * (width // merge_size)
    return total


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
