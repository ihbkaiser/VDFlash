"""Processor-backed visual-token calibration.

The output is a measured calibration manifest, not an estimate. It can be
run without loading a 7B model, which makes it suitable as the first GPU/CPU
pipeline check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .dataset import VideoSample, choose_nearest_calibration, qwen2vl_video_token_count, write_jsonl
from .runtime import process_video


@dataclass(frozen=True)
class VideoCandidate:
    frames: int
    max_pixels: int

    @property
    def candidate_id(self) -> str:
        return f"frames={self.frames}:max_pixels={self.max_pixels}"


DEFAULT_FRAME_COUNTS = (4, 8, 16, 32, 64, 96, 128, 160)
DEFAULT_MAX_PIXELS = tuple(int(value * 28 * 28) for value in (256, 384, 512, 768, 1024))


def candidate_grid(
    frame_counts: Iterable[int] = DEFAULT_FRAME_COUNTS,
    max_pixels: Iterable[int] = DEFAULT_MAX_PIXELS,
) -> list[VideoCandidate]:
    return [VideoCandidate(int(frames), int(pixels)) for frames in frame_counts for pixels in max_pixels]


def calibrate_sample(
    sample: VideoSample,
    dataset_root: str | Path,
    processor: Any,
    targets: Iterable[int],
    tolerance: float,
    candidates: Iterable[VideoCandidate],
) -> list[dict[str, Any]]:
    measured: list[tuple[str, int]] = []
    settings: dict[str, dict[str, int]] = {}
    duration = max(float(sample.duration_sec or 1.0), 1e-3)
    for candidate in candidates:
        fps = candidate.frames / duration
        batch = process_video(
            processor,
            sample.resolved_path(dataset_root),
            sample.question,
            fps=fps,
            max_pixels=candidate.max_pixels,
        )
        grid = batch.get("video_grid_thw") if isinstance(batch, dict) else getattr(batch, "video_grid_thw", None)
        if grid is None:
            raise ValueError(f"processor returned no video_grid_thw for {sample.sample_id}")
        count = qwen2vl_video_token_count(grid.tolist())
        measured.append((candidate.candidate_id, count))
        settings[candidate.candidate_id] = {"frames": candidate.frames, "max_pixels": candidate.max_pixels}

    rows = []
    for target in targets:
        point = choose_nearest_calibration(sample.sample_id, int(target), measured, tolerance)
        rows.append({
            **asdict(point),
            "sample_fingerprint": sample.fingerprint(),
            "candidate_settings": settings.get(point.candidate_id),
            "video_path": str(sample.resolved_path(dataset_root)),
        })
    return rows


def write_calibration(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)
