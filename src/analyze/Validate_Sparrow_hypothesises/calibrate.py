"""Processor-backed visual-token calibration.

The output is a measured calibration manifest, not an estimate. It can be
run without loading a 7B model, which makes it suitable as the first GPU/CPU
pipeline check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset import VideoSample, choose_nearest_calibration, qwen2vl_video_token_count, write_jsonl
from .runtime import process_video


@dataclass(frozen=True)
class VideoCandidate:
    frames: int
    max_pixels: int

    @property
    def candidate_id(self) -> str:
        return f"frames={self.frames}:max_pixels={self.max_pixels}"


# Qwen's visual patch size is 28.  The old grid started at four frames and
# was too sparse around the short-context point, which left many videos more
# than 10% away from the 400-token target.  Keep the grid explicit and
# measured: these values are only processor candidates, never token-count
# estimates.
DEFAULT_FRAME_COUNTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192)
DEFAULT_PIXEL_BUDGETS = (
    128, 160, 192, 208, 224, 240, 256, 288, 320, 352, 384, 448, 512, 640, 768, 1024
)
DEFAULT_MAX_PIXELS = tuple(int(value * 28 * 28) for value in DEFAULT_PIXEL_BUDGETS)


def candidate_grid(
    frame_counts: Iterable[int] = DEFAULT_FRAME_COUNTS,
    max_pixels: Iterable[int] = DEFAULT_MAX_PIXELS,
) -> list[VideoCandidate]:
    candidates: list[VideoCandidate] = []
    seen: set[tuple[int, int]] = set()
    for frames in frame_counts:
        for pixels in max_pixels:
            key = (int(frames), int(pixels))
            if key not in seen:
                candidates.append(VideoCandidate(*key))
                seen.add(key)
    return candidates


def adaptive_candidate_grid(
    frame_counts: Iterable[int] | None = None,
    max_pixels: Iterable[int] | None = None,
) -> list[VideoCandidate]:
    """Return the default adaptive processor grid.

    This named entry point makes the calibration policy testable and lets a
    caller replace either axis without accidentally changing the other one.
    Counts are subsequently taken from ``video_grid_thw`` by
    :func:`calibrate_sample`.
    """

    return candidate_grid(
        DEFAULT_FRAME_COUNTS if frame_counts is None else frame_counts,
        DEFAULT_MAX_PIXELS if max_pixels is None else max_pixels,
    )


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
            "row_id": f"{sample.sample_id}:{int(target)}",
            **asdict(point),
            "calibration_status": point.status,
            "sample_fingerprint": sample.fingerprint(),
            "candidate_settings": settings.get(point.candidate_id),
            "video_path": str(sample.resolved_path(dataset_root)),
        })
    return rows


@dataclass(frozen=True)
class PairedCohort:
    """The same sample IDs qualified at every requested calibration target."""

    target_visual_tokens: tuple[int, ...]
    sample_ids: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    missing_by_target: dict[int, tuple[str, ...]]
    invalid_by_target: dict[int, tuple[str, ...]]
    minimum_samples: int

    @property
    def valid(self) -> bool:
        return len(self.sample_ids) >= self.minimum_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "target_visual_tokens": list(self.target_visual_tokens),
            "sample_ids": list(self.sample_ids),
            "sample_count": len(self.sample_ids),
            "minimum_samples": self.minimum_samples,
            "paired_samples": len(self.sample_ids),
            "missing_by_target": {str(key): list(value) for key, value in self.missing_by_target.items()},
            "invalid_by_target": {str(key): list(value) for key, value in self.invalid_by_target.items()},
        }


def select_paired_cohort(
    rows: Iterable[Mapping[str, Any]],
    targets: Sequence[int],
    minimum_samples: int = 10,
) -> PairedCohort:
    """Select only rows with ``status=ok`` at every calibration milestone.

    Duplicate sample/target measurements are treated as invalid rather than
    silently selecting one.  This prevents a merged or retried calibration
    file from changing the cohort depending on line order.
    """

    target_values = tuple(int(value) for value in targets)
    if not target_values or minimum_samples <= 0:
        raise ValueError("targets must be non-empty and minimum_samples must be positive")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        sample_id = row.get("sample_id")
        target = row.get("target_visual_tokens")
        if sample_id is None or target is None:
            continue
        try:
            key = (str(sample_id), int(target))
        except (TypeError, ValueError):
            continue
        if key[1] in target_values:
            grouped.setdefault(key, []).append(row)

    by_target: dict[int, set[str]] = {target: set() for target in target_values}
    invalid_by_target: dict[int, set[str]] = {target: set() for target in target_values}
    selected_rows: dict[tuple[str, int], dict[str, Any]] = {}
    all_sample_ids = {sample for sample, _target in grouped}
    for sample_id in all_sample_ids:
        for target in target_values:
            values = grouped.get((sample_id, target), [])
            if len(values) != 1:
                if values:
                    invalid_by_target[target].add(sample_id)
                continue
            row = values[0]
            status = row.get("calibration_status", row.get("status"))
            if status != "ok":
                invalid_by_target[target].add(sample_id)
                continue
            actual = row.get("actual_visual_tokens")
            try:
                actual_value = int(actual)
            except (TypeError, ValueError):
                actual_value = 0
            if actual is None or actual_value <= 0:
                invalid_by_target[target].add(sample_id)
                continue
            by_target[target].add(sample_id)
            selected_rows[(sample_id, target)] = row

    paired = set.intersection(*(by_target[target] for target in target_values))
    paired_ids = tuple(sorted(paired))
    missing_by_target = {
        target: tuple(sorted(all_sample_ids - by_target[target]))
        for target in target_values
    }
    result_rows = tuple(
        selected_rows[(sample_id, target)]
        for sample_id in paired_ids
        for target in target_values
    )
    return PairedCohort(
        target_visual_tokens=target_values,
        sample_ids=paired_ids if len(paired_ids) >= minimum_samples else paired_ids,
        rows=result_rows,
        missing_by_target=missing_by_target,
        invalid_by_target={target: tuple(sorted(values)) for target, values in invalid_by_target.items()},
        minimum_samples=int(minimum_samples),
    )


@dataclass(frozen=True)
class HomogeneousCohort:
    """A paired cohort sharing one measured visual-token count per target."""

    target_visual_tokens: tuple[int, ...]
    actual_visual_tokens: tuple[int, ...]
    sample_ids: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    minimum_samples: int

    @property
    def valid(self) -> bool:
        return len(self.sample_ids) >= self.minimum_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "target_visual_tokens": list(self.target_visual_tokens),
            "actual_visual_tokens": list(self.actual_visual_tokens),
            "sample_ids": list(self.sample_ids),
            "sample_count": len(self.sample_ids),
            "minimum_samples": self.minimum_samples,
            "selection": "largest_exact_actual_visual_token_signature",
        }


def select_homogeneous_paired_cohort(
    rows: Iterable[Mapping[str, Any]],
    targets: Sequence[int],
    minimum_samples: int = 10,
) -> HomogeneousCohort:
    """Select the largest paired group with identical measured counts per target.

    This is intended for attention plots whose x-axis is an absolute token
    position.  Samples with different measured visual lengths otherwise have
    different modality boundaries at the same x coordinate.  Rows must be
    unique, calibrated ``ok`` records for each sample/target pair; invalid or
    duplicate pairs cannot enter a homogeneous signature.
    """
    target_values = tuple(int(value) for value in targets)
    if not target_values or minimum_samples <= 0:
        raise ValueError("targets must be non-empty and minimum_samples must be positive")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        sample_id = row.get("sample_id")
        target = row.get("calibration_target_visual_tokens", row.get("target_visual_tokens"))
        if sample_id is None or target is None:
            continue
        try:
            key = (str(sample_id), int(target))
        except (TypeError, ValueError):
            continue
        if key[1] in target_values:
            grouped.setdefault(key, []).append(row)

    signatures: dict[tuple[int, ...], list[tuple[str, dict[int, dict[str, Any]]]]] = {}
    sample_ids = sorted({sample_id for sample_id, _target in grouped})
    for sample_id in sample_ids:
        selected: dict[int, dict[str, Any]] = {}
        actual: list[int] = []
        valid = True
        for target in target_values:
            values = grouped.get((sample_id, target), [])
            if len(values) != 1:
                valid = False
                break
            row = values[0]
            if row.get("calibration_status", row.get("status")) != "ok":
                valid = False
                break
            try:
                count = int(row["actual_visual_tokens"])
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if count <= 0:
                valid = False
                break
            selected[target] = row
            actual.append(count)
        if valid:
            signatures.setdefault(tuple(actual), []).append((sample_id, selected))

    if not signatures:
        return HomogeneousCohort(target_values, (), (), (), int(minimum_samples))
    signature, members = sorted(
        signatures.items(),
        key=lambda item: (-len(item[1]), tuple(item[0])),
    )[0]
    selected_ids = tuple(sorted(sample_id for sample_id, _rows in members))
    selected_by_id = {sample_id: selected_rows for sample_id, selected_rows in members}
    result_rows = tuple(
        selected_by_id[sample_id][target]
        for sample_id in selected_ids
        for target in target_values
    )
    return HomogeneousCohort(
        target_visual_tokens=target_values,
        actual_visual_tokens=tuple(signature),
        sample_ids=selected_ids,
        rows=result_rows,
        minimum_samples=int(minimum_samples),
    )


def write_calibration(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def read_calibration(path: str | Path) -> list[dict[str, Any]]:
    """Read calibration JSONL strictly; a truncated line invalidates the file."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed calibration JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Calibration row at {path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Calibration file is empty: {path}")
    return rows


def audit_calibration(
    rows: Iterable[Mapping[str, Any]],
    targets: Sequence[int],
    minimum_samples: int = 10,
) -> dict[str, Any]:
    """Summarize strict calibration coverage for the final audit gate."""

    materialized = [dict(row) for row in rows]
    target_values = tuple(int(value) for value in targets)
    status_counts = {
        str(target): {
            "ok": sum(
                1 for row in materialized
                if _row_target(row) == target
                and row.get("calibration_status", row.get("status")) == "ok"
            ),
            "non_ok": sum(
                1 for row in materialized
                if _row_target(row) == target
                and row.get("calibration_status", row.get("status")) != "ok"
            ),
        }
        for target in target_values
    }
    cohort = select_paired_cohort(materialized, target_values, minimum_samples)
    return {
        "valid": cohort.valid,
        "targets": list(target_values),
        "status_counts": status_counts,
        "paired_samples": len(cohort.sample_ids),
        "minimum_samples": int(minimum_samples),
        "missing": [] if cohort.valid else [str(target) for target in target_values],
        "cohort": cohort.to_dict(),
    }


def _row_target(row: Mapping[str, Any]) -> int | None:
    try:
        value = row.get("target_visual_tokens")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
