"""Measure a compact adaptive Qwen video-token calibration for VisualCut."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .calibrate import VideoCandidate, select_paired_cohort
from .dataset import (
    choose_nearest_calibration,
    load_vdc_manifest,
    qwen2vl_video_token_count,
)
from .runtime import process_video


TARGETS = (400, 3000, 13000, 25000)
CALIBRATION_VERSION = "v5_decord_matched_nframes_minmax_pixels_target_seed"
FRAME_CHOICES = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192)
PIXEL_PATCH_CHOICES = (
    128, 160, 192, 208, 224, 240, 256, 288, 320, 352, 384, 448, 512, 640, 768, 1024,
    1280, 1536, 1792, 2048, 2560, 3072, 4096,
)
INITIAL_SETTINGS = {
    400: (2, 208),
    3000: (6, 512),
    13000: (24, 512),
    25000: (48, 512),
}

_WORKER_PROCESSOR = None
_WORKER_ROOT: Path | None = None
_WORKER_OUTPUT: Path | None = None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _init_worker(target_model: str, video_root: str, output_dir: str) -> None:
    global _WORKER_PROCESSOR, _WORKER_ROOT, _WORKER_OUTPUT
    from .video_compat import configure_video_reader
    from transformers import AutoProcessor

    configure_video_reader("decord")
    _WORKER_PROCESSOR = AutoProcessor.from_pretrained(
        target_model,
        local_files_only=True,
        trust_remote_code=True,
    )
    _WORKER_ROOT = Path(video_root)
    _WORKER_OUTPUT = Path(output_dir)


def _candidate(frames: int, pixel_patches: int) -> VideoCandidate:
    frames = min(FRAME_CHOICES, key=lambda value: abs(value - int(frames)))
    pixel_patches = min(PIXEL_PATCH_CHOICES, key=lambda value: abs(value - int(pixel_patches)))
    return VideoCandidate(int(frames), int(pixel_patches) * 28 * 28)


def _measure(sample, candidate: VideoCandidate) -> int:
    assert _WORKER_PROCESSOR is not None and _WORKER_ROOT is not None
    batch = process_video(
        _WORKER_PROCESSOR,
        sample.resolved_path(_WORKER_ROOT),
        sample.question,
        nframes=candidate.frames,
        min_pixels=candidate.max_pixels,
        max_pixels=candidate.max_pixels,
    )
    grid = batch.get("video_grid_thw") if isinstance(batch, dict) else getattr(batch, "video_grid_thw", None)
    if grid is None:
        raise ValueError(f"processor returned no video_grid_thw for {sample.sample_id}")
    grid_values = grid.tolist() if hasattr(grid, "tolist") else grid
    return qwen2vl_video_token_count(grid_values)


def _adaptive_candidates(best: VideoCandidate, actual: int, target: int) -> list[VideoCandidate]:
    ratio = target / max(actual, 1)
    factor = math.sqrt(ratio)
    proposals = (
        (round(best.frames * ratio), best.max_pixels / (28 * 28)),
        (best.frames, best.max_pixels / (28 * 28) * ratio),
        (round(best.frames * factor), best.max_pixels / (28 * 28) * factor),
        (round(best.frames / factor), best.max_pixels / (28 * 28) / factor),
    )
    output: list[VideoCandidate] = []
    seen: set[str] = set()
    for frames, pixels in proposals:
        candidate = _candidate(max(1, frames), max(128, round(pixels)))
        if candidate.candidate_id not in seen:
            output.append(candidate)
            seen.add(candidate.candidate_id)
    return output


def _sample_fingerprint(sample) -> str:
    return sample.fingerprint()


def _calibrate_sample(sample) -> list[dict[str, Any]]:
    assert _WORKER_OUTPUT is not None
    fingerprint = _sample_fingerprint(sample)
    safe_id = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:20]
    sample_path = _WORKER_OUTPUT / ".samples" / f"{safe_id}.json"
    if sample_path.is_file():
        cached = json.loads(sample_path.read_text(encoding="utf-8"))
        if (
            cached.get("sample_fingerprint") == fingerprint
            and cached.get("calibration_version") == CALIBRATION_VERSION
        ):
            return list(cached["rows"])

    measured: dict[str, tuple[VideoCandidate, int]] = {}
    failed_measurements: list[dict[str, Any]] = []
    max_source_frames = int(sample.total_frames) if sample.total_frames is not None else FRAME_CHOICES[-1]
    max_source_frames -= max_source_frames % 2
    valid_frames = tuple(frame for frame in FRAME_CHOICES if frame <= max_source_frames)
    if not valid_frames:
        raise ValueError(f"video has fewer than two source frames: {sample.sample_id}")

    def make_candidate(frames: int, patches: int) -> VideoCandidate:
        normalized = _candidate(frames, patches)
        frame_count = min(valid_frames, key=lambda value: abs(value - normalized.frames))
        return VideoCandidate(frame_count, normalized.max_pixels)

    candidates = {
        make_candidate(*settings).candidate_id: make_candidate(*settings)
        for settings in INITIAL_SETTINGS.values()
    }
    for candidate_item in candidates.values():
        try:
            measured[candidate_item.candidate_id] = (
                candidate_item,
                _measure(sample, candidate_item),
            )
        except Exception as exc:
            failed_measurements.append(
                {
                    "candidate_id": candidate_item.candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    results: dict[int, Any] = {}
    for target in TARGETS:
        point = choose_nearest_calibration(
            sample.sample_id,
            target,
            [(key, value[1]) for key, value in measured.items()],
            tolerance=0.10,
        )
        attempts = 0
        while point.status != "ok" and measured and attempts < 4:
            best = measured[point.candidate_id][0]
            new_candidates = [
                make_candidate(item.frames, item.max_pixels // (28 * 28))
                for item in _adaptive_candidates(best, int(point.actual_visual_tokens or 1), target)
            ]
            new_candidates = [item for item in new_candidates if item.candidate_id not in measured]
            if not new_candidates:
                break
            for candidate_item in new_candidates:
                try:
                    measured[candidate_item.candidate_id] = (
                        candidate_item,
                        _measure(sample, candidate_item),
                    )
                except Exception as exc:
                    failed_measurements.append(
                        {
                            "candidate_id": candidate_item.candidate_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            point = choose_nearest_calibration(
                sample.sample_id,
                target,
                [(key, value[1]) for key, value in measured.items()],
                tolerance=0.10,
            )
            attempts += 1
        results[target] = point

    rows = []
    for target in TARGETS:
        point = results[target]
        selected = measured.get(point.candidate_id) if point.candidate_id else None
        rows.append(
            {
                "sample_id": sample.sample_id,
                "target_visual_tokens": target,
                "actual_visual_tokens": point.actual_visual_tokens,
                "candidate_id": point.candidate_id,
                "status": point.status,
                "calibration_status": point.status,
                "relative_error": point.relative_error,
                "source": "processor_video_grid_thw",
                "sample_fingerprint": fingerprint,
                "calibration_version": CALIBRATION_VERSION,
                "video_path": str(sample.resolved_path(_WORKER_ROOT)),
                "candidate_settings": (
                    {
                        "frames": selected[0].frames,
                        "min_pixels": selected[0].max_pixels,
                        "max_pixels": selected[0].max_pixels,
                    }
                    if selected
                    else None
                ),
                "candidate_measurements": [
                    {
                        "candidate_id": key,
                        "frames": value[0].frames,
                        "min_pixels": value[0].max_pixels,
                        "max_pixels": value[0].max_pixels,
                        "actual_visual_tokens": value[1],
                    }
                    for key, value in sorted(measured.items())
                ],
                "failed_measurements": failed_measurements,
            }
        )
    _atomic_json(
        sample_path,
        {
            "sample_fingerprint": fingerprint,
            "calibration_version": CALIBRATION_VERSION,
            "rows": rows,
        },
    )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_vdc_manifest(args.manifest, args.video_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    if args.sample_id:
        selected = set(args.sample_id)
        samples = [sample for sample in samples if sample.sample_id in selected]
    if not samples:
        raise ValueError("no VDC samples selected for calibration")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.target_model), str(args.video_root), str(output)),
    ) as pool:
        futures = {pool.submit(_calibrate_sample, sample): sample.sample_id for sample in samples}
        for index, future in enumerate(as_completed(futures), start=1):
            sample_id = futures[future]
            try:
                rows_by_sample[sample_id] = future.result()
                print(f"[calibration] {index}/{len(futures)} {sample_id} done", flush=True)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[calibration] {index}/{len(futures)} {sample_id} ERROR {failures[-1]['error']}", flush=True)

    calibration_rows = [
        row
        for sample in samples
        for row in rows_by_sample.get(sample.sample_id, [])
    ]
    calibration_path = output / "calibration.jsonl"
    _write_jsonl(calibration_path, calibration_rows)
    cohort = select_paired_cohort(calibration_rows, TARGETS, minimum_samples=args.minimum_samples)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    cohort_manifest = output / "cohort_manifest.jsonl"
    original_rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_id = {
        str(row.get("sample_id") or row.get("video_name") or row.get("id")): row
        for row in original_rows
    }
    _write_jsonl(
        cohort_manifest,
        [manifest_id[sample_id] for sample_id in cohort.sample_ids if sample_id in manifest_id],
    )
    cohort_report = cohort.to_dict()
    cohort_report.update(
        {
            "calibration_file": str(calibration_path),
            "cohort_manifest": str(cohort_manifest),
            "calibration_tolerance": 0.10,
            "candidate_failures": failures,
            "selected_video_fingerprints": {
                sample_id: sample_by_id[sample_id].fingerprint()
                for sample_id in cohort.sample_ids
                if sample_id in sample_by_id
            },
        }
    )
    _atomic_json(output / "cohort_selection.json", cohort_report)
    _atomic_json(output / "calibration_failures.json", {"failures": failures})
    if failures and not args.allow_partial:
        raise RuntimeError(
            f"{len(failures)} of {len(samples)} videos failed calibration; "
            "refusing to release a partial cohort"
        )
    if not cohort.valid:
        raise RuntimeError(
            f"only {len(cohort.sample_ids)} videos met all four measured budgets; "
            f"minimum is {args.minimum_samples}"
        )
    return cohort_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or args.minimum_samples <= 0:
        raise SystemExit("workers and minimum-samples must be positive")
    args.manifest = args.manifest.resolve()
    args.video_root = args.video_root.resolve()
    args.output_dir = args.output_dir.resolve()
    result = calibrate(args)
    print(
        f"[calibration-done] paired_samples={result['sample_count']} "
        f"file={result['calibration_file']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
