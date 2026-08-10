"""Run Qwen3.5 + DFlash on local/Hugging Face video-caption benchmarks.

This module deliberately contains only the benchmark adapter and runner.  The
Qwen3.5/DFlash decoding logic stays in ``qwen35_dflash_video_decode`` and can
therefore be reused by Video-MME and other datasets.

Example::

    python -m src.analyze.Whether_they_are_appliable_for_dDrafter.qwen35_dflash_benchmark \
      --dataset lmms-lab/VideoDetailCaption \
      --video-root /data/VideoDetailCaption \
      --output results/vdc/qwen35_dflash.jsonl \
      --limit 10 --num-frames 64

The output is prediction-oriented JSONL.  It includes the reference answer,
decoded text, losslessness hashes, acceptance metrics, and timing fields.  The
official VideoDetailDescription/VDC LLM judge can consume the prediction and
reference fields separately from this runner.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.analyze.Whether_they_are_appliable_for_dDrafter.qwen35_dflash_video_decode import (
    Qwen35DFlashDecoder,
    load_qwen35_dflash_models,
    normalize_video_inputs,
    sha256_tokens,
)


DEFAULT_PROMPT = (
    "Please provide a detailed description of the video, focusing on the main "
    "subjects, their actions, and the background scenes."
)

VDC_PROMPTS = {
    "detailed": (
        "Provide a faithfully detailed description of the video in more than "
        "three sentences."
    ),
    "short": "Write a one-sentence summary of the video.",
    "main_object": (
        "Describe the main subject, including its attributes, actions, "
        "positions, and movements throughout the video."
    ),
    "camera": (
        "Describe the camera work, including shot types, angles, camera "
        "movements, and changes in perspective."
    ),
    "background": (
        "Describe the background, including objects, location, weather, "
        "time of day, and dynamic elements."
    ),
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first(row: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _with_mp4(name: str) -> str:
    return name if Path(name).suffix else f"{name}.mp4"


def resolve_video_path(row: dict[str, Any], video_root: str | Path) -> Path:
    """Resolve common VideoDetailCaption/VDC video fields to a local file."""

    root = Path(video_root)
    candidates: list[Path] = []
    raw_path = _first(row, ("video_path", "path", "video_file"))
    if isinstance(raw_path, str):
        candidates.append(Path(raw_path))

    raw_name = _first(row, ("video_name", "video", "id"))
    if isinstance(raw_name, str):
        name = _with_mp4(raw_name)
        candidates.extend(
            [
                root / name,
                root / "Test_Videos" / name,
                root / "test_videos" / name,
            ]
        )

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix.lower() == ".mp4":
            expanded.extend(
                [candidate.with_suffix(".MP4"), candidate.with_suffix(".mkv")]
            )
    for candidate in expanded:
        if candidate.exists():
            return candidate
    shown = ", ".join(str(path) for path in expanded[:6]) or "<no path field>"
    raise FileNotFoundError(f"Video not found; tried: {shown}")


def sample_indices(total_frames: int, requested: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("video has no frames")
    count = min(int(requested), total_frames)
    if count < 1:
        raise ValueError("num_frames must be positive")
    return np.linspace(0, total_frames - 1, count).round().astype(int).tolist()


def _video_info(path: Path, requested_frames: int) -> tuple[dict[str, Any], list[int]]:
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        duration = 0.0
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / 1_000_000.0)
        total = int(stream.frames or 0)
        if total <= 0 and duration > 0:
            total = max(1, int(round(duration * fps)))
        return {
            "fps": fps,
            "duration_sec": duration,
            "total_frames": total,
            "width": int(stream.width),
            "height": int(stream.height),
        }, sample_indices(total, requested_frames)
    finally:
        container.close()


def extract_frames(
    path: Path, indices: list[int]
) -> tuple[list[np.ndarray], list[int]]:
    """Decode sampled frames and repair inaccurate container frame counts.

    Some MKV files report one more frame than PyAV can decode.  If a requested
    index is missing, resample against the actual decoded frame count instead
    of aborting the complete benchmark.
    """

    import av

    if not indices:
        raise ValueError("indices must not be empty")

    def decode_wanted(wanted: list[int]) -> tuple[dict[int, np.ndarray], int]:
        container = av.open(str(path))
        try:
            wanted_set = set(wanted)
            frames: dict[int, np.ndarray] = {}
            decoded_count = 0
            for index, frame in enumerate(container.decode(video=0)):
                decoded_count = index + 1
                if index in wanted_set:
                    frames[index] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted_set):
                    break
            return frames, decoded_count
        finally:
            container.close()

    selected, decoded_count = decode_wanted(indices)
    if len(selected) == len(indices):
        return [selected[index] for index in indices], indices

    if decoded_count <= 0:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    repaired_indices = sample_indices(decoded_count, len(indices))
    selected, decoded_count = decode_wanted(repaired_indices)
    if len(selected) != len(repaired_indices):
        raise RuntimeError(
            f"decoded {len(selected)} frames, requested {len(repaired_indices)} "
            f"from {path}"
        )
    return [selected[index] for index in repaired_indices], repaired_indices


def build_inputs(
    processor: Any,
    frames: list[np.ndarray],
    question: str,
    info: dict[str, Any],
    indices: list[int],
    *,
    video_max_pixels: Optional[int] = None,
) -> dict[str, Any]:
    from transformers.video_utils import VideoMetadata

    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=len(frames) / max(info["duration_sec"], 1e-6),
        width=info["width"],
        height=info["height"],
        duration=info["duration_sec"],
        frames_indices=list(indices),
    )
    processor_kwargs: dict[str, Any] = {
        "text": text,
        "videos": [frames],
        "video_metadata": [metadata],
        "return_tensors": "pt",
        "return_mm_token_type_ids": True,
    }
    if video_max_pixels is not None:
        processor_kwargs["size"] = {
            "shortest_edge": 4096,
            "longest_edge": int(video_max_pixels),
        }
    return normalize_video_inputs(processor(**processor_kwargs))


def row_to_prompt(row: dict[str, Any], task: str) -> str:
    question = _first(row, ("question", "prompt", "instruction"))
    if isinstance(question, str) and question.strip():
        return question.strip()
    if task in VDC_PROMPTS:
        return VDC_PROMPTS[task]
    return DEFAULT_PROMPT


def row_to_reference(row: dict[str, Any]) -> Optional[str]:
    value = _first(row, ("answer", "caption", "description", "target"))
    return str(value) if value is not None else None


def load_rows(dataset: str, split: str, *, limit: Optional[int], seed: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    data = load_dataset(dataset, split=split)
    if limit is not None:
        data = data.shuffle(seed=seed).select(range(min(limit, len(data))))
    return [dict(row) for row in data]


def load_manifest(path: str | Path, *, limit: Optional[int]) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]
    return rows


def decode_text(processor: Any, ids: list[int]) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def parse_visual_percentages(value: str) -> list[float]:
    """Parse and validate comma-separated visual percentages."""

    try:
        percentages = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("visual percentages must be numbers, e.g. 100,50,12.5,0") from exc
    if not percentages:
        raise ValueError("at least one visual percentage is required")
    if any(not np.isfinite(item) or item < 0.0 or item > 100.0 for item in percentages):
        raise ValueError("visual percentages must be between 0 and 100")
    return percentages


def _run_key(row: dict[str, Any], visual_percentage: float, config_hash: str) -> str:
    return json.dumps([row["sample_id"], visual_percentage, config_hash], sort_keys=True)


def run(args: argparse.Namespace) -> None:
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit("A GPU is required; use --allow-cpu only for smoke tests")

    visual_percentages = parse_visual_percentages(args.visual_percentages)
    rows = (
        load_manifest(args.manifest, limit=args.limit)
        if args.manifest
        else load_rows(args.dataset, args.split, limit=args.limit, seed=args.seed)
    )
    processor, target, draft = load_qwen35_dflash_models(
        target_model=args.target_model,
        draft_model=args.draft_model,
        device=device,
    )
    decoder = Qwen35DFlashDecoder(
        target,
        draft,
        device=device,
        visual_ratio=visual_percentages[0] / 100.0,
        verify_mode=args.verify_mode,
        block_size=args.block_size,
        stop_token_ids=args.stop_token_ids,
    )
    config_payload = {
        key: getattr(args, key)
        for key in (
            "dataset",
            "split",
            "task",
            "num_frames",
            "video_max_pixels",
            "target_model",
            "draft_model",
            "visual_percentages",
            "verify_mode",
            "block_size",
            "max_new_tokens",
            "temperature",
            "device",
            "seed",
            "stop_token_ids",
        )
    }
    config_hash = _sha256_text(json.dumps(config_payload, sort_keys=True, default=str))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if args.resume and output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("config_hash") == config_hash:
                    if (
                        record.get("status") == "ok"
                        and record.get("outputs_match") is True
                        and "visual_percentage" in record
                    ):
                        existing.add(
                            _run_key(
                                record,
                                float(record["visual_percentage"]),
                                config_hash,
                            )
                        )

    with output.open("a") as handle:
        for index, row in enumerate(rows):
            sample_id = str(_first(row, ("video_name", "video_id", "id"), index))
            row["sample_id"] = sample_id
            question = row_to_prompt(row, args.task)
            reference = row_to_reference(row)
            path: Optional[Path] = None
            info: dict[str, Any] = {}
            indices: list[int] = []
            frames: list[np.ndarray] = []
            preparation_error: Optional[str] = None
            try:
                path = resolve_video_path(row, args.video_root)
                info, indices = _video_info(path, args.num_frames)
                frames, indices = extract_frames(path, indices)
            except Exception as exc:
                preparation_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"[run] PREP FAILED {index + 1}/{len(rows)} "
                    f"{sample_id}: {preparation_error}"
                )
            for visual_percentage in visual_percentages:
                key = _run_key(row, visual_percentage, config_hash)
                if key in existing:
                    continue
                decoder.visual_ratio = visual_percentage / 100.0
                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "dataset": args.dataset,
                    "split": args.split,
                    "task": args.task,
                    "video_path": str(path) if path is not None else None,
                    "question": question,
                    "reference": reference,
                    "visual_percentage": visual_percentage,
                    "verify_mode": args.verify_mode,
                    "requested_frame_count": args.num_frames,
                    "video_max_pixels": args.video_max_pixels,
                    "actual_frame_count": len(frames),
                    "config_hash": config_hash,
                    "status": "error",
                    "error": None,
                }
                inputs = None
                result = None
                greedy_ids = None
                if preparation_error is not None:
                    record["error"] = preparation_error
                else:
                    try:
                        inputs = build_inputs(
                            processor,
                            frames,
                            question,
                            info,
                            indices,
                            video_max_pixels=args.video_max_pixels,
                        )
                        result = decoder.decode(
                            inputs,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                        greedy_ids, greedy_latency = decoder.greedy_reference(
                            inputs, max_new_tokens=args.max_new_tokens
                        )
                        spec_tokens = result.output_ids[0, result.num_input_tokens:].tolist()
                        greedy_tokens = greedy_ids[0].tolist()
                        record.update(
                            {
                                "prediction": decode_text(processor, spec_tokens),
                                "greedy_prediction": decode_text(processor, greedy_tokens),
                                "target_output_hash": sha256_tokens(greedy_tokens),
                                "speculative_output_hash": sha256_tokens(spec_tokens),
                                "outputs_match": spec_tokens == greedy_tokens,
                                "tau_proposal": result.tau_proposal(),
                                "tau_effective": result.tau_effective(),
                                "full_block_rate": result.full_block_rate(),
                                "acceptance_rounds": [r.as_dict() for r in result.acceptance_rounds],
                                "total_input_tokens": int(inputs["input_ids"].shape[1]),
                                "prefill_latency_s": result.prefill_latency_s,
                                "draft_latency_s": result.draft_latency_s,
                                "verify_latency_s": result.verify_latency_s,
                                "end_to_end_latency_s": result.end_to_end_latency_s,
                                "greedy_latency_s": greedy_latency,
                                "target_forward_calls": result.target_forward_calls,
                                "num_output_tokens": result.num_output_tokens,
                                "peak_memory_bytes": result.peak_memory_bytes,
                                "status": "ok" if spec_tokens == greedy_tokens else "mismatch",
                            }
                        )
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        print(f"[run] FAILED {index + 1}/{len(rows)} {sample_id}: {record['error']}")
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                # Release per-run tensors before the next visual percentage;
                # otherwise long-video activations can accumulate on a T4.
                inputs = None
                result = None
                greedy_ids = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            frames = []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5+DFlash on video caption datasets")
    parser.add_argument("--dataset", default="lmms-lab/VideoDetailCaption")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Use a prepared JSONL subset instead of loading/shuffling the full dataset",
    )
    parser.add_argument("--task", default="caption", choices=("caption", *VDC_PROMPTS))
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames sampled per video; 16 is the safe T4 default",
    )
    parser.add_argument(
        "--video-max-pixels",
        type=int,
        default=4_194_304,
        help="Maximum temporal-spatial video pixels passed to the processor; 4M is T4-safe",
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3.5-4B-DFlash")
    parser.add_argument(
        "--visual-percentages",
        default="100,50,12.5,0",
        help="Comma-separated fraction of visual positions retained by the draft",
    )
    parser.add_argument("--verify-mode", choices=("exact", "block"), default="exact")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-token-ids", type=int, nargs="*", default=[248044, 248046])
    parser.set_defaults(func=run)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
