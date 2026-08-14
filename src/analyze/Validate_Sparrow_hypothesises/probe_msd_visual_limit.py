"""Probe the largest MSD visual context that fits on one GPU.

Each invocation loads one model instance and tests increasing processor
settings.  It stops at the first CUDA OOM so the caller can run it once per
GPU in isolated processes.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .dataset import load_vdc_manifest
from .run_msd import _target_reference
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    generate_msd_full_video,
    model_device,
    move_batch_to_device,
    prepare_qwen2vl_prefill,
    process_video,
    load_msd_qwen2vl,
    require_cuda,
)


def _parse_levels(value: str) -> list[tuple[int, int, int]]:
    levels = []
    for item in value.split(","):
        label, frames, pixels = (int(part) for part in item.split(":"))
        levels.append((label, frames, pixels))
    if not levels:
        raise ValueError("at least one probe level is required")
    return levels


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run(args: argparse.Namespace) -> int:
    try:
        require_cuda()
    except RuntimeUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    samples = load_vdc_manifest(args.manifest, args.dataset_root)
    matching = [sample for sample in samples if sample.sample_id == args.sample_id]
    if not matching:
        raise SystemExit(f"sample_id not found in manifest: {args.sample_id}")
    sample = matching[0]
    levels = _parse_levels(args.levels)
    processor = build_qwen2vl_video_processor(
        args.base_model, args.min_pixels, args.processor_max_pixels
    )
    model = load_msd_qwen2vl(
        args.base_model,
        args.msd_model,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )
    base_model = model.base_model
    device = model_device(base_model)
    if args.device_map in {"auto", "model_parallel"}:
        print(f"model device map: {getattr(base_model, 'hf_device_map', 'unavailable')}", flush=True)
    rows: list[dict[str, object]] = []

    for requested, frames, max_pixels in levels:
        fps = frames / max(float(sample.duration_sec or 1.0), 1e-3)
        print(
            f"probe requested={requested} frames={frames} max_pixels={max_pixels}",
            flush=True,
        )
        started = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            batch = process_video(
                processor,
                sample.resolved_path(args.dataset_root),
                sample.question,
                fps=fps,
                max_pixels=max_pixels,
            )
            batch = move_batch_to_device(batch, device)
            prepared = prepare_qwen2vl_prefill(base_model, batch, device)
            actual = int(prepared.video_positions.numel())
            print(f"  prepared actual={actual} input_length={prepared.input_ids.shape[1]}", flush=True)
            if args.skip_target_reference:
                target_tokens, _ar_prefill, _ar_e2e = [], 0.0, 0.0
            else:
                target_tokens, _ar_prefill, _ar_e2e = _target_reference(
                    base_model,
                    batch,
                    prepared.input_ids.shape[1],
                    args.max_new_tokens,
                )
            speculative_output, trace = generate_msd_full_video(
                model, prepared, args.max_new_tokens
            )
            peak = int(torch.cuda.max_memory_allocated(device))
            row = {
                "status": "ok",
                "requested_visual_tokens": requested,
                "actual_visual_tokens": actual,
                "frames": frames,
                "max_pixels": max_pixels,
                "fps": fps,
                "peak_memory_allocated_bytes": peak,
                "peak_memory_allocated_gib": peak / (1024**3),
                "elapsed_seconds": time.perf_counter() - started,
                "decode_seconds": trace["decode_seconds"],
                "target_output_tokens": len(target_tokens),
            }
            if target_tokens:
                speculative_tokens = speculative_output[0, prepared.input_ids.shape[1]:].detach().cpu().tolist()
                common = 0
                for target_token, speculative_token in zip(target_tokens, speculative_tokens):
                    if target_token != speculative_token:
                        break
                    common += 1
                row.update({
                    "speculative_output_tokens": len(speculative_tokens),
                    "lossless": common == len(target_tokens),
                    "lossless_prefix_length": common,
                })
            rows.append(row)
            print(
                f"  OK actual={actual} peak={peak / (1024**3):.2f} GiB "
                f"elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )
            del batch, prepared, target_tokens, speculative_output, trace
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - probe must record the boundary
            peak = int(torch.cuda.max_memory_allocated(device))
            status = "oom" if _is_oom(exc) else "error"
            row = {
                "status": status,
                "requested_visual_tokens": requested,
                "frames": frames,
                "max_pixels": max_pixels,
                "fps": fps,
                "peak_memory_allocated_bytes": peak,
                "peak_memory_allocated_gib": peak / (1024**3),
                "elapsed_seconds": time.perf_counter() - started,
                "error": str(exc),
            }
            rows.append(row)
            print(
                f"  {status.upper()} peak={peak / (1024**3):.2f} GiB: {exc}",
                flush=True,
            )
            if _is_oom(exc):
                break
            gc.collect()
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "sample_id": sample.sample_id,
                "gpu": torch.cuda.get_device_name(device),
                "device_index": torch.cuda.current_device(),
                "base_model": args.base_model,
                "msd_model": args.msd_model,
                "device_map": getattr(base_model, "hf_device_map", args.device_map),
                "results": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote probe results: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--sample-id", default="v_zxr6UZKPDh4")
    parser.add_argument(
        "--levels",
        default=(
            "400:4:200704,3000:8:401408,6000:16:401408,"
            "8000:16:602112,12000:32:602112,16000:64:602112,"
            "20000:64:802816,25000:96:802816"
        ),
        help="comma-separated requested:frames:max_pixels levels",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-model", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="cuda")
    parser.add_argument("--max-memory", help="e.g. 0:22GiB,1:14GiB for --device-map auto")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--processor-max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-target-reference", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
