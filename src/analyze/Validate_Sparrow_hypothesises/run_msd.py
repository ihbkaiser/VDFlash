"""Run the full-input native-video MSD condition on calibrated VDC rows.

Draft-retention conditions are intentionally rejected by this command until
the compacted draft context is verified by the target/draft isolation gate.
This prevents a seemingly working but semantically wrong image-only shortcut
from producing Figure 1(b) evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from .dataset import load_vdc_manifest, write_jsonl
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    generate_msd_full_video,
    prepare_qwen2vl_prefill,
    process_video,
    load_msd_qwen2vl,
    require_cuda,
    validate_native_prefill_parity,
)


def _move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _new_tokens(output: torch.Tensor, input_length: int) -> list[int]:
    return output[0, input_length:].detach().to("cpu").tolist()


def _hash_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:16]


def _target_reference(base_model: Any, batch: Any, input_length: int, max_new_tokens: int) -> tuple[list[int], float, float]:
    start = time.perf_counter()
    with torch.inference_mode():
        base_model(**batch, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with torch.inference_mode():
        output = base_model.generate(
            **batch,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_to_end_seconds = time.perf_counter() - start
    return _new_tokens(output, input_length), prefill_seconds, end_to_end_seconds


def run(args: argparse.Namespace) -> int:
    try:
        require_cuda()
    except RuntimeUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    if args.condition != "full":
        raise SystemExit(
            "Only --condition full is enabled. Draft-retention requires the compacted-draft isolation implementation and is rejected explicitly."
        )
    manifest_path = Path(args.manifest)
    samples = load_vdc_manifest(manifest_path, args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    processor = build_qwen2vl_video_processor(args.base_model, args.min_pixels, args.max_pixels)
    model = load_msd_qwen2vl(args.base_model, args.msd_model)
    base_model = model.base_model
    device = next(base_model.parameters()).device
    rows = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}")
        batch = process_video(processor, sample.resolved_path(args.dataset_root), sample.question, args.fps)
        batch = _move_batch(batch, device)
        prepared = prepare_qwen2vl_prefill(base_model, batch, device)
        parity = validate_native_prefill_parity(base_model, batch, device)
        if not parity["valid"]:
            raise RuntimeError(f"native prefill parity failed for {sample.sample_id}: {parity}")
        target_tokens, ar_prefill, ar_e2e = _target_reference(base_model, batch, prepared.input_ids.shape[1], args.max_new_tokens)
        speculative_ids, trace = generate_msd_full_video(model, prepared, args.max_new_tokens)
        speculative_tokens = _new_tokens(speculative_ids, prepared.input_ids.shape[1])
        row = {
            "row_id": f"{sample.sample_id}:full",
            "paper_figure": "Figure 1(a)",
            "sample_id": sample.sample_id,
            "target_model": args.base_model,
            "temperature": 0.0,
            "target_visual_tokens": int(prepared.video_positions.numel()),
            "actual_visual_tokens": int(prepared.video_positions.numel()),
            "full_target_visual_tokens": int(prepared.video_positions.numel()),
            "target_input_fingerprint": prepared.input_fingerprint,
            "target_input_fingerprint_reference": prepared.input_fingerprint,
            "draft_input_fingerprint": prepared.input_fingerprint,
            "target_output_ids": target_tokens,
            "speculative_output_ids": speculative_tokens,
            "lossless": target_tokens == speculative_tokens,
            "target_output_hash": _hash_tokens(target_tokens),
            "speculative_output_hash": _hash_tokens(speculative_tokens),
            "accepted_prefix_tokens": trace["accepted_prefix_tokens"],
            "acceptance_trace": trace["acceptance_trace"],
            "verification_steps": trace["verification_steps"],
            "prefill_seconds": None,
            "decode_seconds": None,
            "end_to_end_seconds": trace["decode_seconds"],
            "ar_prefill_seconds": ar_prefill,
            "ar_end_to_end_seconds": ar_e2e,
            "video_token_count": trace["video_token_count"],
            "native_prefill_parity": parity,
            "condition": "full",
        }
        rows.append(row)
        if not row["lossless"]:
            raise RuntimeError(f"losslessness failed for {sample.sample_id}")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} lossless MSD rows to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-model", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--output", default="results/sparrow_validation/msd_full.jsonl")
    parser.add_argument("--condition", choices=("full", "retention"), default="full")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
