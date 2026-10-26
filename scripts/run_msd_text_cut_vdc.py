#!/usr/bin/env python3
"""Run MSD 7B text-cut benchmark (zero visual tokens) on Qwen2-VL-7B.

In this experiment:
- Visual retention is 0%: all visual tokens are stripped from Drafter KV cache.
- Text retention is varied: 100%, 75%, 50%, 25%, 0% (min 1 token).
- Strategy: Suffix retention (left-to-right cutting, keeping the suffix text).
- Target model receives full context (visual + text) for AR baseline & verification.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    build_prompt_question,
    load_vdc_manifest,
    write_jsonl,
)
from src.analyze.Validate_Sparrow_hypothesises.run_msd import _target_reference, _hash_tokens
from src.analyze.Validate_Sparrow_hypothesises.runtime import (
    build_qwen2vl_video_processor,
    generate_msd_retention_video,
    load_msd_qwen2vl,
    move_batch_to_device,
    prepare_qwen2vl_prefill,
    process_video,
    PreparedPrefill,
    _hash_tensor,
)


def compact_zero_visual_text_prefill(
    prepared: PreparedPrefill,
    text_retention_pct: float,
) -> tuple[PreparedPrefill, int, int]:
    """Compacts prefill by stripping 100% visual tokens and retaining K% text suffix."""
    keep = torch.ones(prepared.input_ids.shape[1], dtype=torch.bool, device=prepared.input_ids.device)

    # 1. Strip all visual and marker tokens
    marker_ids = {
        prepared.vision_start_token_id,
        prepared.vision_end_token_id,
        prepared.video_token_id,
    }
    for m in marker_ids:
        if m is not None:
            keep &= (prepared.input_ids[0] != int(m))
    if prepared.video_positions.numel():
        keep[prepared.video_positions] = False

    # 2. Text tokens: cut from left to right (suffix retention)
    text_indices = torch.nonzero(keep).flatten()
    full_text_count = int(text_indices.numel())
    keep_count = max(1, round(full_text_count * text_retention_pct / 100.0))
    kept_indices = text_indices[-keep_count:]

    final_keep = torch.zeros_like(keep)
    final_keep[kept_indices] = True

    input_ids = prepared.input_ids[:, final_keep]
    inputs_embeds = prepared.inputs_embeds[:, final_keep, :]
    position_ids = prepared.position_ids[..., final_keep]
    attention_mask = prepared.attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, final_keep]

    compact_prefill = PreparedPrefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=prepared.rope_deltas,
        video_grid_thw=prepared.video_grid_thw,
        video_positions=torch.empty(0, dtype=torch.long, device=input_ids.device),
        input_fingerprint=_hash_tensor(input_ids),
        video_token_id=prepared.video_token_id,
        vision_start_token_id=prepared.vision_start_token_id,
        vision_end_token_id=prepared.vision_end_token_id,
        keep_mask=final_keep.detach().to(device=prepared.input_ids.device),
    )
    return compact_prefill, keep_count, full_text_count


def main():
    parser = argparse.ArgumentParser(description="Run MSD 7B text-cut with zero visual on VDC")
    parser.add_argument("--base-model", default="/path/to/local-resource")
    parser.add_argument("--msd-model", default="/path/to/local-resource")
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/test.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--output", default="results/text_cut_zero_visual/msd_7b/msd_text_cut_vdc.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retention-levels", type=str, default="100,75,50,25,0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=100352)
    parser.add_argument("--max-pixels", type=int, default=200704)
    args = parser.parse_args()

    text_retention_percentages = [float(x.strip()) for x in args.retention_levels.split(",") if x.strip()]
    dev = torch.device(args.device)

    print("=================================================================")
    print(" MSD 7B TEXT-CUT (ZERO VISUAL) BENCHMARK ON QWEN2-VL-7B")
    print("=================================================================")
    print(f"Device:           {args.device}")
    print(f"Base Model:       {args.base_model}")
    print(f"MSD Model:        {args.msd_model}")
    print(f"Text Retention:   {text_retention_percentages} %")
    print(f"Visual Retention: 0.0 % (Stripped completely)")
    print(f"Policy:           Suffix retention (Cut from left to right)")
    print(f"Manifest:         {args.manifest}")
    print(f"Output File:      {args.output}")
    print(f"Limit:            {args.limit}")
    print("=================================================================")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        existing_rows[r.get("row_id", "")] = r
                    except Exception:
                        pass
        print(f"Resume: Found {len(existing_rows)} completed rows in {out_path.name}")

    samples = load_vdc_manifest(args.manifest)
    if args.limit is not None:
        samples = samples[:args.limit]
    print(f"Loaded {len(samples)} samples from {args.manifest}")

    print(f"\nLoading MSD 7B on {args.device}...")
    processor = build_qwen2vl_video_processor(args.base_model, args.min_pixels, args.max_pixels)
    model = load_msd_qwen2vl(args.base_model, args.msd_model, device_map=args.device)
    base_model = getattr(model, "base_model", model)
    print(f"MSD loaded! Memory on {args.device}: {torch.cuda.memory_allocated(dev)/1e9:.2f} GB")

    for sample_idx, sample in enumerate(samples):
        sample_id = sample.sample_id

        all_done = True
        for ret in text_retention_percentages:
            row_id = f"{sample_id}:msd_7b:text_cut:retention-{int(ret)}"
            if row_id not in existing_rows:
                all_done = False
                break
        if all_done:
            continue

        print(f"\n[Sample {sample_idx + 1}/{len(samples)}] Processing {sample_id}...")

        try:
            prompt_question = build_prompt_question(sample, "natural")
            batch = process_video(
                processor,
                sample.resolved_path(args.dataset_root),
                prompt_question,
                args.fps,
                max_pixels=args.max_pixels,
                max_frames=args.max_frames,
            )
            batch = move_batch_to_device(batch, dev)
            prepared = prepare_qwen2vl_prefill(base_model, batch, dev)

            # 1. Target Autoregressive Greedy Baseline
            target_tokens, ar_prefill, ar_e2e = _target_reference(
                base_model, batch, prepared.input_ids.shape[1], args.max_new_tokens
            )
            ar_decode = max(0.0, ar_e2e - ar_prefill)
            target_text = processor.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            target_hash = _hash_tokens(target_tokens)

            print(f"  [AR Baseline] Output tokens: {len(target_tokens)} | Decode: {ar_decode:.3f}s | Text: {repr(target_text[:60])}...")

            # 2. Run Speculative Decoding for each text retention percentage
            for ret in text_retention_percentages:
                row_id = f"{sample_id}:msd_7b:text_cut:retention-{int(ret)}"
                if row_id in existing_rows:
                    continue

                draft, actual_text_tokens, full_text_tokens = compact_zero_visual_text_prefill(prepared, ret)

                spec_ids, trace = generate_msd_retention_video(
                    model,
                    prepared,
                    draft,
                    max_new_tokens=args.max_new_tokens,
                )

                spec_tokens = spec_ids[0, prepared.input_ids.shape[1]:].tolist()
                spec_text = processor.tokenizer.decode(spec_tokens, skip_special_tokens=True).strip()
                spec_hash = _hash_tokens(spec_tokens)

                min_len = min(len(spec_tokens), len(target_tokens))
                common_prefix = 0
                for i in range(min_len):
                    if spec_tokens[i] == target_tokens[i]:
                        common_prefix += 1
                    else:
                        break
                is_lossless = (common_prefix == len(target_tokens))

                decode_s = trace.get("decode_seconds", 0.0)
                e2e_s = trace.get("end_to_end_seconds", 0.0)
                decode_speedup = (ar_decode / decode_s) if decode_s > 0 else 1.0
                e2e_speedup = (ar_e2e / e2e_s) if e2e_s > 0 else 1.0

                acceptance_trace = trace.get("acceptance_trace", [])
                v_steps = len(acceptance_trace)
                tau_eff = (len(spec_tokens) / v_steps) if v_steps > 0 else 1.0

                print(
                    f"  [Text Ret {int(ret):3d}% ({actual_text_tokens}/{full_text_tokens} tok)] "
                    f"Decode speedup: {decode_speedup:.2f}x | E2E: {e2e_speedup:.2f}x | "
                    f"tau_eff: {tau_eff:.2f} | Lossless: {is_lossless} ({common_prefix}/{len(target_tokens)})"
                )

                row = {
                    "row_id": row_id,
                    "sample_id": sample_id,
                    "video_name": sample.video_name,
                    "dataset": "vdc",
                    "drafter": "msd_7b",
                    "selection_policy": "text_suffix_zero_visual",
                    "retention_percentage": ret,
                    "actual_visual_tokens": 0,
                    "full_visual_tokens": int(prepared.video_positions.numel()),
                    "actual_text_tokens": actual_text_tokens,
                    "full_text_tokens": full_text_tokens,
                    "lossless": is_lossless,
                    "lossless_prefix_length": common_prefix,
                    "target_num_tokens": len(target_tokens),
                    "spec_num_tokens": len(spec_tokens),
                    "decode_speedup": decode_speedup,
                    "end_to_end_speedup": e2e_speedup,
                    "tau_effective": tau_eff,
                    "prefill_seconds": trace.get("prefill_seconds", 0.0),
                    "decode_seconds": decode_s,
                    "verification_seconds": trace.get("verification_seconds", decode_s),
                    "end_to_end_seconds": e2e_s,
                    "ar_prefill_seconds": ar_prefill,
                    "ar_decode_seconds": ar_decode,
                    "ar_end_to_end_seconds": ar_e2e,
                    "verification_steps": v_steps,
                    "acceptance_trace": acceptance_trace,
                    "speculative_output_text": spec_text,
                    "speculative_output_hash": spec_hash,
                    "target_output_text": target_text,
                    "target_output_hash": target_hash,
                    "target_ground_truth_answer": getattr(sample, "answer", ""),
                }

                existing_rows[row_id] = row
                write_jsonl(out_path, list(existing_rows.values()))

        except Exception as e:
            print(f"  [ERROR] Failed sample {sample_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nBenchmark completed! Output saved to {out_path}")


if __name__ == "__main__":
    main()
