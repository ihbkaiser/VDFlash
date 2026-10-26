#!/usr/bin/env python3
"""Run DFlash speculative decoding text-cut benchmark (zero visual tokens) on Qwen2.5-VL-3B.

In this experiment:
- Visual retention is 0%: all visual tokens are stripped from Drafter KV cache.
- Text retention is varied: 100%, 75%, 50%, 25%, 0% (min 1 token).
- Strategy: Suffix retention (left-to-right cutting, keeping the suffix text).
- Target model receives full context (visual + text) for AR baseline & verification.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

repo_root = Path(__file__).resolve().parents[1]
specforge_root = repo_root / "src" / "train_Dflash_SpecForge"
for p in (repo_root, specforge_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from src.analyze.Validate_Sparrow_hypothesises.dflash_runtime import find_visual_positions
from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    _compute_position_ids,
    _eos_token_ids,
    _load_draft,
    _target_greedy,
)


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:16]


def load_dataset_records(manifest_path: str | Path, dataset_root: str | Path) -> list[dict[str, Any]]:
    manifest_file = Path(manifest_path)
    records = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = item.get("source_id") or item.get("video_name") or item.get("sample_id")
            video_rel = item.get("local_video_path") or item.get("video_path") or f"{item.get('video_name')}.mp4"
            video_full = Path(dataset_root) / video_rel
            if not video_full.exists():
                alt_paths = [
                    Path(dataset_root) / "Test_Videos" / f"{item.get('video_name')}.mp4",
                    Path(dataset_root) / f"{item.get('video_name')}.mp4",
                ]
                for alt in alt_paths:
                    if alt.exists():
                        video_full = alt
                        break

            question = item.get("question") or item.get("prompt") or "Describe this video in detail."
            records.append({
                "sample_id": sample_id,
                "video_name": item.get("video_name") or sample_id,
                "video_path": str(video_full) if video_full.exists() else None,
                "question": question,
                "raw_metadata": item,
            })
    return records


class PromptWrapper:
    def __init__(self, inputs: dict[str, Any], position_ids: torch.Tensor, target_kwargs: dict[str, Any]):
        self.inputs = inputs
        self.position_ids = position_ids
        self.target_kwargs = target_kwargs


def main():
    parser = argparse.ArgumentParser(description="Run DFlash text-cut with zero visual on VDC")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", default="dataset/qwen25vl-3b-depth1-depth3/qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt")
    parser.add_argument("--draft-config", default="src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash-h32-depth3.json")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/test.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--output-dir", type=Path, default=Path("results/text_cut_zero_visual/dflash"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retention-levels", type=str, default="100,75,50,25,0")
    args = parser.parse_args()

    text_retention_percentages = [float(x.strip()) for x in args.retention_levels.split(",") if x.strip()]
    dev = torch.device(args.device)

    print("=================================================================")
    print(" DFLASH TEXT-CUT (ZERO VISUAL) BENCHMARK ON QWEN2.5-VL-3B")
    print("=================================================================")
    print(f"Device:           {args.device}")
    print(f"Target Model:     {args.target_model}")
    print(f"Draft Checkpoint: {args.checkpoint}")
    print(f"Text Retention:   {text_retention_percentages} %")
    print(f"Visual Retention: 0.0 % (Stripped completely)")
    print(f"Policy:           Suffix retention (Cut from left to right)")
    print(f"Manifest:         {args.manifest}")
    print(f"Output Dir:       {args.output_dir}")
    print(f"Limit:            {args.limit}")
    print("=================================================================")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output = args.output_dir / "dflash_text_cut_vdc.jsonl"
    csv_output = args.output_dir / "draft_iterations_text_cut_vdc.csv"

    # Resume handling
    existing_rows: set[str] = set()
    if jsonl_output.exists():
        with open(jsonl_output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        existing_rows.add(row.get("row_id", ""))
                    except Exception:
                        pass
        print(f"Resume: Found {len(existing_rows)} completed rows in {jsonl_output.name}")

    records = load_dataset_records(args.manifest, args.dataset_root)
    if args.limit is not None:
        records = records[:args.limit]
    print(f"Loaded {len(records)} samples from {args.manifest}")

    print(f"\nLoading Target Model on {args.device}...")
    target = AutoModelForImageTextToText.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map={"model": args.device, "visual": args.device, "lm_head": args.device},
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.target_model)

    print(f"Loading Draft Model on {args.device}...")
    draft, load_s = _load_draft(
        args.checkpoint,
        args.draft_config,
        device=dev,
        dtype=torch.bfloat16,
    )
    print(f"Draft loaded in {load_s:.2f}s! Memory on {args.device}: {torch.cuda.memory_allocated(dev)/1e9:.2f} GB")

    decoder = InstrumentedDFlashDecoder(
        target,
        draft,
        device=dev,
        draft_device=dev,
    )
    stop_token_ids = _eos_token_ids(processor, target)

    if not csv_output.exists():
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "text_retention", "round_idx", "accepted_proposals",
                "effective_emitted", "is_terminal"
            ])

    for sample_idx, sample in enumerate(records):
        sample_id = sample["sample_id"]
        video_path = sample["video_path"]
        question = sample["question"]

        all_done = True
        for ret in text_retention_percentages:
            row_id = f"{sample_id}:dflash:depth{args.depth}:text_cut:retention-{int(ret)}"
            if row_id not in existing_rows:
                all_done = False
                break
        if all_done:
            continue

        print(f"\n[Sample {sample_idx + 1}/{len(records)}] Processing {sample_id}...")

        messages = [
            {
                "role": "user",
                "content": (
                    [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 256 * 28 * 28,
                            "fps": 1.0,
                            "max_frames": 16,
                        },
                        {"type": "text", "text": question},
                    ]
                    if video_path
                    else [{"type": "text", "text": question}]
                ),
            }
        ]

        try:
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            proc_kwargs = {"text": [rendered], "padding": True, "return_tensors": "pt"}
            if image_inputs:
                proc_kwargs["images"] = image_inputs
            if video_inputs:
                proc_kwargs["videos"] = video_inputs
                proc_kwargs.update(dict(video_kwargs or {}))

            batch = processor(**proc_kwargs)
            batch = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
            pos_ids = _compute_position_ids(target, batch)
            target_kwargs = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}

            prompt_len = int(batch["input_ids"].shape[1])
            visual_positions = find_visual_positions(batch["input_ids"], processor=processor)

            # 1. Target Autoregressive Baseline
            prompt_obj = PromptWrapper(batch, pos_ids, target_kwargs)
            target_output, target_timing = _target_greedy(
                target,
                prompt_obj,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=stop_token_ids,
                device=dev,
            )
            target_tokens = target_output[0, prompt_len:].detach().cpu().tolist()
            target_text = processor.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            target_hash = sha256_tokens(target_tokens)
            ar_prefill_s = target_timing["prefill_s"]
            ar_decode_s = target_timing["decode_s"]
            ar_e2e_s = target_timing["end_to_end_s"]

            print(f"  [AR Baseline] Output tokens: {len(target_tokens)} | Decode: {ar_decode_s:.3f}s | Text: {repr(target_text[:60])}...")

            # Base mask with ZERO visual: exclude visual tokens and delimiter markers
            base_text_mask = torch.ones(prompt_len, dtype=torch.bool, device=dev)
            for p_idx in visual_positions:
                base_text_mask[p_idx] = False
            for m in [
                getattr(target.config, "vision_start_token_id", None),
                getattr(target.config, "vision_end_token_id", None),
                getattr(target.config, "video_token_id", None),
            ]:
                if m is not None:
                    base_text_mask[batch["input_ids"][0] == int(m)] = False

            text_positions = [i for i in range(prompt_len) if base_text_mask[i].item() is True]
            full_text_count = len(text_positions)

            # 2. Run DFlash for each text retention percentage
            for ret in text_retention_percentages:
                row_id = f"{sample_id}:dflash:depth{args.depth}:text_cut:retention-{int(ret)}"
                if row_id in existing_rows:
                    continue

                keep_count = max(1, round(full_text_count * ret / 100.0))
                kept_positions = set(text_positions[-keep_count:])
                actual_text_tokens = len(kept_positions)

                keep_mask = torch.zeros(prompt_len, dtype=torch.bool, device=dev)
                for p_idx in kept_positions:
                    keep_mask[p_idx] = True

                spec_res = decoder.decode(
                    input_ids=batch["input_ids"],
                    position_ids=pos_ids,
                    target_kwargs=target_kwargs,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_token_ids,
                    prefill_target_context_keep_mask=keep_mask,
                )
                spec_tokens = spec_res.output_ids[0, prompt_len:].detach().cpu().tolist()
                spec_text = processor.tokenizer.decode(spec_tokens, skip_special_tokens=True).strip()
                spec_hash = sha256_tokens(spec_tokens)

                # Losslessness check
                min_len = min(len(spec_tokens), len(target_tokens))
                common_prefix = 0
                for i in range(min_len):
                    if spec_tokens[i] == target_tokens[i]:
                        common_prefix += 1
                    else:
                        break
                is_lossless = (common_prefix == len(target_tokens))

                decode_speedup = (ar_decode_s / spec_res.decode_s) if spec_res.decode_s > 0 else 1.0
                e2e_speedup = (ar_e2e_s / spec_res.end_to_end_s) if spec_res.end_to_end_s > 0 else 1.0
                tau_eff = spec_res.tau_effective or 1.0
                tau_prop = spec_res.tau_proposal or 0.0

                print(
                    f"  [Text Ret {int(ret):3d}% ({actual_text_tokens}/{full_text_count} tok)] "
                    f"Decode speedup: {decode_speedup:.2f}x | E2E: {e2e_speedup:.2f}x | "
                    f"tau_eff: {tau_eff:.2f} | Lossless: {is_lossless} ({common_prefix}/{len(target_tokens)})"
                )

                row = {
                    "row_id": row_id,
                    "sample_id": sample_id,
                    "video_name": sample["video_name"],
                    "dataset": "vdc",
                    "drafter": "dflash",
                    "draft_depth": args.depth,
                    "selection_policy": "text_suffix_zero_visual",
                    "retention_percentage": ret,
                    "actual_visual_tokens": 0,
                    "full_visual_tokens": len(visual_positions),
                    "actual_text_tokens": actual_text_tokens,
                    "full_text_tokens": full_text_count,
                    "lossless": is_lossless,
                    "lossless_prefix_length": common_prefix,
                    "target_num_tokens": len(target_tokens),
                    "spec_num_tokens": len(spec_tokens),
                    "decode_speedup": decode_speedup,
                    "end_to_end_speedup": e2e_speedup,
                    "tau_effective": tau_eff,
                    "tau_proposal": tau_prop,
                    "prefill_seconds": spec_res.prefill_s,
                    "decode_seconds": spec_res.decode_s,
                    "end_to_end_seconds": spec_res.end_to_end_s,
                    "ar_prefill_seconds": ar_prefill_s,
                    "ar_decode_seconds": ar_decode_s,
                    "ar_end_to_end_seconds": ar_e2e_s,
                    "verification_steps": len(spec_res.acceptance_rounds),
                    "speculative_output_text": spec_text,
                    "speculative_output_hash": spec_hash,
                    "target_output_text": target_text,
                    "target_output_hash": target_hash,
                    "target_ground_truth_answer": sample["raw_metadata"].get("answer", ""),
                }

                with open(jsonl_output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_rows.add(row_id)

                with open(csv_output, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for r_idx, r_dict in enumerate(spec_res.acceptance_rounds):
                        writer.writerow([
                            sample_id,
                            ret,
                            r_idx,
                            r_dict.get("matched_proposals", 0),
                            r_dict.get("effective_emitted_tokens", 0),
                            r_dict.get("is_terminal", False),
                        ])

        except Exception as e:
            print(f"  [ERROR] Failed sample {sample_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nBenchmark completed! Output saved to {jsonl_output}")


if __name__ == "__main__":
    main()
