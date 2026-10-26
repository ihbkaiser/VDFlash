#!/usr/bin/env python3
"""Run full DFlash speculative decoding benchmark on Qwen2.5-VL-3B across 2 GPUs.

Evaluates DFlash against AR baseline and produces directly comparable metrics with MSD:
- Datasets: VidHalluc (100 samples) and VideoChair (100 samples)
- 4 retention levels: 100%, 50%, 25%, 0%
- Telemetry: acceptance_rounds, tau_effective, tau_proposal, timings, decode_speedup, e2e_speedup, losslessness
- Multi-GPU: Asymmetric sharding across RTX 3090 (24GB) on GPU 0 and RTX A4000 (16GB) on GPU 1
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


def select_visual_positions(
    visual_count: int,
    percentage: float,
    policy: str = "suffix",
) -> list[int]:
    """Select indices of visual tokens to retain according to retention policy."""
    if visual_count == 0 or percentage <= 0:
        return []
    if percentage >= 100.0:
        return list(range(visual_count))
    keep = max(1, round(visual_count * percentage / 100.0))
    if policy in ("suffix", "tail"):
        return list(range(visual_count - keep, visual_count))
    elif policy in ("prefix", "head"):
        return list(range(keep))
    elif policy == "uniform":
        return sorted({min(visual_count - 1, (index * visual_count) // keep) for index in range(keep)})
    else:
        raise ValueError(f"Unknown visual retention policy: {policy}")


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
            if "candidates" in item and item["candidates"]:
                opts = "\n".join([f"{chr(65+i)}. {cand}" for i, cand in enumerate(item["candidates"])])
                question = f"{question}\nOptions:\n{opts}\nPlease choose the best option directly."
            elif "choices" in item and item["choices"] and isinstance(item["choices"], dict):
                opts = "\n".join([f"{k}. {v}" for k, v in item["choices"].items()])
                question = f"{question}\nOptions:\n{opts}\nPlease choose the best option directly."

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


def run_benchmark_for_dataset(
    target: Any,
    draft: Any,
    processor: Any,
    dataset_name: str,
    manifest_path: str,
    dataset_root: str,
    output_dir: Path,
    depth: int = 3,
    limit: int | None = None,
    max_new_tokens: int = 128,
    retention_percentages: list[float] | None = None,
    retention_policy: str = "suffix",
):
    if retention_percentages is None:
        retention_percentages = [100.0, 50.0, 25.0, 5.0, 0.0]

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output = output_dir / f"dflash_retention_{retention_policy}_{dataset_name.lower()}.jsonl"
    csv_output = output_dir / f"draft_iterations_{retention_policy}_{dataset_name.lower()}.csv"

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
        print(f"[{dataset_name}] Resume: Found {len(existing_rows)} completed rows in {jsonl_output.name}")

    records = load_dataset_records(manifest_path, dataset_root)
    if limit is not None:
        records = records[:limit]
    print(f"[{dataset_name}] Loaded {len(records)} samples from {manifest_path}")

    stop_token_ids = _eos_token_ids(processor, target)
    decoder = InstrumentedDFlashDecoder(
        target,
        draft,
        device=torch.device("cuda:0"),
        draft_device=torch.device("cuda:1"),
    )

    # Initialize CSV if not exists
    if not csv_output.exists():
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id", "retention", "round_idx", "accepted_proposals",
                "effective_emitted", "is_terminal"
            ])

    for sample_idx, sample in enumerate(records):
        sample_id = sample["sample_id"]
        video_path = sample["video_path"]
        question = sample["question"]

        all_done = True
        for ret in retention_percentages:
            row_id = f"{sample_id}:dflash:depth{depth}:{retention_policy}:retention-{int(ret)}"
            if row_id not in existing_rows:
                all_done = False
                break
        if all_done:
            continue

        print(f"\n[{dataset_name} | Sample {sample_idx + 1}/{len(records)}] Processing {sample_id}...")

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
            batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}
            pos_ids = _compute_position_ids(target, batch)
            target_kwargs = {k: v for k, v in batch.items() if k not in ("input_ids", "attention_mask")}

            prompt_len = int(batch["input_ids"].shape[1])
            visual_positions = find_visual_positions(batch["input_ids"], target=target, processor=processor)

            # 1. Target Autoregressive Greedy Baseline
            prompt_obj = PromptWrapper(batch, pos_ids, target_kwargs)
            target_output, target_timing = _target_greedy(
                target,
                prompt_obj,
                max_new_tokens=max_new_tokens,
                stop_token_ids=stop_token_ids,
                device=torch.device("cuda:0"),
            )
            target_tokens = target_output[0, prompt_len:].detach().cpu().tolist()
            target_text = processor.tokenizer.decode(target_tokens, skip_special_tokens=True).strip()
            target_hash = sha256_tokens(target_tokens)
            ar_prefill_s = target_timing["prefill_s"]
            ar_decode_s = target_timing["decode_s"]
            ar_e2e_s = target_timing["end_to_end_s"]

            print(f"  [AR Baseline] Output tokens: {len(target_tokens)} | Decode: {ar_decode_s:.3f}s | Text: {repr(target_text[:60])}...")

            # 2. Run DFlash Speculative Decoding for each retention percentage
            for ret in retention_percentages:
                row_id = f"{sample_id}:dflash:depth{depth}:{retention_policy}:retention-{int(ret)}"
                if row_id in existing_rows:
                    continue

                if ret >= 100.0:
                    keep_mask = None
                    actual_visual_tokens = len(visual_positions)
                elif ret <= 0.0:
                    actual_visual_tokens = 0
                    keep_mask = torch.ones(prompt_len, dtype=torch.bool, device="cuda:0")
                    for p_idx in visual_positions:
                        keep_mask[p_idx] = False
                else:
                    local_kept = select_visual_positions(
                        len(visual_positions), ret, policy=retention_policy
                    )
                    kept_positions = {visual_positions[i] for i in local_kept}
                    actual_visual_tokens = len(kept_positions)
                    keep_mask = torch.ones(prompt_len, dtype=torch.bool, device="cuda:0")
                    for p_idx in visual_positions:
                        if p_idx not in kept_positions:
                            keep_mask[p_idx] = False

                t0_spec = time.perf_counter()
                spec_res = decoder.decode(
                    input_ids=batch["input_ids"],
                    position_ids=pos_ids,
                    target_kwargs=target_kwargs,
                    max_new_tokens=max_new_tokens,
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
                    f"  [Ret {int(ret):3d}%] Decode speedup: {decode_speedup:.2f}x | "
                    f"tau_eff: {tau_eff:.2f} | Lossless: {is_lossless} (prefix: {common_prefix}/{len(target_tokens)}) | "
                    f"Decode time: {spec_res.decode_s:.3f}s"
                )

                # Write to JSONL
                row = {
                    "row_id": row_id,
                    "sample_id": sample_id,
                    "video_name": sample["video_name"],
                    "dataset": dataset_name,
                    "drafter": "dflash",
                    "draft_depth": depth,
                    "selection_policy": retention_policy,
                    "retention_percentage": ret,
                    "actual_visual_tokens": actual_visual_tokens,
                    "full_visual_tokens": len(visual_positions),
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

                # Append iteration trace to CSV
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

    print(f"\n[{dataset_name}] Benchmark completed! Output saved to {jsonl_output}")


def main():
    parser = argparse.ArgumentParser(description="Run DFlash Speculative Decoding Benchmark on 2 GPUs.")
    parser.add_argument("--target-model", default="/path/to/local-resource")
    parser.add_argument("--checkpoint", default="dataset/qwen25vl-3b-depth1-depth3/qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt")
    parser.add_argument("--draft-config", default="src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash-h32-depth3.json")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dataset", choices=["vidhalluc", "videochair", "vdc", "both", "all"], default="all")
    parser.add_argument("--policy", choices=["suffix", "prefix", "uniform"], default="suffix")
    parser.add_argument("--retention-levels", type=str, default="100,50,25,5,0")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dflash_eval_qwen25vl3b"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    retention_percentages = [float(x.strip()) for x in args.retention_levels.split(",") if x.strip()]

    print("=================================================================")
    print(" DFLASH MULTI-GPU BENCHMARK ON QWEN2.5-VL-3B (RTX 3090 + RTX A4000)")
    print("=================================================================")
    print(f"Target Model:     {args.target_model}")
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Draft Config:     {args.draft_config}")
    print(f"Draft Depth:      {args.depth}")
    print(f"Policy:           {args.policy}")
    print(f"Retention Levels: {retention_percentages}")
    print(f"Dataset:          {args.dataset}")
    print(f"Output Dir:       {args.output_dir}")
    print(f"Limit:            {args.limit}")
    print("=================================================================")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading Target Model across 2 GPUs (device_map=auto, max_memory={0: 18GiB, 1: 12GiB})...")
    processor = AutoProcessor.from_pretrained(args.target_model)
    target = AutoModelForImageTextToText.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "18GiB", 1: "12GiB"},
        attn_implementation="sdpa",
    )
    print(f"Target loaded! GPU 0 VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB | GPU 1 VRAM: {torch.cuda.memory_allocated(1)/1e9:.2f} GB")

    print("\nLoading DFlash Draft Model on cuda:1...")
    draft, load_s = _load_draft(
        args.checkpoint,
        args.draft_config,
        device=torch.device("cuda:1"),
        dtype=torch.bfloat16,
    )
    print(f"Draft loaded in {load_s:.2f}s on cuda:1! Total VRAM: GPU 0 = {torch.cuda.memory_allocated(0)/1e9:.2f} GB, GPU 1 = {torch.cuda.memory_allocated(1)/1e9:.2f} GB")

    datasets_to_run = []
    if args.dataset in {"vidhalluc", "both", "all"}:
        datasets_to_run.append((
            "vidhalluc",
            "dataset/VidHalluc/test.jsonl",
            "dataset/VidHalluc",
        ))
    if args.dataset in {"videochair", "both", "all"}:
        datasets_to_run.append((
            "videochair",
            "dataset/VideoChair/test.jsonl",
            "dataset/VideoChair",
        ))
    if args.dataset in {"vdc", "all"}:
        vdc_manifest = (
            "dataset/VideoDetailCaption/test.jsonl"
            if Path("dataset/VideoDetailCaption/test.jsonl").exists()
            else "dataset/VideoDetailCaption/test_100.jsonl"
        )
        datasets_to_run.append((
            "vdc",
            vdc_manifest,
            "dataset/VideoDetailCaption",
        ))

    for name, manifest, root in datasets_to_run:
        ds_out = args.output_dir / name
        run_benchmark_for_dataset(
            target=target,
            draft=draft,
            processor=processor,
            dataset_name=name,
            manifest_path=manifest,
            dataset_root=root,
            output_dir=ds_out,
            depth=args.depth,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            retention_percentages=retention_percentages,
            retention_policy=args.policy,
        )

    print("\n=================================================================")
    print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f" Results available at: {args.output_dir}")
    print("=================================================================")


if __name__ == "__main__":
    main()
