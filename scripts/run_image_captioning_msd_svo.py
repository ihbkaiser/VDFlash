#!/usr/bin/env python3
"""Run MSD 100% vs 0% Visual Retention Experiment on 50 Image Captioning S-V-O samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "externals" / "MSD" / "EAGLE"))

from src.analyze.Validate_Sparrow_hypothesises.runtime import (
    load_msd_qwen2vl,
    compact_qwen2vl_prefill,
    generate_msd_retention_video,
    PreparedPrefill,
    _hash_tensor,
)


def classify_token_role(token_text: str, svo: dict[str, str]) -> str:
    """Classify token into S, V, O, or Function word."""
    t = token_text.strip().lower()
    if not t or t in {",", ".", "!", "?", "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "is", "are", "and", "by", "its", "her", "his", "their"}:
        return "F"  # Function / syntax word
    
    s = svo.get("subject", "").lower()
    v = svo.get("verb", "").lower()
    o = svo.get("object", "").lower()

    if t in s:
        return "S"
    elif t in v:
        return "V"
    elif t in o:
        return "O"
    return "F"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="/path/to/local-resource")
    parser.add_argument("--msd-model", type=str, default="/path/to/local-resource")
    parser.add_argument("--manifest", type=str, default="dataset/image_captioning_svo_50.jsonl")
    parser.add_argument("--output", type=str, default="results/image_captioning_msd_svo/msd_image_captioning_svo_50.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor and models on GPUs...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = load_msd_qwen2vl(
        args.base_model,
        args.msd_model,
        device_map="model_parallel",
        max_memory="0:22GiB,1:14GiB"
    )

    with open(args.manifest, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:args.limit]

    print(f"Loaded {len(samples)} image captioning samples. Starting experiment...")

    existing_rows = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        existing_rows.add((row["sample_id"], row["retention_percentage"]))
                    except Exception:
                        pass
        print(f"Resume: Found {len(existing_rows)} existing records.")

    visual_device = torch.device("cuda:1")
    image_token_id = int(model.base_model.config.image_token_id)
    vision_start_token_id = getattr(model.base_model.config, "vision_start_token_id", None)
    vision_end_token_id = getattr(model.base_model.config, "vision_end_token_id", None)

    for idx, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        image_path = sample["image_path"]
        question = sample["question"]
        svo = sample.get("svo_structure", {})

        print(f"\n[{idx+1}/{len(samples)}] Sample {sample_id}: {sample.get('file_name')}")
        print(f"  SVO target: S={svo.get('subject')} | V={svo.get('verb')} | O={svo.get('object')}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        input_ids = batch["input_ids"].to("cuda:0")
        image_grid_thw = batch["image_grid_thw"].to(visual_device)
        pixel_values = batch["pixel_values"].to(visual_device)
        attention_mask = batch["attention_mask"].to("cuda:0")

        with torch.inference_mode():
            inputs_embeds = model.base_model.model.embed_tokens(input_ids)
            image_values = pixel_values.to(device=visual_device, dtype=model.base_model.visual.get_dtype())
            image_embeds = model.base_model.visual(image_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask, image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            )
            position_ids, rope_deltas = model.base_model.get_rope_index(input_ids, image_grid_thw, None, attention_mask)

        visual_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten()

        prepared_full = PreparedPrefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            video_grid_thw=image_grid_thw,
            video_positions=visual_positions,
            input_fingerprint=_hash_tensor(input_ids),
            video_token_id=image_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
        )

        # 1. Speculative decoding for 100% and 0%
        spec_results = {}
        for retention in [100.0, 0.0]:
            if (sample_id, retention) in existing_rows:
                print(f"  Retention {retention}% already exists in journal.")
                continue

            prepared_draft = compact_qwen2vl_prefill(prepared_full, retention)
            out_tokens, trace = generate_msd_retention_video(
                model,
                prepared_full,
                prepared_draft,
                max_new_tokens=args.max_new_tokens,
            )
            spec_ids = out_tokens[0, input_ids.shape[1]:].detach().cpu().tolist()
            spec_text = processor.tokenizer.decode(spec_ids, skip_special_tokens=True).strip()

            # Analyze S-V-O per-token acceptance
            pos_records = trace.get("acceptance_by_position", [])
            role_stats = {
                "S": {"total_tokens": 0, "accepted": 0, "fallback": 0},
                "V": {"total_tokens": 0, "accepted": 0, "fallback": 0},
                "O": {"total_tokens": 0, "accepted": 0, "fallback": 0},
                "F": {"total_tokens": 0, "accepted": 0, "fallback": 0},
            }

            for p in pos_records:
                for round_rec in p.get("position_records", []):
                    tok_text = round_rec.get("target_token_text", "")
                    role = classify_token_role(tok_text, svo)
                    role_stats[role]["total_tokens"] += 1
                    if round_rec.get("accepted", 0) == 1:
                        role_stats[role]["accepted"] += 1
                    else:
                        role_stats[role]["fallback"] += 1

            s_acc_rate = (role_stats["S"]["accepted"] / role_stats["S"]["total_tokens"]) if role_stats["S"]["total_tokens"] > 0 else 0.0
            v_acc_rate = (role_stats["V"]["accepted"] / role_stats["V"]["total_tokens"]) if role_stats["V"]["total_tokens"] > 0 else 0.0
            o_acc_rate = (role_stats["O"]["accepted"] / role_stats["O"]["total_tokens"]) if role_stats["O"]["total_tokens"] > 0 else 0.0
            f_acc_rate = (role_stats["F"]["accepted"] / role_stats["F"]["total_tokens"]) if role_stats["F"]["total_tokens"] > 0 else 0.0

            print(f"  Retention {retention:3.0f}% | Acc Length: {trace['accepted_prefix_tokens']:.2f} | Time: {trace['end_to_end_seconds']:.3f}s")
            print(f"    SVO Accept Rates: S={s_acc_rate:.1%} ({role_stats['S']['accepted']}/{role_stats['S']['total_tokens']}) | V={v_acc_rate:.1%} ({role_stats['V']['accepted']}/{role_stats['V']['total_tokens']}) | O={o_acc_rate:.1%} ({role_stats['O']['accepted']}/{role_stats['O']['total_tokens']}) | F={f_acc_rate:.1%} ({role_stats['F']['accepted']}/{role_stats['F']['total_tokens']})")
            print(f"    Output: {spec_text[:80]}...")

            result_row = {
                "sample_id": sample_id,
                "file_name": sample.get("file_name"),
                "retention_percentage": retention,
                "output_text": spec_text,
                "end_to_end_seconds": trace["end_to_end_seconds"],
                "accepted_prefix_tokens": trace["accepted_prefix_tokens"],
                "acceptance_trace": trace["acceptance_trace"],
                "svo_target": svo,
                "role_stats": role_stats,
                "s_acc_rate": s_acc_rate,
                "v_acc_rate": v_acc_rate,
                "o_acc_rate": o_acc_rate,
                "f_acc_rate": f_acc_rate,
            }

            with open(out_file, "a", encoding="utf-8") as out_h:
                out_h.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            existing_rows.add((sample_id, retention))

    print(f"\nAll 50 samples completed! Results recorded to {out_file}")


if __name__ == "__main__":
    main()
