#!/usr/bin/env python3
"""MSD Visual Attention Attribution & Verification Experiment on Qwen2.5-VL-3B.

Hypothesis Test:
Does MSD actually need visual context to correctly propose tokens?
1. Generate response trajectories with Target Model (Qwen2.5-VL-3B-Instruct).
2. For every verified token in the output, capture Target Model attention weights to visual tokens
   (matching the Sparrow Figure 3(b) layer-wise / head-wise formulation).
3. Evaluate MSD Draft (Depth 3) under Full (100% visual retention) vs Cut (0% visual retention).
4. Compute target attention histogram across all tokens to categorize them:
   - Low Visual Attention (< 50th percentile)
   - Medium Visual Attention (50th - 75th percentile)
   - Heavy Visual Attention (> 75th percentile)
   - Ultra-Heavy Visual Attention (> 90th percentile)
5. Compare acceptance rates: If MSD Cut achieves equal or higher acceptance rate on Heavy Visual tokens
   compared to MSD Full -> MSD genuinely does NOT need visual tokens.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parents[1]
specforge_root = repo_root / "src" / "train_Dflash_SpecForge"
for p in (repo_root, specforge_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from specforge.export.checkpoint_io import materialize_draft, resolve_training_state


def bootstrap_ci(diffs: list[float], n_bootstrap: int = 5000, ci: float = 0.95) -> tuple[float, float, float]:
    """Calculate mean and bootstrap confidence interval."""
    if not diffs:
        return 0.0, 0.0, 0.0
    arr = np.array(diffs, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val
    resamples = np.random.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    resample_means = np.mean(resamples, axis=1)
    lower = float(np.percentile(resample_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(resample_means, (1 + ci) / 2 * 100))
    return mean_val, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/path/to/local-resource")
    parser.add_argument("--checkpoint", default="dataset/qwen25vl-3b-msd-checkpoints/qwen25vl-3b-msd-depth3-latest/training_state.pt")
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--output", default="results/22_10_26/msd_attention_experiment_results.json")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("=== STARTING MSD VISUAL ATTENTION ATTRIBUTION EXPERIMENT ===")
    print(f"Target: Qwen2.5-VL-3B-Instruct (Eager Attention, BF16)")
    print(f"Draft:  MSD Depth 3 ({args.checkpoint})")
    print(f"Device: {device}")
    print(f"Cohort: {args.limit} videos (max_new_tokens={args.max_new_tokens})")
    print("=" * 70)

    # 1. Load Target Model
    print("Loading Target Model...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    target = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    ).eval()

    # 2. Load MSD Draft Model
    print("Loading MSD Draft Model...")
    cfg_path = specforge_root / "configs" / "qwen2.5-vl-3b-msd.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = 3
    tmp_cfg_path = "/tmp/qwen25vl_msd_depth3.json"
    with open(tmp_cfg_path, "w") as f:
        json.dump(cfg, f)

    state = resolve_training_state(args.checkpoint)
    draft = materialize_draft(state, tmp_cfg_path)
    draft.embed_tokens.weight.data.copy_(target.get_input_embeddings().weight.data)
    draft.to(device=device, dtype=torch.bfloat16).eval()
    print("Models loaded successfully.")

    # 3. Load manifest records
    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} video records from manifest.")

    # Storage structures
    all_tokens: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    fig3b_accum = np.zeros((36, 16), dtype=np.float64)
    fig3b_token_count = 0

    t_start_all = time.perf_counter()

    for idx, rec in enumerate(records):
        sample_id = rec.get("video_name") or rec.get("sample_id")
        video_rel = rec.get("local_video_path")
        video_full = Path(args.dataset_root) / video_rel
        if not video_full.exists():
            # Try alternate extension or path
            alt = video_full.with_suffix(".mp4" if video_full.suffix == ".mkv" else ".mkv")
            if alt.exists():
                video_full = alt
            else:
                print(f"[{idx+1}/{len(records)}] Video file not found: {video_full}, skipping.")
                continue

        print(f"\n[{idx+1}/{len(records)}] Processing {sample_id} ({rec.get('duration_sec', 0):.1f}s)...")
        question = rec.get("question") or "Please provide a detailed description of the video."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_full),
                        "max_pixels": 128 * 28 * 28,
                        "nframes": 16,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        proc_kwargs = {"text": [rendered], "padding": True, "return_tensors": "pt"}
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
            proc_kwargs.update(dict(video_kwargs or {}))

        batch = processor(**proc_kwargs)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        prompt_len = batch["input_ids"].shape[1]

        # 4. Generate Target ground-truth response
        t0 = time.perf_counter()
        with torch.no_grad():
            out_gen = target.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen_time = time.perf_counter() - t0
        full_ids = out_gen
        gen_tokens = full_ids[0, prompt_len:].tolist()
        gen_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        print(f"  Target generated {len(gen_tokens)} tokens in {gen_time:.2f}s: \"{gen_text[:60]}...\"")

        # 5. Full forward pass to capture Target Attention & Hidden States
        full_batch = dict(batch)
        full_batch["input_ids"] = full_ids
        full_batch["attention_mask"] = torch.ones_like(full_ids, dtype=torch.bool)
        pos_fn = getattr(target, "get_rope_index", getattr(target.model, "get_rope_index"))
        pos_ids, _ = pos_fn(
            input_ids=full_ids,
            image_grid_thw=full_batch.get("image_grid_thw"),
            video_grid_thw=full_batch.get("video_grid_thw"),
            attention_mask=full_batch["attention_mask"],
        )
        full_batch["position_ids"] = pos_ids

        with torch.no_grad():
            t_out = target(**full_batch, output_hidden_states=True, output_attentions=True)

        target_h = t_out.hidden_states[-1]
        embeds = target.get_input_embeddings()(full_ids)
        attentions = t_out.attentions

        # 6. Identify Visual Tokens vs Text Tokens
        v_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        v_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        full_ids_list = full_ids[0].tolist()
        vis_mask_full = torch.zeros(1, full_ids.shape[1] - 1, dtype=torch.bool, device=device)
        vis_positions = []
        in_v = False
        for pos, tid in enumerate(full_ids_list):
            if tid == v_start:
                in_v = True
            elif tid == v_end:
                in_v = False
            elif in_v:
                vis_positions.append(pos)
                if pos > 0 and (pos - 1) < vis_mask_full.shape[1]:
                    vis_mask_full[0, pos - 1] = True

        vis_mask_cut = torch.zeros_like(vis_mask_full)
        vis_pos_t = torch.tensor(vis_positions, device=device, dtype=torch.long)
        print(f"  Prompt length: {prompt_len} | Visual tokens: {len(vis_positions)}")

        # 7. Evaluate MSD Full vs MSD Cut along the generated trajectory
        cond_h = target_h[:, :-1]
        next_emb = embeds[:, 1:]

        with torch.no_grad():
            pred_h_full = draft(
                conditioning_hidden_state=cond_h,
                next_token_embeddings=next_emb,
                visual_embeddings=next_emb,
                visual_token_mask=vis_mask_full,
                position_ids=pos_ids[:, :, 1:],
            )
            pred_h_cut = draft(
                conditioning_hidden_state=cond_h,
                next_token_embeddings=next_emb,
                visual_embeddings=next_emb,
                visual_token_mask=vis_mask_cut,
                position_ids=pos_ids[:, :, 1:],
            )

        logits_full = target.lm_head(pred_h_full)
        logits_cut = target.lm_head(pred_h_cut)
        tokens_full = logits_full.argmax(dim=-1)[0, prompt_len - 1 : prompt_len - 1 + len(gen_tokens)]
        tokens_cut = logits_cut.argmax(dim=-1)[0, prompt_len - 1 : prompt_len - 1 + len(gen_tokens)]
        true_tokens = full_ids[0, prompt_len : prompt_len + len(gen_tokens)]

        match_full = (tokens_full == true_tokens).float()
        match_cut = (tokens_cut == true_tokens).float()

        # 8. Compute Target Attention Mass for each token (Sparrow Figure 3b)
        for i in range(len(gen_tokens)):
            p = prompt_len + i
            total_mass = 0.0
            mid_mass = 0.0
            per_layer_vis = []
            layer_heads_vis = []

            for l in range(len(attentions)):
                # attentions[l]: shape (1, num_heads, seq_len, seq_len)
                head_vis = attentions[l][0, :, p, vis_pos_t].sum(dim=-1).float()  # (num_heads,)
                l_mass = float(head_vis.sum().item())
                per_layer_vis.append(round(l_mass, 4))
                layer_heads_vis.append([round(float(x), 4) for x in head_vis.tolist()])
                total_mass += l_mass
                if 8 <= l <= 20:
                    mid_mass += l_mass

                # Accumulate for Sparrow Fig 3b heatmap
                fig3b_accum[l, :] += head_vis.cpu().numpy()

            fig3b_token_count += 1
            tok_text = processor.tokenizer.decode([true_tokens[i].item()])
            entry = {
                "sample_id": sample_id,
                "position": i,
                "token_id": int(true_tokens[i].item()),
                "token_text": tok_text,
                "vis_mass_all": round(total_mass, 4),
                "vis_mass_l18": per_layer_vis[18],
                "vis_mass_mid": round(mid_mass, 4),
                "per_layer_vis_mass": per_layer_vis,
                "layer_heads_vis": layer_heads_vis,
                "accepted_full": int(match_full[i].item()),
                "accepted_cut": int(match_cut[i].item()),
            }
            all_tokens.append(entry)

        # 9. Speculative lookahead simulation (K=2) to get effective tau
        pos = prompt_len - 1
        acc_full_sim = 0
        acc_cut_sim = 0
        rounds_full = 0
        rounds_cut = 0

        # Full simulation
        p_idx = 0
        while p_idx < len(gen_tokens) - 1:
            rounds_full += 1
            if match_full[p_idx].item() == 1:
                acc_full_sim += 1
                if match_full[p_idx + 1].item() == 1:
                    acc_full_sim += 1
                    p_idx += 3
                else:
                    p_idx += 2
            else:
                p_idx += 1

        # Cut simulation
        p_idx = 0
        while p_idx < len(gen_tokens) - 1:
            rounds_cut += 1
            if match_cut[p_idx].item() == 1:
                acc_cut_sim += 1
                if match_cut[p_idx + 1].item() == 1:
                    acc_cut_sim += 1
                    p_idx += 3
                else:
                    p_idx += 2
            else:
                p_idx += 1

        tau_full = 1.0 + (acc_full_sim / max(1, rounds_full))
        tau_cut = 1.0 + (acc_cut_sim / max(1, rounds_cut))

        print(f"  Match Rate: Full = {match_full.mean().item()*100:.1f}% | Cut = {match_cut.mean().item()*100:.1f}%")
        print(f"  Speculative Tau (K=2): Full = {tau_full:.3f} | Cut = {tau_cut:.3f}")

        video_summaries.append({
            "sample_id": sample_id,
            "duration_sec": rec.get("duration_sec", 0),
            "visual_tokens": len(vis_positions),
            "tokens_count": len(gen_tokens),
            "match_rate_full_pct": round(float(match_full.mean().item() * 100), 2),
            "match_rate_cut_pct": round(float(match_cut.mean().item() * 100), 2),
            "tau_full": round(tau_full, 3),
            "tau_cut": round(tau_cut, 3),
        })

        del t_out, full_batch, logits_full, logits_cut, pred_h_full, pred_h_cut, attentions, target_h, embeds, out_gen, full_ids
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start_all
    print("\n" + "=" * 70)
    print(f"ALL {len(video_summaries)} VIDEOS PROCESSED in {total_time:.1f}s")
    print(f"Total tokens collected: {len(all_tokens)}")
    print("=" * 70)

    # 10. Attention Percentiles & Token Categorization (Primary: Layer 18; Secondary: All Layers)
    vis_scores_l18 = [t["vis_mass_l18"] for t in all_tokens]
    p25_l18 = float(np.percentile(vis_scores_l18, 25))
    p50_l18 = float(np.percentile(vis_scores_l18, 50))
    p75_l18 = float(np.percentile(vis_scores_l18, 75))
    p90_l18 = float(np.percentile(vis_scores_l18, 90))

    vis_scores_all = [t["vis_mass_all"] for t in all_tokens]
    p25_all = float(np.percentile(vis_scores_all, 25))
    p50_all = float(np.percentile(vis_scores_all, 50))
    p75_all = float(np.percentile(vis_scores_all, 75))
    p90_all = float(np.percentile(vis_scores_all, 90))

    print("\n=== LAYER 18 TARGET VISUAL ATTENTION PERCENTILES (SPARROW IVSB m*=18) ===")
    print(f"Min:                     {min(vis_scores_l18):.4f}")
    print(f"25th percentile:         {p25_l18:.4f}")
    print(f"50th percentile (Median):{p50_l18:.4f}")
    print(f"75th percentile (Heavy): {p75_l18:.4f}")
    print(f"90th percentile (Ultra): {p90_l18:.4f}")
    print(f"Max:                     {max(vis_scores_l18):.4f}")

    def assign_cat_l18(score: float) -> str:
        if score < p50_l18:
            return "LOW_VISUAL"
        elif score < p75_l18:
            return "MEDIUM_VISUAL"
        elif score < p90_l18:
            return "HEAVY_VISUAL"
        else:
            return "ULTRA_HEAVY"

    def assign_cat_all(score: float) -> str:
        if score < p50_all:
            return "LOW_VISUAL"
        elif score < p75_all:
            return "MEDIUM_VISUAL"
        elif score < p90_all:
            return "HEAVY_VISUAL"
        else:
            return "ULTRA_HEAVY"

    for t in all_tokens:
        t["category_l18"] = assign_cat_l18(t["vis_mass_l18"])
        t["category_all"] = assign_cat_all(t["vis_mass_all"])
        # default category is Layer 18
        t["category"] = t["category_l18"]

    cat_counts_l18 = {c: sum(1 for t in all_tokens if t["category_l18"] == c) for c in ("LOW_VISUAL", "MEDIUM_VISUAL", "HEAVY_VISUAL", "ULTRA_HEAVY")}
    print("\nLayer 18 Category Distribution:")
    for c, cnt in cat_counts_l18.items():
        print(f"  * {c:<15}: {cnt} tokens ({cnt/len(all_tokens)*100:.1f}%)")

    # 11. Category Comparison for Layer 18
    def analyze_categories(cat_key: str, p75_val: float, score_key: str):
        analysis = {}
        for cat in ("LOW_VISUAL", "MEDIUM_VISUAL", "HEAVY_VISUAL", "ULTRA_HEAVY"):
            cat_tokens = [t for t in all_tokens if t[cat_key] == cat]
            n_cat = len(cat_tokens)
            if n_cat == 0:
                continue
            full_acc = sum(t["accepted_full"] for t in cat_tokens)
            cut_acc = sum(t["accepted_cut"] for t in cat_tokens)
            rate_full = (full_acc / n_cat) * 100.0
            rate_cut = (cut_acc / n_cat) * 100.0
            delta = rate_cut - rate_full

            diffs = [float(t["accepted_cut"] - t["accepted_full"]) * 100.0 for t in cat_tokens]
            mean_d, ci_low, ci_high = bootstrap_ci(diffs)
            sample_words = list(dict.fromkeys(t["token_text"].strip() for t in cat_tokens if len(t["token_text"].strip()) > 1))[:15]

            analysis[cat] = {
                "token_count": n_cat,
                "pass_rate_full_pct": round(rate_full, 2),
                "pass_rate_cut_pct": round(rate_cut, 2),
                "delta_pct": round(delta, 2),
                "delta_ci_95_pct": [round(ci_low, 2), round(ci_high, 2)],
                "sample_words": sample_words,
            }

        heavy_all = [t for t in all_tokens if t[score_key] >= p75_val]
        n_heavy = len(heavy_all)
        h_full_acc = sum(t["accepted_full"] for t in heavy_all)
        h_cut_acc = sum(t["accepted_cut"] for t in heavy_all)
        h_rate_full = (h_full_acc / n_heavy) * 100.0
        h_rate_cut = (h_cut_acc / n_heavy) * 100.0
        h_diffs = [float(t["accepted_cut"] - t["accepted_full"]) * 100.0 for t in heavy_all]
        h_mean_d, h_ci_low, h_ci_high = bootstrap_ci(h_diffs)
        analysis["ALL_HEAVY_VISUAL_TOP25"] = {
            "token_count": n_heavy,
            "pass_rate_full_pct": round(h_rate_full, 2),
            "pass_rate_cut_pct": round(h_rate_cut, 2),
            "delta_pct": round(h_rate_cut - h_rate_full, 2),
            "delta_ci_95_pct": [round(h_ci_low, 2), round(h_ci_high, 2)],
            "sample_words": list(dict.fromkeys(t["token_text"].strip() for t in heavy_all if len(t["token_text"].strip()) > 1))[:15],
        }
        return analysis

    category_analysis_l18 = analyze_categories("category_l18", p75_l18, "vis_mass_l18")
    category_analysis_all = analyze_categories("category_all", p75_all, "vis_mass_all")

    print("\n=== HEAD-TO-HEAD RESULTS FOR LAYER 18 (SPARROW IVSB CORE) ===")
    print(f"{'Category':<24} {'N':<6} {'MSD Full %':<12} {'MSD Cut %':<12} {'Delta (Cut-Full)':<18} {'95% CI'}")
    print("-" * 80)
    for cat, res in category_analysis_l18.items():
        ci_str = f"[{res['delta_ci_95_pct'][0]:+.2f}%, {res['delta_ci_95_pct'][1]:+.2f}%]"
        delta_str = f"{res['delta_pct']:+.2f}%"
        print(f"{cat:<24} {res['token_count']:<6} {res['pass_rate_full_pct']:<12.2f} {res['pass_rate_cut_pct']:<12.2f} {delta_str:<18} {ci_str}")

    # Normalize Fig 3b heatmap matrix
    if fig3b_token_count > 0:
        fig3b_matrix = (fig3b_accum / fig3b_token_count).tolist()
    else:
        fig3b_matrix = fig3b_accum.tolist()

    # 12. Save final JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "draft_model": "qwen25vl-3b-msd-depth3-latest",
            "videos_count": len(video_summaries),
            "total_tokens": len(all_tokens),
            "device": str(device),
            "primary_metric": "Layer 18 Target Visual Attention (Sparrow IVSB m*=18)",
        },
        "percentiles": {
            "p25": round(p25_l18, 4),
            "p50_median": round(p50_l18, 4),
            "p75_heavy": round(p75_l18, 4),
            "p90_ultra": round(p90_l18, 4),
            "min": round(float(min(vis_scores_l18)), 4),
            "max": round(float(max(vis_scores_l18)), 4),
        },
        "percentiles_all": {
            "p25": round(p25_all, 4),
            "p50_median": round(p50_all, 4),
            "p75_heavy": round(p75_all, 4),
            "p90_ultra": round(p90_all, 4),
            "min": round(float(min(vis_scores_all)), 4),
            "max": round(float(max(vis_scores_all)), 4),
        },
        "category_analysis": category_analysis_l18,  # Primary for plot scripts
        "category_analysis_l18": category_analysis_l18,
        "category_analysis_all": category_analysis_all,
        "video_summaries": video_summaries,
        "fig3b_matrix": fig3b_matrix,
        "all_vis_scores": vis_scores_l18,  # Primary scores for histogram
        "all_vis_scores_all": vis_scores_all,
        "tokens_sample": all_tokens[:200],
        "all_tokens": all_tokens,  # All 1,200 tokens
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved complete results to: {output_path}")


if __name__ == "__main__":
    main()
