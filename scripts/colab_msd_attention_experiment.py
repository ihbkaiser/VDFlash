#!/usr/bin/env python3
"""MSD Visual Attention Attribution & Verification Experiment on Qwen2.5-VL-3B.

Hypothesis Test:
Does MSD actually need visual context to correctly propose tokens?
1. Run MSD Speculative Decoding on VideoDetailCaption under Full (100%) vs Cut (0%) retention.
2. For every verified token in the output, capture Target Model attention weights to visual tokens
   (matching the Sparrow Figure 3(b) layer-wise / head-wise formulation).
3. Compute target attention histogram across all accepted/verified tokens to categorize them:
   - Low Visual Attention (< 50th percentile)
   - Medium Visual Attention (50th - 75th percentile)
   - Heavy Visual Attention (> 75th percentile)
   - Ultra-Heavy Visual Attention (> 90th percentile)
4. Compare acceptance rates: If MSD Cut achieves equal or higher acceptance rate on Heavy Visual tokens
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

# Ensure local imports work
for p in ("/content", "/content/specforge"):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, DynamicCache
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
from specforge.export.checkpoint_io import materialize_draft, resolve_training_state


HF_DATASET_REPO = "Tphuc15/vdflash-qwen25vl-3b-checkpoints"


def ensure_video_file(video_rel_path: str, token: str) -> str:
    """Download video from Hugging Face if not already present."""
    cache_path = Path("/content/videos") / Path(video_rel_path).name
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        return str(cache_path)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hf_subpath = f"VideoDetailCaption/{video_rel_path}"
    print(f"Downloading video from HF: {hf_subpath} ...")
    local_downloaded = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=hf_subpath,
        repo_type="dataset",
        token=token,
    )
    import shutil
    shutil.copyfile(local_downloaded, cache_path)
    return str(cache_path)


class MSDAttentionVerifier:
    def __init__(
        self,
        target_model: Qwen2_5_VLForConditionalGeneration,
        draft_model: torch.nn.Module,
        processor: AutoProcessor,
        device: torch.device,
    ):
        self.target = target_model
        self.draft = draft_model
        self.processor = processor
        self.device = device
        self.stop_token_ids = set()
        for attr in ("eos_token_id", "pad_token_id"):
            val = getattr(processor.tokenizer, attr, None)
            if val is not None:
                self.stop_token_ids.add(val)
        im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None:
            self.stop_token_ids.add(im_end)

    def _get_rope_index(self, **kwargs):
        fn = getattr(self.target, "get_rope_index", getattr(getattr(self.target, "model", None), "get_rope_index", None))
        if fn is None:
            raise AttributeError("Neither target nor target.model has get_rope_index")
        return fn(**kwargs)

    def decode_token(self, token_id: int) -> str:
        return self.processor.tokenizer.decode([token_id])

    def _crop_kv_cache(self, pkv: DynamicCache, target_len: int) -> DynamicCache:
        cropped = DynamicCache()
        for l in range(len(pkv.key_cache)):
            k = pkv.key_cache[l][:, :, :target_len, :].contiguous()
            v = pkv.value_cache[l][:, :, :target_len, :].contiguous()
            cropped.update(k, v, l)
        return cropped

    def _extract_token_attention(
        self,
        attentions: tuple[torch.Tensor, ...],
        query_idx: int,
        vis_positions: list[int],
        text_positions: list[int],
    ) -> dict[str, Any]:
        """Extract layer-wise and head-wise visual attention for query token (Sparrow Figure 3b)."""
        num_layers = len(attentions)
        per_layer_vis_mass = []
        layer_heads_vis = []

        vis_pos_tensor = torch.tensor(vis_positions, device=attentions[0].device, dtype=torch.long)
        text_pos_tensor = torch.tensor(text_positions, device=attentions[0].device, dtype=torch.long)

        total_vis_mass = 0.0
        total_text_mass = 0.0
        mid_layer_vis_mass = 0.0  # layers 8 to 20 (Sparrow Core Arena)

        for l in range(num_layers):
            # attn shape: (batch=1, num_heads, q_len, kv_len)
            layer_attn = attentions[l][0, :, query_idx, :]  # (num_heads, kv_len)
            head_vis = layer_attn[:, vis_pos_tensor].sum(dim=-1).float()  # (num_heads,)
            layer_v_mass = float(head_vis.sum().item())
            per_layer_vis_mass.append(round(layer_v_mass, 4))
            layer_heads_vis.append([round(float(x), 4) for x in head_vis.tolist()])

            total_vis_mass += layer_v_mass
            total_text_mass += float(layer_attn[:, text_pos_tensor].sum().item())
            if 8 <= l <= 20:
                mid_layer_vis_mass += layer_v_mass

        n_vis = max(1, len(vis_positions))
        n_text = max(1, len(text_positions))
        vis_density = (total_vis_mass / n_vis)
        text_density = (total_text_mass / n_text)
        density_ratio = (vis_density / max(1e-6, text_density))

        return {
            "vis_mass_all": round(total_vis_mass, 4),
            "vis_mass_mid": round(mid_layer_vis_mass, 4),
            "density_ratio": round(density_ratio, 4),
            "per_layer_vis_mass": per_layer_vis_mass,
            "layer_heads_vis": layer_heads_vis,
        }

    @torch.inference_mode()
    def run_speculative_with_attention(
        self,
        batch: dict[str, Any],
        retention_percentage: float,
        vis_positions: list[int],
        text_positions: list[int],
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        """Run MSD speculative decoding and extract Target Model attention for each token."""
        pos_ids, rope_deltas = self._get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
        delta = rope_deltas[0, 0].item()
        input_ids = batch["input_ids"]
        prompt_len = input_ids.shape[1]

        # Prefill Target
        prefill_out = self.target(
            **batch,
            position_ids=pos_ids,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        past_key_values = prefill_out.past_key_values
        t0 = prefill_out.logits[:, -1, :].argmax(dim=-1).item()

        # Visual retention pruning for draft
        if retention_percentage <= 0:
            prune_positions = list(vis_positions)
        elif retention_percentage >= 100.0:
            prune_positions = []
        else:
            keep_count = max(1, round(len(vis_positions) * retention_percentage / 100.0))
            prune_positions = vis_positions[:-keep_count]

        if not prune_positions:
            draft_pkv = past_key_values
        else:
            draft_pkv = DynamicCache()
            for l in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[l].clone()
                v = past_key_values.value_cache[l].clone()
                v[:, :, prune_positions, :] = 0.0
                k[:, :, prune_positions, :] = 0.0
                draft_pkv.update(k, v, l)

        # Feed t0 into target to get anchor
        p0 = prompt_len + delta
        p_ids_0 = torch.tensor([[[p0]]] * 3, device=self.device)
        t0_out = self.target(
            input_ids=torch.tensor([[t0]], device=self.device),
            position_ids=p_ids_0,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        past_key_values = t0_out.past_key_values
        prev_target_pred = t0_out.logits[:, -1, :].argmax(dim=-1).item()

        t0_attn_meta = self._extract_token_attention(
            t0_out.attentions,
            query_idx=0,
            vis_positions=vis_positions,
            text_positions=text_positions,
        )

        if not prune_positions:
            draft_pkv = past_key_values
            curr_h_draft = t0_out.hidden_states[-1][:, -1:, :]
        else:
            t0_draft_out = self.target(
                input_ids=torch.tensor([[t0]], device=self.device),
                position_ids=p_ids_0,
                past_key_values=draft_pkv,
                use_cache=True,
                output_hidden_states=True,
            )
            draft_pkv = t0_draft_out.past_key_values
            curr_h_draft = t0_draft_out.hidden_states[-1][:, -1:, :]

        curr_tok = t0
        kv_len = prompt_len + 1
        speculative_generated = [t0]
        token_records = [
            {
                "position": 0,
                "token_id": t0,
                "token_text": self.decode_token(t0),
                "accepted": 1,
                "role": "anchor",
                **t0_attn_meta,
            }
        ]

        round_index = 0
        answer_pos = 1
        accepted_count_total = 0
        total_proposals = 0

        while answer_pos < max_new_tokens and curr_tok not in self.stop_token_ids:
            # Draft generates candidate 1
            tok_emb = self.target.get_input_embeddings()(torch.tensor([[curr_tok]], device=self.device))
            vis_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            d_pids_1 = torch.tensor([[[kv_len - 1 + delta]]] * 3, device=self.device)
            d_h1 = self.draft(
                conditioning_hidden_state=curr_h_draft,
                next_token_embeddings=tok_emb,
                visual_embeddings=tok_emb,
                visual_token_mask=vis_mask,
                position_ids=d_pids_1,
            )
            c1 = self.target.lm_head(d_h1).argmax(dim=-1)[0, 0].item()

            # Draft generates candidate 2
            tok_emb2 = self.target.get_input_embeddings()(torch.tensor([[c1]], device=self.device))
            d_pids_2 = torch.tensor([[[kv_len + delta]]] * 3, device=self.device)
            d_h2 = self.draft(
                conditioning_hidden_state=d_h1,
                next_token_embeddings=tok_emb2,
                visual_embeddings=tok_emb2,
                visual_token_mask=vis_mask,
                position_ids=d_pids_2,
            )
            c2 = self.target.lm_head(d_h2).argmax(dim=-1)[0, 0].item()
            total_proposals += 2

            # Target Verification Pass (with attention capture)
            v_pids = torch.tensor([[[kv_len + delta, kv_len + 1 + delta]]] * 3, device=self.device)
            v_inputs = torch.tensor([[c1, c2]], device=self.device)
            v_out = self.target(
                input_ids=v_inputs,
                position_ids=v_pids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
            )

            # Target predictions
            target_pred_after_curr = prev_target_pred
            target_pred_after_c1 = v_out.logits[0, 0].argmax(dim=-1).item()
            target_pred_after_c2 = v_out.logits[0, 1].argmax(dim=-1).item()

            # Extract attentions for c1 and c2
            c1_attn = self._extract_token_attention(
                v_out.attentions, query_idx=0, vis_positions=vis_positions, text_positions=text_positions
            )
            c2_attn = self._extract_token_attention(
                v_out.attentions, query_idx=1, vis_positions=vis_positions, text_positions=text_positions
            )

            if c1 == target_pred_after_curr:
                # c1 is ACCEPTED
                accepted_count_total += 1
                speculative_generated.append(c1)
                token_records.append({
                    "position": answer_pos,
                    "round_index": round_index,
                    "token_id": c1,
                    "token_text": self.decode_token(c1),
                    "accepted": 1,
                    "role": "accepted_proposal",
                    **c1_attn,
                })

                if c2 == target_pred_after_c1:
                    # c2 is ALSO ACCEPTED
                    accepted_count_total += 1
                    speculative_generated.append(c2)
                    token_records.append({
                        "position": answer_pos + 1,
                        "round_index": round_index,
                        "token_id": c2,
                        "token_text": self.decode_token(c2),
                        "accepted": 1,
                        "role": "accepted_proposal",
                        **c2_attn,
                    })

                    # Bonus token after c2
                    bonus_tok = target_pred_after_c2
                    speculative_generated.append(bonus_tok)
                    curr_tok = bonus_tok

                    b_pids = torch.tensor([[[kv_len + 2 + delta]]] * 3, device=self.device)
                    b_out = self.target(
                        input_ids=torch.tensor([[curr_tok]], device=self.device),
                        position_ids=b_pids,
                        past_key_values=v_out.past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    past_key_values = b_out.past_key_values
                    prev_target_pred = b_out.logits[:, -1, :].argmax(dim=-1).item()

                    b_attn = self._extract_token_attention(
                        b_out.attentions, query_idx=0, vis_positions=vis_positions, text_positions=text_positions
                    )
                    token_records.append({
                        "position": answer_pos + 2,
                        "round_index": round_index,
                        "token_id": bonus_tok,
                        "token_text": self.decode_token(bonus_tok),
                        "accepted": 0,
                        "role": "fallback",
                        **b_attn,
                    })

                    if not prune_positions:
                        draft_pkv = past_key_values
                        curr_h_draft = b_out.hidden_states[-1][:, -1:, :]
                    else:
                        d_in_pids = torch.tensor([[[kv_len + delta, kv_len + 1 + delta, kv_len + 2 + delta]]] * 3, device=self.device)
                        d_step = self.target(
                            input_ids=torch.tensor([[c1, c2, curr_tok]], device=self.device),
                            position_ids=d_in_pids,
                            past_key_values=draft_pkv,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        draft_pkv = d_step.past_key_values
                        curr_h_draft = d_step.hidden_states[-1][:, -1:, :]

                    kv_len += 3
                    answer_pos += 3
                else:
                    # c1 accepted, c2 rejected
                    fallback_tok = target_pred_after_c1
                    speculative_generated.append(fallback_tok)
                    curr_tok = fallback_tok

                    pkv_cropped = self._crop_kv_cache(v_out.past_key_values, kv_len + 1)
                    fb_pids = torch.tensor([[[kv_len + 1 + delta]]] * 3, device=self.device)
                    fb_out = self.target(
                        input_ids=torch.tensor([[curr_tok]], device=self.device),
                        position_ids=fb_pids,
                        past_key_values=pkv_cropped,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    past_key_values = fb_out.past_key_values
                    prev_target_pred = fb_out.logits[:, -1, :].argmax(dim=-1).item()

                    fb_attn = self._extract_token_attention(
                        fb_out.attentions, query_idx=0, vis_positions=vis_positions, text_positions=text_positions
                    )
                    token_records.append({
                        "position": answer_pos + 1,
                        "round_index": round_index,
                        "token_id": fallback_tok,
                        "token_text": self.decode_token(fallback_tok),
                        "accepted": 0,
                        "role": "fallback",
                        **fb_attn,
                    })

                    if not prune_positions:
                        draft_pkv = past_key_values
                        curr_h_draft = fb_out.hidden_states[-1][:, -1:, :]
                    else:
                        d_in_pids = torch.tensor([[[kv_len + delta, kv_len + 1 + delta]]] * 3, device=self.device)
                        d_step = self.target(
                            input_ids=torch.tensor([[c1, curr_tok]], device=self.device),
                            position_ids=d_in_pids,
                            past_key_values=draft_pkv,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        draft_pkv = d_step.past_key_values
                        curr_h_draft = d_step.hidden_states[-1][:, -1:, :]

                    kv_len += 2
                    answer_pos += 2
            else:
                # c1 rejected
                fallback_tok = target_pred_after_curr
                speculative_generated.append(fallback_tok)
                curr_tok = fallback_tok

                pkv_cropped = self._crop_kv_cache(v_out.past_key_values, kv_len)
                fb_pids = torch.tensor([[[kv_len + delta]]] * 3, device=self.device)
                fb_out = self.target(
                    input_ids=torch.tensor([[curr_tok]], device=self.device),
                    position_ids=fb_pids,
                    past_key_values=pkv_cropped,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                past_key_values = fb_out.past_key_values
                prev_target_pred = fb_out.logits[:, -1, :].argmax(dim=-1).item()

                fb_attn = self._extract_token_attention(
                    fb_out.attentions, query_idx=0, vis_positions=vis_positions, text_positions=text_positions
                )
                token_records.append({
                    "position": answer_pos,
                    "round_index": round_index,
                    "token_id": fallback_tok,
                    "token_text": self.decode_token(fallback_tok),
                    "accepted": 0,
                    "role": "fallback",
                    **fb_attn,
                })

                if not prune_positions:
                    draft_pkv = past_key_values
                    curr_h_draft = fb_out.hidden_states[-1][:, -1:, :]
                else:
                    fb_draft_pids = torch.tensor([[[kv_len + delta]]] * 3, device=self.device)
                    d_step = self.target(
                        input_ids=torch.tensor([[curr_tok]], device=self.device),
                        position_ids=fb_draft_pids,
                        past_key_values=draft_pkv,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    draft_pkv = d_step.past_key_values
                    curr_h_draft = d_step.hidden_states[-1][:, -1:, :]

                kv_len += 1
                answer_pos += 1

            round_index += 1

        tau = 1.0 + (accepted_count_total / max(1, round_index))
        return {
            "tokens": speculative_generated,
            "text": self.processor.tokenizer.decode(speculative_generated, skip_special_tokens=True).strip(),
            "tau": round(tau, 3),
            "rounds": round_index,
            "accepted_count": accepted_count_total,
            "total_proposals": total_proposals,
            "token_records": token_records,
        }


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
    parser.add_argument("--manifest", default="/content/manifest_15.jsonl")
    parser.add_argument("--checkpoint", default="/content/checkpoints/qwen25vl-3b-msd-depth3-latest/training_state.pt")
    parser.add_argument("--output", default="/content/msd_attention_experiment_results.json")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args, _ = parser.parse_known_args()

    token = os.environ.get("HF_TOKEN") or ""

    print("=== STARTING MSD VISUAL ATTENTION ATTRIBUTION EXPERIMENT ===")
    print(f"Target: Qwen2.5-VL-3B-Instruct (Eager Attention)")
    print(f"Draft: MSD Depth 3 ({args.checkpoint})")
    print(f"Cohort Limit: {args.limit} videos")

    # Load records
    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if args.limit:
        records = records[:args.limit]
    print(f"Loaded {len(records)} sample records.")

    # 1. Load Target Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading Target Model...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    target = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",
    ).eval()

    # 2. Load MSD Draft Model
    print("Loading MSD Draft Model...")
    cfg_path = "/content/configs/qwen2.5-vl-3b-msd.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    cfg["num_hidden_layers"] = 3
    tmp_cfg_path = "/tmp/qwen25vl_msd_depth3.json"
    with open(tmp_cfg_path, "w") as f:
        json.dump(cfg, f)

    state = resolve_training_state(args.checkpoint)
    draft = materialize_draft(state, tmp_cfg_path)
    input_embeds = target.get_input_embeddings().weight.data
    draft.embed_tokens.weight.data.copy_(input_embeds)
    draft.to(device=device, dtype=torch.bfloat16).eval()

    verifier = MSDAttentionVerifier(target, draft, processor, device)

    # Storage for paired tokens
    paired_tokens: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []

    for idx, rec in enumerate(records):
        sample_id = rec.get("video_name") or rec.get("sample_id")
        video_rel = rec.get("local_video_path")
        print(f"\n[{idx+1}/{len(records)}] Processing {sample_id}...")

        # Ensure video is downloaded
        local_video = ensure_video_file(video_rel, token)
        question = rec.get("question") or "Please provide a detailed description of the video."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": local_video,
                        "max_pixels": 256 * 28 * 28,
                        "fps": 1.0,
                        "max_frames": 16,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        proc_kwargs = {"text": [rendered], "padding": True, "return_tensors": "pt"}
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
            proc_kwargs.update(dict(video_kwargs or {}))

        batch = processor(**proc_kwargs)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        prompt_ids = batch["input_ids"][0].tolist()

        # Find visual tokens vs text tokens in prompt
        v_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        v_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vis_positions = []
        in_vis = False
        for pos, tid in enumerate(prompt_ids):
            if tid == v_start:
                in_vis = True
            elif tid == v_end:
                in_vis = False
            elif in_vis:
                vis_positions.append(pos)

        text_positions = [i for i in range(len(prompt_ids)) if i not in vis_positions]
        print(f"  Prompt tokens: {len(prompt_ids)} (Visual: {len(vis_positions)}, Text: {len(text_positions)})")

        # Run Condition 1: MSD Full (100% Visual Context)
        print("  Running MSD Full (100% Visual)...")
        res_full = verifier.run_speculative_with_attention(
            batch,
            retention_percentage=100.0,
            vis_positions=vis_positions,
            text_positions=text_positions,
            max_new_tokens=args.max_new_tokens,
        )

        # Run Condition 2: MSD Cut (0% Visual Context)
        print("  Running MSD Cut (0% Visual)...")
        res_cut = verifier.run_speculative_with_attention(
            batch,
            retention_percentage=0.0,
            vis_positions=vis_positions,
            text_positions=text_positions,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"  Result: MSD Full Tau={res_full['tau']} | MSD Cut Tau={res_cut['tau']}")

        # Pair tokens by position
        recs_full = {r["position"]: r for r in res_full["token_records"]}
        recs_cut = {r["position"]: r for r in res_cut["token_records"]}
        common_positions = sorted(set(recs_full.keys()) & set(recs_cut.keys()))

        vid_token_count = 0
        for pos in common_positions:
            rf = recs_full[pos]
            rc = recs_cut[pos]
            # Since greedy verification ensures identical token ground-truth:
            tok_text = rf["token_text"]
            # Target attention is computed during target verification; average between the two runs for precision
            vis_mass_all = (rf["vis_mass_all"] + rc["vis_mass_all"]) / 2.0
            vis_mass_mid = (rf["vis_mass_mid"] + rc["vis_mass_mid"]) / 2.0
            density_ratio = (rf["density_ratio"] + rc["density_ratio"]) / 2.0

            paired_entry = {
                "sample_id": sample_id,
                "position": pos,
                "token_id": rf["token_id"],
                "token_text": tok_text,
                "vis_mass_all": round(vis_mass_all, 4),
                "vis_mass_mid": round(vis_mass_mid, 4),
                "density_ratio": round(density_ratio, 4),
                "per_layer_vis_mass": rf["per_layer_vis_mass"],
                "layer_heads_vis": rf["layer_heads_vis"],
                "accepted_full": int(rf["accepted"]),
                "accepted_cut": int(rc["accepted"]),
            }
            paired_tokens.append(paired_entry)
            vid_token_count += 1

        video_summaries.append({
            "sample_id": sample_id,
            "tau_full": res_full["tau"],
            "tau_cut": res_cut["tau"],
            "tokens_full": len(res_full["tokens"]),
            "tokens_cut": len(res_cut["tokens"]),
            "paired_tokens_count": vid_token_count,
        })
        torch.cuda.empty_cache()

    print(f"\nTotal paired tokens collected: {len(paired_tokens)}")

    # 3. Target Attention Histogram & Categorization
    vis_scores = [t["vis_mass_all"] for t in paired_tokens]
    p25 = float(np.percentile(vis_scores, 25))
    p50 = float(np.percentile(vis_scores, 50))
    p75 = float(np.percentile(vis_scores, 75))
    p90 = float(np.percentile(vis_scores, 90))

    print("\n=== TARGET VISUAL ATTENTION PERCENTILES ===")
    print(f"25th percentile: {p25:.4f}")
    print(f"50th percentile (Median): {p50:.4f}")
    print(f"75th percentile (Heavy Threshold): {p75:.4f}")
    print(f"90th percentile (Ultra-Heavy Threshold): {p90:.4f}")

    # Categorize tokens
    categories = {
        "LOW_VISUAL": [],       # < 50th percentile
        "MEDIUM_VISUAL": [],    # 50th to 75th percentile
        "HEAVY_VISUAL": [],     # > 75th percentile (Top 25%)
        "ULTRA_HEAVY": [],      # > 90th percentile (Top 10%)
    }

    for t in paired_tokens:
        score = t["vis_mass_all"]
        if score < p50:
            t["category"] = "LOW_VISUAL"
            categories["LOW_VISUAL"].append(t)
        elif score < p75:
            t["category"] = "MEDIUM_VISUAL"
            categories["MEDIUM_VISUAL"].append(t)
        else:
            t["category"] = "HEAVY_VISUAL"
            categories["HEAVY_VISUAL"].append(t)
        
        if score >= p90:
            categories["ULTRA_HEAVY"].append(t)

    # 4. Compare Acceptance Rates: MSD Full vs MSD Cut
    print("\n=== EXPERIMENT RESULTS: ACCEPTANCE BY VISUAL ATTENTION CATEGORY ===")
    cat_analysis = {}
    for cat_name, t_list in categories.items():
        n = len(t_list)
        if n == 0:
            continue
        acc_full = [t["accepted_full"] for t in t_list]
        acc_cut = [t["accepted_cut"] for t in t_list]

        rate_full = float(np.mean(acc_full)) * 100.0
        rate_cut = float(np.mean(acc_cut)) * 100.0
        diffs = [c - f for c, f in zip(acc_cut, acc_full)]
        delta_mean, delta_low, delta_high = bootstrap_ci(diffs)
        delta_pct = delta_mean * 100.0
        delta_low_pct = delta_low * 100.0
        delta_high_pct = delta_high * 100.0

        # Sample words in category
        sample_words = [t["token_text"].strip() for t in t_list[:12] if t["token_text"].strip()]

        cat_analysis[cat_name] = {
            "token_count": n,
            "pass_rate_full_pct": round(rate_full, 2),
            "pass_rate_cut_pct": round(rate_cut, 2),
            "delta_pct": round(delta_pct, 2),
            "delta_ci_95_pct": [round(delta_low_pct, 2), round(delta_high_pct, 2)],
            "sample_words": sample_words,
        }

        print(f"\n[{cat_name}] (N={n})")
        print(f"  MSD Full Pass Rate: {rate_full:.2f}%")
        print(f"  MSD Cut Pass Rate:  {rate_cut:.2f}%")
        print(f"  Delta (Cut - Full): {delta_pct:+.2f}% [95% CI: {delta_low_pct:+.2f}%, {delta_high_pct:+.2f}%]")
        print(f"  Sample Tokens: {sample_words}")

    # Figure 3b Sparrow Layer-wise Attention Matrix (aggregated over all verified tokens)
    layer_count = 36
    head_count = 16
    fig3b_matrix = np.zeros((layer_count, head_count), dtype=float)
    token_weights_count = 0
    for t in paired_tokens:
        layer_heads = t.get("layer_heads_vis")
        if layer_heads and len(layer_heads) == layer_count:
            for l in range(layer_count):
                fig3b_matrix[l, :] += np.array(layer_heads[l], dtype=float)
            token_weights_count += 1

    if token_weights_count > 0:
        fig3b_matrix = fig3b_matrix / token_weights_count

    output_payload = {
        "metadata": {
            "experiment": "MSD Visual Attention Attribution & Verification",
            "model_target": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_draft": "MSD Depth 3",
            "dataset": "VideoDetailCaption (15 videos)",
            "total_paired_tokens": len(paired_tokens),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "percentiles": {
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
        },
        "category_analysis": cat_analysis,
        "video_summaries": video_summaries,
        "fig3b_matrix": fig3b_matrix.tolist(),
        "tokens_sample": paired_tokens[:50],  # first 50 tokens detail
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved successfully to {args.output}!")


if __name__ == "__main__":
    main()
