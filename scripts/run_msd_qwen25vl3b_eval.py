#!/usr/bin/env python3
"""Run full MSD (Multi-modal Speculative Decoding) benchmark on Qwen2.5-VL-3B.

Supports:
- Checkpoints: Depth 3 or Depth 5 MSD draft model
- Datasets: VidHalluc (100 samples) and VideoChair (100 samples)
- 4 retention levels: 100%, 50%, 25%, 0%
- Telemetry: full acceptance_by_position, acceptance_rounds, timings, speedup, losslessness
- Multi-GPU: assignable to any CUDA device
- Resume: skips already completed (sample_id, retention) rows
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parents[1]
specforge_root = repo_root / "src" / "train_Dflash_SpecForge"
for p in (repo_root, specforge_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, DynamicCache
from specforge.export.checkpoint_io import materialize_draft, resolve_training_state
from specforge.modeling.draft.msd import MSDConfig


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:16]


def select_uniform_visual_positions(visual_count: int, percentage: float) -> list[int]:
    if visual_count == 0 or percentage <= 0:
        return []
    if percentage >= 100.0:
        return list(range(visual_count))
    keep = max(1, round(visual_count * percentage / 100.0))
    positions = sorted({min(visual_count - 1, (index * visual_count) // keep) for index in range(keep)})
    return positions


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
                "video_path": str(video_full) if video_full.exists() else None,
                "question": question,
                "raw_metadata": item,
            })
    return records


class MSDSpeculativeEngine:
    def __init__(
        self,
        target_model: Qwen2_5_VLForConditionalGeneration,
        draft_model: torch.nn.Module,
        processor: AutoProcessor,
        device: torch.device,
        proposal_count: int = 2,
    ):
        self.target = target_model
        self.draft = draft_model
        self.processor = processor
        self.device = device
        self.proposal_count = proposal_count
        self.stop_token_ids = set()
        for attr in ("eos_token_id", "pad_token_id"):
            val = getattr(processor.tokenizer, attr, None)
            if val is not None:
                self.stop_token_ids.add(val)
        im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None:
            self.stop_token_ids.add(im_end)

    def decode_token(self, token_id: int) -> str:
        return self.processor.tokenizer.decode([token_id])

    @torch.inference_mode()
    def run_autoregressive(
        self,
        batch: dict[str, Any],
        max_new_tokens: int,
    ) -> tuple[list[int], float, float]:
        """Greedy autoregressive target decoding baseline with repetition_penalty=1.0."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()

        pos_ids, _ = self.target.get_rope_index(
            batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )

        # Prefill timing
        t_prefill_0 = time.perf_counter()
        out = self.target(**batch, position_ids=pos_ids, use_cache=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_seconds = time.perf_counter() - t_prefill_0

        prompt_len = batch["input_ids"].shape[1]
        ar_out = self.target.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        total_time = time.perf_counter() - t_start
        decode_seconds = max(0.0, total_time - prefill_seconds)
        generated = ar_out[0, prompt_len:].tolist()

        return generated, prefill_seconds, decode_seconds

    def _crop_kv_cache(self, past_key_values: Any, target_len: int) -> Any:
        """Crop DynamicCache to target sequence length."""
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(target_len)
            return past_key_values
        if hasattr(past_key_values, "key_cache"):
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][..., :target_len, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][..., :target_len, :]
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = target_len
            return past_key_values
        return past_key_values

    @torch.inference_mode()
    def run_speculative(
        self,
        batch: dict[str, Any],
        retention_percentage: float,
        visual_positions_in_prompt: list[int],
        max_new_tokens: int,
        target_ground_truth_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run MSD speculative decoding with visual token retention compaction."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_start = time.perf_counter()

        pos_ids, rope_deltas = self.target.get_rope_index(
            batch["input_ids"],
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
        delta = rope_deltas[0, 0].item()
        prompt_len = batch["input_ids"].shape[1]

        # 1. Target Prefill on full context
        t_prefill_0 = time.perf_counter()
        out = self.target(**batch, position_ids=pos_ids, use_cache=True, output_hidden_states=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prefill_seconds = time.perf_counter() - t_prefill_0

        past_key_values = out.past_key_values
        t0 = out.logits[:, -1, :].argmax(dim=-1).item()

        # Build Draft KV Cache with visual retention compaction
        local_kept = select_uniform_visual_positions(len(visual_positions_in_prompt), retention_percentage)
        kept_positions = {visual_positions_in_prompt[i] for i in local_kept}
        prune_positions = [pos for pos in visual_positions_in_prompt if pos not in kept_positions]

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

        # Step 0: Feed t0 into target to keep KV in sync and get h(t0) + prediction for token 1
        p0 = prompt_len + delta
        p_ids_0 = torch.tensor([[[p0]]] * 3, device=self.device)
        t0_out = self.target(
            input_ids=torch.tensor([[t0]], device=self.device),
            position_ids=p_ids_0,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = t0_out.past_key_values
        prev_target_pred = t0_out.logits[:, -1, :].argmax(dim=-1).item()

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

        speculative_generated: list[int] = [t0]
        acceptance_rounds: list[dict[str, Any]] = []
        acceptance_by_position: list[dict[str, Any]] = []
        acceptance_trace: list[int] = []

        # Position 0 anchor record
        anchor_rec = {
            "position": 0,
            "round_index": 0,
            "proposed": 0,
            "accepted": 1,
            "role": "anchor",
            "speculative_token_id": t0,
            "speculative_token_text": self.decode_token(t0),
            "target_token_id": t0,
            "target_token_text": self.decode_token(t0),
        }
        acceptance_by_position.append(anchor_rec)

        round_index = 0
        answer_pos = 1
        total_verification_seconds = 0.0
        t_decode_start = time.perf_counter()

        while answer_pos < max_new_tokens and curr_tok not in self.stop_token_ids:
            # Draft proposes K=2 tokens
            # Proposal 1
            tok_emb = self.target.model.embed_tokens(torch.tensor([[curr_tok]], device=self.device))
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

            # Proposal 2
            tok_emb2 = self.target.model.embed_tokens(torch.tensor([[c1]], device=self.device))
            d_pids_2 = torch.tensor([[[kv_len + delta]]] * 3, device=self.device)
            d_h2 = self.draft(
                conditioning_hidden_state=d_h1,
                next_token_embeddings=tok_emb2,
                visual_embeddings=tok_emb2,
                visual_token_mask=vis_mask,
                position_ids=d_pids_2,
            )
            c2 = self.target.lm_head(d_h2).argmax(dim=-1)[0, 0].item()

            # Target Verification Pass
            t_v0 = time.perf_counter()
            v_pids = torch.tensor([[[kv_len + delta, kv_len + 1 + delta]]] * 3, device=self.device)
            v_inputs = torch.tensor([[c1, c2]], device=self.device)
            v_out = self.target(
                input_ids=v_inputs,
                position_ids=v_pids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            total_verification_seconds += (time.perf_counter() - t_v0)

            target_pred_after_curr = prev_target_pred
            target_pred_after_c1 = v_out.logits[0, 0].argmax(dim=-1).item()
            target_pred_after_c2 = v_out.logits[0, 1].argmax(dim=-1).item()

            round_record = {
                "round_index": round_index,
                "answer_position_start": answer_pos - 1,
                "accepted_proposal_count": 0,
                "position_records": [],
            }
            round_pos_records = []
            accepted_k = 0

            if c1 == target_pred_after_curr:
                # c1 accepted!
                accepted_k += 1
                speculative_generated.append(c1)
                rec_0 = {
                    "position": answer_pos,
                    "round_index": round_index,
                    "proposed": 1,
                    "accepted": 1,
                    "role": "accepted_proposal",
                    "speculative_token_id": c1,
                    "speculative_token_text": self.decode_token(c1),
                    "target_token_id": c1,
                    "target_token_text": self.decode_token(c1),
                }
                round_pos_records.append(rec_0)

                if c2 == target_pred_after_c1:
                    # c2 accepted!
                    accepted_k += 1
                    speculative_generated.append(c2)
                    rec_1 = {
                        "position": answer_pos + 1,
                        "round_index": round_index,
                        "proposed": 1,
                        "accepted": 1,
                        "role": "accepted_proposal",
                        "speculative_token_id": c2,
                        "speculative_token_text": self.decode_token(c2),
                        "target_token_id": c2,
                        "target_token_text": self.decode_token(c2),
                    }
                    round_pos_records.append(rec_1)

                    # Bonus fallback token after c2
                    bonus_tok = target_pred_after_c2
                    speculative_generated.append(bonus_tok)
                    rec_b = {
                        "position": answer_pos + 2,
                        "round_index": round_index,
                        "proposed": 0,
                        "accepted": 0,
                        "role": "fallback",
                        "speculative_token_id": bonus_tok,
                        "speculative_token_text": self.decode_token(bonus_tok),
                        "target_token_id": bonus_tok,
                        "target_token_text": self.decode_token(bonus_tok),
                    }
                    round_pos_records.append(rec_b)

                    curr_tok = bonus_tok
                    b_pids = torch.tensor([[[kv_len + 2 + delta]]] * 3, device=self.device)
                    b_out = self.target(
                        input_ids=torch.tensor([[curr_tok]], device=self.device),
                        position_ids=b_pids,
                        past_key_values=v_out.past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_key_values = b_out.past_key_values
                    prev_target_pred = b_out.logits[:, -1, :].argmax(dim=-1).item()

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

                    kv_len = kv_len + 3
                    answer_pos += 3
                else:
                    # c1 accepted, c2 rejected; emit fallback after c1
                    fallback_tok = target_pred_after_c1
                    speculative_generated.append(fallback_tok)
                    rec_1 = {
                        "position": answer_pos + 1,
                        "round_index": round_index,
                        "proposed": 0,
                        "accepted": 0,
                        "role": "fallback",
                        "speculative_token_id": c2,
                        "speculative_token_text": self.decode_token(c2),
                        "target_token_id": fallback_tok,
                        "target_token_text": self.decode_token(fallback_tok),
                    }
                    round_pos_records.append(rec_1)

                    curr_tok = fallback_tok
                    pkv_cropped = self._crop_kv_cache(v_out.past_key_values, kv_len + 1)
                    fb_pids = torch.tensor([[[kv_len + 1 + delta]]] * 3, device=self.device)
                    fb_out = self.target(
                        input_ids=torch.tensor([[curr_tok]], device=self.device),
                        position_ids=fb_pids,
                        past_key_values=pkv_cropped,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past_key_values = fb_out.past_key_values
                    prev_target_pred = fb_out.logits[:, -1, :].argmax(dim=-1).item()

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

                    kv_len = kv_len + 2
                    answer_pos += 2
            else:
                # c1 rejected; emit fallback after curr
                fallback_tok = target_pred_after_curr
                speculative_generated.append(fallback_tok)
                rec_0 = {
                    "position": answer_pos,
                    "round_index": round_index,
                    "proposed": 0,
                    "accepted": 0,
                    "role": "fallback",
                    "speculative_token_id": c1,
                    "speculative_token_text": self.decode_token(c1),
                    "target_token_id": fallback_tok,
                    "target_token_text": self.decode_token(fallback_tok),
                }
                round_pos_records.append(rec_0)

                curr_tok = fallback_tok
                pkv_cropped = self._crop_kv_cache(v_out.past_key_values, kv_len)
                fb_pids = torch.tensor([[[kv_len + delta]]] * 3, device=self.device)
                fb_out = self.target(
                    input_ids=torch.tensor([[curr_tok]], device=self.device),
                    position_ids=fb_pids,
                    past_key_values=pkv_cropped,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = fb_out.past_key_values
                prev_target_pred = fb_out.logits[:, -1, :].argmax(dim=-1).item()

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

                kv_len = kv_len + 1
                answer_pos += 1

            round_record["accepted_proposal_count"] = accepted_k
            round_record["position_records"] = round_pos_records
            acceptance_rounds.append(round_record)
            acceptance_by_position.extend(round_pos_records)
            acceptance_trace.append(accepted_k)
            round_index += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        decode_seconds = time.perf_counter() - t_decode_start
        end_to_end_seconds = time.perf_counter() - t_start

        # Check losslessness against autoregressive ground truth
        lossless = True
        lossless_prefix_len = len(speculative_generated)
        if target_ground_truth_ids is not None:
            min_len = min(len(speculative_generated), len(target_ground_truth_ids))
            common_prefix = 0
            for i in range(min_len):
                if speculative_generated[i] == target_ground_truth_ids[i]:
                    common_prefix += 1
                else:
                    break
            lossless_prefix_len = common_prefix
            lossless = (common_prefix == len(target_ground_truth_ids))

        accepted_prefix_tokens = (sum(acceptance_trace) / len(acceptance_trace)) if acceptance_trace else 0.0

        return {
            "speculative_output_ids": speculative_generated,
            "lossless": lossless,
            "lossless_prefix_length": lossless_prefix_len,
            "acceptance_rounds": acceptance_rounds,
            "acceptance_by_position": acceptance_by_position,
            "acceptance_trace": acceptance_trace,
            "accepted_prefix_tokens": accepted_prefix_tokens,
            "verification_steps": len(acceptance_rounds),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "verification_seconds": total_verification_seconds,
            "speculative_seconds": end_to_end_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "actual_visual_tokens": len(kept_positions),
        }


def run_benchmark_for_config(
    gpu_id: int,
    depth: int,
    dataset_name: str,
    manifest_path: str,
    dataset_root: str,
    output_dir: Path,
    target_model_path: str,
    checkpoint_path: str,
    limit: int | None = None,
    max_new_tokens: int = 128,
    overwrite: bool = False,
):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"\n[{dataset_name} | Depth {depth} | GPU {gpu_id}] Initializing Benchmark...")
    print(f"Target: {target_model_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output = output_dir / f"msd_retention_{dataset_name.lower()}.jsonl"
    csv_output = output_dir / "draft_iterations_detailed.csv"

    if overwrite:
        if jsonl_output.exists():
            jsonl_output.unlink()
        if csv_output.exists():
            csv_output.unlink()

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

    # Load records
    records = load_dataset_records(manifest_path, dataset_root)
    if limit is not None:
        records = records[:limit]
    print(f"Loaded {len(records)} samples from {manifest_path}")

    # Load Target Model
    processor = AutoProcessor.from_pretrained(target_model_path)
    target = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        target_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    ).eval()

    # Load Draft Model
    base_cfg_path = specforge_root / "configs" / "qwen2.5-vl-3b-msd.json"
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg = dict(base_cfg)
    cfg["num_hidden_layers"] = depth
    tmp_cfg_path = f"/tmp/qwen25vl_msd_depth{depth}_gpu{gpu_id}.json"
    with open(tmp_cfg_path, "w") as f:
        json.dump(cfg, f)

    state = resolve_training_state(checkpoint_path)
    draft = materialize_draft(state, tmp_cfg_path)
    draft.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight.data)
    draft.to(device=device, dtype=torch.bfloat16).eval()

    engine = MSDSpeculativeEngine(target, draft, processor, device, proposal_count=2)
    from qwen_vl_utils import process_vision_info

    retention_percentages = [100.0, 50.0, 25.0, 0.0]

    for sample_idx, sample in enumerate(records):
        sample_id = sample["sample_id"]
        video_path = sample["video_path"]
        question = sample["question"]

        # Check if all 4 retention rows for this sample already exist
        all_done = True
        for ret in retention_percentages:
            row_id = f"{sample_id}:native:uniform:retention-{int(ret)}"
            if row_id not in existing_rows:
                all_done = False
                break
        if all_done:
            continue

        print(f"\n[{sample_idx + 1}/{len(records)}] Processing {sample_id}...")

        # Prepare multimodal inputs (capping max_frames=16 to prevent CUDA OOM)
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

        # Find visual tokens
        video_grid_thw = batch.get("video_grid_thw")
        image_grid_thw = batch.get("image_grid_thw")
        vis_positions = []
        if video_grid_thw is not None or image_grid_thw is not None:
            v_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            v_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
            in_vis = False
            for i, tid in enumerate(prompt_ids):
                if tid == v_start:
                    in_vis = True
                elif tid == v_end:
                    in_vis = False
                elif in_vis:
                    vis_positions.append(i)

        # 1. Run Autoregressive Target Baseline once per sample
        print("  Running Autoregressive Baseline...")
        target_ids, ar_prefill_s, ar_decode_s = engine.run_autoregressive(batch, max_new_tokens=max_new_tokens)
        target_text = processor.tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        target_hash = sha256_tokens(target_ids)
        ar_e2e_s = ar_prefill_s + ar_decode_s

        # 2. Run Speculative Decoding across the 4 retention percentages
        for ret in retention_percentages:
            row_id = f"{sample_id}:native:uniform:retention-{int(ret)}"
            if row_id in existing_rows:
                continue

            spec_res = engine.run_speculative(
                batch=batch,
                retention_percentage=ret,
                visual_positions_in_prompt=vis_positions,
                max_new_tokens=max_new_tokens,
                target_ground_truth_ids=target_ids,
            )

            spec_ids = spec_res["speculative_output_ids"]
            spec_text = processor.tokenizer.decode(spec_ids, skip_special_tokens=True).strip()
            spec_hash = sha256_tokens(spec_ids)

            dec_speedup = ar_decode_s / max(spec_res["decode_seconds"], 1e-6)
            e2e_speedup = ar_e2e_s / max(spec_res["end_to_end_seconds"], 1e-6)

            peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

            row = {
                "row_id": row_id,
                "sample_id": sample_id,
                "condition": "retention",
                "paper_figure": "Figure 1(b)",
                "series_id": "vdc_uniform",
                "selection_policy": "uniform",
                "retention_percentage": float(ret),
                "target_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "draft_depth": depth,
                "prompt_question": question,
                "prompt_input_ids": prompt_ids,
                "prompt_input_text": rendered,
                "prompt_variant": "natural",
                "target_output_ids": target_ids,
                "target_output_text": target_text,
                "target_output_hash": target_hash,
                "speculative_output_ids": spec_ids,
                "speculative_output_text": spec_text,
                "speculative_output_hash": spec_hash,
                "lossless": spec_res["lossless"],
                "lossless_prefix_length": spec_res["lossless_prefix_length"],
                "target_visual_tokens": len(vis_positions),
                "actual_visual_tokens": spec_res.get("actual_visual_tokens", len(vis_positions)),
                "video_token_count": len(vis_positions),
                "full_target_visual_tokens": len(vis_positions),
                "max_frames": 16,
                "max_pixels": 256 * 28 * 28,
                "fps": 1.0,
                "temperature": 0.0,
                "visual_value_mode": "real",
                "prefill_seconds": spec_res["prefill_seconds"],
                "decode_seconds": spec_res["decode_seconds"],
                "verification_seconds": spec_res["verification_seconds"],
                "speculative_seconds": spec_res["speculative_seconds"],
                "end_to_end_seconds": spec_res["end_to_end_seconds"],
                "ar_prefill_seconds": ar_prefill_s,
                "ar_decode_seconds": ar_decode_s,
                "ar_end_to_end_seconds": ar_e2e_s,
                "decode_speedup": dec_speedup,
                "end_to_end_speedup": e2e_speedup,
                "acceptance_rounds": spec_res["acceptance_rounds"],
                "acceptance_trace": spec_res["acceptance_trace"],
                "acceptance_by_position": spec_res["acceptance_by_position"],
                "acceptance_position_records": spec_res["acceptance_by_position"],
                "accepted_prefix_tokens": spec_res["accepted_prefix_tokens"],
                "verification_steps": spec_res["verification_steps"],
                "peak_memory_allocated_bytes": peak_mem,
            }

            # Append to file immediately
            with open(jsonl_output, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing_rows.add(row_id)

            print(
                f"    Retention {int(ret):>3}% -> tau={spec_res['accepted_prefix_tokens']:.2f}, "
                f"decode_speedup={dec_speedup:.2f}x, lossless={'yes' if spec_res['lossless'] else 'no'}"
            )

        torch.cuda.empty_cache()

    # Extract iterations detailed CSV
    csv_output = output_dir / "draft_iterations_detailed.csv"
    print(f"\nExtracting draft iterations to {csv_output}...")
    import subprocess
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "extract_draft_iterations.py"),
        "--input", str(jsonl_output),
        "--output", str(csv_output),
    ]
    subprocess.run(cmd, check=True)
    print(f"Done extraction for {dataset_name} Depth {depth}!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--depth", type=int, choices=[3, 5], required=True)
    parser.add_argument("--dataset", choices=["vidhalluc", "videochair", "vdc", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-base", type=Path, default=Path("results/msd_eval_qwen25vl3b"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    target_path = "/path/to/local-resource"
    ckpt_path = f"dataset/qwen25vl-3b-msd-checkpoints/qwen25vl-3b-msd-depth{args.depth}-latest/training_state.pt"

    datasets = ["vidhalluc", "videochair", "vdc"] if args.dataset == "all" else [args.dataset]

    depth_out_dir = args.output_base / f"depth{args.depth}"

    for ds in datasets:
        if ds == "vidhalluc":
            manifest = "dataset/VidHalluc/test.jsonl"
            root = "dataset/VidHalluc"
            max_new = 48
        elif ds == "videochair":
            manifest = "dataset/VideoChair/test.jsonl"
            root = "dataset/VideoChair"
            max_new = 128
        elif ds == "vdc":
            manifest = "dataset/VideoDetailCaption/test_100.jsonl"
            root = "dataset/VideoDetailCaption"
            max_new = 128

        out_dir = depth_out_dir / ds
        run_benchmark_for_config(
            gpu_id=args.gpu_id,
            depth=args.depth,
            dataset_name=ds.capitalize(),
            manifest_path=manifest,
            dataset_root=root,
            output_dir=out_dir,
            target_model_path=target_path,
            checkpoint_path=ckpt_path,
            limit=args.limit,
            max_new_tokens=max_new,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
