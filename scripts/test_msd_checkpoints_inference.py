#!/usr/bin/env python3
"""Evaluate MSD Depth 3 and Depth 5 checkpoints on multimodal video and text samples."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parents[1]
specforge_root = repo_root / "src" / "train_Dflash_SpecForge"
for p in (repo_root, specforge_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from specforge.export.checkpoint_io import materialize_draft, resolve_training_state
from specforge.modeling.draft.msd import MSDConfig


def load_models(
    target_path: str,
    ckpt_depth3_dir: str,
    ckpt_depth5_dir: str,
    device: torch.device,
):
    print("=" * 70)
    print(f"Loading Base Target Model: {target_path}")
    print("=" * 70)
    processor = AutoProcessor.from_pretrained(target_path)
    target = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        target_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    ).eval()

    base_cfg_path = specforge_root / "configs" / "qwen2.5-vl-3b-msd.json"
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))

    drafts = {}
    for depth, ckpt_dir in [(3, ckpt_depth3_dir), (5, ckpt_depth5_dir)]:
        cfg = dict(base_cfg)
        cfg["num_hidden_layers"] = depth
        tmp_cfg_path = f"/tmp/qwen25vl_msd_depth{depth}_eval.json"
        with open(tmp_cfg_path, "w") as f:
            json.dump(cfg, f)

        state = resolve_training_state(f"{ckpt_dir}/training_state.pt")
        print(
            f"Loading MSD Depth {depth}: global_step={state.get('global_step')}, "
            f"epoch={state.get('epoch')}, strategy={state.get('strategy')}"
        )
        draft = materialize_draft(state, tmp_cfg_path)
        draft.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight.data)
        draft.to(device=device, dtype=torch.bfloat16).eval()
        drafts[depth] = draft
        print(f"-> MSD Depth {depth} loaded successfully ({len(draft.layers)} layers).")

    return processor, target, drafts


def evaluate_sample(
    sample_name: str,
    prompt_text: str,
    target: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    drafts: dict[int, torch.nn.Module],
    device: torch.device,
    video_path: str | None = None,
    max_new_tokens: int = 40,
):
    print("\n" + "#" * 70)
    print(f"SAMPLE: {sample_name}")
    print(f"Type: {'Multimodal (Video + Text)' if video_path else 'Text-only'}")
    print(f"Question: {prompt_text}")
    if video_path:
        print(f"Video file: {video_path}")
    print("#" * 70)

    # 1. Format prompt
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
                    },
                    {"type": "text", "text": prompt_text},
                ]
                if video_path
                else [{"type": "text", "text": prompt_text}]
            ),
        }
    ]

    from qwen_vl_utils import process_vision_info

    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    proc_kwargs = {
        "text": [rendered],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        proc_kwargs["images"] = image_inputs
    if video_inputs:
        proc_kwargs["videos"] = video_inputs
        proc_kwargs.update(dict(video_kwargs or {}))

    batch = processor(**proc_kwargs)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    input_ids = batch["input_ids"]
    prompt_len = input_ids.shape[1]

    # 2. Greedy target generation
    t0 = time.perf_counter()
    with torch.no_grad():
        output = target.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    gen_time = time.perf_counter() - t0
    gen_ids = output[0, prompt_len:]
    response_text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"\n[Target Output] ({len(gen_ids)} tokens in {gen_time:.2f}s):")
    print(f'"{response_text}"')

    # 3. Teacher trajectory forward pass
    full_ids = output[:, : prompt_len + len(gen_ids)]
    full_batch = dict(batch)
    full_batch["input_ids"] = full_ids
    if "attention_mask" in full_batch:
        full_batch["attention_mask"] = torch.ones_like(full_ids, dtype=torch.bool)

    image_grid_thw = full_batch.get("image_grid_thw")
    video_grid_thw = full_batch.get("video_grid_thw")
    position_ids, _ = target.get_rope_index(
        input_ids=full_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=full_batch.get("attention_mask"),
    )
    full_batch["position_ids"] = position_ids

    with torch.no_grad():
        target_outputs = target(**full_batch, output_hidden_states=True)
    target_hidden = target_outputs.hidden_states[-1]
    input_embeddings = target.model.embed_tokens(full_ids)

    seq_len = full_ids.shape[1]
    visual_token_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    if video_grid_thw is not None or image_grid_thw is not None:
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        in_vis = False
        for i, tid in enumerate(full_ids[0].tolist()):
            if tid == vision_start_id:
                in_vis = True
            elif tid == vision_end_id:
                in_vis = False
            elif in_vis:
                visual_token_mask[0, i] = True

    # 4. MSD Draft Predictions
    # Assistant tokens start at prompt_len - 1 (predicting the first generated token) up to seq_len - 2
    eval_start = prompt_len - 1
    eval_end = seq_len - 1
    num_generated_eval = eval_end - eval_start

    print(f"\n[Draft Accuracy Evaluation over {num_generated_eval} generated positions]:")
    print("-" * 70)

    # Prepare sequence for draft
    cond_h = target_hidden[:, :-1]
    next_emb = input_embeddings[:, 1:]
    vis_mask = visual_token_mask[:, 1:]
    pos_id = position_ids[:, :, 1:]

    sample_metrics = {}
    for depth in (3, 5):
        draft = drafts[depth]
        with torch.no_grad():
            pred_h = draft(
                conditioning_hidden_state=cond_h,
                next_token_embeddings=next_emb,
                visual_embeddings=next_emb,
                visual_token_mask=vis_mask,
                position_ids=pos_id,
            )

        # Focus on generated assistant token positions:
        # At position t, draft predicts target token t+1 (which is full_ids[0, t+1])
        pred_h_gen = pred_h[:, eval_start:eval_end]
        true_h_gen = target_hidden[:, eval_start + 1 : eval_end + 1]

        cos_sim = F.cosine_similarity(pred_h_gen.float(), true_h_gen.float(), dim=-1).mean().item()

        # Logits projection
        pred_logits = target.lm_head(pred_h_gen)
        pred_tokens = pred_logits.argmax(dim=-1)[0]
        true_tokens = full_ids[0, eval_start + 1 : eval_end + 1]

        # Top-1 accuracy
        top1_match = (pred_tokens == true_tokens).float()
        top1_acc = top1_match.mean().item() * 100

        # Top-5 accuracy
        top5_indices = pred_logits.topk(5, dim=-1).indices[0]
        top5_match = (top5_indices == true_tokens.unsqueeze(-1)).any(dim=-1).float()
        top5_acc = top5_match.mean().item() * 100

        pred_text = processor.tokenizer.decode(pred_tokens, skip_special_tokens=True)

        sample_metrics[depth] = {
            "cosine_similarity": cos_sim,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "correct_tokens": int(top1_match.sum().item()),
            "total_tokens": len(true_tokens),
            "predicted_text": pred_text,
        }

        print(f"MSD Depth {depth}:")
        print(f"  * Hidden Cosine Similarity: {cos_sim:.4f}")
        print(f"  * Next-Token Top-1 Match:   {top1_acc:.1f}% ({int(top1_match.sum().item())}/{len(true_tokens)} tokens)")
        print(f"  * Next-Token Top-5 Match:   {top5_acc:.1f}%")
        print(f"  * Draft Predicted String:   \"{pred_text.strip()}\"")
        print("-" * 70)

    # 5. Speculative 2-token proposal simulation
    print("\n[Speculative Simulation (Lookahead = 2 tokens)]:")
    for depth in (3, 5):
        accepted_tokens = 0
        total_proposals = 0
        rounds = 0
        pos = eval_start
        while pos < eval_end - 1:
            rounds += 1
            # Proposal 1
            cand1 = target.lm_head(pred_h[:, pos : pos + 1]).argmax(dim=-1)[0, 0].item()
            true1 = full_ids[0, pos + 1].item()
            total_proposals += 1
            if cand1 == true1:
                accepted_tokens += 1
                # Proposal 2
                if pos + 1 < eval_end:
                    cand2 = target.lm_head(pred_h[:, pos + 1 : pos + 2]).argmax(dim=-1)[0, 0].item()
                    true2 = full_ids[0, pos + 2].item()
                    total_proposals += 1
                    if cand2 == true2:
                        accepted_tokens += 1
                        pos += 2
                        continue
            pos += 1
        acc_per_round = (accepted_tokens / rounds) if rounds > 0 else 0
        print(
            f"  MSD Depth {depth}: Accepted {accepted_tokens}/{total_proposals} proposals "
            f"across {rounds} verification rounds (Avg {acc_per_round:.2f} tokens/round)"
        )

    # Clean cache
    torch.cuda.empty_cache()
    return sample_metrics


def main():
    target_model_path = "/path/to/local-resource"
    ckpt_depth3_dir = "dataset/qwen25vl-3b-msd-checkpoints/qwen25vl-3b-msd-depth3-latest"
    ckpt_depth5_dir = "dataset/qwen25vl-3b-msd-checkpoints/qwen25vl-3b-msd-depth5-latest"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor, target, drafts = load_models(
        target_model_path, ckpt_depth3_dir, ckpt_depth5_dir, device
    )

    # Test 1: Video QA 1
    evaluate_sample(
        sample_name="VidHalluc #1 (-Hav5URxl74)",
        prompt_text="Is the primary action in the video swirling a toothbrush in water? Answer concisely.",
        target=target,
        processor=processor,
        drafts=drafts,
        device=device,
        video_path="dataset/VidHalluc/Test_Videos/-Hav5URxl74.mp4",
        max_new_tokens=30,
    )

    # Test 2: Video QA 2
    evaluate_sample(
        sample_name="VidHalluc #2 (-ORd4EBliqw_clip_1)",
        prompt_text="Is frying the fish the predominant action captured in the video? Answer concisely.",
        target=target,
        processor=processor,
        drafts=drafts,
        device=device,
        video_path="dataset/VidHalluc/Test_Videos/-ORd4EBliqw_clip_1.mp4",
        max_new_tokens=30,
    )

    # Test 3: Pure text prompt
    evaluate_sample(
        sample_name="Text Prompt (Physics Explanation)",
        prompt_text="Explain what specular reflection is in physics concisely in 1-2 sentences.",
        target=target,
        processor=processor,
        drafts=drafts,
        device=device,
        video_path=None,
        max_new_tokens=35,
    )

    print("\n" + "=" * 70)
    print("ALL EVALUATION SAMPLES FINISHED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
