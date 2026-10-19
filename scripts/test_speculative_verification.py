import json
import time
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from specforge.export.checkpoint_io import materialize_draft, resolve_training_state
from qwen_vl_utils import process_vision_info

target_model_path = "/path/to/local-resource"
ckpt_path = "dataset/qwen25vl-3b-msd-checkpoints/qwen25vl-3b-msd-depth5-latest/training_state.pt"
device = torch.device("cuda:0")

processor = AutoProcessor.from_pretrained(target_model_path)
target = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    target_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
    attn_implementation="sdpa",
).eval()

base_cfg_path = Path("src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-msd.json")
base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
base_cfg["num_hidden_layers"] = 5
tmp_cfg_path = "/tmp/test_depth5.json"
with open(tmp_cfg_path, "w") as f:
    json.dump(base_cfg, f)

state = resolve_training_state(ckpt_path)
draft = materialize_draft(state, tmp_cfg_path)
draft.embed_tokens.weight.data.copy_(target.model.embed_tokens.weight.data)
draft.to(device=device, dtype=torch.bfloat16).eval()

messages = [{
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": "dataset/VidHalluc/Test_Videos/-Hav5URxl74.mp4",
            "max_pixels": 256 * 28 * 28,
            "fps": 1.0,
            "max_frames": 16,
        },
        {"type": "text", "text": "Is there a dog playing in the grass?"}
    ]
}]

rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
img, vid, vid_kw = process_vision_info(messages, return_video_kwargs=True)
batch = processor(text=[rendered], videos=vid, padding=True, return_tensors="pt", **(vid_kw or {}))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

# 1. AR baseline
with torch.no_grad():
    ar_out = target.generate(**batch, max_new_tokens=40, do_sample=False, repetition_penalty=1.0)
prompt_len = batch["input_ids"].shape[1]
ar_tokens = ar_out[0, prompt_len:].tolist()
print("AR tokens:", ar_tokens)
print("AR text:", repr(processor.tokenizer.decode(ar_tokens)))

# 2. Speculative decoding test
pos_ids, rope_deltas = target.get_rope_index(
    batch["input_ids"],
    image_grid_thw=batch.get("image_grid_thw"),
    video_grid_thw=batch.get("video_grid_thw"),
    attention_mask=batch.get("attention_mask"),
)
delta = rope_deltas[0, 0].item()

def crop_kv(pkv, target_len):
    if hasattr(pkv, "crop"):
        pkv.crop(target_len)
        return pkv
    if hasattr(pkv, "key_cache"):
        for i in range(len(pkv.key_cache)):
            pkv.key_cache[i] = pkv.key_cache[i][..., :target_len, :]
            pkv.value_cache[i] = pkv.value_cache[i][..., :target_len, :]
        if hasattr(pkv, "_seen_tokens"):
            pkv._seen_tokens = target_len
        return pkv
    return pkv

with torch.no_grad():
    out = target(**batch, position_ids=pos_ids, use_cache=True, output_hidden_states=True)
    past_key_values = out.past_key_values
    target_hidden = out.hidden_states[-1]
    t0 = out.logits[:, -1, :].argmax(dim=-1).item()

# Now target KV cache has prompt (length = prompt_len).
# We feed t0 into target to put t0 into target KV cache and get h(t0) and target's prediction for token 1
p0 = prompt_len + delta
p_ids_0 = torch.tensor([[[p0]]] * 3, device=device)
with torch.no_grad():
    t0_out = target(input_ids=torch.tensor([[t0]], device=device), position_ids=p_ids_0, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
past_key_values = t0_out.past_key_values
curr_h = t0_out.hidden_states[-1][:, -1:, :]
curr_tok = t0
kv_len = prompt_len + 1
spec_tokens = [t0]
stop_tokens = {processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}
prev_target_pred = t0_out.logits[:, -1, :].argmax(dim=-1).item()

rounds = 0
accepted_proposals = 0
trace = []

with torch.no_grad():
    while len(spec_tokens) < 40 and curr_tok not in stop_tokens:
        rounds += 1
        # Draft proposes c1
        tok_emb = target.model.embed_tokens(torch.tensor([[curr_tok]], device=device))
        vis_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        d_pids_1 = torch.tensor([[[kv_len - 1 + delta]]] * 3, device=device)
        d_h1 = draft(
            conditioning_hidden_state=curr_h,
            next_token_embeddings=tok_emb,
            visual_embeddings=tok_emb,
            visual_token_mask=vis_mask,
            position_ids=d_pids_1,
        )
        c1 = target.lm_head(d_h1).argmax(dim=-1)[0, 0].item()

        # Draft proposes c2
        tok_emb2 = target.model.embed_tokens(torch.tensor([[c1]], device=device))
        d_pids_2 = torch.tensor([[[kv_len + delta]]] * 3, device=device)
        d_h2 = draft(
            conditioning_hidden_state=d_h1,
            next_token_embeddings=tok_emb2,
            visual_embeddings=tok_emb2,
            visual_token_mask=vis_mask,
            position_ids=d_pids_2,
        )
        c2 = target.lm_head(d_h2).argmax(dim=-1)[0, 0].item()

        # Target verifies [c1, c2]
        # Target KV cache currently has kv_len tokens.
        # verify_inputs has shape [1, 2]
        v_pids = torch.tensor([[[kv_len + delta, kv_len + 1 + delta]]] * 3, device=device)
        v_inputs = torch.tensor([[c1, c2]], device=device)
        v_out = target(input_ids=v_inputs, position_ids=v_pids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
        v_logits = v_out.logits
        v_hidden = v_out.hidden_states[-1]

        target_pred_after_curr = prev_target_pred
        target_pred_after_c1 = v_logits[0, 0].argmax(dim=-1).item()
        target_pred_after_c2 = v_logits[0, 1].argmax(dim=-1).item()

        round_accepted = 0
        if c1 == target_pred_after_curr:
            # c1 accepted!
            spec_tokens.append(c1)
            round_accepted += 1
            if c2 == target_pred_after_c1:
                # c2 accepted!
                spec_tokens.append(c2)
                round_accepted += 1
                # Bonus token
                spec_tokens.append(target_pred_after_c2)
                curr_tok = target_pred_after_c2
                # Target KV cache currently has kv_len + 2 (up to c2).
                # Feed bonus token into target to keep KV in sync
                b_pids = torch.tensor([[[kv_len + 2 + delta]]] * 3, device=device)
                b_out = target(input_ids=torch.tensor([[curr_tok]], device=device), position_ids=b_pids, past_key_values=v_out.past_key_values, use_cache=True, output_hidden_states=True)
                past_key_values = b_out.past_key_values
                curr_h = b_out.hidden_states[-1][:, -1:, :]
                prev_target_pred = b_out.logits[:, -1, :].argmax(dim=-1).item()
                kv_len = kv_len + 3
            else:
                # c1 accepted, c2 rejected. Emitted fallback is target_pred_after_c1
                fallback_tok = target_pred_after_c1
                spec_tokens.append(fallback_tok)
                curr_tok = fallback_tok
                # Crop KV cache to keep up to c1 (kv_len + 1)
                pkv_cropped = crop_kv(v_out.past_key_values, kv_len + 1)
                # Feed fallback token into target
                fb_pids = torch.tensor([[[kv_len + 1 + delta]]] * 3, device=device)
                fb_out = target(input_ids=torch.tensor([[curr_tok]], device=device), position_ids=fb_pids, past_key_values=pkv_cropped, use_cache=True, output_hidden_states=True)
                past_key_values = fb_out.past_key_values
                curr_h = fb_out.hidden_states[-1][:, -1:, :]
                prev_target_pred = fb_out.logits[:, -1, :].argmax(dim=-1).item()
                kv_len = kv_len + 2
        else:
            # c1 rejected. Emitted fallback is target_pred_after_curr
            fallback_tok = target_pred_after_curr
            spec_tokens.append(fallback_tok)
            curr_tok = fallback_tok
            # Crop KV cache back to kv_len (discarding [c1, c2])
            pkv_cropped = crop_kv(v_out.past_key_values, kv_len)
            # Feed fallback token into target
            fb_pids = torch.tensor([[[kv_len + delta]]] * 3, device=device)
            fb_out = target(input_ids=torch.tensor([[curr_tok]], device=device), position_ids=fb_pids, past_key_values=pkv_cropped, use_cache=True, output_hidden_states=True)
            past_key_values = fb_out.past_key_values
            curr_h = fb_out.hidden_states[-1][:, -1:, :]
            prev_target_pred = fb_out.logits[:, -1, :].argmax(dim=-1).item()
            kv_len = kv_len + 1

        accepted_proposals += round_accepted
        trace.append(round_accepted)

print("Speculative tokens:", spec_tokens[:len(ar_tokens)])
print("AR tokens:         ", ar_tokens)
print("Exact Match:       ", spec_tokens[:len(ar_tokens)] == ar_tokens)
print(f"Rounds: {rounds}, Accepted proposals: {accepted_proposals}, tau: {accepted_proposals/rounds:.2f}")
