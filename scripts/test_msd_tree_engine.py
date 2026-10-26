import time
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from eagle.model.cnets import Model
from eagle.model.configs import Qwen2VLConfig
from eagle.model.utils import evaluate_posterior
from qwen_vl_utils import process_vision_info


target_path = "/path/to/local-resource"
checkpoint_path = "dataset/qwen25vl-3b-msd-checkpoints/qwen25vl-3b-msd-depth3-latest/training_state.pt"

print("Loading Target Model...")
processor = AutoProcessor.from_pretrained(target_path)
target = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    target_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
).eval()

print("Loading and mapping MSD Depth 3 checkpoint into cnets.Model...")
state = torch.load(checkpoint_path, map_location="cpu")
sd = state["draft_state_dict"]
mapped_sd = {k.replace("fusion_projection.", "fc.") if k.startswith("fusion_projection.") else k: v for k, v in sd.items()}

config = Qwen2VLConfig(
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=3,
    num_attention_heads=16,
    num_key_value_heads=2,
    vocab_size=151936,
    rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]}
)

# Set tree structure: total_tokens=30, depth=4, top_k=8
total_tokens = 30
draft = Model(config, bias=True, total_tokens=total_tokens, depth=4, top_k=8)
draft.load_state_dict(mapped_sd, strict=False)
draft.embed_tokens.weight.data.copy_(target.get_input_embeddings().weight.data)
draft.to(device="cuda:0", dtype=torch.bfloat16).eval()
draft.init_tree()

from src.infer.qwen25vl_ddtree import compact_dynamic_cache
with open("dataset/VideoDetailCaption/test.jsonl") as f:
    item = json.loads(f.readline())

video_path = "dataset/VideoDetailCaption/Test_Videos/" + item.get("video_name", "") + ".mp4"
prompt_text = item.get("question") or "Describe this video in detail."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 256 * 28 * 28, "fps": 1.0, "max_frames": 16},
            {"type": "text", "text": prompt_text}
        ]
    }
]

rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
proc_kwargs = {"text": [rendered], "padding": True, "return_tensors": "pt"}
if video_inputs:
    proc_kwargs["videos"] = video_inputs
    proc_kwargs.update(dict(video_kwargs or {}))

batch = processor(**proc_kwargs)
batch = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in batch.items()}

# Autoregressive baseline
print("\nRunning AR baseline...")
t_ar0 = time.perf_counter()
with torch.no_grad():
    ar_out = target.generate(**batch, max_new_tokens=64, do_sample=False)
t_ar = time.perf_counter() - t_ar0
prompt_len = batch["input_ids"].shape[1]
ar_tokens = ar_out[0, prompt_len:].tolist()
print(f"AR Baseline generated {len(ar_tokens)} tokens in {t_ar:.2f}s: {repr(processor.tokenizer.decode(ar_tokens[:30]))}...")

# Test Tree Speculative Decoding
print("\nTesting MSD Tree Speculative Decoding...")
t_sd0 = time.perf_counter()
with torch.no_grad():
    # 1. Target Prefill
    out = target(**batch, output_hidden_states=True, use_cache=True)
    past_key_values = out.past_key_values
    hidden_states = out.hidden_states[-1]
    t0 = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    
    input_ids = torch.cat((batch["input_ids"], t0), dim=1)
    generated_tokens = [t0.item()]
    
    # First draft tree
    draft.reset()
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft.topK_genrate(
        hidden_states, input_ids, target.lm_head, None
    )
    
    steps = 0
    total_accepted = 0
    stop_ids = {processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")}
    
    while len(generated_tokens) < 64:
        steps += 1
        S = past_key_values.get_seq_length()
        M = draft_tokens.shape[1]
        
        # Build 4D additive attention mask
        additive_mask = torch.full((1, 1, M, S + M), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device="cuda:0")
        additive_mask[..., :S] = 0.0
        additive_mask[..., S:S+M].masked_fill_(tree_mask.to("cuda:0") == 1, 0.0)
        
        pos_ids = (S + tree_position_ids.to("cuda:0")).unsqueeze(0).unsqueeze(0).expand(3, 1, M)
        
        v_out = target(
            input_ids=draft_tokens,
            position_ids=pos_ids,
            attention_mask=additive_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        
        tree_logits = v_out.logits[0, retrieve_indices]
        padding = (torch.zeros(1, 1, dtype=torch.long, device="cuda:0") - 1)
        draft_tokens_pad = torch.cat((draft_tokens, padding), dim=1)
        candidates = draft_tokens_pad[0, retrieve_indices]
        
        best_candidate, accept_length, sample_p = evaluate_posterior(tree_logits, candidates, None)
        accept_len = accept_length.item()
        total_accepted += accept_len
        print('best_candidate index:', best_candidate.item())
        print('candidates[best_candidate]:', [processor.tokenizer.decode([candidates[best_candidate, c].item()]) if candidates[best_candidate, c] >= 0 else '<pad>' for c in range(candidates.shape[1])])
        print('logits argmax along best candidate:', [processor.tokenizer.decode([tree_logits[best_candidate, c].argmax().item()]) for c in range(tree_logits.shape[1])])
        print('sample_p argmax:', processor.tokenizer.decode([sample_p.argmax().item()]))
        top5_sp = torch.topk(sample_p, 5).indices.tolist()
        print('sample_p top 5:', [processor.tokenizer.decode([i]) for i in top5_sp])
        
        # Accepted tokens from best candidate
        for j in range(accept_len):
            tok = candidates[best_candidate, j + 1].item()
            generated_tokens.append(tok)
            if tok in stop_ids:
                break
        if generated_tokens[-1] in stop_ids:
            break
            
        # Bonus fallback token
        bonus_token = torch.argmax(sample_p).item()
        generated_tokens.append(bonus_token)
        if bonus_token in stop_ids:
            break
            
        # Crop KV cache to accepted branch
        best_indices = retrieve_indices[best_candidate, : accept_len + 1].tolist()
        compact_dynamic_cache(past_key_values, S, best_indices)
        
        # Prepare hidden state and input_ids for next tree generation
        accept_hidden = v_out.hidden_states[-1][:, retrieve_indices[best_candidate, : accept_len + 1]]
        token_tensor = torch.tensor([[bonus_token]], device="cuda:0")
        if accept_len > 0:
            input_ids = torch.cat((input_ids, candidates[best_candidate, 1 : accept_len + 1].unsqueeze(0), token_tensor), dim=1)
        else:
            input_ids = torch.cat((input_ids, token_tensor), dim=1)
        
        # Generate next tree
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft.topK_genrate(
            accept_hidden, input_ids, target.lm_head, None
        )

t_sd = time.perf_counter() - t_sd0
print(f"Tree Speculative Decoding generated {len(generated_tokens)} tokens in {t_sd:.2f}s ({steps} steps)!")
print(f"Total accepted tokens: {total_accepted} over {steps} steps -> tau = {1.0 + total_accepted / steps:.2f} tokens/step!")
print(f"Speedup: {t_ar / t_sd:.2f}x")
print("Generated text:", repr(processor.tokenizer.decode(generated_tokens[:30])))
