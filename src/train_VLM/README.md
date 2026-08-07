# DFlash training for Qwen2.5-VL

`src/train_VLM` implements the training objective described in Sections 4.1–4.2
and Appendix A.1 of `externals/dflash/DFlash.pdf`. It is a block drafter, not a
multi-timestep diffusion model: a block contains one clean target anchor and
`block_size - 1` mask tokens, and all masked positions are predicted in one
forward pass.

The target VLM is frozen and evaluated under `torch.inference_mode()`. Only the selected-hidden-state projection, draft
decoder layers and draft norms are saved in the checkpoint. Set
`context_mode` to `full` or `text_only` and train separate checkpoints for the
two representations.

The supplied configs correspond to those two checkpoints:
`config_qwen25vl_7b.json` uses full multimodal context and
`config_qwen25vl_7b_text_only.json` removes image/video token features from the
draft context.

## Real-data two-stage pipeline for one RTX 3090

The default offline pipeline uses frozen `Qwen/Qwen2.5-VL-3B-Instruct` and the
same vanilla DFlash model defined in `model.py`. Prompts, conversations, and
media always come from the selected real source records; no dummy or synthetic
training records are substituted:

- Stage 1 reads `ShareGPT_V3_unfiltered_cleaned_split.json` from
  `anon8231489123/ShareGPT_Vicuna_unfiltered`.
- Stage 2 reads `blip_laion_cc_sbu_558k.json` and its corresponding official
  images from `liuhaotian/LLaVA-Pretrain`.

Each stage has three explicit entrypoints:

```bash
python -m src.train_VLM.prepare_data --config CONFIG.json
python -m src.train_VLM.cache_teacher_features --config CONFIG.json
python -m src.train_VLM.train_draft --config CONFIG.json
```

`prepare_data` streams the real JSON array with `ijson`, validates the original
ShareGPT/LLaVA conversation schema, assigns every valid source record a stable
SHA-256 shuffle score derived from `seed`, and selects the first `max_samples`.
The adjacent `manifest.jsonl.meta.json` records the resolved Hub commit,
annotation SHA-256, split, and every exact selected source ID/index. A local
annotation can be supplied with `--data-path`; local extracted images or a
local ZIP can be supplied with `--image-root` and `--image-archive`.

The official LLaVA image archive is about 27 GB and stores all images in one
ZIP. For a subset run, the default code opens that remote ZIP as a seekable HTTP
range file and extracts only the selected image members. The 32-record smoke
run therefore materializes tens of images rather than downloading the complete
558K archive. Pass `--no-selective-image-download` to require local media.

`cache_teacher_features` is the only training step that loads the full target.
Stage 1 uses the final assistant response already present in each multi-turn
ShareGPT record (`teacher_response_mode=dataset`); it does not append an
instruction or ask the target VLM to regenerate that response. The full Stage 1
presets set `max_seq_length=2048` and `response_max_new_tokens=0`. Stage 2 may
independently use raw-greedy target generation; a generated response that
reaches its token/sequence budget is retained as an exact prefix and marked
truncated. Enable `teacher_require_eos` only when complete generated responses
are mandatory. The cache contains clean token labels, three-axis positions,
the configured selected hidden layers, context positions, truncation decisions,
and source provenance in safetensors shards. It also exports only the frozen
token embedding/LM-head matrix required by vanilla DFlash. `train_draft` then
loads the shards, frozen token I/O, and draft model; the Qwen transformer and
vision encoder are absent from GPU memory. Checkpoints contain only draft and
projection weights plus optimizer/scheduler, per-rank RNG, exact sampler
permutation and in-epoch progress for resume.

Run the complete real-data smoke path, including both stages and final decode
on a real TorchVision MP4, with:

```bash
src/train_VLM/train_3090_smoke.sh
```

The supplied smoke configs select 32 real records per source, train 24 optimizer
steps per stage, assert fixed-subset loss decreases, verify gradients exist only
for draft/projection parameters, save/reload bit-exactly, and initialize Stage 2
from the Stage 1 checkpoint. The `train_3090_small.sh` profile uses 2,048 real
records per stage. Scale the same implementation to 4K or 68K by overriding
`--max-samples`; budget teacher-cache disk space in proportion to sequence
length because selected hidden features are intentionally stored losslessly.

All important values are accepted from JSON/YAML config and as CLI overrides:
target/dataset repo and revision, annotation/image paths, split, sample count,
seed, epochs, LR, micro batch, gradient accumulation, dtype, maximum sequence
length, block/anchor sizes, draft depth, feature count and exact target layer
IDs, context mode, teacher response mode/generation length/EOS policy,
shard/cache/output paths, weights-only `--checkpoint`, full
`--resume`, and device. Existing outputs are not replaced unless
`--overwrite` is explicit. Launch scripts reuse completed stage artifacts; set
`VIDEO_DFLASH_OVERWRITE=1` to rebuild them from the source manifests.

## Full offline pipeline for 8×NVIDIA B200

The canonical copy-paste workflow is documented in the repository-root
`README.md`. The two full presets select 68K real records per stage and use
BF16 distributed teacher caching/training:

- `config_b200_8gpu_stage1_sharegpt.json`
- `config_b200_8gpu_stage2_llava.json`

Launch both caching and training with `torchrun --standalone
--nproc_per_node=8`. Training partitions each shuffled global batch without
padding/repeating the last batch and manually all-reduces weighted gradients.
Every completed optimizer step is durably committed to an atomic recovery
checkpoint before `checkpoints/latest` advances. Long-lived snapshots default
to every 0.5 epoch. Resume restores model, AdamW, cosine scheduler, global
step, exact sample offset/permutation and Python/NumPy/Torch/CUDA RNG state for
each rank; changes to the mathematical training contract are rejected.

Stage 2's `checkpoint` field points directly to the final Stage 1 export. A
Stage 2 `--resume auto`, by contrast, restores Stage 2's own full training
state. Use `python -m src.train_VLM.infer_video` for one-pass greedy
Video-DFlash generation on a real MP4; it reads the architecture contract from
the final Stage 2 checkpoint and prints decoded text plus acceptance/timing
metrics.

## Minimal Video-DFlash smoke test

The video path is vanilla DFlash: Qwen2.5-VL processes the video once during
target prefill, and later iterations contain text token IDs, M-RoPE positions,
and the target KV cache only. Each iteration drafts a flat block, verifies it
with one parallel target forward, crops rejected cache entries, and continues
from the target bonus token. No Sparrow attention, glimpsing, tree decoding, or
video-token selector is used.

After installing the repository-root `requirements.txt` as described in
`README.md`, run the self-contained 3090 smoke test:

```bash
python -m src.train_VLM.smoke_video
```

By default it creates a temporary 8-frame 112×112 MP4, loads
`Qwen/Qwen2.5-VL-3B-Instruct` in BF16 with SDPA, and decodes at most 24 tokens
with batch size one. It prints the realized frame/grid/visual-token counts,
proposals and acceptances, target cache length/shape, visual-prefill count,
peak VRAM, and exact equality against raw greedy target AR. A smaller explicit
run is:

```bash
python -m src.train_VLM.smoke_video \
  --num-frames 4 --size 112 --max-new-tokens 16 --block-size 4
```

Use `--video /path/to/clip.mp4` for a local sample. Frame count, pixel budget,
reader, target attention implementation, block size, draft depth, and target
feature count are CLI parameters and also have corresponding config fields:
`video_num_frames`, `video_min_pixels`, `video_max_pixels`, `video_reader`, and
`target_attn_implementation`. Per-video `nframes`/`fps` and pixel values in a
manifest override those defaults. The current lossless decoder deliberately
supports raw greedy decoding only; it does not reproduce sampling or Qwen's
checkpoint-default repetition penalty.

When `--checkpoint` is supplied, `--video` is mandatory so a trained pipeline
cannot silently fall back to the generated standalone decoder fixture.

## Data flow

1. Prepare a JSONL manifest whose records contain a stable `id` and Qwen chat `messages` (image
   and video content may be local paths). The same manifest format is used for
   both modalities. Configure defaults with the video config fields above, or
   set `nframes`/`fps`, `min_pixels`, and `max_pixels` on each video content
   item when needed.
2. Generate target responses:

   ```bash
   python -m src.train_VLM.prepare_responses \
     --config src/train_VLM/config_qwen25vl_7b.json \
     --input data/train.jsonl \
     --output data/train.target.jsonl
   ```

3. Point `train_manifest` at the generated manifest and train:

   ```bash
   python -m src.train_VLM.train \
     --config src/train_VLM/config_qwen25vl_7b.json
   ```

4. Measure teacher-forced accepted prefixes:

   ```bash
   python -m src.train_VLM.evaluate \
     --config src/train_VLM/config_qwen25vl_7b.json \
     --checkpoint checkpoints/dflash_qwen25vl_full \
     --manifest data/val.target.jsonl
   ```

Use `--decode-benchmark` to compare target autoregressive generation with the
lossless speculative decoder and persist latency, throughput, acceptance, and
peak-memory metrics:

```bash
python -m src.train_VLM.evaluate \
  --config src/train_VLM/config_qwen25vl_7b.json \
  --checkpoint checkpoints/dflash_qwen25vl_full/best \
  --manifest data/val.target.jsonl --decode-benchmark \
  --max-new-tokens 256 --max-samples 32 \
  --output results/dflash_qwen25vl_benchmark.json
```

For a local speculative decode, construct the processor inputs for one prompt
and call `Qwen25VLDFlashDecoder.generate` from `src.train_VLM.vlm_decode`. It
verifies every proposed block against the target posterior, rolls the target KV
cache back at the first mismatch and returns `output_ids` plus
`acceptance_lengths`. The decoder retains projected target-context K/V for each
draft layer, injects only newly verified target features on later iterations,
and never retains noise-block K/V after verification. Checkpoints carry an
implementation version and checkpoints created before the Qwen-compatible
M-RoPE/RMSNorm update are rejected instead of being loaded silently.

The legacy `python -m src.train_VLM.train` trainer uses online target hidden states. A record must
contain `target_response.token_ids`; this preserves the exact target output
used for anchor and label construction. Samples without a complete block or
whose clean sequence exceeds the configured length are reported and skipped;
neither the prompt nor the response is truncated.

An input record before response generation can look like:

```json
{"id":"img-001","messages":[{"role":"user","content":[{"type":"image","image":"/data/cat.jpg"},{"type":"text","text":"Describe the image."}]}]}
```

The response-prepared record adds:

```json
{"target_response":{"token_ids":[123,456,151645],"text":"..."}}
```

Prepared records also contain provenance for the target model/revision,
complete tokenizer-vocabulary fingerprint, processor fingerprint, and greedy
generation settings. Training rejects records whose provenance does not match
the loaded target. Checkpoints are written under `output_dir/checkpoints/`,
retain the last three entries, and include optimizer/scheduler/RNG/epoch state
for resume:

```bash
python -m src.train_VLM.train --config src/train_VLM/config_qwen25vl_7b.json \
  --resume-from-checkpoint checkpoints/dflash_qwen25vl_full/checkpoints/step-00001000
```

The final export remains in `output_dir/`, and the highest validation accepted
prefix checkpoint is at `output_dir/best/`. Use Python 3.11 with the pinned
Torch 2.7.1 / Transformers 4.57.1 dependencies in the repository-root
`requirements.txt` for real Qwen2.5-VL runs.
