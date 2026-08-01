# DFlash training for Qwen2.5-VL

`src/train_VLM` implements the training objective described in Sections 4.1–4.2
and Appendix A.1 of `extetnal/dflash/DFlash.pdf`. It is a block drafter, not a
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

## Data flow

1. Prepare a JSONL manifest whose records contain a stable `id` and Qwen chat `messages` (image
   and video content may be local paths). The same manifest format is used for
   both modalities; set sampling options such as `fps`/`num_frames` through
   `processor_kwargs` when needed.
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
draft layer and never retains noise-block K/V after verification.

The v1 trainer intentionally uses online target hidden states. A record must
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
Torch 2.6 / Transformers 4.57.1 dependencies in `requirements.txt` for real
Qwen2.5-VL runs.
