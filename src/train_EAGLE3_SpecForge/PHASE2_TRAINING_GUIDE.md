# Phase 2 image-captioning training guide

This guide covers standalone offline image-captioning Phase 2 training for
Qwen2.5-VL 3B and 7B EAGLE3. The implementation lives entirely in
`src/train_EAGLE3_SpecForge`; the DFlash tree is not a runtime dependency.

## Requirements

- A completed Phase 1 EAGLE3 draft checkpoint.
- The Phase 1 vocabulary mapping file.
- Qwen2.5-VL 3B or 7B target access for feature capture.
- SGLang 0.5.14 and CUDA for capture.
- Image-caption JSONL and either an image directory or zip/tar archive.

Each source row has this shape:

```json
{
  "id": "example-0001",
  "image": "relative/path/image.jpg",
  "prompt": "Describe this image.",
  "response": "A caption for the image."
}
```

## Configure paths

```bash
export SOURCE_JSONL=/data/captions.jsonl
export IMAGE_ROOT=/data/images
export SPECFORGE_MODEL_SIZE=3b       # change to 7b for the 7B profile
export TARGET_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct
export PHASE1_CHECKPOINT=/data/eagle3-phase1-3b/outputs/phase1-latest
export VOCAB_MAPPING_PATH=/data/eagle3-phase1-3b/hidden_states/vocab_mapping/vocab_mapping.pt
export ARTIFACT_ROOT=/data/eagle3-phase2-3b
export SPECFORGE_GPUS=2
export SPECFORGE_NUM_SAMPLES=68000
export SPECFORGE_COMPRESS=1
```

For an archive, use `IMAGE_ARCHIVE=/data/images.tar` instead of
`IMAGE_ROOT`. The launcher materializes it under
`$ARTIFACT_ROOT/images` and rejects traversal and symlink members.

## Run the phases

Normalize the source manifest:

```bash
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_captioning.sh \
  --phase data
```

Capture image-conditioned EAGLE3 auxiliary/final hidden states:

```bash
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_captioning.sh \
  --phase capture
```

Train the EAGLE3 draft from the captured records:

```bash
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_captioning.sh \
  --phase train
```

Or run all phases:

```bash
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_captioning.sh \
  --phase all
```

The size-specific wrappers select the matching model/draft geometry and
artifact namespace:

```bash
bash src/train_EAGLE3_SpecForge/train_qwen25vl_3b_eagle3_captioning.sh --phase all
bash src/train_EAGLE3_SpecForge/train_qwen25vl_7b_eagle3_captioning.sh --phase all
```

For 2×B200, the checked-in recipes use BF16, EAGLE3's original Phase 1
objective (`learning_rate=5e-5`, `ttt_length=7`, cosine schedule), two samples
per rank and accumulation 16, giving global batch 64. The capture command below
uses two requests per GPU; reduce it to 1 if the installed SGLang build has a
larger per-request vision workspace:

```bash
export SPECFORGE_CAPTURE_BATCH_SIZE=2
```

At the 3072-token upper bound, persisted BF16 features are approximately 3.4
TB for 3B and 6.0 TB for 7B before filesystem overhead. Real captions are
usually shorter; compression is enabled above to reduce storage. Check
`du -sh "$ARTIFACT_ROOT/hidden_states"` before launching training.

A fresh training phase requires the same-size `PHASE1_CHECKPOINT`; it is loaded
as a weights-only warm start. `--resume` is reserved for an existing Phase 2
training checkpoint and restores optimizer, scheduler, RNG, counters, and data
position. It cannot be combined with a fresh warm start.

## Artifacts

The default artifact layout is:

```text
$ARTIFACT_ROOT/
├── shared/qwen25vl_caption_manifest.jsonl
├── shared/qwen25vl_caption_manifest.jsonl.meta.json
├── hidden_states/rows_0-2000/data_0.ckpt
└── outputs/qwen25vl-3b-eagle3-caption-offline/
```

Capture records contain `input_ids`, `loss_mask`, `aux_hidden_state`,
`hidden_state`, and 3-axis `position_ids`. Images and the vision processor are
used only by capture; Phase 2 draft training consumes these persisted tensors.
