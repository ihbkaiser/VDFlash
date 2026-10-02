# MSD shared-storage 1/3/5-depth sweep design

## Goal

Provide one safe Bash entry point that prepares the fixed ShareGPT-68K and
LLaVA-68K inputs, captures Qwen2.5-VL-3B teacher features once per corpus, and
then trains independent 1-, 3-, and 5-layer MSD drafts for the canonical 40
epochs. The training jobs share captured features but never share output or
checkpoint directories.

## Fixed inputs and storage layout

The launcher defaults to the existing shared workspace:

- Storage root: `/workspace/storage-shared/nlp/tungdd11/tungdecoder`
- Target model: `models/qwen25-vl-3b`
- ShareGPT source:
  `ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json`
- LLaVA JSONL:
  `data/llava_dflash_qwen25vl3b_68k/llava_dflash_68k_clean_3b.jsonl`
- LLaVA image root: `LlaVA-Pretrain`

All generated manifests, feature caches, generated recipes, logs, and
checkpoints default to
`/workspace/storage-shared/nlp/tungdd11/tungdecoder/artifacts/qwen25vl_msd_68k`.
Every path remains overridable through named environment variables.

For a LLaVA row whose `image` field is `00009/000090049.jpg`, validation must
resolve the file as
`/workspace/storage-shared/nlp/tungdd11/tungdecoder/LlaVA-Pretrain/00009/000090049.jpg`.

## User interface

The existing `train_qwen25vl_msd_depth_sweep.sh` remains the single entry
point. It supports:

- `--phase data`: convert ShareGPT and validate/materialize the LLaVA manifest.
- `--phase capture`: capture text and visual teacher features.
- `--phase train`: train requested depths from existing feature caches.
- `--phase all`: run all stages in order.
- `--resume`: resume capture/training where supported and reuse valid artifacts.
- `--print-config`: print resolved paths, topology, depths, and output locations
  without starting work.

The default command is:

```bash
bash train_qwen25vl_msd_depth_sweep.sh --phase all
```

`SPECFORGE_MSD_DEPTHS=1,3,5` and `SPECFORGE_GPUS=4` are the defaults. Operators
may restrict a recovery run to one depth or change the GPU count without
editing the script.

## Pipeline and reuse boundaries

The data stage deterministically converts the first 68,000 ShareGPT records
and normalizes the 68,000-row LLaVA JSONL into a manifest under the shared
artifact root. It validates the target model, source files, image root, sample
counts, and at least the manifest's referenced images before scheduling GPU
work.

The capture stage writes separate text and visual feature roots. Capture is
depth-independent because all three MSD drafts consume the same final teacher
hidden state, target input embedding, visual mask, and three-axis position IDs.
Therefore the launcher captures each corpus once, validates the cache, and
reuses it for depths 1, 3, and 5.

The train stage materializes one immutable draft JSON and one recipe per depth.
It runs depths sequentially to avoid GPU and storage-I/O contention. Each run
has a distinct run ID, output directory, checkpoint sequence, and latest
checkpoint link.

## Curriculum and invariants

Every depth uses the same model width, attention configuration, optimizer,
global batch, learning rate, seed, teacher features, and 40-epoch schedule.
Only `num_hidden_layers` changes.

For one-based epoch `e`, LLaVA sampling probability is fixed to:

```text
0.0                         when e <= 20
(e - 20) / 40 * 2          when 20 < e < 40
1.0                         when e == 40
```

Thus epochs 1-20 are ShareGPT-only, epochs 21-39 progressively mix LLaVA, and
epoch 40 is LLaVA-only. This is one paired 40-epoch training run per depth, not
a ShareGPT checkpoint followed by a separately reset LLaVA optimizer run.

## Failure handling and resume behavior

The launcher fails before GPU allocation when a fixed input is missing, a
relative LLaVA image cannot be resolved, the expected record count is wrong,
the global batch is incompatible with GPU count, or a requested depth is not
1, 3, or 5.

Existing non-empty artifacts are never silently overwritten. Without
`--resume`, an existing incomplete feature cache or checkpoint is reported as
an error. With `--resume`, validated captures and completed depths are reused;
an incomplete training depth resumes from its own output directory. A failure
in one depth stops the sequence and leaves earlier depth outputs intact.

## Verification

Tests must cover resolved shared-storage defaults, LLaVA relative-image
resolution, exact 1/3/5 output isolation, 40-epoch curriculum preservation,
dry-run configuration output, invalid path/depth rejection, and resume command
construction. Static verification must not require CUDA, model downloads, or
access to `/workspace/storage-shared`.

The implementation is complete when the focused launcher tests pass, the
existing MSD regression suite remains green, `bash -n` accepts the launcher,
and `--print-config` shows the exact shared paths and three isolated outputs.
