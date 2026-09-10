# Local SpecForge DFlash training tree

This directory is a runnable copy of SpecForge source snapshot
`8cd9fd10dc77693deb1a5cc1b8851f397dd4b906` (version `0.2.0`) and contains the
code required for SpecForge text
DFlash training. It is intentionally separate from `src/train_VLM`, which is
the repository's Qwen2.5-VL/video trainer.

The copied tree contains:

- `specforge/`: the framework package and training/runtime code;
- `scripts/`: dataset and offline hidden-state preparation utilities;
- `configs/`: registered DFlash draft-model JSON configs;
- `examples/configs/`: canonical text DFlash YAML recipes;
- `examples/disagg/`: online/disaggregated launch helpers;
- `patches/`: SGLang capture patches used by online training;
- `tests/`: the upstream unit/integration tests;
- `pyproject.toml`, `version.txt`, and `requirements-rocm.txt`.

Run commands from this directory through `run_specforge.sh`; it adds this
checkout to `PYTHONPATH` without installing over the repository's VLM
environment:

```bash
./run_specforge.sh train --plan \
  --config examples/configs/qwen3-8b-dflash-offline.yaml
```

For a real offline run, prepare text data and target features first:

```bash
python scripts/prepare_data.py --dataset sharegpt
torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py \
  --strategy dflash \
  --target-model-path Qwen/Qwen3-8B \
  --draft-model-config configs/qwen3-8b-dflash.json \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/qwen3-8b-dflash-sharegpt \
  --chat-template qwen --max-length 3072 --tp-size 1 --batch-size 32
./run_specforge.sh train \
  --config examples/configs/qwen3-8b-dflash-offline.yaml
```

ShareGPT exports with a JSON array and `from`/`value` messages are accepted by
the online prompt reader.  A partially copied array is recovered up to its
last complete record (the truncated tail is reported and ignored):

```bash
python scripts/prepare_data.py \
  --dataset sharegpt \
  --data-path /path/to/sample.json \
  --output-path ./cache/dataset
```

Use the resulting JSONL as `data.train_data_path` in an online DFlash config.
Offline DFlash still consumes precomputed target hidden-state `.ckpt` files;
raw conversation JSON cannot replace `data.hidden_states_path` without running
the target capture step first.

## Qwen2.5-VL DFlash on local ShareGPT

The self-contained launcher below prepares text-only ShareGPT conversations,
captures Qwen2.5-VL target features, and trains the 3B and/or 7B DFlash draft.
It only imports code from this SpecForge tree:

```bash
bash train_qwen25vl_dflash_sharegpt_68k.sh --models 3b --phase all
```

Run individual phases when capture and training happen in separate jobs:

```bash
bash train_qwen25vl_dflash_sharegpt_68k.sh --phase data
bash train_qwen25vl_dflash_sharegpt_68k.sh --models 3b --phase capture
bash train_qwen25vl_dflash_sharegpt_68k.sh --models 3b --phase train
```

The B200 defaults use four GPUs, 68,000 rows, sequence length 2,048, global
batch 64, micro-batch 16 per GPU, DDP (`NO_SHARD`), and six epochs. Capture
uses batch 64 per GPU with pinned-memory input and asynchronous output. Override
storage paths or batch sizing with the environment variables shown by `--help`.
Capture supervises only the final assistant turn;
earlier turns remain prompt context. Use `--resume` to continue an interrupted
capture or an existing training checkpoint.

Checkpoints are written separately from captured features:

```text
outputs/qwen25vl-3b-dflash-sharegpt68k/
outputs/qwen25vl-7b-dflash-sharegpt68k/
```

The B200 profile saves every 1,000 optimizer steps and keeps the newest three
checkpoints. Resume a stopped run with the same model, GPU count, batch settings,
and output root:

```bash
bash train_qwen25vl_dflash_sharegpt_68k.sh \
  --gpus 2 --models 3b --phase train --resume
```

Set `OUTPUT_ROOT=/another/path` to relocate checkpoints while keeping large
offline hidden-state files under `ARTIFACT_ROOT`.

SpecForge upstream currently requires its own environment (`torch==2.11.0`,
`transformers==5.8.1`, `sglang==0.5.14`). Do not install those pins into the
root VLM environment, which has separate versions in the repository-level
`requirements.txt`.

## Qwen2.5-VL LLaVA caption Phase 2

The LLaVA caption workflow uses the offline SpecForge DFlash trainer with a
Qwen2.5-VL multimodal capture path and 3-axis M-RoPE feature tensors. Run it
on the server that owns the complete JSONL and image archive/root:

```bash
SOURCE_JSONL=/data/llava_dflash_68k_clean_3b.jsonl \
TARGET_MODEL_PATH=/models/qwen25-vl-3b \
PHASE1_CHECKPOINT=/runs/qwen25vl-phase1/dflash-step10000 \
IMAGE_ARCHIVE=/data/images.zip \
bash train_qwen25vl_dflash_llava_68k.sh --phase all
```

To keep the paths and run parameters in one place, copy
`train_qwen25vl_dflash_llava_68k.env.example` to a private env file, edit it,
and pass it to the launcher:

```bash
cp train_qwen25vl_dflash_llava_68k.env.example qwen25vl_llava_phase2.env
# edit qwen25vl_llava_phase2.env
bash train_qwen25vl_dflash_llava_68k.sh \
  --env-file qwen25vl_llava_phase2.env --phase all
```

The env file is shell-style configuration and is intentionally not committed
when it contains machine-specific paths. `--phase`, `--gpus`, and `--resume`
can still be supplied on the command line for a particular run.

The launcher requires exactly 68,000 valid JSONL records. The normalized
manifest preserves the source `response` as supervision and rejects malformed
tail lines, duplicate IDs, unsafe paths, and missing images. An archive is
materialized safely under the artifact root, using only referenced images.

The capture command requires the pinned SpecForge SGLang environment. Each
feature record contains `input_ids`, `loss_mask`, `hidden_states`, and Qwen
2.5-VL `position_ids`. The final `infer` phase is an HF smoke test that checks
image prefill plus DFlash greedy decoding against target-only greedy decoding;
it is not a production SGLang DFlash serving recipe.

For a two-B200 capture host, the example environment uses two independent
target replicas (`TP=1`, `DP=2`) and batches 16 requests per GPU. Image
processing is prefetched by eight worker-local processors per rank, while
eight writer threads drain feature checkpoints asynchronously. Tune these
without changing the output contract:

```bash
SPECFORGE_GPUS=2
SPECFORGE_CAPTURE_BATCH_SIZE=16
SPECFORGE_CAPTURE_PREPROCESS_WORKERS=8
SPECFORGE_CAPTURE_PREPROCESS_QUEUE=32
SPECFORGE_CAPTURE_IO_THREADS=8
SPECFORGE_CAPTURE_IO_QUEUE=64
SPECFORGE_SGLANG_MEM_FRACTION_STATIC=0.4
```

Keep `SPECFORGE_COMPRESS=0` for maximum capture throughput. After one
successful full preflight, a failed or interrupted capture can be resumed
without repeating that scan by setting `SKIP_PREFLIGHT=1`; existing feature
files are skipped atomically.

From the repository root, `run.sh` applies the complete two-B200 Phase 2
profile, forces `SKIP_PREFLIGHT=1`, reuses existing hidden-state files, and
automatically resumes the latest Phase 2 checkpoint. If no Phase 2 checkpoint
exists, training starts from `PHASE1_CHECKPOINT`:

```bash
cp src/train_Dflash_SpecForge/train_qwen25vl_dflash_llava_68k.env.example \
  qwen25vl_llava_phase2.env
# Edit only the machine-specific paths in qwen25vl_llava_phase2.env.
./run.sh --env-file qwen25vl_llava_phase2.env
```

The default training profile is two replicated BF16 drafts (`NO_SHARD`),
micro-batch 16 per GPU, accumulation 2, global batch 64, FlexAttention,
objective chunks of 256 anchors, and 12 ordered feature-loader workers per
rank. If unusually long image sequences exceed memory, first reduce
`SPECFORGE_MICRO_BATCH_SIZE` to 8; keep the global batch at 64 and the launcher
will recalculate accumulation to 4. Use
`SPECFORGE_ATTENTION_BACKEND=sdpa` only as a compatibility fallback on GPUs
whose shared-memory limit cannot compile the B200 FlexAttention kernel.

The launcher supports both target sizes. The default is 3B; a 7B Phase 2 run
must select the matching draft architecture and use a 7B Phase 1 checkpoint:

```bash
SPECFORGE_MODEL_SIZE=7b
TARGET_MODEL_PATH=/models/qwen25-vl-7b
PHASE1_CHECKPOINT=/runs/qwen25vl-7b-phase1/latest
ARTIFACT_ROOT=/data/artifacts/qwen25vl_7b_dflash_llava68k
```

Never point 3B and 7B at the same `ARTIFACT_ROOT`. Their captured feature
widths differ, so existing files from one size are not valid for the other.
The 7B profile captures target layers `[1, 7, 13, 19, 25]` and uses the same
Qwen2.5-VL three-axis M-RoPE contract as the 3B multimodal profile.

## H3.2 DFlash depth ablation

H3.2 compares DFlash decoder depths such as 1, 2, 3, and 5 while keeping the target
hidden-state inputs fixed at `[1, 9, 17, 25, 33]`. This isolates the number of
DFlash layers from the number of target features. The convenience launcher
captures each Phase 1/Phase 2 feature cache once, then trains an independent
checkpoint for every depth:

```bash
cd src/train_Dflash_SpecForge
SOURCE_JSONL=/data/llava_dflash_68k_clean_3b.jsonl \
TARGET_MODEL_PATH=/models/qwen25-vl-3b \
IMAGE_ARCHIVE=/data/images.zip \
SPECFORGE_MODEL_SIZE=3b \
SPECFORGE_DFLASH_DEPTHS=1,3,5 \
bash train_qwen25vl_dflash_h32.sh
```

The launch creates separate `-h32-depthN` Phase 1/Phase 2 runs for each
configured depth. `--resume` resumes interrupted captures
or checkpoints. Set `SPECFORGE_H32_CAPTURE_FIRST=0` only when the shared
feature caches already exist under the configured H3.2 artifact roots.

Under the hood, `SPECFORGE_DFLASH_DEPTH=1`, `2`, or `3` changes
`num_hidden_layers` and `layer_types` in the materialized draft config, but
does not change `dflash_config.target_layer_ids`. Do not set
`model.draft_num_hidden_layers` in the YAML for this ablation: the generic
model override derives a new target-layer list and would confound H3.2.

For concurrent end-to-end jobs, use one process per depth. Each process runs
Phase 1 and then Phase 2 on its own GPU set while reusing the existing feature
caches:

```bash
export SPECFORGE_MODEL_SIZE=3b
export TARGET_MODEL_PATH=/models/qwen25-vl-3b
export SPECFORGE_PHASE1_ARTIFACT_ROOT=/data/artifacts/qwen25vl_dflash_sharegpt68k
export SPECFORGE_PHASE2_ARTIFACT_ROOT=/data/artifacts/qwen25vl_3b_dflash_llava68k
export SPECFORGE_OUTPUT_ROOT=/data/outputs/dflash_depth_jobs

bash train_qwen25vl_dflash_depth_job.sh --depth 1 --gpu-ids 0,1
```

In a second terminal, change only the depth and GPU IDs:

```bash
SPECFORGE_DFLASH_DEPTH=2 SPECFORGE_GPU_IDS=2,3 \
  bash train_qwen25vl_dflash_depth_job.sh
```

The job launcher assigns a private run/config suffix and data-cache directory
per depth, so the two jobs can share hidden-state files without overwriting
each other's training state.
