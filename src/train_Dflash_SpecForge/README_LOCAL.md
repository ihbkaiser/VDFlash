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
