# EAGLE3 training for Qwen2.5-VL 3B

This directory is the phase-one, text-only training slice of SpecForge for
Qwen2.5-VL 3B. It retains the original SpecForge EAGLE3 runtime and the
offline data/feature preparation path, while omitting unrelated algorithms,
benchmarks, exporters, documentation, assets, and upstream test suites.

## Run

Use the production YAML recipe from the repository root. It is the source of
truth for model, data, optimizer, precision, attention, checkpoint, and
two-GPU training settings:

```bash
SOURCE_DATA=./artifacts/eagle3/sharegpt.jsonl \
TARGET_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase all
```

The launcher supports `data`, `capture`, `train`, and `all` phases. Edit
the YAML recipe to control training; use environment variables only for paths,
dataset sample count, and capture I/O. The checked-in recipe is configured for
2×B200 with BF16, 2048-token text sequences, global batch 4 (batch 1 per GPU
with accumulation 2), ten epochs, and the original EAGLE3 objective settings.
Set `HF_HOME` and artifact paths for the local machine before a real run.

The authoritative model geometry is in
`configs/qwen2.5-vl-3b-eagle3.json`; the text-only offline recipe is in
`examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml`.

For the complete Phase 1 procedure, see
[PHASE1_TRAINING_GUIDE.md](PHASE1_TRAINING_GUIDE.md).

## Direct entry point

```bash
PYTHONPATH=src/train_EAGLE3_SpecForge \
  python -m specforge.cli train \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml
```

Use `--plan` to validate the resolved two-process launch topology without
starting workers.
