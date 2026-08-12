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

SpecForge upstream currently requires its own environment (`torch==2.11.0`,
`transformers==5.8.1`, `sglang==0.5.14`). Do not install those pins into the
root VLM environment, which has separate versions in the repository-level
`requirements.txt`.
