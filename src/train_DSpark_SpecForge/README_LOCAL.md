# DSpark training on the existing Qwen2.5-VL caches

This folder is an isolated SpecForge tree for training DSpark on the existing
Qwen2.5-VL-3B ShareGPT and LLaVA Phase 2 caches. It uses the DSpark objective,
Markov head, and confidence head while keeping DSpark configs, cache validation,
artifacts, and checkpoints separate from `src/train_Dflash_SpecForge`.

The launcher only trains. It does not prepare JSONL, stage images, capture
teacher states, rewrite cache metadata, or modify the DFlash tree.

## One-command two-B200 run with DSpark depth 2

On the company machine, the local ignored file
`src/train_Dflash_SpecForge/train_qwen25vl_dflash_3layer.env` supplies the
ShareGPT, model, LLaVA JSONL, image, and artifact paths. Run:

```bash
bash src/train_DSpark_SpecForge/train_qwen25vl_dspark_2layer_b200.sh
```

This prepares the ShareGPT conversion and LLaVA manifest, captures DSpark
features for both phases into a separate `dspark_depth2_layers_...` artifact
tree, then trains Phase 1 followed by Phase 2 warm-started from Phase 1. It
defaults to GPUs `0,1` and target taps `26,28,30,32,34`; the draft depth is two
DSpark decoder layers. It recaptures features with the DSpark schema because
the old DFlash cache does not contain DSpark's `target_last_hidden_states`.
Partial feature captures are reused if the command is rerun. Use `--resume` to
resume an existing training checkpoint, or `SPECFORGE_ENV_FILE=/path/to/file`
to select another local environment file. `--print-config` shows paths without
starting work.

## Server setup

Use the Python environment already configured for this SpecForge checkout, or
install this copied package into a compatible environment without replacing
its dependencies:

```bash
python -m pip install -e src/train_DSpark_SpecForge --no-deps
```

Copy and edit the environment file for the server:

```bash
cp src/train_DSpark_SpecForge/train_qwen25vl_dspark_two_phase.env.example \
  dspark_qwen25vl_server.env
```

Set `TARGET_MODEL_PATH`, the DFlash cache parent (or both explicit feature
roots), and DSpark output/artifact paths. The default cache layout for one layer
set is:

```text
<SPECFORGE_DFLASH_ARTIFACT_ROOT>/dflash_3layer_<IDs>/phase1/qwen25vl_3b/hidden_states
<SPECFORGE_DFLASH_ARTIFACT_ROOT>/dflash_3layer_<IDs>/phase2/hidden_states
```

The `high_only`, `middle_high`, `middle_high_2`, and `shallow_high` selections
must match the target layers used to make the respective caches. The draft uses
five DSpark decoder layers and block size 7 by default; the training head uses
the DSpark CE, L1, and confidence losses with weights 0.1, 0.9, and 1.0.

## Train

Inspect the resolved paths first, then run both phases:

```bash
bash src/train_DSpark_SpecForge/train_qwen25vl_dspark_two_phase.sh \
  --env-file dspark_qwen25vl_server.env --print-config
bash src/train_DSpark_SpecForge/train_qwen25vl_dspark_two_phase.sh \
  --env-file dspark_qwen25vl_server.env --phase all
```

Phase 1 trains on ShareGPT features. Phase 2 starts from that run's
`*-latest` checkpoint and trains on LLaVA features. Run a single phase with
`--phase phase1` or `--phase phase2`; a Phase 2-only run can set
`PHASE1_CHECKPOINT` explicitly. Use `--resume` to resume the selected DSpark
phase. The default expected cache count is 68,000 per phase and can be changed
with `SPECFORGE_PHASE1_NUM_SAMPLES` and `SPECFORGE_PHASE2_NUM_SAMPLES`.

Before each phase, the launcher checks file count, cache metadata when present,
and 32 evenly spaced feature files. Each feature file must include:

- `input_ids`, `loss_mask`, and Qwen2.5-VL three-axis `position_ids`;
- selected-layer `hidden_states` with width `5 * 2048`;
- `target_last_hidden_states` with width `2048` for DSpark's L1 and confidence losses.

Set `SPECFORGE_CACHE_VALIDATE_SAMPLE_LIMIT=0` to scan every file. The training
loader still validates and loads feature records lazily. Existing DFlash cache
metadata is read without migration or rewriting. If the cache does not include
`target_last_hidden_states`, the DSpark objective cannot be computed from that
cache; the launcher stops before training and identifies the missing tensor.

All resolved configs and data-loader caches are under the DSpark artifact root.
Checkpoints are under the DSpark output root, in `dspark_targets_<IDs>/phase1`
and `phase2`. They do not overwrite DFlash outputs.
