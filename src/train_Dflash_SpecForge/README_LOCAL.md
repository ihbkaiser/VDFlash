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

## Phase 3: visual-10% DFlash self-distillation

The Phase 3 implementation trains a 10%-visual student from a Phase 2 DFlash
checkpoint. Each update uses the same target features, anchors, target tokens,
and block layout for both branches: the frozen DFlash teacher sees the full
visual context, while the student masks all but exactly
`ceil(0.1 * visual_token_count)` visual keys. Text keys remain available to
both branches. The target-token CE, full-vocabulary teacher-to-student KL, and
teacher-correctness gate are selectable as `target_ce`, `teacher_kl`, `dual`,
`gated`, or `gated_mean_scaled` recipes. The default recipe is CE + KL with
temperature 1 and KL weight 0.5.

### Set up the training server environment

Use a fresh Linux x86_64 checkout with Python 3.12. The workspace launcher
resolves its interpreter as `<repository>/.venv/bin/python`, so create and
activate `.venv` at the repository root before installing dependencies. The
[dedicated requirements file](requirements-visual10-selfdistill-py312-cu130.txt)
pins the core framework versions used by this SpecForge snapshot and adds the
Qwen2.5-VL video processing dependencies. It does not modify the repository's
general VLM `requirements.txt`.

```bash
cd /path/to/VDFlash
nvidia-smi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r src/train_Dflash_SpecForge/requirements-visual10-selfdistill-py312-cu130.txt
python -m pip check
```

The PyTorch wheels in this file target CUDA 13.0. Confirm the server has a
compatible NVIDIA driver and GPU before installing: CUDA 13.0 requires Linux
driver 580.65.06 or newer, and CUDA 13 no longer supports Maxwell, Pascal, or
Volta GPUs (including V100). The SGLang 0.5.14 CPython 3.12 wheel also requires
glibc 2.34 or newer. See the [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
and [SGLang 0.5.14 package files](https://pypi.org/project/sglang/0.5.14/).
The NVIDIA driver is managed by the server administrator and is not installed
by pip.

Verify that the active environment sees the expected Python, PyTorch CUDA
runtime, and GPU:

```bash
python --version
python -c 'import torch; print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda); assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0)); assert torch.version.cuda.startswith("13.0")'
```

### Prepare offline target features

Start from the prepared video dataset under `data/dflash_posttraining`. If the
server only has the raw private Hugging Face dataset, first run the `prepare`,
`targets`, and `validate` commands in
[`data/dflash_posttraining/README.md`](../../data/dflash_posttraining/README.md).
Target responses must come from the same target/processor provenance used by
the feature cache. The cache tool stores target-layer features, input token IDs,
response loss masks, and three-axis M-RoPE positions. It does not store teacher
logits; Phase 3 computes those on the fly.

Run a small pilot to estimate the feature-cache size before caching the whole
cohort:

```bash
cd /path/to/VDFlash
PYTHONPATH="src/train_Dflash_SpecForge${PYTHONPATH:+:$PYTHONPATH}" \
  torchrun --nproc_per_node=2 -m src.train_VLM.cache_visual10_selfdistill_features \
  --data-dir artifacts/dflash_posttraining_visual10 \
  --target-config configs/dflash_posttraining_visual10_responses.example.json \
  --draft-config src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json \
  --output-dir artifacts/dflash_visual10_features_pilot \
  --splits train --max-samples 8
```

Then build full features (all ranks need access to the same output filesystem):

```bash
PYTHONPATH="src/train_Dflash_SpecForge${PYTHONPATH:+:$PYTHONPATH}" \
  torchrun --nproc_per_node=4 -m src.train_VLM.cache_visual10_selfdistill_features \
  --data-dir artifacts/dflash_posttraining_visual10 \
  --target-config configs/dflash_posttraining_visual10_responses.example.json \
  --draft-config src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json \
  --output-dir artifacts/dflash_visual10_features
```

The output follows the SpecForge offline feature format:
`features/{train,validation,test}/rows_*/sample_*.ckpt`. Each split includes
layer/phase metadata and a cache contract that prevents accidentally mixing
features from different target revisions, response manifests, or layer lists.
The tool resumes completed samples when the contract matches; pass `--overwrite`
only when you intend to replace the selected split directories.

These caches can be large: Qwen2.5-VL-3B with five 2,048-wide target layers
uses about 10,240 bf16 values per sequence token, before token IDs and positions.
Use the pilot summary to size storage for the actual sequence lengths.

### Train the visual-10 post-training recipes

Use [`examples/configs/qwen25vl-3b-dflash-visual10-selfdistill.yaml`](examples/configs/qwen25vl-3b-dflash-visual10-selfdistill.yaml)
as the shared starting config. One launcher supports both objectives; choose the
objective with `training.dflash_distill_recipe`:

| Recipe | Training signal | Purpose |
| --- | --- | --- |
| `target_ce` | Target-model next-token cross-entropy | Train the 10%-visual student from the target-model signal only. |
| `dual` | Target-model cross-entropy plus KL from the frozen full-visual Phase 2 DFlash teacher | Combine target-model supervision with the full-visual DFlash signal. |

For a fresh run, set `model.draft_model_config` to the exact Phase 2 DFlash
architecture config. Set `model.draft_checkpoint_path` and
`training.dflash_teacher_checkpoint_path` to the same Phase 2 full-visual
checkpoint. Set `model.target_model_path` to the target model. Point
`data.hidden_states_path` and `data.eval_hidden_states_path` at the generated
train and validation feature roots. The example uses one long sequence per GPU,
full FSDP sharding, 128 anchors, and two anchor blocks per full-vocabulary
logits slice; lower anchor/chunk sizes if the server runs out of memory.

The YAML contains placeholder paths. Update those paths on the training server
before launching. The feature cache must already exist; training consumes the
cached target hidden states and does not build them.

#### Target-model signal only (`target_ce`)

```bash
cd src/train_Dflash_SpecForge
CFG=examples/configs/qwen25vl-3b-dflash-visual10-selfdistill.yaml

./run_specforge.sh train --plan \
  --config "$CFG" \
  training.dflash_distill_recipe=target_ce \
  run_id=visual10-target-ce \
  output_dir=/checkpoints/dflash-visual10/target-ce

./run_specforge.sh train \
  --config "$CFG" \
  training.dflash_distill_recipe=target_ce \
  run_id=visual10-target-ce \
  output_dir=/checkpoints/dflash-visual10/target-ce
```

#### Target model plus full-visual DFlash teacher (`dual`)

```bash
cd src/train_Dflash_SpecForge
CFG=examples/configs/qwen25vl-3b-dflash-visual10-selfdistill.yaml

./run_specforge.sh train --plan \
  --config "$CFG" \
  training.dflash_distill_recipe=dual \
  run_id=visual10-dual \
  output_dir=/checkpoints/dflash-visual10/dual

./run_specforge.sh train \
  --config "$CFG" \
  training.dflash_distill_recipe=dual \
  run_id=visual10-dual \
  output_dir=/checkpoints/dflash-visual10/dual
```

`--plan` prints the resolved launch plan; run the second command to start
training. These commands run one recipe per invocation. Use separate
`run_id` and `output_dir` values so the checkpoints and logs do not overlap.
The same pattern applies to the other implemented ablations: `teacher_kl`,
`gated`, and `gated_mean_scaled`. To use the prefix visual selector, add
`training.dflash_visual_selector=prefix`. Keep seeds, data, optimizer steps,
and visual retention fixed when comparing recipes.

Implementation note: `target_ce` skips the teacher forward pass and its KL
loss, but the current model provider still loads the Phase 2 teacher checkpoint
and keeps a frozen teacher draft in memory. The teacher checkpoint remains a
required config field, so account for that extra model's memory use when sizing
the target-only run.

For resume, retain `training.dflash_teacher_checkpoint_path` pointing at the
original Phase 2 checkpoint, remove `model.draft_checkpoint_path` from a copy
of the YAML, and set `training.resume_from` to the Phase 3 checkpoint. The
checkpoint stores only student DFlash weights; the frozen teacher is rebuilt
from Phase 2.

### Greedy parity, acceptance, and runtime evaluation

For the Phase 3 video manifests, use the video-aware evaluator. It reports
greedy output parity, round-end-corrected `tau_proposal` and `tau_effective`,
first-rejection and accepted-prefix survival, actual visual retention,
processor/target/speculative latency, selector/draft/verifier time, and peak
allocated GPU memory plus peak target/draft K/V tensor bytes. It writes both
per-video rows and the aggregate summary:

```bash
cd /path/to/VDFlash
python -m src.train_VLM.evaluate_visual10_selfdistill \
  --target-config configs/dflash_posttraining_visual10_responses.example.json \
  --draft-config /models/qwen2.5-vl-3b-dflash.json \
  --checkpoint /checkpoints/dflash-visual10/dual \
  --manifest artifacts/dflash_posttraining_visual10/responses/validation.jsonl \
  --output artifacts/dflash_visual10_eval/dual-validation.json \
  --max-new-tokens 128 --visual-retention 0.1 \
  --visual-selector temporal_spatial
```

Run the same evaluator on the Phase 2 checkpoint for the full-visual baseline
(`--full-visual`) and the untrained 10% student baseline (Phase 2 checkpoint
with `--visual-retention 0.1`). Use validation to select a recipe, then run the
test manifest once for confirmation. Keep model, selector, video sampling,
generation limit, dtype, and hardware fixed across comparisons. The script
reports one measured pass after configurable warmup; repeat runs and bootstrap
paired videos before making a runtime claim. The older
`scripts/infer_qwen25vl_dflash.py` remains a single-image smoke check.

SpecForge upstream currently requires its own environment (`torch==2.11.0`,
`transformers==5.8.1`, `sglang==0.5.14`). Do not install those pins into the
root VLM environment, which has separate versions in the repository-level
`requirements.txt`.

## Original MSD depth sweep on Qwen2.5-VL-3B

The dedicated `msd` backend reproduces the public `externals/MSD` objective
instead of aliasing EAGLE3 or DFlash. It trains homogeneous 1-, 3-, and 5-layer
draft transformers against one frozen Qwen2.5-VL-3B target. Text positions use
the projected concatenation of the current target hidden state and next-token
embedding; visual positions bypass that projection with the post-vision input
embedding. The loss is SmoothL1 with weight 1.0 plus frozen-LM-head soft-target
cross entropy with weight 0.1. Uniform feature noise uses width 0.2 scaled by
`512 / sequence_length`.

The 40-epoch run is continuous and does not reset the optimizer. Epochs 1-20
select ShareGPT only. Epochs 21-39 select LLaVA with probability
`(epoch - 20) / 40 * 2`, and epoch 40 selects LLaVA only. Corpus selection is
hash-based on seed, epoch, and logical sample index, so checkpoint resume
reconstructs the same mixture.

The launcher defaults to `/path/to/local-resource`:
it reads the ShareGPT JSON, the LLaVA 68k JSONL, images below
`LLaVA-Pretrain/`, and the local Qwen2.5-VL-3B target from that tree. Override
the common prefix with `SPECFORGE_SHARED_STORAGE_ROOT` or override any printed
path individually.

From the SpecForge directory, inspect the resolved configuration without
touching data or GPUs, or run the complete workflow:

```bash
cd /path/to/local-resource
bash train_qwen25vl_msd_depth_sweep.sh --print-config
bash train_qwen25vl_msd_depth_sweep.sh --phase all
```

The train phase runs depths 1, 3, then 5 sequentially. For recovery, rerun only
the required phase; `--resume` preserves existing feature records, resumes an
unfinished checkpoint, and skips a depth whose `.complete` marker exists:

```bash
bash train_qwen25vl_msd_depth_sweep.sh --phase data
bash train_qwen25vl_msd_depth_sweep.sh --phase capture --resume
bash train_qwen25vl_msd_depth_sweep.sh --phase train --resume
```

The immutable text and visual feature caches are shared across all depths.
Generated configs and checkpoints remain isolated:

```text
/path/to/local-resource{1,3,5}/
/path/to/local-resource{1,3,5}/output/
```

Each raw MSD feature record contains `input_ids`, `loss_mask`, the target final
hidden state, the language-model input embeddings after visual replacement, an
explicit visual-token mask, and Qwen2.5-VL three-axis position IDs. Resume
metadata records the depth, objective weights, noise width, target dimensions,
curriculum seed, and curriculum formula version.

This automated launcher is verified at static/config rung R1 only. It has not
read the server-side 68k datasets, run CUDA capture, completed 40 epochs, or
reproduced the paper's acceptance/speed results. Perform a tiny real-cache
overfit and GPU capture smoke test before launching the full sweep.

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

Caches produced by an older VDFlash checkout may contain only `rows_*/data_*.ckpt`
files and no `hidden_state_metadata.json`. The depth-job launcher migrates such
a cache on first use after checking one row's required keys and feature width;
the feature rows themselves are not rewritten. A width or phase/layer mismatch
still stops the run and requires a fresh capture.
