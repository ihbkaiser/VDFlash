# Phase 1 training guide

This guide covers the text-only EAGLE3 training stage for Qwen2.5-VL 3B.
It prepares text conversations, captures Qwen2.5-VL teacher features, and
trains the EAGLE3 draft model from those offline features.

Phase 2 image-captioning fine-tuning is not included in this guide or in the
current Phase 1 launcher.

## 1. Hardware and software

The checked-in recipe is configured for one node with two B200 GPUs:

- BF16 training
- two trainer processes
- 2048-token sequences
- batch size 1 per GPU with accumulation 2
- effective global batch size 4

The target model must be accessible through TARGET_MODEL_PATH, either as a
local directory or a Hugging Face model ID. Hugging Face authentication and
cache configuration should be completed before capture.

The data and train phases operate on local files. The capture phase uses the
SpecForge SGLang backend and therefore requires a compatible sglang
installation. yunchang is not needed with the default
sp_ulysses_size=1, sp_ring_size=1, and flex_attention settings; it is needed
only when enabling USP/ring sequence parallelism.

## 2. Configure the run

Edit this file before training:

~~~text
src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml
~~~

The YAML file controls the training process. Important fields are:

| YAML field | Purpose |
| --- | --- |
| model.target_model_path | Default target model location |
| model.torch_dtype | Training precision; keep bfloat16 on B200 |
| data.max_length | Maximum text sequence length |
| data.dataloader_num_workers | Offline feature-loader workers |
| training.num_epochs | Number of passes over the feature set |
| training.batch_size | Per-GPU micro-batch size |
| training.accumulation_steps | Gradient accumulation |
| training.learning_rate | Draft optimizer learning rate |
| training.lr_scheduler | Learning-rate schedule |
| training.warmup_ratio | Fraction of steps used for warmup |
| training.attention_backend | Draft attention implementation |
| training.save_interval | Checkpoint interval in optimizer steps |
| training.log_interval | Metric logging interval |
| deployment.trainer.nproc_per_node | Number of trainer processes |

The launcher accepts --config FILE or SPECFORGE_CONFIG. It does not replace
the YAML training values with smoke-test settings. Environment variables are
reserved for paths, sample count, GPU/process overrides, and capture I/O.

## 3. Prepare the source data

For a custom ShareGPT file, use JSON or JSONL rows with this structure:

~~~json
{
  "id": "example-0001",
  "conversations": [
    {"from": "human", "value": "Explain this concept."},
    {"from": "gpt", "value": "Here is an explanation."}
  ]
}
~~~

Prepare the configured 68,000-row Phase 1 dataset:

~~~bash
SOURCE_DATA=/data/sharegpt.jsonl \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
SPECFORGE_NUM_SAMPLES=68000 \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase data
~~~

The converted file is written to:

~~~text
/data/artifacts/qwen25vl_eagle3_text/shared/sharegpt_train.jsonl
~~~

The launcher checks the row count and stops if it does not match
SPECFORGE_NUM_SAMPLES.

## 4. Capture teacher features

Feature capture runs the Qwen2.5-VL target and writes the EAGLE3 auxiliary
hidden states, final hidden states, token IDs, masks, and vocabulary mapping:

~~~bash
TARGET_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase capture
~~~

The default capture uses both GPUs as two data-parallel workers (TP=1,
DP=2). Capture output is written below:

~~~text
/data/artifacts/qwen25vl_eagle3_text/hidden_states/
├── rows_0-4096/
│   └── data_0.ckpt
├── ...
└── vocab_mapping/
    └── vocab_mapping.pt
~~~

Capture is resumable:

~~~bash
TARGET_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase capture \
  --resume
~~~

Use SPECFORGE_CAPTURE_BATCH_SIZE, SPECFORGE_CAPTURE_WORKERS,
SPECFORGE_CAPTURE_IO_THREADS, and SPECFORGE_CAPTURE_IO_QUEUE to tune
capture throughput without changing the training recipe.

## 5. Train the EAGLE3 draft

After capture completes, start the real Phase 1 training run:

~~~bash
TARGET_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase train
~~~

The trainer loads the target output head, initializes the EAGLE3 draft, loads
the vocabulary mapping, and consumes the offline feature files. It writes
checkpoints below:

~~~text
/data/artifacts/qwen25vl_eagle3_text/outputs/
└── qwen25vl-3b-eagle3-text-offline/
    ├── qwen25vl-3b-eagle3-text-offline-step1000/
    ├── qwen25vl-3b-eagle3-text-offline-latest
    └── ...
~~~

To resume from the latest checkpoint:

~~~bash
TARGET_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase train \
  --resume
~~~

Resume restores the draft weights, optimizer/scheduler state, step counters,
and training RNG state. Do not reuse an output directory for a different
recipe or model geometry.

## 6. Run all phases

Once the environment and source data are ready, the three stages can be run
sequentially:

~~~bash
SOURCE_DATA=/data/sharegpt.jsonl \
TARGET_MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct \
ARTIFACT_ROOT=/data/artifacts/qwen25vl_eagle3_text \
bash src/train_EAGLE3_SpecForge/train_qwen25vl_eagle3_text.sh \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --phase all
~~~

For a long run, data, capture, and train are usually operated as separate
phases so that each artifact can be checked before the next stage.

## 7. Preflight and monitoring

Validate the resolved two-process plan without starting workers:

~~~bash
PYTHONPATH=src/train_EAGLE3_SpecForge \
python -m specforge.cli train \
  --config src/train_EAGLE3_SpecForge/examples/configs/qwen2.5-vl-3b-eagle3-text-offline.yaml \
  --plan \
  --role all
~~~

Before training, verify:

- the feature count is at least the number of GPUs times the per-GPU batch;
- vocab_mapping/vocab_mapping.pt exists;
- the target model matches Qwen2.5-VL 3B;
- the YAML nproc_per_node matches the visible GPU count;
- no prior checkpoint exists in the output directory unless --resume is used.

Training logs report per-step loss, per-head losses, acceptance rates,
gradient norm, learning rate, optimizer-step time, and global samples per
second. A healthy run should keep all values finite and create checkpoints at
the configured interval.

## 8. Common failures

- Missing sglang during capture: install a SpecForge-compatible SGLang build;
  the train phase can still consume an already-complete feature set without
  starting SGLang.
- Feature-count mismatch: rerun the data phase with the intended
  SPECFORGE_NUM_SAMPLES, then run capture again in a new artifact directory or
  with --resume.
- BF16 Triton errors on T4: BF16 capture/training is intended for B200 or
  other sm80+ GPUs. The production 2×B200 recipe should remain BF16.
- Out-of-memory during capture: lower only SPECFORGE_CAPTURE_BATCH_SIZE or the
  capture worker/I/O settings first. Change model/training hyperparameters in
  the YAML only when intentionally changing the experiment.
- Checkpoint already exists: use a new ARTIFACT_ROOT/OUTPUT_ROOT, or pass
  --resume when continuing the same recipe.

