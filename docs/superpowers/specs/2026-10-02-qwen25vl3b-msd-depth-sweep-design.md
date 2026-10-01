# Qwen2.5-VL-3B MSD 1/3/5-Layer Training Design

Date: 2026-10-02

## Status

Proposed design for user review. Implementation and GPU training are outside
this document. The initial implementation scope is code plus CPU/static tests;
no model, dataset, or large artifact will be downloaded on the current host.

## Objective

Add Multimodal Speculative Decoding (MSD) as a first-class SpecForge
algorithm and provide one automated depth sweep for draft depths 1, 3, and 5
against a frozen Qwen2.5-VL-3B target.

Each depth uses the same deterministic 68,000-example ShareGPT cohort and the
same deterministic 68,000-example LLaVA cohort. Training is one continuous
40-epoch run with two logical phases and one optimizer/scheduler timeline:

- epochs 1-20: text-only;
- epochs 21-39: probabilistic text/visual mixture with visual ratios
  0.05, 0.10, ..., 0.95;
- epoch 40: visual-only.

The implementation must preserve MSD's input-token decoupling and
feature-level training objective. It must not silently substitute the EAGLE3
or DFlash model/objective.

## Source of truth and fidelity policy

The reference algorithm is:

- paper: *Speculative Decoding Reimagined for Multimodal Large Language
  Models*, arXiv:2505.14260v1;
- implementation: `https://github.com/Lyn-Lucy/MSD`, audited at upstream HEAD
  `fd76a5ee2bd107a5a04f05afd0651aab7ba6fab4`;
- vendored reference: `externals/MSD/`.

When the paper and source disagree, the decision is explicit:

- Training length follows the user-selected source-code behavior: 40 epochs,
  split into 20 text-only epochs and 20 progressive multimodal epochs.
- The visual mixture ratio follows the supplied formula exactly. With
  one-based `epoch_now`, it is:

  ```python
  if epoch_now <= total_epochs // 2:
      visual_ratio = 0.0
  elif total_epochs // 2 < epoch_now < total_epochs:
      visual_ratio = (epoch_now - total_epochs // 2) / total_epochs * 2
  else:
      visual_ratio = 1.0
  ```

- Learning rate `5e-5`, cosine decay, and a declared effective global batch
  size of 4 follow the paper. Warmup defaults to 2,000 optimizer steps and
  AdamW betas to `(0.9, 0.95)` following the public DeepSpeed config. All
  resolved values are written into run provenance.
- Dataset selection follows the paper's random 68k cohorts, but makes the
  selection deterministic with a configured seed and records source hashes
  and selected source IDs.
- MSD model semantics and loss weights follow the public implementation:
  feature loss weight `1.0`, soft-label loss weight `0.1`, and uniform
  hidden-state noise with width parameter `0.2`.

## Scope

### In scope

- a registered SpecForge algorithm named `msd`;
- a registered `MSDDraftModel` architecture for Qwen2.5-VL-3B;
- draft-depth configuration for 1, 3, and 5 layers;
- offline target-feature capture contracts for text and image examples;
- exact visual/text input decoupling;
- the MSD feature and soft-label losses;
- deterministic epoch-aware text/visual sampling;
- continuous 40-epoch checkpoint/resume semantics;
- an orchestrator that prepares/captures shared features once and creates
  isolated depth runs;
- CPU/static tests using tiny tensors and tiny configs;
- documentation of commands and artifact layout.

### Out of scope

- launching full GPU training from the current RTX 3050 Ti host;
- downloading Qwen2.5-VL-3B, ShareGPT, LLaVA images, or generated caches;
- claiming numerical replication before GPU runs finish;
- integrating MSD speculative inference into `lmms-eval`;
- changing the vendored `externals/MSD` implementation;
- benchmarking acceptance length, accuracy, latency, or speedup.

## Why a new SpecForge algorithm is required

SpecForge currently registers EAGLE3, P-EAGLE, DFlash, Domino, and DSpark.
Its EAGLE3 contract fixes `num_hidden_layers=1` and declares a text-only
feature contract. MSD additionally requires original multimodal input
embeddings, a visual-token mask, visual/text decoupling, and a different
loss. Reusing `training.strategy=eagle3` would therefore mislabel a different
algorithm as MSD.

The new implementation uses SpecForge's extension points instead:

- immutable algorithm registration;
- draft-model registry;
- offline feature contracts and readers;
- model warm-start and full resume;
- optimizer, scheduler, checkpoint, and distributed launch infrastructure.

## Architecture

### Algorithm package

Create `specforge/algorithms/msd/` with three focused modules:

- `providers.py`: algorithm declaration, model/data/step providers, resume
  contract, and supported capabilities;
- `data.py`: normalization and collation of offline MSD records;
- `model.py`: algorithm wrapper that applies shifting, decoupling, target-head
  projection, and loss computation.

Register `msd` in `specforge/algorithms/builtin.py` without altering existing
algorithm contracts.

### Draft architecture

Create `specforge/modeling/draft/msd.py` containing:

- `MSDDraftConfig`, derived from the Qwen2.5-VL text configuration;
- `MSDDraftModel`, registered under the exact architecture name
  `MSDDraftModel`;
- a frozen token embedding copied from the target;
- a trainable `Linear(2 * hidden_size, hidden_size)` fusion projection;
- 1, 3, or 5 Qwen2.5-compatible decoder layers;
- final normalization where required by the reference decoder contract.

The canonical Qwen2.5-VL-3B draft shape is:

- hidden size: 2,048;
- intermediate size: 11,008;
- attention heads: 16;
- KV heads: 2;
- head dimension: 128;
- vocabulary size: resolved from the target/tokenizer rather than duplicated
  in launch scripts;
- M-RoPE section: `[16, 24, 24]`;
- draft layers: one of 1, 3, or 5.

Depth changes only the number of draft decoder layers. It must not change the
target feature layer, dataset cohort, loss, schedule, tokenizer, or visual
preprocessing.

### Input-token decoupling

For aligned target positions, let `h_t` be the frozen target hidden state,
`e_next` the next input embedding, and `v_t` the original visual embedding.
The draft input is:

```text
text position:   Linear(concat(h_t, e_next))
visual position: v_t
```

The visual path bypasses the fusion projection. The implementation uses an
explicit boolean `visual_token_mask`; it must not infer modality from a
single sentinel token or a fixed visual-token count. This is required because
Qwen2.5-VL supports variable image grids.

All indexing is centralized in a pure helper so unit tests can compare it
against a literal reference implementation. The helper rejects shape,
sequence-length, and mask mismatches.

### Qwen2.5-VL position handling

Offline records retain Qwen2.5-VL three-axis M-RoPE `position_ids`. Collation
pads the sequence dimension without collapsing the three axes. The draft
decoder accepts both ordinary two-dimensional positions for text-only tests
and `[3, batch, sequence]` positions for multimodal data.

## Offline feature contract

Each normalized record provides:

- `input_ids`: target tokenizer IDs;
- `loss_mask`: assistant-answer positions eligible for training;
- `target_hidden_state`: final target-language-model hidden state;
- `input_embeddings`: embeddings presented to the target language model after
  vision encoding/projection and token embedding replacement;
- `visual_token_mask`: positions occupied by image embeddings;
- `position_ids`: Qwen2.5-VL M-RoPE positions;
- provenance: source ID, corpus, target revision, processor contract, and
  source-selection fingerprint.

Teacher features are frozen and captured once per corpus. The three draft
depths reuse them. Cache validation fails closed on target revision, tokenizer,
hidden width, dtype, preprocessing, selected cohort, or tensor-schema
mismatch.

Text examples store an all-false visual mask. Image examples must preserve the
exact variable number and positions of visual embeddings emitted by the
Qwen2.5-VL processor/model.

## Shifting and loss

The normalizer reproduces the public MSD implementation's next-token shift:

- draft conditioning hidden states remain aligned to the current positions;
- token IDs, target hidden features, and input embeddings used as next-token
  supervision are shifted left by one position;
- the final padded position is masked out;
- only assistant positions contribute to the objective.

The loss is:

```text
loss = 1.0 * SmoothL1(predicted_hidden, target_next_hidden)
     + 0.1 * SoftTargetCrossEntropy(
         frozen_lm_head(predicted_hidden),
         softmax(frozen_lm_head(target_next_hidden)),
       )
```

Both terms are normalized by the valid loss-mask count. The target embedding
and LM head remain frozen. Uniform noise is added only to the conditioning
target hidden states, using the reference scaling
`(rand_like(h) - 0.5) * 0.2 * 512 / sequence_length`.

## Dataset preparation

Two immutable manifests are produced:

- `sharegpt68k`: 68,000 valid conversations from
  `ShareGPT_V3_unfiltered_cleaned_split.json`;
- `llava68k`: 68,000 valid image conversations from the configured LLaVA
  instruction source.

Selection is deterministic by seed and stable source ID. Manifests record the
source revision/hash and selected IDs. Invalid or missing-media records are
excluded before selection so each completed manifest contains exactly 68,000
usable records.

The implementation does not substitute LLaVA-Pretrain captions for the
paper's LLaVA instruction-tuning records unless explicitly configured as a
non-replication profile.

## Forty-epoch curriculum

One run owns all 40 epochs. Stage 2 is not a separate optimizer run.
Optimizer, cosine scheduler, global step, RNG state, and sampler state continue
across the epoch-20/21 boundary.

For `total_epoch=40`, the exact visual ratios are:

```text
epochs  1-20: 0.00
epoch      21: 0.05
epoch      22: 0.10
...
epoch      39: 0.95
epoch      40: 1.00
```

At each logical sample draw, the epoch sampler chooses the visual corpus with
probability `current_ratio`; otherwise it chooses the text corpus. The choice
is deterministic for `(seed, epoch, global sample index)` so exact resume does
not depend on Python worker scheduling. This preserves the probabilistic
source-code behavior while making distributed resume reproducible.

Each epoch contains 68,000 logical samples, matching the equal corpus sizes.
The sampler records realized text/visual counts for provenance and validation.

## Configuration and automation

Add:

- one base draft config for Qwen2.5-VL-3B MSD;
- one offline training recipe with the 40-epoch curriculum;
- `train_qwen25vl_msd_68k.sh`, which supports `data`, `capture`, `train`, and
  `all` phases;
- `train_qwen25vl_msd_depth_sweep.sh`, which materializes depth-specific draft
  configs and runs depths `1,3,5` sequentially by default.

Environment variables expose paths, GPU count, seed, depth list, overwrite,
and resume behavior. Default artifact layout:

```text
artifacts/qwen25vl_3b_msd_68k/
  manifests/sharegpt68k.jsonl
  manifests/llava68k.jsonl
  features/sharegpt68k/
  features/llava68k/
  configs/depth1.json
  configs/depth3.json
  configs/depth5.json
  outputs/depth1/
  outputs/depth3/
  outputs/depth5/
```

The orchestrator reuses complete manifests/features and refuses partial or
incompatible caches. `--resume` resumes the same depth's full training state.
Weights-only initialization is not used between logical phases because they
are one continuous run.

## Checkpoint and resume contract

The immutable resume contract contains:

- target model/revision and processor contract;
- draft depth and architecture config;
- both manifest/cache fingerprints;
- total epochs and exact ratio formula/version;
- epoch, global sample offset, and realized mixture counts;
- optimizer/scheduler/loss/noise settings;
- batch topology and gradient accumulation;
- seed and per-rank RNG/sampler state.

Resume rejects any change that affects training mathematics. Starting a new
ablation from weights uses a new output directory and an explicit weights-only
option.

## Error handling

The pipeline fails before training when:

- either manifest has fewer or more than 68,000 usable records;
- media are missing;
- visual mask count does not match captured visual embeddings;
- M-RoPE shapes or sequence lengths disagree;
- feature width differs from 2,048;
- an existing output belongs to another depth or training contract;
- a partial cache is encountered without explicit resume;
- depth is not 1, 3, or 5.

Atomic checkpoint behavior remains owned by SpecForge. No launcher deletes
or overwrites datasets, features, or checkpoints implicitly.

## Static and CPU test plan

Tests require no target checkpoint, dataset download, CUDA, or network.

1. Registry tests verify `msd` is discoverable and advertises text and image
   offline feature contracts.
2. Tiny-config model tests instantiate depths 1, 3, and 5 and verify layer
   counts, parameter monotonicity, and output shapes.
3. Decoupling tests prove visual positions equal original visual embeddings
   and text positions equal the fusion projection reference.
4. Shift tests compare normalized examples with a literal MSD reference on
   text-only and variable-grid visual sequences.
5. Loss tests compare feature loss, soft-label loss, masking, weights, and
   gradients with direct PyTorch formulas.
6. Position tests preserve both two-dimensional and three-axis M-RoPE inputs.
7. Curriculum tests assert ratios for every epoch, especially epochs 20, 21,
   39, and 40.
8. Determinism tests reproduce corpus choices after an interrupted/resumed
   epoch and across simulated data-loader worker orders.
9. Config/launcher tests assert shared feature roots, isolated depth outputs,
   exact 40-epoch settings, and fail-closed resume wiring.
10. Existing SpecForge algorithm/registry/config tests run unchanged to catch
    regressions.

On the current Windows host, verification first discovers an available Python
or WSL interpreter. If none is present, source-level checks can be performed,
but executable pytest results must be reported as unavailable rather than
claimed as passing.

## Acceptance criteria for this implementation pass

- New code is isolated from `externals/MSD`.
- `msd` is a first-class SpecForge strategy, not an EAGLE3 alias.
- Depths 1, 3, and 5 are materialized without changing any other experiment
  variable.
- The exact 40-epoch ratio sequence is unit tested.
- MSD decoupling and both loss terms have independent reference tests.
- Automation commands and artifact layout are documented.
- All runnable focused CPU/static tests pass; unavailable runtime checks are
  explicitly listed.
- No claim is made that training or paper metrics were reproduced.
