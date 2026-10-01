# Qwen2.5-VL-3B MSD Depth Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class, paper-faithful MSD training path for Qwen2.5-VL-3B with draft depths 1, 3, and 5 and the exact 40-epoch text-to-vision curriculum.

**Architecture:** Implement MSD behind SpecForge's algorithm and draft-model registries, with pure tensor helpers for decoupling/loss and a deterministic epoch-aware corpus selector. Extend offline capture records with Qwen2.5-VL input embeddings, visual masks, and M-RoPE positions, then expose a single launcher that shares the two 68k feature caches across isolated depth runs.

**Tech Stack:** Python 3.11, PyTorch, Transformers/SpecForge, Pydantic, pytest, Bash.

---

## Experiment card (R0)

```text
Question:         Can SpecForge faithfully express the public MSD training
                  algorithm for Qwen2.5-VL-3B at draft depths 1, 3, and 5?
Hypothesis:       A dedicated MSD backend will reproduce the reference tensor
                  semantics and create comparable depth-only training runs.
Baseline:         Lyn-Lucy/MSD upstream HEAD fd76a5e, one-layer Qwen2-VL path.
Variants:         Draft decoder depth only: 1 versus 3 versus 5.
Primary metric:   Mean accepted tokens per speculative verification round.
Success:          Full-run depth reports exist with strict lossless decoding;
                  no direction of the depth effect is assumed in advance.
Guardrails:       Target-output equality, finite loss, no target/LM-head grads,
                  matched cohorts/schedule, peak memory and latency recorded.
Data / split:     Deterministic ShareGPT-68k and LLaVA instruction-68k train;
                  disjoint fixed evaluation manifests for later GPU work.
Seeds:            Seed 0 for source replication; additional seeds are future
                  empirical work, not part of this static implementation pass.
Budget:           Current pass stops after CPU/static verification (R1/R2 where
                  dependencies allow); no GPU/model/dataset downloads.
Start rung:       R1 import/config/static checks.
Exploratory-only: Depth quality/speed rankings and paper-level speedup claims.
```

## Verification ladder

- **R0 — protocol:** this plan plus the approved design. Proof: committed
  design and plan files.
- **R1 — static/import/config:** focused pytest suite imports the `msd`
  registration, parses the recipe, validates all 40 ratios, and checks launcher
  dry-run output. Proof: saved pytest console output.
- **R2 — tensor/one-step:** tiny CPU forward/backward has finite loss, finite
  trainable gradients, and no frozen embedding/head gradients. Proof: focused
  pytest output with shape/dtype assertions.
- **R3 — tiny overfit:** deferred; needs a compatible runtime and a small real
  or synthetic feature cache. Required before any GPU launcher smoke.
- **R4-R7:** explicitly outside this static implementation pass. No paper
  replication claim is allowed before these rungs are completed.

Promotion stops at the first missing/red artifact. Passing R1/R2 demonstrates
code-path correctness only.

## File map

Create:

- `src/train_Dflash_SpecForge/specforge/algorithms/msd/__init__.py` — public
  algorithm exports.
- `src/train_Dflash_SpecForge/specforge/algorithms/msd/curriculum.py` — exact
  epoch ratio and stateless corpus choice.
- `src/train_Dflash_SpecForge/specforge/algorithms/msd/data.py` — record
  validation, next-token shifting, padding/collation, and offline reader.
- `src/train_Dflash_SpecForge/specforge/algorithms/msd/model.py` — decoupling,
  uniform noise, reference losses, and training wrapper.
- `src/train_Dflash_SpecForge/specforge/algorithms/msd/providers.py` — immutable
  algorithm/provider registration and resume contract.
- `src/train_Dflash_SpecForge/specforge/modeling/draft/msd.py` — configurable
  Qwen2.5-compatible MSD decoder.
- `src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-msd.json` — base draft
  architecture config.
- `src/train_Dflash_SpecForge/examples/configs/qwen2.5-vl-3b-msd-68k-offline.yaml`
  — canonical run recipe.
- `src/train_Dflash_SpecForge/train_qwen25vl_msd_depth_sweep.sh` — shared-cache
  1/3/5 orchestrator with dry-run support.
- `src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py` — pure algorithm,
  model, data, and gradient tests.
- `src/train_Dflash_SpecForge/tests/test_scripts/test_msd_depth_sweep_launcher.py`
  — launcher/config wiring tests.

Modify:

- `src/train_Dflash_SpecForge/specforge/algorithms/builtin.py` — register MSD.
- `src/train_Dflash_SpecForge/specforge/algorithms/model_providers.py` — build
  the MSD draft/training wrapper and apply depth overrides.
- `src/train_Dflash_SpecForge/specforge/modeling/draft/__init__.py` — import the
  registered architecture.
- `src/train_Dflash_SpecForge/specforge/config/schema.py` — typed MSD weights,
  noise, curriculum, and paired feature-root settings.
- `src/train_Dflash_SpecForge/specforge/training/strategies/base.py` — add the
  MSD step boundary.
- `src/train_Dflash_SpecForge/specforge/algorithms/common/providers.py` — extend
  offline capture layout to accept algorithm-owned captured tensors.
- `src/train_Dflash_SpecForge/scripts/prepare_hidden_states.py` and the local
  capture adapter — persist input embeddings, visual mask, and position IDs.
- `src/train_Dflash_SpecForge/README_LOCAL.md` — usage, fidelity decisions,
  artifact layout, and verification limits.

## Task 1: Exact 40-epoch curriculum

**Files:**
- Create: `src/train_Dflash_SpecForge/specforge/algorithms/msd/curriculum.py`
- Create: `src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py`

- [ ] **Step 1: Write the failing boundary and determinism tests**

```python
from specforge.algorithms.msd.curriculum import (
    choose_visual_sample,
    visual_ratio_for_epoch,
)


def test_msd_40_epoch_ratio_matches_reference_boundaries():
    assert visual_ratio_for_epoch(1, 40) == 0.0
    assert visual_ratio_for_epoch(20, 40) == 0.0
    assert visual_ratio_for_epoch(21, 40) == 0.05
    assert visual_ratio_for_epoch(39, 40) == 0.95
    assert visual_ratio_for_epoch(40, 40) == 1.0


def test_msd_choice_is_stateless_and_resume_stable():
    first = [choose_visual_sample(0, 27, index, 40) for index in range(100)]
    resumed = [choose_visual_sample(0, 27, index, 40) for index in range(50, 100)]
    assert first[50:] == resumed
```

- [ ] **Step 2: Run the test and confirm RED**

Run from `src/train_Dflash_SpecForge`:

```powershell
py -3.11 -m pytest -q tests/test_algorithms/test_msd.py
```

Expected: collection fails because `specforge.algorithms.msd` does not exist.

- [ ] **Step 3: Implement the exact formula and hash-based Bernoulli draw**

```python
def visual_ratio_for_epoch(epoch_now: int, total_epoch: int = 40) -> float:
    if total_epoch <= 0 or epoch_now < 1 or epoch_now > total_epoch:
        raise ValueError("epoch_now must be in [1, total_epoch]")
    if epoch_now <= total_epoch // 2:
        return 0.0
    if epoch_now < total_epoch:
        return (epoch_now - total_epoch // 2) / total_epoch * 2
    return 1.0


def choose_visual_sample(seed: int, epoch_now: int, sample_index: int,
                         total_epoch: int = 40) -> bool:
    ratio = visual_ratio_for_epoch(epoch_now, total_epoch)
    digest = hashlib.sha256(
        f"msd:{seed}:{epoch_now}:{sample_index}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return draw < ratio
```

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Expected: all curriculum tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/specforge/algorithms/msd \
        src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py
git commit -m "Add deterministic MSD curriculum"
```

## Task 2: MSD shifting, visual decoupling, noise, and losses

**Files:**
- Create: `src/train_Dflash_SpecForge/specforge/algorithms/msd/data.py`
- Create: `src/train_Dflash_SpecForge/specforge/algorithms/msd/model.py`
- Modify: `src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py`

- [ ] **Step 1: Add failing reference tests**

```python
def test_decouple_uses_projection_for_text_and_raw_embedding_for_visual():
    hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    next_embeddings = torch.tensor([[[10., 20.], [30., 40.], [50., 60.]]])
    visual_embeddings = torch.tensor([[[7., 8.], [9., 10.], [11., 12.]]])
    visual_mask = torch.tensor([[False, True, False]])
    projection = torch.nn.Linear(4, 2, bias=False)
    projection.weight.data.copy_(torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))
    actual = decouple_msd_inputs(
        hidden, next_embeddings, visual_embeddings, visual_mask, projection
    )
    assert torch.equal(actual[:, 1], visual_embeddings[:, 1])
    assert torch.equal(actual[:, 0], torch.tensor([[11., 22.]]))


def test_msd_loss_matches_literal_reference_and_masks_padding():
    predicted = torch.tensor([[[1., 2.], [2., 3.]]], requires_grad=True)
    target = torch.tensor([[[1.5, 1.5], [99., 99.]]])
    mask = torch.tensor([[1., 0.]])
    head = torch.nn.Linear(2, 3, bias=False)
    result = msd_loss(predicted, target, mask, head, 1.0, 0.1)
    expected_feature = torch.nn.functional.smooth_l1_loss(
        predicted[:, :1], target[:, :1], reduction="mean"
    )
    target_probs = head(target[:, :1]).softmax(-1)
    expected_soft = -(target_probs * head(predicted[:, :1]).log_softmax(-1)).sum(-1).mean()
    assert torch.allclose(result.loss, expected_feature + 0.1 * expected_soft)
```

Also assert shifting, invalid shape rejection, deterministic uniform noise,
finite backward gradients, and frozen-head gradients remain `None`.

- [ ] **Step 2: Run and confirm RED for missing APIs**

- [ ] **Step 3: Implement pure helpers and `MSDLossOutput`**

Implement:

```python
def shift_msd_record(raw: Mapping[str, Tensor], max_len: int) -> dict[str, Tensor]
def decouple_msd_inputs(target_hidden, next_embeddings, visual_embeddings,
                        visual_token_mask, projection) -> Tensor
def add_reference_uniform_noise(hidden, width, *, generator=None) -> Tensor
def msd_loss(predicted, target, loss_mask, frozen_lm_head,
             feature_weight=1.0, soft_weight=0.1) -> MSDLossOutput
```

All helpers validate batch/sequence/hidden shapes and normalize by valid mask
count. `msd_loss` evaluates the frozen head without detaching predicted hidden
states, so gradients reach the draft while head parameters remain frozen.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/specforge/algorithms/msd/{data,model}.py \
        src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py
git commit -m "Implement MSD tensor semantics and loss"
```

## Task 3: Configurable 1/3/5-layer draft model

**Files:**
- Create: `src/train_Dflash_SpecForge/specforge/modeling/draft/msd.py`
- Modify: `src/train_Dflash_SpecForge/specforge/modeling/draft/__init__.py`
- Create: `src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-msd.json`
- Modify: `src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py`

- [ ] **Step 1: Add failing model tests**

Use an 8-wide tiny `MSDConfig` with 2 heads, 1 KV head, intermediate size 16,
vocab 32, and M-RoPE sections `[1, 1, 2]`. Instantiate depths 1, 3, and 5;
assert exact layer counts, strictly increasing trainable parameter counts,
`[batch, sequence, hidden]` output, deterministic eval output, and support for
both `[batch, sequence]` and `[3, batch, sequence]` position IDs.

- [ ] **Step 2: Run and confirm RED because `MSDDraftModel` is unregistered**

- [ ] **Step 3: Implement the model**

`MSDConfig` subclasses `PretrainedConfig`. `MSDDraftModel` subclasses
`PreTrainedModel`, registers with `@register_draft`, owns frozen
`embed_tokens`, trainable `fusion_projection`, homogeneous decoder layers, and
final RMSNorm. The decoder uses causal SDPA, grouped-query attention, SwiGLU,
and Qwen2.5-VL three-axis rotary positions. Its public forward is:

```python
def forward(self, target_hidden, next_token_embeddings, visual_embeddings,
            visual_token_mask, attention_mask=None, position_ids=None):
    hidden = self.decouple_inputs(
        target_hidden=target_hidden,
        next_token_embeddings=next_token_embeddings,
        visual_embeddings=visual_embeddings,
        visual_token_mask=visual_token_mask,
    )
    for layer in self.layers:
        hidden = layer(hidden, attention_mask, position_ids)
    return self.norm(hidden)
```

- [ ] **Step 4: Run model tests and confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/specforge/modeling/draft/msd.py \
        src/train_Dflash_SpecForge/specforge/modeling/draft/__init__.py \
        src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-msd.json \
        src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py
git commit -m "Add configurable MSD draft model"
```

## Task 4: First-class SpecForge registration and training step

**Files:**
- Create: `src/train_Dflash_SpecForge/specforge/algorithms/msd/providers.py`
- Create: `src/train_Dflash_SpecForge/specforge/algorithms/msd/__init__.py`
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/builtin.py`
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/model_providers.py`
- Modify: `src/train_Dflash_SpecForge/specforge/training/strategies/base.py`
- Modify: `src/train_Dflash_SpecForge/specforge/config/schema.py`
- Modify: `src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py`
- Modify: `src/train_Dflash_SpecForge/tests/test_algorithms/test_builtin_parity.py`

- [ ] **Step 1: Add failing registry/config/step tests**

Assert:

```python
registration = builtin_algorithm_registry().resolve("msd")
assert registration.spec.draft.default_architecture == "MSDDraftModel"
assert registration.spec.capabilities.attention_backends == {"sdpa"}
assert registration.spec.modalities == {"multimodal"}
assert registration.spec.draft.fixed_override_values == ()
```

Parse the canonical recipe and reject depths outside `{1,3,5}`, total epochs
other than 40, negative loss weights, and invalid noise width. Build an
`MSDTrainStrategy` around a tiny model and assert one finite scalar loss plus
metrics `feature_loss`, `soft_target_loss`, and `accuracy`.

- [ ] **Step 2: Run and confirm RED**

- [ ] **Step 3: Register providers and add typed settings**

Add `TrainingConfig` fields:

```python
msd_feature_loss_weight: float = Field(default=1.0, ge=0.0)
msd_soft_loss_weight: float = Field(default=0.1, ge=0.0)
msd_noise_width: float = Field(default=0.2, ge=0.0)
msd_total_epochs: int = Field(default=40, gt=0)
msd_curriculum_seed: int = 0
```

The provider resume contract records these fields, depth, visual-ratio formula
version, and target dimensions. `MSDTrainStrategy.required_features` lists all
seven normalized tensors and filters checkpoints to trainable MSD draft state.

- [ ] **Step 4: Run focused and existing registry tests; confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/specforge/algorithms \
        src/train_Dflash_SpecForge/specforge/modeling/draft \
        src/train_Dflash_SpecForge/specforge/config/schema.py \
        src/train_Dflash_SpecForge/specforge/training/strategies/base.py \
        src/train_Dflash_SpecForge/tests/test_algorithms
git commit -m "Register MSD training strategy"
```

## Task 5: Offline MSD feature records and curriculum reader

**Files:**
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/msd/data.py`
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/msd/providers.py`
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/common/providers.py`
- Modify: `src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py`

- [ ] **Step 1: Add failing reader/collator tests**

Create temporary `text/` and `visual/` roots with tiny `.ckpt` records. Assert
the reader exposes the declared seven tensors, epoch 20 selects text only,
epoch 40 selects visual only, resumed sample indices reproduce choices, and
collation right-pads 2D tensors plus `[3, sequence]` M-RoPE tensors correctly.

- [ ] **Step 2: Run and confirm RED**

- [ ] **Step 3: Implement paired feature roots and epoch-aware reader**

Extend the offline provider seam with an MSD-owned reader accepting:

```text
data.hidden_states_path=/data/msd/features/sharegpt68k
data.msd_visual_hidden_states_path=/data/msd/features/llava68k
```

The reader length is 68,000 logical samples per epoch and delegates corpus
choice to `choose_visual_sample`. It emits source/corpus metadata and realized
mixture counters without mutating global RNG state.

- [ ] **Step 4: Run tests and confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/specforge/algorithms/msd \
        src/train_Dflash_SpecForge/specforge/algorithms/common/providers.py \
        src/train_Dflash_SpecForge/tests/test_algorithms/test_msd.py
git commit -m "Add MSD offline curriculum data path"
```

## Task 6: Capture Qwen2.5-VL embeddings, visual masks, and positions

**Files:**
- Modify: `src/train_Dflash_SpecForge/scripts/prepare_hidden_states.py`
- Modify: `src/train_Dflash_SpecForge/specforge/offline_capture/sglang_backend/capture_hooks.py`
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/common/providers.py`
- Modify: `src/train_Dflash_SpecForge/specforge/algorithms/msd/providers.py`
- Create: `src/train_Dflash_SpecForge/tests/test_offline_capture/test_msd_capture.py`

- [ ] **Step 1: Add failing capture-layout tests**

Feed synthetic capture sources with variable visual spans and assert persisted
records contain `input_embeddings`, `visual_token_mask`, and `position_ids` in
addition to target hidden states. Assert missing embeddings, mask/embedding
length mismatch, and non-three-axis multimodal positions fail before writing.

- [ ] **Step 2: Run and confirm RED**

- [ ] **Step 3: Extend capture sources and layout**

Add optional algorithm-owned captured tensors to `OfflineCaptureLayout`, while
leaving all existing layouts unchanged. For MSD/Qwen2.5-VL, hook the language
model input after visual embeddings replace image placeholder embeddings and
persist the model-produced M-RoPE positions plus an explicit image-token mask.

- [ ] **Step 4: Run MSD capture tests and existing offline-capture tests**

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/scripts/prepare_hidden_states.py \
        src/train_Dflash_SpecForge/specforge/offline_capture \
        src/train_Dflash_SpecForge/specforge/algorithms \
        src/train_Dflash_SpecForge/tests/test_offline_capture
git commit -m "Capture multimodal features for MSD"
```

## Task 7: Canonical recipe and automated depth sweep

**Files:**
- Create: `src/train_Dflash_SpecForge/examples/configs/qwen2.5-vl-3b-msd-68k-offline.yaml`
- Create: `src/train_Dflash_SpecForge/train_qwen25vl_msd_depth_sweep.sh`
- Create: `src/train_Dflash_SpecForge/tests/test_scripts/test_msd_depth_sweep_launcher.py`
- Modify: `src/train_Dflash_SpecForge/README_LOCAL.md`

- [ ] **Step 1: Write failing dry-run launcher tests**

Run the launcher with `--print-config` and assert it prints:

```text
TARGET_MODEL_PATH=/models/qwen25-vl-3b
DEPTHS=1,3,5
TOTAL_EPOCHS=40
GLOBAL_BATCH_SIZE=4
LEARNING_RATE=5e-5
TEXT_FEATURE_ROOT=/data/msd/features/sharegpt68k
VISUAL_FEATURE_ROOT=/data/msd/features/llava68k
depth1/output
depth3/output
depth5/output
```

Also assert duplicate/unsupported depths, shared output directories, and
missing required paths return code 2.

- [ ] **Step 2: Run and confirm RED because launcher is absent**

- [ ] **Step 3: Add recipe, launcher, and documentation**

The launcher supports `--phase data|capture|train|all`, `--resume`, and
`--print-config`; defaults to depths `1,3,5`; materializes private draft config
and output directories; and reuses immutable text/visual feature roots. It
never deletes or overwrites artifacts implicitly.

- [ ] **Step 4: Run launcher/config tests and confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add src/train_Dflash_SpecForge/examples/configs/qwen2.5-vl-3b-msd-68k-offline.yaml \
        src/train_Dflash_SpecForge/train_qwen25vl_msd_depth_sweep.sh \
        src/train_Dflash_SpecForge/tests/test_scripts/test_msd_depth_sweep_launcher.py \
        src/train_Dflash_SpecForge/README_LOCAL.md
git commit -m "Add automated MSD depth sweep"
```

## Task 8: Static verification and completion audit

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Run focused R1/R2 tests**

```powershell
py -3.11 -m pytest -q `
  tests/test_algorithms/test_msd.py `
  tests/test_offline_capture/test_msd_capture.py `
  tests/test_scripts/test_msd_depth_sweep_launcher.py
```

Expected artifact: complete console output with zero failures and no CUDA/model
downloads.

- [ ] **Step 2: Run affected SpecForge regression tests**

```powershell
py -3.11 -m pytest -q `
  tests/test_algorithms/test_builtin_parity.py `
  tests/test_algorithms/test_builtin_providers.py `
  tests/test_config/test_schema.py `
  tests/test_config/test_example_draft_config_wiring.py `
  tests/test_offline_capture/test_sglang_backend.py
```

Expected: zero failures.

- [ ] **Step 3: Run repository-level focused tests and static checks**

```powershell
py -3.11 -m pytest -q tests/test_workspace_routing.py
git diff --check
git status --short
```

Expected: tests pass, no whitespace errors, and status contains only intended
changes or is clean after commits.

- [ ] **Step 4: Audit every acceptance criterion**

Confirm with file/test evidence: first-class `msd`, depths 1/3/5 only, exact 40
ratios, visual bypass, reference loss, shared cache roots, isolated outputs,
resume contract, docs, and no empirical replication claims.

- [ ] **Step 5: Record the verified rung accurately**

Report: `Verified through R2` only if the tiny forward/backward test ran. If
environment dependency constraints prevent R2, report the highest green rung
and the exact failing command; never infer success from source inspection.
