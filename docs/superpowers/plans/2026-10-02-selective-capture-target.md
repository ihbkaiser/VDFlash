# Selective Capture Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--capture-target both|text|llava` so the MSD launcher can capture only LLaVA features while preserving the existing default.

**Architecture:** Parse and validate one launcher-level selector, then conditionally validate, guard, create, and execute only the selected capture pipeline. Keep data preparation and training behavior unchanged.

**Tech Stack:** Bash, Python, pytest

---

### Task 1: Lock selective-capture behavior with regression tests

**Files:**
- Modify: `src/train_Dflash_SpecForge/tests/test_scripts/test_msd_depth_sweep_launcher.py`

- [ ] **Step 1: Write the failing LLaVA-only test**

Add a test that passes `--capture-target llava`, points `SHAREGPT_JSONL` at a missing path, and asserts that only `prepare_llava_caption_hidden_states.py` is invoked.

- [ ] **Step 2: Write text-only and invalid-value tests**

Add tests asserting that `text` invokes only `prepare_hidden_states.py`, and an unknown target exits with code 2 before capture commands run.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py -k capture_target
```

Expected: failures because `--capture-target` is currently unknown.

### Task 2: Implement capture target selection

**Files:**
- Modify: `src/train_Dflash_SpecForge/train_qwen25vl_msd_depth_sweep.sh`

- [ ] **Step 1: Parse and validate the flag**

Add `CAPTURE_TARGET=both`, parse `--capture-target VALUE`, document it in usage, and reject values outside `both|text|llava` with exit code 2.

- [ ] **Step 2: Scope validation and guards**

For capture/all phases, validate and guard `SHAREGPT_JSONL`/`TEXT_FEATURE_ROOT` only for `both|text`; validate and guard `LLAVA_MANIFEST`/`IMAGE_ROOT`/`VISUAL_FEATURE_ROOT` only for `both|llava`.

- [ ] **Step 3: Scope capture subprocesses**

Run `prepare_hidden_states.py` only for `both|text`, and `prepare_llava_caption_hidden_states.py` only for `both|llava`. Configure NVRTC once before either selected subprocess.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py
bash -n train_qwen25vl_msd_depth_sweep.sh
```

Expected: all launcher tests pass and Bash syntax exits 0.

### Task 3: Commit and publish

**Files:**
- Modify: `src/train_Dflash_SpecForge/train_qwen25vl_msd_depth_sweep.sh`
- Modify: `src/train_Dflash_SpecForge/tests/test_scripts/test_msd_depth_sweep_launcher.py`

- [ ] **Step 1: Check the final diff**

Run `git diff --check` and confirm only the launcher, its tests, and approved plan artifacts changed.

- [ ] **Step 2: Commit and push**

```bash
git add src/train_Dflash_SpecForge/train_qwen25vl_msd_depth_sweep.sh \
  src/train_Dflash_SpecForge/tests/test_scripts/test_msd_depth_sweep_launcher.py
git commit -m "Add selective MSD capture targets"
git push origin feature/msd-qwen25vl3b
```

- [ ] **Step 3: Report the immutable update command**

Provide the pushed commit hash and the exact `--phase capture --capture-target llava --resume` command.

