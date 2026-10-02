# MSD Shared-Storage Depth Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `train_qwen25vl_msd_depth_sweep.sh` a one-command, resumable ShareGPT-68K/LLaVA-68K feature-capture and 1/3/5-depth 40-epoch training launcher using the real shared-storage paths.

**Architecture:** Keep the existing single Bash orchestrator and Python config materializer. The launcher resolves one shared storage root, prepares two immutable manifests, captures one text and one visual feature cache, then trains isolated depth directories sequentially; shell-level guards prevent accidental cache/checkpoint reuse while `--resume` permits intentional continuation.

**Tech Stack:** Bash, Python 3.11, pytest, PyYAML, SpecForge CLI, PyTorch `torchrun`.

---

## Experiment card and verification ladder

**Question:** Can one deterministic launcher execute the original-MSD Qwen2.5-VL-3B 1/3/5-depth protocol without path ambiguity or cross-depth artifact collisions?

**Hypothesis:** Reusing immutable teacher features while isolating generated configs and checkpoints by depth will preserve the exact 40-epoch curriculum and make interrupted runs safely resumable.

**Baseline:** Commit `0601726`, `train_qwen25vl_msd_depth_sweep.sh`, with generic `/data` and `/models` defaults.

**Variant:** Replace the generic paths with the approved shared-storage layout and add manifest preparation plus artifact/resume guards; model, loss, curriculum, optimizer, and seed remain unchanged.

**Primary metric:** All three depth runs reach epoch 40 with `status=ok` under the generated recipes. **Success:** depth 1, 3, and 5 each produce a final checkpoint and retain `msd_total_epochs=40`.

**Guardrails:** Exactly 68,000 records per corpus; no cache overwrite without `--resume`; unique generated/output directories; finite loss; no change to curriculum ratios or model dimensions.

**Data / split:** ShareGPT text and LLaVA captions are training corpora. No held-out-quality claim is made; evaluation is outside this launcher change.

**Seeds:** Fixed curriculum/training seed `0` for this replication sweep.

**Budget:** No GPU run is authorized by this implementation task. The operator chooses the later cluster budget before R4-R6.

**Start rung:** R1 static/config verification.

**Exploratory-only:** Loss curves, speed, memory, and acceptance behavior from later runs cannot establish paper reproduction without a separate evaluation protocol.

Verification gates:

- R0: this card and the approved design document.
- R1: focused pytest, `bash -n`, and `--print-config`; artifacts are clean test output and the resolved config dump.
- R2: already established for synthetic MSD forward/backward on T4; it does not validate this real-data launcher.
- R3: future tiny-cache overfit; artifact must be a collapsing loss log plus checkpoint.
- R4: future real launcher smoke with a tiny sample/step cap; artifact must include capture logs, a checkpoint, and a successful resume log.
- R5: future short real-data pilot; artifact must be a stable early loss/accuracy curve for all depths.
- R6: future full 40-epoch sweep; artifact must contain all three terminal checkpoints and complete metrics.

Do not promote past a failed rung. This plan implements and verifies R1 only.

## File map

- Modify `train_qwen25vl_msd_depth_sweep.sh`: resolve shared paths, prepare both corpora, guard artifacts, run capture and sequential training.
- Modify `tests/test_scripts/test_msd_depth_sweep_launcher.py`: execute dry-run/data/capture/train shell paths with temporary files and command stubs.
- Modify `README_LOCAL.md`: publish the exact one-command invocation, defaults, phase recovery, and verification boundary.
- Keep `scripts/materialize_msd_sweep.py` unchanged: it already enforces immutable per-depth configs and the 40-epoch schedule.

### Task 1: Lock shared-storage defaults in executable tests

**Files:**
- Modify: `tests/test_scripts/test_msd_depth_sweep_launcher.py`
- Modify: `train_qwen25vl_msd_depth_sweep.sh:1-95`

- [ ] **Step 1: Write the failing dry-run tests**

Add a subprocess helper and assert the approved defaults:

```python
import os
import subprocess


SHARED_ROOT = "/workspace/storage-shared/nlp/tungdd11/tungdecoder"


def run_launcher(*args: str, env: dict[str, str] | None = None):
    process_env = {**os.environ}
    if env:
        process_env.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        env=process_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_msd_launcher_prints_shared_storage_defaults() -> None:
    result = run_launcher("--print-config")
    assert result.returncode == 0, result.stderr
    assert f"TARGET_MODEL_PATH={SHARED_ROOT}/models/qwen25-vl-3b" in result.stdout
    assert (
        f"SHAREGPT_SOURCE={SHARED_ROOT}/ShareGPT/"
        "ShareGPT_V3_unfiltered_cleaned_split.json" in result.stdout
    )
    assert (
        f"LLAVA_SOURCE_JSONL={SHARED_ROOT}/data/"
        "llava_dflash_qwen25vl3b_68k/llava_dflash_68k_clean_3b.jsonl"
        in result.stdout
    )
    assert f"IMAGE_ROOT={SHARED_ROOT}/LlaVA-Pretrain" in result.stdout
    assert "DEPTHS=1,3,5" in result.stdout
    assert "TOTAL_EPOCHS=40" in result.stdout
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py::test_msd_launcher_prints_shared_storage_defaults
```

Expected: FAIL because the launcher still prints `/models` and `/data` defaults and omits `LLAVA_SOURCE_JSONL`.

- [ ] **Step 3: Implement shared-root resolution and complete dry-run output**

Replace the generic path block with:

```bash
SHARED_STORAGE_ROOT=${SPECFORGE_SHARED_STORAGE_ROOT:-/workspace/storage-shared/nlp/tungdd11/tungdecoder}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-"$SHARED_STORAGE_ROOT/artifacts/qwen25vl_msd_68k"}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-"$SHARED_STORAGE_ROOT/models/qwen25-vl-3b"}
SHAREGPT_SOURCE=${SHAREGPT_SOURCE:-"$SHARED_STORAGE_ROOT/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"}
LLAVA_SOURCE_JSONL=${LLAVA_SOURCE_JSONL:-"$SHARED_STORAGE_ROOT/data/llava_dflash_qwen25vl3b_68k/llava_dflash_68k_clean_3b.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-"$SHARED_STORAGE_ROOT/LlaVA-Pretrain"}
SHAREGPT_JSONL=${SHAREGPT_JSONL:-"$ARTIFACT_ROOT/manifests/sharegpt_train.jsonl"}
LLAVA_MANIFEST=${LLAVA_MANIFEST:-"$ARTIFACT_ROOT/manifests/llava68k.jsonl"}
TEXT_FEATURE_ROOT=${TEXT_FEATURE_ROOT:-"$ARTIFACT_ROOT/features/sharegpt68k"}
VISUAL_FEATURE_ROOT=${VISUAL_FEATURE_ROOT:-"$ARTIFACT_ROOT/features/llava68k"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ARTIFACT_ROOT/outputs"}
GENERATED_ROOT=${GENERATED_ROOT:-"$ARTIFACT_ROOT/generated"}
TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
```

Extend `print_config` to emit every input, artifact, topology, and per-depth output path. Validate positive GPU/micro/global batch values before arithmetic.

- [ ] **Step 4: Run tests and shell syntax verification**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py
bash -n train_qwen25vl_msd_depth_sweep.sh
bash train_qwen25vl_msd_depth_sweep.sh --print-config
```

Expected: all launcher tests PASS; syntax exits 0; config contains the exact five approved shared inputs and depth outputs.

- [ ] **Step 5: Commit**

```bash
git add tests/test_scripts/test_msd_depth_sweep_launcher.py train_qwen25vl_msd_depth_sweep.sh
git commit -m "Set MSD shared storage defaults"
```

### Task 2: Prepare and validate both 68K corpora

**Files:**
- Modify: `tests/test_scripts/test_msd_depth_sweep_launcher.py`
- Modify: `train_qwen25vl_msd_depth_sweep.sh:96-125`
- Verify: `tests/test_data/test_llava_caption_manifest.py`

- [ ] **Step 1: Write failing data-phase command tests**

Add a temporary-input test that uses `/bin/echo` instead of Python:

```python
def test_msd_data_phase_prepares_sharegpt_and_llava(tmp_path: Path) -> None:
    sharegpt = tmp_path / "sharegpt.json"
    llava = tmp_path / "llava.jsonl"
    image_root = tmp_path / "images"
    artifact_root = tmp_path / "artifacts"
    sharegpt.write_text("[]", encoding="utf-8")
    llava.write_text("{}\n", encoding="utf-8")
    image_root.mkdir()
    manifests = artifact_root / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "sharegpt_train.jsonl").write_text("{}\n", encoding="utf-8")
    (manifests / "llava68k.jsonl").write_text("{}\n", encoding="utf-8")
    result = run_launcher(
        "--phase", "data",
        env={
            "ARTIFACT_ROOT": str(artifact_root),
            "SHAREGPT_SOURCE": str(sharegpt),
            "LLAVA_SOURCE_JSONL": str(llava),
            "IMAGE_ROOT": str(image_root),
            "PYTHON_BIN": "/bin/echo",
            "SPECFORGE_NUM_SAMPLES": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_data.py --dataset sharegpt" in result.stdout
    assert "scripts/prepare_llava_caption_manifest.py" in result.stdout
    assert f"--image-root {image_root}" in result.stdout
    assert "--expected-records 1" in result.stdout
```

Also add a test with a missing `LLAVA_SOURCE_JSONL` and assert exit 2 before either preparation command runs.

- [ ] **Step 2: Run the data tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py -k "data_phase"
```

Expected: FAIL because the launcher does not invoke `prepare_llava_caption_manifest.py` or validate the LLaVA source during `--phase data`.

- [ ] **Step 3: Implement deterministic two-corpus preparation**

In the data phase, use absolute repository paths and create the manifest directory:

```bash
mkdir -p "$(dirname "$SHAREGPT_JSONL")" "$(dirname "$LLAVA_MANIFEST")"
[[ -f "$SHAREGPT_SOURCE" ]] || { echo "missing SHAREGPT_SOURCE: $SHAREGPT_SOURCE" >&2; exit 2; }
[[ -f "$LLAVA_SOURCE_JSONL" ]] || { echo "missing LLAVA_SOURCE_JSONL: $LLAVA_SOURCE_JSONL" >&2; exit 2; }
[[ -d "$IMAGE_ROOT" ]] || { echo "missing IMAGE_ROOT: $IMAGE_ROOT" >&2; exit 2; }
"$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_data.py" \
  --dataset sharegpt --data-path "$SHAREGPT_SOURCE" \
  --output-path "$(dirname "$SHAREGPT_JSONL")" --sample-size "$EXPECTED_RECORDS"
"$PYTHON_BIN" "$SPECFORGE_DIR/scripts/prepare_llava_caption_manifest.py" \
  --input "$LLAVA_SOURCE_JSONL" --output "$LLAVA_MANIFEST" \
  --image-root "$IMAGE_ROOT" --expected-records "$EXPECTED_RECORDS"
```

The ShareGPT converter's existing limit flag is `--sample-size`; do not pass the capture script's unrelated `--num-samples` spelling here. After both commands, enforce the exact nonblank JSONL row count:

```bash
require_jsonl_count() {
  local label=$1 path=$2 expected=$3 count
  [[ -f "$path" ]] || { echo "missing $label output: $path" >&2; exit 1; }
  count=$(awk 'NF { count += 1 } END { print count + 0 }' "$path")
  if ((count != expected)); then
    echo "$label produced $count records; expected $expected" >&2
    exit 1
  fi
}

require_jsonl_count ShareGPT "$SHAREGPT_JSONL" "$EXPECTED_RECORDS"
require_jsonl_count LLaVA "$LLAVA_MANIFEST" "$EXPECTED_RECORDS"
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py -k "data_phase"
python -m pytest -q tests/test_data/test_llava_caption_manifest.py
```

Expected: PASS, with both commands containing absolute script paths and the same sample count; the manifest test proves a relative image such as `00001/one.jpg` resolves under the supplied image root and missing images fail.

- [ ] **Step 5: Commit**

```bash
git add tests/test_scripts/test_msd_depth_sweep_launcher.py train_qwen25vl_msd_depth_sweep.sh
git commit -m "Prepare MSD text and visual manifests"
```

### Task 3: Guard and resume shared feature capture

**Files:**
- Modify: `tests/test_scripts/test_msd_depth_sweep_launcher.py`
- Modify: `train_qwen25vl_msd_depth_sweep.sh:126-160`

- [ ] **Step 1: Write failing capture and collision tests**

Use temporary prepared manifests, image root, and command stubs:

```python
def test_msd_capture_runs_text_then_visual_commands(tmp_path: Path) -> None:
    sharegpt = tmp_path / "sharegpt_train.jsonl"
    llava = tmp_path / "llava68k.jsonl"
    images = tmp_path / "images"
    target = tmp_path / "target"
    sharegpt.write_text("{}\n", encoding="utf-8")
    llava.write_text("{}\n", encoding="utf-8")
    images.mkdir()
    target.mkdir()
    result = run_launcher(
        "--phase", "capture",
        env={
            "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
            "SHAREGPT_JSONL": str(sharegpt),
            "LLAVA_MANIFEST": str(llava),
            "IMAGE_ROOT": str(images),
            "TARGET_MODEL_PATH": str(target),
            "TORCHRUN_BIN": "/bin/echo",
        },
    )
    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_hidden_states.py --strategy msd" in result.stdout
    assert "scripts/prepare_llava_caption_hidden_states.py --strategy msd" in result.stdout
```

Add another test that creates `features/sharegpt68k/rows_0-2000/data_0.ckpt`, runs capture without `--resume`, and expects exit 1 with `pass --resume`. Run again with `--resume` and expect both capture commands to be constructed.

- [ ] **Step 2: Run the capture tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py -k "capture"
```

Expected: FAIL because `TORCHRUN_BIN` is not used and existing feature artifacts are not guarded.

- [ ] **Step 3: Implement capture guards and resumable commands**

Add:

```bash
feature_count() {
  local root=$1
  [[ -d "$root" ]] || { echo 0; return; }
  find "$root" -type f \( -name '*.ckpt' -o -name '*.ckpt.gz' \) -print 2>/dev/null | wc -l
}

guard_capture_root() {
  local label=$1 root=$2 count
  count=$(feature_count "$root")
  if ((count > 0 && !RESUME)); then
    echo "$label feature cache already contains $count records at $root; pass --resume" >&2
    exit 1
  fi
}
```

Require the target directory and prepared inputs, call both guards, create the feature roots, and replace literal `torchrun` with `"$TORCHRUN_BIN"`. Keep capture depth-independent and omit `--overwrite`; both capture utilities already skip existing records atomically.

- [ ] **Step 4: Run capture tests and shell syntax**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py -k "capture"
bash -n train_qwen25vl_msd_depth_sweep.sh
```

Expected: PASS; capture without resume rejects existing rows, while resume preserves them and constructs both commands.

- [ ] **Step 5: Commit**

```bash
git add tests/test_scripts/test_msd_depth_sweep_launcher.py train_qwen25vl_msd_depth_sweep.sh
git commit -m "Guard resumable MSD feature capture"
```

### Task 4: Isolate, complete, and resume depth training

**Files:**
- Modify: `tests/test_scripts/test_msd_depth_sweep_launcher.py`
- Modify: `train_qwen25vl_msd_depth_sweep.sh:161-205`

- [ ] **Step 1: Write failing sequential-training lifecycle tests**

Add a shared environment helper and a test with temporary feature directories and `PYTHON_BIN=/bin/echo`:

```python
def train_env(tmp_path: Path) -> dict[str, str]:
    text_features = tmp_path / "text-features"
    visual_features = tmp_path / "visual-features"
    target = tmp_path / "target"
    text_features.mkdir()
    visual_features.mkdir()
    target.mkdir()
    return {
        "TARGET_MODEL_PATH": str(target),
        "TEXT_FEATURE_ROOT": str(text_features),
        "VISUAL_FEATURE_ROOT": str(visual_features),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "GENERATED_ROOT": str(tmp_path / "generated"),
        "PYTHON_BIN": "/bin/echo",
    }


def test_msd_train_runs_depths_in_order_and_marks_completion(tmp_path: Path) -> None:
    result = run_launcher("--phase", "train", env=train_env(tmp_path))
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("--depth 1") < result.stdout.index("--depth 3")
    assert result.stdout.index("--depth 3") < result.stdout.index("--depth 5")
    for depth in (1, 3, 5):
        assert f"generated/depth{depth}/train.yaml" in result.stdout
        assert (tmp_path / "outputs" / f"depth{depth}" / ".complete").is_file()
```

The successful stubbed run must create:

```text
outputs/depth1/.complete
outputs/depth3/.complete
outputs/depth5/.complete
```

Add two concrete lifecycle cases:

```python
def test_msd_train_rejects_existing_unfinished_output_without_resume(
    tmp_path: Path,
) -> None:
    env = train_env(tmp_path)
    partial = tmp_path / "outputs" / "depth1" / "output"
    partial.mkdir(parents=True)
    (partial / "partial-file").touch()
    result = run_launcher("--phase", "train", env=env)
    assert result.returncode == 1
    assert "pass --resume" in result.stderr


def test_msd_train_resume_skips_completed_depth(tmp_path: Path) -> None:
    env = train_env(tmp_path)
    marker = tmp_path / "outputs" / "depth1" / ".complete"
    marker.parent.mkdir(parents=True)
    marker.touch()
    result = run_launcher("--phase", "train", "--resume", env=env)
    assert result.returncode == 0, result.stderr
    assert "[train:depth1] complete; skipping" in result.stdout
    assert "generated/depth3/train.yaml" in result.stdout
    assert "generated/depth5/train.yaml" in result.stdout
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py -k "train_"
```

Expected: FAIL because the current loop neither guards output directories nor writes completion markers.

- [ ] **Step 3: Implement per-depth lifecycle handling**

Inside the depth loop use:

```bash
depth_root="$OUTPUT_ROOT/depth${depth}"
output="$depth_root/output"
complete_marker="$depth_root/.complete"
if ((RESUME)) && [[ -f "$complete_marker" ]]; then
  echo "[train:depth${depth}] complete; skipping"
  continue
fi
if ((!RESUME)) && [[ -d "$output" ]] && find "$output" -mindepth 1 -print -quit | grep -q .; then
  echo "depth${depth} output already exists at $output; pass --resume" >&2
  exit 1
fi
resume_args=()
if ((RESUME)) && [[ -d "$output" ]]; then
  resume_args+=("training.resume_from=$output")
fi
"$PYTHON_BIN" -m specforge.cli train --config "$config" "${resume_args[@]}"
mkdir -p "$depth_root"
touch "$complete_marker"
```

Keep materialization before lifecycle launch so resume verifies the immutable recipe. A failed trainer command must stop under `set -e` before the marker is written.

- [ ] **Step 4: Run all launcher/materializer tests**

Run:

```bash
python -m pytest -q tests/test_scripts/test_msd_depth_sweep_launcher.py
```

Expected: PASS with isolated recipes, ordered runs, collision rejection, and completed-depth skipping.

- [ ] **Step 5: Commit**

```bash
git add tests/test_scripts/test_msd_depth_sweep_launcher.py train_qwen25vl_msd_depth_sweep.sh
git commit -m "Add resumable MSD depth lifecycle"
```

### Task 5: Document the exact command and run R1 regression verification

**Files:**
- Modify: `README_LOCAL.md`
- Verify: `tests/test_scripts/test_msd_depth_sweep_launcher.py`
- Verify: `tests/test_algorithms/test_msd.py`
- Verify: `tests/test_modeling/test_msd_draft.py`
- Verify: `tests/test_offline_capture/test_msd_capture.py`
- Verify: `tests/test_data/test_llava_caption_manifest.py`

- [ ] **Step 1: Update operator documentation**

Replace generic override guidance with the exact default command:

```bash
cd /workspace/VDFlash/src/train_Dflash_SpecForge
bash train_qwen25vl_msd_depth_sweep.sh --print-config
bash train_qwen25vl_msd_depth_sweep.sh --phase all
```

Document recovery commands:

```bash
bash train_qwen25vl_msd_depth_sweep.sh --phase data
bash train_qwen25vl_msd_depth_sweep.sh --phase capture --resume
bash train_qwen25vl_msd_depth_sweep.sh --phase train --resume
```

State that shared defaults can be moved with `SPECFORGE_SHARED_STORAGE_ROOT`, that the three depths run sequentially, and that R1 does not authorize or claim a full training result.

- [ ] **Step 2: Run formatting/static checks**

Run:

```bash
bash -n train_qwen25vl_msd_depth_sweep.sh
git diff --check
bash train_qwen25vl_msd_depth_sweep.sh --print-config
```

Expected: all commands exit 0; no whitespace errors; config shows shared paths, 40 epochs, and depths 1/3/5.

- [ ] **Step 3: Run focused and MSD regression suites**

Run:

```bash
python -m pytest -q \
  tests/test_scripts/test_msd_depth_sweep_launcher.py \
  tests/test_algorithms/test_msd.py \
  tests/test_modeling/test_msd_draft.py \
  tests/test_offline_capture/test_msd_capture.py \
  tests/test_data/test_llava_caption_manifest.py
```

Expected: PASS. Proof artifact is the captured pytest output plus the `--print-config` dump.

- [ ] **Step 4: Confirm scope and repository state**

Run:

```bash
git status --short
git diff --stat HEAD~4..HEAD
```

Expected: only the planned launcher, tests, README, spec, and plan are changed. Report “verified through R1”; do not claim real-data capture or 40-epoch completion.

- [ ] **Step 5: Commit**

```bash
git add README_LOCAL.md
git commit -m "Document automated MSD depth sweep"
```
