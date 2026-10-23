"""Regression tests for the Qwen2.5-VL three-layer DFlash launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT_DIR / "train_qwen25vl_dflash_3layer.sh"
BASH_BIN = "bash"
if os.name == "nt":
    BASH_LAUNCHER = f"/{LAUNCHER.drive[0].lower()}{LAUNCHER.as_posix()[2:]}"
    if Path(r"C:\Program Files\Git\bin\bash.exe").exists():
        BASH_BIN = r"C:\Program Files\Git\bin\bash.exe"
else:
    BASH_LAUNCHER = str(LAUNCHER)


def run_print_config(*args: str, env: dict[str, str] | None = None):
    process_env = {**os.environ, "SPECFORGE_SKIP_GPU_CHECK": "1"}
    if env:
        process_env.update(env)
    return subprocess.run(
        [BASH_BIN, BASH_LAUNCHER, "--print-config", *args],
        env=process_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_high_only_preset_resolves_to_three_layer_draft_and_two_gpus():
    result = run_print_config("--set", "high_only")

    assert result.returncode == 0, result.stderr
    assert "TARGET_LAYER_IDS=26,28,30,32,34" in result.stdout
    assert "SPECFORGE_DFLASH_DEPTH=3" in result.stdout
    assert "SPECFORGE_GPUS=2" in result.stdout
    assert "SPECFORGE_GLOBAL_BATCH_SIZE=64" in result.stdout
    assert "SPECFORGE_MICRO_BATCH_SIZE=16" in result.stdout
    assert "SPECFORGE_PHASE1_MAX_LENGTH=2048" in result.stdout
    assert "SPECFORGE_PHASE2_MAX_LENGTH=3072" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0,1" in result.stdout
    assert "outputs/dflash_3layer_26_28_30_32_34" in result.stdout
    assert "SPECFORGE_PHASE1_TARGET_LAYER_IDS=26,28,30,32,34" in result.stdout
    assert "SPECFORGE_PHASE2_TARGET_LAYER_IDS=26,28,30,32,34" in result.stdout


def test_all_supported_presets_preserve_layer_order_and_isolate_artifacts():
    expected = {
        "middle_high": "17,19,25,33,34",
        "middle_high_2": "17,28,30,32,34",
        "shallow_high": "1,28,30,32,34",
    }

    for preset, layer_ids in expected.items():
        result = run_print_config("--set", preset)

        assert result.returncode == 0, result.stderr
        assert f"TARGET_LAYER_IDS={layer_ids}" in result.stdout
        slug = layer_ids.replace(",", "_")
        assert f"artifacts/dflash_3layer_{slug}" in result.stdout
        assert f"outputs/dflash_3layer_{slug}" in result.stdout


def test_custom_layer_option_is_applied_to_both_phases():
    result = run_print_config("--layer", "2,8,16,24,32")

    assert result.returncode == 0, result.stderr
    assert "TARGET_LAYER_IDS=2,8,16,24,32" in result.stdout
    assert "SPECFORGE_PHASE1_TARGET_LAYER_IDS=2,8,16,24,32" in result.stdout
    assert "SPECFORGE_PHASE2_TARGET_LAYER_IDS=2,8,16,24,32" in result.stdout
    assert "outputs/dflash_3layer_2_8_16_24_32" in result.stdout


def test_phase_roots_and_checkpoint_are_layer_specific():
    result = run_print_config(
        "--set",
        "high_only",
        env={"ARTIFACT_ROOT": "/tmp/artifacts", "OUTPUT_ROOT": "/tmp/outputs"},
    )

    assert result.returncode == 0, result.stderr
    assert "PHASE1_ARTIFACT_ROOT=/tmp/artifacts/dflash_3layer_26_28_30_32_34/phase1" in result.stdout
    assert "PHASE2_ARTIFACT_ROOT=/tmp/artifacts/dflash_3layer_26_28_30_32_34/phase2" in result.stdout
    assert "PHASE1_FEATURE_ROOT=/tmp/artifacts/dflash_3layer_26_28_30_32_34/phase1/qwen25vl_3b/hidden_states" in result.stdout
    assert (
        "PHASE1_CHECKPOINT=/tmp/outputs/dflash_3layer_26_28_30_32_34/phase1/"
        "qwen25vl-3b-dflash-sharegpt68k-dflash_3layer_26_28_30_32_34/"
        "qwen25vl-3b-dflash-sharegpt68k-dflash_3layer_26_28_30_32_34-latest"
    ) in result.stdout
    assert "PHASE1_COMMAND=bash " in result.stdout
    assert "train_qwen25vl_dflash_sharegpt_68k.sh --models 3b --phase all --gpus 2" in result.stdout
    assert "train_qwen25vl_dflash_llava_68k.sh --phase all --gpus 2" in result.stdout


def test_phase2_can_select_checkpoint_but_full_run_cannot_mix_checkpoints():
    phase2 = run_print_config(
        "--set",
        "high_only",
        "--phase",
        "phase2",
        "--inner-phase",
        "train",
        env={"PHASE1_CHECKPOINT": "/models/phase1-latest"},
    )
    mixed = run_print_config(
        "--set",
        "high_only",
        env={"PHASE1_CHECKPOINT": "/models/stale-checkpoint"},
    )

    assert phase2.returncode == 0, phase2.stderr
    assert "PHASE1_CHECKPOINT=/models/phase1-latest" in phase2.stdout
    assert "PHASE1_COMMAND=" not in phase2.stdout
    assert "PHASE2_COMMAND=" in phase2.stdout
    assert mixed.returncode == 2
    assert "Phase 1 checkpoint is fixed to this run's output" in mixed.stderr


def test_launcher_rejects_ambiguous_or_invalid_layer_selection():
    both = run_print_config("--set", "high_only", "--layer", "1,2,3,4,5")
    duplicate = run_print_config("--layer", "1,2,3,4,4")
    last_layer = run_print_config("--layer", "17,19,25,33,35")

    assert both.returncode == 2
    assert "choose exactly one of --set or --layer" in both.stderr
    assert duplicate.returncode == 2
    assert "unique" in duplicate.stderr
    assert last_layer.returncode == 2
    assert "following decoder boundary" in last_layer.stderr


def test_launcher_requires_exactly_two_distinct_gpu_ids():
    duplicate = run_print_config("--set", "high_only", "--gpu-ids", "0,0")
    single = run_print_config("--set", "high_only", "--gpu-ids", "0")

    assert duplicate.returncode == 2
    assert "two distinct GPU IDs" in duplicate.stderr
    assert single.returncode == 2
    assert "exactly two GPU IDs" in single.stderr


def test_launcher_rejects_unknown_set_and_phase():
    unknown_set = run_print_config("--set", "unknown")
    unknown_phase = run_print_config("--set", "high_only", "--phase", "capture")

    assert unknown_set.returncode == 2
    assert "unknown layer set" in unknown_set.stderr
    assert unknown_phase.returncode == 2
    assert "--phase must be all, phase1, or phase2" in unknown_phase.stderr


def test_launcher_reports_missing_option_values_as_usage_errors():
    missing_layer = run_print_config("--layer")
    missing_gpu_ids = run_print_config("--set", "high_only", "--gpu-ids")

    assert missing_layer.returncode == 2
    assert "--layer requires a value" in missing_layer.stderr
    assert missing_gpu_ids.returncode == 2
    assert "--gpu-ids requires a value" in missing_gpu_ids.stderr
