"""Regression tests for the dedicated two-B200 hidden-layer launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT_DIR / "train_qwen25vl_dflash_layers_25_27_29_31_33.sh"


def run_print_config(*args: str, env: dict[str, str] | None = None):
    process_env = {**os.environ, "SPECFORGE_SKIP_GPU_CHECK": "1"}
    if env:
        process_env.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--print-config", *args],
        env=process_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_launcher_prints_shared_layers_and_two_b200_profile():
    result = run_print_config()

    assert result.returncode == 0, result.stderr
    assert "TARGET_LAYER_IDS=25,27,29,31,33" in result.stdout
    assert "SPECFORGE_PHASE1_TARGET_LAYER_IDS=25,27,29,31,33" in result.stdout
    assert "SPECFORGE_PHASE2_TARGET_LAYER_IDS=25,27,29,31,33" in result.stdout
    assert "SPECFORGE_GPUS=2" in result.stdout
    assert "SPECFORGE_GLOBAL_BATCH_SIZE=64" in result.stdout
    assert "SPECFORGE_MICRO_BATCH_SIZE=16" in result.stdout
    assert "SPECFORGE_FSDP_SHARDING=NO_SHARD" in result.stdout
    assert "SPECFORGE_ATTENTION_BACKEND=flex_attention" in result.stdout
    assert "SPECFORGE_OBJECTIVE_CHUNK_BLOCKS=256" in result.stdout
    assert "train_qwen25vl_dflash_sharegpt_68k.sh" in result.stdout
    assert "train_qwen25vl_dflash_llava_68k.sh" in result.stdout


def test_launcher_keeps_phase_artifacts_and_outputs_separate():
    result = run_print_config(
        env={
            "ARTIFACT_ROOT": "/tmp/custom-artifacts",
            "OUTPUT_ROOT": "/tmp/custom-outputs",
        }
    )

    assert result.returncode == 0, result.stderr
    assert "PHASE1_ARTIFACT_ROOT=/tmp/custom-artifacts/phase1" in result.stdout
    assert "PHASE2_ARTIFACT_ROOT=/tmp/custom-artifacts/phase2" in result.stdout
    assert "PHASE1_OUTPUT_ROOT=/tmp/custom-outputs/phase1" in result.stdout
    assert "PHASE2_OUTPUT_ROOT=/tmp/custom-outputs/phase2" in result.stdout


def test_launcher_applies_one_override_to_both_phases():
    result = run_print_config(
        env={"SPECFORGE_TARGET_LAYER_IDS": "23,25,27,29,31"}
    )

    assert result.returncode == 0, result.stderr
    assert "TARGET_LAYER_IDS=23,25,27,29,31" in result.stdout
    assert "SPECFORGE_PHASE1_TARGET_LAYER_IDS=23,25,27,29,31" in result.stdout
    assert "SPECFORGE_PHASE2_TARGET_LAYER_IDS=23,25,27,29,31" in result.stdout


def test_launcher_rejects_invalid_wrapper_phase():
    result = run_print_config("--phase", "capture")

    assert result.returncode == 2
    assert "--phase must be all, phase1, or phase2" in result.stderr
