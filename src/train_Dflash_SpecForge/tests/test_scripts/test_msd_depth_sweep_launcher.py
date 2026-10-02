from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from specforge.config import Config


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "train_qwen25vl_msd_depth_sweep.sh"
MATERIALIZER = ROOT / "scripts" / "materialize_msd_sweep.py"
BASE_DRAFT = ROOT / "configs" / "qwen2.5-vl-3b-msd.json"
BASE_RECIPE = ROOT / "examples" / "configs" / "qwen2.5-vl-3b-msd-68k-offline.yaml"
SHARED_ROOT = "/workspace/storage-shared/nlp/tungdd11/tungdecoder"
BASH = r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else "bash"


def run_launcher(*args: str, env: dict[str, str] | None = None):
    process_env = {**os.environ}
    if env:
        process_env.update(env)
    return subprocess.run(
        [BASH, LAUNCHER.as_posix(), *args],
        env=process_env,
        check=False,
        capture_output=True,
        text=True,
    )


def _module():
    spec = importlib.util.spec_from_file_location("materialize_msd_sweep", MATERIALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_msd_launcher_declares_exact_static_sweep_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    for expected in (
        "DEPTHS=1,3,5",
        "TOTAL_EPOCHS=40",
        "GLOBAL_BATCH_SIZE=4",
        "LEARNING_RATE=5e-5",
        "SHARED_STORAGE_ROOT=${SPECFORGE_SHARED_STORAGE_ROOT:-/workspace/storage-shared/nlp/tungdd11/tungdecoder}",
        "SHAREGPT_JSONL=${SHAREGPT_JSONL:-\"$ARTIFACT_ROOT/manifests/sharegpt_train.jsonl\"}",
        "--phase",
        "--resume",
        "--print-config",
    ):
        assert expected in text


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
        "--phase",
        "data",
        env={
            "ARTIFACT_ROOT": artifact_root.as_posix(),
            "SHAREGPT_SOURCE": sharegpt.as_posix(),
            "LLAVA_SOURCE_JSONL": llava.as_posix(),
            "IMAGE_ROOT": image_root.as_posix(),
            "PYTHON_BIN": "/bin/echo",
            "SPECFORGE_NUM_SAMPLES": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_data.py --dataset sharegpt" in result.stdout
    assert "scripts/prepare_llava_caption_manifest.py" in result.stdout
    assert f"--image-root {image_root.as_posix()}" in result.stdout
    assert "--expected-records 1" in result.stdout


def test_msd_data_phase_validates_all_inputs_before_preparation(tmp_path: Path) -> None:
    sharegpt = tmp_path / "sharegpt.json"
    image_root = tmp_path / "images"
    sharegpt.write_text("[]", encoding="utf-8")
    image_root.mkdir()

    result = run_launcher(
        "--phase",
        "data",
        env={
            "ARTIFACT_ROOT": (tmp_path / "artifacts").as_posix(),
            "SHAREGPT_SOURCE": sharegpt.as_posix(),
            "LLAVA_SOURCE_JSONL": (tmp_path / "missing.jsonl").as_posix(),
            "IMAGE_ROOT": image_root.as_posix(),
            "PYTHON_BIN": "/bin/echo",
            "SPECFORGE_NUM_SAMPLES": "1",
        },
    )

    assert result.returncode == 2
    assert "missing LLAVA_SOURCE_JSONL" in result.stderr
    assert "prepare_data.py" not in result.stdout


def capture_env(tmp_path: Path) -> dict[str, str]:
    sharegpt = tmp_path / "sharegpt_train.jsonl"
    llava = tmp_path / "llava68k.jsonl"
    images = tmp_path / "images"
    target = tmp_path / "target"
    sharegpt.write_text("{}\n", encoding="utf-8")
    llava.write_text("{}\n", encoding="utf-8")
    images.mkdir()
    target.mkdir()
    return {
        "SHAREGPT_JSONL": sharegpt.as_posix(),
        "LLAVA_MANIFEST": llava.as_posix(),
        "IMAGE_ROOT": images.as_posix(),
        "TARGET_MODEL_PATH": target.as_posix(),
        "TEXT_FEATURE_ROOT": (tmp_path / "text-features").as_posix(),
        "VISUAL_FEATURE_ROOT": (tmp_path / "visual-features").as_posix(),
        "PYTHON_BIN": "/bin/echo",
        "TORCHRUN_BIN": "/bin/echo",
        "SPECFORGE_GPUS": "1",
        "SPECFORGE_GLOBAL_BATCH_SIZE": "1",
    }


def test_msd_capture_runs_text_then_visual_commands(tmp_path: Path) -> None:
    result = run_launcher("--phase", "capture", env=capture_env(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_hidden_states.py --strategy msd" in result.stdout
    assert "scripts/prepare_llava_caption_hidden_states.py --strategy msd" in result.stdout
    assert result.stdout.index("prepare_hidden_states.py") < result.stdout.index(
        "prepare_llava_caption_hidden_states.py"
    )


def test_msd_capture_rejects_existing_features_without_resume(tmp_path: Path) -> None:
    env = capture_env(tmp_path)
    existing = tmp_path / "text-features" / "rows_0-2000" / "data_0.ckpt"
    existing.parent.mkdir(parents=True)
    existing.touch()

    result = run_launcher("--phase", "capture", env=env)

    assert result.returncode == 1
    assert "pass --resume" in result.stderr
    assert "prepare_hidden_states.py" not in result.stdout


def test_msd_capture_resume_preserves_existing_features(tmp_path: Path) -> None:
    env = capture_env(tmp_path)
    existing = tmp_path / "text-features" / "rows_0-2000" / "data_0.ckpt"
    existing.parent.mkdir(parents=True)
    existing.touch()

    result = run_launcher("--phase", "capture", "--resume", env=env)

    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_hidden_states.py --strategy msd" in result.stdout
    assert "scripts/prepare_llava_caption_hidden_states.py --strategy msd" in result.stdout


def test_msd_materializer_creates_isolated_depth_configs(tmp_path: Path) -> None:
    module = _module()
    outputs = []
    for depth in (1, 3, 5):
        draft_path, recipe_path = module.materialize_run(
            base_draft=BASE_DRAFT,
            base_recipe=BASE_RECIPE,
            generated_root=tmp_path / "generated",
            output_root=tmp_path / "outputs",
            depth=depth,
            target_model_path="/models/qwen25-vl-3b",
            text_feature_root="/data/msd/features/sharegpt68k",
            visual_feature_root="/data/msd/features/llava68k",
            learning_rate=5e-5,
            micro_batch_size=1,
            accumulation_steps=1,
            gpu_count=4,
        )
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        parsed = Config.from_file(str(recipe_path))
        assert draft["num_hidden_layers"] == depth
        assert recipe["training"]["num_epochs"] == 40
        assert recipe["training"]["msd_total_epochs"] == 40
        assert recipe["model"]["draft_num_hidden_layers"] == depth
        assert parsed.training.strategy == "msd"
        assert recipe["data"]["hidden_states_path"].endswith("sharegpt68k")
        assert recipe["data"]["msd_visual_hidden_states_path"].endswith("llava68k")
        outputs.append(recipe["output_dir"])

    assert len(set(outputs)) == 3
    assert Path(outputs[0]).parts[-2:] == ("depth1", "output")
    assert Path(outputs[1]).parts[-2:] == ("depth3", "output")
    assert Path(outputs[2]).parts[-2:] == ("depth5", "output")


@pytest.mark.parametrize("depth", [0, 2, 4, 6])
def test_msd_materializer_rejects_unsupported_depth(depth: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1, 3, or 5"):
        _module().materialize_run(
            base_draft=BASE_DRAFT,
            base_recipe=BASE_RECIPE,
            generated_root=tmp_path,
            output_root=tmp_path,
            depth=depth,
            target_model_path="target",
            text_feature_root="text",
            visual_feature_root="visual",
            learning_rate=5e-5,
            micro_batch_size=1,
            accumulation_steps=1,
            gpu_count=4,
        )
