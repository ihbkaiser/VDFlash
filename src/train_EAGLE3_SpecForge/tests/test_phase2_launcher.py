from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "train_EAGLE3_SpecForge"
CONFIG = PACKAGE / "examples" / "configs" / "qwen2.5-vl-3b-eagle3-caption-offline.yaml"
CONFIG_7B = PACKAGE / "examples" / "configs" / "qwen2.5-vl-7b-eagle3-caption-offline.yaml"
LAUNCHER = PACKAGE / "train_qwen25vl_eagle3_captioning.sh"
LAUNCHER_3B = PACKAGE / "train_qwen25vl_3b_eagle3_captioning.sh"
LAUNCHER_7B = PACKAGE / "train_qwen25vl_7b_eagle3_captioning.sh"


def test_phase2_config_declares_qwen25vl_offline_contract():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["model"]["input_modality"] == "qwen2_5_vl"
    assert config["data"]["hidden_states_path"]
    assert config["data"]["max_length"] == 3072
    assert config["training"]["strategy"] == "eagle3"
    assert config["model"]["draft_checkpoint_path"] is None


def test_launcher_help_and_phase1_checkpoint_contract():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PHASE1_CHECKPOINT" in result.stdout
    assert "data|capture|train|all" in result.stdout
    assert "train_Dflash_SpecForge" not in LAUNCHER.read_text(encoding="utf-8")


def test_launcher_reuses_materialized_archive_root_for_capture_only_runs():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'if [[ -n "$IMAGE_ARCHIVE" && -z "$IMAGE_ROOT" ]]; then' in text


def test_phase2_profiles_are_size_specific_and_use_eagle3_objective():
    config_3b = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_7b = yaml.safe_load(CONFIG_7B.read_text(encoding="utf-8"))

    assert config_3b["model"]["target_model_path"].endswith("3B-Instruct")
    assert config_7b["model"]["target_model_path"].endswith("7B-Instruct")
    assert config_3b["deployment"]["trainer"]["nproc_per_node"] == 2
    assert config_7b["deployment"]["trainer"]["nproc_per_node"] == 2
    assert config_3b["training"]["batch_size"] == 2
    assert config_7b["training"]["batch_size"] == 2
    assert config_3b["training"]["accumulation_steps"] == 16
    assert config_7b["training"]["accumulation_steps"] == 16
    assert config_3b["training"]["learning_rate"] == config_7b["training"]["learning_rate"] == 5e-5
    for config in (config_3b, config_7b):
        assert "objective_chunk_blocks" not in config["training"]
        assert "num_anchors" not in config["training"]
        assert "loss_decay_gamma" not in config["training"]


def test_size_specific_launchers_delegate_to_standalone_eagle3_launcher():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "train_Dflash_SpecForge" not in launcher
    for path, size in ((LAUNCHER_3B, "3b"), (LAUNCHER_7B, "7b")):
        text = path.read_text(encoding="utf-8")
        assert f'SPECFORGE_MODEL_SIZE=${{SPECFORGE_MODEL_SIZE:-{size}}}' in text
        assert "train_qwen25vl_eagle3_captioning.sh" in text
