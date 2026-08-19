from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "train_EAGLE3_SpecForge"


def test_phase1_recipe_is_text_only_qwen25vl_eagle3():
    recipe = (
        PACKAGE / "examples" / "configs" / "qwen2.5-vl-3b-eagle3-text-offline.yaml"
    )
    text = recipe.read_text(encoding="utf-8")

    assert "strategy: eagle3" in text
    assert "input_modality: text" in text
    assert "qwen2.5-vl-3b-eagle3.json" in text
    assert "hidden_states_path:" in text
    assert "vocab_mapping_path:" in text
    assert "image" not in text.lower()


def test_phase1_launcher_exposes_data_capture_and_train_phases():
    launcher = PACKAGE / "train_qwen25vl_eagle3_text.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "--phase" in text
    assert "data|capture|train|all" in text
    assert "scripts/prepare_data.py" in text
    assert "scripts/prepare_hidden_states.py" in text
    assert "--strategy" in text and "eagle3" in text
    assert "-m specforge.cli train" in text
    assert "train_Dflash_SpecForge" not in text
    assert "image" not in text.lower()


def test_phase1_production_recipe_is_two_b200_and_launcher_does_not_shadow_it():
    recipe = (
        PACKAGE / "examples" / "configs" / "qwen2.5-vl-3b-eagle3-text-offline.yaml"
    )
    recipe_text = recipe.read_text(encoding="utf-8")
    launcher = PACKAGE / "train_qwen25vl_eagle3_text.sh"
    launcher_text = launcher.read_text(encoding="utf-8")

    assert "torch_dtype: bfloat16" in recipe_text
    assert "max_length: 2048" in recipe_text
    assert "num_epochs: 10" in recipe_text
    assert "batch_size: 1" in recipe_text
    assert "accumulation_steps: 2" in recipe_text
    assert "nproc_per_node: 2" in recipe_text

    assert "SPECFORGE_CONFIG" in launcher_text
    assert "--config" in launcher_text
    assert "config_value training.num_epochs" in launcher_text
    assert "config_value model.sglang_attention_backend" in launcher_text
    for shadowed_setting in (
        '"data.max_length=$MAX_LENGTH"',
        '"training.num_epochs=$NUM_EPOCHS"',
        '"training.batch_size=$MICRO_BATCH_SIZE"',
        '"training.accumulation_steps=$ACCUMULATION_STEPS"',
        '"training.fsdp_sharding=$FSDP_SHARDING"',
        '"training.save_interval=$SAVE_INTERVAL"',
        '"training.log_interval=$LOG_INTERVAL"',
    ):
        assert shadowed_setting not in launcher_text
