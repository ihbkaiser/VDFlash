from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from specforge.config import Config


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "train_qwen25vl_msd_depth_sweep.sh"
MATERIALIZER = ROOT / "scripts" / "materialize_msd_sweep.py"
BASE_DRAFT = ROOT / "configs" / "qwen2.5-vl-3b-msd.json"
BASE_RECIPE = ROOT / "examples" / "configs" / "qwen2.5-vl-3b-msd-68k-offline.yaml"


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
        "--phase",
        "--resume",
        "--print-config",
    ):
        assert expected in text


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
