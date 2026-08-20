import os
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def test_eagle3_runtime_imports_with_the_installed_torch():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from specforge.algorithms.eagle3.model import Eagle3DraftModel",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_qwen25vl_tied_embedding_can_supply_target_lm_head(tmp_path):
    config = {
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 6,
        "tie_word_embeddings": True,
    }
    weights = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({"model.embed_tokens.weight": weights}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
            }
        ),
        encoding="utf-8",
    )

    from specforge.modeling.target.target_head import TargetHead

    head = TargetHead(str(tmp_path))
    head.load_weights(str(tmp_path), lm_head_key="lm_head.weight")

    assert torch.equal(head.fc.weight, weights)

    typed_head = TargetHead.from_pretrained(
        str(tmp_path),
        lm_head_key="lm_head.weight",
        dtype=torch.float16,
    )
    assert typed_head.fc.weight.dtype == torch.float16


def test_qwen25vl_nested_text_config_builds_target_head():
    from specforge.modeling.target.target_head import TargetHead

    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=4, vocab_size=6),
        tie_word_embeddings=True,
    )
    with patch(
        "specforge.modeling.target.target_head.AutoConfig.from_pretrained",
        return_value=config,
    ):
        head = TargetHead("unused-model-path")

    assert head.hidden_size == 4
    assert head.vocab_size == 6
    assert head.fc.weight.shape == (6, 4)


def test_qwen25vl_text_defaults_to_three_axis_mrope_positions():
    from specforge.algorithms.eagle3.model import OnlineEagle3Model

    runtime = SimpleNamespace(
        attention_backend="sdpa",
        draft_model=SimpleNamespace(
            config=SimpleNamespace(rope_scaling={"type": "mrope"})
        ),
    )

    positions = OnlineEagle3Model._prepare_position_ids(
        runtime,
        None,
        seq_length=5,
        past_key_values_length=0,
        device=torch.device("cpu"),
    )

    assert positions.shape == (3, 1, 5)
    assert torch.equal(positions[0], positions[1])
    assert torch.equal(positions[1], positions[2])
