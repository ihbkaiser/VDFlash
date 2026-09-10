import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from specforge.hidden_state import (
    HIDDEN_STATE_METADATA_FILENAME,
    build_hidden_state_metadata,
    load_hidden_state_metadata,
    parse_target_layer_ids,
    resolve_dflash_config,
    validate_hidden_state_metadata,
    write_hidden_state_metadata,
)


def test_parse_target_layer_ids_accepts_an_ordered_five_layer_list():
    assert parse_target_layer_ids("2, 8, 16, 24, 32") == (2, 8, 16, 24, 32)


@pytest.mark.parametrize(
    "value, message",
    [
        ("1,2,3,4", "exactly 5"),
        ("1,2,3,4,4", "unique"),
        ("1,2,-3,4,5", "non-negative"),
        ("1,2,3,4,35", "valid target decoder"),
    ],
)
def test_parse_target_layer_ids_rejects_invalid_layer_lists(value, message):
    kwargs = {"num_target_layers": 36} if "35" in value else {}
    with pytest.raises(ValueError, match=message):
        parse_target_layer_ids(value, **kwargs)


def test_resolve_dflash_config_preserves_default_and_materializes_override(tmp_path):
    source = tmp_path / "draft.json"
    resolved = tmp_path / "phase2-draft.json"
    source.write_text(
        json.dumps(
            {
                "architectures": ["DFlashDraftModel"],
                "hidden_size": 2048,
                "num_target_layers": 36,
                "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
            }
        ),
        encoding="utf-8",
    )

    payload = resolve_dflash_config(
        source,
        resolved,
        target_layer_ids="2,8,16,24,32",
        phase="phase2",
    )

    assert payload["dflash_config"]["target_layer_ids"] == [2, 8, 16, 24, 32]
    assert json.loads(resolved.read_text(encoding="utf-8")) == payload
    assert resolve_dflash_config(source, tmp_path / "default.json")[
        "dflash_config"
    ]["target_layer_ids"] == [1, 9, 17, 25, 33]


def test_resolve_dflash_config_resizes_draft_but_keeps_five_target_inputs(tmp_path):
    source = tmp_path / "draft.json"
    resolved = tmp_path / "depth3-draft.json"
    source.write_text(
        json.dumps(
            {
                "architectures": ["DFlashDraftModel"],
                "layer_types": ["full_attention"] * 5,
                "num_hidden_layers": 5,
                "num_target_layers": 36,
                "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
            }
        ),
        encoding="utf-8",
    )

    payload = resolve_dflash_config(
        source,
        resolved,
        draft_num_hidden_layers=3,
        phase="phase1",
    )

    assert payload["num_hidden_layers"] == 3
    assert payload["layer_types"] == ["full_attention"] * 3
    assert payload["dflash_config"]["target_layer_ids"] == [1, 9, 17, 25, 33]
    assert json.loads(source.read_text(encoding="utf-8"))["num_hidden_layers"] == 5


def test_hidden_state_metadata_round_trips_and_detects_layer_mismatch(tmp_path):
    metadata = build_hidden_state_metadata(
        target_layer_ids=[1, 9, 17, 25, 33],
        hidden_size=2048,
        phase="phase1",
        target_model="qwen25-vl-3b",
        dtype="bfloat16",
    )
    write_hidden_state_metadata(tmp_path, metadata)

    assert (tmp_path / HIDDEN_STATE_METADATA_FILENAME).is_file()
    assert load_hidden_state_metadata(tmp_path) == metadata
    validate_hidden_state_metadata(
        tmp_path,
        target_layer_ids=[1, 9, 17, 25, 33],
        hidden_size=2048,
        expected_phase="phase1",
    )
    with pytest.raises(ValueError, match="target_layer_ids"):
        validate_hidden_state_metadata(
            tmp_path,
            target_layer_ids=[2, 8, 16, 24, 32],
            hidden_size=2048,
        )


def test_dflash_resume_contract_contains_self_describing_hidden_state_metadata():
    from specforge.algorithms.dflash.providers import resume_contract

    draft_model = SimpleNamespace(
        target_layer_ids=[1, 9, 17, 25, 33],
        config=SimpleNamespace(hidden_size=2048, dtype="bfloat16"),
    )
    training_model = SimpleNamespace(
        block_size=16,
        mask_token_id=151669,
        attention_backend="eager",
        num_anchors=512,
        loss_decay_gamma=7.0,
        loss_type="cross_entropy",
        dpace_alpha=0.0,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(hidden_state_phase="phase2"),
        model=SimpleNamespace(target_model_path="qwen25-vl-3b"),
    )

    metadata = resume_contract(config, draft_model, training_model)[
        "dflash_hidden_state_metadata"
    ]

    assert metadata["target_layer_ids"] == (1, 9, 17, 25, 33)
    assert metadata["feature_dim"] == 5 * 2048
    assert metadata["phase"] == "phase2"


def test_phase_launchers_expose_independent_layer_overrides():
    root = Path(__file__).parents[1]
    phase1 = (root / "train_qwen25vl_dflash_sharegpt_68k.sh").read_text()
    phase2 = (root / "train_qwen25vl_dflash_llava_68k.sh").read_text()

    assert "SPECFORGE_PHASE1_TARGET_LAYER_IDS" in phase1
    assert "SPECFORGE_PHASE2_TARGET_LAYER_IDS" in phase2
    assert "resolve_dflash_config.py" in phase1
    assert "resolve_dflash_config.py" in phase2


def test_dflash_warm_start_resets_projection_when_layer_semantics_change(tmp_path):
    import torch

    from specforge.training.model_loading import warm_start_draft_model

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
            self.block = torch.nn.Linear(2, 2)

    source = Draft()
    destination = Draft()
    with torch.no_grad():
        source.fc.weight.fill_(7.0)
        source.fc.bias.fill_(7.0)
        source.block.weight.fill_(9.0)
        source.block.bias.fill_(9.0)

    checkpoint = tmp_path / "training_state.pt"
    torch.save(
        {
            "strategy": "dflash",
            "draft_state_dict": source.state_dict(),
            "dflash_hidden_state_metadata": {
                "target_layer_ids": (1, 9, 17, 25, 33),
                "hidden_size": 2,
                "feature_dim": 10,
                "phase": "phase1",
            },
        },
        checkpoint,
    )

    report = warm_start_draft_model(
        destination,
        str(checkpoint),
        draft_config=SimpleNamespace(
            dflash_config={"target_layer_ids": [2, 8, 16, 24, 32]}
        ),
        strategy="dflash",
        warm_start_mode="auto",
    )

    assert torch.equal(destination.block.weight, source.block.weight)
    assert not torch.equal(destination.fc.weight, source.fc.weight)
    assert set(report.reset_keys) == {"fc.weight", "fc.bias"}
