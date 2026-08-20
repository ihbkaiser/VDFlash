import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def _fake_prepared(length: int):
    return {
        "input_ids": torch.arange(length, dtype=torch.long).view(1, -1),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "loss_mask": torch.ones(1, length),
        "position_ids": torch.arange(3 * length, dtype=torch.long).view(3, 1, length),
        "multimodal_inputs": {
            "pixel_values": torch.ones(length, 4),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        },
    }


def test_collate_prepared_right_pads_positions_and_media():
    from scripts.prepare_qwen25vl_caption_hidden_states import _collate_prepared

    first = _fake_prepared(length=4)
    second = _fake_prepared(length=6)
    batch, media, lengths = _collate_prepared([first, second], pad_token_id=0)

    assert tuple(batch["input_ids"].shape) == (2, 6)
    assert tuple(batch["position_ids"].shape) == (3, 2, 6)
    assert lengths == [4, 6]
    assert len(media) == 2


def test_saved_record_uses_eagle3_storage_names(tmp_path):
    from scripts.prepare_qwen25vl_caption_hidden_states import _save_record

    path = tmp_path / "row.ckpt"
    _save_record(
        path,
        {
            "input_ids": torch.ones(4, dtype=torch.int32),
            "loss_mask": torch.ones(4),
            "aux_hidden_state": torch.zeros(4, 12),
            "hidden_state": torch.zeros(4, 4),
            "position_ids": torch.zeros(3, 4, dtype=torch.int32),
        },
        compress=False,
    )

    payload = torch.load(path, weights_only=True)
    assert set(payload) == {
        "input_ids",
        "loss_mask",
        "aux_hidden_state",
        "hidden_state",
        "position_ids",
    }


def test_existing_feature_validator_requires_complete_qwen_record(tmp_path):
    from scripts.prepare_qwen25vl_caption_hidden_states import (
        _feature_record_is_complete,
        _save_record,
    )

    path = tmp_path / "row.ckpt"
    _save_record(
        path,
        {
            "input_ids": torch.ones(4, dtype=torch.int32),
            "loss_mask": torch.ones(4),
            "aux_hidden_state": torch.zeros(4, 12),
            "hidden_state": torch.zeros(4, 4),
        },
        compress=False,
    )

    assert not _feature_record_is_complete(path)


def test_existing_feature_validator_accepts_complete_qwen_record(tmp_path):
    from scripts.prepare_qwen25vl_caption_hidden_states import (
        _feature_record_is_complete,
        _save_record,
    )

    path = tmp_path / "row.ckpt"
    _save_record(
        path,
        {
            "input_ids": torch.ones(4, dtype=torch.int32),
            "loss_mask": torch.ones(4),
            "aux_hidden_state": torch.zeros(4, 12),
            "hidden_state": torch.zeros(4, 4),
            "position_ids": torch.zeros(3, 4, dtype=torch.int32),
        },
        compress=False,
    )

    assert _feature_record_is_complete(path)


def test_existing_feature_validator_rejects_nonfinite_states(tmp_path):
    from scripts.prepare_qwen25vl_caption_hidden_states import (
        _feature_record_is_complete,
        _save_record,
    )

    path = tmp_path / "row.ckpt"
    _save_record(
        path,
        {
            "input_ids": torch.ones(4, dtype=torch.int32),
            "loss_mask": torch.ones(4),
            "aux_hidden_state": torch.full((4, 12), float("nan")),
            "hidden_state": torch.zeros(4, 4),
            "position_ids": torch.zeros(3, 4, dtype=torch.int32),
        },
        compress=False,
    )

    assert not _feature_record_is_complete(path)


def test_capture_shape_contract_requires_three_auxiliary_states():
    from scripts.prepare_qwen25vl_caption_hidden_states import (
        _validate_capture_shapes,
    )

    _validate_capture_shapes(
        aux_hidden_states=torch.zeros(4, 36),
        last_hidden_states=torch.zeros(4, 12),
        target_hidden_size=12,
        capture_layers=[1, 2, 3],
    )

    import pytest

    with pytest.raises(ValueError, match="three auxiliary"):
        _validate_capture_shapes(
            aux_hidden_states=torch.zeros(4, 24),
            last_hidden_states=torch.zeros(4, 12),
            target_hidden_size=12,
            capture_layers=[1, 2, 3],
        )
