from __future__ import annotations

import torch

from src.analyze.Validate_Sparrow_hypothesises import dflash_runtime


def test_apply_hidden_context_mask_is_shape_preserving_and_non_mutating():
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    original = hidden.clone()

    masked = dflash_runtime.apply_hidden_context_mask(hidden, torch.tensor([False, True, False, True]))

    assert masked.shape == hidden.shape
    assert masked.dtype == hidden.dtype
    assert torch.equal(hidden, original)
    assert torch.equal(masked[:, [0, 2]], hidden[:, [0, 2]])
    assert torch.count_nonzero(masked[:, [1, 3]]) == 0


def test_visual_retention_mask_keeps_requested_prefix_of_visual_positions():
    mask = dflash_runtime.build_visual_retention_mask(
        total_length=8,
        visual_positions=[1, 2, 3, 4],
        retention_percentage=50,
    )

    assert mask.tolist() == [False, False, False, True, True, False, False, False]


def test_extract_target_hidden_concatenates_configured_layers():
    hidden_states = tuple(torch.full((1, 3, 2), float(index)) for index in range(6))

    selected = dflash_runtime.extract_target_hidden(hidden_states, [0, 3])

    assert selected.shape == (1, 3, 4)
    assert torch.equal(selected[..., :2], torch.ones(1, 3, 2))
    assert torch.equal(selected[..., 2:], torch.full((1, 3, 2), 4.0))


def test_load_dflash_draft_uses_training_state_resolver(monkeypatch):
    state = {"draft_state_dict": {"weight": torch.ones(1)}}
    calls = []

    class Draft:
        pass

    def resolve(path):
        calls.append(("resolve", path))
        return state

    def materialize(received_state, config_path):
        calls.append(("materialize", received_state, config_path))
        return Draft()

    monkeypatch.setattr(dflash_runtime, "resolve_training_state", resolve)
    monkeypatch.setattr(dflash_runtime, "materialize_draft", materialize)

    draft, resolved = dflash_runtime.load_dflash_draft("checkpoint/training_state.pt", "draft.json")

    assert isinstance(draft, Draft)
    assert resolved is state
    assert calls == [
        ("resolve", "checkpoint/training_state.pt"),
        ("materialize", state, "draft.json"),
    ]


def test_find_visual_positions_uses_qwen25_video_token_id():
    class Target:
        class Config:
            video_token_id = 99

        config = Config()

    positions = dflash_runtime.find_visual_positions(
        torch.tensor([[10, 99, 99, 11, 12]]),
        target=Target(),
    )

    assert positions == [1, 2]
