from __future__ import annotations

import pytest
import torch
from torch import nn

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.offline_capture.sglang import OfflineSGLangCapture
from specforge.offline_capture.sglang_backend.capture_hooks import (
    configure_capture_layers,
)


def _sources(length: int = 5) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.arange(length),
        "loss_mask": torch.ones(length),
        "aux_hidden_states": torch.zeros(1, length, 4),
        "last_hidden_states": torch.ones(1, length, 4),
        "input_embeddings": torch.full((1, length, 4), 2.0),
        "visual_token_mask": torch.tensor([False, True, True, False, False]),
        "position_ids": torch.arange(length).repeat(3, 1),
    }


def _layout():
    return (
        builtin_algorithm_registry()
        .resolve("msd")
        .providers.offline_for("multimodal")
        .capture_layout
    )


def test_msd_capture_layout_persists_decoupling_inputs() -> None:
    record = _layout().materialize(_sources())

    assert set(record) == {
        "input_ids",
        "loss_mask",
        "target_hidden_state",
        "input_embeddings",
        "visual_token_mask",
        "position_ids",
    }
    assert record["input_embeddings"].shape == (1, 5, 4)
    assert record["visual_token_mask"].dtype == torch.bool
    assert record["position_ids"].shape == (3, 5)


def test_msd_capture_layout_rejects_missing_post_vision_embeddings() -> None:
    sources = _sources()
    del sources["input_embeddings"]

    with pytest.raises(KeyError, match="input_embeddings"):
        _layout().materialize(sources)


def test_msd_capture_layout_rejects_mask_embedding_length_mismatch() -> None:
    sources = _sources()
    sources["visual_token_mask"] = torch.zeros(4, dtype=torch.bool)

    with pytest.raises(ValueError, match="sequence lengths"):
        _layout().materialize(sources)


def test_msd_capture_layout_requires_three_axis_positions() -> None:
    sources = _sources()
    sources["position_ids"] = torch.arange(5).unsqueeze(0)

    with pytest.raises(ValueError, match="three-axis"):
        _layout().materialize(sources)


def test_msd_qwen_hook_captures_post_vision_language_embeddings() -> None:
    class LanguageModel(nn.Module):
        def forward(self, input_ids=None, input_embeds=None):
            del input_ids
            return input_embeds

    class Model:
        config = type("Config", (), {"model_type": "qwen2_5_vl"})()

        def __init__(self):
            self.language_model = LanguageModel()

    model = Model()
    assert configure_capture_layers(model, [3], capture_method="msd") == (
        "qwen2_5_vl_msd"
    )
    expected = torch.randn(5, 8)
    model.language_model(input_embeds=expected)

    actual = model._specforge_msd_input_capture.consume(5)

    torch.testing.assert_close(actual, expected)


def test_msd_qwen_hook_supports_sglang_0514_model_layout() -> None:
    class LanguageModel(nn.Module):
        def forward(self, input_ids=None, input_embeds=None):
            del input_ids
            return input_embeds

    class Model:
        config = type("Config", (), {"model_type": "qwen2_5_vl"})()
        capture_aux_hidden_states = False

        def __init__(self):
            self.model = LanguageModel()

    model = Model()
    assert configure_capture_layers(model, [3], capture_method="msd") == (
        "qwen2_5_vl_msd"
    )
    expected = torch.randn(5, 8)
    model.model(input_embeds=expected)

    actual = model._specforge_msd_input_capture.consume(5)

    torch.testing.assert_close(actual, expected)


def test_msd_capture_batch_synthesizes_text_mrope_and_carries_visual_mask() -> None:
    class Backend:
        def set_capture_layers(self, layer_ids, *, capture_method):
            self.layer_ids = layer_ids
            self.capture_method = capture_method

        def capture(self, **_kwargs):
            data = [(torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]]), torch.ones(1, 3))]
            last = (torch.ones(2, 4),)
            embeddings = (torch.full((2, 4), 2.0),)
            visual = (torch.tensor([False, True]),)
            return data, (None,), last, embeddings, visual

    capture = OfflineSGLangCapture(Backend())
    capture.set_capture_layers([3], capture_method="msd")
    result = capture.capture(
        input_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 0]]),
        loss_mask=torch.ones(1, 3),
    )

    assert result.input_embeddings.shape == (1, 2, 4)
    assert result.visual_token_mask.tolist() == [[False, True]]
    assert result.position_ids.shape == (3, 1, 3)
