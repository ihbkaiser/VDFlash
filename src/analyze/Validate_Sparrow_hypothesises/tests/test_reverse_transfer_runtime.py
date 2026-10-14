from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.analyze.Validate_Sparrow_hypothesises.reverse_transfer_runtime import (
    _draft_forward,
    _freeze_draft_embedding_branch,
    _mask_features,
)


class FakeDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([nn.Identity()])
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ea_layer = FakeDraft()
        self.base_model = SimpleNamespace(lm_head=nn.Linear(2, 3, bias=False))
        with torch.no_grad():
            self.base_model.lm_head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]))


def prepared() -> SimpleNamespace:
    inner = SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        inputs_embeds=torch.ones(1, 4, 2),
        position_ids=torch.arange(4).view(1, 4),
        video_positions=torch.tensor([1, 2]),
    )
    sample = SimpleNamespace(labels=("A", "B"), correct_label="A")
    return SimpleNamespace(prepared=inner, sample=sample)


def test_feature_masks_zero_the_requested_modality_only():
    value = prepared()
    target_hidden = torch.ones(1, 4, 2)
    visual_hidden, visual_inputs, visual_meta = _mask_features(value, target_hidden, "visual")
    assert torch.all(visual_hidden[:, [0, 3]] == 0)
    assert torch.all(visual_inputs[:, [0, 3]] == 0)
    assert torch.all(visual_hidden[:, [1, 2]] == 1)
    assert visual_meta["draft_visual_embedding_source"] == "target_visual_tower_output"
    text_hidden, text_inputs, text_meta = _mask_features(value, target_hidden, "text")
    assert torch.all(text_hidden[:, [1, 2]] == 0)
    assert torch.all(text_inputs[:, [1, 2]] == 0)
    assert torch.all(text_hidden[:, [0, 3]] == 1)
    assert "target_logits_used" not in text_meta


def test_embedding_branch_freeze_restores_flags():
    model = FakeModel()
    original = [parameter.requires_grad for parameter in model.ea_layer.embed_tokens.parameters()]
    with _freeze_draft_embedding_branch(model):
        assert not any(parameter.requires_grad for parameter in model.ea_layer.embed_tokens.parameters())
    assert [parameter.requires_grad for parameter in model.ea_layer.embed_tokens.parameters()] == original


def test_logits_are_projected_from_msd_draft_hidden_not_target_logits():
    model = FakeModel()
    value = prepared()
    target_hidden = torch.full((1, 4, 2), 99.0)
    prediction = _draft_forward(model, value, target_hidden, "full")
    assert prediction.feature_mask_metadata["target_logits_used"] is False
    assert prediction.feature_mask_metadata["future_answer_token_provided"] is False
    assert prediction.feature_mask_metadata["answer_position"] == 3
    assert prediction.feature_mask_metadata["draft_position_source"] == "model.ea_layer native sequential positions"
    assert model.ea_layer.last_kwargs["position_ids"] is None
    assert torch.equal(prediction.draft_hidden, torch.tensor([[[7.0, 8.0]]]))
    assert torch.equal(prediction.logits, torch.tensor([[7.0, 8.0, 1.0]]))
