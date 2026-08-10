from __future__ import annotations

import torch

from src.analyze.Validate_Sparrow_hypothesises.probes import (
    layerwise_cosine,
    make_modality_masks,
    masked_visual_keys,
    query_only_attention,
    summarize_modality_attention,
)


def test_query_only_attention_matches_manual_softmax_shape():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    values = query_only_attention(query, key, query_index=-1)
    assert values.shape == (1, 2)
    assert torch.allclose(values.sum(dim=-1), torch.ones(1))


def test_attention_summary_keeps_modalities_disjoint():
    masks = make_modality_masks(5, [1, 2], [4])
    attention = torch.tensor([[0.1, 0.2, 0.3, 0.1, 0.3]])
    summary = summarize_modality_attention(attention, masks["instruction"], masks["visual"], masks["text"])
    assert summary["visual_mass"] == 0.5
    assert abs(summary["instruction_mass"] - 0.3) < 1e-6


def test_visual_kv_mask_is_additive_and_does_not_change_shape():
    mask = torch.zeros(1, 1, 3, 5)
    result = masked_visual_keys(mask, torch.tensor([False, True, False, True, False]))
    assert result.shape == mask.shape
    assert torch.isneginf(result[0, 0, 0, 1]) or result[0, 0, 0, 1] < -1e30
    assert result[0, 0, 0, 0] == 0


def test_layerwise_cosine_reports_visual_and_text_separately():
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    hidden = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    result = layerwise_cosine(hidden, embeddings, torch.tensor([True, False]), torch.tensor([False, True]))
    assert result["visual_cosine"] == 1.0
    assert result["text_cosine"] == 0.0
