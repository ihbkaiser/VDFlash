"""Memory-bounded probes for attention dilution and layer internalization."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .metrics import normalized_entropy


def query_only_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: int = -1,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention for one query row without materializing ``L x L``.

    Inputs are ``[batch, heads, query_length, head_dim]`` and
    ``[batch, heads, key_length, head_dim]``. The result is ``[heads, key]``
    for batch size one.
    """

    if query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("query/key must have shape [1, heads, length, head_dim]")
    if query.shape[1] != key.shape[1] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    index = query_index if query_index >= 0 else query.shape[2] + query_index
    if not 0 <= index < query.shape[2]:
        raise IndexError("query_index is outside query sequence")
    scores = torch.matmul(query[:, :, index : index + 1, :], key.transpose(-1, -2)).squeeze(2)
    scores = scores / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        mask = attention_mask
        if mask.ndim == 4:
            mask = mask[:, :, index, :]
        elif mask.ndim == 3:
            mask = mask[:, index, :]
        elif mask.ndim != 2:
            raise ValueError("attention_mask must be [batch,key], [batch,query,key] or [batch,1,query,key]")
        scores = scores + mask
    return torch.softmax(scores, dim=-1)[0]


def summarize_modality_attention(
    attention_by_head: torch.Tensor,
    instruction_mask: torch.Tensor,
    visual_mask: torch.Tensor,
    text_mask: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate query-only attention while preserving per-head diagnostics."""

    if attention_by_head.ndim != 2:
        raise ValueError("attention_by_head must have shape [heads,key_length]")
    masks = {"instruction": instruction_mask, "visual": visual_mask, "text": text_mask}
    if any(mask.ndim != 1 or mask.numel() != attention_by_head.shape[-1] for mask in masks.values()):
        raise ValueError("modality masks must be one-dimensional key-length masks")
    if torch.any((instruction_mask.to(torch.int8) + visual_mask.to(torch.int8) + text_mask.to(torch.int8)) > 1):
        raise ValueError("modality masks overlap")
    mean_attention = attention_by_head.mean(dim=0)
    masses = {name: float(mean_attention[mask].sum().item()) for name, mask in masks.items()}
    visual_values = mean_attention[visual_mask]
    return {
        "heads": int(attention_by_head.shape[0]),
        "key_length": int(attention_by_head.shape[1]),
        "instruction_mass": masses["instruction"],
        "visual_mass": masses["visual"],
        "text_mass": masses["text"],
        "visual_entropy": normalized_entropy(visual_values.detach().float().tolist()),
        "per_head_visual_mass": [float(row[visual_mask].sum().item()) for row in attention_by_head],
    }


def masked_visual_keys(
    attention_mask: torch.Tensor,
    visual_mask: torch.Tensor,
) -> torch.Tensor:
    """Return an additive mask that excludes visual keys from every query."""

    if attention_mask.ndim != 4 or attention_mask.shape[-1] != visual_mask.numel():
        raise ValueError("attention_mask must be [batch, heads, query, key] with matching key length")
    result = attention_mask.clone()
    result[..., visual_mask] = torch.finfo(result.dtype).min
    return result


def layerwise_cosine(
    hidden_states: torch.Tensor,
    input_embeddings: torch.Tensor,
    visual_mask: torch.Tensor,
    text_mask: torch.Tensor,
) -> dict[str, float]:
    """Mean cosine similarity to original embeddings for two modalities."""

    if hidden_states.ndim != 3 or input_embeddings.shape != hidden_states.shape:
        raise ValueError("hidden_states and input_embeddings must have shape [batch, seq, hidden]")
    if hidden_states.shape[0] != 1:
        raise ValueError("layerwise_cosine currently supports batch size one")
    values = torch.nn.functional.cosine_similarity(hidden_states[0], input_embeddings[0], dim=-1)
    if visual_mask.sum() == 0 or text_mask.sum() == 0:
        raise ValueError("visual and text masks must both be non-empty")
    return {
        "visual_cosine": float(values[visual_mask].mean().item()),
        "text_cosine": float(values[text_mask].mean().item()),
    }


def make_modality_masks(
    sequence_length: int,
    visual_positions: list[int],
    instruction_positions: list[int],
) -> Mapping[str, torch.Tensor]:
    visual = torch.zeros(sequence_length, dtype=torch.bool)
    instruction = torch.zeros(sequence_length, dtype=torch.bool)
    visual[torch.as_tensor(visual_positions, dtype=torch.long)] = True
    instruction[torch.as_tensor(instruction_positions, dtype=torch.long)] = True
    text = ~(visual | instruction)
    return {"instruction": instruction, "visual": visual, "text": text}
