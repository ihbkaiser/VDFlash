"""Tensor normalization for offline MSD teacher captures."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def _batched_sequence(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(f"{name} must have shape [S] or [1, S]")


def _batched_hidden(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(f"{name} must have shape [S, H] or [1, S, H]")


def _position_ids(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2 and tensor.shape[0] == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 3 and tensor.shape[:2] == (3, 1):
        return tensor
    raise ValueError("position_ids must have shape [3, S] or [3, 1, S]")


def _shift_left_with_zeros(tensor: Tensor) -> Tensor:
    return torch.cat((tensor[:, 1:], torch.zeros_like(tensor[:, :1])), dim=1)


def normalize_offline_sample(
    raw: Mapping[str, Tensor],
    max_len: int,
) -> dict[str, Tensor]:
    """Validate and convert one raw teacher record to MSD training tensors."""

    if isinstance(max_len, bool) or not isinstance(max_len, int) or max_len < 1:
        raise ValueError("max_len must be a positive integer")
    required = {
        "input_ids",
        "loss_mask",
        "target_hidden_state",
        "input_embeddings",
        "visual_token_mask",
        "position_ids",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise KeyError(f"missing MSD capture fields: {', '.join(missing)}")

    input_ids = _batched_sequence(raw["input_ids"], "input_ids")
    loss_mask = _batched_sequence(raw["loss_mask"], "loss_mask")
    target_hidden = _batched_hidden(
        raw["target_hidden_state"], "target_hidden_state"
    )
    input_embeddings = _batched_hidden(raw["input_embeddings"], "input_embeddings")
    visual_mask = _batched_sequence(raw["visual_token_mask"], "visual_token_mask")
    position_ids = _position_ids(raw["position_ids"])

    sequence_lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        target_hidden.shape[1],
        input_embeddings.shape[1],
        visual_mask.shape[1],
        position_ids.shape[2],
    }
    if len(sequence_lengths) != 1:
        raise ValueError("MSD capture sequence lengths must match")
    if target_hidden.shape[-1] != input_embeddings.shape[-1]:
        raise ValueError("target and input embedding hidden sizes must match")

    sequence_length = min(input_ids.shape[1], max_len)
    input_ids = input_ids[:, :sequence_length]
    loss_mask = loss_mask[:, :sequence_length].clone()
    target_hidden = target_hidden[:, :sequence_length]
    input_embeddings = input_embeddings[:, :sequence_length]
    visual_mask = visual_mask[:, :sequence_length].to(torch.bool)
    position_ids = position_ids[:, :, :sequence_length]
    loss_mask[:, -1] = 0

    return {
        "input_ids": _shift_left_with_zeros(input_ids),
        "loss_mask": loss_mask,
        "target_hidden_state": _shift_left_with_zeros(target_hidden),
        "conditioning_hidden_state": target_hidden,
        "next_token_embeddings": _shift_left_with_zeros(input_embeddings),
        "visual_embeddings": input_embeddings,
        "visual_token_mask": visual_mask,
        "position_ids": position_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }


__all__ = ["normalize_offline_sample"]
