"""Reference tensor operations and objective for MSD training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class MSDLossOutput:
    """Scalar MSD objective terms and masked token accuracy."""

    loss: Tensor
    feature_loss: Tensor
    soft_target_loss: Tensor
    accuracy: Tensor


class MSDTrainingModel(nn.Module):
    """Composite training shell required by the shared FSDP trainer."""

    def __init__(self, draft_model: nn.Module) -> None:
        super().__init__()
        self.draft_model = draft_model

    def forward(self, *args, **kwargs):
        return self.draft_model(*args, **kwargs)


def decouple_msd_inputs(
    conditioning_hidden_state: Tensor,
    next_token_embeddings: Tensor,
    visual_embeddings: Tensor,
    visual_token_mask: Tensor,
    fusion_projection: nn.Module,
) -> Tensor:
    """Apply text fusion while bypassing it at visual positions."""

    if conditioning_hidden_state.ndim != 3:
        raise ValueError("conditioning_hidden_state must have shape [B, S, H]")
    if next_token_embeddings.shape != conditioning_hidden_state.shape:
        raise ValueError("next_token_embeddings must match conditioning hidden state")
    if visual_embeddings.shape != conditioning_hidden_state.shape:
        raise ValueError("visual_embeddings must match conditioning hidden state")
    if visual_token_mask.shape != conditioning_hidden_state.shape[:2]:
        raise ValueError("visual_token_mask must have shape [B, S]")

    text_inputs = fusion_projection(
        torch.cat((conditioning_hidden_state, next_token_embeddings), dim=-1)
    )
    if text_inputs.shape != conditioning_hidden_state.shape:
        raise ValueError("fusion_projection must map 2 * hidden_size to hidden_size")
    return torch.where(
        visual_token_mask.to(torch.bool).unsqueeze(-1),
        visual_embeddings,
        text_inputs,
    )


def add_reference_uniform_noise(
    hidden_states: Tensor,
    width: float = 0.2,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add original MSD's sequence-scaled uniform perturbation."""

    if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
        raise ValueError("hidden_states must have shape [B, S, H] with S > 0")
    if width < 0:
        raise ValueError("width must be non-negative")
    noise = torch.rand(
        hidden_states.shape,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        generator=generator,
    )
    scale = float(width) * 512.0 / hidden_states.shape[1]
    return hidden_states + (noise - 0.5) * scale


def _head_logits(hidden_states: Tensor, head: nn.Module) -> Tensor:
    if isinstance(head, nn.Linear):
        bias = head.bias.detach().float() if head.bias is not None else None
        return F.linear(hidden_states.float(), head.weight.detach().float(), bias)
    return head(hidden_states.float())


def msd_loss(
    predicted_hidden_state: Tensor,
    target_hidden_state: Tensor,
    loss_mask: Tensor,
    frozen_lm_head: nn.Module,
    feature_weight: float = 1.0,
    soft_target_weight: float = 0.1,
) -> MSDLossOutput:
    """Compute masked SmoothL1 plus frozen-head soft-target cross entropy."""

    if predicted_hidden_state.ndim != 3:
        raise ValueError("predicted_hidden_state must have shape [B, S, H]")
    if target_hidden_state.shape != predicted_hidden_state.shape:
        raise ValueError("target_hidden_state must match predicted_hidden_state")
    if loss_mask.shape != predicted_hidden_state.shape[:2]:
        raise ValueError("loss_mask must have shape [B, S]")
    if feature_weight < 0 or soft_target_weight < 0:
        raise ValueError("MSD loss weights must be non-negative")

    mask = loss_mask.to(dtype=torch.float32)
    valid_tokens = mask.sum()
    if valid_tokens.item() <= 0:
        raise ValueError("loss_mask must contain at least one valid token")
    denominator = valid_tokens + 1e-5

    predicted = predicted_hidden_state.float()
    target = target_hidden_state.detach().float()
    feature_per_token = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
    feature_loss = (feature_per_token * mask).sum() / denominator

    predicted_logits = _head_logits(predicted, frozen_lm_head)
    with torch.no_grad():
        target_logits = _head_logits(target, frozen_lm_head)
        target_probabilities = F.softmax(target_logits, dim=-1)
    soft_per_token = -(
        target_probabilities * F.log_softmax(predicted_logits, dim=-1)
    ).sum(-1)
    soft_target_loss = (soft_per_token * mask).sum() / denominator
    correct = predicted_logits.argmax(-1).eq(target_logits.argmax(-1)).float()
    accuracy = (correct * mask).sum() / valid_tokens
    total = feature_weight * feature_loss + soft_target_weight * soft_target_loss
    return MSDLossOutput(total, feature_loss, soft_target_loss, accuracy)


__all__ = [
    "MSDLossOutput",
    "MSDTrainingModel",
    "add_reference_uniform_noise",
    "decouple_msd_inputs",
    "msd_loss",
]
