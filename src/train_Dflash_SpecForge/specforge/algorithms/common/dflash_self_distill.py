"""Two-source DFlash self-distillation with reduced visual context."""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
)
from specforge.core.chunking import checkpointed_chunk_reduce


_RECIPES = {
    "dual",
    "target_ce",
    "teacher_kl",
    "gated",
    "gated_mean_scaled",
}
_SELECTORS = {"prefix", "temporal_spatial"}


def _select_temporal_spatial(
    visual_positions: torch.Tensor,
    position_ids: torch.Tensor,
    keep_count: int,
) -> torch.Tensor:
    """Select exactly K positions with temporal and within-frame coverage."""

    positions = visual_positions.to(dtype=torch.long, device="cpu")
    coordinates = position_ids[:, positions].transpose(0, 1).to(dtype=torch.long)
    temporal_ids, group_ids, group_counts = torch.unique(
        coordinates[:, 0],
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    num_groups = int(temporal_ids.numel())
    if keep_count >= int(positions.numel()):
        return positions
    if keep_count < num_groups:
        # When fewer visual slots than temporal groups are available, spread
        # the selected groups over time instead of taking a contiguous prefix.
        if keep_count == 1:
            chosen_group_ids = [num_groups // 2]
        else:
            chosen_group_ids = [
                round(index * (num_groups - 1) / (keep_count - 1))
                for index in range(keep_count)
            ]
        quotas = [1] * len(chosen_group_ids)
    else:
        chosen_group_ids = list(range(num_groups))
        counts = [int(count) for count in group_counts.tolist()]
        total_visual = int(positions.numel())
        raw_quotas = [keep_count * count / total_visual for count in counts]
        quotas = [min(count, int(value)) for count, value in zip(counts, raw_quotas)]
        remaining = keep_count - sum(quotas)
        order = sorted(
            range(num_groups),
            key=lambda index: (raw_quotas[index] - int(raw_quotas[index]), -index),
            reverse=True,
        )
        while remaining:
            progressed = False
            for index in order:
                if quotas[index] < counts[index]:
                    quotas[index] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                raise RuntimeError("could not apportion visual retention quota")

    # Sort all positions by temporal group once, then visit the small number
    # of frames/groups. Spatial coordinates are sorted with tensor ops rather
    # than Python work per visual token; this selector runs inside training.
    grouped_offsets = torch.argsort(group_ids, stable=True)
    group_offsets = [0]
    group_offsets.extend(int(count) for count in group_counts.cumsum(0).tolist())
    selected_visual_offsets = []
    for group_id, quota in zip(chosen_group_ids, quotas):
        members = grouped_offsets[group_offsets[group_id] : group_offsets[group_id + 1]]
        height = coordinates[members, 1]
        width = coordinates[members, 2]
        width_span = int(width.max().item()) - int(width.min().item()) + 1
        spatial_key = (height - height.min()) * width_span + (width - width.min())
        spatial_order = torch.argsort(spatial_key, stable=True)
        if quota >= int(members.numel()):
            picks = torch.arange(members.numel())
        else:
            picks = torch.floor(
                (torch.arange(quota, dtype=torch.float64) + 0.5)
                * members.numel()
                / quota
            ).long()
        selected_visual_offsets.append(members[spatial_order[picks]])
    selected_offsets = torch.cat(selected_visual_offsets)
    selected = positions[selected_offsets]
    if selected.numel() != keep_count or torch.unique(selected).numel() != keep_count:
        raise RuntimeError(
            f"visual selector produced {selected.numel()} unique positions; "
            f"expected {keep_count}"
        )
    return selected


def build_visual_context_keep_mask(
    input_ids: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    *,
    visual_token_ids: Sequence[int],
    retention: float,
    selector: str,
) -> torch.Tensor:
    """Keep every text key and exactly ceil(retention * visual_count) visual keys."""

    if selector not in _SELECTORS:
        raise ValueError(f"unknown visual selector {selector!r}")
    if not 0.0 < retention <= 1.0:
        raise ValueError("visual retention must be in (0, 1]")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, sequence], got {input_ids.shape}")
    keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if not visual_token_ids:
        raise ValueError("self-distillation requires configured visual token IDs")
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in visual_token_ids:
        visual_mask |= input_ids == int(token_id)
    if selector == "temporal_spatial" and position_ids is None:
        raise ValueError("temporal_spatial selection requires Qwen M-RoPE position_ids")

    if position_ids is not None:
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            batch_positions = position_ids
        elif position_ids.ndim == 3 and position_ids.shape[1] == 3:
            batch_positions = position_ids.transpose(0, 1)
        else:
            raise ValueError(
                "visual selection requires position_ids [3,batch,sequence] or "
                f"[batch,3,sequence], got {tuple(position_ids.shape)}"
            )
        if batch_positions.shape[1:] != input_ids.shape:
            raise ValueError("position_ids batch/sequence dimensions must match input_ids")
    else:
        batch_positions = None

    for batch_index in range(input_ids.shape[0]):
        positions = torch.nonzero(visual_mask[batch_index], as_tuple=False).flatten()
        visual_count = int(positions.numel())
        if visual_count == 0:
            continue
        keep_count = min(visual_count, math.ceil(retention * visual_count))
        if selector == "prefix":
            selected = positions[:keep_count]
        else:
            position_row = batch_positions[:, batch_index]
            visual_indices = positions.detach().cpu()
            selected_indices = _select_temporal_spatial(
                visual_indices,
                position_row.detach().cpu(),
                keep_count,
            )
            selected = selected_indices.to(device=input_ids.device)
        keep_mask[batch_index, positions] = False
        keep_mask[batch_index, selected] = True
        if int((keep_mask[batch_index] & visual_mask[batch_index]).sum().item()) != keep_count:
            raise RuntimeError("visual retention mask did not preserve the exact K count")
    return keep_mask


class OnlineDFlashSelfDistillModel(OnlineDFlashModel):
    """Train a reduced-visual DFlash student against target and full DFlash."""

    def __init__(
        self,
        *,
        visual_token_ids: Sequence[int],
        visual_retention: float = 0.1,
        visual_selector: str = "temporal_spatial",
        recipe: str = "dual",
        kd_weight: float = 0.5,
        kd_temperature: float = 1.0,
        teacher_draft_model: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if recipe not in _RECIPES:
            raise ValueError(f"recipe must be one of {sorted(_RECIPES)}, got {recipe!r}")
        if visual_selector not in _SELECTORS:
            raise ValueError(
                f"visual_selector must be one of {sorted(_SELECTORS)}, got {visual_selector!r}"
            )
        if not 0.0 < visual_retention <= 1.0:
            raise ValueError("visual_retention must be in (0, 1]")
        if kd_weight < 0.0:
            raise ValueError("kd_weight must be >= 0")
        if kd_temperature <= 0.0:
            raise ValueError("kd_temperature must be > 0")
        if self.loss_type != "dflash":
            raise ValueError("self-distillation currently requires loss_type='dflash'")
        self.teacher_draft_model = (
            teacher_draft_model
            if teacher_draft_model is not None
            else copy.deepcopy(self.draft_model)
        )
        self.teacher_draft_model.requires_grad_(False)
        self.teacher_draft_model.eval()
        self.lm_head.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)
        self.visual_token_ids = tuple(sorted({int(token_id) for token_id in visual_token_ids}))
        self.visual_retention = float(visual_retention)
        self.visual_selector = visual_selector
        self.recipe = recipe
        self.kd_weight = float(kd_weight)
        self.kd_temperature = float(kd_temperature)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_draft_model.eval()
        self.lm_head.eval()
        self.embed_tokens.eval()
        return self

    def _distill_chunk_terms(
        self,
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, num_blocks, block_size, hidden_size = student_hidden.shape
        temperature = self.kd_temperature
        with torch.no_grad():
            teacher_logits = self.lm_head(
                teacher_hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
            ).reshape(batch_size, num_blocks, block_size, -1).float()
            teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            teacher_ids = teacher_logits.argmax(dim=-1)

        student_logits = self.lm_head(
            student_hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1).float()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_ids = student_logits.argmax(dim=-1)
        student_ce = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        kd = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1) * (temperature**2)

        target_correct = target_ids == teacher_ids
        student_correct = target_ids == student_ids
        weights = weight_mask
        decay_gamma = self.loss_decay_gamma
        if decay_gamma is not None and decay_gamma > 0:
            positions = torch.arange(block_size, device=student_hidden.device).view(1, 1, -1)
            decay = torch.exp(
                -(positions - 1).clamp(min=0).float() / float(decay_gamma)
            )
            weights = weights * decay
        weighted_gate = weights * target_correct.float()
        return (
            (student_ce * weights).sum(),
            (kd * weights).sum(),
            (kd * weighted_gate).sum(),
            weighted_gate.sum(),
            weights.sum(),
            (student_correct.float() * weights).sum(),
            (target_correct.float() * weights).sum(),
            ((student_correct & target_correct).float() * weights).sum(),
            ((target_correct & ~student_correct).float() * weights).sum(),
            ((student_correct & ~target_correct).float() * weights).sum(),
            ((~student_correct & ~target_correct).float() * weights).sum(),
        )

    def _target_ce_chunk_terms(
        self,
        _teacher_hidden: Optional[torch.Tensor],
        student_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Target-only ablation with the same additive reduction contract."""

        batch_size, num_blocks, block_size, hidden_size = student_hidden.shape
        logits = self.lm_head(
            student_hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1).float()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        predicted = logits.argmax(dim=-1)
        weights = weight_mask
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(block_size, device=student_hidden.device).view(1, 1, -1)
            decay = torch.exp(
                -(positions - 1).clamp(min=0).float() / float(self.loss_decay_gamma)
            )
            weights = weights * decay
        zeros = weights.new_zeros(())
        correct = (predicted == target_ids).float() * weights
        return (
            (ce * weights).sum(),
            zeros,
            zeros,
            zeros,
            weights.sum(),
            correct.sum(),
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        if self.attention_backend == "flex_attention" and not self._flex_available():
            raise ValueError("flex_attention is not available on this device; use sdpa/eager.")
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        anchors, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        student_context_mask = build_visual_context_keep_mask(
            input_ids,
            position_ids,
            visual_token_ids=self.visual_token_ids,
            retention=self.visual_retention,
            selector=self.visual_selector,
        )

        # Both branches consume the same clean sequence, target features,
        # anchors, mask tokens, and draft positions. Only the student context
        # keys at dropped visual locations are hidden.
        teacher_output = None
        if self.recipe != "target_ce":
            with torch.no_grad():
                _, _, teacher_output = self._forward_draft_blocks(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                    position_ids=position_ids,
                    anchor_positions=anchors,
                    block_keep_mask=block_keep_mask,
                    draft_model=self.teacher_draft_model,
                )
        _, _, student_output = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            position_ids=position_ids,
            anchor_positions=anchors,
            block_keep_mask=block_keep_mask,
            context_keep_mask=student_context_mask,
        )

        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        label_indices = anchors.unsqueeze(-1) + offsets
        valid_labels = label_indices < seq_len
        safe_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchors.size(1), -1),
            2,
            safe_indices,
        )
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_labels.float()
        weight_mask = weight_mask * (offsets > 0).float()
        supervised = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchors.size(1), -1),
            2,
            safe_indices,
        )
        weight_mask = weight_mask * supervised

        teacher_hidden = (
            teacher_output.reshape(batch_size, anchors.size(1), self.block_size, -1)
            if teacher_output is not None
            else None
        )
        student_hidden = student_output.reshape(
            batch_size, anchors.size(1), self.block_size, -1
        )
        (
            ce_num,
            kd_num,
            gated_kd_num,
            gate_num,
            weight_den,
            student_correct,
            teacher_correct,
            both_correct,
            teacher_only,
            student_only,
            both_wrong,
        ) = checkpointed_chunk_reduce(
            (
                self._target_ce_chunk_terms
                if self.recipe == "target_ce"
                else self._distill_chunk_terms
            ),
            teacher_hidden,
            student_hidden,
            target_ids,
            weight_mask,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        denominator = weight_den.clamp_min(1.0)
        ce_loss = ce_num / denominator
        kd_loss = kd_num / denominator
        gated_kd_loss = gated_kd_num / denominator
        gate_coverage = gate_num / denominator
        if self.recipe == "target_ce":
            objective_num = ce_num
        elif self.recipe == "teacher_kl":
            objective_num = self.kd_weight * kd_num
        elif self.recipe == "gated":
            objective_num = ce_num + self.kd_weight * gated_kd_num
        elif self.recipe == "gated_mean_scaled":
            objective_num = ce_num + self.kd_weight * gate_coverage.detach() * kd_num
        else:
            objective_num = ce_num + self.kd_weight * kd_num
        loss = objective_num / denominator
        ratio_metrics = {
            "acc": (student_correct.detach(), denominator.detach()),
            "student_acc": (student_correct.detach(), denominator.detach()),
            "ce_loss": (ce_num.detach(), denominator.detach()),
        }
        if teacher_output is not None:
            ratio_metrics.update(
                {
                    "teacher_acc": (teacher_correct.detach(), denominator.detach()),
                    "both_correct": (both_correct.detach(), denominator.detach()),
                    "teacher_only_correct": (teacher_only.detach(), denominator.detach()),
                    "student_only_correct": (student_only.detach(), denominator.detach()),
                    "both_wrong": (both_wrong.detach(), denominator.detach()),
                    "gate_coverage": (gate_num.detach(), denominator.detach()),
                    "kd_loss": (kd_num.detach(), denominator.detach()),
                    "gated_kd_loss": (gated_kd_num.detach(), denominator.detach()),
                }
            )
        metrics: Dict[str, object] = {
            "accuracy_denom": denominator.detach(),
            "ratio_metrics": ratio_metrics,
            "loss_terms": (objective_num, denominator.detach()),
            "visual_tokens_total": int(
                sum(
                    int(((input_ids == token_id).sum()).item())
                    for token_id in self.visual_token_ids
                )
            ),
            "visual_tokens_retained": int(
                sum(
                    int(((input_ids == token_id) & student_context_mask).sum().item())
                    for token_id in self.visual_token_ids
                )
            ),
        }
        accuracy = student_correct / denominator
        return loss, accuracy, metrics

    @staticmethod
    def _flex_available() -> bool:
        from specforge.algorithms.common.dflash_family_model import FLEX_ATTENTION_AVAILABLE

        return FLEX_ATTENTION_AVAILABLE


__all__ = [
    "OnlineDFlashSelfDistillModel",
    "build_visual_context_keep_mask",
]
