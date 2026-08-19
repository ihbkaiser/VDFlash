# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""EAGLE3 per-step features, forward pass, loss, and projection."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch


@dataclass(frozen=True)
class StepOutput:
    """Per-step result and optional globally normalized loss terms."""

    loss: torch.Tensor
    metrics: Dict[str, Any]
    ratio_metrics: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    loss_terms: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


@dataclass(frozen=True)
class StepContext:
    """Training-schedule state passed into the strategy."""

    global_step: int = 0
    total_steps: Optional[int] = None


class DraftTrainStrategy(abc.ABC):
    name: str
    required_features: set

    @abc.abstractmethod
    def trainable_module(self) -> nn.Module:
        """Return the module whose parameters the optimizer owns."""

    def validate_batch(self, batch: TrainBatch) -> None:
        missing = {f for f in self.required_features if f not in batch.tensors}
        if missing:
            raise ValueError(
                f"{self.name} batch missing required features {sorted(missing)}; "
                f"present={sorted(batch.tensors)}"
            )

    @abc.abstractmethod
    def forward_loss(
        self, batch: TrainBatch, ctx: Optional["StepContext"] = None
    ) -> StepOutput: ...

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Select the keys this strategy persists as draft weights."""
        return state_dict


def _prepare_eagle_target(
    *,
    target_head: Optional[nn.Module],
    target_repr: Optional[str],
    input_ids: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize stored hidden states or logits for an EAGLE3 forward."""
    if target_repr == "hidden_state":
        if target_head is None:
            raise ValueError(
                "target_repr='hidden_state' requires a target_head to re-run "
                "the lm_head projection"
            )
        input_ids, target, loss_mask = target_head.preprocess(
            input_ids, target, loss_mask
        )
        target = target_head(target.to(device))
        return input_ids.to(device), target, loss_mask.to(device)
    return input_ids.to(device), target.to(device), loss_mask.to(device)


class Eagle3TrainStrategy(DraftTrainStrategy):
    """EAGLE3 TTT strategy wrapping the SpecForge EAGLE3 model."""

    name = "eagle3"
    required_features = {
        "input_ids",
        "attention_mask",
        "loss_mask",
        "hidden_state",
        "target",
    }

    def __init__(
        self,
        eagle3_model: nn.Module,
        *,
        target_head: Optional[nn.Module] = None,
        ploss_decay: float = 0.8,
        compact_teacher: bool = False,
        compact_teacher_chunk_size: Optional[int] = None,
    ) -> None:
        self.eagle3_model = eagle3_model
        self.target_head = target_head
        self.ploss_decay = ploss_decay
        self.compact_teacher = compact_teacher
        self.compact_teacher_chunk_size = compact_teacher_chunk_size
        if compact_teacher:
            self._validate_compact_teacher()

    def _validate_compact_teacher(self) -> None:
        """Validate the offline compact-teacher contract before training."""
        if self.target_head is None:
            raise ValueError(
                "compact teacher requires the offline target_head; it is not "
                "available for online capture"
            )

        model = self.eagle3_model
        if not hasattr(model, "draft_model") and hasattr(model, "module"):
            model = model.module
        draft_model = getattr(model, "draft_model", None)
        if draft_model is None:
            raise ValueError(
                "compact teacher requires an EAGLE3 model with a draft_model"
            )

        from specforge.core.compact_teacher import (
            validate_compact_teacher_enabled,
            validate_vocab_mapping_consistency,
        )

        target_head_weight = getattr(
            getattr(self.target_head, "fc", None), "weight", None
        )
        vocab_size = (
            int(target_head_weight.shape[0])
            if target_head_weight is not None and target_head_weight.dim() >= 1
            else int(
                getattr(getattr(self.target_head, "config", None), "vocab_size", 0)
            )
        )
        draft_vocab_size = int(
            getattr(
                getattr(draft_model, "config", None),
                "draft_vocab_size",
                int(draft_model.t2d.sum().item()),
            )
        )
        validate_compact_teacher_enabled(
            is_online=False,
            draft_vocab_size=draft_vocab_size,
            vocab_size=vocab_size,
            t2d=draft_model.t2d,
            target_head_weight=target_head_weight,
            chunk_size=self.compact_teacher_chunk_size,
        )
        validate_vocab_mapping_consistency(draft_model.t2d, draft_model.d2t)

    def trainable_module(self) -> nn.Module:
        return self.eagle3_model

    def _device(self) -> torch.device:
        return next(self.eagle3_model.parameters()).device

    def _prepare_target(
        self,
        target_repr: Optional[str],
        input_ids: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _prepare_eagle_target(
            target_head=self.target_head,
            target_repr=target_repr,
            input_ids=input_ids,
            target=target,
            loss_mask=loss_mask,
            device=device,
        )

    def forward_loss(
        self, batch: TrainBatch, ctx: Optional[StepContext] = None
    ) -> StepOutput:
        self.validate_batch(batch)
        t = batch.tensors
        device = self._device()
        target_repr = batch.metadata.get("target_repr")

        compact_kwargs: Dict[str, Any] = {}
        if self.compact_teacher:
            if target_repr != "hidden_state":
                raise ValueError(
                    "compact teacher is offline-only and requires "
                    "target_repr='hidden_state'"
                )
            input_ids, target_hidden, loss_mask = self.target_head.preprocess(
                t["input_ids"], t["target"], t["loss_mask"]
            )
            input_ids = input_ids.to(device)
            target_hidden = target_hidden.to(device)
            loss_mask = loss_mask.to(device)
            from specforge.core.compact_teacher import build_offline_teacher_inputs

            target, compact_kwargs = build_offline_teacher_inputs(
                compact=True,
                target_model=self.target_head,
                target_hidden=target_hidden,
                chunk_size_arg=self.compact_teacher_chunk_size,
            )
        else:
            input_ids, target, loss_mask = self._prepare_target(
                target_repr, t["input_ids"], t["target"], t["loss_mask"], device
            )

        position_ids = t.get("position_ids")
        (
            plosses,
            acceptance_rates,
            acces,
            acc_corrects,
            acc_denoms,
            metric_losses,
            metric_loss_denoms,
        ) = self.eagle3_model(
            input_ids=input_ids,
            attention_mask=t["attention_mask"].to(device),
            loss_mask=loss_mask,
            target=target,
            hidden_states=t["hidden_state"].to(device),
            position_ids=position_ids.to(device) if position_ids is not None else None,
            **compact_kwargs,
        )
        weights = [self.ploss_decay**i for i in range(len(plosses))]
        loss = sum(weights[i] * plosses[i] for i in range(len(plosses)))
        return StepOutput(
            loss=loss,
            metrics={
                "plosses": [p.detach() for p in plosses],
                "acces": [a.detach() for a in acces],
                "acceptance_rates": [a.detach() for a in acceptance_rates],
                "acc_corrects": [c.detach() for c in acc_corrects],
                "acc_denoms": [d.detach() for d in acc_denoms],
                "metric_losses": [m.detach() for m in metric_losses],
                "metric_loss_denoms": [d.detach() for d in metric_loss_denoms],
            },
        )

    def checkpoint_state_filter(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Persist trainable draft weights, omitting a frozen copied embedding."""
        embed_frozen = all(
            not p.requires_grad
            for n, p in self.eagle3_model.named_parameters()
            if "embed" in n.lower()
        )
        return {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k and not (embed_frozen and "embed" in k.lower())
        }


__all__ = ["DraftTrainStrategy", "Eagle3TrainStrategy", "StepOutput", "StepContext"]
