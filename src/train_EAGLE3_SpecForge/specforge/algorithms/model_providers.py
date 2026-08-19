# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""EAGLE3 model and configuration providers used by the phase-one runtime.

Heavy imports remain inside callables so resolving the EAGLE3 registry stays
dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from specforge.config import Config


@dataclass
class AlgorithmModelParts:
    """Model and frozen target components returned to the assembler."""

    model: Any
    target_head: Any = None
    capture_layers: Optional[List[int]] = None


def _torch_dtype(cfg: Config):
    import torch

    return getattr(torch, cfg.model.torch_dtype)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _warm_start(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    *,
    allow_missing_embedding: bool = False,
) -> None:
    if not cfg.model.draft_checkpoint_path:
        return
    from specforge.training.model_loading import warm_start_draft_model

    warm_start_draft_model(
        draft_model,
        cfg.model.draft_checkpoint_path,
        draft_config=draft_config,
        strategy=cfg.training.strategy,
        allow_missing_embedding=allow_missing_embedding,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )


def _load_vocab_mapping(cfg: Config, draft_model: Any) -> None:
    if cfg.model.vocab_mapping_path:
        draft_model.load_vocab_mapping(cfg.model.vocab_mapping_path)


def build_eagle3_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_model = AutoDraftModel.from_config(
        draft_config,
        attention_backend=cfg.training.attention_backend,
        torch_dtype=_torch_dtype(cfg),
    )
    _warm_start(
        cfg,
        draft_model,
        draft_config,
        allow_missing_embedding=True,
    )
    _load_vocab_mapping(cfg, draft_model)
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
        )
    draft_model.freeze_embedding()
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def resolve_eagle_capture_layers(
    cfg: Config, draft_config: Any, target_config: Any
) -> List[int]:
    """Resolve the three auxiliary target layers used by EAGLE3."""

    layers = cfg.model.aux_hidden_state_layer_ids
    if layers is None:
        eagle_config = (
            draft_config.get("eagle_config", {})
            if isinstance(draft_config, dict)
            else getattr(draft_config, "eagle_config", {})
        ) or {}
        layers = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if layers is None:
        target_config = getattr(target_config, "text_config", target_config)
        num_layers = int(target_config.num_hidden_layers)
        layers = [1, num_layers // 2 - 1, num_layers - 4]
    layers = list(layers)
    if len(layers) != 3 or any(not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError(
            "resolved EAGLE capture layers must contain exactly three "
            f"non-negative integers, got {layers!r}"
        )
    return layers


def build_eagle3_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    _tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.eagle3.model import OnlineEagle3Model

    model = OnlineEagle3Model(
        draft_model=draft_model,
        length=cfg.training.ttt_length,
        attention_backend=cfg.training.attention_backend,
        lk_loss_type=cfg.training.lk_loss_type,
        kl_scale=cfg.training.kl_scale,
        kl_decay=cfg.training.kl_decay,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    target_head = None
    if cfg.mode == "offline" or (
        cfg.deployment.mode == "disaggregated" and cfg.training.role == "consumer"
    ):
        from specforge.modeling.target.target_head import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
            dtype=_torch_dtype(cfg),
        )
    return AlgorithmModelParts(model=model, target_head=target_head)


def eagle3_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "compact_teacher": cfg.training.compact_teacher,
        "compact_teacher_chunk_size": cfg.training.compact_teacher_chunk_size,
    }


__all__ = [
    "AlgorithmModelParts",
    "build_eagle3_draft",
    "build_eagle3_model",
    "eagle3_strategy_kwargs",
    "resolve_eagle_capture_layers",
]
