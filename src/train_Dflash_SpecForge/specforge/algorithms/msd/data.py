"""Tensor normalization for offline MSD teacher captures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Iterator, Optional

import torch
from torch import Tensor

from .curriculum import choose_visual_sample, visual_ratio_for_epoch

NORMALIZER_ID = "msd_qwen25vl_offline_v1"
RAW_FEATURE_KEYS = (
    "input_ids",
    "loss_mask",
    "target_hidden_state",
    "input_embeddings",
    "visual_token_mask",
    "position_ids",
)


def validate_msd_capture_record(record: Mapping[str, Tensor]) -> None:
    """Fail before persistence when a raw MSD capture cannot be normalized."""

    for name in RAW_FEATURE_KEYS:
        value = record.get(name)
        if not isinstance(value, Tensor):
            raise TypeError(f"MSD capture feature {name!r} must be a tensor")

    input_ids = record["input_ids"]
    loss_mask = record["loss_mask"]
    target = record["target_hidden_state"]
    embeddings = record["input_embeddings"]
    visual_mask = record["visual_token_mask"]
    positions = record["position_ids"]
    if positions.ndim == 3 and positions.shape[:2] == (3, 1):
        position_length = positions.shape[2]
    elif positions.ndim == 2 and positions.shape[0] == 3:
        position_length = positions.shape[1]
    else:
        raise ValueError(
            "MSD capture position_ids must use three-axis [3, sequence] positions"
        )

    def sequence_length(tensor: Tensor, name: str) -> int:
        if name in {"target_hidden_state", "input_embeddings"}:
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                return int(tensor.shape[1])
            if tensor.ndim == 2:
                return int(tensor.shape[0])
            raise ValueError(f"MSD capture {name} must have [S, H] or [1, S, H]")
        if tensor.ndim == 2 and tensor.shape[0] == 1:
            return int(tensor.shape[1])
        if tensor.ndim == 1:
            return int(tensor.shape[0])
        raise ValueError(f"MSD capture {name} must have [S] or [1, S]")

    lengths = {
        sequence_length(input_ids, "input_ids"),
        sequence_length(loss_mask, "loss_mask"),
        sequence_length(target, "target_hidden_state"),
        sequence_length(embeddings, "input_embeddings"),
        sequence_length(visual_mask, "visual_token_mask"),
        int(position_length),
    }
    if len(lengths) != 1:
        raise ValueError("MSD capture sequence lengths must match")
    if embeddings.shape[-1] != target.shape[-1]:
        raise ValueError("MSD capture hidden widths must match")
    if visual_mask.dtype != torch.bool:
        raise ValueError("MSD visual_token_mask must have boolean dtype")


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
    # The public MSD loader feeds token t+1 (including post-vision embeddings)
    # beside target hidden state t, so the visual bypass mask shifts with it.
    next_token_embeddings = _shift_left_with_zeros(input_embeddings)

    return {
        "input_ids": _shift_left_with_zeros(input_ids),
        "loss_mask": loss_mask,
        "target_hidden_state": _shift_left_with_zeros(target_hidden),
        "conditioning_hidden_state": target_hidden,
        "next_token_embeddings": next_token_embeddings,
        "visual_embeddings": next_token_embeddings,
        "visual_token_mask": _shift_left_with_zeros(visual_mask),
        "position_ids": position_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }


def build_offline_reader(
    hidden_states_path: str,
    *,
    run_id: str,
    ttt_length: int,
    max_len: int,
):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy="msd",
        feature_keys=RAW_FEATURE_KEYS,
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


class MSDPairedManifestReader:
    """Expose matched text/visual caches as one deterministic epoch cohort."""

    def __init__(
        self,
        text_root: str,
        visual_root: str,
        *,
        run_id: str,
        ttt_length: int,
        max_len: int,
        epoch_now: int,
        curriculum_seed: int = 0,
        total_epoch: int = 40,
    ) -> None:
        from specforge.runtime.data_plane.offline_reader import (
            OfflineManifestReader,
            list_feature_files,
        )

        text_files = list_feature_files(text_root)
        visual_files = list_feature_files(visual_root)
        if not text_files or not visual_files:
            raise ValueError(
                "MSD text and visual feature cohorts must both be non-empty "
                f"(text={len(text_files)}, visual={len(visual_files)})"
            )
        cohort_size = min(len(text_files), len(visual_files))
        common = {
            "run_id": run_id,
            "strategy": "msd",
            "feature_keys": RAW_FEATURE_KEYS,
            "target_repr": "hidden_state",
            "ttt_length": ttt_length,
            "max_len": max_len,
        }
        self.text_refs = OfflineManifestReader(text_root, **common).read()[
            :cohort_size
        ]
        self.visual_refs = OfflineManifestReader(visual_root, **common).read()[
            :cohort_size
        ]
        self.run_id = run_id
        self.epoch_now = epoch_now
        self.curriculum_seed = int(curriculum_seed)
        self.total_epoch = total_epoch
        self.visual_ratio = visual_ratio_for_epoch(epoch_now, total_epoch)
        self.realized_counts = {"text": 0, "visual": 0}

    def __iter__(self) -> Iterator:
        self.realized_counts = {"text": 0, "visual": 0}
        for index, (text_ref, visual_ref) in enumerate(
            zip(self.text_refs, self.visual_refs, strict=True)
        ):
            is_visual = choose_visual_sample(
                self.curriculum_seed,
                self.epoch_now,
                index,
                self.total_epoch,
            )
            corpus = "visual" if is_visual else "text"
            source = visual_ref if is_visual else text_ref
            self.realized_counts[corpus] += 1
            yield replace(
                source,
                sample_id=f"{self.run_id}:e{self.epoch_now:02d}:{index:08d}",
                metadata={
                    **source.metadata,
                    "msd_corpus": corpus,
                    "msd_epoch": self.epoch_now,
                    "msd_sample_index": index,
                    "msd_visual_ratio": self.visual_ratio,
                },
            )

    def read(self, limit: Optional[int] = None) -> list:
        refs = []
        for index, ref in enumerate(self):
            if limit is not None and index >= limit:
                break
            refs.append(ref)
        return refs


def build_paired_offline_reader(
    text_root: str,
    visual_root: str,
    *,
    run_id: str,
    ttt_length: int,
    max_len: int,
    epoch_now: int,
    curriculum_seed: int = 0,
    total_epoch: int = 40,
) -> MSDPairedManifestReader:
    return MSDPairedManifestReader(
        text_root,
        visual_root,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        epoch_now=epoch_now,
        curriculum_seed=curriculum_seed,
        total_epoch=total_epoch,
    )


def build_offline_normalizer(max_len: int, **_topology):
    return partial(normalize_offline_sample, max_len=max_len)


def build_offline_collator():
    """Right-pad MSD tensors, including three-axis M-RoPE positions."""

    required = (
        "input_ids",
        "loss_mask",
        "target_hidden_state",
        "conditioning_hidden_state",
        "next_token_embeddings",
        "visual_embeddings",
        "visual_token_mask",
        "position_ids",
        "attention_mask",
    )

    def collate(features):
        if not features:
            raise ValueError("cannot collate an empty MSD feature batch")
        missing = [
            (index, key)
            for index, feature in enumerate(features)
            for key in required
            if key not in feature
        ]
        if missing:
            raise KeyError(f"MSD feature batch is missing required keys: {missing}")
        max_length = max(int(item["input_ids"].shape[1]) for item in features)

        def pad(tensor: Tensor, axis: int) -> Tensor:
            amount = max_length - int(tensor.shape[axis])
            if amount < 0:
                raise ValueError("MSD feature is longer than input_ids")
            if amount == 0:
                return tensor
            shape = list(tensor.shape)
            shape[axis] = amount
            return torch.cat((tensor, tensor.new_zeros(shape)), dim=axis)

        return {
            key: torch.cat(
                [pad(item[key], 2 if key == "position_ids" else 1) for item in features],
                dim=1 if key == "position_ids" else 0,
            )
            for key in required
        }

    return collate


__all__ = [
    "NORMALIZER_ID",
    "RAW_FEATURE_KEYS",
    "build_offline_collator",
    "build_offline_normalizer",
    "build_offline_reader",
    "build_paired_offline_reader",
    "normalize_offline_sample",
    "validate_msd_capture_record",
]
