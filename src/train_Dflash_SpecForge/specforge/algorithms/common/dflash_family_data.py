"""Shared DFlash-family normalization and padding adapters."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.collation import pad_and_concatenate_features
from specforge.data.loss_mask import has_consecutive_supervised_tokens

NORMALIZER_ID = "dflash_family_offline_v1"
QWEN25VL_NORMALIZER_ID = "dflash_qwen25vl_offline_v1"
DSPARK_NORMALIZER_ID = "dspark_offline_v1"


def _normalize_hidden_states(
    raw,
    key: str,
    max_len: int,
    *,
    description: str,
):
    hidden_states = raw[key]
    if hidden_states.dim() == 3:
        if hidden_states.shape[0] != 1:
            raise ValueError(
                f"offline {description} must have shape [seq, width] or "
                f"[1, seq, width], got {tuple(hidden_states.shape)}"
            )
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.dim() != 2:
        raise ValueError(
            f"offline {description} must have shape [seq, width] or "
            f"[1, seq, width], got {tuple(hidden_states.shape)}"
        )
    return hidden_states[:max_len].unsqueeze(0)


def normalize_offline_sample(raw, max_len: int):
    """Normalize raw DFlash/Domino capture tensors without target projection."""

    input_ids = raw["input_ids"][:max_len].unsqueeze(0)
    loss_mask = raw["loss_mask"][:max_len].unsqueeze(0)
    hidden_states = _normalize_hidden_states(
        raw,
        "hidden_states",
        max_len,
        description="DFlash-family hidden_states",
    )
    lengths = {
        input_ids.shape[1],
        loss_mask.shape[1],
        hidden_states.shape[1],
    }
    if len(lengths) != 1:
        raise ValueError(
            "offline DFlash-family features have mismatched sequence lengths "
            f"after truncation: input_ids={input_ids.shape[1]}, "
            f"loss_mask={loss_mask.shape[1]}, "
            f"hidden_states={hidden_states.shape[1]}"
        )
    if not has_consecutive_supervised_tokens(loss_mask[0]):
        raise ValueError(
            "offline DFlash-family samples require two consecutive supervised tokens"
        )
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_states": hidden_states,
    }


def normalize_dspark_offline_sample(raw, max_len: int):
    """Normalize DSpark capture tensors, including target final-layer states."""

    normalized = normalize_offline_sample(raw, max_len)
    target_last_hidden_states = _normalize_hidden_states(
        raw,
        "target_last_hidden_states",
        max_len,
        description="DSpark target_last_hidden_states",
    )
    expected_length = normalized["input_ids"].shape[1]
    if target_last_hidden_states.shape[1] != expected_length:
        raise ValueError(
            "offline DSpark features have mismatched sequence lengths after "
            f"truncation: input_ids={expected_length}, "
            "target_last_hidden_states="
            f"{target_last_hidden_states.shape[1]}"
        )
    return {
        **normalized,
        "target_last_hidden_states": target_last_hidden_states,
    }


def normalize_qwen25vl_offline_sample(raw, max_len: int):
    """Normalize DFlash features carrying Qwen2.5-VL 3-axis positions."""

    normalized = normalize_offline_sample(raw, max_len)
    position_ids = raw.get("position_ids")
    if position_ids is None:
        raise KeyError("offline Qwen2.5-VL features require position_ids")
    if position_ids.dim() == 2:
        if position_ids.shape[0] != 3:
            raise ValueError(
                "Qwen2.5-VL position_ids must have shape [3, seq] or "
                f"[3, 1, seq], got {tuple(position_ids.shape)}"
            )
        position_ids = position_ids.unsqueeze(1)
    elif position_ids.dim() == 3:
        if position_ids.shape[0] != 3 or position_ids.shape[1] != 1:
            raise ValueError(
                "Qwen2.5-VL position_ids must have shape [3, seq] or "
                f"[3, 1, seq], got {tuple(position_ids.shape)}"
            )
    else:
        raise ValueError(
            "Qwen2.5-VL position_ids must have shape [3, seq] or "
            f"[3, 1, seq], got {tuple(position_ids.shape)}"
        )
    position_ids = position_ids[..., :max_len]
    expected_length = normalized["input_ids"].shape[1]
    if position_ids.shape[-1] != expected_length:
        raise ValueError(
            "offline Qwen2.5-VL features have mismatched sequence lengths after "
            f"truncation: input_ids={expected_length}, "
            f"position_ids={position_ids.shape[-1]}"
        )
    return {**normalized, "position_ids": position_ids}


def build_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
    feature_keys=("input_ids", "loss_mask", "hidden_states"),
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=tuple(feature_keys),
        target_repr=None,
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_dspark_offline_reader(
    strategy,
    hidden_states_path,
    *,
    run_id,
    ttt_length,
    max_len,
):
    # Transitional runtime import; the composition root will inject this port.
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        strategy=strategy,
        feature_keys=(
            "input_ids",
            "loss_mask",
            "hidden_states",
            "target_last_hidden_states",
        ),
        target_repr="hidden_state",
        ttt_length=ttt_length,
        max_len=max_len,
    )


def build_offline_normalizer(max_len, **_topology):
    return partial(normalize_offline_sample, max_len=max_len)


def build_dspark_offline_normalizer(max_len, **_topology):
    return partial(normalize_dspark_offline_sample, max_len=max_len)


def build_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
            },
            required_keys=("input_ids", "loss_mask", "hidden_states"),
        )

    return collate


def build_qwen25vl_collator():
    """Collate DFlash features with [axes, batch, sequence] positions."""

    def collate(features):
        if not features:
            raise ValueError("cannot collate an empty feature batch")
        import torch

        max_length = max(int(item["input_ids"].shape[-1]) for item in features)

        def pad(tensor, axis):
            length = int(tensor.shape[axis])
            if length == max_length:
                return tensor
            shape = list(tensor.shape)
            shape[axis] = max_length - length
            return torch.cat([tensor, tensor.new_zeros(shape)], dim=axis)

        batch = {
            "input_ids": torch.cat([pad(item["input_ids"], 1) for item in features], dim=0),
            "loss_mask": torch.cat([pad(item["loss_mask"], 1) for item in features], dim=0),
            "hidden_states": torch.cat(
                [pad(item["hidden_states"], 1) for item in features], dim=0
            ),
            "position_ids": torch.cat(
                [pad(item["position_ids"], 2) for item in features], dim=1
            ),
        }
        return batch

    return collate


def build_dspark_collator():
    def collate(features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={
                "input_ids": 1,
                "loss_mask": 1,
                "hidden_states": 1,
                "target_last_hidden_states": 1,
            },
            required_keys=(
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
            ),
        )

    return collate


__all__ = [
    "DSPARK_NORMALIZER_ID",
    "NORMALIZER_ID",
    "QWEN25VL_NORMALIZER_ID",
    "build_collator",
    "build_qwen25vl_collator",
    "build_dspark_collator",
    "build_dspark_offline_normalizer",
    "build_dspark_offline_reader",
    "build_offline_normalizer",
    "build_offline_reader",
    "normalize_dspark_offline_sample",
    "normalize_offline_sample",
    "normalize_qwen25vl_offline_sample",
]
