"""Qwen2.5-VL and SpecForge boundaries used by DFlash validation stages."""

from __future__ import annotations

import math
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _ensure_specforge_importable() -> None:
    specforge_root = Path(__file__).resolve().parents[2] / "train_Dflash_SpecForge"
    if str(specforge_root) not in sys.path:
        sys.path.insert(0, str(specforge_root))


def resolve_training_state(checkpoint_path: str):
    _ensure_specforge_importable()
    from specforge.export.checkpoint_io import resolve_training_state as _resolve

    return _resolve(checkpoint_path)


def materialize_draft(state: dict[str, Any], config_path: str):
    _ensure_specforge_importable()
    from specforge.export.checkpoint_io import materialize_draft as _materialize

    return _materialize(state, config_path)


def load_dflash_draft(checkpoint_path: str, config_path: str):
    """Resolve and materialize one DFlash training checkpoint."""

    state = resolve_training_state(checkpoint_path)
    draft = materialize_draft(state, config_path)
    return draft, state


def extract_target_hidden(hidden_states: Sequence[torch.Tensor], layer_ids: Sequence[int]) -> torch.Tensor:
    """Concatenate target hidden states selected by DFlash target layer IDs.

    Transformers returns the embedding output at index zero, so configured
    target layer ``N`` is read from ``hidden_states[N + 1]``.
    """

    if hidden_states is None:
        raise RuntimeError("target forward did not return hidden_states")
    try:
        selected = [hidden_states[int(layer_id) + 1] for layer_id in layer_ids]
    except (IndexError, TypeError) as exc:
        raise RuntimeError("target hidden_states do not contain configured DFlash layers") from exc
    if not selected:
        raise ValueError("layer_ids must not be empty")
    return torch.cat(selected, dim=-1)


def find_visual_positions(
    input_ids: torch.Tensor,
    *,
    target: Any | None = None,
    processor: Any | None = None,
) -> list[int]:
    """Locate Qwen2.5-VL video placeholder positions in one prompt."""

    token_id = getattr(getattr(target, "config", None), "video_token_id", None)
    if token_id is None:
        token_id = getattr(processor, "video_token_id", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    if token_id is None:
        token_id = getattr(tokenizer, "video_token_id", None)
    if token_id is None:
        raise ValueError("Qwen2.5-VL target/processor has no video_token_id")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    return (input_ids[0] == int(token_id)).nonzero(as_tuple=False).flatten().tolist()


def input_fingerprint(inputs: Mapping[str, Any]) -> str:
    """Hash token/grid preprocessing values without serializing video tensors."""

    digest = hashlib.sha256()
    for key in ("input_ids", "attention_mask", "video_grid_thw", "second_per_grid_ts"):
        value = inputs.get(key)
        if torch.is_tensor(value):
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif value is not None:
            digest.update(key.encode("utf-8"))
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()[:16]


def build_visual_retention_mask(
    *,
    total_length: int,
    visual_positions: Sequence[int],
    retention_percentage: int,
) -> torch.Tensor:
    """Return a boolean mask of hidden-context rows to zero.

    Visual positions are retained in their existing order.  Text positions are
    never masked.  The first ``ceil(retention * visual_count)`` visual rows are
    retained, making the operation deterministic and easy to audit.
    """

    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    if retention_percentage < 0 or retention_percentage > 100:
        raise ValueError("retention_percentage must be between 0 and 100")
    positions = [int(position) for position in visual_positions]
    if any(position < 0 or position >= total_length for position in positions):
        raise ValueError("visual_positions must be within total_length")

    mask = torch.zeros(total_length, dtype=torch.bool)
    retained_count = math.ceil(len(positions) * retention_percentage / 100.0)
    for position in positions[retained_count:]:
        mask[position] = True
    return mask


def apply_hidden_context_mask(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero selected sequence rows in a copied DFlash conditioning tensor."""

    if hidden.ndim < 2:
        raise ValueError("hidden context must have at least batch and sequence dimensions")
    normalized = mask.to(device=hidden.device, dtype=torch.bool)
    if normalized.ndim == 2:
        if normalized.shape[0] != hidden.shape[0]:
            raise ValueError("2D hidden mask batch dimension does not match hidden context")
        if normalized.shape[1] != hidden.shape[1]:
            raise ValueError("hidden mask sequence dimension does not match hidden context")
        view_shape = normalized.shape + (1,) * (hidden.ndim - 2)
    elif normalized.ndim == 1:
        if normalized.shape[0] != hidden.shape[1]:
            raise ValueError("hidden mask sequence dimension does not match hidden context")
        view_shape = (1, normalized.shape[0]) + (1,) * (hidden.ndim - 2)
    else:
        raise ValueError("hidden mask must be one- or two-dimensional")
    return hidden.masked_fill(normalized.reshape(view_shape), 0)


def load_qwen25vl_target(*args: Any, **kwargs: Any):
    """Lazy target loader hook kept out of CPU-only module imports.

    The concrete loader is intentionally supplied by the orchestration stage,
    because device maps, quantization, and processor options vary by host.
    """

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_name = kwargs.pop("model_name", args[0] if args else None)
    if not model_name:
        raise ValueError("model_name is required")
    processor = AutoProcessor.from_pretrained(model_name, **kwargs.pop("processor_kwargs", {}))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    return model, processor


__all__ = [
    "apply_hidden_context_mask",
    "build_visual_retention_mask",
    "extract_target_hidden",
    "find_visual_positions",
    "input_fingerprint",
    "load_dflash_draft",
    "load_qwen25vl_target",
]
