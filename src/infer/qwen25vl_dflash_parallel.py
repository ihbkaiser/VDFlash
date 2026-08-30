"""Device placement helpers for Qwen2.5-VL DFlash inference."""

from __future__ import annotations

from typing import Mapping


def parse_max_memory(
    value: str | Mapping[int, str] | None,
) -> dict[int, str] | None:
    """Parse the ``device_id:budget`` syntax accepted by Accelerate."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return {int(index): str(budget) for index, budget in value.items()}

    result: dict[int, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            index, budget = item.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                "max_memory must use comma-separated DEVICE:BUDGET values"
            ) from exc
        if not index.strip() or not budget.strip():
            raise ValueError(
                "max_memory must use comma-separated DEVICE:BUDGET values"
            )
        result[int(index.strip())] = budget.strip()
    if not result:
        raise ValueError("max_memory must contain at least one device budget")
    return result


def build_qwen25vl_model_parallel_map(
    num_hidden_layers: int,
    prefix_layers: int = 3,
    suffix_layers: int = 2,
) -> dict[str, int]:
    """Place Qwen2.5-VL's boundary modules and decoder layers on two GPUs.

    GPU 0 owns the vision tower, embedding/final path, and DFlash's draft
    device. GPU 1 owns the middle language-model layers.
    """

    num_hidden_layers = int(num_hidden_layers)
    prefix_layers = int(prefix_layers)
    suffix_layers = int(suffix_layers)
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if prefix_layers < 0 or suffix_layers < 0:
        raise ValueError("prefix_layers and suffix_layers must be non-negative")
    if prefix_layers + suffix_layers > num_hidden_layers:
        raise ValueError("prefix_layers and suffix_layers cannot exceed layer count")

    mapping: dict[str, int] = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        "model.language_model.norm": 0,
        "lm_head": 0,
    }
    for index in range(num_hidden_layers):
        is_boundary = index < prefix_layers or index >= num_hidden_layers - suffix_layers
        mapping[f"model.language_model.layers.{index}"] = 0 if is_boundary else 1
    return mapping
