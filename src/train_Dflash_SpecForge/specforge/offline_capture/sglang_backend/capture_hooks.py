"""Compatibility helpers for strategy-specific SGLang capture hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _text_decoder_with_capture_state(model: Any) -> Any | None:
    candidates = (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "language_model", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers_to_capture"):
            return candidate
    return None


def configure_capture_layers(
    model: Any,
    layer_ids: Sequence[int] | None,
    *,
    capture_method: str,
) -> str:
    """Use a native hook, or Qwen2.5-VL's text-decoder capture state."""

    setter_name = {
        "eagle3": "set_eagle3_layers_to_capture",
        "dflash": "set_dflash_layers_to_capture",
        "dspark": "set_dspark_layers_to_capture",
    }.get(capture_method)
    if setter_name is None:
        raise ValueError(
            "offline SGLang capture method must be 'eagle3', 'dflash', or "
            f"'dspark', got {capture_method!r}"
        )

    setter = getattr(model, setter_name, None)
    if callable(setter):
        setter(layer_ids)
        return "native"

    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    decoder = _text_decoder_with_capture_state(model)
    if (
        capture_method in {"dflash", "dspark"}
        and model_type == "qwen2_5_vl"
        and decoder is not None
        and hasattr(model, "capture_aux_hidden_states")
    ):
        if layer_ids is None:
            raise ValueError(
                f"{capture_method.upper()} requires explicit layer_ids for "
                "Qwen2.5-VL text capture"
            )
        resolved = list(layer_ids)
        if (
            not resolved
            or any(isinstance(value, bool) or not isinstance(value, int) for value in resolved)
            or any(value < 0 for value in resolved)
            or len(set(resolved)) != len(resolved)
        ):
            raise ValueError(
                "Qwen2.5-VL text capture layer_ids must be distinct "
                f"non-negative integers, got {resolved!r}"
            )

        num_layers = getattr(getattr(decoder, "config", None), "num_hidden_layers", None)
        if isinstance(num_layers, int) and any(value + 1 >= num_layers for value in resolved):
            raise ValueError(
                "Qwen2.5-VL text capture layers must leave a following decoder "
                f"boundary (depth={num_layers}), got {resolved!r}"
            )

        model.capture_aux_hidden_states = True
        # SGLang stores the residual stream before layer i, so HF layer k is
        # available at the boundary before layer k + 1.
        decoder.layers_to_capture = [value + 1 for value in resolved]
        return "qwen2_5_vl_text"

    raise RuntimeError(
        f"target model does not expose SGLang capture hook {setter_name!r}; "
        "the only built-in fallback is text-only DFlash/DSpark capture for "
        "Qwen2.5-VL"
    )


__all__ = ["configure_capture_layers"]
