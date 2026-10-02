"""Compatibility helpers for strategy-specific SGLang capture hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class MSDInputEmbeddingCapture:
    """Capture the post-vision language-model input without changing outputs."""

    def __init__(self, module: Any) -> None:
        self.value: torch.Tensor | None = None
        self.handle = module.register_forward_pre_hook(
            self._capture,
            with_kwargs=True,
        )

    def _capture(self, _module, args, kwargs) -> None:
        value = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
        if value is None:
            floating = [
                item
                for item in args
                if isinstance(item, torch.Tensor)
                and item.ndim >= 2
                and torch.is_floating_point(item)
            ]
            if floating:
                value = floating[-1]
        if isinstance(value, torch.Tensor):
            self.value = value.detach()

    def consume(self, expected_tokens: int) -> torch.Tensor:
        value, self.value = self.value, None
        if value is None:
            raise RuntimeError(
                "MSD capture did not observe post-vision language input embeddings"
            )
        flattened = value.reshape(-1, value.shape[-1])
        if flattened.shape[0] != expected_tokens:
            raise RuntimeError(
                "MSD captured embedding/token length mismatch: "
                f"{flattened.shape[0]} != {expected_tokens}"
            )
        return flattened


def _attach_msd_input_capture(model: Any) -> MSDInputEmbeddingCapture:
    existing = getattr(model, "_specforge_msd_input_capture", None)
    if isinstance(existing, MSDInputEmbeddingCapture):
        return existing
    candidates = (
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
    )
    module = next(
        (
            candidate
            for candidate in candidates
            if callable(getattr(candidate, "register_forward_pre_hook", None))
        ),
        None,
    )
    if module is None:
        raise RuntimeError(
            "Qwen2.5-VL target does not expose a hookable language model for MSD"
        )
    capture = MSDInputEmbeddingCapture(module)
    model._specforge_msd_input_capture = capture
    return capture


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
        "msd": "set_msd_layers_to_capture",
    }.get(capture_method)
    if setter_name is None:
        raise ValueError(
            "offline SGLang capture method must be eagle3, dflash, dspark, "
            f"or msd; got {capture_method!r}"
        )

    setter = getattr(model, setter_name, None)
    if callable(setter):
        setter(layer_ids)
        return "native"

    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    decoder = _text_decoder_with_capture_state(model)
    if capture_method == "msd" and model_type == "qwen2_5_vl":
        _attach_msd_input_capture(model)
        if decoder is not None and hasattr(model, "capture_aux_hidden_states"):
            model.capture_aux_hidden_states = False
            decoder.layers_to_capture = []
        return "qwen2_5_vl_msd"
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


__all__ = ["MSDInputEmbeddingCapture", "configure_capture_layers"]
