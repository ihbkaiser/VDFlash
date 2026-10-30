"""Local SGLang capture for algorithm-owned offline feature preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class OfflineCaptureBatch:
    """Generic batched auxiliary and final target states."""

    hidden_states: Optional[torch.Tensor]
    last_hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: Optional[torch.Tensor] = None
    input_embeddings: Optional[torch.Tensor] = None
    visual_token_mask: Optional[torch.Tensor] = None


class OfflineSGLangCapture:
    """Frozen local target used by ``scripts/prepare_hidden_states.py`` only."""

    def __init__(self, backend) -> None:
        self._backend = backend
        self.capture_layers: Optional[List[int]] = None
        self.capture_method = "eagle3"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "OfflineSGLangCapture":
        from .sglang_backend import OfflineSGLangCaptureBackend

        backend = OfflineSGLangCaptureBackend.build(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(backend)

    def set_capture_layers(
        self,
        layer_ids: Optional[List[int]] = None,
        *,
        capture_method: str = "eagle3",
    ) -> None:
        self.capture_layers = layer_ids
        self.capture_method = capture_method
        self._backend.set_capture_layers(
            layer_ids,
            capture_method=capture_method,
        )

    def capture(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        multimodal_inputs: Optional[list[dict]] = None,
    ) -> OfflineCaptureBatch:
        capture_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
        # Preserve compatibility with small test/custom backends that still
        # implement the original text-only capture signature.
        if position_ids is not None:
            capture_kwargs["position_ids"] = position_ids
        if multimodal_inputs is not None:
            capture_kwargs["multimodal_inputs"] = multimodal_inputs
        backend_output = self._backend.capture(**capture_kwargs)
        if len(backend_output) == 5:
            data, aux_states, last_states, input_embeddings, visual_masks = (
                backend_output
            )
        else:
            data, aux_states, last_states = backend_output
            input_embeddings = visual_masks = None
        position_batch = None
        if position_ids is not None:
            position_batch = position_ids
        elif self.capture_method == "msd":
            base = attention_mask.to(torch.long).cumsum(-1).sub(1).clamp_min(0)
            position_batch = base.unsqueeze(0).expand(3, -1, -1)
        if self.capture_method == "msd" and (
            input_embeddings is None or visual_masks is None
        ):
            raise RuntimeError("MSD backend must return input embeddings and visual masks")
        return OfflineCaptureBatch(
            hidden_states=(
                None
                if all(value is None for value in aux_states)
                else pad_sequence(list(aux_states), batch_first=True)
            ),
            last_hidden_states=pad_sequence(list(last_states), batch_first=True),
            input_ids=torch.cat([row[0] for row in data], dim=0),
            attention_mask=torch.cat([row[1] for row in data], dim=0),
            loss_mask=torch.cat([row[2] for row in data], dim=0),
            position_ids=position_batch,
            input_embeddings=(
                pad_sequence(list(input_embeddings), batch_first=True)
                if input_embeddings is not None
                else None
            ),
            visual_token_mask=(
                pad_sequence(list(visual_masks), batch_first=True)
                if visual_masks is not None
                else None
            ),
        )


def load_offline_capture(
    pretrained_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> OfflineSGLangCapture:
    """Load the local SGLang target for offline hidden-state preparation."""

    return OfflineSGLangCapture.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


# Compatibility aliases for callers of the original EAGLE3-only surface.
OfflineEagle3CaptureBatch = OfflineCaptureBatch
OfflineEagle3SGLangCapture = OfflineSGLangCapture


def load_offline_eagle3_capture(
    pretrained_model_name_or_path: str,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> OfflineSGLangCapture:
    return load_offline_capture(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


__all__ = [
    "OfflineCaptureBatch",
    "OfflineEagle3CaptureBatch",
    "OfflineEagle3SGLangCapture",
    "OfflineSGLangCapture",
    "load_offline_capture",
    "load_offline_eagle3_capture",
]
