"""Standalone Qwen2.5-VL preparation for EAGLE3 Phase 2 capture."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch

from specforge.data.qwen25vl_manifest import safe_relative_image_path


def safe_image_path(image_root: str | os.PathLike[str], relative: str) -> Path:
    """Resolve an existing image below the configured image root."""

    resolved = safe_relative_image_path(image_root, relative)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _message_with_image_path(
    record: dict[str, Any],
    image_path: Path,
    *,
    image_min_pixels: int,
    image_max_pixels: int,
) -> list[dict[str, Any]]:
    messages = copy.deepcopy(record.get("messages"))
    if messages is None:
        prompt = record.get("prompt")
        response = record.get("response")
        if isinstance(prompt, str) and isinstance(response, str):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt.replace("<image>", "").strip()},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ]
    if not isinstance(messages, list):
        raise ValueError(f"record {record.get('id')} has no normalized messages")
    found_image = False
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for item in message["content"]:
            if isinstance(item, dict) and item.get("type") == "image":
                item["image"] = image_path.resolve().as_uri()
                item["min_pixels"] = image_min_pixels
                item["max_pixels"] = image_max_pixels
                found_image = True
    if not found_image:
        raise ValueError(f"record {record.get('id')} has no image content item")
    return messages


def _fallback_image_inputs(messages: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for message in messages:
        for item in message.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_uri = item.get("image")
            if not isinstance(image_uri, str):
                raise ValueError("Qwen2.5-VL image content has no image URI")
            parsed = urlparse(image_uri)
            image_path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(image_uri)
            try:
                from PIL import Image

                values.append(Image.open(image_path).convert("RGB"))
            except (ImportError, OSError):
                # Some processors accept local paths directly. Keep this fallback
                # usable for dependency-light unit tests as well.
                values.append(str(image_path))
    if not values:
        raise ValueError("Qwen2.5-VL prompt contains no image inputs")
    return values


def _prepare_prompt_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        image_inputs = _fallback_image_inputs(messages)
        video_inputs = []
        video_kwargs = {}
    else:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
    processor_kwargs: dict[str, Any] = {
        "text": [rendered],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
        processor_kwargs.update(dict(video_kwargs or {}))
    return dict(processor(**processor_kwargs))


def _response_token_ids(
    processor: Any,
    prompt_messages: list[dict[str, Any]],
    response: str,
) -> list[int]:
    prompt_rendered = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_rendered = processor.apply_chat_template(
        prompt_messages
        + [{"role": "assistant", "content": [{"type": "text", "text": response}]}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_rendered.startswith(prompt_rendered):
        raise RuntimeError("Qwen chat template did not preserve the prompt prefix")
    suffix = full_rendered[len(prompt_rendered) :]
    return list(processor.tokenizer(suffix, add_special_tokens=False).input_ids)


def _config_id(config: Any, text_config: Any, name: str) -> int:
    value = getattr(config, name, None)
    if value is None:
        value = getattr(text_config, name, None)
    if value is None:
        raise ValueError(f"Qwen2.5-VL config is missing {name}")
    return int(value)


def compute_qwen25vl_position_ids(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor,
    config: Any,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Qwen2.5-VL three-axis image/text M-RoPE positions."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("position preparation expects [1, sequence] input_ids")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError(
            "image_grid_thw must have shape [images, 3], got "
            f"{tuple(image_grid_thw.shape)}"
        )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape")

    vision_config = getattr(config, "vision_config", config)
    spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
    if spatial_merge_size <= 0:
        raise ValueError("Qwen2.5-VL spatial_merge_size must be positive")
    text_config = getattr(config, "text_config", config)
    image_token_id = _config_id(config, text_config, "image_token_id")
    vision_start_token_id = _config_id(config, text_config, "vision_start_token_id")

    active_mask = attention_mask[0].bool()
    active_ids = input_ids[0][active_mask]
    active_values = active_ids.tolist()
    image_count = sum(
        1
        for index, token in enumerate(active_values[:-1])
        if token == vision_start_token_id
        and active_values[index + 1] == image_token_id
    )
    if image_count != int(image_grid_thw.shape[0]):
        raise ValueError(
            "image token/grid mismatch: "
            f"tokens={image_count}, grids={int(image_grid_thw.shape[0])}"
        )

    pieces: list[torch.Tensor] = []
    cursor = 0
    image_index = 0
    next_scalar_position = 0
    while image_index < image_count:
        try:
            vision_start = active_values.index(vision_start_token_id, cursor)
        except ValueError as exc:
            raise ValueError("Qwen2.5-VL image start token is missing") from exc
        image_start = vision_start + 1
        if image_start >= len(active_values) or active_values[image_start] != image_token_id:
            raise ValueError("Qwen2.5-VL image start is not followed by image tokens")
        if image_start > cursor:
            text_length = image_start - cursor
            pieces.append(
                torch.arange(text_length, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + next_scalar_position
            )
            next_scalar_position = int(pieces[-1].max().item()) + 1

        t, h, w = [int(value) for value in image_grid_thw[image_index].tolist()]
        if min(t, h, w) <= 0 or h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(
                "image grid dimensions must be positive and divisible by "
                f"spatial_merge_size={spatial_merge_size}"
            )
        grid_h = h // spatial_merge_size
        grid_w = w // spatial_merge_size
        image_token_count = t * grid_h * grid_w
        image_end = image_start + image_token_count
        if image_end > len(active_values) or any(
            token != image_token_id for token in active_values[image_start:image_end]
        ):
            raise ValueError("image placeholder/grid mismatch")
        # Qwen2.5-VL's official get_rope_index uses zero seconds-per-grid for
        # images. A non-zero temporal stride is reserved for video inputs.
        time = torch.zeros(
            (t * grid_h * grid_w,), dtype=torch.long, device=input_ids.device
        )
        height = (
            torch.arange(grid_h, device=input_ids.device)
            .view(1, -1, 1)
            .expand(t, -1, grid_w)
            .flatten()
        )
        width = (
            torch.arange(grid_w, device=input_ids.device)
            .view(1, 1, -1)
            .expand(t, grid_h, -1)
            .flatten()
        )
        pieces.append(
            torch.stack([time, height, width])
            + next_scalar_position
        )
        next_scalar_position = int(pieces[-1].max().item()) + 1
        cursor = image_end
        image_index += 1

    if cursor < len(active_values):
        text_length = len(active_values) - cursor
        pieces.append(
            torch.arange(text_length, device=input_ids.device)
            .view(1, -1)
            .expand(3, -1)
            + next_scalar_position
        )
    if not pieces:
        raise ValueError("Qwen2.5-VL input has no active tokens")
    active_positions = torch.cat(pieces, dim=1).to(dtype=input_ids.dtype)
    if active_positions.shape[-1] != int(active_mask.sum().item()):
        raise AssertionError("position construction did not cover all active tokens")
    positions = torch.ones(
        3,
        1,
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    positions[:, 0, active_mask] = active_positions
    return positions


def prepare_training_example(
    processor: Any,
    target_config: Any,
    record: dict[str, Any],
    *,
    image_root: str | os.PathLike[str],
    max_length: int = 3072,
    image_min_pixels: int = 200704,
    image_max_pixels: int = 200704,
) -> dict[str, Any]:
    """Prepare one normalized image-caption row for teacher capture."""

    if max_length < 2:
        raise ValueError("max_length must be at least two")
    image_path = safe_image_path(image_root, record["image"])
    messages = _message_with_image_path(
        record,
        image_path,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    prompt_messages = messages[:1]
    inputs = _prepare_prompt_inputs(processor, prompt_messages)
    prompt_ids = inputs.get("input_ids")
    if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2:
        raise ValueError("Qwen processor must return [1, sequence] input_ids")
    response_ids = _response_token_ids(processor, prompt_messages, record["response"])
    if len(response_ids) < 2:
        raise ValueError(f"record {record.get('id')} response has fewer than two tokens")
    available = max_length - int(prompt_ids.shape[-1])
    if available < 2:
        raise ValueError(f"record {record.get('id')} leaves fewer than two response tokens")
    response_truncated = len(response_ids) > available
    if response_truncated:
        response_ids = response_ids[: max(0, available - 2)] + response_ids[-2:]
    response_tensor = torch.tensor(
        response_ids,
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    ).view(1, -1)
    full_ids = torch.cat([prompt_ids, response_tensor], dim=-1)
    inputs["input_ids"] = full_ids
    inputs["attention_mask"] = torch.ones_like(full_ids)
    for key in ("mm_token_type_ids", "token_type_ids"):
        value = inputs.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if value.shape[-1] == prompt_ids.shape[-1]:
            inputs[key] = torch.cat(
                [value, value.new_zeros((*value.shape[:-1], response_tensor.shape[-1]))],
                dim=-1,
            )
        elif value.shape[-1] != full_ids.shape[-1]:
            raise ValueError(f"{key} length does not match prompt/full input")
    loss_mask = torch.zeros_like(full_ids, dtype=torch.float32)
    loss_mask[:, prompt_ids.shape[-1] :] = 1.0
    image_grid_thw = inputs.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError("Qwen processor did not return image_grid_thw")
    position_ids = compute_qwen25vl_position_ids(
        full_ids,
        image_grid_thw=image_grid_thw,
        config=target_config,
        attention_mask=inputs["attention_mask"],
    )
    return {
        "input_ids": full_ids,
        "attention_mask": inputs["attention_mask"],
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "multimodal_inputs": {
            key: value
            for key, value in inputs.items()
            if key not in {"input_ids", "attention_mask", "position_ids"}
            and isinstance(value, torch.Tensor)
        },
        "response_truncated": response_truncated,
    }


__all__ = [
    "compute_qwen25vl_position_ids",
    "prepare_training_example",
    "safe_image_path",
]
