"""Qwen2.5-VL input preparation shared by capture and HF smoke inference."""

from __future__ import annotations

import copy
import os
from pathlib import Path, PurePosixPath
from typing import Any

import torch


def safe_image_path(image_root: str | os.PathLike[str], relative: str) -> Path:
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe image path: {relative!r}")
    root = Path(image_root).expanduser().resolve()
    resolved = (root / Path(*path.parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"image path escapes image root: {relative!r}")
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
    messages = copy.deepcopy(record["messages"])
    inserted = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                item["image"] = image_path.resolve().as_uri()
                item["min_pixels"] = image_min_pixels
                item["max_pixels"] = image_max_pixels
                inserted = True
    if not inserted:
        raise ValueError(f"record {record.get('id')} has no image content item")
    return messages


def _prepare_prompt_inputs(processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "qwen-vl-utils is required for Qwen2.5-VL image capture"
        ) from exc
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
    )
    processor_kwargs = {
        "text": [rendered],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    # qwen-vl-utils returns ``video_inputs=[]`` and ``{"fps": []}`` for an
    # image-only conversation.  Recent Transformers/huggingface_hub releases
    # validate ``fps`` as a scalar before Qwen's processor can ignore it, so
    # forwarding those empty video values raises during image preprocessing.
    # Video-only keyword arguments are meaningful only when a video is present.
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
        processor_kwargs.update(dict(video_kwargs or {}))
    inputs = processor(
        **processor_kwargs,
    )
    return dict(inputs)


def _response_token_ids(processor: Any, prompt_messages: list[dict[str, Any]], response: str) -> list[int]:
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


def compute_qwen25vl_position_ids(
    input_ids: torch.Tensor,
    *,
    image_grid_thw: torch.Tensor,
    config: Any,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Qwen2.5-VL's official 3-axis image/text position layout."""

    vision_config = getattr(config, "vision_config", config)
    spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
    tokens_per_second = float(getattr(vision_config, "tokens_per_second", 25))
    text_config = getattr(config, "text_config", config)

    def config_id(name: str) -> int:
        value = getattr(config, name, None)
        if value is None:
            value = getattr(text_config, name, None)
        if value is None:
            raise ValueError(f"Qwen2.5-VL config is missing {name}")
        return int(value)

    image_token_id = config_id("image_token_id")
    vision_start_token_id = config_id("vision_start_token_id")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("position preparation currently expects [1, sequence] input_ids")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    positions = torch.ones(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    image_index = 0
    active_ids = input_ids[0][attention_mask[0] == 1]
    token_list = active_ids.tolist()
    starts = torch.argwhere(active_ids == vision_start_token_id).flatten()
    image_count = int(
        (active_ids[starts + 1] == image_token_id).sum().item()
    ) if starts.numel() else 0
    if image_count != int(image_grid_thw.shape[0]):
        raise ValueError(
            "image token/grid mismatch: "
            f"tokens={image_count}, grids={int(image_grid_thw.shape[0])}"
        )
    pieces: list[torch.Tensor] = []
    start = 0
    for _ in range(image_count):
        end = token_list.index(image_token_id, start)
        t, h, w = [int(value) for value in image_grid_thw[image_index].tolist()]
        image_index += 1
        grid_t = t
        grid_h = h // spatial_merge_size
        grid_w = w // spatial_merge_size
        text_len = end - start
        text_start = pieces[-1].max().item() + 1 if pieces else 0
        if text_len:
            pieces.append(
                torch.arange(text_len, device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
                + text_start
            )
        time = (
            torch.arange(grid_t, device=input_ids.device)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            * tokens_per_second
        ).long().flatten()
        height = (
            torch.arange(grid_h, device=input_ids.device)
            .view(1, -1, 1)
            .expand(grid_t, -1, grid_w)
            .flatten()
        )
        width = (
            torch.arange(grid_w, device=input_ids.device)
            .view(1, 1, -1)
            .expand(grid_t, grid_h, -1)
            .flatten()
        )
        pieces.append(
            torch.stack([time, height, width])
            + text_len
            + text_start
        )
        start = end + grid_t * grid_h * grid_w
    if start < len(token_list):
        text_start = pieces[-1].max().item() + 1 if pieces else 0
        pieces.append(
            torch.arange(len(token_list) - start, device=input_ids.device)
            .view(1, -1)
            .expand(3, -1)
            + text_start
        )
    active_positions = torch.cat(pieces, dim=1)
    positions[:, 0, attention_mask[0] == 1] = active_positions
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
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Prepare one manifest row and retain the dataset response verbatim."""

    image_path = safe_image_path(image_root, record["image"])
    messages = _message_with_image_path(
        record,
        image_path,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )
    prompt_messages = messages[:1]
    inputs = _prepare_prompt_inputs(processor, prompt_messages)
    prompt_ids = inputs["input_ids"]
    response_ids = _response_token_ids(processor, prompt_messages, record["response"])
    if len(response_ids) < 2:
        raise ValueError(
            f"record {record.get('id')} response has fewer than two target tokens"
        )
    available = max_length - int(prompt_ids.shape[1])
    if available < 16:
        raise ValueError(
            f"record {record.get('id')} leaves only {available} response tokens"
        )
    response_truncated = len(response_ids) > available
    if response_truncated:
        terminal = response_ids[-2:] if len(response_ids) >= 2 else response_ids[-1:]
        content_budget = available - len(terminal)
        if content_budget < 1:
            raise ValueError(f"record {record.get('id')} has no response content budget")
        response_ids = response_ids[:content_budget] + terminal
    response_tensor = torch.tensor(
        response_ids,
        dtype=prompt_ids.dtype,
        device=prompt_ids.device,
    ).view(1, -1)
    full_ids = torch.cat([prompt_ids, response_tensor], dim=1)
    full_length = int(full_ids.shape[1])
    if full_length > max_length:
        raise AssertionError("response truncation failed to enforce max_length")
    inputs["input_ids"] = full_ids
    inputs["attention_mask"] = torch.ones_like(full_ids)
    for key in ("mm_token_type_ids", "token_type_ids"):
        value = inputs.get(key)
        if not torch.is_tensor(value):
            continue
        if value.shape[-1] == prompt_ids.shape[-1]:
            inputs[key] = torch.cat(
                [value, value.new_zeros((*value.shape[:-1], response_tensor.shape[-1]))],
                dim=-1,
            )
        elif value.shape[-1] != full_ids.shape[-1]:
            raise ValueError(
                f"{key} length {value.shape[-1]} does not match prompt/full input"
            )
    loss_mask = torch.zeros_like(full_ids, dtype=torch.float32)
    loss_mask[:, int(prompt_ids.shape[1]) :] = 1.0
    image_grid_thw = inputs.get("image_grid_thw")
    if not torch.is_tensor(image_grid_thw):
        raise ValueError("Qwen processor did not return image_grid_thw")
    position_ids = compute_qwen25vl_position_ids(
        full_ids,
        image_grid_thw=image_grid_thw,
        config=target_config,
        attention_mask=inputs["attention_mask"],
    )
    inputs["position_ids"] = position_ids
    return {
        "input_ids": full_ids,
        "attention_mask": inputs["attention_mask"],
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "multimodal_inputs": {
            key: value
            for key, value in inputs.items()
            if key not in {"input_ids", "attention_mask", "position_ids"}
            and torch.is_tensor(value)
        },
        "response_truncated": response_truncated,
    }


def prepare_inference_prompt(
    processor: Any,
    target_config: Any,
    record: dict[str, Any],
    *,
    image_root: str | os.PathLike[str],
    image_min_pixels: int = 200704,
    image_max_pixels: int = 200704,
) -> dict[str, Any]:
    """Prepare a user image prompt and its target-compatible positions."""

    image_path = safe_image_path(image_root, record["image"])
    messages = _message_with_image_path(
        record,
        image_path,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )[:1]
    inputs = _prepare_prompt_inputs(processor, messages)
    input_ids = inputs["input_ids"]
    inputs["attention_mask"] = torch.ones_like(input_ids)
    inputs["position_ids"] = compute_qwen25vl_position_ids(
        input_ids,
        image_grid_thw=inputs["image_grid_thw"],
        config=target_config,
        attention_mask=inputs["attention_mask"],
    )
    inputs["multimodal_inputs"] = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids"}
        and torch.is_tensor(value)
    }
    return inputs


__all__ = [
    "compute_qwen25vl_position_ids",
    "prepare_training_example",
    "prepare_inference_prompt",
    "safe_image_path",
]
