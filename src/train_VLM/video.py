from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch


_VISUAL_CONTENT_TYPES = {"image", "image_url", "video"}


@dataclass(frozen=True)
class VideoProcessorMetadata:
    """Small, serializable summary of the materialized Qwen video input."""

    rendered_prompt: str
    frame_counts: tuple[int, ...]
    video_grid_thw: tuple[tuple[int, int, int], ...]


def messages_have_visual_content(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in _VISUAL_CONTENT_TYPES:
                return True
    return False


def _apply_media_defaults(
    messages: list[dict[str, Any]],
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_num_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    """Copy messages and fill missing per-video Qwen preprocessing fields."""

    materialized: list[dict[str, Any]] = []
    for message in messages:
        copied_message = dict(message)
        content = message.get("content")
        if not isinstance(content, list):
            materialized.append(copied_message)
            continue
        copied_content = []
        for item in content:
            copied_item = dict(item) if isinstance(item, dict) else item
            if isinstance(copied_item, dict) and copied_item.get("type") in {"image", "image_url"}:
                if image_min_pixels is not None:
                    copied_item.setdefault("min_pixels", image_min_pixels)
                if image_max_pixels is not None:
                    copied_item.setdefault("max_pixels", image_max_pixels)
            if isinstance(copied_item, dict) and copied_item.get("type") == "video":
                if (
                    video_num_frames is not None
                    and "nframes" not in copied_item
                    and "fps" not in copied_item
                ):
                    copied_item["nframes"] = video_num_frames
                if video_min_pixels is not None:
                    copied_item.setdefault("min_pixels", video_min_pixels)
                if video_max_pixels is not None:
                    copied_item.setdefault("max_pixels", video_max_pixels)
            copied_content.append(copied_item)
        copied_message["content"] = copied_content
        materialized.append(copied_message)
    return materialized


def _frame_count(value: Any) -> int | None:
    """Infer T from qwen-vl-utils' usual [T, C, H, W] video value."""

    if torch.is_tensor(value) and value.ndim >= 4:
        return int(value.shape[0])
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 4:
        return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _merge_processor_kwargs(
    video_kwargs: dict[str, Any], processor_kwargs: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(video_kwargs)
    for key, value in (processor_kwargs or {}).items():
        if key in merged and merged[key] != value:
            raise ValueError(
                f"processor kwarg {key!r} conflicts with the value produced by qwen-vl-utils"
            )
        merged[key] = value
    return merged


def prepare_qwen_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    device: torch.device,
    processor_kwargs: dict[str, Any] | None = None,
    video_reader: str = "torchvision",
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_num_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> tuple[dict[str, Any], VideoProcessorMetadata]:
    """Materialize one Qwen chat, including image/video tensors when present.

    Pure-text records retain the lightweight ``apply_chat_template`` path used
    by the existing tests and trainer. Multimodal records deliberately follow
    Qwen2.5-VL's explicit ``process_vision_info -> processor`` path so video FPS
    metadata and ``second_per_grid_ts`` are not lost.
    """

    materialized_messages = _apply_media_defaults(
        messages,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        video_num_frames=video_num_frames,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
    )
    if not messages_have_visual_content(materialized_messages):
        inputs = processor.apply_chat_template(
            materialized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **(processor_kwargs or {}),
        )
        values = dict(inputs)
        values = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in values.items()
        }
        return values, VideoProcessorMetadata("", (), ())

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - exercised by the real smoke test
        raise RuntimeError(
            "qwen-vl-utils is required for Qwen2.5-VL image/video records"
        ) from exc

    # qwen-vl-utils reads this environment variable when selecting its backend.
    # setdefault preserves an explicit choice made by the caller.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", video_reader)
    rendered = processor.apply_chat_template(
        materialized_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        materialized_messages, return_video_kwargs=True
    )
    call_kwargs = _merge_processor_kwargs(dict(video_kwargs or {}), processor_kwargs)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **call_kwargs,
    )
    values = dict(inputs)
    values = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in values.items()
    }

    frame_counts = []
    for video in video_inputs or ():
        count = _frame_count(video)
        if count is not None:
            frame_counts.append(count)
    grid = values.get("video_grid_thw")
    grid_values: tuple[tuple[int, int, int], ...] = ()
    if torch.is_tensor(grid):
        grid_values = tuple(tuple(int(x) for x in row) for row in grid.detach().cpu().tolist())
    return values, VideoProcessorMetadata(
        rendered_prompt=rendered,
        frame_counts=tuple(frame_counts),
        video_grid_thw=grid_values,
    )
