from __future__ import annotations

from dataclasses import dataclass
import inspect
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


def _install_qwen_pyav_fallback(vision_process: Any) -> None:
    """Use PyAV when qwen-vl-utils falls back to removed torchvision video IO."""
    readers = getattr(vision_process, "VIDEO_READER_BACKENDS", None)
    if not isinstance(readers, dict) or "torchvision" not in readers:
        return
    if getattr(vision_process, "_vdflash_pyav_fallback_installed", False):
        return

    original_fetch_video = getattr(vision_process, "fetch_video", None)
    fetch_source = inspect.getsource(original_fetch_video) if original_fetch_video else ""
    returns_metadata = "video_metadata, sample_fps" in fetch_source

    def read_video_with_pyav(ele: dict[str, Any]) -> tuple[Any, ...]:
        import av
        import numpy as np

        video_path = str(ele["video"])
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        container = av.open(video_path)
        try:
            if not container.streams.video:
                raise ValueError(f"video has no video stream: {video_path}")
            stream = container.streams.video[0]
            # Avoid the threaded decoder path that fails on some of these clips.
            stream.thread_type = "NONE"
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            if rate is None or float(rate) <= 0:
                raise ValueError(f"video has no usable frame rate: {video_path}")
            video_fps = float(rate)
            total_frames = int(stream.frames or 0)
            if total_frames <= 0:
                duration = stream.duration
                if duration is not None and stream.time_base is not None:
                    total_frames = round(float(duration * stream.time_base) * video_fps)
            if total_frames <= 0:
                # A second pass is unnecessary for the finite local corpus; retain
                # decoded frames only when the container omits frame-count metadata.
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
                total_frames = len(decoded)
                if not total_frames:
                    raise ValueError(f"video contains no decodable frames: {video_path}")
                frame_source = decoded
            else:
                frame_source = None

            calculate_range = getattr(vision_process, "calculate_video_frame_range", None)
            if calculate_range is not None:
                start_frame, end_frame, range_frames = calculate_range(
                    ele, total_frames, video_fps
                )
            else:
                start_frame, end_frame = 0, total_frames - 1
                range_frames = total_frames
                if ele.get("video_start") is not None:
                    start_frame = max(0, min(total_frames - 1, round(float(ele["video_start"]) * video_fps)))
                if ele.get("video_end") is not None:
                    end_frame = max(start_frame, min(total_frames - 1, round(float(ele["video_end"]) * video_fps)))
                range_frames = end_frame - start_frame + 1
            smart_nframes = getattr(vision_process, "smart_nframes", None)
            if smart_nframes is not None:
                nframes = int(smart_nframes(ele, total_frames=range_frames, video_fps=video_fps))
            elif "nframes" in ele:
                nframes = min(range_frames, max(2, int(ele["nframes"])))
            else:
                nframes = min(range_frames, max(2, round(range_frames / video_fps * float(ele.get("fps", 2.0)))))
            indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
            if frame_source is not None:
                selected = [frame_source[index] for index in indices]
            else:
                index_set = set(indices)
                selected = []
                for index, frame in enumerate(container.decode(stream)):
                    if index in index_set:
                        selected.append(frame.to_ndarray(format="rgb24"))
                if len(selected) != len(indices):
                    raise RuntimeError(
                        f"PyAV decoded {len(selected)} of {len(indices)} requested frames from {video_path}"
                    )
            video = torch.from_numpy(np.stack(selected)).permute(0, 3, 1, 2)
            sample_fps = nframes / max(range_frames, 1e-6) * video_fps
            if returns_metadata:
                metadata = {
                    "fps": video_fps,
                    "frames_indices": indices,
                    "total_num_frames": range_frames,
                    "video_backend": "pyav",
                }
                return video, metadata, sample_fps
            return video, sample_fps
        finally:
            container.close()

    readers["torchvision"] = read_video_with_pyav
    vision_process._vdflash_pyav_fallback_installed = True


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
    try:
        from qwen_vl_utils import vision_process
    except ImportError as exc:  # pragma: no cover - handled by the import above
        raise RuntimeError("qwen-vl-utils is required for Qwen2.5-VL video records") from exc
    _install_qwen_pyav_fallback(vision_process)
    rendered = processor.apply_chat_template(
        materialized_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        materialized_messages, return_video_kwargs=True
    )
    # qwen-vl-utils may return video-only metadata such as ``fps=[]`` for an
    # image-only request. Recent Transformers processors strictly validate FPS
    # as a scalar, so do not forward video kwargs when there are no videos.
    if video_inputs is None or len(video_inputs) == 0:
        video_kwargs = {}
    else:
        # qwen-vl-utils returns one FPS value per video, while current
        # Qwen2.5-VL processor kwargs declare ``fps`` as a scalar. This
        # function materializes one sample at a time, so a single video can be
        # safely unwrapped without changing its sampling rate.
        video_kwargs = dict(video_kwargs or {})
        fps = video_kwargs.get("fps")
        if isinstance(fps, (list, tuple)):
            if len(fps) != 1:
                raise ValueError(
                    "the current Qwen2.5-VL processor requires scalar fps; "
                    f"received {len(fps)} video FPS values for one sample"
                )
            video_kwargs["fps"] = fps[0]
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


def prepare_qwen_message_batch(
    processor: Any,
    messages_batch: list[list[dict[str, Any]]],
    *,
    processor_kwargs: dict[str, Any] | None = None,
    video_reader: str = "torchvision",
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_num_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
) -> dict[str, Any]:
    """Prepare a padded Qwen batch while keeping video FPS metadata per sample."""
    if not messages_batch:
        raise ValueError("messages_batch must not be empty")
    materialized = [
        _apply_media_defaults(
            messages,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            video_num_frames=video_num_frames,
            video_min_pixels=video_min_pixels,
            video_max_pixels=video_max_pixels,
        )
        for messages in messages_batch
    ]
    if any(not messages_have_visual_content(messages) for messages in materialized):
        raise ValueError("batched Qwen VLM preparation expects visual content in every example")
    old_padding_side = getattr(processor.tokenizer, "padding_side", "right")
    encoded_samples: list[dict[str, Any]] = []
    try:
        processor.tokenizer.padding_side = "left"
        for messages in materialized:
            inputs, _metadata = prepare_qwen_messages(
                processor,
                messages,
                device=torch.device("cpu"),
                processor_kwargs=processor_kwargs,
                video_reader=video_reader,
                image_min_pixels=image_min_pixels,
                image_max_pixels=image_max_pixels,
                video_num_frames=video_num_frames,
                video_min_pixels=video_min_pixels,
                video_max_pixels=video_max_pixels,
            )
            encoded_samples.append(inputs)
    finally:
        processor.tokenizer.padding_side = old_padding_side

    text_keys = {"input_ids", "attention_mask", "token_type_ids"}
    text_features = [
        {
            key: sample[key][0]
            for key in text_keys
            if key in sample
        }
        for sample in encoded_samples
    ]
    padded_text = processor.tokenizer.pad(
        text_features, padding=True, return_tensors="pt"
    )
    result = dict(padded_text)

    sequence_lengths = [int(sample["input_ids"].shape[-1]) for sample in encoded_samples]
    padded_sequence_length = int(result["input_ids"].shape[-1])
    pad_on_left = getattr(processor.tokenizer, "padding_side", "right") == "left"
    other_keys = set().union(*(sample.keys() for sample in encoded_samples)) - text_keys
    for key in other_keys:
        values = [sample[key] for sample in encoded_samples if key in sample]
        if len(values) != len(encoded_samples):
            raise ValueError(f"Qwen processor output {key!r} is missing from part of the batch")
        if all(torch.is_tensor(value) for value in values):
            is_sequence_tensor = all(
                value.ndim >= 2 and int(value.shape[-1]) == sequence_lengths[index]
                for index, value in enumerate(values)
            )
            if is_sequence_tensor and any(
                length != padded_sequence_length for length in sequence_lengths
            ):
                padded_values = []
                for value, length in zip(values, sequence_lengths):
                    amount = padded_sequence_length - length
                    padding = (amount, 0) if pad_on_left else (0, amount)
                    padded_value = torch.nn.functional.pad(value, padding, value=0)
                    if padded_value.ndim >= 3 and padded_value.shape[0] != 1:
                        batch_axis = 1
                    else:
                        batch_axis = 0
                    padded_values.append((padded_value, batch_axis))
                batch_axes = {axis for _value, axis in padded_values}
                if len(batch_axes) != 1:
                    raise ValueError(
                        f"Qwen sequence tensor {key!r} has incompatible batch axes"
                    )
                batch_axis = batch_axes.pop()
                result[key] = torch.cat(
                    [value for value, _axis in padded_values], dim=batch_axis
                )
                continue
            if values[0].ndim == 0:
                result[key] = torch.stack(values)
            else:
                result[key] = torch.cat(values, dim=0)
        elif all(isinstance(value, (list, tuple)) for value in values):
            result[key] = [item for value in values for item in value]
        else:
            result[key] = values
    return result
