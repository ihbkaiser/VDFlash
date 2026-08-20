"""Qwen2.5-VL request conversion for the standalone EAGLE3 capture backend."""

from __future__ import annotations

from typing import Any

import torch


def _contiguous_ranges(values: list[int], token_id: int) -> list[tuple[int, int]]:
    positions = [index for index, value in enumerate(values) if value == token_id]
    if not positions:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            ranges.append((start, previous))
            start = position
        previous = position
    ranges.append((start, previous))
    return ranges


def build_qwen25vl_multimodal_inputs(
    *,
    input_ids: torch.Tensor,
    media: dict[str, Any],
    position_ids: torch.Tensor | None,
    image_token_id: int,
):
    """Build SGLang's lazy multimodal request object from processor tensors."""

    pixel_values = media.get("pixel_values")
    image_grid_thw = media.get("image_grid_thw")
    if not isinstance(pixel_values, torch.Tensor):
        raise ValueError("Qwen2.5-VL capture requires pixel_values")
    if not isinstance(image_grid_thw, torch.Tensor):
        raise ValueError("Qwen2.5-VL capture requires image_grid_thw")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[-1] != 3:
        raise ValueError("image_grid_thw must have shape [images, 3]")
    values = input_ids.view(-1).tolist()
    offsets = _contiguous_ranges(values, int(image_token_id))
    if not offsets:
        raise ValueError("Qwen2.5-VL input has no image placeholder tokens")
    if len(offsets) != int(image_grid_thw.shape[0]):
        raise ValueError(
            "image placeholder/grid mismatch: "
            f"offsets={len(offsets)}, grids={int(image_grid_thw.shape[0])}"
        )
    if position_ids is not None:
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            mrope_positions = position_ids[:, 0, :]
        elif position_ids.ndim == 2 and position_ids.shape[0] == 3:
            mrope_positions = position_ids
        else:
            raise ValueError(
                "Qwen2.5-VL capture requires [3, sequence] M-RoPE positions"
            )
    else:
        mrope_positions = None

    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    items = []
    pixel_cursor = 0
    for image_index, (start, end) in enumerate(offsets):
        grid_t, grid_h, grid_w = [
            int(value) for value in image_grid_thw[image_index].tolist()
        ]
        if min(grid_t, grid_h, grid_w) <= 0:
            raise ValueError("image_grid_thw dimensions must be positive")
        feature_count = grid_t * grid_h * grid_w
        feature = pixel_values[pixel_cursor : pixel_cursor + feature_count]
        if feature.shape[0] != feature_count:
            raise ValueError("pixel_values/image_grid_thw feature count mismatch")
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[(start, end)],
            model_specific_data={
                "image_grid_thw": image_grid_thw[image_index : image_index + 1]
            },
        )
        item.set_pad_value()
        items.append(item)
        pixel_cursor += feature_count

    padded_input_ids = values.copy()
    for item in items:
        for start, end in item.offsets:
            padded_input_ids[start : end + 1] = [item.pad_value] * (
                end - start + 1
            )
    return MultimodalInputs(
        mm_items=items,
        padded_input_ids=padded_input_ids,
        im_token_id=int(image_token_id),
        mrope_positions=mrope_positions,
    )


__all__ = ["build_qwen25vl_multimodal_inputs"]
