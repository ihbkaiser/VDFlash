import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


class _FakeTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(input_ids=[60, 61, 62])


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        prompt = "USER:<image>Describe\nASSISTANT:"
        if add_generation_prompt:
            return prompt
        return prompt + "A cat.<|end|>"

    def __call__(self, *, text, images, padding, return_tensors):
        assert text == ["USER:<image>Describe\nASSISTANT:"]
        assert len(images) == 1
        assert padding is True
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor(
                [[151644, 151652, 151655, 151655, 151655, 151655, 151653, 42]]
            ),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "pixel_values": torch.ones(16, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
        }


def _target_config():
    return SimpleNamespace(
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=25),
        text_config=SimpleNamespace(),
    )


def test_prepare_training_example_masks_only_response(tmp_path):
    from specforge.qwen25vl import prepare_training_example

    Image.new("RGB", (224, 224), color="white").save(tmp_path / "cat.jpg")
    example = prepare_training_example(
        _FakeProcessor(),
        _target_config(),
        {
            "id": "x",
            "image": "cat.jpg",
            "prompt": "Describe",
            "response": "A cat.",
        },
        image_root=tmp_path,
        max_length=64,
        image_min_pixels=200704,
        image_max_pixels=200704,
    )

    assert tuple(example["input_ids"].shape) == (1, 11)
    assert tuple(example["loss_mask"].shape) == (1, 11)
    assert example["loss_mask"][0, 0].item() == 0
    assert example["loss_mask"].sum().item() == 3
    assert tuple(example["position_ids"].shape) == (3, 1, 11)
    assert "pixel_values" in example["multimodal_inputs"]


def test_position_ids_reject_image_grid_mismatch():
    from specforge.qwen25vl import compute_qwen25vl_position_ids

    with pytest.raises(ValueError, match="image token/grid mismatch"):
        compute_qwen25vl_position_ids(
            torch.tensor([[1, 2, 3]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            config=_target_config(),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
        )


def test_position_ids_use_three_axes_for_image_grid():
    from specforge.qwen25vl import compute_qwen25vl_position_ids

    positions = compute_qwen25vl_position_ids(
        torch.tensor([[151652, 151655, 151655, 151655, 151655, 151653]]),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        config=_target_config(),
    )

    assert tuple(positions.shape) == (3, 1, 6)
    assert not torch.equal(positions[1], positions[2])


def test_position_ids_support_multiple_images_in_order():
    from specforge.qwen25vl import compute_qwen25vl_position_ids

    positions = compute_qwen25vl_position_ids(
        torch.tensor(
            [
                [
                    151652,
                    151655,
                    151655,
                    151655,
                    151655,
                    151653,
                    151652,
                    151655,
                    151655,
                    151655,
                    151655,
                    151653,
                ]
            ]
        ),
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
        config=_target_config(),
    )

    assert tuple(positions.shape) == (3, 1, 12)
    assert not torch.equal(positions[1, 0, 1:5], positions[2, 0, 1:5])
    assert not torch.equal(positions[1, 0, 7:11], positions[2, 0, 7:11])


def test_image_mrope_temporal_axis_is_zero_for_all_image_patches():
    from specforge.qwen25vl import compute_qwen25vl_position_ids

    positions = compute_qwen25vl_position_ids(
        torch.tensor(
            [[151652] + [151655] * 8 + [151653]],
        ),
        image_grid_thw=torch.tensor([[2, 4, 4]]),
        config=_target_config(),
    )

    assert torch.equal(
        positions[0, 0, 1:9],
        torch.full((8,), positions[0, 0, 1], dtype=torch.long),
    )
