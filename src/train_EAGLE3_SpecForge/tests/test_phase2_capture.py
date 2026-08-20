import sys
from pathlib import Path

import pytest
import torch


PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def test_capture_wrapper_forwards_multimodal_arguments():
    from specforge.offline_capture.sglang import OfflineSGLangCapture

    seen = {}

    class Backend:
        def capture(self, **kwargs):
            seen.update(kwargs)
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            loss_mask = kwargs["loss_mask"]
            return (
                [(input_ids, attention_mask, loss_mask)],
                [torch.zeros(4, 12)],
                [torch.zeros(4, 4)],
            )

    wrapper = OfflineSGLangCapture(Backend())
    batch = wrapper.capture(
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        loss_mask=torch.ones(1, 4),
        position_ids=torch.zeros(3, 1, 4, dtype=torch.long),
        multimodal_inputs=[{"pixel_values": torch.ones(1, 4)}],
    )

    assert seen["position_ids"].shape == (3, 1, 4)
    assert len(seen["multimodal_inputs"]) == 1
    assert tuple(batch.hidden_states.shape) == (1, 4, 12)
    assert tuple(batch.last_hidden_states.shape) == (1, 4, 4)


def test_multimodal_builder_rejects_missing_grid():
    from specforge.offline_capture.sglang_backend.multimodal import (
        build_qwen25vl_multimodal_inputs,
    )

    with pytest.raises(ValueError, match="image_grid_thw"):
        build_qwen25vl_multimodal_inputs(
            input_ids=torch.tensor([151655, 151655]),
            media={"pixel_values": torch.ones(2, 4)},
            position_ids=torch.zeros(3, 2, dtype=torch.long),
            image_token_id=151655,
        )


def test_multimodal_builder_rejects_image_grid_placeholder_mismatch():
    from specforge.offline_capture.sglang_backend.multimodal import (
        build_qwen25vl_multimodal_inputs,
    )

    with pytest.raises(ValueError, match="placeholder/grid mismatch"):
        build_qwen25vl_multimodal_inputs(
            input_ids=torch.tensor([151655, 42, 151655]),
            media={
                "pixel_values": torch.ones(4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            position_ids=torch.zeros(3, 3, dtype=torch.long),
            image_token_id=151655,
        )
