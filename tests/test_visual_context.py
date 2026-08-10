import pytest
import torch

from src.analyze.Whether_they_are_appliable_for_dDrafter.qwen35_dflash_benchmark import (
    parse_visual_percentages,
)
from src.analyze.Whether_they_are_appliable_for_dDrafter.qwen35_dflash_video_decode import (
    draft_context_positions,
)


def _positions(percentage: float) -> torch.Tensor:
    # text, vision start, four visual tokens, vision end, text
    input_ids = torch.tensor([[10, 53, 57, 57, 57, 57, 54, 11]])
    return draft_context_positions(
        input_ids,
        vision_start_id=53,
        vision_end_id=54,
        video_token_id=57,
        visual_ratio=percentage / 100.0,
    )


def test_visual_percentages_keep_uniform_visual_positions():
    assert _positions(100).tolist() == list(range(8))
    assert _positions(50).tolist() == [0, 2, 5, 7]
    assert _positions(12.5).tolist() == [0, 2, 7]
    assert _positions(0).tolist() == [0, 7]


def test_visual_percentage_parser():
    assert parse_visual_percentages("100,50,12.5,0") == [100.0, 50.0, 12.5, 0.0]
    with pytest.raises(ValueError):
        parse_visual_percentages("101")
    with pytest.raises(ValueError):
        parse_visual_percentages("not-a-number")
