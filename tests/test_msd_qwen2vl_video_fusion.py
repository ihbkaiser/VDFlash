from pathlib import Path
import sys

import torch
from torch import nn


EAGLE_ROOT = Path(__file__).resolve().parents[1] / "externals" / "MSD" / "EAGLE"
if str(EAGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EAGLE_ROOT))

from eagle.model.cnets import (  # noqa: E402
    _fuse_msd_video_hidden,
    _prepare_msd_video_embeddings,
)


def test_msd_video_path_preserves_noncontiguous_video_pad_features():
    embedding = nn.Embedding(151_657, 4)
    input_ids = torch.tensor([[10, 151_656, 11, 12, 151_656, 13]])
    visual_features = torch.full((1, 6, 4), -3.0)

    fused_embeddings, visual_mask = _prepare_msd_video_embeddings(
        input_ids,
        visual_features,
        embedding,
    )

    assert visual_mask.tolist() == [[False, True, False, False, True, False]]
    torch.testing.assert_close(fused_embeddings[visual_mask], visual_features[visual_mask])
    torch.testing.assert_close(
        fused_embeddings[~visual_mask],
        embedding(input_ids)[~visual_mask],
    )

    hidden_states = torch.ones((6, 4))
    video_mask = visual_mask[0]
    fused_hidden = _fuse_msd_video_hidden(
        nn.Linear(8, 4),
        fused_embeddings[0],
        hidden_states,
        video_mask,
    )
    torch.testing.assert_close(fused_hidden[video_mask], fused_embeddings[0][video_mask])


def test_msd_video_path_rejects_misaligned_visual_embeddings():
    embedding = nn.Embedding(151_657, 4)
    input_ids = torch.tensor([[10, 151_656, 11]])
    visual_features = torch.zeros((1, 2, 4))

    try:
        _prepare_msd_video_embeddings(input_ids, visual_features, embedding)
    except ValueError as exc:
        assert "must align" in str(exc)
    else:
        raise AssertionError("misaligned target video features should be rejected")

    try:
        _prepare_msd_video_embeddings(input_ids, None, embedding)
    except ValueError as exc:
        assert "required" in str(exc)
    else:
        raise AssertionError("video-pad IDs without target visual features should fail")
