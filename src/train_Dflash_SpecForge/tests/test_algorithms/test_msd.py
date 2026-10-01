from __future__ import annotations

import pytest
import torch

from specforge.algorithms.msd.curriculum import (
    choose_visual_sample,
    visual_ratio_for_epoch,
)
from specforge.algorithms.msd.data import normalize_offline_sample
from specforge.algorithms.msd.model import (
    add_reference_uniform_noise,
    decouple_msd_inputs,
    msd_loss,
)


@pytest.mark.parametrize(
    ("epoch_now", "expected"),
    [
        (1, 0.0),
        (20, 0.0),
        (21, 0.05),
        (22, 0.10),
        (39, 0.95),
        (40, 1.0),
    ],
)
def test_msd_40_epoch_ratio_matches_reference_boundaries(
    epoch_now: int,
    expected: float,
) -> None:
    assert visual_ratio_for_epoch(epoch_now, 40) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("epoch_now", "total_epoch"),
    [(0, 40), (41, 40), (1, 0), (1, 39)],
)
def test_msd_ratio_rejects_invalid_or_noncanonical_epochs(
    epoch_now: int,
    total_epoch: int,
) -> None:
    with pytest.raises(ValueError):
        visual_ratio_for_epoch(epoch_now, total_epoch)


def test_msd_choice_is_stateless_and_resume_stable() -> None:
    first = [choose_visual_sample(0, 27, index, 40) for index in range(100)]
    resumed = [
        choose_visual_sample(0, 27, index, 40) for index in range(50, 100)
    ]

    assert first[50:] == resumed
    assert any(first)
    assert not all(first)


def test_msd_choice_obeys_pure_text_and_pure_visual_epochs() -> None:
    assert not any(choose_visual_sample(0, 20, index, 40) for index in range(64))
    assert all(choose_visual_sample(0, 40, index, 40) for index in range(64))


def test_msd_normalizer_reproduces_next_token_shift() -> None:
    raw = {
        "input_ids": torch.tensor([10, 11, 12, 13]),
        "loss_mask": torch.tensor([0, 1, 1, 1]),
        "target_hidden_state": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        ),
        "input_embeddings": torch.tensor(
            [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]]
        ),
        "visual_token_mask": torch.tensor([False, True, True, False]),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 4, 5, 3],
                [0, 6, 7, 3],
            ]
        ),
    }

    normalized = normalize_offline_sample(raw, max_len=4)

    assert normalized["input_ids"].tolist() == [[11, 12, 13, 0]]
    assert normalized["loss_mask"].tolist() == [[0, 1, 1, 0]]
    assert normalized["target_hidden_state"].tolist() == [
        [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0]]
    ]
    assert normalized["next_token_embeddings"].tolist() == [
        [[30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [0.0, 0.0]]
    ]
    assert torch.equal(
        normalized["conditioning_hidden_state"], raw["target_hidden_state"]
    )
    assert torch.equal(normalized["visual_embeddings"], raw["input_embeddings"])
    assert normalized["visual_token_mask"].tolist() == [[False, True, True, False]]
    assert tuple(normalized["position_ids"].shape) == (3, 1, 4)


def test_msd_normalizer_rejects_mismatched_sequence_lengths() -> None:
    raw = {
        "input_ids": torch.tensor([1, 2, 3]),
        "loss_mask": torch.ones(3),
        "target_hidden_state": torch.ones(1, 3, 2),
        "input_embeddings": torch.ones(1, 2, 2),
        "visual_token_mask": torch.zeros(3, dtype=torch.bool),
        "position_ids": torch.arange(3).repeat(3, 1),
    }

    with pytest.raises(ValueError, match="sequence lengths"):
        normalize_offline_sample(raw, max_len=3)


def test_decouple_uses_projection_for_text_and_raw_embedding_for_visual() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    next_embeddings = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
    )
    visual_embeddings = torch.tensor([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    visual_mask = torch.tensor([[False, True, False]])
    projection = torch.nn.Linear(4, 2, bias=False)
    projection.weight.data.copy_(
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    )

    actual = decouple_msd_inputs(
        hidden,
        next_embeddings,
        visual_embeddings,
        visual_mask,
        projection,
    )

    assert torch.equal(actual[:, 0], torch.tensor([[11.0, 22.0]]))
    assert torch.equal(actual[:, 1], visual_embeddings[:, 1])
    assert torch.equal(actual[:, 2], torch.tensor([[55.0, 66.0]]))


def test_reference_uniform_noise_is_seeded_and_scaled_by_sequence_length() -> None:
    hidden = torch.zeros(1, 4, 2)
    first = add_reference_uniform_noise(
        hidden,
        width=0.2,
        generator=torch.Generator().manual_seed(7),
    )
    second = add_reference_uniform_noise(
        hidden,
        width=0.2,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(first, second)
    assert first.abs().max().item() <= 12.8 + 1e-5


def test_msd_loss_matches_literal_reference_and_masks_padding() -> None:
    predicted = torch.tensor([[[1.0, 2.0], [2.0, 3.0]]], requires_grad=True)
    target = torch.tensor([[[1.5, 1.5], [99.0, 99.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    head = torch.nn.Linear(2, 3, bias=False)
    head.weight.data.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    )
    for parameter in head.parameters():
        parameter.requires_grad = False

    result = msd_loss(predicted, target, mask, head, 1.0, 0.1)

    expected_feature = torch.nn.functional.smooth_l1_loss(
        predicted[:, :1], target[:, :1], reduction="mean"
    )
    target_probs = head(target[:, :1]).softmax(-1)
    expected_soft = -(
        target_probs * head(predicted[:, :1]).log_softmax(-1)
    ).sum(-1).mean()
    assert torch.allclose(result.feature_loss, expected_feature)
    assert torch.allclose(result.soft_target_loss, expected_soft)
    assert torch.allclose(result.loss, expected_feature + 0.1 * expected_soft)

    result.loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert head.weight.grad is None


def test_msd_loss_rejects_an_empty_loss_mask() -> None:
    head = torch.nn.Linear(2, 3, bias=False)
    with pytest.raises(ValueError, match="at least one valid token"):
        msd_loss(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2),
            head,
        )
