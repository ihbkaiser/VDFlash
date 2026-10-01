from __future__ import annotations

import pytest

from specforge.algorithms.msd.curriculum import (
    choose_visual_sample,
    visual_ratio_for_epoch,
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
