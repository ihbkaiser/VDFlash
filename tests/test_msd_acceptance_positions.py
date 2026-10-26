import torch

from src.analyze.Validate_Sparrow_hypothesises.runtime import (
    acceptance_round_record,
    cap_msd_acceptance_to_remaining,
    use_msd_author_mode,
)


def test_round_positions_count_root_as_target_fallback_not_draft_acceptance():
    first = acceptance_round_record(
        prompt_length=10,
        current_length=10,
        accept_length=2,
        round_index=0,
    )
    second = acceptance_round_record(
        prompt_length=10,
        current_length=13,
        accept_length=1,
        round_index=1,
    )

    assert first["accepted_proposal_positions"] == [1, 2]
    assert first["fallback_position"] == 0
    assert first["emitted_token_count"] == 3
    assert second["accepted_proposal_positions"] == [4]
    assert second["fallback_position"] == 3
    assert second["emitted_token_count"] == 2


def test_final_round_clips_accepted_proposals_to_exact_remaining_output_cap():
    logits = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    sample_p = torch.zeros(7)

    accepted, next_sample, stop = cap_msd_acceptance_to_remaining(
        logits,
        best_candidate=1,
        accept_length=4,
        sample_p=sample_p,
        remaining_tokens=2,
    )

    assert accepted == 1
    assert stop is True
    torch.testing.assert_close(next_sample, logits[1, 1])


def test_author_msd_mode_is_enabled_and_restored_around_inference():
    from pathlib import Path
    import sys

    eagle_root = Path(__file__).resolve().parents[1] / "externals" / "MSD" / "EAGLE"
    if str(eagle_root) not in sys.path:
        sys.path.insert(0, str(eagle_root))
    from eagle.model.utils import temp_cache

    original = temp_cache.use_msd
    temp_cache.use_msd = False
    try:
        with use_msd_author_mode():
            assert temp_cache.use_msd is True
        assert temp_cache.use_msd is False
    finally:
        temp_cache.use_msd = original
