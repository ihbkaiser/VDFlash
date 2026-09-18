from src.analyze.Validate_Sparrow_hypothesises.runtime import acceptance_round_record


def test_acceptance_round_record_maps_proposals_and_target_fallback_to_positions():
    record = acceptance_round_record(
        prompt_length=100,
        current_length=105,
        accept_length=2,
        round_index=3,
    )

    assert record == {
        "round_index": 3,
        "answer_position_start": 5,
        "accepted_proposal_count": 2,
        "accepted_proposal_positions": [5, 6],
        "fallback_position": 7,
        "absolute_position_start": 105,
        "emitted_token_count": 3,
    }
