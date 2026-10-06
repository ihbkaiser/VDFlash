from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.raw_trace import (
    enrich_position_trace,
    enrich_round_trace,
)


def test_enrich_position_trace_preserves_acceptance_and_text_alignment():
    trace = [{
        "position": 0,
        "proposed": 1,
        "accepted": 1,
        "rate": 1.0,
    }, {
        "position": 1,
        "proposed": 1,
        "accepted": 0,
        "rate": 0.0,
    }]

    result = enrich_position_trace(trace, [10, 11], [10, 12], lambda value: f"t{value}")

    assert result == [
        {
            "position": 0,
            "proposed": 1,
            "accepted": 1,
            "rate": 1.0,
            "target_token_id": 10,
            "target_token_text": "t10",
            "speculative_token_id": 10,
            "speculative_token_text": "t10",
        },
        {
            "position": 1,
            "proposed": 1,
            "accepted": 0,
            "rate": 0.0,
            "target_token_id": 11,
            "target_token_text": "t11",
            "speculative_token_id": 12,
            "speculative_token_text": "t12",
        },
    ]


def test_enrich_round_trace_expands_msd_rounds_to_position_text_records():
    trace = [{
        "round_index": 0,
        "answer_position_start": 0,
        "accepted_proposal_count": 1,
        "accepted_proposal_positions": [0],
        "fallback_position": 1,
        "emitted_token_count": 2,
    }]

    result = enrich_round_trace(trace, [10, 11], [10, 12], lambda value: f"t{value}")

    assert result["positions"] == [
        {
            "position": 0,
            "round_index": 0,
            "role": "accepted_proposal",
            "proposed": 1,
            "accepted": 1,
            "target_token_id": 10,
            "target_token_text": "t10",
            "speculative_token_id": 10,
            "speculative_token_text": "t10",
        },
        {
            "position": 1,
            "round_index": 0,
            "role": "fallback",
            "proposed": 0,
            "accepted": 0,
            "target_token_id": 11,
            "target_token_text": "t11",
            "speculative_token_id": 12,
            "speculative_token_text": "t12",
        },
    ]
    assert result["rounds"][0]["position_records"] == result["positions"]
