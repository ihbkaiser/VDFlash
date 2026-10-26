import pytest

from scripts.run_msd_visual_verification_7b import (
    _aggregate_repeated_arm_runs,
    _validate_acceptance_trace,
)


def test_acceptance_trace_accounts_for_root_and_each_accepted_proposal_once():
    trace = {
        "acceptance_by_position": [
            {
                "answer_position_start": 0,
                "accepted_proposal_count": 2,
                "accepted_proposal_positions": [1, 2],
                "fallback_position": 0,
                "emitted_token_count": 3,
            },
            {
                "answer_position_start": 3,
                "accepted_proposal_count": 1,
                "accepted_proposal_positions": [4],
                "fallback_position": 3,
                "emitted_token_count": 2,
            },
        ]
    }

    _validate_acceptance_trace(trace, output_length=5)


def test_acceptance_trace_rejects_overlapping_round_positions():
    trace = {
        "acceptance_by_position": [
            {
                "answer_position_start": 0,
                "accepted_proposal_count": 1,
                "accepted_proposal_positions": [0],
                "fallback_position": 1,
                "emitted_token_count": 2,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="fallback"):
        _validate_acceptance_trace(trace, output_length=2)


def test_repeated_arm_runs_report_median_latency_and_require_stable_output():
    base = {
        "speculative_output_ids": [4, 5],
        "acceptance_by_position": [{"round_index": 0}],
        "end_to_end_seconds": 1.0,
        "decode_seconds": 0.8,
        "prefill_seconds": 0.2,
    }
    runs = [
        dict(base),
        {**base, "end_to_end_seconds": 3.0, "decode_seconds": 2.8},
        {**base, "end_to_end_seconds": 2.0, "decode_seconds": 1.8},
    ]

    result = _aggregate_repeated_arm_runs(runs)

    assert result["end_to_end_seconds"] == 2.0
    assert result["end_to_end_seconds_by_repeat"] == [1.0, 3.0, 2.0]
    assert result["timing_repeats"] == 3
    with pytest.raises(RuntimeError, match="not deterministic"):
        _aggregate_repeated_arm_runs([base, {**base, "speculative_output_ids": [4, 6]}])
