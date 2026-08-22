from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infer.reject_position_audit import (
    JsonlLMReviewer,
    build_blind_prompt,
    build_informed_prompt,
    cross_validate_round,
    extract_rule_event,
    extract_rule_rounds,
    review_lm_response,
    summarize_rule_events,
    validate_round_fields,
    write_audit_artifacts,
)


def _sample() -> dict:
    return {
        "sample_id": "sample-1",
        "question": "Describe the video.",
        "reference": "A person walks outside.",
        "target_baseline": {"output_tokens": [1, 2, 3, 4, 5, 6, 7, 8]},
    }


def _checkpoint() -> dict:
    return {
        "label": "checkpoint-a",
        "num_output_tokens": 8,
        "acceptance": {
            "acceptance_rounds": [
                {
                    "proposal_count": 3,
                    "matched_proposals": 1,
                    "effective_emitted_tokens": 2,
                    "draft_proposal_text": " walks through outside",
                    "block_text": "A person",
                    "next_anchor_text": "walks",
                    "is_partial_block": False,
                    "is_terminal": False,
                },
                {
                    "proposal_count": 3,
                    "matched_proposals": 3,
                    "effective_emitted_tokens": 4,
                    "draft_proposal_text": " in the scene",
                    "block_text": "walks through outside",
                    "next_anchor_text": ".",
                    "is_partial_block": False,
                    "is_terminal": False,
                },
                {
                    "proposal_count": 1,
                    "matched_proposals": 0,
                    "effective_emitted_tokens": 1,
                    "draft_proposal_text": " extra",
                    "block_text": ".",
                    "next_anchor_text": "<|endoftext|>",
                    "is_partial_block": True,
                    "is_terminal": True,
                },
            ]
        },
    }


def test_extract_rule_event_uses_first_unmatched_proposal_and_absolute_position():
    event = extract_rule_event(_sample(), _checkpoint(), 0, previous_emitted=0)

    assert event["rule_reject"] is True
    assert event["rule_reject_index_in_block"] == 1
    assert event["rule_generated_position_0based"] == 2
    assert event["rule_relative_position"] == pytest.approx(2 / 7)
    assert event["rule_region"] == "early"
    assert event["is_partial_block"] is False


def test_extract_rule_event_does_not_call_full_acceptance_a_reject():
    event = extract_rule_event(_sample(), _checkpoint(), 1, previous_emitted=2)

    assert event["rule_reject"] is False
    assert event["rule_reject_index_in_block"] is None
    assert event["rule_generated_position_0based"] is None
    assert event["rule_region"] is None


def test_extract_rule_rounds_accumulates_previous_emitted_tokens(tmp_path: Path):
    payload = _sample()
    payload["checkpoints"] = [_checkpoint()]
    (tmp_path / "sample_000.json").write_text(json.dumps(payload), encoding="utf-8")

    rounds = extract_rule_rounds(tmp_path)

    assert len(rounds) == 3
    assert rounds[2]["previous_effective_emitted_tokens"] == 6
    assert rounds[2]["rule_generated_position_0based"] == 7
    assert rounds[2]["rule_region"] == "late"


def test_validate_round_fields_rejects_invalid_acceptance_prefix():
    errors = validate_round_fields(
        {
            "proposal_count": 2,
            "matched_proposals": 3,
            "effective_emitted_tokens": 1,
        }
    )

    assert any("matched_proposals" in error for error in errors)


def test_summarize_rule_events_reports_rates_and_regions():
    rounds = [
        {
            "checkpoint": "a",
            "sample_id": "s1",
            "rule_reject": True,
            "rule_region": "early",
            "rule_relative_position": 0.2,
            "rule_issues": [],
        },
        {
            "checkpoint": "a",
            "sample_id": "s1",
            "rule_reject": False,
            "rule_region": None,
            "rule_relative_position": None,
            "rule_issues": [],
        },
        {
            "checkpoint": "b",
            "sample_id": "s2",
            "rule_reject": True,
            "rule_region": "late",
            "rule_relative_position": 0.9,
            "rule_issues": [],
        },
    ]

    summary = summarize_rule_events(rounds)

    assert summary["total_rounds"] == 3
    assert summary["reject_rounds"] == 2
    assert summary["no_reject_rounds"] == 1
    assert summary["reject_rate"] == pytest.approx(2 / 3)
    assert summary["region_counts"] == {"early": 1, "middle": 0, "late": 1}
    assert summary["negative_control_rounds_available"] == 1


def test_prompts_keep_blind_review_free_of_rule_conclusion():
    record = {
        "sample_id": "s1",
        "checkpoint": "a",
        "proposal_count": 3,
        "draft_proposal_text": "a blue object",
        "block_text": "a",
        "next_anchor_text": "red",
        "rule_reject": True,
        "rule_reject_index_in_block": 1,
        "rule_region": "early",
    }

    blind = build_blind_prompt(record)
    informed = build_informed_prompt(record)

    assert "rule_reject" not in blind
    assert "matched_proposals" not in blind
    assert "rule_reject" in informed
    assert "rule_reject_index_in_block" in informed


def test_lm_response_validation_checks_position_and_region():
    valid = review_lm_response(
        {
            "reject_detected": True,
            "reject_position_in_block": 1,
            "position_region": "early",
            "confidence": "high",
        },
        {
            "proposal_count": 3,
            "rule_reject": True,
            "rule_region": "early",
        },
    )
    invalid = review_lm_response(
        {
            "reject_detected": False,
            "reject_position_in_block": 9,
            "position_region": "late",
        },
        {"proposal_count": 3, "rule_reject": False, "rule_region": None},
    )

    assert valid["valid"] is True
    assert invalid["valid"] is False
    assert invalid["issues"]


def test_cross_validate_marks_agreement_and_disagreement():
    rule = {
        "proposal_count": 3,
        "rule_reject": True,
        "rule_reject_index_in_block": 1,
        "rule_region": "early",
        "rule_issues": [],
    }
    blind = {
        "reject_detected": True,
        "reject_position_in_block": 1,
        "position_region": "early",
        "confidence": "high",
    }
    informed = {
        "reject_detected": True,
        "reject_position_in_block": 1,
        "position_region": "early",
        "confidence": "high",
        "rule_result_valid": True,
        "agreement": True,
        "position_consistent": True,
    }

    agreement = cross_validate_round(rule, blind, informed)
    disagreement = cross_validate_round(
        rule,
        {"reject_detected": False, "confidence": "low"},
        {
            "reject_detected": True,
            "reject_position_in_block": 1,
            "position_region": "early",
            "rule_result_valid": False,
            "agreement": False,
            "position_consistent": False,
        },
    )

    assert agreement["final_status"] == "validated"
    assert agreement["blind_reject_agreement"] is True
    assert disagreement["final_status"] == "disagreement"


def test_cross_validate_rejects_malformed_informed_lm_response():
    rule = {
        "proposal_count": 3,
        "rule_reject": True,
        "rule_reject_index_in_block": 1,
        "rule_region": "early",
        "rule_issues": [],
    }
    informed = {
        "reject_detected": "yes",
        "reject_position_in_block": 1,
        "position_region": "early",
        "rule_result_valid": True,
        "agreement": True,
        "position_consistent": True,
    }

    result = cross_validate_round(rule, None, informed)

    assert result["final_status"] == "lm_invalid"


def test_write_audit_artifacts_creates_machine_readable_outputs(tmp_path: Path):
    paths = write_audit_artifacts(
        [
            {
                "sample_id": "s1",
                "checkpoint": "a",
                "round_index": 0,
                "rule_reject": True,
                "rule_region": "early",
                "rule_relative_position": 0.2,
                "rule_issues": [],
            }
        ],
        tmp_path,
        metadata={"source": "test"},
    )

    assert set(paths) == {"rounds", "reject_events", "summary", "report"}
    assert all(path.exists() for path in paths.values())
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["total_rounds"] == 1


def test_jsonl_reviewer_replays_phase_specific_response(tmp_path: Path):
    path = tmp_path / "lm.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "checkpoint": "a",
                "round_index": 2,
                "phase": "blind",
                "response": {"reject_detected": True, "reject_position_in_block": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reviewer = JsonlLMReviewer(path)

    response = reviewer.review(
        {"sample_id": "s1", "checkpoint": "a", "round_index": 2},
        informed=False,
    )

    assert response["reject_detected"] is True
    assert reviewer.review(
        {"sample_id": "s1", "checkpoint": "a", "round_index": 2},
        informed=True,
    )["missing_replay"] is True
