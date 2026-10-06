"""Tests for raw metric collection from completed DFlash journals."""

from __future__ import annotations

from pathlib import Path

from src.analyze.Validate_Sparrow_hypothesises.analyze_run_metrics import (
    acceptance_round_records,
    flatten_row,
    group_metrics,
    metric_coverage,
    output_detail_records,
    output_position_records,
    position_metrics,
)


def _row() -> dict:
    return {
        "status": "ok",
        "sample_id": "sample-1",
        "prompt_variant": "natural",
        "visual_condition": "full",
        "dataset": "vdc",
        "training_corpus": "llava68k",
        "draft_depth": 3,
        "actual_visual_tokens": 3128,
        "draft_context_visual_tokens": 3128,
        "target_visual_tokens": 3128,
        "accepted_prefix_tokens": 2.0,
        "accepted_effective_tokens": 3.0,
        "metrics": {
            "tau": 3.0,
            "tau_proposal": 2.0,
            "tau_effective": 3.0,
            "speculative_output_tokens": 64,
            "target_output_tokens": 64,
        },
        "timing": {
            "speculative": {
                "prefill_s": 1.0,
                "draft_s": 0.2,
                "verify_s": 0.3,
                "decode_s": 0.4,
                "end_to_end_s": 1.9,
                "tokens_per_second": 33.7,
            },
            "target": {
                "prefill_s": 1.2,
                "decode_s": 0.8,
                "end_to_end_s": 2.0,
                "tokens_per_second": 32.0,
            },
        },
        "speedup": {"end_to_end": 1.05},
        "lossless": True,
        "lossless_prefix_length": 64,
        "target_output_hash": "same",
        "speculative_output_hash": "same",
        "acceptance_by_position": [
            {"position": 0, "proposed": 0, "accepted": 0, "rate": None},
            {"position": 1, "proposed": 2, "accepted": 1, "rate": 0.5},
            {"position": 2, "proposed": 3, "accepted": 2, "rate": 2 / 3},
        ],
        "acceptance_rounds": [
            {"proposal_count": 5, "matched_proposals": 3},
        ],
    }


def test_flatten_row_collects_existing_acceptance_timing_and_output_fields():
    flat = flatten_row(_row(), source=Path("journal.jsonl"), row_index=0)

    assert flat["source"] == "journal.jsonl"
    assert flat["accepted_prefix_tokens"] == 2.0
    assert flat["accepted_effective_tokens"] == 3.0
    assert flat["tau_proposal"] == 2.0
    assert flat["tau_effective"] == 3.0
    assert flat["acceptance_rate"] == 3 / 5
    assert flat["speculative_prefill_s"] == 1.0
    assert flat["speculative_end_to_end_s"] == 1.9
    assert flat["target_end_to_end_s"] == 2.0
    assert flat["speedup_end_to_end"] == 1.05
    assert flat["speculative_output_tokens"] == 64
    assert flat["target_output_tokens"] == 64
    assert flat["lossless"] == 1.0
    assert flat["lossless_prefix_length"] == 64
    assert flat["target_output_hash"] == "same"
    assert flat["speculative_output_hash"] == "same"
    assert flat["acceptance_by_position"] == _row()["acceptance_by_position"]


def test_acceptance_rate_uses_full_round_proposals_not_truncated_positions():
    row = _row()
    row["acceptance_by_position"] = [
        {"position": 0, "proposed": 1, "accepted": 1, "rate": 1.0},
    ]
    row["acceptance_rounds"] = [
        {"proposal_count": 15, "matched_proposals": 2},
    ]

    flat = flatten_row(row, source=Path("journal.jsonl"), row_index=0)

    assert flat["acceptance_rate"] == 2 / 15


def test_group_metrics_summarizes_raw_metrics_without_hypothesis_contrasts():
    rows = [flatten_row(_row(), source=Path("journal.jsonl"), row_index=0)]
    summary = group_metrics(rows, group_fields=("prompt_variant", "visual_condition"))

    names = {row["metric"] for row in summary}
    assert "accepted_prefix_tokens" in names
    assert "acceptance_rate" in names
    assert "speculative_prefill_s" in names
    assert "lossless" in names
    assert not any("interaction" in name or "delta" in name for name in names)


def test_position_metrics_preserves_raw_position_acceptance_by_group():
    positions = position_metrics([_row()], group_fields=("visual_condition",))

    assert positions == [
        {
            "visual_condition": "full",
            "sample_id": "sample-1",
            "position": 1,
            "proposed": 2,
            "accepted": 1,
            "rate": 0.5,
        },
        {
            "visual_condition": "full",
            "sample_id": "sample-1",
            "position": 2,
            "proposed": 3,
            "accepted": 2,
            "rate": 2 / 3,
        },
    ]


def test_metric_coverage_marks_missing_runtime_fields_without_imputing_them():
    rows = [
        flatten_row(_row(), source=Path("journal.jsonl"), row_index=0),
        {"accepted_prefix_tokens": None, "acceptance_rate": None},
    ]

    coverage = {row["metric"]: row for row in metric_coverage(rows)}

    assert coverage["accepted_prefix_tokens"] == {
        "metric": "accepted_prefix_tokens",
        "total": 2,
        "present": 1,
        "missing": 1,
        "coverage": 0.5,
    }
    assert coverage["acceptance_rate"]["missing"] == 1


class _FakeTokenizer:
    def decode(self, ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(f"<{int(token)}>" for token in ids)


def test_output_detail_records_keep_per_position_acceptance_ids_and_text():
    row = _row()
    row.update(
        {
            "target_output_ids": [10, 11, 12],
            "speculative_output_ids": [10, 99, 12],
            "acceptance_rounds": [
                {
                    "answer_position_start": 0,
                    "accepted_proposal_positions": [0],
                    "proposal_positions": [0, 1],
                    "block_token_ids": [10, 11],
                    "draft_proposal_token_ids": [10, 99],
                    "effective_emitted_tokens": 2,
                    "matched_proposals": 1,
                }
            ],
        }
    )
    flat = flatten_row(row, source=Path("journal.jsonl"), row_index=0)
    tokenizer = _FakeTokenizer()

    positions = output_position_records([flat], tokenizer=tokenizer)
    details = output_detail_records([flat], tokenizer=tokenizer)
    rounds = acceptance_round_records([flat], tokenizer=tokenizer)

    assert positions[1]["position"] == 1
    assert positions[1]["accepted_tokens_at_position"] == 1
    assert positions[1]["target_token_id"] == 11
    assert positions[1]["target_token_text"] == "<11>"
    assert positions[1]["speculative_token_text"] == "<99>"
    assert details[0]["target_output_text"] == "<10><11><12>"
    assert details[0]["speculative_output_text"] == "<10><99><12>"
    assert rounds[0]["block_text"] == "<10><11>"
    assert rounds[0]["draft_proposal_text"] == "<10><99>"


def test_flatten_row_normalizes_flat_msd_runtime_fields():
    row = {
        "sample_id": "msd-1",
        "condition": "retention",
        "retention_percentage": 50.0,
        "target_visual_tokens": 25000,
        "actual_visual_tokens": 12500,
        "target_output_ids": [10, 11],
        "speculative_output_ids": [10, 11],
        "target_output_hash": "same",
        "speculative_output_hash": "same",
        "lossless": True,
        "lossless_prefix_length": 2,
        "accepted_prefix_tokens": 1.5,
        "prefill_seconds": 2.0,
        "decode_seconds": 3.0,
        "end_to_end_seconds": 5.0,
        "ar_prefill_seconds": 2.5,
        "ar_decode_seconds": 4.0,
        "ar_end_to_end_seconds": 6.5,
        "end_to_end_speedup": 1.3,
        "acceptance_by_position": [
            {"position": 0, "proposed": 1, "accepted": 1, "rate": 1.0},
        ],
    }

    flat = flatten_row(row, source=Path("msd.jsonl"), row_index=0)

    assert flat["retention_percentage"] == 50.0
    assert flat["accepted_prefix_tokens"] == 1.5
    assert flat["acceptance_rate"] == 1.0
    assert flat["speculative_prefill_s"] == 2.0
    assert flat["speculative_end_to_end_s"] == 5.0
    assert flat["target_prefill_s"] == 2.5
    assert flat["target_end_to_end_s"] == 6.5
    assert flat["speedup_end_to_end"] == 1.3
    assert flat["speculative_output_tokens"] == 2
    assert flat["target_output_tokens"] == 2


def test_flatten_row_exposes_mcq2_answer_choice_format_and_token_cap_metrics():
    row = _row()
    row.update(
        {
            "prompt_variant": "mcq2",
            "correct_option": "B",
            "max_new_tokens": 128,
            "target_output_text": "(B) A person walks across the room carrying a box.",
            "speculative_output_text": "(B) A person walks across the room carrying a box.",
        }
    )
    row["metrics"].update(
        {"target_output_tokens": 11, "speculative_output_tokens": 11}
    )

    flat = flatten_row(row, source=Path("journal.jsonl"), row_index=0)

    assert flat["target_answer_word_count"] == 9
    assert flat["speculative_answer_word_count"] == 9
    assert flat["target_choice_correct"] == 1.0
    assert flat["speculative_choice_correct"] == 1.0
    assert flat["target_expected_format_compliant"] == 1.0
    assert flat["speculative_expected_format_compliant"] == 1.0
    assert flat["max_new_tokens"] == 128
    assert flat["target_reached_output_cap"] == 0.0
    assert flat["speculative_reached_output_cap"] == 0.0


def test_h21_analyzer_writes_response_metric_plot(tmp_path):
    import json

    from src.analyze.Validate_Sparrow_hypothesises.analyze_run_metrics import analyze

    raw_path = tmp_path / "h21.jsonl"
    row = _row()
    row.update(
        {
            "prompt_variant": "mcq2",
            "correct_option": "B",
            "max_new_tokens": 128,
            "target_output_text": "(B) A person walks across the room carrying a box.",
            "speculative_output_text": "(B) A person walks across the room carrying a box.",
        }
    )
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    manifest = analyze("h21", [raw_path], tmp_path / "summary")

    assert "response_metrics.png" in manifest["plots"]
    assert (tmp_path / "summary" / "response_metrics.png").is_file()
