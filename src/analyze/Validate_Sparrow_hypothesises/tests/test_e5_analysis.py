from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.e5_analysis import (
    pair_acceptance_rows,
    pair_attention_rows,
    summarize_paired,
)


def _attention_row(mode: str, visual_mass: float, visual_density: float) -> dict:
    return {
        "sample_id": "sample-1",
        "modality": "summary",
        "visual_condition": "full",
        "attention_policy": "last_instruction",
        "visual_value_mode": mode,
        "visual_mass": visual_mass,
        "visual_attention_density": visual_density,
        "visual_density_ratio": visual_density * 10,
        "visual_effective_key_count": 100.0,
    }


def test_pair_attention_rows_matches_real_and_zero_by_sample_condition_policy():
    pairs = pair_attention_rows([
        _attention_row("real", 0.8, 0.01),
        _attention_row("zero", 0.8, 0.01),
    ])

    assert len(pairs) == 1
    assert pairs[0]["attention_visual_mass_real_minus_zero"] == 0.0
    assert pairs[0]["attention_visual_density_real_minus_zero"] == 0.0


def test_pair_acceptance_rows_preserves_real_zero_difference():
    real = [{
        "sample_id": "sample-1",
        "condition": "retention",
        "retention_percentage": 25.0,
        "accepted_prefix_tokens": 2.0,
        "lossless": True,
    }]
    zero = [{**real[0], "accepted_prefix_tokens": 3.0, "lossless": False}]

    pairs = pair_acceptance_rows(real, zero)

    assert len(pairs) == 1
    assert pairs[0]["accepted_prefix_tokens_real_minus_zero"] == -1.0
    assert pairs[0]["lossless_real"] is True
    assert pairs[0]["lossless_zero"] is False


def test_summarize_paired_reports_sample_count_and_bootstrap_fields():
    rows = [
        {"condition": "full", "delta": -1.0},
        {"condition": "full", "delta": 1.0},
    ]

    summary = summarize_paired(rows, group_fields=("condition",), value_field="delta")

    assert summary["full"]["n"] == 2
    assert summary["full"]["mean"] == 0.0
    assert "ci95_low" in summary["full"]
    assert "ci95_high" in summary["full"]
