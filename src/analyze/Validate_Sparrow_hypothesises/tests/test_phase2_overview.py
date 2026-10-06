from pathlib import Path

from scripts import plot_phase2_overview
from scripts.plot_phase2_overview import (
    acceptance_length_series,
    plot_acceptance_length_lines,
)


def _summary_rows():
    rows = []
    for retention, mean in (("100.0", 1.2), ("50.0", 1.5), ("25.0", 1.8), ("0.0", 3.6)):
        rows.append(
            {
                "metric": "accepted_prefix_tokens",
                "retention_percentage": retention,
                "mean": str(mean),
                "std": "0.1",
                "n": "2",
            }
        )
    return rows


def test_acceptance_length_series_adds_bonus_token_and_preserves_retention_order():
    series = acceptance_length_series(
        _summary_rows(),
        [{"retention_percentage": value} for value in ("100.0", "50.0", "25.0", "0.0")],
    )

    assert series["accepted_prefix_tokens"][0][0] == 1.2
    assert series["accepted_effective_tokens"][0][0] == 2.2
    assert [item[0] for item in series["accepted_prefix_tokens"]] == [1.2, 1.5, 1.8, 3.6]


def test_plot_acceptance_length_lines_writes_a_nonempty_png(tmp_path: Path):
    output = tmp_path / "acceptance_length_lines.png"
    plot_acceptance_length_lines(
        _summary_rows(),
        [{"retention_percentage": value} for value in ("100.0", "50.0", "25.0", "0.0")],
        ["100", "50", "25", "0"],
        output,
        title="test",
        xlabel="retention",
    )

    assert output.is_file()
    assert output.stat().st_size > 0


def test_h21_output_progress_acc_length_is_sample_balanced_and_excludes_invalid_rounds():
    summarize = getattr(plot_phase2_overview, "summarize_h21_output_progress", None)
    assert callable(summarize), "H2.1 output-prefix summary should be available"

    metric_rows = [
        {
            "row_id": "sample-a:natural:full",
            "training_corpus": "llava68k",
            "prompt_variant": "natural",
            "visual_condition": "full",
            "target_output_tokens": "8",
            "tau_proposal": "2",
        },
        {
            "row_id": "sample-b:natural:full",
            "training_corpus": "llava68k",
            "prompt_variant": "natural",
            "visual_condition": "full",
            "target_output_tokens": "4",
            "tau_proposal": "1",
        },
    ]
    round_rows = [
        {"row_id": "sample-a:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 1, "matched_proposals": 0},
        {"row_id": "sample-a:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 2, "matched_proposals": 2},
        {"row_id": "sample-a:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 5, "matched_proposals": 4},
        {"row_id": "sample-a:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 7, "matched_proposals": 5, "is_terminal": True},
        {"row_id": "sample-a:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 8, "matched_proposals": 3, "is_partial_block": True},
        {"row_id": "sample-b:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 1, "matched_proposals": 2},
        {"row_id": "sample-b:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 2, "matched_proposals": 0},
        {"row_id": "sample-b:natural:full", "training_corpus": "llava68k", "prompt_variant": "natural", "visual_condition": "full", "answer_position_start": 3, "matched_proposals": 1},
    ]

    rows = summarize(metric_rows, round_rows)
    observed = [(row["progress_pct"], row["mean_acc_length"], row["n_valid_samples"]) for row in rows]

    assert observed == [(25, 0.0, 1), (50, 1.5, 2), (75, 1.5, 2), (100, 1.5, 2)]
