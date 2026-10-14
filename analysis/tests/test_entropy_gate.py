import pytest

from analysis.entropy_gate import (
    add_derived_metrics,
    compute_signal_comparison,
    compute_decile_summary,
    simulate_chain_gates,
)


def _row(sample_id, task, t, j, cut_entropy, necessity, full_match, cut_match):
    return {
        "sample_id": sample_id,
        "task": task,
        "t": t,
        "j": j,
        "draft_entropy_cut": cut_entropy,
        "draft_prob_cut": 0.95,
        "draft_margin_cut": 0.8,
        "draft_attention_mass_full": 2.0,
        "necessity": necessity,
        "argmax_match_full": full_match,
        "argmax_match_cut": cut_match,
        "target_tv_full": 0.1,
        "target_tv_cut": 0.3,
    }


def test_derived_metrics_and_decile_summary_mark_target_defined_visual_heavy_rows():
    rows = [
        _row("s0", "chartqa", 0, 1, 0.1, 0.4, 1, 0),
        _row("s0", "chartqa", 0, 2, 0.2, 0.4, 1, 0),
        _row("s1", "textvqa_val", 0, 1, 1.1, 0.05, 1, 1),
        _row("s1", "textvqa_val", 0, 2, 1.2, 0.05, 1, 1),
    ]

    add_derived_metrics(rows)
    assert rows[0]["gamma_greedy"] == 1
    assert rows[0]["gamma_tv"] == pytest.approx(0.2)
    assert rows[0]["visual_heavy_target_match"] is True
    assert rows[2]["visual_heavy_target_match"] is False
    assert rows[0]["necessity_heavy_03"] is True
    assert rows[0]["cut_confident_wrong"] is True

    summary = compute_decile_summary(rows, threshold=0.3)
    assert len(summary) == 10
    assert sum(item["count"] for item in summary) == len(rows)
    assert any(item["visual_heavy_pct"] > 0 for item in summary if item["count"])


def test_chain_gate_simulation_reports_full_fraction_and_length():
    rows = [
        _row("s0", "chartqa", 0, 1, 0.1, 0.4, 1, 0),
        _row("s0", "chartqa", 0, 2, 1.1, 0.4, 1, 0),
        _row("s0", "chartqa", 0, 3, 0.1, 0.4, 0, 0),
    ]
    add_derived_metrics(rows)

    results = simulate_chain_gates(rows, thresholds=[0.5])
    by_policy = {item["policy"]: item for item in results}
    assert by_policy["always_full"]["length"] == 2
    assert by_policy["always_cut"]["length"] == 0
    assert by_policy["entropy_gate"]["full_fraction"] == 1 / 3
    assert by_policy["oracle"]["length"] == 2


def test_signal_comparison_uses_target_defined_visual_heaviness():
    rows = [
        _row("s0", "chartqa", 0, 1, 0.1, 0.4, 1, 0),
        _row("s0", "chartqa", 0, 2, 0.2, 0.4, 1, 0),
        _row("s1", "chartqa", 0, 1, 1.1, 0.05, 1, 1),
        _row("s1", "chartqa", 0, 2, 1.2, 0.05, 1, 1),
    ]
    add_derived_metrics(rows)

    result = compute_signal_comparison(rows)
    assert {item["signal"] for item in result} == {
        "one_minus_cut_prob",
        "cut_top2_prob",
        "cut_entropy",
        "inverse_cut_margin",
        "full_visual_attention",
    }
    assert all(item["positive_count"] == 2 for item in result)
