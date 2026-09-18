import json

import pytest

import src.analyze.plot_hypothesis_report as report_plots
from src.analyze.plot_hypothesis_report import (
    load_jsonl_rows,
    load_position_csv,
    summarize_h1b_attention,
    summarize_h2_acceptance,
    summarize_pilot_attention,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_load_jsonl_rows_adds_dataset_label_and_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write_jsonl(path, [{"sample_id": "a", "value": 1}, {"sample_id": "b", "value": 2}])
    path.write_text(path.read_text() + "\n")

    rows = load_jsonl_rows([path], dataset_label="VDC50")

    assert rows == [
        {"sample_id": "a", "value": 1, "dataset": "VDC50"},
        {"sample_id": "b", "value": 2, "dataset": "VDC50"},
    ]


def test_summarize_h1b_attention_uses_real_value_rows_and_groups_correctly():
    rows = [
        {
            "dataset": "VDC50",
            "visual_condition": "full",
            "visual_value_mode": "real",
            "attention_policy": "last_instruction",
            "visual_mass": 0.4,
            "text_mass": 0.5,
            "instruction_mass": 0.1,
        },
        {
            "dataset": "VDC50",
            "visual_condition": "full",
            "visual_value_mode": "real",
            "attention_policy": "last_instruction",
            "visual_mass": 0.6,
            "text_mass": 0.3,
            "instruction_mass": 0.1,
        },
        {
            "dataset": "VDC50",
            "visual_condition": "full",
            "visual_value_mode": "zero",
            "attention_policy": "last_instruction",
            "visual_mass": 0.99,
            "text_mass": 0.01,
            "instruction_mass": 0.0,
        },
    ]

    summary = summarize_h1b_attention(rows)

    assert summary[("VDC50", "last_instruction", "full")] == {
        "n": 2,
        "visual_mass": pytest.approx(0.5),
        "text_mass": pytest.approx(0.4),
        "instruction_mass": pytest.approx(0.1),
    }


def test_summarize_h2_acceptance_groups_prompt_and_retention():
    rows = [
        {"prompt_variant": "natural", "condition": "full", "accepted_prefix_tokens": 1.0},
        {"prompt_variant": "natural", "condition": "full", "accepted_prefix_tokens": 3.0},
        {"prompt_variant": "answer_hint", "condition": "deleted", "accepted_prefix_tokens": 4.0},
    ]

    summary = summarize_h2_acceptance(rows)

    assert summary[("natural", "full")] == {"n": 2, "mean": pytest.approx(2.0)}
    assert summary[("answer_hint", "deleted")] == {"n": 1, "mean": pytest.approx(4.0)}


def test_summarize_pilot_attention_preserves_value_mode_and_groups():
    rows = [
        {
            "visual_value_mode": "real",
            "attention_policy": "last_instruction",
            "visual_condition": "full",
            "modality": "summary",
            "visual_mass": 0.4,
            "text_mass": 0.5,
            "instruction_mass": 0.1,
        },
        {
            "visual_value_mode": "zero",
            "attention_policy": "last_instruction",
            "visual_condition": "full",
            "modality": "summary",
            "visual_mass": 0.4,
            "text_mass": 0.5,
            "instruction_mass": 0.1,
        },
        {
            "visual_value_mode": "real",
            "attention_policy": "last_instruction",
            "visual_condition": "full",
            "modality": "visual",
            "visual_mass": 1.0,
            "text_mass": 0.0,
            "instruction_mass": 0.0,
        },
    ]

    summary = summarize_pilot_attention(rows)

    assert summary[("real", "last_instruction", "full")] == {
        "n": 1,
        "visual_mass": pytest.approx(0.4),
        "text_mass": pytest.approx(0.5),
        "instruction_mass": pytest.approx(0.1),
    }
    assert summary[("zero", "last_instruction", "full")]["n"] == 1


def test_load_position_csv_reads_machine_readable_position_source(tmp_path):
    path = tmp_path / "position.csv"
    path.write_text(
        "dataset,condition,bin_0_10,bin_10_25,bin_25_50,bin_50_75,bin_75_100\n"
        "VDC50,Full,0.1,0.2,0.3,0.4,0.5\n"
    )

    result = load_position_csv(path)

    assert result == {
        "VDC50": {"Full": [0.1, 0.2, 0.3, 0.4, 0.5]}
    }


def test_load_h31_dflash_statistics_preserves_condition_order_and_ci(tmp_path):
    path = tmp_path / "h31.csv"
    path.write_text(
        "bin,bin_end_fraction,bin_start_fraction,ci_high,ci_low,mean,n,visual_condition\n"
        "1,0.25,0.125,0.20,0.10,0.15,50,zero\n"
        "0,0.125,0.0,0.30,0.20,0.25,50,zero\n"
        "0,0.125,0.0,0.31,0.21,0.26,50,full\n"
    )

    loader = getattr(report_plots, "load_h31_dflash_statistics", None)
    assert loader is not None
    result = loader(path)

    assert result["Full"]["mean"] == [pytest.approx(0.26)]
    assert result["Zero"]["mean"] == [pytest.approx(0.25), pytest.approx(0.15)]
    assert result["Zero"]["ci_low"] == [pytest.approx(0.20), pytest.approx(0.10)]
    assert result["Zero"]["n"] == [50, 50]


def test_h31_pilot_condition_labels_are_explicit_for_provenance_plot():
    labeler = getattr(report_plots, "h31_pilot_condition_label", None)
    assert labeler is not None

    assert labeler("F") == "Full"
    assert labeler("R25") == "Reduced 25%"
    assert labeler("D0") == "Deleted"


def test_report_generation_excludes_n2_pilot_figures(tmp_path, monkeypatch):
    called = []

    monkeypatch.setattr(report_plots, "load_jsonl_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(report_plots, "load_csv_rows", lambda *args, **kwargs: [])

    def recorder(name):
        def _record(*args, **kwargs):
            called.append(name)
            return []

        return _record

    for name in (
        "plot_h1b_attention",
        "plot_h1b_value_ablation",
        "plot_h2_acceptance",
        "plot_qwen2_sweeps",
        "plot_h1a_layer_proxy",
    ):
        monkeypatch.setattr(report_plots, name, recorder(name))

    def forbidden(*args, **kwargs):
        pytest.fail("underpowered pilot figure should not be generated")

    monkeypatch.setattr(report_plots, "plot_e5_zero_value_pilot", forbidden)
    monkeypatch.setattr(report_plots, "plot_h3_position_acceptance", forbidden)
    monkeypatch.setattr(report_plots, "plot_h3_backend_cohort_provenance", forbidden)

    report_plots.generate_figures(
        tmp_path,
        h1b_vdc_paths=[],
        h1b_mvbench_paths=[],
        h2_paths=[],
        e5_attention_real_path="e5-real.jsonl",
        e5_attention_zero_path="e5-zero.jsonl",
        e5_acceptance_real_path="e5-accept-real.jsonl",
        e5_acceptance_zero_path="e5-accept-zero.jsonl",
        h3_position_csv="h3-legacy.csv",
        h3_pilot_path="summary.md",
        h3_dflash_full_stats_path="h3-full.csv",
        legacy_length_csv="length.csv",
        legacy_retention_csv="retention.csv",
        layer_proxy_csv="layer.csv",
    )

    manifest = json.loads((tmp_path / "figure_manifest.json").read_text())
    assert "plot_e5_zero_value_pilot" not in called
    assert "plot_h3_position_acceptance" not in called
    assert "plot_h3_backend_cohort_provenance" not in called
    assert "e5_pilot_attention_real" in manifest["excluded_sources"]
    assert "h3_msd_pilot" in manifest["excluded_sources"]
