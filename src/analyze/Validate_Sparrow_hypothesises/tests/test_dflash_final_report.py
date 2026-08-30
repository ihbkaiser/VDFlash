from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analyze.Validate_Sparrow_hypothesises.build_dflash_final_report import (
    build_report,
    build_summary,
    summarize_attention_rows,
    summarize_decode_rows,
    summarize_layer_rows,
)


def test_summarize_decode_rows_preserves_status_counts_and_metrics() -> None:
    rows = [
        {
            "status": "ok",
            "length_target": 400,
            "target_visual_tokens": 468,
            "metrics": {"tau_effective": 3.0},
            "speedup": {"end_to_end": 2.0},
        },
        {
            "status": "mismatch",
            "length_target": 400,
            "target_visual_tokens": 432,
            "metrics": {"tau_effective": 1.0},
            "speedup": {"end_to_end": 1.5},
        },
        {
            "status": "unsupported",
            "length_target": 400,
            "target_visual_tokens": 400,
        },
        {"status": "error", "length_target": 400},
        {"status": "unexpected", "length_target": 400},
    ]

    summary = summarize_decode_rows(rows, "length_target")

    assert summary == [
        {
            "condition": 400,
            "n": 5,
            "valid_n": 2,
            "ok": 1,
            "mismatch": 1,
            "unsupported": 1,
            "error": 1,
            "unknown": 1,
            "lossless_rate": 1 / 2,
            "tau_effective_mean": 2.0,
            "accepted_length_mean": 2.0,
            "speculative_latency_mean": None,
            "target_latency_mean": None,
            "speedup_mean": 1.75,
        }
    ]


def test_summarize_attention_rows_groups_target_and_layer() -> None:
    rows = [
        {"target_visual_tokens": 400, "layer_index": 0, "context_attention_mass": 0.6},
        {"target_visual_tokens": 400, "layer_index": 0, "context_attention_mass": 0.8},
        {"target_visual_tokens": 3000, "layer_index": 1, "context_attention_mass": 0.4},
    ]

    assert summarize_attention_rows(rows) == [
        {"target_visual_tokens": 400, "layer_index": 0, "n": 2, "context_attention_mass_mean": 0.7},
        {"target_visual_tokens": 3000, "layer_index": 1, "n": 1, "context_attention_mass_mean": 0.4},
    ]


def test_summarize_layer_rows_separates_diagnostics() -> None:
    rows = [
        {
            "experiment": "qwen25vl_target_visual_kv",
            "layer_index": 0,
            "metrics": {"diagnostic_output_length": 100},
        },
        {
            "experiment": "qwen25vl_target_attention",
            "layer_index": 0,
            "metrics": {"visual_attention_mass": 0.25},
        },
        {
            "experiment": "qwen25vl_target_hidden_cosine",
            "layer_index": 0,
            "metrics": {"visual_cosine": 0.9, "text_cosine": 0.1},
        },
    ]

    summary = summarize_layer_rows(rows)

    assert summary["visual_kv"] == [{"layer_index": 0, "n": 1, "diagnostic_output_length_mean": 100.0}]
    assert summary["attention"] == [{"layer_index": 0, "n": 1, "visual_attention_mass_mean": 0.25}]
    assert summary["cosine"] == [
        {"layer_index": 0, "n": 1, "text_cosine_mean": 0.1, "visual_cosine_mean": 0.9}
    ]


def test_build_summary_reads_source_audits(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    layers = tmp_path / "layers"
    primary.mkdir()
    layers.mkdir()
    (primary / "dflash_audit.json").write_text(
        json.dumps({"coverage_valid": False, "mismatch_rows": 2, "unsupported_rows": [{}]}),
        encoding="utf-8",
    )
    (layers / "dflash_audit.json").write_text(
        json.dumps({"coverage_valid": True, "mismatch_rows": 0, "unsupported_rows": []}),
        encoding="utf-8",
    )
    (primary / "figure1a_length_sweep.jsonl").write_text(
        json.dumps({"status": "mismatch", "length_target": 400}) + "\n",
        encoding="utf-8",
    )
    (primary / "figure1b_target_hidden_visual_retention.jsonl").write_text("", encoding="utf-8")
    (primary / "figure2_dflash_context_attention.jsonl").write_text("", encoding="utf-8")
    (layers / "figure3_3b_6_target_diagnostics.jsonl").write_text("", encoding="utf-8")

    summary = build_summary(primary, layers)

    assert summary["status"] == "INCOMPLETE DIAGNOSTIC"
    assert summary["primary_audit"]["mismatch_rows"] == 2
    assert summary["primary_audit"]["unsupported_rows"] == [{}]


def test_build_summary_fails_fast_when_a_required_jsonl_is_missing(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    layers = tmp_path / "layers"
    primary.mkdir()
    layers.mkdir()
    audit = {
        "coverage_valid": True,
        "mismatch_rows": 0,
        "unsupported_rows": [],
        "error_rows": [],
        "invalid_rows": [],
        "coverage_gaps": {},
    }
    (primary / "dflash_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (layers / "dflash_audit.json").write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="figure1a_length_sweep.jsonl"):
        build_summary(primary, layers)


def test_build_summary_uses_layer_audit_to_filter_rows_and_status(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    layers = tmp_path / "layers"
    primary.mkdir()
    layers.mkdir()
    clean_audit = {
        "coverage_valid": True,
        "mismatch_rows": 0,
        "unsupported_rows": [],
        "error_rows": [],
        "invalid_rows": [],
        "coverage_gaps": {},
    }
    layer_audit = {**clean_audit, "unsupported_rows": [{"index": 1, "reason": "probe failed"}]}
    (primary / "dflash_audit.json").write_text(json.dumps(clean_audit), encoding="utf-8")
    (layers / "dflash_audit.json").write_text(json.dumps(layer_audit), encoding="utf-8")
    (primary / "figure1a_length_sweep.jsonl").write_text("{}\n", encoding="utf-8")
    (primary / "figure1b_target_hidden_visual_retention.jsonl").write_text("{}\n", encoding="utf-8")
    (primary / "figure2_dflash_context_attention.jsonl").write_text("{}\n", encoding="utf-8")
    (layers / "figure3_3b_6_target_diagnostics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "experiment": "qwen25vl_target_visual_kv",
                    "layer_index": 0,
                    "metrics": {"diagnostic_output_length": 10},
                }),
                json.dumps({
                    "experiment": "qwen25vl_target_visual_kv",
                    "layer_index": 1,
                    "status": "unsupported",
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_summary(primary, layers)

    assert summary["status"] == "INCOMPLETE DIAGNOSTIC"
    assert summary["row_counts"]["layer_diagnostics"] == 2
    assert summary["audited_row_counts"]["layer_diagnostics"] == 1
    assert summary["layers"]["visual_kv"] == [
        {"layer_index": 0, "n": 1, "diagnostic_output_length_mean": 10.0}
    ]


def test_build_report_writes_figures_and_machine_readable_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    primary = tmp_path / "primary"
    layers = tmp_path / "layers"
    output = tmp_path / "report"
    primary.mkdir()
    layers.mkdir()
    audit = {
        "coverage_valid": True,
        "mismatch_rows": 0,
        "unsupported_rows": [],
        "error_rows": [],
        "invalid_rows": [],
        "coverage_gaps": {},
    }
    (primary / "dflash_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (layers / "dflash_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (primary / "figure1a_length_sweep.jsonl").write_text(
        json.dumps({
            "status": "ok",
            "length_target": 400,
            "target_visual_tokens": 400,
            "metrics": {"tau_effective": 2.0},
            "speedup": {"end_to_end": 1.5},
        }) + "\n",
        encoding="utf-8",
    )
    retention_rows = [
        {
            "status": "ok",
            "retention_percentage": percentage,
            "metrics": {"tau_effective": 2.0},
            "speedup": {"end_to_end": 1.5},
        }
        for percentage in (100, 25, 10, 5, 1, 0)
    ]
    (primary / "figure1b_target_hidden_visual_retention.jsonl").write_text(
        "\n".join(json.dumps(row) for row in retention_rows) + "\n",
        encoding="utf-8",
    )
    (primary / "figure2_dflash_context_attention.jsonl").write_text(
        json.dumps({
            "target_visual_tokens": 400,
            "layer_index": 0,
            "context_attention_mass": 0.5,
            "noise_attention_mass": 0.5,
        }) + "\n",
        encoding="utf-8",
    )
    (layers / "figure3_3b_6_target_diagnostics.jsonl").write_text(
        "\n".join([
            json.dumps({"experiment": "qwen25vl_target_visual_kv", "layer_index": 0, "metrics": {"diagnostic_output_length": 10}}),
            json.dumps({"experiment": "qwen25vl_target_attention", "layer_index": 0, "metrics": {"visual_attention_mass": 0.2}}),
            json.dumps({"experiment": "qwen25vl_target_hidden_cosine", "layer_index": 0, "metrics": {"visual_cosine": 0.8, "text_cosine": 0.1}}),
        ]) + "\n",
        encoding="utf-8",
    )

    build_report(primary, layers, output)

    assert (output / "REPORT_FINAL.md").is_file()
    assert (output / "summary.json").is_file()
    for stem in (
        "figure1_insight_summary",
        "figure2_insight_attention",
        "figure3_insight_layer_analysis",
        "figure6_insight_retention",
    ):
        for extension in ("png", "pdf", "svg"):
            assert (output / f"{stem}.{extension}").is_file()
    figure_svg = (output / "figure1_insight_summary.svg").read_text(encoding="utf-8")
    for label in ("100", "25", "10", "5", "1", "0"):
        assert f"<!-- {label} -->" in figure_svg
    report = (output / "REPORT_FINAL.md").read_text(encoding="utf-8")
    assert "DFlash proxy" in report
    assert "per-token modality traces" in report
    assert "head-summed visual attention" in report


def test_report_embeds_figures_and_explains_each_experiment(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    primary = tmp_path / "primary"
    layers = tmp_path / "layers"
    output = tmp_path / "report"
    primary.mkdir()
    layers.mkdir()
    audit = {
        "coverage_valid": True,
        "mismatch_rows": 0,
        "unsupported_rows": [],
        "error_rows": [],
        "invalid_rows": [],
        "coverage_gaps": {},
    }
    (primary / "dflash_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (layers / "dflash_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (primary / "figure1a_length_sweep.jsonl").write_text(
        json.dumps({
            "status": "ok",
            "length_target": 400,
            "target_visual_tokens": 400,
            "metrics": {"tau_effective": 2.0},
            "timing": {"speculative": {"end_to_end_s": 2.0}, "target": {"end_to_end_s": 3.0}},
            "speedup": {"end_to_end": 1.5},
        }) + "\n",
        encoding="utf-8",
    )
    (primary / "figure1b_target_hidden_visual_retention.jsonl").write_text(
        json.dumps({
            "status": "ok",
            "retention_percentage": 100,
            "metrics": {"tau_effective": 2.0},
            "timing": {"speculative": {"end_to_end_s": 2.0}, "target": {"end_to_end_s": 3.0}},
            "speedup": {"end_to_end": 1.5},
        }) + "\n",
        encoding="utf-8",
    )
    (primary / "figure2_dflash_context_attention.jsonl").write_text(
        json.dumps({
            "target_visual_tokens": 400,
            "layer_index": 0,
            "context_attention_mass": 0.5,
            "noise_attention_mass": 0.5,
        }) + "\n",
        encoding="utf-8",
    )
    (layers / "figure3_3b_6_target_diagnostics.jsonl").write_text(
        "\n".join([
            json.dumps({"experiment": "qwen25vl_target_visual_kv", "layer_index": 0, "metrics": {"diagnostic_output_length": 10}}),
            json.dumps({"experiment": "qwen25vl_target_attention", "layer_index": 0, "metrics": {"visual_attention_mass": 0.2}}),
            json.dumps({"experiment": "qwen25vl_target_hidden_cosine", "layer_index": 0, "metrics": {"visual_cosine": 0.8, "text_cosine": 0.1}}),
        ]) + "\n",
        encoding="utf-8",
    )

    build_report(primary, layers, output)
    report = (output / "REPORT_FINAL.md").read_text(encoding="utf-8")

    assert "![Figure 1" in report
    assert "![Figure 2" in report
    assert "![Figure 3" in report
    assert "![Figure 6" in report
    assert "## Experiment 1" in report
    assert "## Experiment 2" in report
    assert "## Experiment 3" in report
    assert "## Experiment 4" in report
    assert "Insight thực nghiệm" in report
    assert "50 video" in report
    assert "max_new_tokens=256" in report
    assert "hidden_context_mask" in report
    assert "eager mode" in report
    assert "visual-KV" in report
    assert "input embedding" in report
