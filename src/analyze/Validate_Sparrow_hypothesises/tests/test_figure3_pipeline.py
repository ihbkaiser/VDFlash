from __future__ import annotations

import json
from pathlib import Path

from src.analyze.Validate_Sparrow_hypothesises.figure3_pipeline import (
    build_figure3_statistics,
    build_panel_commands,
    load_figure3_bundle,
    validate_figure3_summaries,
    write_figure3_metadata,
)
from src.analyze.Validate_Sparrow_hypothesises.cli import build_parser as build_cli_parser
from src.analyze.Validate_Sparrow_hypothesises.run_paper_experiments import build_parser as build_all_parser


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
TASKS = ("action_prediction",)


def _summary(
    figure: str,
    *,
    records: int = 2,
    rows: int = 6,
    layers: int = 3,
    manifest_sha: str = "same",
    model: str = MODEL,
) -> dict:
    common = {
        "paper_figure": figure,
        "model": model,
        "manifest": "/tmp/selected.jsonl",
        "manifest_sha256": manifest_sha,
        "tasks": list(TASKS),
        "num_records": records,
        "num_error_records": 0,
        "coverage": 1.0,
        "layer_count": layers,
        "dtype": "bfloat16",
        "device_map": "auto",
        "quantized": False,
        "fps": 8.0,
        "max_frames": 8,
        "min_pixels": 200704,
        "max_pixels": 151200,
    }
    if figure == "Figure 3(a)":
        common.update(
            {
                "num_scored_rows": rows,
                "expected_rows_if_complete": rows,
                "layer_cut_points": [0, 2],
            }
        )
    else:
        common.update(
            {
                "num_success_rows": rows,
                "expected_rows_if_complete": rows,
                "layer_indices": list(range(layers)),
                "layer_index_convention": "zero_based_native_decoder_layer",
            }
        )
    return common


def test_current_figure3_requires_matching_model_manifest_and_preprocessing():
    a = _summary("Figure 3(a)", manifest_sha="same")
    b = _summary("Figure 3(b)", manifest_sha="different")

    result = validate_figure3_summaries(a, b, expected_model=MODEL)

    assert not result["valid"]
    assert any(issue["code"] == "manifest_mismatch" for issue in result["issues"])


def test_panel_commands_share_explicit_current_figure3_configuration(tmp_path: Path):
    manifest = tmp_path / "selected.jsonl"
    output_dir = tmp_path / "figure3"

    commands = build_panel_commands(
        python="python",
        model=MODEL,
        manifest=manifest,
        output_dir=output_dir,
        tasks=TASKS,
        limit_per_task=2,
        dtype="bfloat16",
        device_map="auto",
        quantized=False,
    )

    assert commands["a"][-2:] == ["--plot-output", str(output_dir / "figure3a.png")]
    assert "--model" in commands["b"] and MODEL in commands["b"]
    assert "--tasks" in commands["a"] and "action_prediction" in commands["a"]
    assert str(manifest) in commands["a"] and str(manifest) in commands["b"]


def test_statistics_and_metadata_are_written_in_figure3_directory(tmp_path: Path):
    a_summary = _summary("Figure 3(a)")
    b_summary = _summary("Figure 3(b)")
    a_rows = [
        {"task": "action_prediction", "layer_cut": None, "correct": True},
        {"task": "action_prediction", "layer_cut": 0, "correct": False},
    ]
    b_rows = [{"layer": 0, "visual_mass": 0.25, "per_head_visual_mass": [0.25]}]

    statistics = build_figure3_statistics(a_rows, b_rows)
    files = write_figure3_metadata(
        tmp_path,
        a_summary,
        b_summary,
        a_rows,
        b_rows,
        audit={"valid": True, "issues": []},
    )

    assert statistics["figure3a"]
    assert statistics["figure3b"]
    assert (tmp_path / "figure3a_statistics.csv").is_file()
    assert (tmp_path / "figure3b_statistics.csv").is_file()
    assert "figure3_metadata.json" in files
    metadata = json.loads((tmp_path / "figure3_metadata.json").read_text())
    assert metadata["model"] == MODEL
    assert metadata["manifest_sha256"] == "same"


def test_full_orchestrator_exposes_current_figure3_configuration(tmp_path: Path):
    args = build_all_parser().parse_args(
        [
            "--include-current-figure3",
            "--figure3-model",
            MODEL,
            "--figure3-manifest",
            str(tmp_path / "selected.jsonl"),
            "--figure3-output-dir",
            str(tmp_path / "figure3"),
            "--figure3-tasks",
            "action_prediction",
            "--figure3-limit-per-task",
            "2",
            "--figure3-dtype",
            "float32",
        ]
    )

    assert args.include_current_figure3 is True
    assert args.figure3_model == MODEL
    assert args.figure3_tasks == ["action_prediction"]
    assert args.figure3_limit_per_task == 2
    assert args.figure3_dtype == "float32"


def test_package_cli_exposes_direct_current_figure3_command(tmp_path: Path):
    args = build_cli_parser().parse_args(
        [
            "figure3",
            "--model",
            MODEL,
            "--manifest",
            str(tmp_path / "selected.jsonl"),
            "--output-dir",
            str(tmp_path / "figure3"),
        ]
    )

    assert args.command == "figure3"
    assert args.model == MODEL
    assert args.output_dir == str(tmp_path / "figure3")


def test_report_externalizes_legacy_figure3_rows_and_keeps_current_provenance(tmp_path: Path):
    bundle_dir = tmp_path / "figure3"
    a_summary = _summary("Figure 3(a)")
    b_summary = _summary("Figure 3(b)")
    (bundle_dir / "figure3a.jsonl").parent.mkdir(parents=True)
    (bundle_dir / "figure3a.jsonl").write_text("\n")
    (bundle_dir / "figure3b.jsonl").write_text("\n")
    (bundle_dir / "figure3a.summary.json").write_text(json.dumps(a_summary))
    (bundle_dir / "figure3b.summary.json").write_text(json.dumps(b_summary))
    write_figure3_metadata(
        bundle_dir,
        a_summary,
        b_summary,
        [],
        [],
        audit={"valid": True, "issues": []},
    )
    bundle = load_figure3_bundle(bundle_dir)
    legacy_rows = [
        {"row_id": "legacy-a", "paper_figure": "Figure 3"},
        {"row_id": "legacy-b", "paper_figure": "Figure 3(b)"},
        {
            "row_id": "f6",
            "paper_figure": "Figure 6 / Appendix D",
            "sample_id": "sample",
            "target_model": "Qwen/Qwen2-VL-7B-Instruct",
            "temperature": 0.0,
            "target_visual_tokens": 3000,
            "actual_visual_tokens": 3000,
            "target_input_fingerprint": "same",
            "draft_input_fingerprint": "same",
            "layer": 1,
            "visual_cosine": 0.9,
            "text_cosine": 0.8,
        },
    ]

    from src.analyze.Validate_Sparrow_hypothesises.report import build_report
    report = build_report(legacy_rows, __import__(
        "src.analyze.Validate_Sparrow_hypothesises.paper_contract",
        fromlist=["DEFAULT_CONTRACT"],
    ).DEFAULT_CONTRACT, current_figure3=bundle)

    assert report["current_figure3"]["audit"]["valid"] is True
    assert all(row["paper_figure"] not in {"Figure 3", "Figure 3(b)"} for row in report["_rows"])


def test_report_writer_links_current_figure3_bundle(tmp_path: Path):
    bundle_dir = tmp_path / "report" / "figure3"
    a_summary = _summary("Figure 3(a)")
    b_summary = _summary("Figure 3(b)")
    (bundle_dir / "figure3a.jsonl").parent.mkdir(parents=True)
    (bundle_dir / "figure3a.jsonl").write_text("\n")
    (bundle_dir / "figure3b.jsonl").write_text("\n")
    (bundle_dir / "figure3a.summary.json").write_text(json.dumps(a_summary))
    (bundle_dir / "figure3b.summary.json").write_text(json.dumps(b_summary))
    (bundle_dir / "figure3_insight_layer_analysis.png").write_bytes(b"png")
    write_figure3_metadata(bundle_dir, a_summary, b_summary, [], [], audit={"valid": True, "issues": []})
    bundle = load_figure3_bundle(bundle_dir)
    from src.analyze.Validate_Sparrow_hypothesises.paper_contract import DEFAULT_CONTRACT
    from src.analyze.Validate_Sparrow_hypothesises.report import build_report, write_report

    report = build_report([], DEFAULT_CONTRACT, current_figure3=bundle)
    write_report(tmp_path / "report", report)
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    markdown = (tmp_path / "report" / "REPORT.md").read_text()
    assert "figure3/figure3_insight_layer_analysis.png" in summary["plots"]
    assert "Current Qwen2.5-VL-3B Figure 3" in markdown


def test_audit_command_accepts_external_current_figure3_bundle(tmp_path: Path):
    args = build_cli_parser().parse_args(
        [
            "audit",
            "--input",
            str(tmp_path / "results.jsonl"),
            "--current-figure3-dir",
            str(tmp_path / "figure3"),
        ]
    )
    assert args.current_figure3_dir == str(tmp_path / "figure3")


def test_external_figure3_coverage_still_requires_legacy_figure6():
    from src.analyze.Validate_Sparrow_hypothesises.coverage import build_coverage
    from src.analyze.Validate_Sparrow_hypothesises.paper_contract import DEFAULT_CONTRACT

    coverage = build_coverage(
        [],
        DEFAULT_CONTRACT,
        excluded_figures={"Figure 3", "Figure 3(b)"},
    )

    assert any(name.startswith("Figure 6 / Appendix D") for name in coverage.observed["missing"])
