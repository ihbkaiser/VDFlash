from __future__ import annotations

from pathlib import Path

from src.analyze.Validate_Sparrow_hypothesises.dflash_audit import audit_dflash_rows
from src.analyze.Validate_Sparrow_hypothesises.dflash_report import write_dflash_report
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import (
    DFLASH_STAGE_ORDER,
    _stage_row_key,
    _stage_row_sink,
    _read_stage_rows,
    build_parser,
    default_output_dir,
    run_dflash_experiments,
)


def _row(experiment, *, status="ok", semantic_status="direct", **extra):
    row = {
        "backend": "dflash",
        "experiment": experiment,
        "semantic_status": semantic_status,
        "target_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "draft_checkpoint": "dataset/qwen25vl-3b-dflash-llava68k-latest/training_state.pt",
        "draft_config": "draft.json",
        "sample_id": "sample-0",
        "input_fingerprint": "input-0",
        "status": status,
        "metrics": {},
    }
    if status == "ok" and experiment in {"length_sweep", "target_hidden_visual_retention"}:
        row.update(target_output_ids=[1, 2], speculative_output_ids=[1, 2])
    row.update(extra)
    return row


def test_audit_separates_semantics_and_checks_losslessness():
    rows = [
        _row("length_sweep"),
        _row(
            "target_hidden_visual_retention",
            semantic_status="adapted",
            retention_percentage=100,
            full_target_input_fingerprint="full-0",
            target_input_fingerprint="full-0",
        ),
        _row(
            "dflash_context_attention",
            semantic_status="adapted",
            query_policy="draft_block",
            attention_source="dflash_context",
        ),
        _row(
            "qwen25vl_target_hidden_cosine",
            semantic_status="target_side_diagnostic",
            target_output_ids=None,
            speculative_output_ids=None,
            layer_index=4,
        ),
    ]

    audit = audit_dflash_rows(rows)

    assert audit["valid_rows"] == 4
    assert audit["semantic_status_counts"] == {
        "direct": 1,
        "adapted": 2,
        "target_side_diagnostic": 1,
    }
    assert audit["lossless_rows"] == 2
    assert audit["invalid_rows"] == []


def test_audit_reports_retention_fingerprint_mismatch():
    rows = [
        _row(
            "target_hidden_visual_retention",
            semantic_status="adapted",
            retention_percentage=100,
            full_target_input_fingerprint="full-0",
            target_input_fingerprint="full-0",
        ),
        _row(
            "target_hidden_visual_retention",
            semantic_status="adapted",
            retention_percentage=0,
            full_target_input_fingerprint="full-1",
            target_input_fingerprint="full-1",
        ),
    ]

    audit = audit_dflash_rows(rows)

    assert audit["retention_fingerprint_errors"] == ["sample-0"]


def test_audit_can_report_missing_requested_grid_values():
    audit = audit_dflash_rows(
        [_row("length_sweep", length_target=3000)],
        expected_grid={"length_targets": [400, 3000]},
    )

    assert audit["coverage_gaps"] == {"length_targets": [400]}
    assert audit["coverage_valid"] is False


def test_audit_marks_runtime_error_rows_as_not_covered():
    audit = audit_dflash_rows(
        [_row("length_sweep", status="unsupported", error="OOM")]
    )

    assert len(audit["error_rows"]) == 1
    assert audit["coverage_valid"] is False


def test_dflash_parser_defaults_are_isolated_and_stage_order_is_stable():
    args = build_parser().parse_args(["--dry-run"])

    assert args.checkpoint.endswith("training_state.pt")
    assert args.target_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert DFLASH_STAGE_ORDER == ("length", "retention", "attention", "layers", "report")
    assert Path(default_output_dir()).name.startswith("sparrow_validation_dflash_qwen25vl3b_")


def test_preflight_reports_artifact_paths_and_cuda_state_without_loading_models():
    args = build_parser().parse_args(["preflight"])

    result = run_dflash_experiments(args)

    assert result["preflight"]["missing_paths"] == []
    assert result["preflight"]["checkpoint_bytes"] > 0
    assert isinstance(result["preflight"]["cuda_available"], bool)


def test_dry_run_reports_missing_paths_instead_of_claiming_readiness(tmp_path):
    args = build_parser().parse_args(["--dry-run", "--checkpoint", str(tmp_path / "missing.pt")])

    result = run_dflash_experiments(args)

    assert str(tmp_path / "missing.pt") in result["preflight"]["missing_paths"]
    assert result["preflight"]["ready_for_model_run"] is False


def test_stage_row_key_distinguishes_partial_resume_conditions():
    assert _stage_row_key(
        "length",
        {"sample_id": "sample-0", "length_target": 3000},
    ) == ("sample-0", "length", "3000")
    assert _stage_row_key(
        "retention",
        {"sample_id": "sample-0", "retention_percentage": 25},
    ) == ("sample-0", "retention", "25")
    assert _stage_row_key(
        "attention",
        {"sample_id": "sample-0", "target_visual_tokens": 3000},
    ) == ("sample-0", "attention", "3000")


def test_stage_journal_survives_a_truncated_final_line(tmp_path):
    path = tmp_path / "length.jsonl"
    row = {"sample_id": "sample-0", "length_target": 3000, "status": "ok"}

    with _stage_row_sink(path, append=False) as sink:
        sink(row)
    path.write_text(path.read_text(encoding="utf-8") + '{"truncated"', encoding="utf-8")

    assert _read_stage_rows(path) == [row]


def test_report_writes_dflash_only_artifacts(tmp_path):
    rows = [_row("length_sweep")]

    report_path = write_dflash_report(rows, tmp_path)

    assert report_path == tmp_path / "REPORT.md"
    assert report_path.is_file()
    assert "DFlash" in report_path.read_text(encoding="utf-8")
    assert (tmp_path / "dflash_audit.json").is_file()
