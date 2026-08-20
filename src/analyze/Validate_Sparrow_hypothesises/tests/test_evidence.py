from __future__ import annotations

from pathlib import Path

from src.analyze.Validate_Sparrow_hypothesises.evidence import (
    build_final_evidence,
    MalformedJsonlError,
    collect_evidence,
    read_jsonl_strict,
    select_evidence,
)


def _row(row_id: str, **overrides):
    row = {"row_id": row_id, "paper_figure": "Figure 2", "calibration_status": "ok"}
    row.update(overrides)
    return row


def test_evidence_filters_runtime_and_lossless_rows():
    result = select_evidence([
        _row("ok"),
        _row("error", status="error"),
        _row("parity", native_prefill_parity={"valid": False}),
        _row("mismatch", target_output_ids=[1], speculative_output_ids=[2]),
        _row("ablation", paper_figure="Figure 3", target_output_ids=[1], speculative_output_ids=[2]),
    ])
    assert [row["row_id"] for row in result.evidence_rows] == ["ok", "ablation"]
    assert {row["diagnostic_reason"] for row in result.diagnostic_rows} == {
        "runtime_error", "native_prefill_parity_failure", "lossless_mismatch"
    }


def test_evidence_last_retry_wins_and_old_row_is_diagnostic():
    result = select_evidence([_row("same", value="old"), _row("same", value="new")])
    assert result.evidence_rows[0]["value"] == "new"
    assert result.diagnostic_rows[0]["diagnostic_reason"] == "duplicate_row_id_superseded"


def test_evidence_uses_runner_lossless_prefix_with_post_eos_tail():
    result = select_evidence([
        _row(
            "prefix-lossless",
            paper_figure="Figure 1(a)",
            lossless=True,
            target_output_ids=[1, 2],
            speculative_output_ids=[1, 2, 3],
        )
    ])
    assert [row["row_id"] for row in result.evidence_rows] == ["prefix-lossless"]


def test_malformed_jsonl_is_rejected_and_excluded(tmp_path: Path):
    path = tmp_path / "truncated.jsonl"
    path.write_text('{"row_id": "ok"}\n{"row_id":', encoding="utf-8")
    try:
        read_jsonl_strict(path)
    except MalformedJsonlError:
        pass
    else:  # pragma: no cover
        raise AssertionError("malformed JSONL was accepted")
    result = collect_evidence([path])
    assert not result.evidence_rows
    assert result.malformed_files == (str(path),)
    assert result.diagnostic_rows[0]["diagnostic_reason"] == "malformed_jsonl"


def _figure2_summary(row_id: str, sample_id: str, target: int, actual: int):
    return _row(
        row_id,
        paper_figure="Figure 2",
        sample_id=sample_id,
        modality="summary",
        attention_source="msd_draft",
        attention_policy="last_instruction",
        calibration_target_visual_tokens=target,
        actual_visual_tokens=actual,
        calibration_status="ok",
        attention_weights=[1.0],
        instruction_positions=[],
        visual_positions=[],
        text_positions=[0],
    )


def test_final_evidence_replaces_only_figure2_with_homogeneous_summary_rows():
    rows = [
        {"row_id": "f1", "paper_figure": "Figure 1(a)"},
        _figure2_summary("a-400", "a", 400, 572),
        _figure2_summary("a-3000", "a", 3000, 2912),
        _figure2_summary("b-400", "b", 400, 572),
        _figure2_summary("b-3000", "b", 3000, 2912),
        _figure2_summary("c-400", "c", 400, 560),
        _figure2_summary("c-3000", "c", 3000, 2900),
        {"row_id": "f3", "paper_figure": "Figure 3"},
    ]

    result = build_final_evidence(rows, figure2_targets=(400, 3000), minimum_samples=2)

    assert [row["row_id"] for row in result.figure2_rows] == ["a-400", "a-3000", "b-400", "b-3000"]
    assert result.cohort["sample_ids"] == ["a", "b"]
    assert result.cohort["actual_visual_tokens"] == [572, 2912]
    assert len(result.diagnostic_rows) == 2
    assert {row["diagnostic_reason"] for row in result.diagnostic_rows} == {"figure2_nonhomogeneous_cohort"}


def test_final_evidence_fails_closed_without_minimum_homogeneous_cohort():
    rows = [
        _figure2_summary("a-400", "a", 400, 572),
        _figure2_summary("a-3000", "a", 3000, 2912),
    ]
    try:
        build_final_evidence(rows, figure2_targets=(400, 3000), minimum_samples=2)
    except ValueError as exc:
        assert "homogeneous" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mixed/undersized Figure 2 cohort was accepted")
