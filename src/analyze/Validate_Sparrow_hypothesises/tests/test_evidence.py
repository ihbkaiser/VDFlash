from __future__ import annotations

from pathlib import Path

from src.analyze.Validate_Sparrow_hypothesises.evidence import (
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
