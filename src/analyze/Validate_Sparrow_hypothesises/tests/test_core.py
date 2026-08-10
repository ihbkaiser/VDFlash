from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.audit import audit_losslessness, audit_rows
from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    choose_nearest_calibration,
    qwen2vl_video_token_count,
)
from src.analyze.Validate_Sparrow_hypothesises.metrics import (
    common_prefix_length,
    normalized_entropy,
    rouge_l,
)
from src.analyze.Validate_Sparrow_hypothesises.paper_contract import DEFAULT_CONTRACT, validate_contract
from src.analyze.Validate_Sparrow_hypothesises.runtime import select_visual_positions


def test_default_contract_matches_paper_milestones():
    assert validate_contract(DEFAULT_CONTRACT) == []
    assert DEFAULT_CONTRACT.visual_token_milestones == (400, 3000, 13000, 25000)
    assert DEFAULT_CONTRACT.layer_cut_points[5] == 20


def test_video_token_count_uses_qwen_grid_and_merge_size():
    assert qwen2vl_video_token_count([[2, 14, 14]]) == 98
    assert qwen2vl_video_token_count([[1, 14, 14], [1, 14, 14]]) == 98


def test_calibration_is_nearest_and_marks_tolerance():
    point = choose_nearest_calibration("sample", 400, [("a", 360), ("b", 470)], tolerance=0.10)
    assert point.candidate_id == "a"
    assert point.status == "ok"
    point = choose_nearest_calibration("sample", 400, [("a", 500)], tolerance=0.10)
    assert point.status == "out_of_tolerance"


def test_metrics_are_deterministic_and_bounded():
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert rouge_l("a b c", "a b c") == 1.0
    assert 0.0 <= normalized_entropy([0.5, 0.5]) <= 1.0


def _valid_row(**overrides):
    row = {
        "row_id": "sample:400",
        "paper_figure": "Figure 1(a)",
        "sample_id": "sample",
        "target_model": DEFAULT_CONTRACT.msd_target_model,
        "temperature": 0.0,
        "target_visual_tokens": 400,
        "actual_visual_tokens": 392,
        "target_input_fingerprint": "same-target",
        "target_input_fingerprint_reference": "same-target",
        "draft_input_fingerprint": "draft",
    }
    row.update(overrides)
    return row


def test_paper_audit_rejects_target_draft_leak():
    report = audit_rows([_valid_row(paper_figure="Figure 1(b)", full_target_visual_tokens=399)], DEFAULT_CONTRACT)
    assert not report.valid
    assert any(issue.code == "target_draft_leak" for issue in report.issues)


def test_paper_audit_rejects_wrong_attention_query_and_missing_layer_mask():
    attention_row = _valid_row(
        row_id="sample:attention",
        paper_figure="Figure 2",
        attention_query="first_instruction",
    )
    layer_row = _valid_row(row_id="sample:layer", paper_figure="Figure 3")
    report = audit_rows([attention_row, layer_row], DEFAULT_CONTRACT)
    codes = {issue.code for issue in report.issues}
    assert "wrong_attention_query" in codes
    assert "missing_layer_intervention" in codes


def test_losslessness_audit_compares_token_ids_not_text():
    good = {"row_id": "a", "target_output_ids": [1, 2], "speculative_output_ids": [1, 2]}
    bad = {"row_id": "b", "target_output_ids": [1, 2], "speculative_output_ids": [1, 3]}
    report = audit_losslessness([good, bad])
    assert not report.valid
    assert any(issue.code == "lossless_mismatch" for issue in report.issues)


def test_visual_selector_is_deterministic_and_tie_stable():
    assert select_visual_positions(8, 0).tolist() == []
    assert select_visual_positions(8, 100).tolist() == list(range(8))
    assert select_visual_positions(8, 25, [1.0] * 8).tolist() == [0, 1]
    assert select_visual_positions(8, 25, [0.1, 0.9, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6]).tolist() == [1, 2]
