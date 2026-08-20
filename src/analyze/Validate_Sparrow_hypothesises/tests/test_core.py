from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analyze.Validate_Sparrow_hypothesises.audit import (
    audit_figure2_homogeneous,
    audit_losslessness,
    audit_rows,
)
from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    choose_nearest_calibration,
    qwen2vl_video_token_count,
)
from src.analyze.Validate_Sparrow_hypothesises.calibrate import (
    adaptive_candidate_grid,
    audit_calibration,
    select_paired_cohort,
    select_homogeneous_paired_cohort,
)
from src.analyze.Validate_Sparrow_hypothesises.metrics import (
    common_prefix_length,
    normalized_entropy,
    rouge_l,
)
from src.analyze.Validate_Sparrow_hypothesises.model_analysis import (
    PreparedQwen2,
    PreparedQwen25,
    find_instruction_masks,
    prepare_qwen2vl_prefill,
    prepare_qwen25_prefill,
)
from src.analyze.Validate_Sparrow_hypothesises.paper_statistics import build_paper_statistics, summarize
from src.analyze.Validate_Sparrow_hypothesises.paper_contract import DEFAULT_CONTRACT, validate_contract
from src.analyze.Validate_Sparrow_hypothesises.report import build_report
from src.analyze.Validate_Sparrow_hypothesises.plots import (
    _cohort_modality_ranges,
    _contiguous_ranges,
    _figure2_region_statistics,
)
from src.analyze.Validate_Sparrow_hypothesises.run_draft_attention import _strict_preceding_attention
from src.analyze.Validate_Sparrow_hypothesises.runtime import (
    PreparedPrefill,
    compact_qwen2vl_prefill,
    last_position_lm_head,
    select_visual_positions,
)


def test_default_contract_matches_paper_milestones():
    assert validate_contract(DEFAULT_CONTRACT) == []
    assert DEFAULT_CONTRACT.visual_token_milestones == (400, 3000, 13000, 25000)
    assert DEFAULT_CONTRACT.layer_cut_points[5] == 20


def test_plot_modality_positions_are_split_into_contiguous_ranges():
    assert _contiguous_ranges([0, 1, 2, 5, 7, 8, 8]) == [(0, 2), (5, 5), (7, 8)]


def test_plot_cohort_modality_ranges_follow_all_sample_boundaries():
    rows = [
        {"sample_id": "a", "modality": "text", "token_position": 0},
        {"sample_id": "a", "modality": "visual", "token_position": 1},
        {"sample_id": "a", "modality": "visual", "token_position": 2},
        {"sample_id": "a", "modality": "instruction", "token_position": 3},
        {"sample_id": "b", "modality": "text", "token_position": 0},
        {"sample_id": "b", "modality": "visual", "token_position": 1},
        {"sample_id": "b", "modality": "visual", "token_position": 2},
        {"sample_id": "b", "modality": "visual", "token_position": 3},
        {"sample_id": "b", "modality": "instruction", "token_position": 4},
        {"sample_id": "c", "modality": "text", "token_position": 0},
        {"sample_id": "c", "modality": "visual", "token_position": 1},
        {"sample_id": "c", "modality": "visual", "token_position": 2},
        {"sample_id": "c", "modality": "visual", "token_position": 3},
        {"sample_id": "c", "modality": "instruction", "token_position": 4},
    ]
    assert _cohort_modality_ranges(rows) == [
        ("text", 0, 0),
        ("visual", 1, 3),
        ("instruction", 4, 4),
    ]


def test_figure2_region_statistics_report_mass_and_mean_weight_per_token():
    rows = [
        {
            "row_id": "a",
            "attention_weights": [0.2, 0.3, 0.5],
            "instruction_positions": [0],
            "visual_positions": [1],
            "text_positions": [2],
            "instruction_mass": 0.2,
            "visual_mass": 0.3,
            "text_mass": 0.5,
        },
        {
            "row_id": "b",
            "attention_weights": [0.4, 0.2, 0.4],
            "instruction_positions": [0],
            "visual_positions": [1],
            "text_positions": [2],
            "instruction_mass": 0.4,
            "visual_mass": 0.2,
            "text_mass": 0.4,
        },
    ]
    stats = _figure2_region_statistics(rows)
    assert abs(stats["instruction"]["mass_mean"] - 0.3) < 1e-9
    assert abs(stats["visual"]["mass_mean"] - 0.25) < 1e-9
    assert abs(stats["text"]["mean_weight_per_token"] - 0.45) < 1e-9


def test_report_uses_homogeneous_figure2_source_without_dropping_full_audit_rows():
    full = [
        {"row_id": "f2-a", "paper_figure": "Figure 2"},
        {"row_id": "f2-b", "paper_figure": "Figure 2"},
        {"row_id": "f3", "paper_figure": "Figure 3"},
    ]
    selected = [{"row_id": "f2-selected", "paper_figure": "Figure 2"}]

    report = build_report(full, DEFAULT_CONTRACT, figure2_rows=selected)

    assert report["diagnostic_counts"]["Figure 2"] == 2
    assert [row["row_id"] for row in report["_rows"] if row["paper_figure"] == "Figure 2"] == ["f2-selected"]


def test_draft_attention_keeps_only_strictly_preceding_keys():
    captured = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4, 0.0],
         [0.05, 0.05, 0.1, 0.2, 0.6]],
    ])
    result = _strict_preceding_attention(captured, [2, 4])
    assert torch.allclose(result[0, 0], torch.tensor([1 / 3, 2 / 3, 0.0, 0.0, 0.0]))
    assert torch.allclose(result[0, 1], torch.tensor([1 / 8, 1 / 8, 1 / 4, 1 / 2, 0.0]))
    assert torch.allclose(result.sum(dim=-1), torch.ones(1, 2))


def test_last_position_lm_head_patches_accelerate_saved_forward():
    head = torch.nn.Linear(4, 7)
    original_forward = head.forward
    head._old_forward = original_forward

    def accelerated_forward(hidden_states):
        return head._old_forward(hidden_states)

    head.forward = accelerated_forward
    model = SimpleNamespace(lm_head=head)
    inputs = torch.randn(1, 8, 4)

    with last_position_lm_head(model, threshold=4):
        assert model.lm_head(inputs).shape == (1, 1, 7)

    assert model.lm_head(inputs).shape == (1, 8, 7)
    assert head._old_forward == original_forward


def test_video_token_count_uses_qwen_grid_and_merge_size():
    assert qwen2vl_video_token_count([[2, 14, 14]]) == 98
    assert qwen2vl_video_token_count([[1, 14, 14], [1, 14, 14]]) == 98


def test_calibration_is_nearest_and_marks_tolerance():
    point = choose_nearest_calibration("sample", 400, [("a", 360), ("b", 470)], tolerance=0.10)
    assert point.candidate_id == "a"
    assert point.status == "ok"
    point = choose_nearest_calibration("sample", 400, [("a", 500)], tolerance=0.10)
    assert point.status == "out_of_tolerance"


def test_adaptive_grid_covers_short_frames_and_dense_short_context_pixels():
    candidates = adaptive_candidate_grid()
    assert any(candidate.frames < 4 for candidate in candidates)
    short_pixels = {candidate.max_pixels for candidate in candidates if candidate.max_pixels <= 256 * 28 * 28}
    assert len(short_pixels) >= 6


def test_paired_cohort_requires_ok_at_every_target_and_minimum():
    rows = []
    for sample_id in ("a", "b", "c"):
        for target in (400, 3000):
            rows.append({
                "sample_id": sample_id,
                "target_visual_tokens": target,
                "actual_visual_tokens": target,
                "status": "ok",
            })
    rows[-1]["status"] = "out_of_tolerance"
    cohort = select_paired_cohort(rows, (400, 3000), minimum_samples=2)
    assert cohort.sample_ids == ("a", "b")
    assert cohort.valid
    assert "c" in cohort.invalid_by_target[3000]


def test_homogeneous_paired_cohort_prefers_exact_visual_length_signature():
    rows = []
    for sample_id, actual_by_target in {
        "a": (560, 2912),
        "b": (560, 2912),
        "c": (560, 2912),
        "d": (572, 2992),
    }.items():
        for target, actual in zip((400, 3000), actual_by_target):
            rows.append({
                "sample_id": sample_id,
                "calibration_target_visual_tokens": target,
                "actual_visual_tokens": actual,
                "calibration_status": "ok",
            })
    cohort = select_homogeneous_paired_cohort(rows, (400, 3000), minimum_samples=3)
    assert cohort.valid
    assert cohort.sample_ids == ("a", "b", "c")
    assert cohort.actual_visual_tokens == (560, 2912)


def test_figure2_homogeneous_audit_reports_exact_target_coverage():
    rows = []
    for sample_id in ("a", "b"):
        for target, actual in ((400, 572), (3000, 2912)):
            rows.append({
                "row_id": f"{sample_id}:{target}",
                "paper_figure": "Figure 2",
                "sample_id": sample_id,
                "target_model": DEFAULT_CONTRACT.msd_target_model,
                "temperature": 0.0,
                "target_visual_tokens": target,
                "calibration_target_visual_tokens": target,
                "actual_visual_tokens": actual,
                "calibration_status": "ok",
                "target_input_fingerprint": "same",
                "draft_input_fingerprint": "same",
                "attention_query": "last_instruction",
                "attention_key_scope": "strict_preceding",
                "query_position": 3,
                "instruction_positions": [0],
                "visual_positions": [1],
                "text_positions": [2],
                "modality": "summary",
                "attention_source": "msd_draft",
                "attention_policy": "last_instruction",
                "attention_weights": [0.2, 0.3, 0.5, 0.0],
            })

    report = audit_figure2_homogeneous(rows, DEFAULT_CONTRACT, (400, 3000), minimum_samples=2)

    assert report["valid"]
    assert report["checked_rows"] == 4
    assert report["coverage"]["paired_samples"] == 2
    assert report["coverage"]["targets"]["400"]["actual_visual_tokens"] == 572


def test_calibration_audit_reports_paired_coverage_without_promoting_outliers():
    rows = [
        {"sample_id": sample, "target_visual_tokens": target,
         "actual_visual_tokens": target, "status": "ok"}
        for sample in ("a", "b")
        for target in (400, 3000)
    ]
    rows[-1]["status"] = "out_of_tolerance"
    summary = audit_calibration(rows, (400, 3000), minimum_samples=1)
    assert summary["valid"]
    assert summary["paired_samples"] == 1
    assert summary["status_counts"]["3000"]["non_ok"] == 1


def test_layer_analysis_exposes_qwen2_contract_names():
    assert PreparedQwen2 is PreparedQwen25
    assert prepare_qwen2vl_prefill is prepare_qwen25_prefill


def test_metrics_are_deterministic_and_bounded():
    assert common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert rouge_l("a b c", "a b c") == 1.0
    assert 0.0 <= normalized_entropy([0.5, 0.5]) <= 1.0


def test_paper_statistics_report_n_mean_and_bootstrap_interval():
    summary = summarize([1, 2, 3], replicates=200, seed=7)
    assert summary["n"] == 3
    assert summary["mean"] == 2.0
    assert summary["ci95_low"] <= summary["mean"] <= summary["ci95_high"]

    rows = [
        {
            "paper_figure": "Figure 1(a)",
            "sample_id": "a",
            "calibration_target_visual_tokens": 400,
            "actual_visual_tokens": 392,
            "accepted_prefix_tokens": 4,
            "prefill_seconds": 1.0,
        },
        {
            "paper_figure": "Figure 1(a)",
            "sample_id": "b",
            "calibration_target_visual_tokens": 400,
            "actual_visual_tokens": 408,
            "accepted_prefix_tokens": 6,
            "prefill_seconds": 1.2,
        },
        {
            "paper_figure": "Figure 2",
            "modality": "summary",
            "sample_id": "a",
            "target_visual_tokens": 400,
            "attention_policy": "last_instruction",
            "visual_mass": 0.4,
            "text_mass": 0.5,
            "instruction_mass": 0.1,
            "visual_entropy": 0.8,
        },
        {
            "paper_figure": "Figure 3",
            "sample_id": "a",
            "layer_cut": 20,
            "prefix_agreement": 1.0,
            "rouge_l": 1.0,
            "native_answer_rouge_l": 0.9,
            "ablated_answer_rouge_l": 0.9,
            "answer_quality_delta": 0.0,
        },
    ]
    statistics = build_paper_statistics(rows, replicates=200, seed=7)
    assert statistics["figure1a"][0]["visual_tokens"] == 400
    assert statistics["figure1a"][0]["accepted_prefix_tokens"]["n"] == 2
    assert statistics["figure2"][0]["visual_tokens"] == 400
    assert statistics["figure2"][0]["visual_mass"]["mean"] == 0.4
    assert statistics["figure3a"][0]["answer_quality_delta"]["mean"] == 0.0


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


def test_retention_audit_allows_zero_visual_draft_tokens():
    row = _valid_row(
        paper_figure="Figure 1(b)",
        actual_visual_tokens=0,
        full_target_visual_tokens=400,
        retention_percentage=0.0,
    )
    assert audit_rows([row], DEFAULT_CONTRACT).valid


def test_attention_probe_selects_final_user_instruction_token():
    class Tokenizer:
        all_special_ids = [0, 100, 101, 102]

        def encode(self, text, add_special_tokens=False):
            return [100, 101]

    class Processor:
        tokenizer = Tokenizer()

    ids = torch.tensor([[0, 10, 151656, 151656, 20, 21, 100, 101, 102]])
    masks = find_instruction_masks(ids, Processor(), [2, 3])
    assert masks["attention_query"] == "last_instruction"
    assert masks["instruction_positions"] == [4, 5]
    assert masks["query_index"] == 5
    assert not (set(masks["instruction_positions"]) & set(masks["visual_positions"]))


def test_analysis_rows_require_their_provenance_and_use_contract_models():
    attention = _valid_row(
        row_id="sample:attention",
        paper_figure="Figure 2",
        target_visual_tokens=400,
        actual_visual_tokens=400,
        attention_query="last_instruction",
        query_position=2,
        instruction_positions=[2],
        visual_positions=[1],
        text_positions=[0, 3],
    )
    layer = _valid_row(
        row_id="sample:layer",
        paper_figure="Figure 3",
        target_model=DEFAULT_CONTRACT.layer_target_model,
        visual_kv_masked_from=20,
        layer_cut=20,
        native_answer_rouge_l=0.8,
        ablated_answer_rouge_l=0.79,
        answer_quality_delta=-0.01,
    )
    retention = _valid_row(
        row_id="sample:cosine",
        paper_figure="Figure 6 / Appendix D",
        target_model=DEFAULT_CONTRACT.layer_target_model,
        layer=20,
        visual_cosine=0.2,
        text_cosine=0.8,
    )
    report = audit_rows([attention, layer, retention], DEFAULT_CONTRACT)
    assert report.valid


def test_audit_accepts_strict_preceding_attention_without_query_in_mask():
    row = _valid_row(
        row_id="strict-attention",
        paper_figure="Figure 2",
        attention_query="last_instruction",
        attention_key_scope="strict_preceding",
        query_position=3,
        instruction_positions=[2],
        visual_positions=[1],
        text_positions=[0],
    )
    assert audit_rows([row], DEFAULT_CONTRACT).valid


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


def test_layer_audit_requires_vdc_answer_quality_metrics():
    row = _valid_row(
        row_id="sample:layer-quality",
        paper_figure="Figure 3",
        target_model=DEFAULT_CONTRACT.layer_target_model,
        visual_kv_masked_from=20,
        layer_cut=20,
    )
    report = audit_rows([row], DEFAULT_CONTRACT)
    assert not report.valid
    assert any(issue.code == "missing_answer_quality" for issue in report.issues)


def test_layer_audit_rejects_out_of_range_vdc_answer_quality_metrics():
    row = _valid_row(
        row_id="sample:layer-quality-range",
        paper_figure="Figure 3",
        target_model=DEFAULT_CONTRACT.layer_target_model,
        visual_kv_masked_from=20,
        layer_cut=20,
        native_answer_rouge_l=1.2,
        ablated_answer_rouge_l=0.8,
        answer_quality_delta=-0.4,
    )
    report = audit_rows([row], DEFAULT_CONTRACT)
    assert not report.valid
    assert any(issue.code == "invalid_answer_quality" for issue in report.issues)


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


def test_retention_compacts_only_the_draft_and_removes_markers_at_zero():
    prepared = PreparedPrefill(
        input_ids=torch.tensor([[10, 151652, 151656, 151656, 151653, 20]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        inputs_embeds=torch.randn(1, 6, 4),
        position_ids=torch.arange(6).view(1, 1, 6).expand(3, 1, 6),
        rope_deltas=torch.tensor([0]),
        video_grid_thw=None,
        video_positions=torch.tensor([2, 3]),
        input_fingerprint="full",
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    half = compact_qwen2vl_prefill(prepared, 50)
    empty = compact_qwen2vl_prefill(prepared, 0)
    assert half.input_ids.shape[1] == 5
    assert half.video_positions.tolist() == [2]
    assert empty.input_ids.tolist() == [[10, 20]]
    assert empty.video_positions.tolist() == []


def test_losslessness_audit_warns_on_late_near_tie():
    from src.analyze.Validate_Sparrow_hypothesises.audit import audit_losslessness

    rows = [
        {"row_id": "late", "target_output_ids": [1, 2, 3, 4, 5, 6, 7, 8],
         "speculative_output_ids": [1, 2, 3, 4, 5, 6, 7, 9]},
        {"row_id": "early", "target_output_ids": [1, 2, 3, 4],
         "speculative_output_ids": [1, 9, 3, 4]},
        {"row_id": "exact", "target_output_ids": [1, 2, 3],
         "speculative_output_ids": [1, 2, 3]},
    ]
    report = audit_losslessness(rows)
    codes = {(issue.code, issue.severity) for issue in report.issues}
    assert ("near_tie_divergence", "warning") in codes
    assert ("lossless_mismatch", "error") in codes
    # The early divergence still fails the gate; the late one is only a warning.
    assert not report.valid


def test_losslessness_audit_allows_longer_speculative_tail():
    from src.analyze.Validate_Sparrow_hypothesises.audit import audit_losslessness

    rows = [{"row_id": "tail", "target_output_ids": [1, 2, 3],
             "speculative_output_ids": [1, 2, 3, 4]}]
    assert audit_losslessness(rows).valid


def test_audit_allows_zero_draft_tokens_for_remove_all_series():
    row = _valid_row(
        row_id="remove-all",
        paper_figure="Figure 1(a)",
        series_id="msd_remove_all",
        actual_visual_tokens=0,
        condition="retention",
        retention_percentage=0.0,
    )
    assert audit_rows([row], DEFAULT_CONTRACT).valid
