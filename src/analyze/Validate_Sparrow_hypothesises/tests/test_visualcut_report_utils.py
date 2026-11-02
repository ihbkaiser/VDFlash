from __future__ import annotations

from src.analyze.Validate_Sparrow_hypothesises.visualcut_report_utils import (
    extract_case_metrics,
    is_usable_case,
    sample_evenly,
    summarize_cases,
)
from src.analyze.Validate_Sparrow_hypothesises.compute_config_reports_data import (
    _latest_case_rows,
)


def test_report_contrasts_pair_each_condition_to_the_full_sample_set():
    all_data = {
        ("config", 400, "full", "sample-1"): {"tau_effective": 2.0, "lossless": True},
        ("config", 400, "full", "sample-2"): {"tau_effective": 4.0, "lossless": True},
        ("config", 400, "cut", "sample-1"): {"tau_effective": 3.0, "lossless": False},
    }

    condition_stats, contrasts = summarize_cases(
        all_data,
        ["config"],
        [400],
        ["full", "cut"],
    )

    assert condition_stats["config"]["400_full"]["tau_effective"]["n"] == 2
    assert condition_stats["config"]["400_cut"]["tau_effective"]["n"] == 1
    contrast = contrasts["config"]["400_cut_minus_full"]["tau_effective"]
    assert contrast["n"] == 1
    assert contrast["mean"] == 1.0


def test_report_metrics_skip_failed_capture_and_preserve_nonlossless_case():
    nonlossless = {
        "status": "mismatch",
        "capture_and_timing_outputs_match": True,
        "lossless": False,
        "metrics": {"tau_effective": 2.0},
    }
    capture_mismatch = {
        "status": "mismatch",
        "capture_and_timing_outputs_match": False,
        "lossless": True,
    }

    assert is_usable_case(nonlossless) is True
    assert extract_case_metrics(nonlossless)["tau_effective"] == 2.0
    assert is_usable_case(capture_mismatch) is False


def test_sample_evenly_covers_both_ends_of_the_shard_list():
    items = list(range(10))

    selected = sample_evenly(items, 4)

    assert selected == [0, 3, 6, 9]


def test_report_loader_uses_the_latest_row_for_a_case(tmp_path):
    journal = tmp_path / "cases.jsonl"
    journal.write_text(
        '{"checkpoint_name":"config","target_visual_tokens":400,'
        '"visual_condition":"full","sample_id":"sample-1", "status":"ok"}\n'
        '{"checkpoint_name":"config","target_visual_tokens":400,'
        '"visual_condition":"full","sample_id":"sample-1", "status":"error"}\n',
        encoding="utf-8",
    )

    latest = _latest_case_rows([journal])

    assert latest[("config", 400, "full", "sample-1")]["status"] == "error"
    assert is_usable_case(latest[("config", 400, "full", "sample-1")]) is False
