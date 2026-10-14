from __future__ import annotations

import json

from src.analyze.Validate_Sparrow_hypothesises.reverse_transfer_report import build_report


def row(sample_id: str, condition: str, accuracy: float) -> dict:
    order = "video_text" if condition in {"E1", "E3", "E5"} else "text_video"
    conditioning = {"E1": "full", "E2": "full", "E3": "visual", "E4": "visual", "E5": "text", "E6": "text"}[condition]
    return {
        "status": "ok",
        "evidence_eligible": True,
        "dataset": "mvbench",
        "sample_id": sample_id,
        "video_id": sample_id,
        "task_type": "toy",
        "condition": condition,
        "input_order": order,
        "msd_conditioning": conditioning,
        "permutation_id": "identity",
        "accuracy": accuracy,
        "correct_option_probability": 0.8,
        "nll": 0.2,
        "margin": 1.0,
        "prediction_entropy": 0.5,
    }


def test_report_filters_ineligible_rows_and_writes_outputs(tmp_path):
    rows = []
    for sample_id, value in (("v0", 1.0), ("v1", 0.0)):
        rows.extend([row(sample_id, "E3", value), row(sample_id, "E4", min(1.0, value + 0.0))])
    rows.append({**row("bad", "E4", 1.0), "evidence_eligible": False})
    payload = build_report(rows, tmp_path, bootstrap_repeats=20)
    assert payload["raw_rows"] == 5
    assert payload["evidence_rows"] == 4
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "paired_contrasts.csv").exists()
    assert (tmp_path / "REPORT.md").exists()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["prediction_source_guard"].startswith("Primary rows")


def test_report_excludes_label_permutations_from_primary_summaries(tmp_path):
    identity = row("v0", "E3", 1.0)
    permutation = {**identity, "permutation_id": "swap01", "accuracy": 0.0}

    payload = build_report([identity, permutation], tmp_path, bootstrap_repeats=20)

    assert payload["raw_rows"] == 2
    assert payload["primary_identity_rows"] == 1
    assert payload["evidence_rows"] == 2
    primary = [summary for summary in payload["summaries"] if summary["condition"] == "E3"]
    assert len(primary) == 1
    assert primary[0]["samples"] == 1
