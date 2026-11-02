from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.analyze.Validate_Sparrow_hypothesises import run_dflash_visualcut_attention as runner
from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_visualcut_attention import (
    _read_expected_case_ids,
)
from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_visualcut_attention import (
    _attach_reference_quality,
    _quality_fields,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import (
    _resolve_reference_answer,
)


def test_worker_metadata_reads_requested_generation_limit(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "_git_state", lambda _repo: {"revision": "test"})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H200")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: "H200 test")
    args = SimpleNamespace(
        repo_root=tmp_path,
        device="cpu",
        checkpoint_name="dflash_test",
        target_model="target",
        manifest=tmp_path / "manifest.jsonl",
        calibration=tmp_path / "calibration.jsonl",
        cohort_selection=tmp_path / "cohort.json",
        output_dir=tmp_path / "run",
        visual_tokens=400,
        requested_max_new_tokens=256,
        dtype="bf16",
        target_attention="sdpa",
    )

    metadata = runner._worker_metadata(
        args,
        "dflash_test_v400",
        tmp_path / "checkpoint",
        tmp_path / "config.json",
    )

    assert metadata["max_new_tokens"] == 256
    assert metadata["gpu_name"] == "NVIDIA H200"


def test_campaign_case_matrix_covers_chunked_balanced_run_and_limits():
    records = [
        {
            "sample_id": f"sample-{index}",
            "calibration_by_target": {
                target: {"status": "ok"} for target in runner.TARGETS
            },
        }
        for index in range(1, 5)
    ]

    expected = runner._campaign_expected_case_ids(
        records,
        {record["sample_id"] for record in records},
        ["dflash_3layer_1_9_17_25_33"],
        num_chunks=2,
        sample_limit=1,
    )

    assert len(expected) == len(runner.TARGETS) * 2 * len(runner.CONDITIONS)
    assert "dflash_3layer_1_9_17_25_33:sample-1:400:full" in expected
    assert "dflash_3layer_1_9_17_25_33:sample-3:400:full" in expected
    assert "dflash_3layer_1_9_17_25_33:sample-2:400:full" not in expected


def test_runner_jsonl_reader_repairs_an_incomplete_final_line(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"row_id":"complete"}\n{"row_id":', encoding="utf-8")

    rows = runner._read_jsonl(path)

    assert rows == [{"row_id": "complete"}]
    assert path.read_text(encoding="utf-8") == '{"row_id":"complete"}\n'


def test_analyzer_reads_explicit_expected_case_matrix(tmp_path: Path):
    expected_ids = ["balanced:sample-1:400:full", "balanced:sample-1:400:cut"]
    (tmp_path / "run_config.json").write_text(
        '{"expected_case_ids": ["balanced:sample-1:400:full", '
        '"balanced:sample-1:400:cut"], "expected_row_count": 2}',
        encoding="utf-8",
    )

    assert _read_expected_case_ids(tmp_path) == set(expected_ids)

    (tmp_path / "run_config.json").write_text(
        '{"expected_case_ids": ["balanced:sample-1:400:full"], '
        '"expected_row_count": 2}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_row_count"):
        _read_expected_case_ids(tmp_path)


def test_reference_answer_falls_back_when_manifest_reference_is_null():
    reference, source = _resolve_reference_answer(
        {"reference_answer": None, "answer": "A person walks by a truck."}
    )

    assert reference == "A person walks by a truck."
    assert source == "answer"


def test_quality_fields_keep_nonlossless_quality_and_report_delta():
    fields = _quality_fields(
        {
            "reference_answer": "A person walks by a truck.",
            "quality_metrics": {
                "reference_source": "answer",
                "target": {"bleu": 0.8, "rouge_l": 0.9, "coverage": 1.0},
                "speculative": {"bleu": 0.7, "rouge_l": 0.8, "coverage": 0.9},
            },
            "lossless": False,
        }
    )

    assert fields["quality_reference_available"] is True
    assert abs(fields["rouge_l_delta_vs_target"] + 0.1) < 1e-9
    assert abs(fields["coverage_delta_vs_target"] + 0.1) < 1e-9


def test_reference_quality_can_be_backfilled_for_pilot_journal_rows():
    rows = [
        {
            "sample_id": "video-1",
            "target_output_text": "A dog runs through grass.",
            "speculative_output_text": "A dog runs through grass.",
        }
    ]

    _attach_reference_quality(rows, {"video-1": "A dog runs through grass."})

    assert rows[0]["reference_answer"] == "A dog runs through grass."
    assert rows[0]["quality_metrics"]["reference_source"] == "run_manifest"
    assert rows[0]["quality_metrics"]["speculative"]["rouge_l"] == 1.0
