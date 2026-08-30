from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch

from src.analyze.Validate_Sparrow_hypothesises.dflash_audit import audit_dflash_rows
from src.analyze.Validate_Sparrow_hypothesises.dflash_report import write_dflash_report
from src.analyze.Validate_Sparrow_hypothesises import run_dflash_experiments as orchestrator
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import (
    DFLASH_STAGE_ORDER,
    _calibration_settings,
    _stage_row_key,
    _stage_row_sink,
    _read_stage_rows,
    build_parser,
    default_output_dir,
    run_dflash_experiments,
)
from src.analyze.Validate_Sparrow_hypothesises.dataset import VideoSample


def test_calibration_settings_can_allow_out_of_tolerance_points():
    sample = {
        "calibration_by_target": {
            400: {
                "status": "out_of_tolerance",
                "candidate_settings": {"frames": 4, "max_pixels": 200704},
            }
        }
    }

    settings, error = _calibration_settings(
        sample,
        400,
        allow_out_of_tolerance=True,
    )

    assert error is None
    assert settings == {"frames": 4, "max_pixels": 200704}


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


def test_audit_separates_runtime_errors_from_unsupported_rows():
    audit = audit_dflash_rows(
        [
            _row("length_sweep", status="error", error="OOM"),
            _row("length_sweep", status="unsupported", error="not calibrated"),
        ]
    )

    assert len(audit["error_rows"]) == 1
    assert len(audit["unsupported_rows"]) == 1
    assert audit["error_rows"][0]["error"] == "OOM"
    assert audit["coverage_valid"] is False


def test_audit_counts_mismatch_and_unsupported_in_main_result_statistics():
    audit = audit_dflash_rows(
        [
            _row(
                "length_sweep",
                status="mismatch",
                target_output_ids=[1],
                speculative_output_ids=[2],
            ),
            _row("length_sweep", status="unsupported", error="not calibrated"),
            _row("length_sweep", status="error", error="OOM"),
        ]
    )

    assert audit["result_rows"] == 3
    assert audit["result_status_counts"] == {
        "error": 1,
        "mismatch": 1,
        "unsupported": 1,
    }
    assert audit["experiment_counts"] == {"length_sweep": 3}
    assert audit["valid_rows"] == 1


def test_report_exposes_all_result_statuses_in_primary_summary(tmp_path):
    rows = [
        _row(
            "length_sweep",
            status="mismatch",
            target_output_ids=[1],
            speculative_output_ids=[2],
        ),
        _row("length_sweep", status="unsupported", error="not calibrated"),
    ]

    report_path = write_dflash_report(rows, tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert "- Result rows: **2**" in report
    assert "- Mismatch rows: **1**" in report
    assert "- Unsupported rows: **1**" in report


def test_dflash_parser_defaults_are_isolated_and_stage_order_is_stable():
    args = build_parser().parse_args(["--dry-run"])

    assert args.checkpoint.endswith("training_state.pt")
    assert args.target_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert DFLASH_STAGE_ORDER == ("length", "retention", "attention", "layers", "report")
    assert Path(default_output_dir()).name.startswith("sparrow_validation_dflash_qwen25vl3b_")


def test_dflash_parser_accepts_model_parallel_target_placement():
    args = build_parser().parse_args(
        [
            "length",
            "--device-map",
            "model_parallel",
            "--max-memory",
            "0:22GiB,1:14GiB",
        ]
    )

    assert args.device_map == "model_parallel"
    assert args.max_memory == "0:22GiB,1:14GiB"


def test_layer_stage_keeps_flash_backend_for_masked_generation(monkeypatch, tmp_path):
    class FakeTarget:
        config = SimpleNamespace(_attn_implementation="flash_attention_2")

    target = FakeTarget()
    sample = VideoSample(
        video_name="sample-0",
        question="question",
        answer="answer",
        local_video_path="sample.mp4",
    )

    monkeypatch.setattr(
        "src.analyze.Validate_Sparrow_hypothesises.dataset.load_vdc_manifest",
        lambda *_args, **_kwargs: [sample],
    )
    monkeypatch.setattr(
        "src.infer.qwen25vl_dflash_compare._load_target",
        lambda *_args, **_kwargs: (object(), target, 0.0),
    )
    monkeypatch.setattr(
        "src.infer.qwen25vl_dflash_compare._load_draft",
        lambda *_args, **_kwargs: (object(), 0.0),
    )
    prompt = SimpleNamespace(inputs={"input_ids": torch.tensor([[1, 2]])})
    monkeypatch.setattr(
        orchestrator,
        "_prepare_prompt",
        lambda **_kwargs: (prompt, [0], "fingerprint", {"frames": 4}),
    )

    seen_backend = []

    def probe(**_kwargs):
        seen_backend.append(target.config._attn_implementation)
        assert target.config._attn_implementation == "flash_attention_2"
        return [{"experiment": "qwen25vl_target_visual_kv", "layer_index": 0}]

    monkeypatch.setattr(orchestrator, "_target_side_probe", probe)
    args = build_parser().parse_args(
        [
            "layers",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
            "--layer-visual-targets",
            "3000",
            "--layer-cut-points",
            "0",
            "--allow-out-of-tolerance",
        ]
    )

    orchestrator.run_dflash_experiments(args)
    assert seen_backend == ["flash_attention_2"]


def test_two_gpu_dflash_launcher_declares_model_parallel_contract():
    launcher = Path(
        "src/analyze/Validate_Sparrow_hypothesises/"
        "run_dflash_model_parallel_2gpu.sh"
    )

    assert launcher.is_file()
    contents = launcher.read_text(encoding="utf-8")
    assert 'GPUS="${GPUS:-0,1}"' in contents
    assert "--device-map model_parallel" in contents
    assert 'MAX_MEMORY="${MAX_MEMORY:-0:22GiB,1:14GiB}"' in contents
    assert "calibration_complete_20260823_currentenv.jsonl" in contents
    assert "--allow-out-of-tolerance" in contents
    assert "--dry-run" in contents


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
