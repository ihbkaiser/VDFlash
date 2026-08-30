from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infer.build_dflash_epoch_comparison_report import (
    EpochSpec,
    build_report,
    compare_epoch_bundles,
    load_epoch_bundle,
)


def _checkpoint(label: str, *, match: bool, tau: float, e2e: float, correct: bool) -> dict:
    family = "llava68k" if "llava" in label else "sharegpt68k"
    return {
        "label": label,
        "status": "ok" if match else "mismatch",
        "outputs_match": match,
        "speculative_output_hash": f"{family}-same" if match else f"{family}-{label}",
        "task_metrics": {"correct": correct},
        "text_metrics": {"bleu": 0.40 if correct else 0.20, "rouge_l": 0.50 if correct else 0.30},
        "acceptance": {"tau": tau},
        "speedup": {"esr": 1.5 if tau > 2 else 1.1, "dsr": 1.4 if tau > 2 else 1.0},
        "timing": {"tokens_per_second": 20.0 if tau > 2 else 15.0, "end_to_end_s": e2e},
    }


def _write_run(directory: Path, *, epoch: str, dataset: str, device: str, duplicate: bool = False) -> None:
    directory.mkdir(parents=True)
    checkpoint_prefix = "qwen25vl-3b-dflash-20e-" if epoch == "20e" else "qwen25vl-3b-dflash-"
    labels = (
        f"{checkpoint_prefix}llava68k-latest",
        f"{checkpoint_prefix}sharegpt68k-latest",
    )
    count = 2
    samples = []
    for index in range(count):
        sample_id = "duplicate" if duplicate else f"sample:{index:03d}"
        samples.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "task": "action_prediction",
                "device": device,
                "dtype": "torch.bfloat16",
                "preprocessing": {"num_frames": 8},
                "target_baseline": {
                    "output_hash": f"target-{index}",
                    "end_to_end_s": 1.0,
                    "task_metrics": {"correct": index == 0},
                    "text_metrics": {"bleu": 0.50, "rouge_l": 0.60},
                },
                "checkpoints": [
                    _checkpoint(labels[0], match=index == 0, tau=2.0, e2e=0.8, correct=index == 0),
                    _checkpoint(labels[1], match=index == 0, tau=1.5, e2e=0.9, correct=index == 0),
                ],
            }
        )
        (directory / f"sample_{index:03d}.json").write_text(json.dumps(samples[-1]), encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "total_samples": count,
                "completed_samples": count,
                "lossless_samples": 1,
                "mismatch_samples": 1,
                "runtime_errors": 0,
                "run_completed": True,
            }
        ),
        encoding="utf-8",
    )


def _spec(root: Path, epoch: str, *, vdc_device: str = "cuda:0", duplicate: bool = False) -> EpochSpec:
    mvbench = root / epoch / "mvbench"
    vdc = root / epoch / "vdc"
    _write_run(mvbench, epoch=epoch, dataset="mvbench", device="cuda:0", duplicate=duplicate)
    _write_run(vdc, epoch=epoch, dataset="vdc50", device=vdc_device, duplicate=duplicate)
    return EpochSpec(epoch=epoch, mvbench_dir=mvbench, vdc50_dir=vdc)


def test_compare_epoch_bundles_pairs_ids_and_computes_deltas(tmp_path: Path) -> None:
    old = load_epoch_bundle(_spec(tmp_path, "6e"))
    new = load_epoch_bundle(_spec(tmp_path, "20e", vdc_device="cuda:1"))

    result = compare_epoch_bundles(old, new)

    mvbench = result["datasets"]["mvbench"]
    vdc = result["datasets"]["vdc50"]
    assert mvbench["paired_samples"] == 2
    assert mvbench["performance_comparable"] is True
    assert mvbench["checkpoints"]["llava68k"]["6e"]["lossless_rate"] == 0.5
    assert mvbench["checkpoints"]["llava68k"]["20e"]["lossless_rate"] == 0.5
    assert mvbench["checkpoints"]["llava68k"]["delta"]["tau"] == 0.0
    assert vdc["performance_comparable"] is False
    assert vdc["target_hash_changes"] == 0


def test_load_epoch_bundle_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "6e", duplicate=True)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_epoch_bundle(spec)


def test_build_report_emits_tables_and_six_figure_pairs(tmp_path: Path) -> None:
    specs = [
        _spec(tmp_path, "6e"),
        _spec(tmp_path, "20e", vdc_device="cuda:1"),
    ]

    build_report(specs, tmp_path / "report")

    output = tmp_path / "report"
    assert (output / "REPORT.md").is_file()
    assert (output / "summary.json").is_file()
    for name in ("epoch_comparison.csv", "performance.csv", "task_accuracy.csv", "paired_changes.csv"):
        assert (output / name).is_file()
    for figure in (
        "figure1_epoch_overview",
        "figure2_mvbench_task_accuracy",
        "figure3_mvbench_performance",
        "figure4_vdc_quality",
        "figure5_vdc_performance",
        "figure6_epoch_delta_heatmap",
    ):
        assert (output / f"{figure}.png").is_file()
        assert (output / f"{figure}.pdf").is_file()
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "cross-GPU" in report
    assert "figure6_epoch_delta_heatmap.png" in report
