from __future__ import annotations

import json
from pathlib import Path

from src.infer.build_dflash_infer_report import (
    RunSpec,
    aggregate_run,
    build_report,
    compare_hashes,
)


LLAVA = "qwen25vl-3b-dflash-llava68k-latest"
SHAREGPT = "qwen25vl-3b-dflash-sharegpt68k-latest"


def _write_run(
    directory: Path,
    *,
    dataset: str = "mvbench",
    condition: str = "full",
    sample_count: int = 2,
    second_prediction: str = "B",
) -> None:
    directory.mkdir()
    samples = []
    for index in range(sample_count):
        samples.append(
            {
                "sample_index": index,
                "sample_id": f"task:{index:06d}",
                "task": "action_prediction",
                "dataset_format": dataset,
                "device": "cuda:0",
                "dtype": "torch.bfloat16",
                "visual_ablation": {
                    "enabled": condition != "full",
                    "mode": "zero" if condition != "cut" else "cut",
                    "requested_layer_ids": [25, 33] if condition == "exp2_zero" else [],
                },
                "preprocessing": {"num_frames": 8, "video_min_pixels": 50176, "video_max_pixels": 50176},
                "target_baseline": {
                    "output_hash": f"target-{index}",
                    "end_to_end_s": 1.0,
                    "task_metrics": {"correct": index == 0, "target_option": "A"},
                    "text_metrics": {"bleu": 0.5, "rouge_l": 0.4, "coverage": 0.3},
                },
                "checkpoints": [
                    {
                        "label": LLAVA,
                        "status": "ok" if index == 0 else "mismatch",
                        "outputs_match": index == 0,
                        "speculative_output_hash": f"target-{index}" if index == 0 else f"llava-{index}",
                        "task_metrics": {"correct": index == 0},
                        "text_metrics": {"bleu": 0.45, "rouge_l": 0.35, "coverage": 0.25},
                        "timing": {
                            "tokens_per_second": 10.0 + index,
                            "end_to_end_s": 0.5 + index,
                            "speedup_vs_target": 2.0,
                        },
                        "acceptance": {"tau": 2.0, "mean_accepted_proposals": 1.0},
                        "speedup": {"esr": 2.0, "dsr": 1.8},
                    },
                    {
                        "label": SHAREGPT,
                        "status": "ok" if index == 0 else "mismatch",
                        "outputs_match": index == 0,
                        "speculative_output_hash": f"target-{index}" if index == 0 else f"sharegpt-{index}",
                        "task_metrics": {"correct": index == 0},
                        "text_metrics": {"bleu": 0.44, "rouge_l": 0.34, "coverage": 0.24},
                        "timing": {
                            "tokens_per_second": 9.0 + index,
                            "end_to_end_s": 0.6 + index,
                            "speedup_vs_target": 1.8,
                        },
                        "acceptance": {"tau": 1.8, "mean_accepted_proposals": 0.8},
                        "speedup": {"esr": 1.8, "dsr": 1.6},
                    },
                ],
            }
        )
        (directory / f"sample_{index:03d}.json").write_text(json.dumps(samples[-1]), encoding="utf-8")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "total_samples": sample_count,
                "completed_samples": sample_count,
                "lossless_samples": 1,
                "mismatch_samples": sample_count - 1,
                "runtime_errors": 0,
                "run_completed": True,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_run_computes_coverage_accuracy_and_performance(tmp_path: Path) -> None:
    run_dir = tmp_path / "mvbench_full"
    _write_run(run_dir)

    result = aggregate_run(run_dir, RunSpec("mvbench", "Full visual", "full", run_dir, 16))

    expected_coverage = {
        "total": 2,
        "completed": 2,
        "lossless": 1,
        "mismatch": 1,
        "runtime_errors": 0,
        "sample_file_count": 2,
        "sample_ids_unique": True,
        "sample_indices_contiguous": True,
        "summary_consistent": True,
    }
    assert {key: result["coverage"][key] for key in expected_coverage} == expected_coverage
    assert result["coverage"]["lossless_by_checkpoint"][LLAVA] == 1
    assert result["checkpoints"][LLAVA]["accuracy"] == 0.5
    assert result["checkpoints"][LLAVA]["lossless_rate"] == 0.5
    assert result["checkpoints"][LLAVA]["performance"]["tokens_per_second"] == 10.5
    assert result["task_accuracy"][LLAVA]["action_prediction"] == 0.5
    assert result["config"]["mode"] == "full"


def test_aggregate_run_uses_sample_truth_when_summary_is_stale(tmp_path: Path) -> None:
    run_dir = tmp_path / "stale_summary"
    _write_run(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"lossless_samples": 2, "mismatch_samples": 0})
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = aggregate_run(run_dir, RunSpec("mvbench", "Full visual", "full", run_dir, 16))

    assert result["coverage"]["lossless"] == 1
    assert result["coverage"]["mismatch"] == 1
    assert result["coverage"]["source_summary_lossless"] == 2
    assert result["coverage"]["summary_consistent"] is False


def test_compare_hashes_counts_target_and_checkpoint_changes(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    condition_dir = tmp_path / "condition"
    _write_run(reference_dir)
    _write_run(condition_dir, condition="exp2_zero", second_prediction="C")

    reference = aggregate_run(reference_dir, RunSpec("mvbench", "Full visual", "full", reference_dir, 16))
    condition = aggregate_run(condition_dir, RunSpec("mvbench", "EXP2 zero", "exp2_zero", condition_dir, 16))

    changed = compare_hashes(reference["samples"], condition["samples"], [LLAVA, SHAREGPT])

    assert changed == {"target": 0, LLAVA: 0, SHAREGPT: 0}

    condition["samples"][1]["target_baseline"]["output_hash"] = "new-target"
    condition["samples"][1]["checkpoints"][0]["speculative_output_hash"] = "new-llava"
    changed = compare_hashes(reference["samples"], condition["samples"], [LLAVA, SHAREGPT])
    assert changed == {"target": 1, LLAVA: 1, SHAREGPT: 0}


def test_build_report_writes_report_tables_and_figures(tmp_path: Path) -> None:
    full_dir = tmp_path / "mvbench_full"
    zero_dir = tmp_path / "mvbench_zero"
    _write_run(full_dir)
    _write_run(zero_dir, condition="exp2_zero")
    specs = [
        RunSpec("mvbench", "Full visual", "full", full_dir, 16),
        RunSpec("mvbench", "EXP2 zero", "exp2_zero", zero_dir, 16),
    ]

    build_report(specs, tmp_path / "report")

    output = tmp_path / "report"
    assert (output / "REPORT.md").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "scenario_summary.csv").is_file()
    assert (output / "performance.csv").is_file()
    assert (output / "figure1_lossless_rate.png").is_file()
    assert (output / "figure2_mvbench_accuracy.png").is_file()
    assert (output / "figure3_latency_throughput.png").is_file()
    assert (output / "figure4_vdc_quality_ablation.png").is_file()
    assert (output / "figure5_speculative_timing_speedup.png").is_file()
    assert (output / "figure6_speculative_acceptance_speedup.png").is_file()
    assert (output / "speculative_metrics.csv").is_file()
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    assert "CONFIRMED" in report
    assert "EXPLORATORY" in report
    assert "Performance" in report
    assert "Speculative decoding metrics" in report
    assert "Speedup vs target (×)" in report
    assert "figure5_speculative_timing_speedup.png" in report
    assert "figure6_speculative_acceptance_speedup.png" in report


def test_build_report_embeds_figures_at_relevant_sections(tmp_path: Path) -> None:
    full_dir = tmp_path / "mvbench_full"
    _write_run(full_dir)

    build_report(
        [RunSpec("mvbench", "Full visual", "full", full_dir, 16)],
        tmp_path / "report",
    )

    report = (tmp_path / "report" / "REPORT.md").read_text(encoding="utf-8")
    figure1 = report.index("![Figure 1")
    figure2 = report.index("![Figure 2")
    figure3 = report.index("![Figure 3")
    figure4 = report.index("![Figure 4")
    mvbench_section = report.index("## 1. Kết quả theo từng kịch bản — MVBench")
    first_mvbench_scenario = report.index("### 1.1")
    performance_section = report.index("## 2. So sánh performance theo dataset")
    vdc_section = report.index("### VDC50", performance_section)
    confirmed_section = report.index("## 3. CONFIRMED")

    assert figure1 < mvbench_section
    assert mvbench_section < figure2 < first_mvbench_scenario
    assert performance_section < figure3 < report.index("### MVBench", performance_section)
    assert vdc_section < figure4 < confirmed_section
