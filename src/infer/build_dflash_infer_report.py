"""Build a reproducible report from completed DFlash inference batches.

The module is intentionally CPU-only: it reads the per-sample JSON artifacts
written by :mod:`qwen25vl_dflash_compare`, aggregates them, and renders a
Vietnamese Markdown report with PNG/PDF figures. It never loads a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from src.workspace import resolve_workspace_path, workspace_path


LLAVA_CHECKPOINT = "qwen25vl-3b-dflash-llava68k-latest"
SHAREGPT_CHECKPOINT = "qwen25vl-3b-dflash-sharegpt68k-latest"
CHECKPOINTS = (LLAVA_CHECKPOINT, SHAREGPT_CHECKPOINT)
DEFAULT_OUTPUT_DIR = workspace_path(
    "results", "infer", "dflash_inference_report_2026-08-25"
)


@dataclass(frozen=True)
class RunSpec:
    """Metadata needed to identify one completed benchmark condition."""

    dataset: str
    display_name: str
    condition: str
    path: Path
    output_budget: int


def default_run_specs(root: Path | None = None) -> list[RunSpec]:
    """Return the eight primary MVBench/VDC50 conditions in report order."""

    root = resolve_workspace_path(root or "results/infer")
    return [
        RunSpec("mvbench", "Full visual", "full", root / "mvbench100_full_20260823", 16),
        RunSpec("mvbench", "EXP2 zero (L25,33)", "exp2_zero", root / "mvbench100_exp2_zero_20260823", 16),
        RunSpec(
            "mvbench",
            "EXP1 zero (L1,9,17,25,33)",
            "exp1_zero",
            root / "mvbench100_exp1_zero_20260823",
            16,
        ),
        RunSpec(
            "mvbench",
            "EXP1 cut (L1,9,17,25,33)",
            "exp1_cut",
            root / "mvbench100_exp1_cut_20260823",
            16,
        ),
        RunSpec("vdc50", "Full visual", "full", root / "vdc50_exp_full_dflash_20260823", 256),
        RunSpec("vdc50", "EXP2 zero (L25,33)", "exp2_zero", root / "vdc50_exp2_high_zero_20260823", 256),
        RunSpec(
            "vdc50",
            "EXP1 zero (L1,9,17,25,33)",
            "exp1_zero",
            root / "vdc50_exp1_all_zero_20260823",
            256,
        ),
        RunSpec(
            "vdc50",
            "EXP1 cut (L1,9,17,25,33)",
            "exp1_cut",
            root / "vdc50_exp1_all_cut_20260823",
            256,
        ),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_samples(directory: Path) -> list[dict[str, Any]]:
    """Read and sort all per-sample reports in a batch directory."""

    paths = sorted(directory.glob("sample_*.json"))
    if not paths:
        raise FileNotFoundError(f"no sample JSON files found in {directory}")
    samples = [_read_json(path) for path in paths]
    return sorted(samples, key=lambda row: int(row.get("sample_index", 0)))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    return fmean(numbers) if numbers else None


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _bool_mean(values: Iterable[Any]) -> float | None:
    booleans = [value for value in values if isinstance(value, bool)]
    return fmean(booleans) if booleans else None


def _checkpoint_rows(samples: Sequence[Mapping[str, Any]], label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for sample in samples:
        for checkpoint in sample.get("checkpoints", []):
            if isinstance(checkpoint, Mapping) and checkpoint.get("label") == label:
                rows.append(checkpoint)
                break
    return rows


def _task_accuracy(samples: Sequence[Mapping[str, Any]], label: str) -> dict[str, float]:
    values: dict[str, list[bool]] = defaultdict(list)
    for sample in samples:
        task = str(sample.get("task", "unknown"))
        for checkpoint in sample.get("checkpoints", []):
            if checkpoint.get("label") == label:
                correct = _nested(checkpoint, "task_metrics", "correct")
                if isinstance(correct, bool):
                    values[task].append(correct)
                break
    return {task: fmean(results) for task, results in sorted(values.items()) if results}


def _target_task_accuracy(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, list[bool]] = defaultdict(list)
    for sample in samples:
        correct = _nested(sample, "target_baseline", "task_metrics", "correct")
        if isinstance(correct, bool):
            values[str(sample.get("task", "unknown"))].append(correct)
    return {task: fmean(results) for task, results in sorted(values.items()) if results}


def _config_from_samples(spec: RunSpec, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = samples[0]
    ablation = first.get("visual_ablation") or {}
    preprocessing = first.get("preprocessing") or {}
    layers = ablation.get("requested_layer_ids") or []
    mode = "full" if spec.condition == "full" else ablation.get("mode", "unknown")
    return {
        "dataset": spec.dataset,
        "condition": spec.condition,
        "mode": mode,
        "layers": list(layers),
        "device": first.get("device"),
        "dtype": first.get("dtype"),
        "num_frames": preprocessing.get("num_frames"),
        "video_min_pixels": preprocessing.get("video_min_pixels"),
        "video_max_pixels": preprocessing.get("video_max_pixels"),
        "max_new_tokens": spec.output_budget,
    }


def aggregate_run(directory: Path, spec: RunSpec) -> dict[str, Any]:
    """Aggregate one benchmark directory while retaining raw samples for pairing."""

    samples = load_samples(directory)
    summary = _read_json(directory / "summary.json")
    sample_ids = [str(sample.get("sample_id", sample.get("sample_index"))) for sample in samples]
    sample_indices = [sample.get("sample_index") for sample in samples]
    sample_indices_contiguous = sample_indices == list(range(len(samples)))
    checkpoint_rows = {label: _checkpoint_rows(samples, label) for label in CHECKPOINTS}
    checkpoint_presence = {
        str(sample.get("sample_id", sample.get("sample_index"))): {
            checkpoint.get("label")
            for checkpoint in sample.get("checkpoints", [])
            if isinstance(checkpoint, Mapping)
        }
        for sample in samples
    }
    lossless_by_checkpoint = {
        label: sum(bool(row.get("outputs_match")) for row in rows)
        for label, rows in checkpoint_rows.items()
    }
    lossless_all_checkpoints = sum(
        all(label in checkpoint_presence[sample_id] for label in CHECKPOINTS)
        and all(
            bool(next(
                checkpoint.get("outputs_match")
                for checkpoint in sample.get("checkpoints", [])
                if checkpoint.get("label") == label
            ))
            for label in CHECKPOINTS
        )
        for sample_id, sample in zip(sample_ids, samples)
    )
    runtime_error_samples = sum(
        any(str(checkpoint.get("status")) == "error" for checkpoint in sample.get("checkpoints", []))
        for sample in samples
    )
    recomputed_mismatch = len(samples) - lossless_all_checkpoints
    source_summary = {
        "total": int(summary.get("total_samples", -1)),
        "completed": int(summary.get("completed_samples", -1)),
        "lossless": int(summary.get("lossless_samples", -1)),
        "mismatch": int(summary.get("mismatch_samples", -1)),
        "runtime_errors": int(summary.get("runtime_errors", -1)),
        "run_completed": bool(summary.get("run_completed", False)),
    }
    summary_consistent = (
        source_summary["total"] == len(samples)
        and source_summary["completed"] == len(samples)
        and source_summary["lossless"] == lossless_all_checkpoints
        and source_summary["mismatch"] == recomputed_mismatch
        and source_summary["runtime_errors"] == runtime_error_samples
        and source_summary["run_completed"]
    )
    coverage = {
        "total": len(samples),
        "completed": len(samples),
        "lossless": lossless_all_checkpoints,
        "mismatch": recomputed_mismatch,
        "runtime_errors": runtime_error_samples,
        "lossless_by_checkpoint": lossless_by_checkpoint,
        "sample_file_count": len(samples),
        "sample_ids_unique": len(sample_ids) == len(set(sample_ids)),
        "sample_indices_contiguous": sample_indices_contiguous,
        "summary_consistent": summary_consistent,
        "source_summary_total": source_summary["total"],
        "source_summary_completed": source_summary["completed"],
        "source_summary_lossless": source_summary["lossless"],
        "source_summary_mismatch": source_summary["mismatch"],
        "source_summary_runtime_errors": source_summary["runtime_errors"],
        "source_summary_run_completed": source_summary["run_completed"],
    }
    result: dict[str, Any] = {
        "dataset": spec.dataset,
        "display_name": spec.display_name,
        "condition": spec.condition,
        "path": str(directory),
        "config": _config_from_samples(spec, samples),
        "coverage": coverage,
        "target": {
            "accuracy": _bool_mean(_nested(sample, "target_baseline", "task_metrics", "correct") for sample in samples),
            "bleu": _mean(_nested(sample, "target_baseline", "text_metrics", "bleu") for sample in samples),
            "rouge_l": _mean(_nested(sample, "target_baseline", "text_metrics", "rouge_l") for sample in samples),
            "coverage": _mean(_nested(sample, "target_baseline", "text_metrics", "coverage") for sample in samples),
            "end_to_end_s": _mean(_nested(sample, "target_baseline", "end_to_end_s") for sample in samples),
        },
        "checkpoints": {},
        "task_accuracy": {"target": _target_task_accuracy(samples)},
        "samples": samples,
    }
    for label in CHECKPOINTS:
        rows = checkpoint_rows[label]
        status_counts = Counter(str(row.get("status", "missing")) for row in rows)
        result["checkpoints"][label] = {
            "n": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "lossless_count": sum(bool(row.get("outputs_match")) for row in rows),
            "lossless_rate": _bool_mean(row.get("outputs_match") for row in rows),
            "accuracy": _bool_mean(_nested(row, "task_metrics", "correct") for row in rows),
            "bleu": _mean(_nested(row, "text_metrics", "bleu") for row in rows),
            "rouge_l": _mean(_nested(row, "text_metrics", "rouge_l") for row in rows),
            "coverage": _mean(_nested(row, "text_metrics", "coverage") for row in rows),
            "performance": {
                "tau": _mean(_nested(row, "acceptance", "tau") for row in rows),
                "esr": _mean(_nested(row, "speedup", "esr") for row in rows),
                "dsr": _mean(_nested(row, "speedup", "dsr") for row in rows),
                "tokens_per_second": _mean(_nested(row, "timing", "tokens_per_second") for row in rows),
                "end_to_end_s": _mean(_nested(row, "timing", "end_to_end_s") for row in rows),
                "speedup_vs_target": _mean(_nested(row, "timing", "speedup_vs_target") for row in rows),
            },
        }
        result["task_accuracy"][label] = _task_accuracy(samples, label)
    return result


def _hash_map(samples: Sequence[Mapping[str, Any]], label: str | None) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for sample in samples:
        sample_id = str(sample.get("sample_id", sample.get("sample_index")))
        if label is None:
            value = _nested(sample, "target_baseline", "output_hash")
        else:
            value = None
            for checkpoint in sample.get("checkpoints", []):
                if checkpoint.get("label") == label:
                    value = checkpoint.get("speculative_output_hash")
                    break
        values[sample_id] = str(value) if value is not None else None
    return values


def compare_hashes(
    reference_samples: Sequence[Mapping[str, Any]],
    condition_samples: Sequence[Mapping[str, Any]],
    labels: Sequence[str] = CHECKPOINTS,
) -> dict[str, int]:
    """Count output-hash changes for paired samples between two conditions."""

    result: dict[str, int] = {}
    ref_target = _hash_map(reference_samples, None)
    cond_target = _hash_map(condition_samples, None)
    if set(ref_target) != set(cond_target):
        raise ValueError("reference and condition sample IDs do not match")
    result["target"] = sum(ref_target[key] != cond_target[key] for key in ref_target)
    for label in labels:
        ref = _hash_map(reference_samples, label)
        cond = _hash_map(condition_samples, label)
        result[label] = sum(ref[key] != cond[key] for key in ref)
    return result


def _public_run(run: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "samples"}


def _find_reference(runs: Sequence[Mapping[str, Any]], dataset: str) -> Mapping[str, Any] | None:
    return next((run for run in runs if run["dataset"] == dataset and run["condition"] == "full"), None)


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _number(value)
    if number is None:
        return str(value)
    if percent:
        return f"{number * 100:.1f}%"
    return f"{number:.{digits}f}"


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _scenario_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        row = {
            "dataset": run["dataset"],
            "scenario": run["display_name"],
            "condition": run["condition"],
            "mode": run["config"]["mode"],
            "layers": ",".join(str(layer) for layer in run["config"]["layers"]),
            "device": run["config"]["device"],
            "same_device_as_full": run.get("comparability", {}).get("same_device_as_full"),
            "summary_consistent": run["coverage"]["summary_consistent"],
            "total": run["coverage"]["total"],
            "completed": run["coverage"]["completed"],
            "lossless": run["coverage"]["lossless"],
            "mismatch": run["coverage"]["mismatch"],
            "runtime_errors": run["coverage"]["runtime_errors"],
            "target_accuracy": run["target"]["accuracy"],
            "target_bleu": run["target"]["bleu"],
            "target_rouge_l": run["target"]["rouge_l"],
        }
        rows.append(row)
    return rows


def _performance_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for label, metrics in run["checkpoints"].items():
            row = {
                "dataset": run["dataset"],
                "scenario": run["display_name"],
                "condition": run["condition"],
                "checkpoint": label,
                "device": run["config"]["device"],
                "same_device_as_full": run.get("comparability", {}).get("same_device_as_full"),
                "comparability": run.get("comparability", {}).get("note"),
                "lossless_rate": metrics["lossless_rate"],
                "accuracy": metrics["accuracy"],
                "bleu": metrics["bleu"],
                "rouge_l": metrics["rouge_l"],
                "coverage": metrics["coverage"],
                **metrics["performance"],
            }
            rows.append(row)
    return rows


def _speculative_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the speculative-decoding metrics used by the dedicated plots."""

    rows = []
    for run in runs:
        target_end_to_end = run["target"]["end_to_end_s"]
        for label, metrics in run["checkpoints"].items():
            performance = metrics["performance"]
            speculative_end_to_end = performance["end_to_end_s"]
            end_to_end_speedup = (
                target_end_to_end / speculative_end_to_end
                if target_end_to_end is not None and speculative_end_to_end
                else None
            )
            rows.append(
                {
                    "dataset": run["dataset"],
                    "scenario": run["display_name"],
                    "condition": run["condition"],
                    "checkpoint": label,
                    "device": run["config"]["device"],
                    "same_device_as_full": run.get("comparability", {}).get("same_device_as_full"),
                    "comparability": run.get("comparability", {}).get("note"),
                    "target_end_to_end_s": target_end_to_end,
                    "speculative_end_to_end_s": speculative_end_to_end,
                    "end_to_end_speedup_ratio": end_to_end_speedup,
                    "speedup_vs_target": performance["speedup_vs_target"],
                    "tau": performance["tau"],
                    "esr": performance["esr"],
                    "dsr": performance["dsr"],
                    "tokens_per_second": performance["tokens_per_second"],
                    "lossless_rate": metrics["lossless_rate"],
                }
            )
    return rows


def _task_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run["dataset"] != "mvbench":
            continue
        for task, target in run["task_accuracy"]["target"].items():
            row = {
                "dataset": run["dataset"],
                "scenario": run["display_name"],
                "condition": run["condition"],
                "task": task,
                "target": target,
                LLAVA_CHECKPOINT: run["task_accuracy"].get(LLAVA_CHECKPOINT, {}).get(task),
                SHAREGPT_CHECKPOINT: run["task_accuracy"].get(SHAREGPT_CHECKPOINT, {}).get(task),
            }
            rows.append(row)
    return rows


def _hash_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        changes = run.get("hash_changes_vs_full")
        if changes is None:
            continue
        rows.append(
            {
                "dataset": run["dataset"],
                "scenario": run["display_name"],
                "condition": run["condition"],
                "device": run["config"]["device"],
                "same_device_as_full": run.get("comparability", {}).get("same_device_as_full"),
                **changes,
            }
        )
    return rows


def _save_figure(fig: Any, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )
    return plt


def _condition_labels(runs: Sequence[Mapping[str, Any]], dataset: str) -> list[str]:
    return [run["display_name"] for run in runs if run["dataset"] == dataset]


def render_figures(runs: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Render six compact figures from the already aggregated data."""

    plt = _setup_matplotlib()
    colors = {"mvbench": "#2166ac", "vdc50": "#b2182b"}
    checkpoint_colors = {LLAVA_CHECKPOINT: "#2166ac", SHAREGPT_CHECKPOINT: "#67a9cf"}

    # Figure 1: lossless/coverage overview.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, ("mvbench", "vdc50")):
        subset = [run for run in runs if run["dataset"] == dataset]
        if not subset:
            axis.text(0.5, 0.5, f"Không có dữ liệu {dataset}", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = [run["display_name"] for run in subset]
        values = [run["checkpoints"][LLAVA_CHECKPOINT]["lossless_rate"] * 100 for run in subset]
        bars = axis.bar(range(len(labels)), values, color=colors[dataset], alpha=0.85)
        axis.set_ylim(0, 105)
        axis.set_ylabel("Lossless rate (%)")
        axis.set_title(f"{dataset.upper()} — output agreement")
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center", fontsize=9)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 1. DFlash lossless output agreement by scenario", fontsize=13)
    _save_figure(fig, output, "figure1_lossless_rate")
    plt.close(fig)

    # Figure 2: task-level MVBench accuracy.
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    mvbench = [run for run in runs if run["dataset"] == "mvbench"]
    tasks = sorted({task for run in mvbench for task in run["task_accuracy"][LLAVA_CHECKPOINT]})
    if not mvbench or not tasks:
        axis.text(0.5, 0.5, "Không có dữ liệu MVBench", ha="center", va="center")
        axis.set_axis_off()
    else:
        width = 0.8 / len(mvbench)
        for index, run in enumerate(mvbench):
            values = [run["task_accuracy"][LLAVA_CHECKPOINT].get(task, 0) * 100 for task in tasks]
            positions = [task_index + (index - (len(mvbench) - 1) / 2) * width for task_index in range(len(tasks))]
            axis.bar(positions, values, width=width, label=run["display_name"], alpha=0.85)
        axis.set_xticks(range(len(tasks)), [task.replace("_", " ") for task in tasks], rotation=20, ha="right")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title("Figure 2. MVBench task accuracy (llava68k checkpoint)")
        axis.legend(ncol=2, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, output, "figure2_mvbench_accuracy")
    plt.close(fig)

    # Figure 3: latency and throughput.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_index, dataset in enumerate(("mvbench", "vdc50")):
        subset = [run for run in runs if run["dataset"] == dataset]
        labels = [run["display_name"] for run in subset]
        for col_index, metric in enumerate(("end_to_end_s", "tokens_per_second")):
            axis = axes[row_index][col_index]
            if not subset:
                axis.text(0.5, 0.5, f"Không có dữ liệu {dataset}", ha="center", va="center")
                axis.set_axis_off()
                continue
            width = 0.35
            positions = list(range(len(subset)))
            for offset, label in enumerate(CHECKPOINTS):
                values = [run["checkpoints"][label]["performance"][metric] for run in subset]
                bars = axis.bar(
                    [position + (offset - 0.5) * width for position in positions],
                    values,
                    width=width,
                    label=label.replace("qwen25vl-3b-dflash-", ""),
                    color=checkpoint_colors[label],
                    alpha=0.9,
                )
                for bar, run in zip(bars, subset):
                    if not run.get("comparability", {}).get("same_device_as_full", True):
                        bar.set_hatch("//")
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.set_ylabel("Seconds" if metric == "end_to_end_s" else "Tokens/s")
            suffix = " (*)" if any(
                not run.get("comparability", {}).get("same_device_as_full", True) for run in subset
            ) else ""
            axis.set_title(
                f"{dataset.upper()} — {('end-to-end latency' if metric == 'end_to_end_s' else 'throughput')}{suffix}"
            )
            axis.grid(axis="y", alpha=0.25)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8)
    has_cross_device = any(
        not run.get("comparability", {}).get("same_device_as_full", True)
        for run in runs
        if run["dataset"] == "vdc50"
    )
    if has_cross_device:
        fig.suptitle(
            "Figure 3. DFlash inference performance by dataset and checkpoint "
            "(hatched VDC bars = cross-GPU exploratory comparison)",
            fontsize=12,
        )
    else:
        fig.suptitle("Figure 3. DFlash inference performance by dataset and checkpoint", fontsize=13)
    _save_figure(fig, output, "figure3_latency_throughput")
    plt.close(fig)

    # Figure 4: VDC quality and paired hash changes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    vdc = [run for run in runs if run["dataset"] == "vdc50"]
    labels = [run["display_name"] for run in vdc]
    if not vdc:
        for axis in axes:
            axis.text(0.5, 0.5, "Không có dữ liệu VDC50", ha="center", va="center")
            axis.set_axis_off()
    else:
        width = 0.35
        positions = list(range(len(vdc)))
        for offset, metric in enumerate(("bleu", "rouge_l")):
            values = [run["checkpoints"][LLAVA_CHECKPOINT][metric] for run in vdc]
            axes[0].bar(
                [position + (offset - 0.5) * width for position in positions],
                values,
                width=width,
                label=metric.upper().replace("_L", "-L"),
                color=("#b2182b", "#ef8a62")[offset],
            )
        axes[0].set_xticks(positions, labels, rotation=25, ha="right")
        axes[0].set_ylabel("Score")
        axes[0].set_title("VDC50 — DFlash quality (llava68k)")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.25)

        changes = [run.get("hash_changes_vs_full", {"target": 0, LLAVA_CHECKPOINT: 0, SHAREGPT_CHECKPOINT: 0}) for run in vdc]
        for offset, key in enumerate(("target", LLAVA_CHECKPOINT, SHAREGPT_CHECKPOINT)):
            values = [change[key] / run["coverage"]["total"] * 100 for change, run in zip(changes, vdc)]
            axes[1].bar(
                [position + (offset - 1) * 0.25 for position in positions],
                values,
                width=0.25,
                label=key.replace("qwen25vl-3b-dflash-", ""),
            )
        axes[1].set_xticks(positions, labels, rotation=25, ha="right")
        axes[1].set_ylabel("Changed hashes vs Full (%)")
        axes[1].set_title("VDC50 — paired output changes")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 4. VDC50 quality and ablation effects", fontsize=13)
    _save_figure(fig, output, "figure4_vdc_quality_ablation")
    plt.close(fig)

    # Figure 5: target-vs-speculative inference time and end-to-end speedup.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row_index, dataset in enumerate(("mvbench", "vdc50")):
        subset = [run for run in runs if run["dataset"] == dataset]
        labels = [run["display_name"] for run in subset]
        time_axis, speedup_axis = axes[row_index]
        if not subset:
            for axis in (time_axis, speedup_axis):
                axis.text(0.5, 0.5, f"Không có dữ liệu {dataset}", ha="center", va="center")
                axis.set_axis_off()
            continue
        positions = list(range(len(subset)))
        time_series = [
            ("Target baseline", [run["target"]["end_to_end_s"] for run in subset], "#777777"),
            *[
                (
                    label.replace("qwen25vl-3b-dflash-", ""),
                    [run["checkpoints"][label]["performance"]["end_to_end_s"] for run in subset],
                    checkpoint_colors[label],
                )
                for label in CHECKPOINTS
            ],
        ]
        width = 0.23
        for offset, (label, values, color) in enumerate(time_series):
            bars = time_axis.bar(
                [position + (offset - 1) * width for position in positions],
                values,
                width=width,
                label=label,
                color=color,
                alpha=0.9,
            )
            for bar, run in zip(bars, subset):
                if not run.get("comparability", {}).get("same_device_as_full", True):
                    bar.set_hatch("//")
        time_axis.set_xticks(positions, labels, rotation=25, ha="right")
        time_axis.set_ylabel("Seconds")
        time_axis.set_title(f"{dataset.upper()} — target vs speculative E2E time")
        time_axis.grid(axis="y", alpha=0.25)
        if row_index == 0:
            time_axis.legend(fontsize=8)

        for offset, label in enumerate(CHECKPOINTS):
            values = [
                run["target"]["end_to_end_s"]
                / max(run["checkpoints"][label]["performance"]["end_to_end_s"], 1e-12)
                for run in subset
            ]
            bars = speedup_axis.bar(
                [position + (offset - 0.5) * width for position in positions],
                values,
                width=width,
                label=label.replace("qwen25vl-3b-dflash-", ""),
                color=checkpoint_colors[label],
                alpha=0.9,
            )
            for bar, run in zip(bars, subset):
                if not run.get("comparability", {}).get("same_device_as_full", True):
                    bar.set_hatch("//")
        speedup_axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
        speedup_axis.set_xticks(positions, labels, rotation=25, ha="right")
        speedup_axis.set_ylabel("E2E speedup (×)")
        speedup_axis.set_title(f"{dataset.upper()} — target E2E / speculative E2E")
        speedup_axis.grid(axis="y", alpha=0.25)
        if row_index == 0:
            speedup_axis.legend(fontsize=8)
    has_cross_device = any(
        not run.get("comparability", {}).get("same_device_as_full", True)
        for run in runs
        if run["dataset"] == "vdc50"
    )
    title = "Figure 5. Speculative inference time and end-to-end speedup"
    if has_cross_device:
        title += " (hatched VDC bars = cross-GPU exploratory comparison)"
    fig.suptitle(title, fontsize=12 if has_cross_device else 13)
    _save_figure(fig, output, "figure5_speculative_timing_speedup")
    plt.close(fig)

    # Figure 6: acceptance and end-to-end/decode-only speedup statistics.
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    metric_specs = (
        ("tau", "τ (effective emitted tokens/round)", "tokens/round"),
        ("esr", "ESR (end-to-end speedup)", "Speedup (×)"),
        ("dsr", "DSR (decode-only speedup)", "Speedup (×)"),
    )
    for row_index, dataset in enumerate(("mvbench", "vdc50")):
        subset = [run for run in runs if run["dataset"] == dataset]
        labels = [run["display_name"] for run in subset]
        if not subset:
            for axis in axes[row_index]:
                axis.text(0.5, 0.5, f"Không có dữ liệu {dataset}", ha="center", va="center")
                axis.set_axis_off()
            continue
        positions = list(range(len(subset)))
        width = 0.35
        for col_index, (metric, title, ylabel) in enumerate(metric_specs):
            axis = axes[row_index][col_index]
            for offset, label in enumerate(CHECKPOINTS):
                values = [run["checkpoints"][label]["performance"][metric] for run in subset]
                bars = axis.bar(
                    [position + (offset - 0.5) * width for position in positions],
                    values,
                    width=width,
                    label=label.replace("qwen25vl-3b-dflash-", ""),
                    color=checkpoint_colors[label],
                    alpha=0.9,
                )
                for bar, run in zip(bars, subset):
                    if not run.get("comparability", {}).get("same_device_as_full", True):
                        bar.set_hatch("//")
            axis.set_xticks(positions, labels, rotation=25, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{dataset.upper()} — {title}")
            axis.grid(axis="y", alpha=0.25)
            if row_index == 0 and col_index == 0:
                axis.legend(fontsize=8)
    title = "Figure 6. Speculative acceptance and speedup statistics"
    if has_cross_device:
        title += " (hatched VDC bars = cross-GPU exploratory comparison)"
    fig.suptitle(title, fontsize=12 if has_cross_device else 13)
    _save_figure(fig, output, "figure6_speculative_acceptance_speedup")
    plt.close(fig)


def _markdown_report(runs: Sequence[Mapping[str, Any]], output: Path) -> str:
    speculative_rows = _speculative_rows(runs)

    def metric_range(
        dataset: str,
        key: str,
        *,
        condition: str | None = None,
        same_device: bool | None = None,
    ) -> str:
        values = [
            _number(row.get(key))
            for row in speculative_rows
            if row["dataset"] == dataset
            and (condition is None or row["condition"] == condition)
            and (same_device is None or row["same_device_as_full"] is same_device)
        ]
        numbers = [value for value in values if value is not None]
        if not numbers:
            return "—"
        return f"{min(numbers):.2f}–{max(numbers):.2f}"

    mvbench_speedup = metric_range("mvbench", "end_to_end_speedup_ratio")
    mvbench_tau = metric_range("mvbench", "tau")
    mvbench_esr = metric_range("mvbench", "esr")
    mvbench_dsr = metric_range("mvbench", "dsr")
    vdc_same_speedup = metric_range("vdc50", "end_to_end_speedup_ratio", same_device=True)
    vdc_same_tau = metric_range("vdc50", "tau", same_device=True)
    vdc_cut_speedup = metric_range("vdc50", "end_to_end_speedup_ratio", condition="exp1_cut")
    vdc_cut_esr = metric_range("vdc50", "esr", condition="exp1_cut")
    vdc_cut_dsr = metric_range("vdc50", "dsr", condition="exp1_cut")

    complete = all(
        run["coverage"]["completed"] == run["coverage"]["total"]
        and run["coverage"]["runtime_errors"] == 0
        and run["coverage"]["summary_consistent"]
        and run["coverage"]["sample_ids_unique"]
        and run["coverage"]["sample_indices_contiguous"]
        for run in runs
    )
    lines = [
        "# Báo cáo tổng hợp DFlash Inference trên MVBench và VDC50",
        "",
        f"**Trạng thái vận hành:** `{'COMPLETED' if complete else 'INCOMPLETE'} — diagnostic benchmark`",
        "",
        "Báo cáo được sinh hoàn toàn từ các JSON artifact của các lần chạy đã hoàn tất; không chạy lại model/GPU và không sửa artifact gốc.",
        "",
        "## Executive summary",
        "",
        "- MVBench gồm 4 điều kiện, mỗi điều kiện 500 mẫu; VDC50 gồm 4 điều kiện, mỗi điều kiện 50 mẫu.",
        "- Trên MVBench, accuracy khoảng 58.0–58.2%; các ablation gần như giữ nguyên output so với Full.",
        "- Trên VDC50, hai điều kiện zero giữ nguyên output so với Full; cut thay đổi decoding rõ rệt và tăng lossless rate từ 4% lên 12%.",
        f"- Speculative metrics: MVBench có E2E speedup khoảng {mvbench_speedup}×, τ {mvbench_tau} token/round, ESR {mvbench_esr}× và DSR {mvbench_dsr}×; VDC50 Full/zero có E2E speedup {vdc_same_speedup}× trên cùng GPU.",
        "- VDC cut chạy trên GPU khác với Full/zero, do đó các kết luận về speedup hoặc chất lượng của cut được giữ ở mức exploratory.",
        "",
        "## Protocol và phạm vi",
        "",
        "| Dataset | Số điều kiện | Mẫu/điều kiện | Frames | Output budget | Checkpoints |",
        "|---|---:|---:|---:|---:|---|",
        "| MVBench | 4 | 500 | 8 | 16 tokens | llava68k, sharegpt68k |",
        "| VDC50 | 4 | 50 | 8 | 256 tokens | llava68k, sharegpt68k |",
        "",
        "`Lossless` nghĩa là toàn bộ speculative output token trùng target output token ở cấp sample.",
        "",
        "## Trực quan tổng quan",
        "",
        "![Figure 1 — tỷ lệ lossless theo kịch bản](figure1_lossless_rate.png)",
        "",
        "*Hình 1. Tỷ lệ output lossless của các checkpoint DFlash trên từng kịch bản.*",
        "",
    ]

    for dataset, title in (("mvbench", "MVBench"), ("vdc50", "VDC50")):
        lines.extend([f"## 1. Kết quả theo từng kịch bản — {title}", ""])
        if dataset == "mvbench":
            lines.extend(
                [
                    "![Figure 2 — accuracy theo task trên MVBench](figure2_mvbench_accuracy.png)",
                    "",
                    "*Hình 2. Accuracy theo task của checkpoint llava68k trên MVBench; dùng để đối chiếu nhanh giữa các ablation.*",
                    "",
                ]
            )
        for index, run in enumerate((run for run in runs if run["dataset"] == dataset), start=1):
            config = run["config"]
            coverage = run["coverage"]
            lines.extend(
                [
                    f"### 1.{index} {run['display_name']}",
                    "",
                    f"- Mode: `{config['mode']}`; layers: `{config['layers'] or 'full visual context'}`.",
                    f"- Device/dtype: `{config['device']}` / `{config['dtype']}`; frames: `{config['num_frames']}`; output budget: `{config['max_new_tokens']}`.",
                    f"- Coverage: `{coverage['completed']}/{coverage['total']}`; runtime errors: `{coverage['runtime_errors']}`; lossless: `{coverage['lossless']}/{coverage['total']}`; summary consistency: `{coverage['summary_consistent']}`.",
                    f"- Performance comparability: `{run.get('comparability', {}).get('note', 'not available')}`.",
                    "",
                    "| Model group | Accuracy | BLEU | ROUGE-L | τ | ESR | DSR | Tokens/s | E2E (s) | Speedup vs target (×) |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            lines.append(
                f"| Target baseline | {_fmt(run['target']['accuracy'], percent=True)} | {_fmt(run['target']['bleu'])} | {_fmt(run['target']['rouge_l'])} | — | — | — | — | {_fmt(run['target']['end_to_end_s'])} | — |"
            )
            for label in CHECKPOINTS:
                metrics = run["checkpoints"][label]
                perf = metrics["performance"]
                lines.append(
                    f"| {label.replace('qwen25vl-3b-dflash-', '')} | {_fmt(metrics['accuracy'], percent=True)} | {_fmt(metrics['bleu'])} | {_fmt(metrics['rouge_l'])} | {_fmt(perf['tau'])} | {_fmt(perf['esr'])} | {_fmt(perf['dsr'])} | {_fmt(perf['tokens_per_second'])} | {_fmt(perf['end_to_end_s'])} | {_fmt(perf['speedup_vs_target'])} |"
                )
            if dataset == "mvbench":
                lines.extend(["", "Task accuracy (target / llava68k):"])
                for task, target in run["task_accuracy"]["target"].items():
                    dflash = run["task_accuracy"][LLAVA_CHECKPOINT].get(task)
                    lines.append(f"- `{task}`: {_fmt(target, percent=True)} / {_fmt(dflash, percent=True)}")
            else:
                lines.extend(
                    [
                        "",
                        f"- Hash changes vs Full: target `{run.get('hash_changes_vs_full', {}).get('target', 0)}`; llava68k `{run.get('hash_changes_vs_full', {}).get(LLAVA_CHECKPOINT, 0)}`; sharegpt68k `{run.get('hash_changes_vs_full', {}).get(SHAREGPT_CHECKPOINT, 0)}`.",
                    ]
                )
            lines.append("")

    lines.extend(
        [
            "## 2. So sánh performance theo dataset",
            "",
            "![Figure 3 — latency và throughput theo dataset](figure3_latency_throughput.png)",
            "",
            "*Hình 3. So sánh latency end-to-end và throughput; các cột VDC khác GPU được hatch và chỉ nên đọc ở mức exploratory.*",
            "",
            "### Speculative decoding metrics",
            "",
            "![Figure 5 — inference time và speedup end-to-end](figure5_speculative_timing_speedup.png)",
            "",
            "*Hình 5. Thời gian suy luận target/DFlash và speedup end-to-end; đường gạch ngang tương ứng speedup = 1×.*",
            "",
            "![Figure 6 — τ, ESR và DSR](figure6_speculative_acceptance_speedup.png)",
            "",
            "*Hình 6. Các metric speculative: τ là số token phát ra hiệu dụng mỗi acceptance round, ESR là speedup end-to-end, DSR là speedup phần decode.*",
            "",
            f"Các metric trong hai hình trên được đọc như sau: E2E speedup > 1× nghĩa là DFlash nhanh hơn target baseline ở cấp end-to-end; ESR phản ánh cùng xu hướng trên từng sample rồi lấy trung bình; DSR chỉ xét decode. MVBench đạt E2E speedup {mvbench_speedup}× và τ {mvbench_tau}; VDC50 Full/zero đạt E2E speedup {vdc_same_speedup}× và τ {vdc_same_tau}. Vì vậy không nên đồng nhất ESR/DSR với chất lượng output hoặc với speedup cross-GPU.",
            "",
            "### MVBench",
            "",
            "MVBench cho thấy DFlash giữ accuracy gần như bằng target baseline trong cả bốn điều kiện. Full, EXP1-zero và EXP1-cut đều đạt 58.2%; EXP2-zero đạt 58.0%. Các giá trị E2E speedup, ESR và DSR đều lớn hơn 1× trong artifact hiện tại, nhưng biên lợi ích chỉ ở mức vừa phải và throughput/latency vẫn phụ thuộc output length và trạng thái runtime.",
            "",
            "### VDC50",
            "",
            f"VDC50 có chất lượng caption tuyệt đối thấp (BLEU khoảng 0.047–0.050; ROUGE-L khoảng 0.235–0.241), vì vậy phần này phù hợp hơn cho phân tích decoding/ablation hơn là tuyên bố chất lượng caption tổng quát. Full và hai zero có chỉ số quality gần như trùng nhau; cut có E2E speedup {vdc_cut_speedup}×, ESR {vdc_cut_esr}× và DSR {vdc_cut_dsr}× trong số liệu ghi nhận nhưng chạy trên GPU khác, vì vậy đây chưa phải bằng chứng về lợi ích tốc độ của cut.",
            "",
            "![Figure 4 — quality và hash changes trên VDC50](figure4_vdc_quality_ablation.png)",
            "",
            "*Hình 4. Chất lượng caption và số hash thay đổi so với Full trên VDC50; cut cần được diễn giải cùng giới hạn cross-GPU.*",
            "",
            "## 3. CONFIRMED",
            "",
            "- Tám batch chính đều có đủ sample files, hoàn tất và không có runtime error.",
            "- MVBench accuracy và output agreement gần như không đổi qua các ablation.",
            "- VDC zero layer 25/33 và zero layer 1/9/17/25/33 không đổi output hash so với Full trên 50/50 mẫu.",
            "- VDC cut đổi output hash trên 47/50 speculative outputs và tăng lossless rate quan sát được lên 12%.",
            "",
            "## 4. EXPLORATORY INSIGHTS",
            "",
            "1. **Zeroing chưa làm thay đổi output trong các vị trí được thử.** Việc zero visual hidden ở các layer đã chọn không làm thay đổi output trên VDC50 và chỉ tạo thay đổi rất nhỏ trên MVBench; dữ liệu này chưa đủ để xác định bottleneck nội tại.",
            "2. **Cut tác động mạnh hơn zero.** Cut thay đổi context sequence vật lý, do đó decoding path thay đổi rõ ràng; tuy nhiên output khác target nhiều hơn không đồng nghĩa với chất lượng tốt hơn.",
            "3. **Không thấy speed benefit ổn định từ cut trong artifact hiện tại.** Cut giảm visual positions nhưng E2E chậm hơn Full, phù hợp với khả năng overhead của đường triển khai hiện tại lớn hơn lợi ích giảm sequence length.",
            "4. **Checkpoint ảnh hưởng hiệu năng nhiều hơn accuracy.** Hai checkpoint thường cho cùng output/accuracy, nhưng τ, throughput và latency khác nhau, đặc biệt trên VDC50.",
            "5. **τ cao không tự động bảo đảm E2E speedup cao.** Trên VDC50 Full/zero cùng GPU, llava có τ khoảng 2.40–2.47 và E2E speedup khoảng 1.46–1.50×, trong khi sharegpt có τ khoảng 1.63–1.66 và E2E speedup chỉ khoảng 1.02–1.03×; overhead của checkpoint và đường decode vẫn là yếu tố quyết định.",
            "6. **Khoảng cách ESR–DSR phản ánh overhead ngoài decode.** DSR chỉ đo phần decode còn ESR bao gồm end-to-end; khi hai giá trị lệch nhau, acceptance tốt chưa chắc chuyển thành lợi ích E2E tương ứng.",
            "",
            "## 5. INCOMPLETE / LIMITATIONS",
            "",
            "- VDC cut chạy trên `cuda:1`, trong khi Full và zero chạy trên `cuda:0`; target baseline của cut cũng khác Full ở 48/50 mẫu. Vì vậy không dùng kết quả này để khẳng định cut nhanh hơn hoặc tốt hơn.",
            "- Các lần chạy MVBench ban đầu từng bị `Killed`, nhưng các lần resume đã hoàn tất và artifact cuối cùng đã được kiểm tra lại bằng sample counts, summary fields và lossless recomputation.",
            "- VDC không có accuracy kiểu multiple-choice; BLEU/ROUGE ở đây là metric mô tả, không phải bằng chứng về chất lượng video QA tổng quát.",
            "- Báo cáo cũ `qwen25vl_3b_dflash_vdc50_8frames_isolated_20260820` được giữ như reference, không tính thành condition thứ năm vì schema cũ thiếu metadata ablation.",
            "",
            "## 6. Recommended next",
            "",
            "Với giới hạn OOM hiện tại, kết quả này đủ để chốt báo cáo. Nếu có thể tối ưu sau này, ưu tiên profiling hoặc chạy subset ngắn cho VDC cut; không cần rerun full trước khi sử dụng các kết luận đã xác nhận ở trên.",
            "",
            "## 7. Figures and machine-readable artifacts",
            "",
            "- [Figure 1 — lossless rate](figure1_lossless_rate.png)",
            "- [Figure 2 — MVBench task accuracy](figure2_mvbench_accuracy.png)",
            "- [Figure 3 — latency/throughput](figure3_latency_throughput.png)",
            "- [Figure 4 — VDC quality and ablation](figure4_vdc_quality_ablation.png)",
            "- [Figure 5 — speculative timing/speedup](figure5_speculative_timing_speedup.png)",
            "- [Figure 6 — speculative acceptance/speedup](figure6_speculative_acceptance_speedup.png)",
            "- [Scenario summary CSV](scenario_summary.csv)",
            "- [Performance CSV](performance.csv)",
            "- [Speculative metrics CSV](speculative_metrics.csv)",
            "- [MVBench task accuracy CSV](mvbench_task_accuracy.csv)",
            "- [Hash comparison CSV](hash_comparison.csv)",
            "- [Summary JSON](summary.json)",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(specs: Sequence[RunSpec], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Aggregate runs, render figures, and write the complete report bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [aggregate_run(spec.path, spec) for spec in specs]
    for run in runs:
        reference = _find_reference(runs, run["dataset"])
        same_device = reference is None or run["config"]["device"] == reference["config"]["device"]
        same_dtype = reference is None or run["config"]["dtype"] == reference["config"]["dtype"]
        run["comparability"] = {
            "same_device_as_full": same_device,
            "same_dtype_as_full": same_dtype,
            "note": (
                "same-device comparison"
                if same_device and same_dtype
                else "cross-device or dtype comparison; exploratory only"
            ),
        }
        if reference is not None and run["condition"] != "full":
            run["hash_changes_vs_full"] = compare_hashes(
                reference["samples"], run["samples"], CHECKPOINTS
            )
        elif run["condition"] == "full":
            run["hash_changes_vs_full"] = {"target": 0, **{label: 0 for label in CHECKPOINTS}}

    scenario_rows = _scenario_rows(runs)
    performance_rows = _performance_rows(runs)
    speculative_rows = _speculative_rows(runs)
    task_rows = _task_rows(runs)
    hash_rows = _hash_rows(runs)
    _csv_write(
        output_dir / "scenario_summary.csv",
        scenario_rows,
        list(scenario_rows[0]) if scenario_rows else ["dataset", "scenario"],
    )
    _csv_write(
        output_dir / "performance.csv",
        performance_rows,
        list(performance_rows[0]) if performance_rows else ["dataset", "scenario", "checkpoint"],
    )
    _csv_write(
        output_dir / "speculative_metrics.csv",
        speculative_rows,
        list(speculative_rows[0]) if speculative_rows else ["dataset", "scenario", "checkpoint"],
    )
    _csv_write(
        output_dir / "mvbench_task_accuracy.csv",
        task_rows,
        list(task_rows[0]) if task_rows else ["dataset", "scenario", "task"],
    )
    _csv_write(
        output_dir / "hash_comparison.csv",
        hash_rows,
        list(hash_rows[0]) if hash_rows else ["dataset", "scenario", "target"],
    )
    render_figures(runs, output_dir)

    payload = {
        "report": "DFlash inference benchmark report",
        "datasets": ["mvbench", "vdc50"],
        "runs": [_public_run(run) for run in runs],
        "source_policy": "completed per-sample inference artifacts; no GPU rerun",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(_markdown_report(runs, output_dir), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--results-root", type=Path, default=workspace_path("results", "infer")
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results_root = resolve_workspace_path(args.results_root)
    output_dir = resolve_workspace_path(args.output_dir)
    build_report(default_run_specs(results_root), output_dir)
    print(f"Wrote DFlash inference report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
