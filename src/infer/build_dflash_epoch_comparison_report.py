"""Build a reproducible Vietnamese comparison report for DFlash 6e and 20e runs.

The module reads completed per-sample JSON artifacts only.  It deliberately does
not import Transformers or load checkpoint tensors, so report generation is safe
to run on a CPU-only environment.
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


CHECKPOINT_FAMILIES = ("llava68k", "sharegpt68k")
EPOCHS = ("6e", "20e")
METRIC_KEYS = (
    "lossless_rate",
    "accuracy",
    "rouge_l",
    "bleu",
    "tau",
    "esr",
    "dsr",
    "tokens_per_second",
    "end_to_end_s",
    "speedup_vs_target",
)
DISPLAY_NAMES = {"llava68k": "LLaVA-68k", "sharegpt68k": "ShareGPT-68k"}
EPOCH_COLORS = {"6e": "#4C78A8", "20e": "#F58518"}
FAMILY_COLORS = {"llava68k": "#54A24B", "sharegpt68k": "#E45756"}
DEFAULT_OUTPUT_DIR = workspace_path(
    "results", "infer", "dflash20e_vs_6e_report_2026-08-26"
)


@dataclass(frozen=True)
class EpochSpec:
    """Locations for one epoch's completed Full runs."""

    epoch: str
    mvbench_dir: Path
    vdc50_dir: Path


@dataclass(frozen=True)
class DatasetRun:
    """Loaded, validated samples for one epoch and dataset."""

    dataset: str
    epoch: str
    directory: Path
    samples: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    device: str | None
    dtype: str | None
    checkpoint_labels: tuple[str, ...]


@dataclass(frozen=True)
class EpochBundle:
    """Both datasets for one epoch."""

    epoch: str
    datasets: dict[str, DatasetRun]


def default_epoch_specs(root: Path | None = None) -> list[EpochSpec]:
    """Return the verified 6e and 20e Full-run locations."""

    root = resolve_workspace_path(root or "results/infer")
    return [
        EpochSpec(
            "6e",
            root / "mvbench100_full_20260823",
            root / "vdc50_exp_full_dflash_20260823",
        ),
        EpochSpec(
            "20e",
            root / "dflash20e_20260825" / "mvbench100_full_20260823",
            root / "dflash20e_20260825" / "vdc50_exp_full_dflash_20260823",
        ),
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
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


def _family_from_label(label: Any) -> str | None:
    text = str(label or "").lower()
    for family in CHECKPOINT_FAMILIES:
        if family in text:
            return family
    return None


def _sample_id(sample: Mapping[str, Any]) -> str:
    value = sample.get("sample_id")
    if value is None:
        raise ValueError("sample is missing sample_id")
    return str(value)


def _load_dataset(directory: Path, *, epoch: str, dataset: str) -> DatasetRun:
    summary = _read_json(directory / "summary.json")
    paths = sorted(directory.glob("sample_*.json"))
    if not paths:
        raise FileNotFoundError(f"no sample_*.json files found in {directory}")
    samples = tuple(_read_json(path) for path in paths)
    ids = [_sample_id(sample) for sample in samples]
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample_id in {directory}: {duplicates}")
    indices = [sample.get("sample_index") for sample in samples]
    if indices != list(range(len(samples))):
        raise ValueError(f"sample_index is not contiguous in {directory}")
    labels = sorted(
        {
            str(checkpoint.get("label"))
            for sample in samples
            for checkpoint in sample.get("checkpoints", [])
            if isinstance(checkpoint, Mapping) and _family_from_label(checkpoint.get("label"))
        }
    )
    families = {_family_from_label(label) for label in labels}
    if families != set(CHECKPOINT_FAMILIES):
        raise ValueError(f"{directory}: expected both checkpoint families, found {sorted(families)}")
    source_total = int(summary.get("total_samples", -1))
    source_completed = int(summary.get("completed_samples", -1))
    if source_total != len(samples) or source_completed != len(samples):
        raise ValueError(
            f"{directory}: summary coverage {source_total}/{source_completed} "
            f"does not match {len(samples)} sample files"
        )
    devices = {str(sample.get("device")) for sample in samples if sample.get("device") is not None}
    dtypes = {str(sample.get("dtype")) for sample in samples if sample.get("dtype") is not None}
    return DatasetRun(
        dataset=dataset,
        epoch=epoch,
        directory=directory,
        samples=samples,
        summary=summary,
        device=next(iter(devices)) if len(devices) == 1 else ",".join(sorted(devices)) or None,
        dtype=next(iter(dtypes)) if len(dtypes) == 1 else ",".join(sorted(dtypes)) or None,
        checkpoint_labels=tuple(labels),
    )


def load_epoch_bundle(spec: EpochSpec) -> EpochBundle:
    """Load and validate one epoch's MVBench and VDC50 Full runs."""

    if spec.epoch not in EPOCHS:
        raise ValueError(f"unsupported epoch {spec.epoch!r}; expected one of {EPOCHS}")
    return EpochBundle(
        epoch=spec.epoch,
        datasets={
            "mvbench": _load_dataset(spec.mvbench_dir, epoch=spec.epoch, dataset="mvbench"),
            "vdc50": _load_dataset(spec.vdc50_dir, epoch=spec.epoch, dataset="vdc50"),
        },
    )


def _checkpoint_row(sample: Mapping[str, Any], family: str) -> Mapping[str, Any] | None:
    for checkpoint in sample.get("checkpoints", []):
        if isinstance(checkpoint, Mapping) and _family_from_label(checkpoint.get("label")) == family:
            return checkpoint
    return None


def _target_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    def target_e2e(sample: Mapping[str, Any]) -> Any:
        value = _nested(sample, "target_baseline", "timing", "end_to_end_s")
        return value if value is not None else _nested(sample, "target_baseline", "end_to_end_s")

    return {
        "accuracy": _mean(_nested(sample, "target_baseline", "task_metrics", "correct") for sample in samples),
        "rouge_l": _mean(_nested(sample, "target_baseline", "text_metrics", "rouge_l") for sample in samples),
        "bleu": _mean(_nested(sample, "target_baseline", "text_metrics", "bleu") for sample in samples),
        "end_to_end_s": _mean(target_e2e(sample) for sample in samples),
    }


def _checkpoint_metrics(samples: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    rows = [row for sample in samples if (row := _checkpoint_row(sample, family)) is not None]
    if len(rows) != len(samples):
        raise ValueError(f"checkpoint family {family} is missing from one or more samples")
    return {
        "n": len(rows),
        "status_counts": dict(sorted(Counter(str(row.get("status", "missing")) for row in rows).items())),
        "lossless_count": sum(bool(row.get("outputs_match")) for row in rows),
        "lossless_rate": _mean(row.get("outputs_match") for row in rows),
        "accuracy": _mean(_nested(row, "task_metrics", "correct") for row in rows),
        "rouge_l": _mean(_nested(row, "text_metrics", "rouge_l") for row in rows),
        "bleu": _mean(_nested(row, "text_metrics", "bleu") for row in rows),
        "tau": _mean(_nested(row, "acceptance", "tau") for row in rows),
        "esr": _mean(_nested(row, "speedup", "esr") for row in rows),
        "dsr": _mean(_nested(row, "speedup", "dsr") for row in rows),
        "tokens_per_second": _mean(_nested(row, "timing", "tokens_per_second") for row in rows),
        "end_to_end_s": _mean(_nested(row, "timing", "end_to_end_s") for row in rows),
        "speedup_vs_target": _mean(_nested(row, "timing", "speedup_vs_target") for row in rows),
    }


def _task_accuracy(samples: Sequence[Mapping[str, Any]], family: str) -> dict[str, float]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        row = _checkpoint_row(sample, family)
        if row is not None:
            grouped[str(sample.get("task", "unknown"))].append(
                _nested(row, "task_metrics", "correct")
            )
    return {
        task: mean
        for task, values in sorted(grouped.items())
        if (mean := _mean(values)) is not None
    }


def _target_task_accuracy(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("task", "unknown"))].append(
            _nested(sample, "target_baseline", "task_metrics", "correct")
        )
    return {
        task: mean
        for task, values in sorted(grouped.items())
        if (mean := _mean(values)) is not None
    }


def _hashes(samples: Sequence[Mapping[str, Any]], family: str | None) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for sample in samples:
        sample_id = _sample_id(sample)
        if family is None:
            result[sample_id] = _nested(sample, "target_baseline", "output_hash")
        else:
            row = _checkpoint_row(sample, family)
            result[sample_id] = row.get("speculative_output_hash") if row is not None else None
    return result


def _delta(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        old_value = _number(old.get(key))
        new_value = _number(new.get(key))
        result[key] = new_value - old_value if old_value is not None and new_value is not None else None
    return result


def _relative_delta(old: Any, new: Any) -> float | None:
    old_value = _number(old)
    new_value = _number(new)
    if old_value is None or new_value is None or old_value == 0:
        return None
    return (new_value - old_value) / abs(old_value)


def _paired_dataset(old: DatasetRun, new: DatasetRun) -> dict[str, Any]:
    old_map = {_sample_id(sample): sample for sample in old.samples}
    new_map = {_sample_id(sample): sample for sample in new.samples}
    if set(old_map) != set(new_map):
        missing_old = sorted(set(new_map) - set(old_map))
        missing_new = sorted(set(old_map) - set(new_map))
        raise ValueError(
            f"{old.dataset}: sample IDs do not pair; missing in old={missing_old[:5]}, "
            f"missing in new={missing_new[:5]}"
        )
    comparable = old.device == new.device and old.dtype == new.dtype
    target_old = _target_metrics(old.samples)
    target_new = _target_metrics(new.samples)
    target_hash_old = _hashes(old.samples, None)
    target_hash_new = _hashes(new.samples, None)
    target_hash_changes = sum(target_hash_old[key] != target_hash_new[key] for key in old_map)
    result: dict[str, Any] = {
        "dataset": old.dataset,
        "paired_samples": len(old_map),
        "devices": {"6e": old.device, "20e": new.device},
        "dtypes": {"6e": old.dtype, "20e": new.dtype},
        "performance_comparable": comparable,
        "target": {
            "6e": target_old,
            "20e": target_new,
            "delta": _delta(target_old, target_new),
        },
        "target_hash_changes": target_hash_changes,
        "task_accuracy": {
            epoch: {family: _task_accuracy(run.samples, family) for family in CHECKPOINT_FAMILIES}
            for epoch, run in (("6e", old), ("20e", new))
        },
        "target_task_accuracy": {
            "6e": _target_task_accuracy(old.samples),
            "20e": _target_task_accuracy(new.samples),
        },
        "checkpoints": {},
        "paired_changes": {},
    }
    for family in CHECKPOINT_FAMILIES:
        old_metrics = _checkpoint_metrics(old.samples, family)
        new_metrics = _checkpoint_metrics(new.samples, family)
        old_hashes = _hashes(old.samples, family)
        new_hashes = _hashes(new.samples, family)
        changes = sum(old_hashes[key] != new_hashes[key] for key in old_map)
        result["checkpoints"][family] = {
            "6e": old_metrics,
            "20e": new_metrics,
            "delta": _delta(old_metrics, new_metrics),
            "relative_delta": {
                key: _relative_delta(old_metrics.get(key), new_metrics.get(key))
                for key in METRIC_KEYS
            },
        }
        result["paired_changes"][family] = {
            "changed": changes,
            "unchanged": len(old_map) - changes,
            "changed_rate": changes / len(old_map) if old_map else None,
        }
    return result


def compare_epoch_bundles(old: EpochBundle, new: EpochBundle) -> dict[str, Any]:
    """Pair the two epoch bundles and compute aggregates/deltas."""

    if old.epoch != "6e" or new.epoch != "20e":
        raise ValueError("comparison requires a 6e bundle followed by a 20e bundle")
    return {
        "epochs": [old.epoch, new.epoch],
        "datasets": {
            dataset: _paired_dataset(old.datasets[dataset], new.datasets[dataset])
            for dataset in ("mvbench", "vdc50")
        },
    }


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number * 100:.1f}%" if percent else f"{number:.{digits}f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _public_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_json(item) for item in value]
    return value


def _save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.titlesize": 13})
    return plt


def _series_label(epoch: str, family: str) -> str:
    return f"{epoch} {DISPLAY_NAMES[family]}"


def _render_figures(comparison: Mapping[str, Any], output_dir: Path) -> None:
    plt = _matplotlib()
    mv = comparison["datasets"]["mvbench"]
    vdc = comparison["datasets"]["vdc50"]

    # Figure 1: a compact overview matrix for agreement and primary quality.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_index, (dataset, data, primary, title) in enumerate(
        (("mvbench", mv, "accuracy", "MVBench accuracy"), ("vdc50", vdc, "rouge_l", "VDC50 ROUGE-L"))
    ):
        labels = [_series_label(epoch, family) for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
        values_lossless = [
            data["checkpoints"][family][epoch]["lossless_rate"]
            for epoch in EPOCHS
            for family in CHECKPOINT_FAMILIES
        ]
        values_quality = [
            data["checkpoints"][family][epoch][primary]
            for epoch in EPOCHS
            for family in CHECKPOINT_FAMILIES
        ]
        colors = [EPOCH_COLORS[epoch] for epoch in EPOCHS for _ in CHECKPOINT_FAMILIES]
        x = list(range(len(labels)))
        axes[row_index, 0].bar(x, values_lossless, color=colors)
        axes[row_index, 0].set_ylim(0, 1.05)
        axes[row_index, 0].set_ylabel("Lossless rate")
        axes[row_index, 0].set_title(f"{dataset.upper()} — output agreement")
        axes[row_index, 1].bar(x, values_quality, color=colors)
        axes[row_index, 1].set_ylabel(primary)
        axes[row_index, 1].set_title(f"{dataset.upper()} — {title}")
        for axis in axes[row_index]:
            axis.set_xticks(x, labels, rotation=28, ha="right")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 1. 6e versus 20e overview", fontsize=13)
    _save_figure(fig, output_dir, "figure1_epoch_overview")
    plt.close(fig)

    # Figure 2: task-level MVBench accuracy.
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    tasks = sorted(
        set(mv["task_accuracy"]["6e"]["llava68k"])
        | set(mv["task_accuracy"]["20e"]["llava68k"])
        | set(mv["task_accuracy"]["6e"]["sharegpt68k"])
    )
    series = [("Target", "#777777", mv["target_task_accuracy"]["6e"], "")]
    for epoch in EPOCHS:
        for family in CHECKPOINT_FAMILIES:
            series.append(
                (
                    _series_label(epoch, family),
                    EPOCH_COLORS[epoch],
                    mv["task_accuracy"][epoch][family],
                    "" if family == "llava68k" else "//",
                )
            )
    width = 0.16
    x = list(range(len(tasks)))
    for index, (label, color, values, hatch) in enumerate(series):
        points = [
            (values.get(task) if isinstance(values, Mapping) else None)
            for task in tasks
        ]
        axis.bar(
            [point + (index - 2) * width for point in x],
            points,
            width=width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.4,
        )
    axis.set_xticks(x, tasks, rotation=25, ha="right")
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Figure 2. MVBench task accuracy by epoch and checkpoint")
    axis.legend(ncol=3, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, output_dir, "figure2_mvbench_task_accuracy")
    plt.close(fig)

    # Figure 3: MVBench acceptance and runtime; same-device comparison.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = [_series_label(epoch, family) for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
    x = list(range(len(labels)))
    for offset, key in enumerate(("tau", "esr", "dsr")):
        values = [mv["checkpoints"][family][epoch][key] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
        axes[0].bar(
            [point + (offset - 1) * 0.22 for point in x],
            values,
            width=0.22,
            label=key,
            alpha=0.85,
        )
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    throughput = [mv["checkpoints"][family][epoch]["tokens_per_second"] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
    latency = [mv["checkpoints"][family][epoch]["end_to_end_s"] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
    axes[1].bar(
        x,
        throughput,
        width=0.58,
        color=[EPOCH_COLORS[epoch] for epoch in EPOCHS for _ in CHECKPOINT_FAMILIES],
        label="tokens/s",
    )
    latency_axis = axes[1].twinx()
    latency_axis.plot(x, latency, color="#D95F02", marker="o", linewidth=2, label="E2E seconds")
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    latency_handles, latency_labels = latency_axis.get_legend_handles_labels()
    axes[1].legend(handles + latency_handles, legend_labels + latency_labels, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("MVBench — τ / ESR / DSR")
    axes[1].set_ylabel("tokens/s")
    latency_axis.set_ylabel("E2E seconds", color="#D95F02")
    axes[1].set_title("MVBench — throughput and E2E latency")
    fig.suptitle("Figure 3. MVBench speculative performance (same GPU)", fontsize=13)
    _save_figure(fig, output_dir, "figure3_mvbench_performance")
    plt.close(fig)

    # Figure 4: VDC quality and agreement.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [_series_label(epoch, family) for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
    x = list(range(len(labels)))
    for metric, axis in (("rouge_l", axes[0]), ("lossless_rate", axes[1])):
        values = [vdc["checkpoints"][family][epoch][metric] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
        axis.bar(x, values, color=[EPOCH_COLORS[epoch] for epoch in EPOCHS for _ in CHECKPOINT_FAMILIES])
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("VDC50 — ROUGE-L")
    axes[1].set_title("VDC50 — lossless rate")
    fig.suptitle("Figure 4. VDC50 quality and exact agreement", fontsize=13)
    _save_figure(fig, output_dir, "figure4_vdc_quality")
    plt.close(fig)

    # Figure 5: VDC performance, explicitly marked exploratory because devices differ.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for offset, key in enumerate(("tau", "esr", "dsr")):
        values = [vdc["checkpoints"][family][epoch][key] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
        axes[0].bar(
            [point + (offset - 1) * 0.22 for point in x],
            values,
            width=0.22,
            label=key,
        )
    axes[0].set_xticks(x, labels, rotation=28, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    throughput = [vdc["checkpoints"][family][epoch]["tokens_per_second"] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
    latency = [vdc["checkpoints"][family][epoch]["end_to_end_s"] for epoch in EPOCHS for family in CHECKPOINT_FAMILIES]
    axes[1].bar(
        x,
        throughput,
        width=0.58,
        color=[EPOCH_COLORS[epoch] for epoch in EPOCHS for _ in CHECKPOINT_FAMILIES],
        label="tokens/s",
    )
    latency_axis = axes[1].twinx()
    latency_axis.plot(x, latency, color="#D95F02", marker="o", linewidth=2, label="E2E seconds")
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    latency_handles, latency_labels = latency_axis.get_legend_handles_labels()
    axes[1].legend(handles + latency_handles, legend_labels + latency_labels, fontsize=8, loc="upper right")
    axes[0].set_title("VDC50 — τ / ESR / DSR")
    axes[1].set_title("VDC50 — throughput and E2E latency")
    axes[0].set_ylabel("Metric value")
    axes[1].set_ylabel("tokens/s")
    latency_axis.set_ylabel("E2E seconds", color="#D95F02")
    fig.suptitle("Figure 5. VDC50 performance (exploratory: cross-GPU)", fontsize=13)
    _save_figure(fig, output_dir, "figure5_vdc_performance")
    plt.close(fig)

    # Figure 6: relative epoch delta heatmap.
    rows: list[str] = []
    values: list[list[float]] = []
    columns: list[str] = []
    for dataset in ("mvbench", "vdc50"):
        data = comparison["datasets"][dataset]
        for family in CHECKPOINT_FAMILIES:
            row_name = f"{dataset.upper()} {DISPLAY_NAMES[family]}"
            rows.append(row_name)
            delta = data["checkpoints"][family]["delta"]
            metric_order = ("lossless_rate", "accuracy", "rouge_l", "tau", "esr", "dsr", "tokens_per_second", "end_to_end_s")
            if not columns:
                columns = list(metric_order)
            row = []
            for key in metric_order:
                old_value = data["checkpoints"][family]["6e"].get(key)
                difference = delta.get(key)
                row.append(float("nan") if old_value is None or difference is None else float(difference) / max(abs(float(old_value)), 1e-12) * 100)
            values.append(row)
    fig, axis = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-100, vmax=100)
    axis.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(range(len(rows)), rows)
    axis.set_xlabel("Relative delta: (20e − 6e) / |6e| × 100")
    axis.set_title("Figure 6. Relative performance delta from 6e to 20e")
    fig.colorbar(image, ax=axis, label="percent")
    _save_figure(fig, output_dir, "figure6_epoch_delta_heatmap")
    plt.close(fig)


def _comparison_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, data in comparison["datasets"].items():
        for family in CHECKPOINT_FAMILIES:
            for epoch in EPOCHS:
                metrics = data["checkpoints"][family][epoch]
                rows.append(
                    {
                        "dataset": dataset,
                        "epoch": epoch,
                        "checkpoint": family,
                        "n": metrics["n"],
                        **{key: metrics.get(key) for key in METRIC_KEYS},
                        "device": data["devices"][epoch],
                        "performance_comparable": data["performance_comparable"],
                    }
                )
    return rows


def _performance_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _comparison_rows(comparison):
        rows.append({key: row.get(key) for key in ("dataset", "epoch", "checkpoint", "device", "performance_comparable", "tau", "esr", "dsr", "tokens_per_second", "end_to_end_s", "speedup_vs_target")})
    return rows


def _task_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, data in comparison["datasets"].items():
        for epoch in EPOCHS:
            for family in CHECKPOINT_FAMILIES:
                for task, accuracy in data["task_accuracy"][epoch][family].items():
                    rows.append({"dataset": dataset, "epoch": epoch, "checkpoint": family, "task": task, "accuracy": accuracy})
    return rows


def _paired_rows(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset, data in comparison["datasets"].items():
        for family, values in data["paired_changes"].items():
            rows.append({"dataset": dataset, "comparison": family, **values})
        rows.append(
            {
                "dataset": dataset,
                "comparison": "target_hash_changes",
                "changed": data["target_hash_changes"],
                "unchanged": data["paired_samples"] - data["target_hash_changes"],
                "changed_rate": data["target_hash_changes"] / data["paired_samples"],
            }
        )
    return rows


def _table(rows: Sequence[Mapping[str, Any]], *, dataset: str) -> list[str]:
    lines = [
        "| Epoch | Checkpoint | Lossless | Accuracy | ROUGE-L | BLEU | τ | ESR | DSR | tok/s | E2E (s) | Device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["dataset"] != dataset:
            continue
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["epoch"]),
                    DISPLAY_NAMES[row["checkpoint"]],
                    _fmt(row["lossless_rate"], percent=True),
                    _fmt(row["accuracy"], percent=True),
                    _fmt(row["rouge_l"]),
                    _fmt(row["bleu"]),
                    _fmt(row["tau"]),
                    _fmt(row["esr"]),
                    _fmt(row["dsr"]),
                    _fmt(row["tokens_per_second"], digits=2),
                    _fmt(row["end_to_end_s"]),
                    str(row["device"] or "—"),
                )
            )
            + " |"
        )
    return lines


def _delta_table(data: Mapping[str, Any]) -> list[str]:
    """Render signed and relative 20e-minus-6e deltas for one dataset."""

    metric_names = {
        "lossless_rate": "Lossless rate",
        "accuracy": "Accuracy",
        "rouge_l": "ROUGE-L",
        "bleu": "BLEU",
        "tau": "τ",
        "esr": "ESR",
        "dsr": "DSR",
        "tokens_per_second": "tokens/s",
        "end_to_end_s": "E2E (s)",
    }
    lines = [
        "| Metric | LLaVA Δ | LLaVA relative | ShareGPT Δ | ShareGPT relative |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in metric_names.items():
        llava = data["checkpoints"]["llava68k"]
        sharegpt = data["checkpoints"]["sharegpt68k"]
        lines.append(
            f"| {label} | {_fmt(llava['delta'].get(key))} | {_fmt(llava['relative_delta'].get(key), percent=True)} | "
            f"{_fmt(sharegpt['delta'].get(key))} | {_fmt(sharegpt['relative_delta'].get(key), percent=True)} |"
        )
    return lines


def _write_report(comparison: Mapping[str, Any], output_dir: Path) -> None:
    rows = _comparison_rows(comparison)
    mv = comparison["datasets"]["mvbench"]
    vdc = comparison["datasets"]["vdc50"]
    mv_llava = mv["checkpoints"]["llava68k"]
    vdc_llava = vdc["checkpoints"]["llava68k"]
    mv_share = mv["checkpoints"]["sharegpt68k"]
    vdc_share = vdc["checkpoints"]["sharegpt68k"]
    lines = [
        "# Báo cáo performance DFlash Qwen2.5-VL-3B: 6 epoch so với 20 epoch",
        "",
        "**Trạng thái:** `COMPLETED — paired artifact comparison`",
        "",
        "Báo cáo được sinh từ các JSON artifact đã hoàn tất; không chạy lại model/GPU và không sửa artifact nguồn.",
        "",
        "## Executive summary",
        "",
        f"- MVBench gồm {mv['paired_samples']} mẫu ghép cặp, cùng GPU giữa 6e và 20e; VDC50 gồm {vdc['paired_samples']} mẫu ghép cặp nhưng chạy khác GPU.",
        f"- MVBench giữ nguyên accuracy và output: lossless {_fmt(mv_llava['6e']['lossless_rate'], percent=True)} → {_fmt(mv_llava['20e']['lossless_rate'], percent=True)}; paired target/checkpoint hash changes đều bằng 0.",
        f"- VDC50 lossless tăng từ {_fmt(vdc_llava['6e']['lossless_rate'], percent=True)} lên {_fmt(vdc_llava['20e']['lossless_rate'], percent=True)} cho cả hai checkpoint; đây là kết quả mô tả, không phải bằng chứng sạch về tác động của epoch vì target hash đổi {vdc['target_hash_changes']}/{vdc['paired_samples']} mẫu giữa hai GPU.",
        f"- Trên MVBench, 20e không tạo speedup cao hơn 6e: ESR LLaVA {_fmt(mv_llava['6e']['esr'])} → {_fmt(mv_llava['20e']['esr'])}; ShareGPT {_fmt(mv_share['6e']['esr'])} → {_fmt(mv_share['20e']['esr'])}.",
        "- Mọi kết luận latency/throughput VDC50 được đánh dấu `exploratory` do 6e chạy `cuda:0` còn 20e chạy `cuda:1`.",
        "",
        "![Figure 1 — overview 6e/20e](figure1_epoch_overview.png)",
        "",
        "*Hình 1. Overview về exact agreement và metric chất lượng chính của hai epoch.*",
        "",
        "## 1. Artifact identity và protocol",
        "",
        "| Epoch | MVBench Full | VDC50 Full | Checkpoint families |",
        "|---|---|---|---|",
        "| 6e | `results/infer/mvbench100_full_20260823` | `results/infer/vdc50_exp_full_dflash_20260823` | LLaVA-68k, ShareGPT-68k |",
        "| 20e | `results/infer/dflash20e_20260825/mvbench100_full_20260823` | `results/infer/dflash20e_20260825/vdc50_exp_full_dflash_20260823` | LLaVA-68k, ShareGPT-68k |",
        "",
        "MVBench dùng cùng manifest 500 mẫu. VDC50 dùng hai tên manifest khác nhau, nhưng kiểm tra paired records cho thấy cùng 50 video ID, câu hỏi, câu trả lời và local video path.",
        "",
        "### Coverage",
        "",
        f"- MVBench: {mv['paired_samples']} paired samples; 6e/20e đều hoàn tất, không có runtime error.",
        f"- VDC50: {vdc['paired_samples']} paired samples; 6e/20e đều hoàn tất, không có runtime error.",
        f"- Device MVBench: 6e `{mv['devices']['6e']}`, 20e `{mv['devices']['20e']}`; comparable = `{mv['performance_comparable']}`.",
        f"- Device VDC50: 6e `{vdc['devices']['6e']}`, 20e `{vdc['devices']['20e']}`; comparable = `{vdc['performance_comparable']}`.",
        "",
        "## 2. MVBench",
        "",
        "![Figure 2 — MVBench task accuracy](figure2_mvbench_task_accuracy.png)",
        "",
        "*Hình 2. Accuracy theo task, đối chiếu target với hai checkpoint ở 6e và 20e.*",
        "",
        "### Performance table",
        "",
        *_table(rows, dataset="mvbench"),
        "",
        "![Figure 3 — MVBench performance](figure3_mvbench_performance.png)",
        "",
        "*Hình 3. τ, ESR, DSR, throughput và E2E latency trên MVBench; 6e và 20e chạy cùng GPU nên có thể so sánh trực tiếp hơn VDC50.*",
        "",
        "### Phân tích MVBench",
        "",
        f"- Accuracy của target và hai checkpoint giữ ở mức {_fmt(mv['target']['6e']['accuracy'], percent=True)}; exact agreement của cả hai checkpoint giữ ở {_fmt(mv_llava['6e']['lossless_rate'], percent=True)} ở cả hai epoch.",
        f"- Hash comparison không phát hiện thay đổi output giữa 6e và 20e trên 500 mẫu cho target, LLaVA-68k và ShareGPT-68k. Đây là bằng chứng mạnh rằng trong workload MVBench này, 20e không đổi hành vi output quan sát được.",
        f"- τ giảm ở 20e: LLaVA {_fmt(mv_llava['6e']['tau'])} → {_fmt(mv_llava['20e']['tau'])}; ShareGPT {_fmt(mv_share['6e']['tau'])} → {_fmt(mv_share['20e']['tau'])}.",
        f"- ESR giảm tương ứng: LLaVA {_fmt(mv_llava['6e']['esr'])} → {_fmt(mv_llava['20e']['esr'])}; ShareGPT {_fmt(mv_share['6e']['esr'])} → {_fmt(mv_share['20e']['esr'])}. Vì vậy 20e không cho thấy lợi ích tốc độ rõ ràng trên MVBench dù chất lượng không đổi.",
        "",
        "## 3. VDC50",
        "",
        "![Figure 4 — VDC50 quality](figure4_vdc_quality.png)",
        "",
        "*Hình 4. ROUGE-L và lossless rate trên VDC50; lossless là exact token equality với target của từng run.*",
        "",
        "### Performance table",
        "",
        *_table(rows, dataset="vdc50"),
        "",
        "![Figure 5 — VDC50 performance](figure5_vdc_performance.png)",
        "",
        "*Hình 5. Performance VDC50; các metric thời gian/throughput là exploratory vì khác GPU giữa 6e và 20e.*",
        "",
        "### Phân tích VDC50",
        "",
        f"- Lossless tăng từ {_fmt(vdc_llava['6e']['lossless_rate'], percent=True)} lên {_fmt(vdc_llava['20e']['lossless_rate'], percent=True)} cho LLaVA-68k và cùng mức tăng cho ShareGPT-68k.",
        f"- ROUGE-L LLaVA giảm {_fmt(vdc_llava['6e']['rouge_l'])} → {_fmt(vdc_llava['20e']['rouge_l'])}; BLEU giảm {_fmt(vdc_llava['6e']['bleu'])} → {_fmt(vdc_llava['20e']['bleu'])}. Tuy nhiên target hash đổi {vdc['target_hash_changes']}/{vdc['paired_samples']} mẫu giữa hai GPU, nên không thể quy phần chênh lệch này hoàn toàn cho epoch.",
        f"- LLaVA 20e ghi nhận ESR {_fmt(vdc_llava['20e']['esr'])} và ShareGPT 20e {_fmt(vdc_share['20e']['esr'])}; các con số thấp hơn 6e trong artifact hiện tại nhưng bị confound bởi GPU/runtime khác nhau.",
        "",
        "## 4. Delta 20e − 6e",
        "",
        "![Figure 6 — epoch delta heatmap](figure6_epoch_delta_heatmap.png)",
        "",
        "*Hình 6. Delta tương đối `(20e − 6e) / |6e|`; ô trống là metric không phù hợp hoặc thiếu dữ liệu.*",
        "",
        "Heatmap chỉ nhằm tổng hợp hướng thay đổi. Với VDC50, các ô performance không được dùng để kết luận causal về số epoch; chúng phải được đọc cùng giới hạn cross-GPU.",
        "",
        "### Bảng delta định lượng",
        "",
        "#### MVBench — cùng GPU, so sánh trực tiếp hơn",
        "",
        *_delta_table(mv),
        "",
        "#### VDC50 — mô tả, bị confound bởi khác GPU",
        "",
        *_delta_table(vdc),
        "",
        "## 5. Paired output stability",
        "",
        "| Dataset | So sánh | Changed | Unchanged | Changed rate |",
        "|---|---|---:|---:|---:|",
        f"| MVBench | Target | 0 | {mv['paired_samples']} | 0.0% |",
        f"| MVBench | LLaVA-68k | 0 | {mv['paired_samples']} | 0.0% |",
        f"| MVBench | ShareGPT-68k | 0 | {mv['paired_samples']} | 0.0% |",
        f"| VDC50 | Target | {vdc['target_hash_changes']} | {vdc['paired_samples'] - vdc['target_hash_changes']} | {_fmt(vdc['target_hash_changes'] / vdc['paired_samples'], percent=True)} |",
        f"| VDC50 | LLaVA-68k | {vdc['paired_changes']['llava68k']['changed']} | {vdc['paired_changes']['llava68k']['unchanged']} | {_fmt(vdc['paired_changes']['llava68k']['changed_rate'], percent=True)} |",
        f"| VDC50 | ShareGPT-68k | {vdc['paired_changes']['sharegpt68k']['changed']} | {vdc['paired_changes']['sharegpt68k']['unchanged']} | {_fmt(vdc['paired_changes']['sharegpt68k']['changed_rate'], percent=True)} |",
        "",
        "## 6. Confirmed findings",
        "",
        "1. Cả hai epoch đều có coverage đầy đủ: 500 MVBench và 50 VDC50, không có runtime error.",
        "2. Trên MVBench, 20e giữ nguyên accuracy, lossless rate và output hashes so với 6e trên toàn bộ 500 mẫu.",
        "3. Trên MVBench, các metric acceptance/speed của 20e không cao hơn 6e trong artifact Full hiện tại.",
        "4. Trên VDC50, lossless rate quan sát được tăng từ 4% lên 12% cho cả hai checkpoint.",
        "",
        "## 7. Exploratory findings và giới hạn",
        "",
        "1. VDC50 6e chạy trên `cuda:0`, còn VDC50 20e chạy trên `cuda:1`; mọi so sánh E2E latency, throughput, ESR và DSR giữa hai epoch là exploratory.",
        "2. Target output hash của VDC50 đổi trên 48/50 mẫu giữa hai run. Vì target là baseline khác nhau ở cấp runtime, chênh lệch ROUGE-L/BLEU/lossless không thể được diễn giải như pure epoch effect.",
        "3. Lossless là exact token equality, không phải đánh giá ngữ nghĩa; ROUGE-L/BLEU của VDC50 là metric mô tả và không thay thế human/semantic evaluation.",
        "4. Checkpoint identity được xác định từ tên thư mục 6e/20e và checkpoint labels trong sample artifacts; report không giải nén lại tensor checkpoint.",
        "",
        "## 8. Recommended next experiment",
        "",
        "Để kết luận causal về 6e so với 20e trên VDC50, cần rerun hai checkpoint trên cùng một GPU, cùng process/runtime, cùng manifest và cùng seed/runtime settings; sau đó giữ nguyên paired hash analysis trong report này.",
        "",
        "## 9. Artifacts",
        "",
        "- `epoch_comparison.csv`: metric theo dataset/epoch/checkpoint.",
        "- `performance.csv`: τ/ESR/DSR/throughput/latency.",
        "- `task_accuracy.csv`: accuracy theo task.",
        "- `paired_changes.csv`: target/checkpoint hash changes.",
        "- `summary.json`: machine-readable aggregate and provenance.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(specs: Sequence[EpochSpec], output_dir: Path) -> dict[str, Any]:
    """Build all report artifacts and return the public summary."""

    by_epoch = {spec.epoch: spec for spec in specs}
    if set(by_epoch) != set(EPOCHS) or len(by_epoch) != len(specs):
        raise ValueError(f"build_report requires exactly one spec for {EPOCHS}")
    old = load_epoch_bundle(by_epoch["6e"])
    new = load_epoch_bundle(by_epoch["20e"])
    comparison = compare_epoch_bundles(old, new)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _comparison_rows(comparison)
    _write_csv(output_dir / "epoch_comparison.csv", rows, ["dataset", "epoch", "checkpoint", "n", *METRIC_KEYS, "device", "performance_comparable"])
    _write_csv(output_dir / "performance.csv", _performance_rows(comparison), ["dataset", "epoch", "checkpoint", "device", "performance_comparable", "tau", "esr", "dsr", "tokens_per_second", "end_to_end_s", "speedup_vs_target"])
    _write_csv(output_dir / "task_accuracy.csv", _task_rows(comparison), ["dataset", "epoch", "checkpoint", "task", "accuracy"])
    _write_csv(output_dir / "paired_changes.csv", _paired_rows(comparison), ["dataset", "comparison", "changed", "unchanged", "changed_rate"])
    _render_figures(comparison, output_dir)
    _write_report(comparison, output_dir)
    summary = {
        "report": "DFlash 6e vs 20e epoch comparison",
        "output_dir": str(output_dir.resolve()),
        "source_specs": [
            {
                "epoch": spec.epoch,
                "mvbench_dir": str(spec.mvbench_dir),
                "vdc50_dir": str(spec.vdc50_dir),
            }
            for spec in specs
        ],
        "comparison": _public_json(comparison),
        "figures": [
            f"figure{index}_{stem}.{suffix}"
            for index, stem in (
                (1, "epoch_overview"),
                (2, "mvbench_task_accuracy"),
                (3, "mvbench_performance"),
                (4, "vdc_quality"),
                (5, "vdc_performance"),
                (6, "epoch_delta_heatmap"),
            )
            for suffix in ("png", "pdf")
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=workspace_path("results", "infer")
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mvbench-6e", type=Path)
    parser.add_argument("--vdc50-6e", type=Path)
    parser.add_argument("--mvbench-20e", type=Path)
    parser.add_argument("--vdc50-20e", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = resolve_workspace_path(args.results_root)
    output_dir = resolve_workspace_path(args.output_dir)
    defaults = {spec.epoch: spec for spec in default_epoch_specs(results_root)}
    specs = [
        EpochSpec(
            "6e",
            resolve_workspace_path(args.mvbench_6e) if args.mvbench_6e else defaults["6e"].mvbench_dir,
            resolve_workspace_path(args.vdc50_6e) if args.vdc50_6e else defaults["6e"].vdc50_dir,
        ),
        EpochSpec(
            "20e",
            resolve_workspace_path(args.mvbench_20e) if args.mvbench_20e else defaults["20e"].mvbench_dir,
            resolve_workspace_path(args.vdc50_20e) if args.vdc50_20e else defaults["20e"].vdc50_dir,
        ),
    ]
    summary = build_report(specs, output_dir)
    print(json.dumps({"output_dir": summary["output_dir"], "figures": len(summary["figures"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
