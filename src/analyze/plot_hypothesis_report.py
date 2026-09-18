"""Generate report figures from the recorded hypothesis-validation artifacts.

The module deliberately keeps data loading and aggregation separate from the
plotting functions.  This makes the numbers used in the report auditable and
keeps the plotting command independent of model/GPU code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONDITIONS = ("full", "reduced25", "deleted")
CONDITION_LABELS = {
    "full": "Full",
    "reduced25": "Reduced 25%",
    "deleted": "Deleted",
}
DATASET_COLORS = {
    "visual": "#4C78A8",
    "text": "#F58518",
    "instruction": "#54A24B",
}


def load_jsonl_rows(
    paths: Iterable[str | Path],
    *,
    dataset_label: str | None = None,
    value_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Load non-empty JSONL records and optionally attach a dataset label."""

    rows: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object in {path}:{line_number}")
                record = dict(row)
                if dataset_label is not None:
                    record["dataset"] = dataset_label
                if value_mode is not None:
                    record["visual_value_mode"] = value_mode
                rows.append(record)
    return rows


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV and convert numeric cells to floats when possible."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if value is None or value == "":
                    row[key] = value
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot calculate the mean of an empty group")
    return sum(values) / len(values)


def _finite_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _condition_label(row: Mapping[str, Any]) -> str:
    """Normalize condition names from attention and MSD output schemas."""

    retention = _finite_float(row, "retention_percentage")
    if retention is not None:
        if math.isclose(retention, 100.0):
            return "full"
        if math.isclose(retention, 25.0):
            return "reduced25"
        if math.isclose(retention, 0.0):
            return "deleted"
    condition = str(row.get("condition", "")).lower()
    if condition in {"full", "f"}:
        return "full"
    if condition in {"retention", "reduced", "reduced25", "r25", "r"}:
        return "reduced25"
    if condition in {"deleted", "delete", "zero", "d", "d0", "cut"}:
        return "deleted"
    raise ValueError(f"Unknown visual condition: {row.get('condition')!r}")


def _group_rows(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def summarize_h1b_attention(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float | int]]:
    """Summarize real-value H1b attention mass by dataset/policy/condition."""

    real_rows = (
        row for row in rows if str(row.get("visual_value_mode", "real")) == "real"
    )
    summary: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for key, group in _group_rows(
        real_rows, ("dataset", "attention_policy", "visual_condition")
    ).items():
        dataset, policy, raw_condition = key
        condition = _condition_label({"condition": raw_condition})
        metrics: dict[str, list[float]] = defaultdict(list)
        for row in group:
            for metric in ("visual_mass", "text_mass", "instruction_mass"):
                value = _finite_float(row, metric)
                if value is not None:
                    metrics[metric].append(value)
        if not all(metrics.get(metric) for metric in ("visual_mass", "text_mass", "instruction_mass")):
            continue
        summary[(str(dataset), str(policy), condition)] = {
            "n": min(len(metrics[metric]) for metric in metrics),
            **{metric: _mean(values) for metric, values in metrics.items()},
        }
    return summary


def summarize_h1b_value_ablation(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, float | int]]:
    """Summarize real-value draft-logit changes and KL diagnostics."""

    real_rows = (
        row for row in rows if str(row.get("visual_value_mode", "real")) == "real"
    )
    summary: dict[tuple[str, str], dict[str, float | int]] = {}
    for key, group in _group_rows(real_rows, ("dataset", "visual_condition")).items():
        dataset, raw_condition = key
        condition = _condition_label({"condition": raw_condition})
        values: dict[str, list[float]] = defaultdict(list)
        top1: list[float] = []
        for row in group:
            for metric in (
                "value_ablation_logit_max_abs_delta",
                "value_ablation_kl_forward",
                "value_ablation_kl_reverse",
            ):
                value = _finite_float(row, metric)
                if value is not None:
                    values[metric].append(value)
            match = row.get("value_ablation_top1_match")
            if isinstance(match, bool):
                top1.append(float(match))
        if not values.get("value_ablation_logit_max_abs_delta"):
            continue
        summary[(str(dataset), condition)] = {
            "n": len(values["value_ablation_logit_max_abs_delta"]),
            "logit_delta": _mean(values["value_ablation_logit_max_abs_delta"]),
            "kl_forward": _mean(values.get("value_ablation_kl_forward", [float("nan")])),
            "kl_reverse": _mean(values.get("value_ablation_kl_reverse", [float("nan")])),
            "top1_match": _mean(top1) if top1 else float("nan"),
        }
    return summary


def summarize_pilot_attention(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float | int]]:
    """Summarize summary-level E5 attention rows by value/policy/condition."""

    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("modality") != "summary":
            continue
        value_mode = str(row.get("visual_value_mode") or "real")
        condition = _condition_label({"condition": row.get("visual_condition")})
        groups[(value_mode, str(row.get("attention_policy")), condition)].append(row)
    summary: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for key, group in groups.items():
        metrics: dict[str, list[float]] = defaultdict(list)
        for row in group:
            for metric in ("visual_mass", "text_mass", "instruction_mass"):
                value = _finite_float(row, metric)
                if value is not None:
                    metrics[metric].append(value)
        if not all(metrics.get(metric) for metric in ("visual_mass", "text_mass", "instruction_mass")):
            continue
        summary[key] = {
            "n": min(len(metrics[metric]) for metric in metrics),
            **{metric: _mean(values) for metric, values in metrics.items()},
        }
    return summary


def summarize_h2_acceptance(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, float | int]]:
    """Summarize accepted-prefix tokens by prompt variant and visual condition."""

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_float(row, "accepted_prefix_tokens")
        if value is None:
            continue
        prompt = str(row.get("prompt_variant", "natural"))
        condition = _condition_label(row)
        groups[(prompt, condition)].append(value)
    return {
        key: {"n": len(values), "mean": _mean(values)}
        for key, values in groups.items()
    }


def load_position_table(path: str | Path) -> dict[str, dict[str, list[float]]]:
    """Read the recorded DFlash position-wise table from a Markdown report."""

    bins = ("0–10%", "10–25%", "25–50%", "50–75%", "75–100%")
    result: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6:
            continue
        label = cells[0]
        if label.startswith("DFlash "):
            label = label.removeprefix("DFlash ")
        if not (label.startswith("MVBench ") or label.startswith("VDC50 ")):
            continue
        try:
            values = [float(cell) for cell in cells[1:]]
        except ValueError:
            continue
        dataset, condition = label.split(maxsplit=1)
        result[dataset][condition] = values
    return dict(result)


def load_position_csv(path: str | Path) -> dict[str, dict[str, list[float]]]:
    """Read the machine-readable DFlash position-wise acceptance source."""

    result: dict[str, dict[str, list[float]]] = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = str(row.get("dataset", ""))
            condition = str(row.get("condition", ""))
            if not dataset or not condition:
                continue
            try:
                result[dataset][condition] = [
                    float(row[key])
                    for key in (
                        "bin_0_10",
                        "bin_10_25",
                        "bin_25_50",
                        "bin_50_75",
                        "bin_75_100",
                    )
                ]
            except (KeyError, TypeError, ValueError):
                continue
    return dict(result)


def load_h31_dflash_statistics(
    path: str | Path,
) -> dict[str, dict[str, list[float] | list[int]]]:
    """Load full-cohort DFlash position statistics, sorted by normalized bin."""

    rows_by_condition: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    condition_names = {"full": "Full", "zero": "Zero", "cut": "Cut"}
    for row in load_csv_rows(path):
        raw_condition = str(row.get("visual_condition", "")).lower()
        condition = condition_names.get(raw_condition)
        if condition is None:
            continue
        try:
            rows_by_condition[condition].append(
                {
                    "bin": int(float(row["bin"])),
                    "mean": float(row["mean"]),
                    "ci_low": float(row["ci_low"]),
                    "ci_high": float(row["ci_high"]),
                    "n": int(float(row["n"])),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    result: dict[str, dict[str, list[float] | list[int]]] = {}
    for condition, rows in rows_by_condition.items():
        ordered = sorted(rows, key=lambda row: int(row["bin"]))
        result[condition] = {
            "bin": [int(row["bin"]) for row in ordered],
            "mean": [float(row["mean"]) for row in ordered],
            "ci_low": [float(row["ci_low"]) for row in ordered],
            "ci_high": [float(row["ci_high"]) for row in ordered],
            "n": [int(row["n"]) for row in ordered],
        }
    return result


def load_msd_position_pilot(path: str | Path) -> dict[str, list[float]]:
    """Read the small MSD position pilot table from its recorded summary."""

    result: dict[str, list[float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in {"F", "R25", "D0"}:
            continue
        try:
            result[cells[0]] = [float(cell) for cell in cells[1:]]
        except ValueError:
            continue
    return result


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.clf()
    return paths


def plot_h1b_attention(
    summary: Mapping[tuple[str, str, str], Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    plt = _plt()
    datasets = ("VDC50", "MVBench")
    policies = ("last_instruction", "all_text")
    metrics = ("visual_mass", "text_mass", "instruction_mass")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    x = list(range(len(CONDITIONS)))
    width = 0.24
    for row_index, dataset in enumerate(datasets):
        for col_index, policy in enumerate(policies):
            axis = axes[row_index][col_index]
            for metric_index, metric in enumerate(metrics):
                values = [
                    float(summary[(dataset, policy, condition)][metric])
                    for condition in CONDITIONS
                ]
                positions = [position + (metric_index - 1) * width for position in x]
                axis.bar(
                    positions,
                    values,
                    width=width,
                    color=DATASET_COLORS[metric.split("_")[0]],
                    label=metric.removesuffix("_mass").capitalize(),
                )
            axis.set_title(f"{dataset} — {policy.replace('_', ' ')}")
            axis.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
            axis.set_ylim(0, 1.05)
            axis.grid(axis="y", alpha=0.25)
            if col_index == 0:
                axis.set_ylabel("Attention mass")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle("H1b — attention allocation under visual-position ablation", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_figure(fig, output_dir, "h1b_attention_mass")


def plot_h1b_value_ablation(
    summary: Mapping[tuple[str, str], Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    plt = _plt()
    datasets = ("VDC50", "MVBench")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    x = list(range(len(CONDITIONS)))
    for axis, dataset in zip(axes, datasets):
        values = [float(summary[(dataset, condition)]["logit_delta"]) for condition in CONDITIONS]
        bars = axis.bar(x, values, color=["#4C78A8", "#F58518", "#54A24B"])
        axis.set_title(dataset)
        axis.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
        axis.set_ylabel("Mean max |Δ draft logit|")
        axis.grid(axis="y", alpha=0.25)
        for bar, condition in zip(bars, CONDITIONS):
            row = summary[(dataset, condition)]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"KL={float(row['kl_forward']):.3g}\ntop-1={float(row['top1_match']) * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("H1b — visual-value ablation changes logits without changing captured attention")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "h1b_value_ablation")


def plot_h2_acceptance(
    summary: Mapping[tuple[str, str], Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    plt = _plt()
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(CONDITIONS)))
    styles = {
        "natural": ("#4C78A8", "Natural"),
        "answer_hint": ("#E45756", "Answer-hint"),
    }
    for prompt, (color, label) in styles.items():
        values = [float(summary[(prompt, condition)]["mean"]) for condition in CONDITIONS]
        axis.plot(x, values, marker="o", linewidth=2, color=color, label=label)
        for position, value in zip(x, values):
            axis.annotate(f"{value:.2f}", (position, value), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    axis.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    axis.set_ylabel("Mean accepted-prefix tokens")
    axis.set_title("H2.1 — answer hint × visual retention")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "h2_1_acceptance_interaction")


def plot_e5_zero_value_pilot(
    attention_summary: Mapping[tuple[str, str, str], Mapping[str, Any]],
    acceptance_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Plot all E5 pilot summary rows without expanding them into a long table."""

    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ("visual_mass", "text_mass", "instruction_mass")
    metric_colors = (DATASET_COLORS["visual"], DATASET_COLORS["text"], DATASET_COLORS["instruction"])
    x = list(range(len(CONDITIONS)))
    mode_styles = {"real": ("Real", ""), "zero": ("Zero", "//")}
    for axis, policy in zip(axes[0], ("last_instruction", "all_text")):
        width = 0.34
        for mode_index, (mode, (label, hatch)) in enumerate(mode_styles.items()):
            bottoms = [0.0] * len(CONDITIONS)
            positions = [position + (mode_index - 0.5) * width for position in x]
            for metric, color in zip(metrics, metric_colors):
                values = [
                    float(attention_summary[(mode, policy, condition)][metric])
                    for condition in CONDITIONS
                ]
                axis.bar(
                    positions,
                    values,
                    width=width,
                    bottom=bottoms,
                    color=color,
                    hatch=hatch,
                    edgecolor="white",
                    label=f"{label} — {metric.removesuffix('_mass')}" if policy == "last_instruction" else "_nolegend_",
                )
                bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_title(f"E5 attention — {policy.replace('_', ' ')}")
        axis.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Attention mass")
        axis.grid(axis="y", alpha=0.25)

    acceptance_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in acceptance_rows:
        value = _finite_float(row, "accepted_prefix_tokens")
        if value is None:
            continue
        mode = str(row.get("visual_value_mode") or "real")
        condition = _condition_label(row)
        acceptance_groups[(mode, condition)].append(value)
    axis = axes[1][0]
    for mode, color, label in (("real", "#4C78A8", "Real"), ("zero", "#E45756", "Zero")):
        available = [(condition, _mean(acceptance_groups[(mode, condition)])) for condition in CONDITIONS if (mode, condition) in acceptance_groups]
        if available:
            axis.plot(
                [CONDITIONS.index(condition) for condition, _ in available],
                [value for _, value in available],
                marker="o",
                color=color,
                linewidth=2,
                label=label,
            )
    axis.set_title("E5 acceptance pilot")
    axis.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    axis.set_ylabel("Accepted-prefix tokens")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axes[1][1].axis("off")
    axes[1][1].text(
        0.02,
        0.95,
        "E5: VDC@3K pilot\n"
        "Attention: n=2 samples per mode/policy/condition\n"
        "Acceptance: Real n=2 per condition; Zero n=2 Full only\n"
        "Hatched bars = zero visual values\n"
        "Full statistics remain in the source JSONL files.",
        va="top",
        fontsize=11,
        transform=axes[1][1].transAxes,
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle("E5 — zero-value pilot: values change logits, not captured attention", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save_figure(fig, output_dir, "h1b_zero_value_pilot")


def plot_qwen2_sweeps(
    length_rows: Sequence[Mapping[str, Any]],
    retention_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    series_styles = {
        "msd_keep_visual": ("#4C78A8", "Keep visual"),
        "msd_remove_all": ("#E45756", "Remove all"),
    }
    for series, (color, label) in series_styles.items():
        rows = sorted(
            (row for row in length_rows if row.get("series_id") == series),
            key=lambda row: float(row["visual_tokens"]),
        )
        axes[0].plot(
            [float(row["visual_tokens"]) / 1000 for row in rows],
            [float(row["accepted_prefix_tokens.mean"]) for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_xlabel("Target visual tokens (K)")
    axes[0].set_ylabel("Mean accepted-prefix tokens")
    axes[0].set_title("Length sweep")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    retention = sorted(retention_rows, key=lambda row: float(row["retention_percentage"]), reverse=True)
    axes[1].plot(
        [float(row["retention_percentage"]) for row in retention],
        [float(row["accepted_prefix_tokens.mean"]) for row in retention],
        marker="o",
        color="#54A24B",
    )
    axes[1].set_xlabel("Retained visual positions (%)")
    axes[1].set_ylabel("Mean accepted-prefix tokens")
    axes[1].set_title("Retention sweep — last instruction")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Qwen2/MSD legacy evidence used as supporting context")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "qwen2_msd_sweeps")


def plot_h1a_layer_proxy(layer_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    plt = _plt()
    rows = sorted(layer_rows, key=lambda row: float(row["layer"]))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    layers = [float(row["layer"]) for row in rows]
    axis.plot(layers, [float(row["visual_cosine.mean"]) for row in rows], marker="o", label="Visual cosine")
    axis.plot(layers, [float(row["text_cosine.mean"]) for row in rows], marker="o", label="Text cosine")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Cosine proxy")
    axis.set_title("H1a — layerwise information-retention proxy")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "h1a_layerwise_proxy")


def plot_h3_position_acceptance(
    dflash_table: Mapping[str, Mapping[str, Sequence[float]]],
    msd_pilot: Mapping[str, Sequence[float]],
    output_dir: Path,
) -> list[Path]:
    plt = _plt()
    bins = ("0–10", "10–25", "25–50", "50–75", "75–100")
    x = list(range(len(bins)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    styles = {"Full": ("#4C78A8", "o"), "Zero": ("#F58518", "s"), "Cut": ("#54A24B", "^")}
    for dataset, axis in zip(("MVBench", "VDC50"), axes[:2]):
        for condition, (color, marker) in styles.items():
            if condition in dflash_table.get(dataset, {}):
                axis.plot(x, dflash_table[dataset][condition], color=color, marker=marker, label=condition)
        axis.set_title(f"DFlash — {dataset}")
        axis.set_xticks(x, bins, rotation=25)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Acceptance rate")
    axes[0].legend(frameon=False)
    if msd_pilot:
        inset = axes[2]
        pilot_styles = {"F": ("#4C78A8", "o"), "R25": ("#F58518", "s"), "D0": ("#54A24B", "^")}
        for condition, (color, marker) in pilot_styles.items():
            if condition in msd_pilot:
                inset.plot(x, msd_pilot[condition], color=color, marker=marker, label=condition)
        inset.set_title("MSD pilot VDC@3K (n=2)")
        inset.set_xticks(x, bins, rotation=25)
        inset.legend(frameon=False)
        inset.grid(alpha=0.25)
    fig.suptitle("H3.1 — acceptance by answer position (backend/cohort caveat applies)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(fig, output_dir, "h3_1_position_acceptance")


def h31_pilot_condition_label(raw_condition: str) -> str:
    """Return the explicit condition label used in the MSD H3.1 pilot."""

    return {
        "F": "Full",
        "R25": "Reduced 25%",
        "D0": "Deleted",
    }.get(str(raw_condition), str(raw_condition))


def plot_h3_backend_cohort_provenance(
    msd_pilot: Mapping[str, Sequence[float]],
    dflash_full: Mapping[str, Mapping[str, Sequence[float]]],
    output_dir: Path,
) -> list[Path]:
    """Plot H3.1's historical MSD pilot beside the full DFlash study."""

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {
        "Full": "#4C78A8",
        "Reduced 25%": "#F58518",
        "Deleted": "#54A24B",
        "Zero": "#F58518",
        "Cut": "#54A24B",
    }
    markers = {"Full": "o", "Zero": "s", "Cut": "^"}

    pilot_bins = ("0–10%", "10–25%", "25–50%", "50–75%", "75–100%")
    pilot_axis = axes[0]
    for raw_condition, values in msd_pilot.items():
        condition = h31_pilot_condition_label(raw_condition)
        pilot_axis.plot(
            range(len(values)),
            values,
            marker={"Full": "o", "Reduced 25%": "s", "Deleted": "^"}.get(condition, "o"),
            color=colors.get(condition, "#999999"),
            label=condition,
        )
    pilot_axis.set_title("MSD / Qwen2-VL-7B\nVDC@3K pilot — n=2")
    pilot_axis.text(
        0.02,
        0.96,
        "Exploratory; no full-cohort CI",
        transform=pilot_axis.transAxes,
        va="top",
        fontsize=9,
        color="#9C2C2C",
    )
    pilot_axis.set_xticks(range(len(pilot_bins)), pilot_bins, rotation=25)
    pilot_axis.set_xlabel("Normalized answer position")
    pilot_axis.set_ylabel("Acceptance rate")
    pilot_axis.grid(alpha=0.25)

    full_axis = axes[1]
    for condition, stats in dflash_full.items():
        bins = list(stats.get("bin", []))
        means = [float(value) for value in stats.get("mean", [])]
        ci_low = [float(value) for value in stats.get("ci_low", [])]
        ci_high = [float(value) for value in stats.get("ci_high", [])]
        if not bins or not means:
            continue
        x = list(range(len(means)))
        lower = [mean - low for mean, low in zip(means, ci_low)]
        upper = [high - mean for mean, high in zip(means, ci_high)]
        full_axis.errorbar(
            x,
            means,
            yerr=[lower, upper],
            marker=markers.get(condition, "o"),
            color=colors.get(condition, "#999999"),
            capsize=3,
            linewidth=1.8,
            label=condition,
        )
    full_axis.set_title("DFlash / Qwen2.5-VL-3B\nVDC50 full study — n=50")
    full_axis.text(
        0.02,
        0.96,
        "Bootstrap 95% CI over samples",
        transform=full_axis.transAxes,
        va="top",
        fontsize=9,
        color="#245A9C",
    )
    full_axis.set_xticks(
        range(len(next(iter(dflash_full.values()))["mean"])),
        ("0–12.5%", "12.5–25%", "25–37.5%", "37.5–50%", "50–62.5%", "62.5–75%", "75–87.5%", "87.5–100%"),
        rotation=35,
    )
    full_axis.set_xlabel("Normalized answer position")
    full_axis.grid(alpha=0.25)
    full_axis.legend(frameon=False)
    fig.suptitle(
        "H3.1 provenance: MSD pilot n=2 versus DFlash full n=50\n"
        "The panels are not pooled; backend and cohort differ",
        y=1.03,
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "h3_1_backend_cohort_provenance")


def generate_figures(
    output_dir: str | Path,
    *,
    h1b_vdc_paths: Sequence[str | Path],
    h1b_mvbench_paths: Sequence[str | Path],
    h2_paths: Sequence[str | Path],
    e5_attention_real_path: str | Path,
    e5_attention_zero_path: str | Path,
    e5_acceptance_real_path: str | Path,
    e5_acceptance_zero_path: str | Path,
    h3_position_csv: str | Path,
    h3_pilot_path: str | Path,
    h3_dflash_full_stats_path: str | Path,
    legacy_length_csv: str | Path,
    legacy_retention_csv: str | Path,
    layer_proxy_csv: str | Path,
) -> list[Path]:
    """Generate report figures and record excluded pilot sources separately."""

    output = Path(output_dir)
    h1b_rows = load_jsonl_rows(h1b_vdc_paths, dataset_label="VDC50")
    h1b_rows += load_jsonl_rows(h1b_mvbench_paths, dataset_label="MVBench")
    h2_rows: list[dict[str, Any]] = []
    for path in h2_paths:
        h2_rows.extend(load_jsonl_rows([path]))

    h1b_attention = summarize_h1b_attention(h1b_rows)
    h1b_values = summarize_h1b_value_ablation(h1b_rows)
    h2_summary = summarize_h2_acceptance(h2_rows)
    # The E5 pilot and MSD H3.1 position/provenance plots are intentionally
    # excluded: they contain n=2 observations and are not suitable for the
    # report's quantitative evidence layer.
    figures: list[Path] = []
    figures += plot_h1b_attention(h1b_attention, output)
    figures += plot_h1b_value_ablation(h1b_values, output)
    figures += plot_h2_acceptance(h2_summary, output)
    figures += plot_qwen2_sweeps(
        load_csv_rows(legacy_length_csv), load_csv_rows(legacy_retention_csv), output
    )
    figures += plot_h1a_layer_proxy(load_csv_rows(layer_proxy_csv), output)

    manifest = {
        "h1b_rows": len(h1b_rows),
        "h2_rows": len(h2_rows),
        "figures": [path.name for path in figures],
        "sources": {
            "h1b_vdc": [str(path) for path in h1b_vdc_paths],
            "h1b_mvbench": [str(path) for path in h1b_mvbench_paths],
            "h2": [str(path) for path in h2_paths],
            "legacy_length": str(legacy_length_csv),
            "legacy_retention": str(legacy_retention_csv),
            "layer_proxy": str(layer_proxy_csv),
        },
        "excluded_sources": {
            "e5_pilot_attention_real": str(e5_attention_real_path),
            "e5_pilot_attention_zero": str(e5_attention_zero_path),
            "e5_pilot_acceptance_real": str(e5_acceptance_real_path),
            "e5_pilot_acceptance_zero": str(e5_acceptance_zero_path),
            "h3_legacy_position_csv": str(h3_position_csv),
            "h3_msd_pilot": str(h3_pilot_path),
            "h3_dflash_full_statistics_not_plotted_here": str(h3_dflash_full_stats_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return figures


def _default_paths(root: Path) -> dict[str, Any]:
    h1b_dir = root / "results" / "h1b_h21_20260909"
    return {
        "h1b_vdc": sorted(h1b_dir.glob("attention_kl_vdc50_part*.jsonl")),
        "h1b_mv": [
            path
            for path in sorted(h1b_dir.glob("attention_kl_mvbench_*.jsonl"))
            if "pilot" not in path.name.lower()
        ],
        "h2": [h1b_dir / "msd_natural_vdc50.jsonl", h1b_dir / "msd_answer_hint_vdc50.jsonl"],
        "e5_attention_real": root / "results" / "hypothesis_validation_gpu_20260907" / "msd_draft_attention_cohort5_3k_natural_f_r_d.jsonl",
        "e5_attention_zero": root / "results" / "hypothesis_validation_gpu_20260908" / "attention_cohort5_3k_z_natural_f_r_d.jsonl",
        "e5_acceptance_real": root / "results" / "hypothesis_validation_gpu_20260907" / "msd_cohort5_3k_f_r_d.jsonl",
        "e5_acceptance_zero": root / "results" / "hypothesis_validation_gpu_20260907" / "msd_cohort5_3k_z_natural.jsonl",
        "h3_position_csv": root / "results" / "hypothesis_report_figures_20260910" / "h3_dflash_position_acceptance.csv",
        "h3_pilot": root / "results" / "hypothesis_validation_gpu_20260907" / "SUMMARY.md",
        "h3_dflash_full_stats": root / "results" / "e7_dflash_h3_20260911" / "h31_full" / "analysis" / "h31_dflash_position_statistics.csv",
        "legacy_length": root / "results" / "sparrow_validation_qwen2vl_final_20260819" / "figure1a_statistics.csv",
        "legacy_retention": root / "results" / "sparrow_validation_qwen2vl_final_20260819" / "figure1b_statistics.csv",
        "layer_proxy": root / "results" / "sparrow_validation_qwen2vl_final_20260819" / "figure6_statistics.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hypothesis_report_figures_20260910"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    paths = _default_paths(args.root)
    generate_figures(
        args.output_dir,
        h1b_vdc_paths=paths["h1b_vdc"],
        h1b_mvbench_paths=paths["h1b_mv"],
        h2_paths=paths["h2"],
        e5_attention_real_path=paths["e5_attention_real"],
        e5_attention_zero_path=paths["e5_attention_zero"],
        e5_acceptance_real_path=paths["e5_acceptance_real"],
        e5_acceptance_zero_path=paths["e5_acceptance_zero"],
        h3_position_csv=paths["h3_position_csv"],
        h3_pilot_path=paths["h3_pilot"],
        h3_dflash_full_stats_path=paths["h3_dflash_full_stats"],
        legacy_length_csv=paths["legacy_length"],
        legacy_retention_csv=paths["legacy_retention"],
        layer_proxy_csv=paths["layer_proxy"],
    )
    print(f"Generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
