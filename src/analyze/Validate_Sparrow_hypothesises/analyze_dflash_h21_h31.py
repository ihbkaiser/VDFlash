"""Analyze DFlash E7/H3.1 journals and generate report-ready figures.

The unit of resampling is the VDC sample, not an individual decode round.
This keeps paired condition comparisons valid and makes incomplete journals
visible instead of silently turning them into a pseudo-large sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONDITIONS = ("full", "reduced", "deleted")
H31_CONDITIONS = ("full", "zero", "cut")
PROMPTS = ("natural", "answer_hint")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def bootstrap_mean(values: Sequence[float], *, seed: int = 42, replicates: int = 2000) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    if len(clean) == 1:
        value = clean[0]
        return {"n": 1, "mean": value, "ci_low": value, "ci_high": value}
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, replicates)):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return {"n": len(clean), "mean": sum(clean) / len(clean), "ci_low": low, "ci_high": high}


def paired_difference(left: Mapping[str, float], right: Mapping[str, float]) -> list[float]:
    return [float(left[key]) - float(right[key]) for key in sorted(set(left) & set(right))]


def _valid_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("status") == "ok"]


def _metric_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        grouped[(str(row["prompt_variant"]), str(row["visual_condition"]))].append(float(value))
    result = []
    for (prompt, condition), values in sorted(grouped.items()):
        result.append({"prompt_variant": prompt, "visual_condition": condition, "metric": metric, **bootstrap_mean(values)})
    return result


def analyze_e7(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    valid = _valid_rows(rows)
    stats: dict[str, Any] = {
        "rows_total": len(rows),
        "rows_ok": len(valid),
        "rows_error": len(rows) - len(valid),
        "sample_ids": sorted({str(row.get("sample_id")) for row in valid}),
        "metrics": {},
        "attention": [],
        "attention_grouped": [],
    }
    for metric in ("accepted_prefix_tokens", "accepted_effective_tokens", "lossless"):
        stats["metrics"][metric] = _metric_rows(valid, metric)

    attention_fields = (
        "visual_mass", "question_text_mass", "answer_hint_mass",
        "visual_density", "question_text_density", "answer_hint_density",
        "visual_density_ratio_vs_context_uniform",
        "question_text_density_ratio_vs_context_uniform",
        "answer_hint_density_ratio_vs_context_uniform",
        "answer_hint_density_vs_question_text",
        "answer_hint_mass_share_of_question_plus_hint",
    )
    for row in valid:
        summary = row.get("attention_summary") or {}
        for field in attention_fields:
            if field in summary and summary[field] is not None:
                stats["attention"].append({
                    "sample_id": row.get("sample_id"),
                    "prompt_variant": row.get("prompt_variant"),
                    "visual_condition": row.get("visual_condition"),
                    "metric": field,
                    "value": float(summary[field]),
                    "key_count": float(summary.get(field.replace("_mass", "_key_count"), 0.0))
                    if field.endswith("_mass") else None,
                })

    attention_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for item in stats["attention"]:
        attention_values[(str(item["prompt_variant"]), str(item["visual_condition"]), str(item["metric"]))].append(float(item["value"]))
    for (prompt, condition, metric), values in sorted(attention_values.items()):
        stats["attention_grouped"].append({
            "prompt_variant": prompt,
            "visual_condition": condition,
            "metric": metric,
            **bootstrap_mean(values, seed=120 + len(stats["attention_grouped"])),
        })

    value_maps: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in valid:
        value = row.get("accepted_prefix_tokens")
        if value is not None:
            value_maps[(str(row["sample_id"]), str(row["prompt_variant"]))][str(row["visual_condition"])] = float(value)
    natural_delta = {}
    hint_delta = {}
    for (sample_id, prompt), values in value_maps.items():
        if "full" in values and "deleted" in values:
            (natural_delta if prompt == "natural" else hint_delta)[sample_id] = values["deleted"] - values["full"]
    interaction = paired_difference(hint_delta, natural_delta)
    stats["deleted_minus_full"] = {
        "natural": bootstrap_mean(list(natural_delta.values()), seed=101),
        "answer_hint": bootstrap_mean(list(hint_delta.values()), seed=102),
        "interaction_answer_hint_minus_natural": bootstrap_mean(interaction, seed=103),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "e7_dflash_group_statistics.csv", stats["metrics"]["accepted_prefix_tokens"])
    _write_csv(output_dir / "e7_dflash_attention_statistics.csv", stats["attention"])
    _write_csv(output_dir / "e7_dflash_attention_group_statistics.csv", stats["attention_grouped"])
    return stats


def _position_values(row: Mapping[str, Any], *, bins: int = 8) -> dict[int, list[float]]:
    trace = row.get("acceptance_by_position") or []
    output_length = max(1, len(trace))
    result: dict[int, list[float]] = defaultdict(list)
    for item in trace:
        rate = item.get("rate")
        if rate is None:
            continue
        position = int(item["position"])
        # Position zero is the target anchor and normally has no proposal.
        normalized = min(bins - 1, int((position / output_length) * bins))
        result[normalized].append(float(rate))
    return result


def analyze_h31(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    valid = _valid_rows(rows)
    stats: dict[str, Any] = {
        "rows_total": len(rows),
        "rows_ok": len(valid),
        "rows_error": len(rows) - len(valid),
        "sample_ids": sorted({str(row.get("sample_id")) for row in valid}),
        "position_bins": [],
        "position_deltas": [],
        "condition_summary": [],
        "early_late": {},
    }
    bins = 8
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    sample_condition_bin: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in valid:
        condition = str(row["visual_condition"])
        for bin_index, values in _position_values(row, bins=bins).items():
            sample_condition_bin[(str(row["sample_id"]), condition, bin_index)].append(sum(values) / len(values))
    for (sample_id, condition, bin_index), values in sample_condition_bin.items():
        del sample_id
        grouped[(condition, bin_index)].append(sum(values) / len(values))
    for condition in H31_CONDITIONS:
        for bin_index in range(bins):
            values = grouped.get((condition, bin_index), [])
            summary = bootstrap_mean(values, seed=200 + bin_index)
            stats["position_bins"].append({
                "visual_condition": condition,
                "bin": bin_index,
                "bin_start_fraction": bin_index / bins,
                "bin_end_fraction": (bin_index + 1) / bins,
                **summary,
            })

    # Paired per-position contrasts are the direct statistic for H3.1.  They
    # answer whether Zero/Cut differ from Full at the beginning or end of the
    # answer, rather than relying on visual comparison of three raw curves.
    for condition in ("zero", "cut"):
        for bin_index in range(bins):
            full_values = {
                sample_id: sum(values) / len(values)
                for (sample_id, row_condition, row_bin), values in sample_condition_bin.items()
                if row_condition == "full" and row_bin == bin_index
            }
            condition_values = {
                sample_id: sum(values) / len(values)
                for (sample_id, row_condition, row_bin), values in sample_condition_bin.items()
                if row_condition == condition and row_bin == bin_index
            }
            deltas = paired_difference(condition_values, full_values)
            stats["position_deltas"].append({
                "visual_condition": condition,
                "contrast": f"{condition}_minus_full",
                "bin": bin_index,
                "bin_start_fraction": bin_index / bins,
                "bin_end_fraction": (bin_index + 1) / bins,
                **bootstrap_mean(deltas, seed=500 + 10 * bin_index + (0 if condition == "zero" else 1)),
            })

    for condition in H31_CONDITIONS:
        values = [float(row["accepted_prefix_tokens"]) for row in valid if row.get("visual_condition") == condition and row.get("accepted_prefix_tokens") is not None]
        stats["condition_summary"].append({"visual_condition": condition, **bootstrap_mean(values, seed=250 + H31_CONDITIONS.index(condition))})

    # Early = first half of normalized answer positions; late = second half.
    early_late_values: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {"early": {}, "late": {}}
    )
    for (sample_id, condition, bin_index), values in sample_condition_bin.items():
        early_late_values[condition]["early" if bin_index < bins / 2 else "late"][sample_id] = sum(values) / len(values)
    full = early_late_values["full"]
    for condition in ("zero", "cut"):
        early_delta = paired_difference(early_late_values[condition]["early"], full["early"])
        late_delta = paired_difference(early_late_values[condition]["late"], full["late"])
        common_ids = sorted(
            set(early_late_values[condition]["early"])
            & set(early_late_values[condition]["late"])
            & set(full["early"])
            & set(full["late"])
        )
        late_minus_early = [
            (early_late_values[condition]["late"][sample_id] - full["late"][sample_id])
            - (early_late_values[condition]["early"][sample_id] - full["early"][sample_id])
            for sample_id in common_ids
        ]
        stats["early_late"][condition] = {
            "early_delta_vs_full": bootstrap_mean(early_delta, seed=300),
            "late_delta_vs_full": bootstrap_mean(late_delta, seed=301),
            "late_minus_early": bootstrap_mean(late_minus_early, seed=302),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "h31_dflash_position_statistics.csv", stats["position_bins"])
    _write_csv(output_dir / "h31_dflash_position_delta_statistics.csv", stats["position_deltas"])
    return stats


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_e7(stats: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = stats["metrics"]["accepted_prefix_tokens"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(rows)))
    labels = [f"{r['prompt_variant']}\n{r['visual_condition']}" for r in rows]
    means = [r["mean"] if r["mean"] is not None else float("nan") for r in rows]
    low = [r["ci_low"] if r["ci_low"] is not None else float("nan") for r in rows]
    high = [r["ci_high"] if r["ci_high"] is not None else float("nan") for r in rows]
    ax.errorbar(x, means, yerr=[[m - l for m, l in zip(means, low)], [h - m for h, m in zip(high, means)]], fmt="o", capsize=4)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("accepted proposal tokens (mean, 95% bootstrap CI)")
    ax.set_title("E7-DFlash: Natural versus answer-hint acceptance")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "e7_dflash_acceptance.png", dpi=180)
    fig.savefig(output_dir / "e7_dflash_acceptance.pdf")
    plt.close(fig)

    attention = stats["attention_grouped"]
    panel_metrics = (
        ("visual_mass", "question_text_mass", "answer_hint_mass"),
        ("visual_density_ratio_vs_context_uniform", "question_text_density_ratio_vs_context_uniform", "answer_hint_density_ratio_vs_context_uniform"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    groups = [(prompt, condition) for prompt in PROMPTS for condition in CONDITIONS]
    x = list(range(len(groups)))
    width = 0.25
    for ax, metrics in zip(axes, panel_metrics):
        for offset, metric in enumerate(metrics):
            values = {
                (str(row["prompt_variant"]), str(row["visual_condition"])): row
                for row in attention if row["metric"] == metric
            }
            means = [values.get(group, {}).get("mean", 0.0) or 0.0 for group in groups]
            ax.bar([value + (offset - 1) * width for value in x], means, width=width, label=metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("attention mass")
    axes[1].set_ylabel("density / context-uniform")
    axes[1].set_xticks(x, [f"{prompt}\n{condition}" for prompt, condition in groups], rotation=25, ha="right")
    axes[0].set_title("E7-DFlash: attention mass and density-normalized concentration")
    fig.tight_layout()
    fig.savefig(output_dir / "e7_dflash_attention_mass_density.png", dpi=180)
    fig.savefig(output_dir / "e7_dflash_attention_mass_density.pdf")
    plt.close(fig)


def _plot_h31(stats: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for condition in H31_CONDITIONS:
        rows = [row for row in stats["position_bins"] if row["visual_condition"] == condition]
        x = [row["bin_start_fraction"] + 0.0625 for row in rows]
        y = [row["mean"] if row["mean"] is not None else float("nan") for row in rows]
        low = [row["ci_low"] if row["ci_low"] is not None else float("nan") for row in rows]
        high = [row["ci_high"] if row["ci_high"] is not None else float("nan") for row in rows]
        ax.errorbar(x, y, yerr=[[a - b for a, b in zip(y, low)], [c - a for c, a in zip(high, y)]], marker="o", capsize=3, label=condition)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="half-answer boundary")
    ax.set_xlabel("normalized answer position")
    ax.set_ylabel("proposal acceptance rate")
    ax.set_title("H3.1-DFlash: position-wise acceptance")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "h31_dflash_position_acceptance.png", dpi=180)
    fig.savefig(output_dir / "h31_dflash_position_acceptance.pdf")
    plt.close(fig)


def _plot_h31_deltas(stats: Mapping[str, Any], output_dir: Path) -> None:
    """Plot the paired position-wise contrasts used by the H3.1 decision rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"zero": "#d95f02", "cut": "#1b9e77"}
    labels = {"zero": "zero − full", "cut": "cut − full"}
    for condition in ("zero", "cut"):
        rows = [row for row in stats["position_deltas"] if row["visual_condition"] == condition]
        x = [row["bin_start_fraction"] + 0.0625 for row in rows]
        y = [row["mean"] if row["mean"] is not None else float("nan") for row in rows]
        low = [row["ci_low"] if row["ci_low"] is not None else float("nan") for row in rows]
        high = [row["ci_high"] if row["ci_high"] is not None else float("nan") for row in rows]
        ax.errorbar(
            x,
            y,
            yerr=[[a - b for a, b in zip(y, low)], [c - a for c, a in zip(high, y)]],
            marker="o",
            capsize=3,
            color=colors[condition],
            label=labels[condition],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="half-answer boundary")
    ax.set_xlabel("normalized answer position")
    ax.set_ylabel("paired acceptance-rate difference")
    ax.set_title("H3.1-DFlash: position-wise contrast versus Full")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "h31_dflash_position_delta.png", dpi=180)
    fig.savefig(output_dir / "h31_dflash_position_delta.pdf")
    plt.close(fig)


def _plot_h31_statistical_summary(stats: Mapping[str, Any], output_dir: Path) -> None:
    """Plot the sample-level statistics used to decide H3.1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Overall accepted-prefix statistic by visual condition.
    condition_labels = {"full": "Full", "zero": "Zero", "cut": "Cut"}
    condition_rows = list(stats["condition_summary"])
    x = list(range(len(condition_rows)))
    means = [float(row["mean"]) for row in condition_rows]
    low = [mean - float(row["ci_low"]) for mean, row in zip(means, condition_rows)]
    high = [float(row["ci_high"]) - mean for mean, row in zip(means, condition_rows)]
    axes[0].errorbar(x, means, yerr=[low, high], fmt="o", capsize=4, color="#2878b5")
    axes[0].set_xticks(x, [condition_labels[str(row["visual_condition"])] for row in condition_rows])
    axes[0].set_ylabel("accepted proposal tokens")
    axes[0].set_title("Overall acceptance (n=50)")
    axes[0].grid(axis="y", alpha=0.25)

    # Early/late paired deltas versus Full and the registered interaction.
    conditions = ("zero", "cut")
    labels = ("Zero−Full", "Cut−Full")
    xpos = [0, 1]
    width = 0.34
    for offset, (condition, label, color) in enumerate(
        zip(conditions, labels, ("#d95f02", "#1b9e77"))
    ):
        summary = stats["early_late"][condition]
        row_means = [
            float(summary["early_delta_vs_full"]["mean"]),
            float(summary["late_delta_vs_full"]["mean"]),
        ]
        row_low = [
            row_means[index] - float(summary[key]["ci_low"])
            for index, key in enumerate(("early_delta_vs_full", "late_delta_vs_full"))
        ]
        row_high = [
            float(summary[key]["ci_high"]) - row_means[index]
            for index, key in enumerate(("early_delta_vs_full", "late_delta_vs_full"))
        ]
        centers = [value + (offset - 0.5) * width for value in xpos]
        axes[1].errorbar(
            centers,
            row_means,
            yerr=[row_low, row_high],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(xpos, ("Early half", "Late half"))
    axes[1].set_ylabel("paired Δ acceptance vs Full")
    axes[1].set_title("Early/late contrast (n=50)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    # Registered H3.1 decision statistic: late delta minus early delta.
    interaction_means = [
        float(stats["early_late"][condition]["late_minus_early"]["mean"])
        for condition in conditions
    ]
    interaction_low = [
        mean - float(stats["early_late"][condition]["late_minus_early"]["ci_low"])
        for mean, condition in zip(interaction_means, conditions)
    ]
    interaction_high = [
        float(stats["early_late"][condition]["late_minus_early"]["ci_high"]) - mean
        for mean, condition in zip(interaction_means, conditions)
    ]
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].errorbar(
        [0, 1],
        interaction_means,
        yerr=[interaction_low, interaction_high],
        fmt="o",
        capsize=4,
        color="#756bb1",
    )
    axes[2].set_xticks([0, 1], labels)
    axes[2].set_ylabel("late Δ − early Δ")
    axes[2].set_title("Registered H3.1 interaction")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("H3.1-DFlash: statistical summary with bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(output_dir / "h31_dflash_statistical_summary.png", dpi=180)
    fig.savefig(output_dir / "h31_dflash_statistical_summary.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e7", type=Path)
    parser.add_argument("--h31", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    if args.e7:
        e7_stats = analyze_e7(read_jsonl(args.e7), args.output_dir)
        _plot_e7(e7_stats, args.output_dir)
        result["e7"] = e7_stats
    if args.h31:
        h31_stats = analyze_h31(read_jsonl(args.h31), args.output_dir)
        _plot_h31(h31_stats, args.output_dir)
        _plot_h31_deltas(h31_stats, args.output_dir)
        _plot_h31_statistical_summary(h31_stats, args.output_dir)
        result["h3_1"] = h31_stats
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"rows_total": value["rows_total"], "rows_ok": value["rows_ok"], "rows_error": value["rows_error"]} for key, value in result.items()}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
