"""Paired analysis and plots for the expanded E5 Real/Zero experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .paper_statistics import summarize


CONDITIONS = ("full", "reduced25", "deleted")
CONDITION_LABELS = {
    "full": "Full",
    "reduced25": "Reduced 25%",
    "deleted": "Deleted",
}


def condition_key(row: Mapping[str, Any]) -> str:
    """Normalize E5 attention and acceptance condition names."""

    value = row.get("retention_percentage")
    if value is not None:
        try:
            retention = float(value)
        except (TypeError, ValueError):
            retention = math.nan
        if math.isclose(retention, 100.0):
            return "full"
        if math.isclose(retention, 25.0):
            return "reduced25"
        if math.isclose(retention, 0.0):
            return "deleted"
    raw = str(row.get("visual_condition") or row.get("condition") or "").lower()
    if raw in {"full", "f"}:
        return "full"
    if raw in {"reduced", "reduced25", "retention", "r25", "r"}:
        return "reduced25"
    if raw in {"deleted", "delete", "zero", "d0", "d", "cut"}:
        return "deleted"
    raise ValueError(f"Unknown E5 condition: {raw!r}")


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _key_for_group(values: Sequence[Any]) -> Any:
    return values[0] if len(values) == 1 else tuple(values)


def pair_attention_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair summary attention rows by sample, condition and query policy.

    The returned difference is always ``Real - Zero``.  Incomplete or
    duplicated groups are omitted; the caller can compare the returned pair
    count with the expected Cartesian product for a completeness audit.
    """

    groups: dict[tuple[str, str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"real": [], "zero": []}
    )
    for row in rows:
        if row.get("modality") != "summary":
            continue
        mode = str(row.get("visual_value_mode") or "real").lower()
        if mode not in {"real", "zero"}:
            continue
        key = (
            str(row.get("sample_id")),
            condition_key(row),
            str(row.get("attention_policy")),
        )
        groups[key][mode].append(row)

    result: list[dict[str, Any]] = []
    metric_names = {
        "visual_mass": "visual_mass",
        "text_mass": "text_mass",
        "instruction_mass": "instruction_mass",
        "visual_attention_density": "visual_density",
        "text_attention_density": "text_density",
        "instruction_attention_density": "instruction_density",
        "visual_density_ratio": "visual_density_ratio",
        "text_density_ratio": "text_density_ratio",
        "instruction_density_ratio": "instruction_density_ratio",
        "visual_effective_key_count": "visual_effective_key_count",
        "text_effective_key_count": "text_effective_key_count",
        "instruction_effective_key_count": "instruction_effective_key_count",
    }
    for (sample_id, condition, policy), modes in sorted(groups.items()):
        if len(modes["real"]) != 1 or len(modes["zero"]) != 1:
            continue
        real = modes["real"][0]
        zero = modes["zero"][0]
        paired: dict[str, Any] = {
            "sample_id": sample_id,
            "visual_condition": condition,
            "attention_policy": policy,
        }
        for source_name, output_name in metric_names.items():
            real_value = _finite(real.get(source_name))
            zero_value = _finite(zero.get(source_name))
            paired[f"attention_{output_name}_real"] = real_value
            paired[f"attention_{output_name}_zero"] = zero_value
            paired[f"attention_{output_name}_real_minus_zero"] = (
                real_value - zero_value
                if real_value is not None and zero_value is not None
                else None
            )
        paired["value_ablation_logit_max_abs_delta"] = _finite(
            real.get("value_ablation_logit_max_abs_delta")
        )
        paired["value_ablation_kl_forward"] = _finite(real.get("value_ablation_kl_forward"))
        paired["value_ablation_top1_match"] = real.get("value_ablation_top1_match")
        result.append(paired)
    return result


def pair_acceptance_rows(
    real_rows: Iterable[Mapping[str, Any]],
    zero_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pair Real/Zero acceptance rows by sample and visual condition."""

    groups: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"real": [], "zero": []}
    )
    for mode, rows in (("real", real_rows), ("zero", zero_rows)):
        for row in rows:
            if _finite(row.get("accepted_prefix_tokens")) is None:
                continue
            key = (str(row.get("sample_id")), condition_key(row))
            groups[key][mode].append(row)
    result: list[dict[str, Any]] = []
    for (sample_id, condition), modes in sorted(groups.items()):
        if len(modes["real"]) != 1 or len(modes["zero"]) != 1:
            continue
        real = modes["real"][0]
        zero = modes["zero"][0]
        real_value = float(real["accepted_prefix_tokens"])
        zero_value = float(zero["accepted_prefix_tokens"])
        result.append({
            "sample_id": sample_id,
            "visual_condition": condition,
            "accepted_prefix_tokens_real": real_value,
            "accepted_prefix_tokens_zero": zero_value,
            "accepted_prefix_tokens_real_minus_zero": real_value - zero_value,
            "lossless_real": real.get("lossless"),
            "lossless_zero": zero.get("lossless"),
            "lossless_real_minus_zero": int(bool(real.get("lossless"))) - int(bool(zero.get("lossless"))),
        })
    return result


def summarize_paired(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    value_field: str,
) -> dict[Any, dict[str, Any]]:
    """Return descriptive and bootstrap statistics for paired differences."""

    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(value_field))
        if value is None:
            continue
        groups[tuple(row.get(field) for field in group_fields)].append(value)
    return {
        _key_for_group(key): summarize(values, replicates=5000, seed=42)
        for key, values in sorted(groups.items(), key=lambda item: item[0])
    }


def _summary_for_field(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    value_field: str,
) -> dict[Any, dict[str, Any]]:
    return summarize_paired(rows, group_fields=group_fields, value_field=value_field)


def _json_keyed(summary: Mapping[Any, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Convert tuple group keys to stable JSON object keys."""

    result: dict[str, Mapping[str, Any]] = {}
    for key, value in summary.items():
        if isinstance(key, tuple):
            rendered = "|".join(str(part) for part in key)
        else:
            rendered = str(key)
        result[rendered] = value
    return result


def _difference_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
    value_field: str,
) -> dict[str, dict[str, Any]]:
    """Report exact equality and maximum paired difference by group."""

    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(value_field))
        if value is not None:
            groups[tuple(row.get(field) for field in group_fields)].append(value)
    result = {}
    for key, values in sorted(groups.items(), key=lambda item: item[0]):
        result_key = "|".join(str(part) for part in key)
        result[result_key] = {
            "n": len(values),
            "exact_equal_count": sum(value == 0.0 for value in values),
            "exact_equal_rate": sum(value == 0.0 for value in values) / len(values),
            "max_abs_difference": max(abs(value) for value in values),
        }
    return result


def _lossless_rates(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        condition = condition_key(row)
        if isinstance(row.get("lossless"), bool):
            groups[condition].append(row["lossless"])
    return {
        condition: {
            "n": len(values),
            "lossless_count": sum(values),
            "lossless_rate": sum(values) / len(values),
        }
        for condition, values in sorted(groups.items())
    }


def build_statistics(
    attention_rows: Sequence[Mapping[str, Any]],
    acceptance_real_rows: Sequence[Mapping[str, Any]],
    acceptance_zero_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the complete E5 coverage and paired-statistics object."""

    attention_summaries = [row for row in attention_rows if row.get("modality") == "summary"]
    attention_pairs = pair_attention_rows(attention_summaries)
    acceptance_pairs = pair_acceptance_rows(acceptance_real_rows, acceptance_zero_rows)
    attention_grouped: dict[str, Any] = {}
    attention_metrics = (
        "visual_mass",
        "text_mass",
        "instruction_mass",
        "visual_density",
        "text_density",
        "instruction_density",
        "visual_density_ratio",
        "text_density_ratio",
        "instruction_density_ratio",
        "visual_effective_key_count",
        "text_effective_key_count",
        "instruction_effective_key_count",
    )
    for metric in attention_metrics:
        attention_grouped[metric] = {
            "paired_difference_real_minus_zero": summarize_paired(
                attention_pairs,
                group_fields=("visual_condition", "attention_policy"),
                value_field=f"attention_{metric}_real_minus_zero",
            ),
            "real": summarize_paired(
                attention_pairs,
                group_fields=("visual_condition", "attention_policy"),
                value_field=f"attention_{metric}_real",
            ),
            "zero": summarize_paired(
                attention_pairs,
                group_fields=("visual_condition", "attention_policy"),
                value_field=f"attention_{metric}_zero",
            ),
        }
    acceptance_grouped = {
        "accepted_prefix_tokens": {
            "paired_difference_real_minus_zero": summarize_paired(
                acceptance_pairs,
                group_fields=("visual_condition",),
                value_field="accepted_prefix_tokens_real_minus_zero",
            ),
            "real": summarize_paired(
                acceptance_pairs,
                group_fields=("visual_condition",),
                value_field="accepted_prefix_tokens_real",
            ),
            "zero": summarize_paired(
                acceptance_pairs,
                group_fields=("visual_condition",),
                value_field="accepted_prefix_tokens_zero",
            ),
        },
        "lossless_difference_real_minus_zero": summarize_paired(
            acceptance_pairs,
            group_fields=("visual_condition",),
            value_field="lossless_real_minus_zero",
        ),
    }
    for metric in attention_grouped:
        for mode in ("paired_difference_real_minus_zero", "real", "zero"):
            attention_grouped[metric][mode] = _json_keyed(attention_grouped[metric][mode])
    acceptance_grouped["accepted_prefix_tokens"]["paired_difference_real_minus_zero"] = _json_keyed(
        acceptance_grouped["accepted_prefix_tokens"]["paired_difference_real_minus_zero"]
    )
    acceptance_grouped["accepted_prefix_tokens"]["real"] = _json_keyed(
        acceptance_grouped["accepted_prefix_tokens"]["real"]
    )
    acceptance_grouped["accepted_prefix_tokens"]["zero"] = _json_keyed(
        acceptance_grouped["accepted_prefix_tokens"]["zero"]
    )
    acceptance_grouped["lossless_difference_real_minus_zero"] = _json_keyed(
        acceptance_grouped["lossless_difference_real_minus_zero"]
    )
    attention_exact = {}
    for metric in ("visual_mass", "visual_density", "visual_density_ratio"):
        attention_exact[metric] = _difference_diagnostics(
            attention_pairs,
            group_fields=("visual_condition", "attention_policy"),
            value_field=f"attention_{metric}_real_minus_zero",
        )
    attention_ids = {str(row.get("sample_id")) for row in attention_summaries}
    attention_real_zero_ids = {
        pair["sample_id"] for pair in attention_pairs
    }
    acceptance_real_ids = {str(row.get("sample_id")) for row in acceptance_real_rows}
    acceptance_zero_ids = {str(row.get("sample_id")) for row in acceptance_zero_rows}
    expected_sample_count = max(
        len(attention_ids), len(acceptance_real_ids), len(acceptance_zero_ids)
    )
    return {
        "protocol": {
            "dataset": "VDC test subset",
            "target_visual_tokens": 3000,
            "retention_conditions": ["full", "reduced25", "deleted"],
            "attention_policies": ["last_instruction", "all_text"],
            "value_modes": ["real", "zero"],
            "acceptance_max_new_tokens": 64,
            "paired_difference_convention": "Real - Zero",
            "density_definition": "modality mass / mean eligible strict-preceding key count",
        },
        "coverage": {
            "attention_summary_rows": len(attention_summaries),
            "attention_unique_sample_ids": len(attention_ids),
            "expected_sample_count": expected_sample_count,
            "attention_expected_rows": expected_sample_count * 3 * 2 * 2,
            "attention_paired_rows": len(attention_pairs),
            "attention_paired_unique_sample_ids": len(attention_real_zero_ids),
            "acceptance_real_rows": len(acceptance_real_rows),
            "acceptance_zero_rows": len(acceptance_zero_rows),
            "acceptance_expected_rows": expected_sample_count * 3,
            "acceptance_real_unique_sample_ids": len(acceptance_real_ids),
            "acceptance_zero_unique_sample_ids": len(acceptance_zero_ids),
            "acceptance_paired_rows": len(acceptance_pairs),
            "acceptance_paired_unique_sample_ids": len({row["sample_id"] for row in acceptance_pairs}),
            "attention_error_rows": sum(row.get("status") == "error" for row in attention_rows),
            "acceptance_real_error_rows": sum(row.get("status") == "error" for row in acceptance_real_rows),
            "acceptance_zero_error_rows": sum(row.get("status") == "error" for row in acceptance_zero_rows),
        },
        "attention": attention_grouped,
        "exact_pair_diagnostics": {
            "attention": attention_exact,
            "acceptance": {
                "accepted_prefix_tokens": _difference_diagnostics(
                    acceptance_pairs,
                    group_fields=("visual_condition",),
                    value_field="accepted_prefix_tokens_real_minus_zero",
                ),
            },
        },
        "acceptance": acceptance_grouped,
        "lossless_rates": {
            "real": _lossless_rates(acceptance_real_rows),
            "zero": _lossless_rates(acceptance_zero_rows),
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    plt = _plt()
    plt.close(fig)
    return paths


def _mean_or_nan(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else math.nan


def plot_attention_density(
    pairs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Plot mass, density, density ratio and key count for the visual group."""

    plt = _plt()
    output = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        ("attention_visual_mass", "Visual attention mass", False),
        ("attention_visual_density", "Visual attention density", True),
        ("attention_visual_density_ratio", "Visual density ratio vs uniform", False),
        ("attention_visual_effective_key_count", "Eligible visual key count", False),
    )
    colors = {"real": "#4C78A8", "zero": "#E45756"}
    for axis, (field, title, log_y) in zip(axes.flat, panels):
        for mode, label in (("real", "Real"), ("zero", "Zero")):
            values = []
            for condition in CONDITIONS:
                sample_values = [
                    float(row[f"{field}_{mode}"])
                    for row in pairs
                    if row.get("visual_condition") == condition
                    and row.get("attention_policy") == "last_instruction"
                    and row.get(f"{field}_{mode}") is not None
                ]
                values.append(_mean_or_nan(sample_values))
            axis.plot(
                range(len(CONDITIONS)),
                values,
                marker="o",
                linewidth=2,
                color=colors[mode],
                label=label,
            )
        axis.set_title(title)
        axis.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS])
        axis.grid(alpha=0.25)
        if log_y:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if field == "attention_visual_density_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Uniform")
        axis.legend(frameon=False)
    fig.suptitle("E5 expanded — visual attention mass versus per-key density (n=33)")
    fig.tight_layout()
    return _save_figure(fig, output, "h1b_e5_attention_density")


def plot_real_zero_paired(
    attention_pairs: Sequence[Mapping[str, Any]],
    acceptance_pairs: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Plot paired Real-minus-Zero effects with bootstrap intervals."""

    plt = _plt()
    output = Path(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = (
        (
            axes[0],
            attention_pairs,
            "attention_visual_mass_real_minus_zero",
            "Attention mass difference\nReal − Zero",
            ("visual_condition", "attention_policy"),
        ),
        (
            axes[1],
            attention_pairs,
            "attention_visual_density_ratio_real_minus_zero",
            "Visual density-ratio difference\nReal − Zero",
            ("visual_condition", "attention_policy"),
        ),
        (
            axes[2],
            acceptance_pairs,
            "accepted_prefix_tokens_real_minus_zero",
            "Accepted-prefix difference\nReal − Zero",
            ("visual_condition",),
        ),
    )
    colors = {"last_instruction": "#4C78A8", "all_text": "#54A24B"}
    for axis, rows, field, title, _ in panels:
        if rows is acceptance_pairs:
            summaries = summarize_paired(rows, group_fields=("visual_condition",), value_field=field)
            x_labels = [CONDITION_LABELS[c] for c in CONDITIONS]
            means = [summaries.get(c, {}).get("mean", math.nan) for c in CONDITIONS]
            lows = [summaries.get(c, {}).get("ci95_low", math.nan) for c in CONDITIONS]
            highs = [summaries.get(c, {}).get("ci95_high", math.nan) for c in CONDITIONS]
            axis.errorbar(
                range(len(CONDITIONS)),
                means,
                yerr=[[m - l for m, l in zip(means, lows)], [h - m for h, m in zip(highs, means)]],
                marker="o",
                linewidth=2,
                color="#E45756",
                capsize=4,
                label="Acceptance",
            )
        else:
            for policy in ("last_instruction", "all_text"):
                summaries = summarize_paired(
                    rows,
                    group_fields=("visual_condition", "attention_policy"),
                    value_field=field,
                )
                means = []
                lows = []
                highs = []
                for condition in CONDITIONS:
                    summary = summaries.get((condition, policy), {})
                    means.append(summary.get("mean", math.nan))
                    lows.append(summary.get("ci95_low", math.nan))
                    highs.append(summary.get("ci95_high", math.nan))
                axis.errorbar(
                    range(len(CONDITIONS)),
                    means,
                    yerr=[[m - l for m, l in zip(means, lows)], [h - m for h, m in zip(highs, means)]],
                    marker="o",
                    linewidth=2,
                    color=colors[policy],
                    capsize=4,
                    label=policy.replace("_", " "),
                )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(range(len(CONDITIONS)), [CONDITION_LABELS[c] for c in CONDITIONS], rotation=15)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("E5 expanded — paired Real/Zero effects (95% bootstrap CI)")
    fig.tight_layout()
    return _save_figure(fig, output, "h1b_e5_real_zero_paired")


def analyze_files(
    *,
    attention_path: str | Path,
    acceptance_real_path: str | Path,
    acceptance_zero_path: str | Path,
    output_dir: str | Path,
    figure_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze JSONL artifacts and write CSV, JSON and figure outputs."""

    def load(path: str | Path) -> list[dict[str, Any]]:
        rows = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    attention_rows = load(attention_path)
    acceptance_real_rows = load(acceptance_real_path)
    acceptance_zero_rows = load(acceptance_zero_path)
    attention_pairs = pair_attention_rows(attention_rows)
    acceptance_pairs = pair_acceptance_rows(acceptance_real_rows, acceptance_zero_rows)
    statistics_object = build_statistics(
        attention_rows, acceptance_real_rows, acceptance_zero_rows
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "e5_attention_pairs.csv", attention_pairs)
    _write_csv(output / "e5_acceptance_pairs.csv", acceptance_pairs)
    (output / "e5_expanded_statistics.json").write_text(
        json.dumps(statistics_object, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if figure_dir is not None:
        figure_output = Path(figure_dir)
        figures = plot_attention_density(attention_pairs, figure_output)
        figures += plot_real_zero_paired(attention_pairs, acceptance_pairs, figure_output)
        statistics_object["figures"] = [str(path) for path in figures]
        (output / "e5_expanded_statistics.json").write_text(
            json.dumps(statistics_object, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return statistics_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", required=True, type=Path)
    parser.add_argument("--acceptance-real", required=True, type=Path)
    parser.add_argument("--acceptance-zero", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    analyze_files(
        attention_path=args.attention,
        acceptance_real_path=args.acceptance_real,
        acceptance_zero_path=args.acceptance_zero,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
    )
    print(f"Wrote E5 analysis to {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
