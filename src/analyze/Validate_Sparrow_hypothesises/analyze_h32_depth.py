"""Analyze the H3.2 DFlash depth-controlled experiment.

The causal unit is one sample.  Rows at each draft depth are paired by
``dataset/corpus/sample_id/condition``; bootstrap resampling never treats
individual answer tokens or speculative rounds as independent observations.
Depth-3 minus depth-1 remains the registered primary interaction; additional
depths, such as depth-5, are reported through pairwise extensions.
Full minus Zero is the primary visual-information contrast because Zero keeps
the visual positions in the denominator/context but removes their values.
Full minus Cut is reported as a secondary context-length-sensitive contrast.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CONDITIONS = ("full", "zero", "cut")
METRIC = "accepted_effective_tokens"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def bootstrap_mean(
    values: Sequence[float], *, seed: int, replicates: int = 4000
) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = random.Random(seed)
    means = [
        sum(clean[rng.randrange(len(clean))] for _ in clean) / len(clean)
        for _ in range(max(1, replicates))
    ]
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return {
        "n": len(clean),
        "mean": mean,
        "ci_low": means[low_index],
        "ci_high": means[high_index],
    }


def _cell(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row.get("dataset", "unknown")),
        str(row.get("training_corpus", "unknown")),
        int(row.get("draft_depth")),
        str(row.get("sample_id")),
        str(row.get("visual_condition")),
    )


def _valid_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("status") == "ok" and row.get(METRIC) is not None
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _group_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    visual_counts: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["training_corpus"]),
            int(row["draft_depth"]),
            str(row["visual_condition"]),
        )
        grouped[key].append(float(row[METRIC]))
        if row.get("actual_visual_tokens") is not None:
            visual_counts[key].append(float(row["actual_visual_tokens"]))

    output: list[dict[str, Any]] = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        dataset, corpus, depth, condition = key
        stats = bootstrap_mean(values, seed=1000 + index)
        count_stats = bootstrap_mean(visual_counts.get(key, []), seed=2000 + index)
        output.append(
            {
                "dataset": dataset,
                "training_corpus": corpus,
                "draft_depth": depth,
                "visual_condition": condition,
                "metric": METRIC,
                "actual_visual_tokens_mean": count_stats["mean"],
                **stats,
            }
        )
    return output


def _paired_statistics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, int, str, str], float] = {}
    for row in rows:
        dataset, corpus, depth, sample_id, condition = _cell(row)
        values[(dataset, corpus, depth, sample_id, condition)] = float(row[METRIC])

    output: list[dict[str, Any]] = []
    group_keys = sorted({key[:2] for key in values})
    contrasts = (("full", "zero", "full_minus_zero"), ("full", "cut", "full_minus_cut"))
    for dataset, corpus in group_keys:
        depths = sorted(
            {
                key[2]
                for key in values
                if key[:2] == (dataset, corpus)
            }
        )
        for left_condition, right_condition, contrast_name in contrasts:
            for depth in depths:
                sample_ids = sorted(
                    {
                        key[3]
                        for key in values
                        if key[:3] == (dataset, corpus, depth)
                        and key[4] == left_condition
                    }
                    & {
                        key[3]
                        for key in values
                        if key[:3] == (dataset, corpus, depth)
                        and key[4] == right_condition
                    }
                )
                deltas = [
                    values[(dataset, corpus, depth, sample_id, left_condition)]
                    - values[(dataset, corpus, depth, sample_id, right_condition)]
                    for sample_id in sample_ids
                ]
                output.append(
                    {
                        "dataset": dataset,
                        "training_corpus": corpus,
                        "draft_depth": depth,
                        "contrast": contrast_name,
                        "left_condition": left_condition,
                        "right_condition": right_condition,
                        **bootstrap_mean(deltas, seed=3000 + depth),
                    }
                )

        # Pairwise difference-in-differences.  Depth3-depth1 is primary;
        # depth5-depth1 and depth5-depth3 are extensions.
        for lower_depth, higher_depth in combinations(depths, 2):
            common = sorted(
                {
                    key[3]
                    for key in values
                    if key[:3] == (dataset, corpus, lower_depth)
                    and key[4] == "full"
                }
                & {
                    key[3]
                    for key in values
                    if key[:3] == (dataset, corpus, lower_depth)
                    and key[4] == "zero"
                }
                & {
                    key[3]
                    for key in values
                    if key[:3] == (dataset, corpus, higher_depth)
                    and key[4] == "full"
                }
                & {
                    key[3]
                    for key in values
                    if key[:3] == (dataset, corpus, higher_depth)
                    and key[4] == "zero"
                }
            )
            interaction = [
                (
                    values[(dataset, corpus, higher_depth, sample_id, "full")]
                    - values[(dataset, corpus, higher_depth, sample_id, "zero")]
                )
                - (
                    values[(dataset, corpus, lower_depth, sample_id, "full")]
                    - values[(dataset, corpus, lower_depth, sample_id, "zero")]
                )
                for sample_id in common
            ]
            output.append(
                {
                    "dataset": dataset,
                    "training_corpus": corpus,
                    "draft_depth": f"{higher_depth}_minus_{lower_depth}",
                    "contrast": "depth_interaction_full_minus_zero",
                    "left_condition": f"full_minus_zero@depth{higher_depth}",
                    "right_condition": f"full_minus_zero@depth{lower_depth}",
                    **bootstrap_mean(
                        interaction,
                        seed=4000 + higher_depth * 10 + lower_depth,
                    ),
                }
            )
    return output


def _hash_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_sample: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    by_depth: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["dataset"]), str(row["training_corpus"]), str(row["sample_id"]))
        target_hash = row.get("target_output_hash")
        if target_hash:
            by_sample[key].add(str(target_hash))
            by_depth[key + (int(row["draft_depth"]),)].add(str(target_hash))
    sample_mismatches = [key for key, hashes in by_sample.items() if len(hashes) > 1]
    depth_mismatches = [key for key, hashes in by_depth.items() if len(hashes) > 1]
    return {
        "samples_with_multiple_target_hashes": len(sample_mismatches),
        "depth_cells_with_multiple_target_hashes": len(depth_mismatches),
        "target_hash_consistent": not sample_mismatches,
        "target_hash_consistent_within_depth": not depth_mismatches,
        "sample_mismatch_keys": [list(key) for key in sample_mismatches[:20]],
    }


def _plot_group_statistics(stats: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    groups = sorted({(str(row["dataset"]), str(row["training_corpus"])) for row in stats})
    if not groups:
        return
    figure, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 4.6), squeeze=False)
    for axis, (dataset, corpus) in zip(axes[0], groups):
        subset = [row for row in stats if (str(row["dataset"]), str(row["training_corpus"])) == (dataset, corpus)]
        for condition, color in zip(CONDITIONS, ("#2878b5", "#d95f02", "#7570b3")):
            points = sorted(
                [row for row in subset if row["visual_condition"] == condition],
                key=lambda row: int(row["draft_depth"]),
            )
            if not points:
                continue
            x = [int(row["draft_depth"]) for row in points]
            y = [float(row["mean"]) for row in points]
            low = [y_i - float(row["ci_low"]) for y_i, row in zip(y, points)]
            high = [float(row["ci_high"]) - y_i for y_i, row in zip(y, points)]
            axis.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=condition, color=color)
        axis.set_xticks(sorted({int(row["draft_depth"]) for row in subset}))
        axis.set_xlabel("Draft depth (number of layers)")
        axis.set_ylabel("Accepted effective tokens (τ_eff)")
        axis.set_title(f"{dataset.upper()} · {corpus}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    figure.suptitle("H3.2: acceptance by DFlash draft depth")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_interactions(paired: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in paired
        if row["contrast"] == "depth_interaction_full_minus_zero"
        and row.get("mean") is not None
    ]
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["dataset"]),
            str(row["training_corpus"]),
            int(str(row["draft_depth"]).split("_", 1)[0]),
        ),
    )
    labels = [
        f"{str(row['dataset']).upper()}\n{row['training_corpus']}\n"
        f"D{str(row['draft_depth']).replace('_minus_', '−D')}"
        for row in rows
    ]
    means = [float(row["mean"]) for row in rows]
    lows = [mean - float(row["ci_low"]) for mean, row in zip(means, rows)]
    highs = [float(row["ci_high"]) - mean for mean, row in zip(means, rows)]
    figure, axis = plt.subplots(figsize=(7.5, 4.7))
    axis.axhline(0.0, color="black", linewidth=0.8)
    colors = [
        "#1b9e77" if str(row["draft_depth"]).startswith("3_") else "#7570b3"
        for row in rows
    ]
    for index, (mean, low, high, color) in enumerate(zip(means, lows, highs, colors)):
        axis.errorbar(
            index,
            mean,
            yerr=[[low], [high]],
            fmt="o",
            capsize=4,
            color=color,
        )
    axis.set_xticks(range(len(rows)), labels)
    axis.set_ylabel("(Full−Zero) higher depth − (Full−Zero) lower depth")
    axis.set_title("H3.2: pairwise depth interactions")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_visual_sensitivity(paired: Sequence[Mapping[str, Any]], output: Path) -> None:
    """Plot Full−Zero at each depth, the primitive contrast behind H3.2."""
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in paired
        if row["contrast"] == "full_minus_zero"
        and row.get("mean") is not None
        and isinstance(row.get("draft_depth"), int)
    ]
    groups = sorted({(str(row["dataset"]), str(row["training_corpus"])) for row in rows})
    if not rows or not groups:
        return
    figure, axes = plt.subplots(1, len(groups), figsize=(5.4 * len(groups), 4.6), squeeze=False)
    for axis, group in zip(axes[0], groups):
        subset = sorted(
            [row for row in rows if (str(row["dataset"]), str(row["training_corpus"])) == group],
            key=lambda row: int(row["draft_depth"]),
        )
        x = [int(row["draft_depth"]) for row in subset]
        y = [float(row["mean"]) for row in subset]
        low = [y_i - float(row["ci_low"]) for y_i, row in zip(y, subset)]
        high = [float(row["ci_high"]) - y_i for y_i, row in zip(y, subset)]
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.errorbar(x, y, yerr=[low, high], marker="o", capsize=4, color="#1b9e77")
        axis.set_xticks(sorted({int(row["draft_depth"]) for row in subset}))
        axis.set_xlabel("Draft depth")
        axis.set_ylabel("Full − Zero (τ_eff)")
        axis.set_title(f"{group[0].upper()} · {group[1]}")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("H3.2: visual-value sensitivity by draft depth")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze(paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    all_rows = [row for path in paths for row in read_jsonl(path)]
    valid = _valid_rows(all_rows)
    stats = _group_statistics(valid)
    paired = _paired_statistics(valid)
    errors = [row for row in all_rows if row.get("status") != "ok"]
    expected_cells = defaultdict(set)
    observed_cells = defaultdict(set)
    for row in all_rows:
        group = (str(row.get("dataset")), str(row.get("training_corpus")))
        expected_cells[group].add((int(row.get("draft_depth")), str(row.get("sample_id")), str(row.get("visual_condition"))))
    for row in valid:
        group = (str(row["dataset"]), str(row["training_corpus"]))
        observed_cells[group].add((int(row["draft_depth"]), str(row["sample_id"]), str(row["visual_condition"])))
    coverage = {
        f"{dataset}/{corpus}": {
            "rows_total": len(expected_cells[(dataset, corpus)]),
            "rows_ok": len(observed_cells[(dataset, corpus)]),
            "complete": expected_cells[(dataset, corpus)] == observed_cells[(dataset, corpus)],
        }
        for dataset, corpus in sorted(expected_cells)
    }
    result = {
        "experiment": "H3.2",
        "primary_metric": METRIC,
        "primary_contrast": "depth_interaction_full_minus_zero",
        "rows_total": len(all_rows),
        "rows_ok": len(valid),
        "rows_error": len(errors),
        "coverage": coverage,
        "target_hash_audit": _hash_audit(valid),
        "group_statistics": stats,
        "paired_statistics": paired,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "h32_group_statistics.csv", stats)
    _write_csv(output_dir / "h32_paired_statistics.csv", paired)
    (output_dir / "h32_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_group_statistics(stats, output_dir / "h32_acceptance_by_depth.png")
    _plot_interactions(paired, output_dir / "h32_primary_depth_interaction.png")
    _plot_visual_sensitivity(paired, output_dir / "h32_visual_sensitivity.png")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(args.inputs, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["rows_error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
