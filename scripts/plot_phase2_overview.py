#!/usr/bin/env python3
"""Create compact overview plots from the descriptive Phase 2 summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "full": "#2563eb",
    "zero": "#f59e0b",
    "cut": "#dc2626",
    "prefix": "#475569",
    "effective": "#0f766e",
}


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def lookup(rows: list[dict[str, str]], group: dict[str, str], metric: str) -> tuple[float, float] | None:
    for row in rows:
        if row.get("metric") != metric:
            continue
        if any(row.get(key, "") != value for key, value in group.items()):
            continue
        try:
            if int(float(row.get("n", "0"))) <= 0:
                return None
            return float(row["mean"]), float(row.get("std", "0") or 0.0)
        except (KeyError, TypeError, ValueError):
            return None
    return None


def acceptance_length_series(
    rows: list[dict[str, str]], groups: list[dict[str, str]]
) -> dict[str, list[tuple[float, float] | None]]:
    """Return prefix and effective acceptance-length series for ordered groups.

    ``accepted_prefix_tokens`` is the number of accepted draft proposals k.
    ``accepted_effective_tokens`` is k+1, including the target bonus token. MSD
    does not emit the latter field, so it is derived from the prefix series.
    """

    prefix = [lookup(rows, group, "accepted_prefix_tokens") for group in groups]
    effective = [lookup(rows, group, "accepted_effective_tokens") for group in groups]
    derived_effective = []
    for prefix_value, effective_value in zip(prefix, effective):
        if effective_value is not None:
            derived_effective.append(effective_value)
        elif prefix_value is not None:
            derived_effective.append((prefix_value[0] + 1.0, prefix_value[1]))
        else:
            derived_effective.append(None)
    return {
        "accepted_prefix_tokens": prefix,
        "accepted_effective_tokens": derived_effective,
    }


def _draw_acceptance_length_lines(
    ax,
    rows: list[dict[str, str]],
    groups: list[dict[str, str]],
    labels: list[str],
    *,
    title: str,
    xlabel: str,
    color_prefix: str = COLORS["prefix"],
    color_effective: str = COLORS["effective"],
    show_legend: bool = True,
):
    series = acceptance_length_series(rows, groups)
    x = np.arange(len(groups))

    def values(metric):
        items = series[metric]
        return (
            np.array([item[0] if item is not None else np.nan for item in items]),
            np.array([item[1] if item is not None else 0.0 for item in items]),
        )

    prefix, prefix_std = values("accepted_prefix_tokens")
    effective, effective_std = values("accepted_effective_tokens")
    ax.errorbar(
        x,
        effective,
        yerr=effective_std,
        color=color_effective,
        marker="o",
        markersize=6,
        linewidth=2.4,
        capsize=4,
        label="Acc length hiệu dụng (k+1)",
        zorder=3,
    )
    ax.errorbar(
        x,
        prefix,
        yerr=prefix_std,
        color=color_prefix,
        marker="D",
        markersize=5,
        linewidth=1.8,
        linestyle="--",
        capsize=3,
        label="Accepted prefix (k)",
        zorder=2,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tokens / speculative round")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="best")


def plot_acceptance_length_lines(
    rows: list[dict[str, str]],
    groups: list[dict[str, str]],
    labels: list[str],
    output: Path,
    *,
    title: str,
    xlabel: str,
):
    """Write a polished two-series acceptance-length line plot."""

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    _draw_acceptance_length_lines(
        ax,
        rows,
        groups,
        labels,
        title=title,
        xlabel=xlabel,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _h21_sample_key(row: dict) -> tuple[str, str]:
    """Disambiguate H2.1 sample IDs reused across draft checkpoints."""

    return str(row.get("training_corpus", "")), str(row.get("row_id", ""))


def summarize_h21_output_progress(
    metric_rows: list[dict[str, str]],
    round_rows: list[dict],
    progress_points: tuple[int, ...] = (25, 50, 75, 100),
) -> list[dict]:
    """Summarize per-sample acc length inside cumulative output prefixes.

    Each round is assigned by its fractional output progress
    (``answer_position_start / target_output_tokens``). Only full,
    non-terminal rounds are included, matching the existing ``tau_proposal``
    metric. A sample contributes its mean over eligible rounds; group means
    are then calculated across contributing samples so longer outputs do not
    receive extra weight. The 100% result is checked against recorded tau.
    """

    samples = {}
    for row in metric_rows:
        key = _h21_sample_key(row)
        if key in samples:
            raise ValueError(f"Duplicate H2.1 sample key: {key}")
        samples[key] = row

    rounds_by_sample: dict[tuple[str, str], list[dict]] = {}
    for row in round_rows:
        key = _h21_sample_key(row)
        if key not in samples:
            raise ValueError(f"Acceptance round has no matching H2.1 sample: {key}")
        rounds_by_sample.setdefault(key, []).append(row)

    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    round_counts: dict[tuple[str, str, str, int], int] = {}
    total_samples: dict[tuple[str, str, str, int], int] = {}

    for key, sample in samples.items():
        try:
            output_length = int(float(sample["target_output_tokens"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing target output length for H2.1 sample {key}") from exc
        if output_length <= 0:
            raise ValueError(f"Non-positive target output length for H2.1 sample {key}")

        valid_rounds = []
        for item in rounds_by_sample.get(key, []):
            if item.get("is_partial_block", False) or item.get("is_terminal", False):
                continue
            start = item.get("answer_position_start")
            matched = item.get("matched_proposals")
            if start is None or matched is None:
                continue
            valid_rounds.append((int(start), float(matched)))

        for percent in progress_points:
            if not 0 < percent <= 100:
                raise ValueError(f"Invalid output progress point: {percent}")
            # answer_position_start is the number of target output tokens
            # already emitted before this round. A round exactly on a prefix
            # boundary belongs to the next cumulative prefix.
            prefix_rounds = [
                accepted
                for start, accepted in valid_rounds
                if start < output_length * percent / 100.0
            ]
            group_key = (
                str(sample.get("training_corpus", "")),
                str(sample.get("prompt_variant", "")),
                str(sample.get("visual_condition", "")),
                percent,
            )
            total_samples[group_key] = total_samples.get(group_key, 0) + 1
            if prefix_rounds:
                grouped.setdefault(group_key, []).append(statistics.mean(prefix_rounds))
                round_counts[group_key] = round_counts.get(group_key, 0) + len(prefix_rounds)

        logged_tau = sample.get("tau_proposal")
        if logged_tau not in (None, ""):
            full_rounds = [accepted for _start, accepted in valid_rounds]
            if not full_rounds or abs(statistics.mean(full_rounds) - float(logged_tau)) > 1e-8:
                raise ValueError(f"100% round trace does not reproduce logged tau for {key}")
        elif valid_rounds:
            raise ValueError(f"H2.1 sample has valid rounds but missing logged tau: {key}")

    output = []
    for group_key in sorted(total_samples):
        corpus, prompt, visual, percent = group_key
        values = grouped.get(group_key, [])
        output.append(
            {
                "training_corpus": corpus,
                "prompt_variant": prompt,
                "visual_condition": visual,
                "progress_pct": percent,
                "mean_acc_length": statistics.mean(values) if values else None,
                "std_acc_length": statistics.stdev(values) if len(values) > 1 else None,
                "n_valid_samples": len(values),
                "n_valid_rounds": round_counts.get(group_key, 0),
                "n_total_samples": total_samples[group_key],
            }
        )
    return output


def plot_h21_output_progress(summary_rows: list[dict], output: Path) -> None:
    """Draw a 2×3 H2.1 progress chart split by checkpoint and prompt."""

    corpora = ("llava68k", "sharegpt68k")
    prompts = ("natural", "mcq", "mcq2")
    conditions = ("full", "zero", "cut")
    condition_labels = {"full": "Full", "zero": "Zero", "cut": "Cut"}
    corpus_labels = {
        "llava68k": "LLaVA68K · depth 3",
        "sharegpt68k": "ShareGPT68K · depth 3",
    }
    prompt_labels = {"natural": "Natural", "mcq": "MCQ", "mcq2": "MCQ2"}
    colors = {"full": COLORS["full"], "zero": COLORS["zero"], "cut": COLORS["cut"]}
    progress_points = (25, 50, 75, 100)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="col")

    for row_index, corpus in enumerate(corpora):
        for col_index, prompt in enumerate(prompts):
            ax = axes[row_index, col_index]
            panel_rows = [
                row
                for row in summary_rows
                if row["training_corpus"] == corpus and row["prompt_variant"] == prompt
            ]
            for condition in conditions:
                condition_rows = {
                    row["progress_pct"]: row
                    for row in panel_rows
                    if row["visual_condition"] == condition
                }
                means = [
                    condition_rows[pct]["mean_acc_length"]
                    if pct in condition_rows
                    else np.nan
                    for pct in progress_points
                ]
                ax.plot(
                    progress_points,
                    means,
                    color=colors[condition],
                    marker="o",
                    linewidth=2.2,
                    markersize=5.5,
                    label=condition_labels[condition],
                )
            ax.set_title(f"{prompt_labels[prompt]} · {corpus_labels[corpus]}", fontsize=10.5, fontweight="bold")
            ax.set_xticks(progress_points, ["25%", "50%", "75%", "100%"])
            ax.set_xlim(20, 105)
            ax.grid(axis="y", alpha=0.22, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_index == 0:
                ax.set_ylabel("Mean accepted proposals / valid round")
            if row_index == 1:
                ax.set_xlabel("Cumulative target-output progress")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "H2.1 — Acc length theo tiến độ output tích lũy",
        fontsize=16,
        fontweight="bold",
        y=1.015,
    )
    fig.text(
        0.5,
        0.015,
        "Mỗi sample được lấy trung bình các vòng hợp lệ bắt đầu trong prefix rồi mới gộp theo nhóm; n theo từng điểm nằm trong CSV đi kèm.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92), h_pad=2.0, w_pad=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_h21_output_progress(input_dir: Path) -> None:
    metric_rows = load_summary(input_dir / "metric_records.csv")
    with (input_dir / "acceptance_round_records.jsonl").open(encoding="utf-8") as handle:
        round_rows = [json.loads(line) for line in handle if line.strip()]
    summary_rows = summarize_h21_output_progress(metric_rows, round_rows)
    output_csv = input_dir / "acc_length_output_progress.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_h21_output_progress(summary_rows, input_dir / "acc_length_output_progress.png")



def summarize_h31_output_progress(
    raw_rows: list[dict],
    progress_points: tuple[int, ...] = tuple(range(10, 101, 10)),
) -> list[dict]:
    """Summarize MSD acc length inside cumulative output prefixes.

    MSD records one accepted_proposal_count per speculative round. Unlike
    DFlash rows, the generic matched_proposals field is not populated, so
    this summary reads the MSD-native round trace directly. Values are first
    averaged within each sample and then across samples so long outputs do not
    receive extra weight.
    """

    retentions = (100.0, 50.0, 25.0, 0.0)
    grouped: dict[tuple[float, int], list[float]] = {}
    round_counts: dict[tuple[float, int], int] = {}
    total_samples: dict[tuple[float, int], int] = {}
    seen_keys: set[tuple[str, float]] = set()

    for row in raw_rows:
        if row.get("status") not in (None, "ok") or row.get("condition") != "retention":
            continue
        try:
            retention = float(row["retention_percentage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"MSD row has invalid retention: {row.get('row_id')}") from exc
        if retention not in retentions:
            raise ValueError(f"Unexpected H3.1 retention: {retention}")
        try:
            sample_id = str(row["sample_id"])
            target_output_tokens = len(row["target_output_ids"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"MSD row is missing target output data: {row.get('row_id')}") from exc
        if target_output_tokens <= 0:
            raise ValueError(f"Non-positive target output length: {row.get('row_id')}")
        sample_key = (sample_id, retention)
        if sample_key in seen_keys:
            raise ValueError(f"Duplicate H3.1 sample/retention key: {sample_key}")
        seen_keys.add(sample_key)

        valid_rounds: list[tuple[int, float]] = []
        for item in row.get("acceptance_rounds") or []:
            if not isinstance(item, dict):
                continue
            start = item.get("answer_position_start")
            accepted = item.get("accepted_proposal_count")
            if not isinstance(start, (int, float)) or not isinstance(accepted, (int, float)):
                continue
            valid_rounds.append((int(start), float(accepted)))
        if not valid_rounds:
            raise ValueError(f"MSD row has no valid acceptance rounds: {row.get('row_id')}")

        logged_acc_length = row.get("accepted_prefix_tokens")
        logged_mean = statistics.fmean(value for _start, value in valid_rounds)
        if not isinstance(logged_acc_length, (int, float)) or not math.isclose(
            logged_mean, float(logged_acc_length), rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError(
                f"100% trace does not reproduce accepted_prefix_tokens for {row.get('row_id')}: "
                f"{logged_mean} != {logged_acc_length}"
            )

        for percent in progress_points:
            if not 0 < percent <= 100:
                raise ValueError(f"Invalid output progress point: {percent}")
            key = (retention, percent)
            total_samples[key] = total_samples.get(key, 0) + 1
            if percent == 100:
                # Include the terminal MSD round so the 100% point reproduces
                # the runtime's recorded accepted_prefix_tokens exactly.
                prefix_rounds = [accepted for _start, accepted in valid_rounds]
            else:
                prefix_rounds = [
                    accepted
                    for start, accepted in valid_rounds
                    if start < target_output_tokens * percent / 100.0
                ]
            if prefix_rounds:
                grouped.setdefault(key, []).append(statistics.fmean(prefix_rounds))
                round_counts[key] = round_counts.get(key, 0) + len(prefix_rounds)

    if len(seen_keys) != 200:
        raise ValueError(f"Expected 200 valid H3.1 MSD rows, found {len(seen_keys)}")

    output: list[dict] = []
    for retention in retentions:
        for percent in progress_points:
            key = (retention, percent)
            values = grouped.get(key, [])
            output.append(
                {
                    "retention_percentage": int(retention),
                    "progress_pct": percent,
                    "mean_acc_length": statistics.fmean(values) if values else None,
                    "std_acc_length": statistics.stdev(values) if len(values) > 1 else None,
                    "n_valid_samples": len(values),
                    "n_valid_rounds": round_counts.get(key, 0),
                    "n_total_samples": total_samples.get(key, 0),
                }
            )
    return output


def plot_h31_output_progress(summary_rows: list[dict], output: Path) -> None:
    """Draw H3.1 MSD acc length by cumulative target-output progress."""

    retention_order = (100, 50, 25, 0)
    labels = {
        100: "100% visual retention",
        50: "50% visual retention",
        25: "25% visual retention",
        0: "0% visual retention",
    }
    colors = {100: "#2563eb", 50: "#f59e0b", 25: "#16a34a", 0: "#dc2626"}
    progress_points = tuple(range(10, 101, 10))
    fig, ax = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    for retention in retention_order:
        rows = {
            int(row["progress_pct"]): row
            for row in summary_rows
            if int(row["retention_percentage"]) == retention
        }
        means = np.array(
            [
                rows[pct]["mean_acc_length"]
                if rows[pct]["mean_acc_length"] is not None
                else np.nan
                for pct in progress_points
            ],
            dtype=float,
        )
        stds = np.array(
            [rows[pct]["std_acc_length"] or 0.0 for pct in progress_points],
            dtype=float,
        )
        ax.errorbar(
            progress_points,
            means,
            yerr=stds,
            color=colors[retention],
            marker="o",
            markersize=5,
            linewidth=2.2,
            capsize=3,
            label=labels[retention],
        )
    ax.set_title("H3.1 MSD — Acc length theo tiến độ output", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Tỷ lệ output tích lũy so với output length")
    ax.set_ylabel("Mean accepted proposals / speculative round")
    ax.set_xticks(progress_points, [f"{pct}%" for pct in progress_points])
    ax.set_xlim(7, 103)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.text(
        0.5,
        0.005,
        "Mỗi sample được lấy trung bình các round bắt đầu trong prefix, rồi mới gộp theo retention; thanh sai số là ±1 SD giữa các sample.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_h31_output_progress(input_dir: Path) -> None:
    """Build H3.1 output-progress CSV/PNG from the raw MSD journal."""

    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    raw_rows: list[dict] = []
    for source in manifest["input_files"]:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        with source_path.open(encoding="utf-8") as handle:
            raw_rows.extend(json.loads(line) for line in handle if line.strip())
    summary_rows = summarize_h31_output_progress(raw_rows)
    output_csv = input_dir / "acc_length_output_progress.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_h31_output_progress(summary_rows, input_dir / "acc_length_output_progress.png")

def panel(ax, rows, groups, metric, title, labels, ylabel=None):
    values = [lookup(rows, group, metric) for group in groups]
    x = np.arange(len(groups))
    means = np.array([item[0] if item is not None else np.nan for item in values])
    stds = np.array([item[1] if item is not None else 0.0 for item in values])
    ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, linewidth=1.4, markersize=4)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    if ylabel:
        ax.set_ylabel(ylabel)


def save_h21(input_dir: Path):
    rows = load_summary(input_dir / "metric_summary.csv")
    groups = [
        {"prompt_variant": variant, "visual_condition": condition}
        for variant in ("mcq", "natural")
        for condition in ("full", "zero", "cut")
    ]
    labels = [f"{g['prompt_variant']}\n{g['visual_condition']}" for g in groups]
    specs = [
        ("accepted_prefix_tokens", "Accepted prefix", "tokens"),
        ("acceptance_rate", "Acceptance rate", None),
        ("speedup_end_to_end", "End-to-end speedup", "x"),
        ("speculative_end_to_end_s", "Speculative end-to-end", "seconds"),
        ("speculative_output_tokens", "Speculative output", "tokens"),
        ("lossless", "Lossless fraction", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        panel(ax, rows, groups, metric, title, labels, ylabel)
    fig.suptitle("H2.1 descriptive overview; mean ± std", fontsize=14)
    fig.savefig(input_dir / "overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, constrained_layout=True)
    for ax, variant in zip(axes, ("mcq", "natural")):
        variant_groups = [
            {"prompt_variant": variant, "visual_condition": condition}
            for condition in ("full", "zero", "cut")
        ]
        _draw_acceptance_length_lines(
            ax,
            rows,
            variant_groups,
            ["Full", "Zero", "Cut"],
            title=f"{variant.upper()} prompt",
            xlabel="Draft visual condition",
            show_legend=variant == "mcq",
        )
    fig.suptitle("H2.1 — Acc length theo format prompt", fontsize=16, fontweight="bold")
    fig.savefig(input_dir / "acceptance_length_lines.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_h31(input_dir: Path):
    rows = load_summary(input_dir / "metric_summary.csv")
    retentions = (100, 50, 25, 0)
    groups = [{"retention_percentage": str(float(retention))} for retention in retentions]
    labels = [str(retention) for retention in retentions]
    specs = [
        ("accepted_prefix_tokens", "Accepted prefix", "tokens"),
        ("acceptance_rate", "Acceptance rate*", None),
        ("speedup_end_to_end", "End-to-end speedup", "x"),
        ("speculative_end_to_end_s", "Speculative end-to-end", "seconds"),
        ("speculative_output_tokens", "Speculative output", "tokens"),
        ("draft_context_visual_tokens", "Draft visual tokens in context", "tokens"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.flat, specs):
        panel(ax, rows, groups, metric, title, labels, ylabel)
    fig.suptitle("H3.1 MSD retention overview; mean ± std", fontsize=14)
    fig.text(
        0.5,
        0.005,
        "* H3.1 acceptance rate is not comparable to DFlash: fallback records have proposed=0 and are normalized to 1.0.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(input_dir / "overview.png", dpi=180)
    plt.close(fig)

    plot_acceptance_length_lines(
        rows,
        groups,
        ["100%", "50%", "25%", "0%"],
        input_dir / "acceptance_length_lines.png",
        title="H3.1 MSD — Acc length theo visual retention",
        xlabel="Visual retention",
    )


def save_h32(input_dir: Path):
    rows = load_summary(input_dir / "metric_summary.csv")
    specs = [
        ("accepted_prefix_tokens", "Accepted prefix", "tokens"),
        ("acceptance_rate", "Acceptance rate", None),
        ("speedup_end_to_end", "End-to-end speedup", "x"),
        ("speculative_end_to_end_s", "Speculative end-to-end", "seconds"),
        ("lossless", "Lossless fraction", None),
        ("draft_context_visual_tokens", "Draft visual tokens in context", "tokens"),
    ]
    for dataset in ("mvbench", "vdc"):
        groups = [
            {
                "dataset": dataset,
                "training_corpus": corpus,
                "draft_depth": str(depth),
                "visual_condition": condition,
            }
            for corpus in ("llava68k", "sharegpt68k")
            for depth in (1, 3, 5)
            for condition in ("full", "zero", "cut")
        ]
        labels = [
            f"{g['training_corpus'].replace('68k', '')}\nd{g['draft_depth']}-{g['visual_condition']}"
            for g in groups
        ]
        fig, axes = plt.subplots(2, 3, figsize=(19, 9), constrained_layout=True)
        for ax, (metric, title, ylabel) in zip(axes.flat, specs):
            panel(ax, rows, groups, metric, title, labels, ylabel)
        fig.suptitle(f"H3.2 {dataset} depth overview; mean ± std", fontsize=14)
        fig.savefig(input_dir / f"overview_{dataset}.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True, constrained_layout=True)
        for ax, corpus in zip(axes, ("llava68k", "sharegpt68k")):
            for condition in ("full", "zero", "cut"):
                groups = [
                    {
                        "dataset": dataset,
                        "training_corpus": corpus,
                        "draft_depth": str(depth),
                        "visual_condition": condition,
                    }
                    for depth in (1, 3, 5)
                ]
                series = acceptance_length_series(rows, groups)
                x = np.arange(3)
                effective = series["accepted_effective_tokens"]
                prefix = series["accepted_prefix_tokens"]
                effective_mean = np.array([item[0] if item else np.nan for item in effective])
                effective_std = np.array([item[1] if item else 0.0 for item in effective])
                prefix_mean = np.array([item[0] if item else np.nan for item in prefix])
                prefix_std = np.array([item[1] if item else 0.0 for item in prefix])
                color = COLORS[condition]
                ax.errorbar(
                    x,
                    effective_mean,
                    yerr=effective_std,
                    color=color,
                    marker="o",
                    linewidth=2.0,
                    capsize=3,
                    label=f"{condition} · acc length",
                )
                ax.errorbar(
                    x,
                    prefix_mean,
                    yerr=prefix_std,
                    color=color,
                    marker="D",
                    linewidth=1.3,
                    linestyle="--",
                    alpha=0.72,
                    capsize=2,
                    label=f"{condition} · prefix",
                )
            ax.set_title(corpus.replace("68k", " 68K"), fontsize=12, fontweight="bold")
            ax.set_xlabel("Draft depth")
            ax.set_xticks(np.arange(3), ["1", "3", "5"])
            ax.grid(axis="y", alpha=0.22, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("Tokens / speculative round")
        axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="best")
        fig.suptitle(f"H3.2 {dataset.upper()} — Acc length theo depth", fontsize=16, fontweight="bold")
        fig.savefig(
            input_dir / f"acceptance_length_lines_{dataset}.png",
            dpi=220,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)



def plot_h32_tau_effective(rows: list[dict[str, str]], output: Path) -> None:
    """Draw H3.2 tau effective across draft depths."""

    panels = (
        ("mvbench", "llava68k"),
        ("mvbench", "sharegpt68k"),
        ("vdc", "llava68k"),
        ("vdc", "sharegpt68k"),
    )
    condition_labels = {"full": "Full", "zero": "Zero", "cut": "Cut"}
    depths = (1, 3, 5)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharey=True, constrained_layout=True)
    axes_flat = axes.flat
    for ax, (dataset, corpus) in zip(axes_flat, panels):
        for condition in ("full", "zero", "cut"):
            groups = [
                {
                    "dataset": dataset,
                    "training_corpus": corpus,
                    "draft_depth": str(depth),
                    "visual_condition": condition,
                }
                for depth in depths
            ]
            values = [lookup(rows, group, "tau_effective") for group in groups]
            means = np.array([item[0] if item is not None else np.nan for item in values])
            stds = np.array([item[1] if item is not None else 0.0 for item in values])
            ax.errorbar(
                depths,
                means,
                yerr=stds,
                color=COLORS[condition],
                marker="o",
                markersize=5.5,
                linewidth=2.2,
                capsize=3,
                label=condition_labels[condition],
            )
        corpus_label = corpus.replace("68k", " 68K")
        ax.set_title(f"{dataset.upper()} · {corpus_label}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("DFlash draft depth")
        ax.set_xticks(depths, [str(depth) for depth in depths])
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Tau effective (tokens / speculative round)")
    axes[1, 0].set_ylabel("Tau effective (tokens / speculative round)")
    axes[0, 1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("H3.2 — Tau effective theo depth, dataset và corpus", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Điểm là mean; thanh sai số là ±1 SD. MVBench n=1,000 mỗi nhóm trước khi lọc trace; VDC n=50 mỗi nhóm.",
        ha="center",
        fontsize=8.8,
        color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_h32_tau_effective(input_dir: Path) -> None:
    """Write the H3.2 tau-effective CSV and figure."""

    rows = load_summary(input_dir / "metric_summary.csv")
    tau_rows = [row for row in rows if row.get("metric") == "tau_effective"]
    expected = {
        (dataset, corpus, str(depth), condition)
        for dataset in ("mvbench", "vdc")
        for corpus in ("llava68k", "sharegpt68k")
        for depth in (1, 3, 5)
        for condition in ("full", "zero", "cut")
    }
    observed = {
        (row["dataset"], row["training_corpus"], row["draft_depth"], row["visual_condition"])
        for row in tau_rows
    }
    if observed != expected:
        raise ValueError(f"Unexpected H3.2 tau-effective groups: {sorted(observed ^ expected)}")
    output_csv = input_dir / "tau_effective.csv"
    fields = [
        "dataset",
        "training_corpus",
        "draft_depth",
        "visual_condition",
        "metric",
        "n",
        "mean",
        "std",
        "median",
        "min",
        "max",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            tau_rows,
            key=lambda item: (
                item["dataset"],
                item["training_corpus"],
                int(item["draft_depth"]),
                ("full", "zero", "cut").index(item["visual_condition"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fields})
    plot_h32_tau_effective(tau_rows, input_dir / "tau_effective_h32.png")


def _load_h32_progress_samples(path: Path) -> list[dict[str, str]]:
    """Read only sample fields needed for H3.2 output-progress aggregation."""

    old_limit = csv.field_size_limit()
    csv.field_size_limit(100_000_000)
    try:
        samples = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = (
                "row_id",
                "dataset",
                "training_corpus",
                "draft_depth",
                "visual_condition",
                "target_output_tokens",
                "accepted_prefix_tokens",
                "tau_proposal",
            )
            for row in reader:
                samples.append({field: row.get(field, "") for field in fields})
        return samples
    finally:
        csv.field_size_limit(old_limit)


def summarize_h32_output_progress(
    metric_rows: list[dict[str, str]],
    round_rows,
    progress_points: tuple[int, ...] = tuple(range(10, 101, 10)),
) -> list[dict]:
    """Summarize H3.2 acc length inside cumulative output prefixes."""

    expected_depths = {"1", "3", "5"}
    sample_info: dict[tuple[str, str, str, str, str], dict] = {}
    total_samples: dict[tuple[str, str, str, str, int], int] = {}
    for row in metric_rows:
        row_id = row.get("row_id", "")
        depth = str(row["draft_depth"])
        group = (
            row["dataset"],
            row["training_corpus"],
            depth,
            row["visual_condition"],
        )
        sample_key = (*group, row_id)
        if not row_id or sample_key in sample_info:
            raise ValueError(f"Missing or duplicate H3.2 sample key: {sample_key}")
        target_length = int(float(row["target_output_tokens"]))
        if depth not in expected_depths or target_length <= 0:
            raise ValueError(f"Invalid H3.2 sample metadata: {sample_key}")
        sample_info[sample_key] = {
            "group": group,
            "target_length": target_length,
            "logged_acc_length": row.get("tau_proposal") or row.get("accepted_prefix_tokens") or "",
        }
        for percent in progress_points:
            key = (*group, percent)
            total_samples[key] = total_samples.get(key, 0) + 1

    rounds_by_sample: dict[tuple[str, str, str, str, str], list[tuple[int, float]]] = {
        sample_key: [] for sample_key in sample_info
    }
    for item in round_rows:
        row_id = item.get("row_id", "")
        group = (
            str(item.get("dataset", "")),
            str(item.get("training_corpus", "")),
            str(item.get("draft_depth", "")),
            str(item.get("visual_condition", "")),
        )
        sample_key = (*group, row_id)
        if sample_key not in sample_info:
            continue
        if item.get("is_partial_block") or item.get("is_terminal"):
            continue
        start = item.get("answer_position_start")
        accepted = item.get("matched_proposals")
        if not isinstance(start, (int, float)) or not isinstance(accepted, (int, float)):
            continue
        rounds_by_sample[sample_key].append((int(start), float(accepted)))

    grouped: dict[tuple[str, str, str, str, int], list[float]] = {}
    round_counts: dict[tuple[str, str, str, str, int], int] = {}
    for sample_key, info in sample_info.items():
        valid_rounds = rounds_by_sample[sample_key]
        if not valid_rounds:
            continue
        logged = info["logged_acc_length"]
        if logged:
            logged_mean = statistics.fmean(value for _start, value in valid_rounds)
            if not math.isclose(logged_mean, float(logged), rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(
                    f"100% trace does not reproduce logged acc length for {sample_key}: "
                    f"{logged_mean} != {logged}"
                )
        group = info["group"]
        for percent in progress_points:
            if percent == 100:
                selected = [value for _start, value in valid_rounds]
            else:
                cutoff = info["target_length"] * percent / 100.0
                selected = [value for start, value in valid_rounds if start < cutoff]
            if not selected:
                continue
            key = (*group, percent)
            grouped.setdefault(key, []).append(statistics.fmean(selected))
            round_counts[key] = round_counts.get(key, 0) + len(selected)

    output: list[dict] = []
    for dataset in ("mvbench", "vdc"):
        for corpus in ("llava68k", "sharegpt68k"):
            for depth in ("1", "3", "5"):
                for condition in ("full", "zero", "cut"):
                    for percent in progress_points:
                        key = (dataset, corpus, depth, condition, percent)
                        values = grouped.get(key, [])
                        output.append(
                            {
                                "dataset": dataset,
                                "training_corpus": corpus,
                                "draft_depth": int(depth),
                                "visual_condition": condition,
                                "progress_pct": percent,
                                "mean_acc_length": statistics.fmean(values) if values else None,
                                "std_acc_length": statistics.stdev(values) if len(values) > 1 else None,
                                "n_valid_samples": len(values),
                                "n_valid_rounds": round_counts.get(key, 0),
                                "n_total_samples": total_samples.get(key, 0),
                            }
                        )
    return output

def plot_h32_output_progress(summary_rows: list[dict], depth: int, output: Path) -> None:
    """Draw one four-panel H3.2 output-progress figure for one depth."""

    panels = (
        ("mvbench", "llava68k"),
        ("mvbench", "sharegpt68k"),
        ("vdc", "llava68k"),
        ("vdc", "sharegpt68k"),
    )
    progress_points = tuple(range(10, 101, 10))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharey=True, constrained_layout=True)
    for ax, (dataset, corpus) in zip(axes.flat, panels):
        for condition in ("full", "zero", "cut"):
            rows = {
                int(row["progress_pct"]): row
                for row in summary_rows
                if row["dataset"] == dataset
                and row["training_corpus"] == corpus
                and int(row["draft_depth"]) == depth
                and row["visual_condition"] == condition
            }
            means = np.array(
                [rows[pct]["mean_acc_length"] if rows[pct]["mean_acc_length"] is not None else np.nan for pct in progress_points],
                dtype=float,
            )
            stds = np.array(
                [rows[pct]["std_acc_length"] or 0.0 for pct in progress_points],
                dtype=float,
            )
            ax.errorbar(
                progress_points,
                means,
                yerr=stds,
                color=COLORS[condition],
                marker="o",
                markersize=4.5,
                linewidth=2.0,
                capsize=2.5,
                label=condition.capitalize(),
            )
        ax.set_title(f"{dataset.upper()} · {corpus.replace('68k', ' 68K')}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Cumulative target-output progress")
        ax.set_xticks(progress_points, [f"{pct}%" for pct in progress_points])
        ax.set_xlim(7, 103)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Mean acc length (accepted proposals / round)")
    axes[1, 0].set_ylabel("Mean acc length (accepted proposals / round)")
    axes[0, 1].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(f"H3.2 — Acc length theo tiến độ output · depth {depth}", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Mỗi sample được lấy trung bình trên các round hợp lệ trong prefix rồi mới gộp; thanh sai số là ±1 SD giữa các sample.",
        ha="center",
        fontsize=8.8,
        color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_h32_output_progress(input_dir: Path) -> None:
    """Write H3.2 output-progress CSV and one figure per draft depth."""

    metric_rows = _load_h32_progress_samples(input_dir / "metric_records.csv")
    round_path = input_dir / "acceptance_round_records.jsonl"
    with round_path.open(encoding="utf-8") as handle:
        round_rows = (json.loads(line) for line in handle if line.strip())
        summary_rows = summarize_h32_output_progress(metric_rows, round_rows)

    output_csv = input_dir / "acc_length_output_progress_h32.csv"
    fields = [
        "dataset",
        "training_corpus",
        "draft_depth",
        "visual_condition",
        "progress_pct",
        "mean_acc_length",
        "std_acc_length",
        "n_valid_samples",
        "n_valid_rounds",
        "n_total_samples",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    for depth in (1, 3, 5):
        plot_h32_output_progress(
            summary_rows,
            depth,
            input_dir / f"acc_length_output_progress_h32_depth{depth}.png",
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=("h21", "h21-progress", "h31", "h31-progress", "h32", "h32-tau-effective", "h32-output-progress"),
        required=True,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    args.input_dir.mkdir(parents=True, exist_ok=True)
    {
        "h21": save_h21,
        "h21-progress": save_h21_output_progress,
        "h31": save_h31,
        "h31-progress": save_h31_output_progress,
        "h32": save_h32,
        "h32-tau-effective": save_h32_tau_effective,
        "h32-output-progress": save_h32_output_progress,
    }[
        args.experiment
    ](args.input_dir)


if __name__ == "__main__":
    main()
