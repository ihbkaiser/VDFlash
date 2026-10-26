#!/usr/bin/env python3
"""Plot the author-checkpoint Qwen2-VL-7B MSD verification results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/22_10_26/msd_qwen2vl7b_author_visual_verification_colab_corrected.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/22_10_26/msd_qwen2vl7b_author_visual_verification_colab_corrected_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("markdown/22_10_26/figures"))
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.input) if row.get("status") == "ok"]
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    strict = [
        row for row in rows
        if row["arms"]["full"]["lossless"] and row["arms"]["cut"]["lossless"]
    ]
    if not rows or not strict:
        raise SystemExit("No successful rows or no strict-lossless pairs to plot")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_color = "#4263a5"
    cut_color = "#d17a35"

    # Figure 9: the target-attention distribution and the paired acceptance rates.
    token_rows = [token for row in rows for token in row["target_visual_attention"]]
    masses = np.asarray(
        [token["visual_attention_mass_mid_layer"] for token in token_rows], dtype=float
    )
    thresholds = summary["attention_thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1), constrained_layout=True)
    ax = axes[0]
    ax.hist(masses, bins=36, color="#6a8caf", alpha=0.88, edgecolor="white", linewidth=0.35)
    for key, label, color in (
        ("p50", "P50", "#3a8f5b"),
        ("p75", "P75 (top-quartile cutoff)", "#c46c2e"),
        ("p90", "P90", "#9c3d4b"),
    ):
        value = thresholds[key]
        ax.axvline(value, color=color, linestyle="--", linewidth=1.7, label=f"{label}: {value:.2f}")
    ax.set_title("Target visual-attention mass")
    ax.set_xlabel("Visual-attention mass at decoder block 14 (0-based)")
    ax.set_ylabel("Target output tokens (n)")
    ax.legend(frameon=False, fontsize=8)

    categories = [
        ("LOW_VISUAL", "Low\n< P50"),
        ("MEDIUM_VISUAL", "Medium\nP50–P75"),
        ("HEAVY_VISUAL", "Heavy\nP75–P90"),
        ("ULTRA_HEAVY", "Ultra-heavy\n≥ P90"),
        ("ALL_HEAVY_VISUAL_TOP25", "Top quartile\n≥ P75"),
    ]
    x = np.arange(len(categories))
    width = 0.36
    ax = axes[1]
    for offset, arm, color, label in (
        (-width / 2, "full", full_color, "Full video-pad context"),
        (width / 2, "cut", cut_color, "Video-pad slots cut"),
    ):
        values = [summary["category_summary"][key][f"{arm}_acceptance_pct"] for key, _ in categories]
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
    counts = [summary["category_summary"][key]["token_count"] for key, _ in categories]
    ax.set_xticks(x, [f"{label}\n(n={count})" for (_, label), count in zip(categories, counts)])
    ax.set_ylim(0, 92)
    ax.set_ylabel("Accepted MSD proposals (%)")
    ax.set_title("Acceptance by target-attention group")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Author MSD checkpoint with Qwen2-VL-7B-Instruct · Colab L4 · 25 VDC videos", fontsize=13)
    fig.savefig(args.output_dir / "figure11_msd7b_colab_corrected_attention_acceptance.png", dpi=200)
    plt.close(fig)

    # Figure 12: paired video-bootstrap effects on the 25 exact-lossless pairs.
    bootstrap = summary["paired_video_bootstrap"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    def forest(ax: Any, entries: list[tuple[str, list[float], float]], xlabel: str, zero: float) -> None:
        y = np.arange(len(entries))
        for position, (label, interval, scale) in enumerate(entries):
            mean, low, high = [value * scale for value in interval]
            ax.errorbar(
                mean,
                position,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color="#3f754f" if low > zero else "#6a8caf",
                capsize=3,
                markersize=6,
            )
        ax.axvline(zero, color="#777777", linestyle="--", linewidth=1)
        ax.set_yticks(y, [entry[0] for entry in entries])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)

    forest(
        axes[0],
        [
            (
                f"Top-quartile (+{100 * bootstrap['heavy_top_quartile_cut_minus_full_rate'][0]:.1f} pp)",
                bootstrap["heavy_top_quartile_cut_minus_full_rate"],
                100,
            ),
            (
                f"All output (+{100 * bootstrap['all_output_cut_minus_full_rate'][0]:.1f} pp)",
                bootstrap["all_output_cut_minus_full_rate"],
                100,
            ),
        ],
        "Cut − Full acceptance (percentage points)",
        0,
    )
    axes[0].set_title("Acceptance change")
    axes[0].set_xlim(-1.5, 6)

    forest(
        axes[1],
        [(f"τ effective (+{bootstrap['tau_effective_cut_minus_full'][0]:.2f})", bootstrap["tau_effective_cut_minus_full"], 1)],
        "Cut − Full accepted tokens / round",
        0,
    )
    axes[1].set_title("Drafting efficiency")
    axes[1].set_xlim(-0.05, 0.12)

    speed_entries = [
        (
            f"T_AR / T_Full ({bootstrap['end_to_end_speedup_vs_ar_full'][0]:.2f}×)",
            bootstrap["end_to_end_speedup_vs_ar_full"],
            1,
        ),
        (
            f"T_AR / T_Cut ({bootstrap['end_to_end_speedup_vs_ar_cut'][0]:.2f}×)",
            bootstrap["end_to_end_speedup_vs_ar_cut"],
            1,
        ),
        (
            f"T_Full / T_Cut ({bootstrap['end_to_end_time_ratio_full_over_cut'][0]:.2f}×)",
            bootstrap["end_to_end_time_ratio_full_over_cut"],
            1,
        ),
    ]
    forest(axes[2], speed_entries, "Paired E2E time ratio (×; >1 favors named MSD arm)", 1)
    axes[2].set_title("End-to-end timing")
    axes[2].set_xlim(0.90, 1.82)
    fig.suptitle("Paired video bootstrap · exact-lossless pairs (n=25) · 3 timing repeats", fontsize=13)
    fig.savefig(args.output_dir / "figure12_msd7b_colab_corrected_paired_effects.png", dpi=200)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
