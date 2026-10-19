#!/usr/bin/env python3
"""Plot statistical benchmarks and comparative figures for VidHalluc and VideoChair."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RETENTION_LEVELS = [100, 50, 25, 0]
RET_LABELS = ["100% (Full)", "50%", "25%", "0% (Text-only)"]
RET_COLORS = ["#2563eb", "#f59e0b", "#16a34a", "#dc2626"]


def plot_speedup_comparison(output_dir: Path):
    """Plot decode speedup vs end-to-end speedup for VidHalluc and VideoChair."""
    # Data from runs
    # VidHalluc: short answers (prefill-dominated)
    vid_decode = [0.63, 0.72, 0.84, 1.14]
    vid_e2e = [1.19, 1.28, 1.35, 1.48]

    # VideoChair: long captions (decode-dominated)
    chair_decode = [0.67, 1.03, 1.12, 1.22]
    chair_e2e = [0.77, 1.12, 1.21, 1.29]

    x = np.arange(len(RETENTION_LEVELS))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: VidHalluc
    rects1 = ax1.bar(x - width/2, vid_decode, width, label="Decode Speedup", color="#3b82f6", edgecolor="black", linewidth=0.8)
    rects2 = ax1.bar(x + width/2, vid_e2e, width, label="End-to-End Speedup", color="#10b981", edgecolor="black", linewidth=0.8)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Autoregressive Baseline (1.0x)")
    ax1.set_title("VidHalluc (Short QA / Prefill-Dominated)", fontweight="bold")
    ax1.set_ylabel("Speedup Ratio (vs AR)")
    ax1.set_xlabel("Visual Retention Level")
    ax1.set_xticks(x)
    ax1.set_xticklabels(RET_LABELS)
    ax1.set_ylim(0, 1.8)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Add data values on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: VideoChair
    rects3 = ax2.bar(x - width/2, chair_decode, width, label="Decode Speedup", color="#f59e0b", edgecolor="black", linewidth=0.8)
    rects4 = ax2.bar(x + width/2, chair_e2e, width, label="End-to-End Speedup", color="#8b5cf6", edgecolor="black", linewidth=0.8)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Autoregressive Baseline (1.0x)")
    ax2.set_title("VideoChair (Long Captions / Decode-Dominated)", fontweight="bold")
    ax2.set_ylabel("Speedup Ratio (vs AR)")
    ax2.set_xlabel("Visual Retention Level")
    ax2.set_xticks(x)
    ax2.set_xticklabels(RET_LABELS)
    ax2.set_ylim(0, 1.8)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}x", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Comparative Speedup Dynamics: Decode vs End-to-End Across Visual Retention Levels", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = output_dir / "hallucination_speedup_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_zero_acc_and_length(output_dir: Path):
    """Plot Zero-Acceptance Rate (%) and Mean Accepted Length per Iteration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Data
    ret_vals = [100, 50, 25, 0]
    vid_zero = [34.80, 28.52, 18.80, 6.17]
    chair_zero = [17.54, 5.69, 4.58, 3.50]

    vid_len = [1.04, 1.26, 1.60, 2.63]
    chair_len = [1.52, 2.78, 3.18, 3.46]

    # Panel 1: Zero-acc rate
    ax1.plot(ret_vals, vid_zero, marker="o", markersize=8, linewidth=2.5, color="#dc2626", label="VidHalluc (100 samples)")
    ax1.plot(ret_vals, chair_zero, marker="s", markersize=8, linewidth=2.5, color="#2563eb", label="VideoChair (100 samples)")
    ax1.set_title("Zero-Acceptance Rate (%) vs Retention Level\n(Lower is Better)", fontweight="bold")
    ax1.set_xlabel("Visual Retention (%)")
    ax1.set_ylabel("Zero-Acceptance Iterations (%)")
    ax1.set_xticks([100, 50, 25, 0])
    ax1.set_xticklabels(["100%", "50%", "25%", "0%"])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    for x, y in zip(ret_vals, vid_zero):
        ax1.annotate(f"{y:.1f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color="#dc2626", fontweight="bold")
    for x, y in zip(ret_vals, chair_zero):
        ax1.annotate(f"{y:.1f}%", (x, y), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9, color="#2563eb", fontweight="bold")

    # Panel 2: Mean accepted length
    ax2.plot(ret_vals, vid_len, marker="o", markersize=8, linewidth=2.5, color="#16a34a", label="VidHalluc (100 samples)")
    ax2.plot(ret_vals, chair_len, marker="s", markersize=8, linewidth=2.5, color="#8b5cf6", label="VideoChair (100 samples)")
    ax2.set_title("Mean Accepted Tokens per Round\n(Higher is Better)", fontweight="bold")
    ax2.set_xlabel("Visual Retention (%)")
    ax2.set_ylabel("Accepted Tokens / Draft Round")
    ax2.set_xticks([100, 50, 25, 0])
    ax2.set_xticklabels(["100%", "50%", "25%", "0%"])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    for x, y in zip(ret_vals, vid_len):
        ax2.annotate(f"{y:.2f}", (x, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9, color="#16a34a", fontweight="bold")
    for x, y in zip(ret_vals, chair_len):
        ax2.annotate(f"{y:.2f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color="#8b5cf6", fontweight="bold")

    plt.suptitle("Draft Model Alignment Dynamics: Removing Visual Noise Enhances Acceptance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = output_dir / "hallucination_zero_acc_and_length.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_difficulty_distribution(output_dir: Path):
    """Plot difficulty category distribution (CONTENT, EASY, HARD) for both datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    categories = ["CONTENT\n(Nouns/Verbs)", "EASY\n(Punctuation)", "HARD\n(Prepositions/Other)"]
    vid_shares = [61.5, 27.9, 10.6]
    chair_shares = [66.3, 22.7, 11.1]

    colors = ["#3b82f6", "#10b981", "#ef4444"]

    # VidHalluc
    wedges1, texts1, autotexts1 = ax1.pie(
        vid_shares, labels=categories, autopct="%1.1f%%", startangle=140,
        colors=colors, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts1:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax1.set_title("VidHalluc (5,963 Iterations)\nDifficulty Breakdown", fontweight="bold")

    # VideoChair
    wedges2, texts2, autotexts2 = ax2.pie(
        chair_shares, labels=categories, autopct="%1.1f%%", startangle=140,
        colors=colors, wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts2:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax2.set_title("VideoChair (36,099 Iterations)\nDifficulty Breakdown", fontweight="bold")

    plt.suptitle("Iteration Difficulty Distribution Across Video Benchmarks", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = output_dir / "hallucination_difficulty_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_comprehensive_dashboard(output_dir: Path):
    """Create a 4-panel comprehensive dashboard for paper & presentation."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    # Panel (a): VidHalluc Speedup
    x = np.arange(4)
    w = 0.35
    axs[0, 0].bar(x - w/2, [0.63, 0.72, 0.84, 1.14], w, label="Decode Speedup", color="#3b82f6", edgecolor="black")
    axs[0, 0].bar(x + w/2, [1.19, 1.28, 1.35, 1.48], w, label="End-to-End Speedup", color="#10b981", edgecolor="black")
    axs[0, 0].axhline(1.0, color="red", linestyle="--", alpha=0.7)
    axs[0, 0].set_title("(a) VidHalluc: Decode vs End-to-End Speedup", fontweight="bold")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(RET_LABELS)
    axs[0, 0].set_ylabel("Speedup Ratio")
    axs[0, 0].set_ylim(0, 1.7)
    axs[0, 0].legend()
    axs[0, 0].grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (b): VideoChair Speedup
    axs[0, 1].bar(x - w/2, [0.67, 1.03, 1.12, 1.22], w, label="Decode Speedup", color="#f59e0b", edgecolor="black")
    axs[0, 1].bar(x + w/2, [0.77, 1.12, 1.21, 1.29], w, label="End-to-End Speedup", color="#8b5cf6", edgecolor="black")
    axs[0, 1].axhline(1.0, color="red", linestyle="--", alpha=0.7)
    axs[0, 1].set_title("(b) VideoChair: Decode vs End-to-End Speedup", fontweight="bold")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(RET_LABELS)
    axs[0, 1].set_ylabel("Speedup Ratio")
    axs[0, 1].set_ylim(0, 1.7)
    axs[0, 1].legend()
    axs[0, 1].grid(axis="y", linestyle=":", alpha=0.5)

    # Panel (c): Zero-Acc Rates
    axs[1, 0].plot([100, 50, 25, 0], [34.80, 28.52, 18.80, 6.17], marker="o", color="#dc2626", linewidth=2.5, label="VidHalluc")
    axs[1, 0].plot([100, 50, 25, 0], [17.54, 5.69, 4.58, 3.50], marker="s", color="#2563eb", linewidth=2.5, label="VideoChair")
    axs[1, 0].set_title("(c) Zero-Acceptance Rate (%)", fontweight="bold")
    axs[1, 0].set_xlabel("Retention Level (%)")
    axs[1, 0].set_ylabel("Zero-Acc Iterations (%)")
    axs[1, 0].set_xticks([100, 50, 25, 0])
    axs[1, 0].set_xticklabels(["100%", "50%", "25%", "0%"])
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle=":", alpha=0.5)

    # Panel (d): Mean Accepted Length
    axs[1, 1].plot([100, 50, 25, 0], [1.04, 1.26, 1.60, 2.63], marker="o", color="#16a34a", linewidth=2.5, label="VidHalluc")
    axs[1, 1].plot([100, 50, 25, 0], [1.52, 2.78, 3.18, 3.46], marker="s", color="#8b5cf6", linewidth=2.5, label="VideoChair")
    axs[1, 1].set_title("(d) Mean Accepted Tokens per Round", fontweight="bold")
    axs[1, 1].set_xlabel("Retention Level (%)")
    axs[1, 1].set_ylabel("Accepted Tokens / Round")
    axs[1, 1].set_xticks([100, 50, 25, 0])
    axs[1, 1].set_xticklabels(["100%", "50%", "25%", "0%"])
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Multi-modal Speculative Decoding (MSD): Performance Synthesis Across Benchmarks", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_path = output_dir / "hallucination_comprehensive_dashboard.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    output_dir = Path("results/6_10_26")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_speedup_comparison(output_dir)
    plot_zero_acc_and_length(output_dir)
    plot_difficulty_distribution(output_dir)
    plot_comprehensive_dashboard(output_dir)
    print("All figures successfully created in results/6_10_26/!")


if __name__ == "__main__":
    main()
