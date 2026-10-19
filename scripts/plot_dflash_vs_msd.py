#!/usr/bin/env python3
"""Plot publication-quality figures comparing DFlash vs MSD on Qwen2.5-VL-3B."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style setup
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

RETENTIONS = ["100% (Full)", "50%", "25%", "0% (Cut)"]
OUTPUT_DIR = Path("markdown/Dflash_Sparrow_exp/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Empirical data
# VidHalluc
msd_vh_decode = [0.65, 0.72, 0.84, 1.14]
dflash_vh_decode = [1.55, 1.57, 1.60, 1.60]

msd_vh_tau = [1.00, 1.00, 1.00, 1.00]
dflash_vh_tau = [1.95, 1.96, 1.96, 1.97]

# VideoChair
msd_vc_decode = [0.67, 1.03, 1.12, 1.22]
dflash_vc_decode = [1.78, 1.76, 1.75, 1.69]

msd_vc_tau = [1.00, 1.00, 1.00, 1.00]
dflash_vc_tau = [2.41, 2.41, 2.38, 2.27]

msd_vc_chair_i = [92.05, 92.00, 92.01, 92.49]
dflash_vc_chair_i = [71.15, 71.15, 71.15, 71.15]

msd_vc_precision = [7.95, 8.00, 7.99, 7.51]
dflash_vc_precision = [28.85, 28.85, 28.85, 28.85]


def plot_speedup():
    x = np.arange(len(RETENTIONS))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: VidHalluc
    ax1.bar(x - width/2, msd_vh_decode, width, label="MSD (Depth 3)", color="#94a3b8", edgecolor="black", linewidth=0.8)
    ax1.bar(x + width/2, dflash_vh_decode, width, label="DFlash (Depth 3)", color="#3b82f6", edgecolor="black", linewidth=0.8)
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="AR Baseline (1.0x)")
    ax1.set_title("VidHalluc (Short QA)", fontweight="bold")
    ax1.set_ylabel("Decode Speedup (vs AR Baseline)")
    ax1.set_xlabel("Visual Context Retention")
    ax1.set_xticks(x)
    ax1.set_xticklabels(RETENTIONS)
    ax1.set_ylim(0, 2.2)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Panel 2: VideoChair
    ax2.bar(x - width/2, msd_vc_decode, width, label="MSD (Depth 3)", color="#94a3b8", edgecolor="black", linewidth=0.8)
    ax2.bar(x + width/2, dflash_vc_decode, width, label="DFlash (Depth 3)", color="#10b981", edgecolor="black", linewidth=0.8)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="AR Baseline (1.0x)")
    ax2.set_title("VideoChair (Detailed Captioning)", fontweight="bold")
    ax2.set_ylabel("Decode Speedup (vs AR Baseline)")
    ax2.set_xlabel("Visual Context Retention")
    ax2.set_xticks(x)
    ax2.set_xticklabels(RETENTIONS)
    ax2.set_ylim(0, 2.2)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.suptitle("Decode Speedup Comparison: DFlash vs. MSD across Visual Retentions", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUTPUT_DIR / "fig1_dflash_vs_msd_speedup.png")
    plt.close()
    print("Saved fig1_dflash_vs_msd_speedup.png")


def plot_tau_and_acceptance():
    x = np.arange(len(RETENTIONS))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: VidHalluc Tau
    ax1.bar(x - width/2, msd_vh_tau, width, label="MSD (Depth 3)", color="#cbd5e1", edgecolor="black", linewidth=0.8)
    ax1.bar(x + width/2, dflash_vh_tau, width, label="DFlash (Depth 3)", color="#6366f1", edgecolor="black", linewidth=0.8)
    ax1.set_title("VidHalluc Effective Emitted Tokens ($\tau_{eff}$)", fontweight="bold")
    ax1.set_ylabel("$\tau_{eff}$ (Tokens / Round)")
    ax1.set_xlabel("Visual Context Retention")
    ax1.set_xticks(x)
    ax1.set_xticklabels(RETENTIONS)
    ax1.set_ylim(0, 3.2)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Panel 2: VideoChair Tau
    ax2.bar(x - width/2, msd_vc_tau, width, label="MSD (Depth 3)", color="#cbd5e1", edgecolor="black", linewidth=0.8)
    ax2.bar(x + width/2, dflash_vc_tau, width, label="DFlash (Depth 3)", color="#8b5cf6", edgecolor="black", linewidth=0.8)
    ax2.set_title("VideoChair Effective Emitted Tokens ($\tau_{eff}$)", fontweight="bold")
    ax2.set_ylabel("$\tau_{eff}$ (Tokens / Round)")
    ax2.set_xlabel("Visual Context Retention")
    ax2.set_xticks(x)
    ax2.set_xticklabels(RETENTIONS)
    ax2.set_ylim(0, 3.2)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.suptitle("Speculative Emitted Length ($\tau_{eff}$): DFlash vs. MSD", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUTPUT_DIR / "fig2_dflash_vs_msd_tau.png")
    plt.close()
    print("Saved fig2_dflash_vs_msd_tau.png")


def plot_hallucination_metrics():
    x = np.arange(len(RETENTIONS))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: CHAIR_i (lower is better)
    ax1.bar(x - width/2, msd_vc_chair_i, width, label="MSD (Depth 3)", color="#ef4444", edgecolor="black", linewidth=0.8)
    ax1.bar(x + width/2, dflash_vc_chair_i, width, label="DFlash (Depth 3)", color="#f59e0b", edgecolor="black", linewidth=0.8)
    ax1.set_title("VideoChair Instance Hallucination Rate ($CHAIR_i$)", fontweight="bold")
    ax1.set_ylabel("$CHAIR_i$ (%) - Lower is Better")
    ax1.set_xlabel("Visual Context Retention")
    ax1.set_xticks(x)
    ax1.set_xticklabels(RETENTIONS)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Panel 2: Object Precision (higher is better)
    ax2.bar(x - width/2, msd_vc_precision, width, label="MSD (Depth 3)", color="#94a3b8", edgecolor="black", linewidth=0.8)
    ax2.bar(x + width/2, dflash_vc_precision, width, label="DFlash (Depth 3)", color="#10b981", edgecolor="black", linewidth=0.8)
    ax2.set_title("VideoChair Object Precision ($1 - CHAIR_i$)", fontweight="bold")
    ax2.set_ylabel("Object Precision (%) - Higher is Better")
    ax2.set_xlabel("Visual Context Retention")
    ax2.set_xticks(x)
    ax2.set_xticklabels(RETENTIONS)
    ax2.set_ylim(0, 40)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.suptitle("Object Hallucination & Precision on VideoChair: DFlash vs. MSD", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUTPUT_DIR / "fig3_dflash_vs_msd_hallucination.png")
    plt.close()
    print("Saved fig3_dflash_vs_msd_hallucination.png")


if __name__ == "__main__":
    plot_speedup()
    plot_tau_and_acceptance()
    plot_hallucination_metrics()
    print(f"All figures generated successfully in {OUTPUT_DIR}!")
