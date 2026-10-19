#!/usr/bin/env python3
"""Generate comprehensive publication-quality comparison plots across cut levels for BAO_CAO_VIDHALLUC_VIDEOCHAIR.md"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling settings
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

OUTPUT_DIR = Path("results/6_10_26")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUT_LEVELS = [100.0, 50.0, 25.0, 0.0]
CUT_LABELS = ["100% (No Cut)", "50% Cut", "25% Cut", "0% (Full Cut / Text-only)"]
CUT_TICKS = ["100%", "50%", "25%", "0%"]

# Exact metrics extracted from runs
VID_METRICS = {
    "retention": [100.0, 50.0, 25.0, 0.0],
    "decode_speedup": [0.631, 0.718, 0.836, 1.140],
    "e2e_speedup": [1.194, 1.276, 1.351, 1.485],
    "decode_sec": [1.899, 1.710, 1.468, 1.026],
    "prefill_sec": [2.489, 2.433, 2.409, 2.389],
    "e2e_sec": [4.388, 4.143, 3.877, 3.415],
    "ar_e2e_sec": [5.238, 5.238, 5.238, 5.238],
    "ar_decode_sec": [1.163, 1.163, 1.163, 1.163],
    "ar_prefill_sec": [4.075, 4.075, 4.075, 4.075],
    "mean_acc_len": [1.042, 1.262, 1.599, 2.628],
    "zero_acc_rate": [34.80, 28.52, 18.80, 6.17],
    "t0_acc_rate": [65.20, 71.48, 81.20, 93.83],
    "verification_steps": [18.16, 16.62, 14.47, 10.38],
    "lossless_rate": [95.0, 94.0, 93.0, 94.0],
}

CHAIR_METRICS = {
    "retention": [100.0, 50.0, 25.0, 0.0],
    "decode_speedup": [0.672, 1.025, 1.124, 1.215],
    "e2e_speedup": [0.772, 1.117, 1.208, 1.290],
    "decode_sec": [10.961, 7.325, 6.646, 6.176],
    "prefill_sec": [1.100, 1.077, 1.068, 1.058],
    "e2e_sec": [12.061, 8.402, 7.715, 7.234],
    "ar_e2e_sec": [9.509, 9.509, 9.509, 9.509],
    "ar_decode_sec": [7.350, 7.350, 7.350, 7.350],
    "ar_prefill_sec": [2.159, 2.159, 2.159, 2.159],
    "mean_acc_len": [1.516, 2.778, 3.177, 3.463],
    "zero_acc_rate": [17.54, 5.69, 4.58, 3.50],
    "t0_acc_rate": [82.46, 94.31, 95.42, 96.50],
    "verification_steps": [125.69, 85.08, 77.66, 72.56],
    "lossless_rate": [78.0, 81.0, 76.0, 78.0],
}


def plot_fig1_speedup_by_cut():
    """Plot 1: Speedup across cut levels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(CUT_LEVELS))
    w = 0.35

    # VidHalluc
    r1 = ax1.bar(x - w/2, VID_METRICS["decode_speedup"], w, label="Decode Speedup", color="#3b82f6", edgecolor="black")
    r2 = ax1.bar(x + w/2, VID_METRICS["e2e_speedup"], w, label="End-to-End Speedup", color="#10b981", edgecolor="black")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="AR Baseline (1.0x)")
    ax1.set_title("VidHalluc (100 samples / Short QA)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CUT_TICKS)
    ax1.set_xlabel("Visual Retention Level (Độ giữ lại Visual Tokens)")
    ax1.set_ylabel("Tốc độ Tăng tốc (Speedup vs AR Baseline)")
    ax1.set_ylim(0, 1.8)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    for rect in r1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}x", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in r2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}x", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # VideoChair
    r3 = ax2.bar(x - w/2, CHAIR_METRICS["decode_speedup"], w, label="Decode Speedup", color="#f59e0b", edgecolor="black")
    r4 = ax2.bar(x + w/2, CHAIR_METRICS["e2e_speedup"], w, label="End-to-End Speedup", color="#8b5cf6", edgecolor="black")
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="AR Baseline (1.0x)")
    ax2.set_title("VideoChair (100 samples / Long Captions)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CUT_TICKS)
    ax2.set_xlabel("Visual Retention Level (Độ giữ lại Visual Tokens)")
    ax2.set_ylabel("Tốc độ Tăng tốc (Speedup vs AR Baseline)")
    ax2.set_ylim(0, 1.8)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    for rect in r3:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}x", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in r4:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}x", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Hình 2: So sánh Tốc độ Tăng tốc (Decode vs End-to-End Speedup) theo 4 Mức Cắt Visual", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig1_speedup_by_cut_level.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_fig2_efficiency_by_cut():
    """Plot 2: Speculative Efficiency (Accepted Length, Zero-Acc Rate, Verification Steps)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))
    x = [100, 50, 25, 0]

    # Panel 1: Mean Accepted Length
    ax1.plot(x, VID_METRICS["mean_acc_len"], marker="o", markersize=8, color="#16a34a", linewidth=2.5, label="VidHalluc")
    ax1.plot(x, CHAIR_METRICS["mean_acc_len"], marker="s", markersize=8, color="#8b5cf6", linewidth=2.5, label="VideoChair")
    ax1.set_title("(a) Chiều dài Chấp nhận TB / Round\n(Mean Accepted Length - Cao hơn là tốt hơn)", fontweight="bold")
    ax1.set_xlabel("Visual Retention (%)")
    ax1.set_ylabel("Số tokens được accept / Round")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CUT_TICKS)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()
    for pt_x, pt_y in zip(x, VID_METRICS["mean_acc_len"]):
        ax1.annotate(f"{pt_y:.2f}", (pt_x, pt_y), xytext=(0, -15), textcoords="offset points", ha="center", color="#16a34a", fontweight="bold", fontsize=9)
    for pt_x, pt_y in zip(x, CHAIR_METRICS["mean_acc_len"]):
        ax1.annotate(f"{pt_y:.2f}", (pt_x, pt_y), xytext=(0, 8), textcoords="offset points", ha="center", color="#8b5cf6", fontweight="bold", fontsize=9)

    # Panel 2: Zero-Acceptance Rate
    ax2.plot(x, VID_METRICS["zero_acc_rate"], marker="o", markersize=8, color="#dc2626", linewidth=2.5, label="VidHalluc")
    ax2.plot(x, CHAIR_METRICS["zero_acc_rate"], marker="s", markersize=8, color="#2563eb", linewidth=2.5, label="VideoChair")
    ax2.set_title("(b) Tỷ lệ Zero-Acceptance (%)\n(Round bị từ chối 0 token - Thấp hơn là tốt hơn)", fontweight="bold")
    ax2.set_xlabel("Visual Retention (%)")
    ax2.set_ylabel("Tỷ lệ Rounds Zero-Acc (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CUT_TICKS)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    for pt_x, pt_y in zip(x, VID_METRICS["zero_acc_rate"]):
        ax2.annotate(f"{pt_y:.1f}%", (pt_x, pt_y), xytext=(0, 7), textcoords="offset points", ha="center", color="#dc2626", fontweight="bold", fontsize=9)
    for pt_x, pt_y in zip(x, CHAIR_METRICS["zero_acc_rate"]):
        ax2.annotate(f"{pt_y:.1f}%", (pt_x, pt_y), xytext=(0, -15), textcoords="offset points", ha="center", color="#2563eb", fontweight="bold", fontsize=9)

    # Panel 3: Verification Steps Count
    ax3.plot(x, VID_METRICS["verification_steps"], marker="o", markersize=8, color="#f59e0b", linewidth=2.5, label="VidHalluc")
    ax3.plot(x, CHAIR_METRICS["verification_steps"], marker="s", markersize=8, color="#0ea5e9", linewidth=2.5, label="VideoChair")
    ax3.set_title("(c) Số Vòng Xác minh TB (Steps)\n(Verification Rounds - Ít hơn là nhanh hơn)", fontweight="bold")
    ax3.set_xlabel("Visual Retention (%)")
    ax3.set_ylabel("Số vòng xác minh trung bình (rounds)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(CUT_TICKS)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()
    for pt_x, pt_y in zip(x, VID_METRICS["verification_steps"]):
        ax3.annotate(f"{pt_y:.1f}", (pt_x, pt_y), xytext=(0, 7), textcoords="offset points", ha="center", color="#f59e0b", fontweight="bold", fontsize=9)
    for pt_x, pt_y in zip(x, CHAIR_METRICS["verification_steps"]):
        ax3.annotate(f"{pt_y:.1f}", (pt_x, pt_y), xytext=(0, 7), textcoords="offset points", ha="center", color="#0ea5e9", fontweight="bold", fontsize=9)

    plt.suptitle("Hình 3: Hiệu năng Suy đoán Đa phương thức (Độ dài Chấp nhận, Zero-Acc, và Số Vòng Xác minh) theo Độ Cắt Visual", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig2_efficiency_by_cut_level.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_fig3_latency_breakdown_by_cut():
    """Plot 3: Latency breakdown (Prefill vs Decode) compared to AR baseline."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # VidHalluc
    labels_v = ["AR Baseline", "100% Retention", "50% Cut", "25% Cut", "0% Text-Only"]
    prefill_v = [VID_METRICS["ar_prefill_sec"][0]] + VID_METRICS["prefill_sec"]
    decode_v = [VID_METRICS["ar_decode_sec"][0]] + VID_METRICS["decode_sec"]
    x_v = np.arange(len(labels_v))

    p1 = ax1.bar(x_v, prefill_v, label="Prefill Time (Nạp Video)", color="#6366f1", edgecolor="black", width=0.55)
    p2 = ax1.bar(x_v, decode_v, bottom=prefill_v, label="Decode Time (Sinh Token)", color="#f43f5e", edgecolor="black", width=0.55)
    ax1.set_title("VidHalluc: Phân bổ Độ trễ Prefill vs Decode (Giây)", fontweight="bold")
    ax1.set_xticks(x_v)
    ax1.set_xticklabels(labels_v, rotation=15)
    ax1.set_ylabel("Thời gian thực thi (Giây)")
    ax1.set_ylim(0, 6.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    for i in range(len(labels_v)):
        total = prefill_v[i] + decode_v[i]
        ax1.annotate(f"{total:.2f}s\n(P:{prefill_v[i]:.2f}s, D:{decode_v[i]:.2f}s)", (x_v[i], total),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    # VideoChair
    labels_c = ["AR Baseline", "100% Retention", "50% Cut", "25% Cut", "0% Text-Only"]
    prefill_c = [CHAIR_METRICS["ar_prefill_sec"][0]] + CHAIR_METRICS["prefill_sec"]
    decode_c = [CHAIR_METRICS["ar_decode_sec"][0]] + CHAIR_METRICS["decode_sec"]
    x_c = np.arange(len(labels_c))

    p3 = ax2.bar(x_c, prefill_c, label="Prefill Time (Nạp Video)", color="#6366f1", edgecolor="black", width=0.55)
    p4 = ax2.bar(x_c, decode_c, bottom=prefill_c, label="Decode Time (Sinh Token)", color="#f43f5e", edgecolor="black", width=0.55)
    ax2.set_title("VideoChair: Phân bổ Độ trễ Prefill vs Decode (Giây)", fontweight="bold")
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(labels_c, rotation=15)
    ax2.set_ylabel("Thời gian thực thi (Giây)")
    ax2.set_ylim(0, 14.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    for i in range(len(labels_c)):
        total = prefill_c[i] + decode_c[i]
        ax2.annotate(f"{total:.2f}s\n(P:{prefill_c[i]:.2f}s, D:{decode_c[i]:.2f}s)", (x_c[i], total),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Hình 4: Phân rã Định lượng Độ trễ Prefill vs Decode giữa MSD và Autoregressive Gốc", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig3_latency_breakdown_by_cut_level.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_fig4_t0_and_lossless():
    """Plot 4: Token 0 Acceptance Rate and Lossless Parity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = [100, 50, 25, 0]

    # Panel 1: T0 acceptance rate
    ax1.plot(x, VID_METRICS["t0_acc_rate"], marker="o", markersize=8, color="#059669", linewidth=2.5, label="VidHalluc")
    ax1.plot(x, CHAIR_METRICS["t0_acc_rate"], marker="s", markersize=8, color="#2563eb", linewidth=2.5, label="VideoChair")
    ax1.set_title("(a) Tỷ lệ Chấp nhận Token Đầu tiên ($t_0$ Acceptance Rate)\n(Phản ánh khả năng khởi đầu thành công của Round)", fontweight="bold")
    ax1.set_xlabel("Visual Retention (%)")
    ax1.set_ylabel("Tỷ lệ $t_0$ được Target Accept (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CUT_TICKS)
    ax1.set_ylim(55, 100)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left")
    for pt_x, pt_y in zip(x, VID_METRICS["t0_acc_rate"]):
        ax1.annotate(f"{pt_y:.1f}%", (pt_x, pt_y), xytext=(0, -15), textcoords="offset points", ha="center", color="#059669", fontweight="bold", fontsize=9)
    for pt_x, pt_y in zip(x, CHAIR_METRICS["t0_acc_rate"]):
        ax1.annotate(f"{pt_y:.1f}%", (pt_x, pt_y), xytext=(0, 7), textcoords="offset points", ha="center", color="#2563eb", fontweight="bold", fontsize=9)

    # Panel 2: Lossless Rate
    ax2.plot(x, VID_METRICS["lossless_rate"], marker="o", markersize=8, color="#7c3aed", linewidth=2.5, label="VidHalluc")
    ax2.plot(x, CHAIR_METRICS["lossless_rate"], marker="s", markersize=8, color="#ea580c", linewidth=2.5, label="VideoChair")
    ax2.set_title("(b) Tỷ lệ Trùng khớp Tuyệt đối (Exact Lossless Parity)\n(Độ trung thực so với Target Greedy Baseline)", fontweight="bold")
    ax2.set_xlabel("Visual Retention (%)")
    ax2.set_ylabel("Tỷ lệ Lossless (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CUT_TICKS)
    ax2.set_ylim(60, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left")
    for pt_x, pt_y in zip(x, VID_METRICS["lossless_rate"]):
        ax2.annotate(f"{pt_y:.1f}%", (pt_x, pt_y), xytext=(0, 7), textcoords="offset points", ha="center", color="#7c3aed", fontweight="bold", fontsize=9)
    for pt_x, pt_y in zip(x, CHAIR_METRICS["lossless_rate"]):
        ax2.annotate(f"{pt_y:.1f}%", (pt_x, pt_y), xytext=(0, -15), textcoords="offset points", ha="center", color="#ea580c", fontweight="bold", fontsize=9)

    plt.suptitle("Hình 5: Khả năng Khởi đầu Chuỗi ($t_0$ Acceptance) và Tính Toàn vẹn Suy luận (Lossless Rate) theo Độ Cắt", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig4_t0_acceptance_and_lossless.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    plot_fig1_speedup_by_cut()
    plot_fig2_efficiency_by_cut()
    plot_fig3_latency_breakdown_by_cut()
    plot_fig4_t0_and_lossless()
    print("All cut-level comparison figures successfully generated in results/6_10_26/!")


if __name__ == "__main__":
    main()
