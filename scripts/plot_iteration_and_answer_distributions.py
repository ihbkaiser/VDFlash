#!/usr/bin/env python3
"""Plot tree depth survival curves and answer distribution for BAO_CAO_VIDHALLUC_VIDEOCHAIR.md"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style
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


def plot_tree_depth_survival():
    """Plot Tree Depth Survival Curves (Depth 1 to 5) for 100% vs 0% Retention."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    depths = [1, 2, 3, 4, 5]

    # VidHalluc
    v_100 = [65.2, 27.4, 7.3, 3.4, 0.9]
    v_0 = [93.8, 75.8, 47.3, 27.2, 18.7]

    ax1.plot(depths, v_100, marker="o", markersize=8, linewidth=2.5, color="#2563eb", label="100% Visual Retention")
    ax1.plot(depths, v_0, marker="s", markersize=8, linewidth=2.5, color="#dc2626", label="0% Retention (Text-Only)")
    ax1.set_title("VidHalluc: Tỷ lệ Sống sót theo Độ sâu Cây\n(Tree Depth Survival Rate)", fontweight="bold")
    ax1.set_xlabel("Độ sâu Cây Draft (Tree Depth >= k tokens)")
    ax1.set_ylabel("Tỷ lệ Iterations Chấp nhận (%)")
    ax1.set_xticks(depths)
    ax1.set_xticklabels(["Depth 1", "Depth 2", "Depth 3", "Depth 4", "Depth 5"])
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    for x, y in zip(depths, v_100):
        ax1.annotate(f"{y:.1f}%", (x, y), xytext=(0, -15), textcoords="offset points", ha="center", color="#2563eb", fontweight="bold", fontsize=9)
    for x, y in zip(depths, v_0):
        ax1.annotate(f"{y:.1f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", color="#dc2626", fontweight="bold", fontsize=9)

    # VideoChair
    c_100 = [82.5, 49.3, 16.2, 3.2, 0.4]
    c_0 = [96.5, 86.4, 69.0, 52.2, 42.2]

    ax2.plot(depths, c_100, marker="o", markersize=8, linewidth=2.5, color="#f59e0b", label="100% Visual Retention")
    ax2.plot(depths, c_0, marker="s", markersize=8, linewidth=2.5, color="#10b981", label="0% Retention (Text-Only)")
    ax2.set_title("VideoChair: Tỷ lệ Sống sót theo Độ sâu Cây\n(Tree Depth Survival Rate)", fontweight="bold")
    ax2.set_xlabel("Độ sâu Cây Draft (Tree Depth >= k tokens)")
    ax2.set_ylabel("Tỷ lệ Iterations Chấp nhận (%)")
    ax2.set_xticks(depths)
    ax2.set_xticklabels(["Depth 1", "Depth 2", "Depth 3", "Depth 4", "Depth 5"])
    ax2.set_ylim(-2, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    for x, y in zip(depths, c_100):
        ax2.annotate(f"{y:.1f}%", (x, y), xytext=(0, -15), textcoords="offset points", ha="center", color="#f59e0b", fontweight="bold", fontsize=9)
    for x, y in zip(depths, c_0):
        ax2.annotate(f"{y:.1f}%", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", color="#10b981", fontweight="bold", fontsize=9)

    plt.suptitle("Hình 7: Khả năng Sinh tồn của Cây Suy đoán (Tree Depth Survival Curve): 100% vs 0% Retention", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig5_tree_depth_survival_curve.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_timeline_and_answer_distribution():
    """Plot timeline progression and answer percentage distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1: VidHalluc Answer Distribution across 4 cut levels
    cuts = ["100% Retention", "50% Cut", "25% Cut", "0% Text-Only"]
    x = np.arange(len(cuts))
    w = 0.25

    yes_pct = [23.0, 23.0, 23.0, 23.0]
    no_pct = [16.0, 16.0, 16.0, 16.0]
    desc_pct = [61.0, 61.0, 61.0, 61.0]

    ax1.bar(x - w, yes_pct, w, label="Khẳng định 'Yes'", color="#10b981", edgecolor="black")
    ax1.bar(x, no_pct, w, label="Phủ định 'No'", color="#ef4444", edgecolor="black")
    ax1.bar(x + w, desc_pct, w, label="Mô tả Mở (Descriptive)", color="#3b82f6", edgecolor="black")

    ax1.set_title("VidHalluc: Tỷ lệ Phần trăm Câu trả lời Output (%)\n(Bảo toàn Bất biến 100% qua 4 Độ Cắt)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cuts)
    ax1.set_ylabel("Tỷ lệ Phần trăm trong Output (%)")
    ax1.set_ylim(0, 75)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    for i in range(len(cuts)):
        ax1.annotate(f"{yes_pct[i]:.0f}%", (x[i] - w, yes_pct[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        ax1.annotate(f"{no_pct[i]:.0f}%", (x[i], no_pct[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        ax1.annotate(f"{desc_pct[i]:.0f}%", (x[i] + w, desc_pct[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Performance over Timeline (VideoChair)
    timeline_labels = ["Rounds 0-5\n(Mở đầu)", "Rounds 6-15\n(Đoạn đầu)", "Rounds 16-30\n(Đoạn giữa)", "Rounds 31-60\n(Nội dung)", "Rounds 60+\n(Đoạn kết)"]
    x_tl = np.arange(len(timeline_labels))
    w_tl = 0.35

    len_100 = [0.51, 0.98, 1.18, 1.40, 1.82]
    len_0 = [2.42, 2.65, 2.99, 3.72, 4.23]

    ax2.bar(x_tl - w_tl/2, len_100, w_tl, label="100% Visual Retention", color="#f59e0b", edgecolor="black")
    ax2.bar(x_tl + w_tl/2, len_0, w_tl, label="0% Retention (Text-Only)", color="#8b5cf6", edgecolor="black")

    ax2.set_title("VideoChair: Chiều dài Accept TB theo Timeline Thế hệ\n(Từ Mở đầu câu đến Cuối đoạn văn)", fontweight="bold")
    ax2.set_xticks(x_tl)
    ax2.set_xticklabels(timeline_labels)
    ax2.set_ylabel("Số tokens được Accept / Round")
    ax2.set_ylim(0, 5.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    for i in range(len(timeline_labels)):
        ax2.annotate(f"{len_100[i]:.2f}", (x_tl[i] - w_tl/2, len_100[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        ax2.annotate(f"{len_0[i]:.2f}", (x_tl[i] + w_tl/2, len_0[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Hình 8: Phân bố Câu trả lời Output và Động học Chấp nhận theo Timeline Văn bản", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig6_timeline_and_answer_distribution.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    plot_tree_depth_survival()
    plot_timeline_and_answer_distribution()
    print("New figures successfully created in results/6_10_26/!")


if __name__ == "__main__":
    main()
