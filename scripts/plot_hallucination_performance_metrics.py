#!/usr/bin/env python3
"""Vẽ biểu đồ các chỉ số Performance & Hallucination (CHAIR, Accuracy, Precision, Recall)

Tạo biểu đồ trực quan hóa lưu vào results/6_10_26/fig12_hallucination_performance_metrics.png
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set style
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

with open(OUTPUT_DIR / "hallucination_performance_metrics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vid_data = data["vidhalluc"]
chair_data = data["videochair"]

cuts = ["100% (Full)", "50% Cut", "25% Cut", "0% (Text-Only)"]
cut_keys = ["100", "50", "25", "0"]
x = np.arange(len(cuts))
w = 0.25

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: VidHalluc Performance & Hallucination Rate
bin_acc = [vid_data[k]["bin_acc"] * 100 for k in cut_keys]
trans_acc = [vid_data[k]["trans_change_acc"] * 100 for k in cut_keys]
bin_halluc = [(1.0 - vid_data[k]["bin_acc"]) * 100 for k in cut_keys]

rects1 = ax1.bar(x - w, bin_acc, w, label="BinaryQA Accuracy (Yes/No)", color="#10b981", edgecolor="black")
rects2 = ax1.bar(x, trans_acc, w, label="Scene Transition Detection", color="#3b82f6", edgecolor="black")
rects3 = ax1.bar(x + w, bin_halluc, w, label="Binary Hallucination Rate", color="#ef4444", edgecolor="black")

ax1.set_title("VidHalluc: Độ chính xác & Tỷ lệ Ảo giác theo Độ Cắt\n(Bảo toàn Bất biến 100% qua 4 Mức Cắt)", fontweight="bold")
ax1.set_ylabel("Tỷ lệ Phần trăm (%)")
ax1.set_xlabel("Mức Độ Cắt Visual Context")
ax1.set_xticks(x)
ax1.set_xticklabels(cuts)
ax1.set_ylim(0, 100)
ax1.grid(axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

for i in range(len(cuts)):
    ax1.annotate(f"{bin_acc[i]:.1f}%", (x[i] - w, bin_acc[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax1.annotate(f"{trans_acc[i]:.1f}%", (x[i], trans_acc[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax1.annotate(f"{bin_halluc[i]:.1f}%", (x[i] + w, bin_halluc[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

# Panel 2: VideoChair (CHAIR Metrics)
chair_i = [chair_data[k]["chair_i"] * 100 for k in cut_keys]
obj_rec = [chair_data[k]["object_recall"] * 100 for k in cut_keys]
obj_prec = [chair_data[k]["object_precision"] * 100 for k in cut_keys]

rects4 = ax2.bar(x - w, chair_i, w, label="CHAIR_i (Tỷ lệ Object Ảo giác)", color="#f59e0b", edgecolor="black")
rects5 = ax2.bar(x, obj_rec, w, label="Object Recall (Độ phủ Thực thể)", color="#8b5cf6", edgecolor="black")
rects6 = ax2.bar(x + w, obj_prec, w, label="Object Precision (Độ chuẩn xác)", color="#06b6d4", edgecolor="black")

ax2.set_title("VideoChair: Chỉ số Ảo giác CHAIR_i, Recall & Precision\n(Tính Ổn định Tuyệt đối giữa 100% vs 0% Cut)", fontweight="bold")
ax2.set_ylabel("Tỷ lệ Phần trăm (%)")
ax2.set_xlabel("Mức Độ Cắt Visual Context")
ax2.set_xticks(x)
ax2.set_xticklabels(cuts)
ax2.set_ylim(0, 110)
ax2.grid(axis="y", linestyle=":", alpha=0.6)
ax2.legend(loc="upper right")

for i in range(len(cuts)):
    ax2.annotate(f"{chair_i[i]:.1f}%", (x[i] - w, chair_i[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax2.annotate(f"{obj_rec[i]:.1f}%", (x[i], obj_rec[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax2.annotate(f"{obj_prec[i]:.1f}%", (x[i] + w, obj_prec[i]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Hình 9: Đánh giá Định lượng Hiệu năng Task & Chỉ số Ảo giác (VidHalluc & VideoChair CHAIR)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

out_file = OUTPUT_DIR / "fig12_hallucination_performance_metrics.png"
plt.savefig(out_file)
plt.close()
print(f"Đã lưu biểu đồ vào {out_file}!")
