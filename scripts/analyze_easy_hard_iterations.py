#!/usr/bin/env python3
"""Comprehensive analysis and visualization of Easy, Hard, and Visual-Heavy Iterations in MSD on VDC."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Linguistic token vocabulary classification
EASY_T0_TOKENS = {
    # Determiners & Demonstratives
    " a", " an", " the", " this", " that", " these", " those",
    # Prepositions & Core Function Words
    " of", " in", " on", " at", " with", " to", " by", " from", " into", " through",
    # Linking / Auxiliary Verbs
    " is", " are", " was", " were", " be", " been", " has", " have",
    # Formulaic n-gram collocations in video captioning
    " variety", " objects", " scenes", " followed", " transitions", " including", " series",
}

HARD_T0_TOKENS = {
    # Sentence Starters (after sentence period / section boundary)
    " The", "The", "In", "Next", "However", "Meanwhile",
    # Temporal Transitions & Branching Conjunctions
    " then", " and", " but", " while",
    # High-Entropy Domain Nouns / Open Subjects
    " video", " scene", " image", " clip",
}

# Concrete physical objects, actions, and visual sensory attributes
VISUAL_HEAVY_T0_TOKENS = {
    # Objects & Entities
    "person", "man", "woman", "boy", "girl", "people", "child",
    "clock", "camera", "boat", "airplane", "plane", "car", "vehicle",
    "bird", "dog", "cat", "animal", "flag", "map", "cube", "tie", "ball",
    "tree", "flower", "water", "river", "beach", "sky", "cloud", "clouds",
    "building", "house", "room", "wall", "window", "door", "table", "chair",
    "guitar", "umbrella", "sandwich", "food", "cup", "bottle",
    # Specific visual actions / postures / motions
    "standing", "holding", "walking", "running", "jumping", "flying",
    "swimming", "dancing", "singing", "playing", "sitting", "riding",
    "looking", "wearing", "moving", "shifts", "pans",
    # Specific visual colors & sensory attributes
    "white", "black", "blue", "red", "green", "yellow", "colorful", "vibrant",
    "fluffy", "wooden", "metallic"
}

RETENTION_ORDER = (100, 50, 25, 0)
RETENTION_LABELS = {
    100: "100% Visual (Full)",
    50: "50% Visual",
    25: "25% Visual",
    0: "0% Visual (None)",
}
CATEGORIES = ("EASY", "HARD", "GENERAL_CONTENT", "VISUAL_HEAVY")


def classify_round(token_0: str) -> str:
    """Classify a speculative round based on its first proposed token t_0."""
    t = token_0.strip().lower()
    raw = token_0
    if raw in EASY_T0_TOKENS or t in {"a", "an", "the", "this", "that", "these", "those", "in", "on", "at", "of", "with", "to", "is", "are"}:
        return "EASY"
    if raw in HARD_T0_TOKENS or raw in {" The", "The", " then", " and", " video", " scene", " image"}:
        return "HARD"
    if t in VISUAL_HEAVY_T0_TOKENS:
        return "VISUAL_HEAVY"
    return "GENERAL_CONTENT"


def load_and_analyze_rounds(jsonl_path: Path) -> tuple[dict[int, dict[str, list[dict]]], list[dict], list[dict]]:
    """Parse raw journal and extract detailed round metrics grouped by retention and difficulty."""
    data: dict[int, dict[str, list[dict]]] = {
        ret: {cat: [] for cat in CATEGORIES} for ret in RETENTION_ORDER
    }
    
    decile_counts: dict[tuple[int, str, int], int] = defaultdict(int)
    decile_totals: dict[tuple[int, int], int] = defaultdict(int)

    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in (None, "ok") or row.get("condition") != "retention":
                continue
            try:
                ret = int(float(row["retention_percentage"]))
            except (KeyError, TypeError, ValueError):
                continue
            if ret not in RETENTION_ORDER:
                continue

            target_len = len(row.get("target_output_ids") or [])
            rounds = row.get("acceptance_rounds") or []
            
            for rnd in rounds:
                pos = rnd.get("position_records") or []
                if not pos:
                    continue
                t0_text = pos[0].get("speculative_token_text", "")
                cat = classify_round(t0_text)
                acc = int(rnd.get("accepted_proposal_count", 0))
                t0_pass = 1 if pos[0].get("accepted") == 1 else 0
                ans_start = int(rnd.get("answer_position_start", 0))

                pct = (ans_start / max(1, target_len)) * 100.0
                decile = min(9, int(pct // 10))
                decile_counts[(ret, cat, decile)] += 1
                decile_totals[(ret, decile)] += 1

                round_info = {
                    "round_index": int(rnd.get("round_index", 0)),
                    "token_0": t0_text,
                    "accepted_count": acc,
                    "t0_pass": t0_pass,
                    "category": cat,
                    "emitted_count": int(rnd.get("emitted_token_count", acc + 1)),
                    "answer_position_start": ans_start,
                    "target_len": target_len,
                    "decile": decile,
                }
                data[ret][cat].append(round_info)

    summary_rows: list[dict] = []
    for ret in RETENTION_ORDER:
        tot_rounds = sum(len(data[ret][c]) for c in CATEGORIES)
        for cat in CATEGORIES:
            r_list = data[ret][cat]
            n = len(r_list)
            if n == 0:
                continue
            accs = [r["accepted_count"] for r in r_list]
            p0 = [r["t0_pass"] for r in r_list]
            
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs, ddof=1)) if n > 1 else 0.0
            sem_acc = std_acc / math.sqrt(n) if n > 0 else 0.0
            p0_rate = float(np.mean(p0)) * 100.0
            zero_rate = float(np.mean([a == 0 for a in accs])) * 100.0
            d2_rate = float(np.mean([a >= 2 for a in accs])) * 100.0
            d3_rate = float(np.mean([a >= 3 for a in accs])) * 100.0
            d4_rate = float(np.mean([a >= 4 for a in accs])) * 100.0
            d5_rate = float(np.mean([a >= 5 for a in accs])) * 100.0

            summary_rows.append({
                "retention_percentage": ret,
                "category": cat,
                "n_rounds": n,
                "pct_of_total_rounds": round((n / tot_rounds) * 100.0, 2),
                "t0_pass_rate_pct": round(p0_rate, 2),
                "zero_acc_rate_pct": round(zero_rate, 2),
                "mean_acc_tokens": round(mean_acc, 3),
                "std_acc_tokens": round(std_acc, 3),
                "sem_acc_tokens": round(sem_acc, 3),
                "depth_ge_2_pct": round(d2_rate, 2),
                "depth_ge_3_pct": round(d3_rate, 2),
                "depth_ge_4_pct": round(d4_rate, 2),
                "depth_ge_5_pct": round(d5_rate, 2),
            })

    timeline_rows: list[dict] = []
    for ret in (100, 0):
        for decile in range(10):
            tot = decile_totals.get((ret, decile), 0)
            if tot == 0:
                continue
            e_cnt = decile_counts.get((ret, "EASY", decile), 0)
            h_cnt = decile_counts.get((ret, "HARD", decile), 0)
            g_cnt = decile_counts.get((ret, "GENERAL_CONTENT", decile), 0)
            v_cnt = decile_counts.get((ret, "VISUAL_HEAVY", decile), 0)
            timeline_rows.append({
                "retention_percentage": ret,
                "progress_decile": f"{decile*10}-{(decile+1)*10}%",
                "decile_idx": decile,
                "easy_pct": round(e_cnt / tot * 100.0, 2),
                "hard_pct": round(h_cnt / tot * 100.0, 2),
                "general_content_pct": round(g_cnt / tot * 100.0, 2),
                "visual_heavy_pct": round(v_cnt / tot * 100.0, 2),
                "total_rounds": tot,
            })

    return data, summary_rows, timeline_rows


def plot_easy_vs_hard_figures(
    summary_rows: list[dict],
    timeline_rows: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_map = {(r["retention_percentage"], r["category"]): r for r in summary_rows}

    # -------------------------------------------------------------
    # Figure 1: 3-Panel Core Performance Comparison (4 Categories)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    
    cats = ["EASY", "HARD", "GENERAL_CONTENT", "VISUAL_HEAVY"]
    cat_labels = ["DỄ (Easy)", "KHÓ (Hard)", "Nội dung chung", "Visual-Heavy"]
    x = np.arange(len(cats))
    width = 0.35

    # Panel A: Token 0 Pass Rate (%)
    ax_a = axes[0]
    p0_100 = [row_map[(100, c)]["t0_pass_rate_pct"] for c in cats]
    p0_0 = [row_map[(0, c)]["t0_pass_rate_pct"] for c in cats]
    ax_a.bar(x - width/2, p0_100, width, label="100% Visual (Full)", color="#2563eb", alpha=0.9)
    ax_a.bar(x + width/2, p0_0, width, label="0% Visual (None)", color="#dc2626", alpha=0.9)
    ax_a.set_title("A. Tỷ lệ qua Cửa ải Token đầu t_0 (%)", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("Tỷ lệ t_0 được Accept (%)", fontsize=10)
    ax_a.set_xticks(x, cat_labels, fontsize=9.5)
    ax_a.set_ylim(0, 115)
    ax_a.grid(axis="y", linestyle="--", alpha=0.3)
    ax_a.legend(frameon=True, facecolor="white", fontsize=9, loc="lower right")

    for i in range(len(cats)):
        ax_a.text(i - width/2, p0_100[i] + 2, f"{p0_100[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
        ax_a.text(i + width/2, p0_0[i] + 2, f"{p0_0[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#991b1b")

    # Panel B: Mean Accepted Tokens / Round
    ax_b = axes[1]
    m_100 = [row_map[(100, c)]["mean_acc_tokens"] for c in cats]
    m_0 = [row_map[(0, c)]["mean_acc_tokens"] for c in cats]
    err_100 = [row_map[(100, c)]["sem_acc_tokens"] for c in cats]
    err_0 = [row_map[(0, c)]["sem_acc_tokens"] for c in cats]
    ax_b.bar(x - width/2, m_100, width, yerr=err_100, label="100% Visual", color="#2563eb", alpha=0.9, capsize=4)
    ax_b.bar(x + width/2, m_0, width, yerr=err_0, label="0% Visual", color="#dc2626", alpha=0.9, capsize=4)
    ax_b.set_title("B. Số Token Accept trung bình / Round", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Mean Accepted Proposals (Tokens)", fontsize=10)
    ax_b.set_xticks(x, cat_labels, fontsize=9.5)
    ax_b.set_ylim(0, 5.2)
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)
    ax_b.legend(frameon=True, facecolor="white", fontsize=9, loc="upper left")

    for i in range(len(cats)):
        ax_b.text(i - width/2, m_100[i] + 0.15, f"{m_100[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
        ax_b.text(i + width/2, m_0[i] + 0.15, f"{m_0[i]:.2f}", ha="center", fontsize=8.5, fontweight="bold", color="#991b1b")

    # Panel C: Zero-Acceptance Rate (%) (Round sụp đổ)
    ax_c = axes[2]
    z_100 = [row_map[(100, c)]["zero_acc_rate_pct"] for c in cats]
    z_0 = [row_map[(0, c)]["zero_acc_rate_pct"] for c in cats]
    ax_c.bar(x - width/2, z_100, width, label="100% Visual", color="#2563eb", alpha=0.9)
    ax_c.bar(x + width/2, z_0, width, label="0% Visual", color="#dc2626", alpha=0.9)
    ax_c.set_title("C. Tỷ lệ Round Sụp đổ (0 Token Accept)", fontsize=11, fontweight="bold")
    ax_c.set_ylabel("Tỷ lệ Round bị lãng phí (%)", fontsize=10)
    ax_c.set_xticks(x, cat_labels, fontsize=9.5)
    ax_c.set_ylim(0, 85)
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)
    ax_c.legend(frameon=True, facecolor="white", fontsize=9, loc="upper right")

    for i in range(len(cats)):
        ax_c.text(i - width/2, z_100[i] + 1.8, f"{z_100[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#1e3a8a")
        ax_c.text(i + width/2, z_0[i] + 1.8, f"{z_0[i]:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#991b1b")

    fig.suptitle(
        "So sánh Hiệu năng giữa 4 Hạng mục Iteration (MSD trên VDC-50)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    p1 = output_dir / "easy_vs_hard_performance_bars.png"
    fig.savefig(p1, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # Figure 2: Survival Curve by Depth (ge 1 to ge 5 tokens)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    depth_labels = ["≥1 tok", "≥2 toks", "≥3 toks", "≥4 toks", "≥5 toks"]
    d_keys = ["t0_pass_rate_pct", "depth_ge_2_pct", "depth_ge_3_pct", "depth_ge_4_pct", "depth_ge_5_pct"]

    # Panel A: 100% Visual Retention
    ax_1 = axes[0]
    easy_d_100 = [row_map[(100, "EASY")][k] for k in d_keys]
    hard_d_100 = [row_map[(100, "HARD")][k] for k in d_keys]
    gen_d_100 = [row_map[(100, "GENERAL_CONTENT")][k] for k in d_keys]
    vis_d_100 = [row_map[(100, "VISUAL_HEAVY")][k] for k in d_keys]

    ax_1.plot(depth_labels, easy_d_100, marker="o", linewidth=2.2, color="#16a34a", label="Iter DỄ (Easy)")
    ax_1.plot(depth_labels, vis_d_100, marker="D", linewidth=2.4, color="#f59e0b", label="Iter Visual-Heavy")
    ax_1.plot(depth_labels, gen_d_100, marker="s", linewidth=1.8, color="#2563eb", linestyle="--", label="Iter Nội dung chung")
    ax_1.plot(depth_labels, hard_d_100, marker="^", linewidth=2.2, color="#dc2626", label="Iter KHÓ (Hard)")
    ax_1.set_title("A. 100% Visual Retention (Có Video)", fontsize=11, fontweight="bold")
    ax_1.set_ylabel("Tỷ lệ Round giữ được độ sâu (%)", fontsize=10)
    ax_1.set_ylim(-2, 105)
    ax_1.grid(True, linestyle="--", alpha=0.3)
    ax_1.legend(frameon=True, facecolor="white", fontsize=9, loc="upper right")

    # Panel B: 0% Visual Retention
    ax_2 = axes[1]
    easy_d_0 = [row_map[(0, "EASY")][k] for k in d_keys]
    hard_d_0 = [row_map[(0, "HARD")][k] for k in d_keys]
    gen_d_0 = [row_map[(0, "GENERAL_CONTENT")][k] for k in d_keys]
    vis_d_0 = [row_map[(0, "VISUAL_HEAVY")][k] for k in d_keys]

    ax_2.plot(depth_labels, easy_d_0, marker="o", linewidth=2.2, color="#16a34a", label="Iter DỄ (Easy)")
    ax_2.plot(depth_labels, vis_d_0, marker="D", linewidth=2.4, color="#f59e0b", label="Iter Visual-Heavy")
    ax_2.plot(depth_labels, gen_d_0, marker="s", linewidth=1.8, color="#2563eb", linestyle="--", label="Iter Nội dung chung")
    ax_2.plot(depth_labels, hard_d_0, marker="^", linewidth=2.2, color="#dc2626", label="Iter KHÓ (Hard)")
    ax_2.set_title("B. 0% Visual Retention (Không Video)", fontsize=11, fontweight="bold")
    ax_2.set_ylabel("Tỷ lệ Round giữ được độ sâu (%)", fontsize=10)
    ax_2.set_ylim(-2, 105)
    ax_2.grid(True, linestyle="--", alpha=0.3)
    ax_2.legend(frameon=True, facecolor="white", fontsize=9, loc="lower left")

    fig.suptitle(
        "Tỷ lệ sống sót chuỗi Draft theo Chiều sâu Cây (Draft Tree Survival Curve)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    p2 = output_dir / "easy_vs_hard_depth_survival.png"
    fig.savefig(p2, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # Figure 3: Timeline Distribution across Output Progress
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    t_100 = [r for r in timeline_rows if r["retention_percentage"] == 100]
    deciles = [r["progress_decile"] for r in t_100]
    easy_p = [r["easy_pct"] for r in t_100]
    vis_p = [r["visual_heavy_pct"] for r in t_100]
    gen_p = [r["general_content_pct"] for r in t_100]
    hard_p = [r["hard_pct"] for r in t_100]

    idx_x = np.arange(len(deciles))
    ax.bar(idx_x, easy_p, label="Iter DỄ (Function/Template)", color="#16a34a", alpha=0.85)
    ax.bar(idx_x, vis_p, bottom=easy_p, label="Iter Visual-Heavy", color="#f59e0b", alpha=0.85)
    bottom_gen = np.array(easy_p) + np.array(vis_p)
    ax.bar(idx_x, gen_p, bottom=bottom_gen, label="Iter Nội dung chung", color="#3b82f6", alpha=0.85)
    bottom_hard = bottom_gen + np.array(gen_p)
    ax.bar(idx_x, hard_p, bottom=bottom_hard, label="Iter KHÓ (Đầu câu / Rẽ nhánh)", color="#ef4444", alpha=0.85)

    ax.set_title("Phân bố 4 Hạng Iteration dọc theo Tiến trình Câu trả lời (VDC-50)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tiến trình độ dài câu trả lời (Output Decile)", fontsize=10.5)
    ax.set_ylabel("Tỷ trọng trong các vòng lặp (%)", fontsize=10.5)
    ax.set_xticks(idx_x, deciles, fontsize=9.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", fontsize=9, loc="upper right")

    p3 = output_dir / "easy_hard_timeline_distribution.png"
    fig.savefig(p3, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p3}")


def main() -> None:
    jsonl_path = Path("results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl")
    output_dir = Path("results/6_10_26")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing journal: {jsonl_path}...")
    data, summary_rows, timeline_rows = load_and_analyze_rounds(jsonl_path)

    summary_csv = output_dir / "easy_hard_iteration_comparison.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved: {summary_csv} ({len(summary_rows)} rows)")

    timeline_csv = output_dir / "easy_hard_timeline_distribution.csv"
    with timeline_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timeline_rows)
    print(f"Saved: {timeline_csv} ({len(timeline_rows)} rows)")

    plot_easy_vs_hard_figures(summary_rows, timeline_rows, output_dir)
    print("Completed script execution successfully!")


if __name__ == "__main__":
    main()
