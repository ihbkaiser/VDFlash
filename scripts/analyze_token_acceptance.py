#!/usr/bin/env python3
"""Analyze which tokens and iteration contexts are easily accepted in MSD speculative decoding."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_token_and_context_acceptance(
    jsonl_path: Path, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Token-level statistics (Accepted vs Fallback)
    token_stats_100 = defaultdict(lambda: {"accepted": 0, "fallback": 0})
    token_stats_0 = defaultdict(lambda: {"accepted": 0, "fallback": 0})

    # 2. Preceding token context -> next round acceptance length
    prev_tok_acc_100 = defaultdict(list)
    prev_tok_acc_0 = defaultdict(list)

    # 3. Draft tree depth statistics (number of tokens accepted in round: >=1, >=2, >=3, >=4, >=5)
    depth_stats = {
        100: defaultdict(int),
        0: defaultdict(int),
    }

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
            if ret not in (100, 0):
                continue

            rounds = row.get("acceptance_rounds") or []
            target_token_stats = token_stats_100 if ret == 100 else token_stats_0
            target_prev_acc = prev_tok_acc_100 if ret == 100 else prev_tok_acc_0

            for i, r in enumerate(rounds):
                acc_cnt = int(r.get("accepted_proposal_count", 0))
                depth_stats[ret]["total_rounds"] += 1
                for d in range(1, 6):
                    if acc_cnt >= d:
                        depth_stats[ret][f"depth_{d}"] += 1

                for p in r.get("position_records", []):
                    tok = p.get("speculative_token_text", "")
                    if p.get("role") == "accepted_proposal":
                        target_token_stats[tok]["accepted"] += 1
                    else:
                        target_token_stats[tok]["fallback"] += 1

                # Preceding context
                if i > 0:
                    prev_r = rounds[i - 1]
                    prev_records = prev_r.get("position_records", [])
                    if prev_records:
                        last_tok = prev_records[-1].get("speculative_token_text", "")
                        target_prev_acc[last_tok].append(acc_cnt)

    # Save token acceptance CSV
    token_csv_path = output_dir / "token_acceptance_comparison.csv"
    tokens_to_analyze = [
        # Formulaic collocations (Cụm thành ngữ cố định)
        (" variety", "Formulaic Collocation"),
        (" objects", "Formulaic Collocation"),
        (" followed", "Formulaic Collocation"),
        (" transitions", "Formulaic Collocation"),
        (" including", "Formulaic Collocation"),
        (" series", "Formulaic Collocation"),
        # Branching connectors (Liên từ / Điểm rẽ nhánh)
        (" and", "Branching Connector"),
        (" of", "Branching Connector"),
        (",", "Punctuation Junction"),
        (" with", "Branching Connector"),
        (" then", "Branching Connector"),
        (" in", "Preposition Connector"),
        (" on", "Preposition Connector"),
        # Sentence starters / Boundaries (Mở đầu / Ngắt câu)
        ("The", "Sentence Starter (Initial)"),
        (" The", "Sentence Starter (New Sentence)"),
        (".", "Sentence Ending"),
        # Common articles / verbs (Mạo từ / Động từ chuẩn)
        (" a", "Determiner / Article"),
        (" the", "Determiner / Article"),
        (" video", "Domain Noun (Boilerplate)"),
        (" scene", "Domain Noun (Boilerplate)"),
        (" features", "Domain Verb (Boilerplate)"),
        (" shows", "Domain Verb (Boilerplate)"),
        # Visual-specific content words (Từ phụ thuộc thị giác)
        (" standing", "Visual Action"),
        (" clock", "Visual Object"),
        (" white", "Visual Attribute (Color)"),
        (" person", "Visual Object"),
        (" background", "Visual Attribute (Spatial)"),
    ]

    csv_rows = []
    for tok, category in tokens_to_analyze:
        s100 = token_stats_100[tok]
        tot100 = s100["accepted"] + s100["fallback"]
        rate100 = (s100["accepted"] / tot100 * 100) if tot100 > 0 else 0.0

        s0 = token_stats_0[tok]
        tot0 = s0["accepted"] + s0["fallback"]
        rate0 = (s0["accepted"] / tot0 * 100) if tot0 > 0 else 0.0

        delta = rate0 - rate100
        csv_rows.append(
            {
                "token_text": repr(tok),
                "category": category,
                "acc_rate_100_pct": round(rate100, 2),
                "accepted_100": s100["accepted"],
                "total_100": tot100,
                "acc_rate_0_pct": round(rate0, 2),
                "accepted_0": s0["accepted"],
                "total_0": tot0,
                "delta_pct": round(delta, 2),
            }
        )

    with token_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved token comparison CSV: {token_csv_path}")

    # Plot 1: Token Acceptance by Category (Bar Chart)
    _plot_token_category_bars(csv_rows, output_dir / "token_acceptance_by_category.png")

    # Plot 2: Acceptance by Preceding Context
    _plot_preceding_context(prev_tok_acc_100, prev_tok_acc_0, output_dir / "acceptance_by_preceding_context.png")

    # Plot 3: Depth in draft tree
    _plot_tree_depth_acceptance(depth_stats, output_dir / "acceptance_by_tree_depth.png")


def _plot_token_category_bars(csv_rows: list[dict], output_path: Path) -> None:
    # Select key representative tokens
    key_tokens = [
        " variety", " objects", " followed",  # Collocation
        " and", ",", " of", " then",          # Branching
        " The", "The", ".",                    # Sentence Starters
        " a", " the", " video",                # Determiners & domain nouns
        " standing", " clock", " white",      # Visual specific
    ]
    data = [r for r in csv_rows if eval(r["token_text"]) in key_tokens]

    tokens_display = [eval(r["token_text"]).strip() if eval(r["token_text"]).strip() else "',' (comma)" for r in data]
    # For The / The distinction:
    for idx, r in enumerate(data):
        raw = eval(r["token_text"])
        if raw == "The":
            tokens_display[idx] = "The (Cap/Start)"
        elif raw == " The":
            tokens_display[idx] = " The (Space-Cap)"
        elif raw == ",":
            tokens_display[idx] = "Comma (,)"
        elif raw == ".":
            tokens_display[idx] = "Period (.)"

    rates_100 = [r["acc_rate_100_pct"] for r in data]
    rates_0 = [r["acc_rate_0_pct"] for r in data]

    x = np.arange(len(tokens_display))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6.8), constrained_layout=True)
    bars_100 = ax.bar(x - width / 2, rates_100, width, label="100% Visual Retention", color="#2563eb", alpha=0.9)
    bars_0 = ax.bar(x + width / 2, rates_0, width, label="0% Visual Retention (No Visual)", color="#dc2626", alpha=0.9)

    ax.set_title(
        "Tỷ lệ chấp nhận theo loại Token: Khi nào Draft dễ được Accept nhất? (MSD trên VDC-50)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Tỷ lệ được Target chấp nhận (Acceptance Rate %)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tokens_display, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10.5, loc="upper right")

    # Annotate group boundaries
    ax.axvline(2.5, color="#94a3b8", linestyle=":", linewidth=1.2)
    ax.axvline(6.5, color="#94a3b8", linestyle=":", linewidth=1.2)
    ax.axvline(9.5, color="#94a3b8", linestyle=":", linewidth=1.2)
    ax.axvline(12.5, color="#94a3b8", linestyle=":", linewidth=1.2)

    ax.text(1.0, 103, "Cụm cố định (Collocation)", ha="center", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.text(4.5, 103, "Liên từ & Rẽ nhánh", ha="center", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.text(8.0, 103, "Mở đầu / Kết câu", ha="center", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.text(11.0, 103, "Mạo từ & Từ khóa", ha="center", fontsize=8.5, fontweight="bold", color="#1e293b")
    ax.text(14.0, 103, "Chi tiết thị giác", ha="center", fontsize=8.5, fontweight="bold", color="#1e293b")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def _plot_preceding_context(
    prev_100: dict[str, list[int]], prev_0: dict[str, list[int]], output_path: Path
) -> None:
    contexts = [
        (" a", "Determiner ('a')"),
        (" of", "Preposition ('of')"),
        (" with", "Preposition ('with')"),
        (" video", "Noun ('video')"),
        (" and", "Conjunction ('and')"),
        (",", "Punctuation (Comma)"),
        (" The", "Cap (' The')"),
        (".", "Period (Sentence End)"),
    ]

    labels = [desc for _, desc in contexts]
    means_100 = [float(np.mean(prev_100[tok])) if prev_100[tok] else 0.0 for tok, _ in contexts]
    means_0 = [float(np.mean(prev_0[tok])) if prev_0[tok] else 0.0 for tok, _ in contexts]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    ax.bar(x - width / 2, means_100, width, label="100% Retention (Full Visual)", color="#2563eb", alpha=0.9)
    ax.bar(x + width / 2, means_0, width, label="0% Retention (No Visual)", color="#dc2626", alpha=0.9)

    ax.set_title(
        "Hiệu năng Round tiếp theo tùy thuộc vào Ngữ cảnh đứng trước (Preceding Context)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Số token được chấp nhận ở round kế tiếp (Mean tokens)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10.5, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def _plot_tree_depth_acceptance(depth_stats: dict[int, dict], output_path: Path) -> None:
    tot_100 = depth_stats[100]["total_rounds"]
    tot_0 = depth_stats[0]["total_rounds"]

    depths = ["≥1 token", "≥2 tokens", "≥3 tokens", "≥4 tokens", "≥5 tokens"]
    rates_100 = [depth_stats[100][f"depth_{d}"] / tot_100 * 100 for d in range(1, 6)]
    rates_0 = [depth_stats[0][f"depth_{d}"] / tot_0 * 100 for d in range(1, 6)]

    x = np.arange(len(depths))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    ax.bar(x - width / 2, rates_100, width, label="100% Retention (Full Visual)", color="#2563eb", alpha=0.9)
    ax.bar(x + width / 2, rates_0, width, label="0% Retention (No Visual)", color="#dc2626", alpha=0.9)

    ax.set_title(
        "Tỷ lệ các Round chấp nhận Draft theo Chiều sâu Chuỗi đề xuất (Draft Depth)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Tỷ lệ % các round đạt độ sâu (Survival %)", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(depths, fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    analyze_token_and_context_acceptance(
        Path("results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl"),
        Path("results/6_10_26"),
    )
    print("All token acceptance analysis completed successfully!")
