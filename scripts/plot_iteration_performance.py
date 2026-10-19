#!/usr/bin/env python3
"""Compute per-draft-iteration statistics and plot performance curves for MSD on VDC."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RETENTION_ORDER = (100, 50, 25, 0)
RETENTION_LABELS = {
    100: "100% visual retention",
    50: "50% visual retention",
    25: "25% visual retention",
    0: "0% visual retention",
}
RETENTION_COLORS = {
    100: "#2563eb",  # Blue
    50: "#f59e0b",   # Amber
    25: "#16a34a",   # Green
    0: "#dc2626",    # Red
}


def load_msd_data(jsonl_path: Path) -> dict[int, list[dict]]:
    """Load and group valid MSD samples by retention percentage."""
    samples_by_ret: dict[int, list[dict]] = {ret: [] for ret in RETENTION_ORDER}
    seen_keys: set[tuple[str, int]] = set()

    with jsonl_path.open(encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in (None, "ok") or row.get("condition") != "retention":
                continue
            try:
                ret = int(float(row["retention_percentage"]))
            except (KeyError, TypeError, ValueError):
                continue
            if ret not in samples_by_ret:
                continue

            sample_id = str(row.get("sample_id", f"sample_{line_num}"))
            key = (sample_id, ret)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            samples_by_ret[ret].append(row)

    for ret, items in samples_by_ret.items():
        if len(items) != 50:
            print(f"Warning: Retention {ret}% has {len(items)} samples (expected 50)")
    return samples_by_ret


def compute_iteration_stats(samples_by_ret: dict[int, list[dict]]) -> tuple[list[dict], dict[int, dict]]:
    """Compute per-iteration acceptance and emission statistics."""
    csv_rows: list[dict] = []
    ret_trajectories: dict[int, dict] = {}

    for ret in RETENTION_ORDER:
        samples = samples_by_ret[ret]
        # For each sample, collect sequence of accepted proposals and emitted tokens
        sample_accepted_seqs: list[list[int]] = []
        sample_emitted_seqs: list[list[int]] = []

        max_rounds = 0
        for s in samples:
            rounds = s.get("acceptance_rounds") or []
            acc_list = [int(r.get("accepted_proposal_count", 0)) for r in rounds]
            emit_list = [int(r.get("emitted_token_count", a + 1)) for r, a in zip(rounds, acc_list)]
            sample_accepted_seqs.append(acc_list)
            sample_emitted_seqs.append(emit_list)
            if len(acc_list) > max_rounds:
                max_rounds = len(acc_list)

        # For each iteration index t
        iter_records = []
        for t in range(max_rounds):
            active_acc = [seq[t] for seq in sample_accepted_seqs if len(seq) > t]
            active_emit = [seq[t] for seq in sample_emitted_seqs if len(seq) > t]
            n_active = len(active_acc)
            if n_active == 0:
                continue

            mean_acc = float(np.mean(active_acc))
            std_acc = float(np.std(active_acc, ddof=1)) if n_active > 1 else 0.0
            sem_acc = std_acc / math.sqrt(n_active) if n_active > 0 else 0.0
            ci95_low = mean_acc - 1.96 * sem_acc
            ci95_high = mean_acc + 1.96 * sem_acc
            median_acc = float(np.median(active_acc))
            min_acc = int(np.min(active_acc))
            max_acc = int(np.max(active_acc))

            mean_emit = float(np.mean(active_emit))

            # Cumulative values: sum of tokens up to iteration t for samples active at t
            cum_acc_list = [sum(seq[: t + 1]) for seq in sample_accepted_seqs if len(seq) > t]
            cum_emit_list = [sum(seq[: t + 1]) for seq in sample_emitted_seqs if len(seq) > t]
            cum_mean_acc = float(np.mean(cum_acc_list))
            cum_mean_emit = float(np.mean(cum_emit_list))

            record = {
                "retention_percentage": ret,
                "draft_iteration": t,
                "n_active_samples": n_active,
                "mean_accepted_tokens": round(mean_acc, 4),
                "std_accepted_tokens": round(std_acc, 4),
                "sem_accepted_tokens": round(sem_acc, 4),
                "ci95_low": round(ci95_low, 4),
                "ci95_high": round(ci95_high, 4),
                "median_accepted_tokens": round(median_acc, 2),
                "min_accepted_tokens": min_acc,
                "max_accepted_tokens": max_acc,
                "mean_emitted_tokens": round(mean_emit, 4),
                "cumulative_mean_accepted": round(cum_mean_acc, 4),
                "cumulative_mean_emitted": round(cum_mean_emit, 4),
            }
            csv_rows.append(record)
            iter_records.append(record)

        ret_trajectories[ret] = {
            "max_rounds": max_rounds,
            "records": iter_records,
            "sample_accepted_seqs": sample_accepted_seqs,
            "sample_emitted_seqs": sample_emitted_seqs,
        }

    return csv_rows, ret_trajectories


def plot_iteration_performance(
    ret_trajectories: dict[int, dict],
    output_path: Path,
    max_focus_iter: int = 60,
) -> None:
    """Create comprehensive 4-panel plot showing iteration-by-iteration behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Panel A: Draft Acceptance Length (Iter 0 to max_focus_iter)
    ax_a = axes[0, 0]
    for ret in RETENTION_ORDER:
        records = ret_trajectories[ret]["records"]
        iters = [r["draft_iteration"] for r in records if r["draft_iteration"] <= max_focus_iter]
        means = [r["mean_accepted_tokens"] for r in records if r["draft_iteration"] <= max_focus_iter]
        sems = [r["sem_accepted_tokens"] for r in records if r["draft_iteration"] <= max_focus_iter]

        iters_np = np.array(iters)
        means_np = np.array(means)
        sems_np = np.array(sems)

        ax_a.plot(
            iters_np,
            means_np,
            color=RETENTION_COLORS[ret],
            label=RETENTION_LABELS[ret],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            alpha=0.9,
        )
        ax_a.fill_between(
            iters_np,
            means_np - sems_np,
            means_np + sems_np,
            color=RETENTION_COLORS[ret],
            alpha=0.15,
        )

    ax_a.set_title("A. Mean Accepted Proposals per Draft Iteration (Iter 0–60)", fontsize=11.5, fontweight="bold")
    ax_a.set_xlabel("Draft Iteration Index (Round)", fontsize=10)
    ax_a.set_ylabel("Accepted Draft Tokens (Mean ± SEM)", fontsize=10)
    ax_a.set_xlim(-1, max_focus_iter + 1)
    ax_a.set_ylim(-0.2, 5.2)
    ax_a.grid(True, linestyle="--", alpha=0.3)
    ax_a.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9, loc="upper right")

    # Panel B: Early Iterations Detail (Iter 0 to 20)
    ax_b = axes[0, 1]
    early_max = 20
    for ret in RETENTION_ORDER:
        records = ret_trajectories[ret]["records"]
        iters = [r["draft_iteration"] for r in records if r["draft_iteration"] <= early_max]
        means = [r["mean_accepted_tokens"] for r in records if r["draft_iteration"] <= early_max]
        sems = [r["sem_accepted_tokens"] for r in records if r["draft_iteration"] <= early_max]

        ax_b.errorbar(
            iters,
            means,
            yerr=sems,
            color=RETENTION_COLORS[ret],
            label=RETENTION_LABELS[ret],
            linewidth=2.0,
            marker="s",
            markersize=4.5,
            capsize=3,
            alpha=0.95,
        )

    ax_b.set_title("B. Early Iterations Detail & Initial Dip (Iter 0–20)", fontsize=11.5, fontweight="bold")
    ax_b.set_xlabel("Draft Iteration Index (Round)", fontsize=10)
    ax_b.set_ylabel("Accepted Draft Tokens (Mean ± SEM)", fontsize=10)
    ax_b.set_xticks(range(0, early_max + 1, 2))
    ax_b.set_xlim(-0.5, early_max + 0.5)
    ax_b.set_ylim(-0.2, 5.0)
    ax_b.grid(True, linestyle="--", alpha=0.3)
    ax_b.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9, loc="upper right")

    # Panel C: Cumulative Emitted Output Tokens
    ax_c = axes[1, 0]
    for ret in RETENTION_ORDER:
        records = ret_trajectories[ret]["records"]
        iters = [r["draft_iteration"] for r in records if r["draft_iteration"] <= 120]
        cum_emit = [r["cumulative_mean_emitted"] for r in records if r["draft_iteration"] <= 120]

        ax_c.plot(
            iters,
            cum_emit,
            color=RETENTION_COLORS[ret],
            label=RETENTION_LABELS[ret],
            linewidth=2.4,
            alpha=0.9,
        )

    ax_c.axhline(378, color="#64748b", linestyle=":", linewidth=1.5, label="Target Mean Length (~378 tok)")
    ax_c.set_title("C. Cumulative Emitted Tokens per Draft Iteration", fontsize=11.5, fontweight="bold")
    ax_c.set_xlabel("Draft Iteration Index (Round)", fontsize=10)
    ax_c.set_ylabel("Cumulative Output Tokens (Mean)", fontsize=10)
    ax_c.set_xlim(0, 120)
    ax_c.grid(True, linestyle="--", alpha=0.3)
    ax_c.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9, loc="lower right")

    # Panel D: Active Sequence Count (Survival Curve)
    ax_d = axes[1, 1]
    for ret in RETENTION_ORDER:
        records = ret_trajectories[ret]["records"]
        iters = [r["draft_iteration"] for r in records if r["draft_iteration"] <= 220]
        n_active = [r["n_active_samples"] for r in records if r["draft_iteration"] <= 220]

        ax_d.plot(
            iters,
            n_active,
            color=RETENTION_COLORS[ret],
            label=RETENTION_LABELS[ret],
            linewidth=2.2,
            alpha=0.9,
        )

    ax_d.set_title("D. Active Samples Remaining (Survival Curve)", fontsize=11.5, fontweight="bold")
    ax_d.set_xlabel("Draft Iteration Index (Round)", fontsize=10)
    ax_d.set_ylabel("Number of Active Sequences (out of 50)", fontsize=10)
    ax_d.set_xlim(0, 220)
    ax_d.set_ylim(-1, 52)
    ax_d.grid(True, linestyle="--", alpha=0.3)
    ax_d.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=9, loc="upper right")

    fig.suptitle(
        "MSD trên VDC-50 — Thống kê hiệu năng chi tiết theo từng Iteration Draft",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_single_iteration_focus(
    ret_trajectories: dict[int, dict],
    output_path: Path,
    max_iter: int = 80,
) -> None:
    """Create a single high-resolution plot focusing on acceptance rate per iteration."""
    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)

    for ret in RETENTION_ORDER:
        records = ret_trajectories[ret]["records"]
        iters = [r["draft_iteration"] for r in records if r["draft_iteration"] <= max_iter]
        means = [r["mean_accepted_tokens"] for r in records if r["draft_iteration"] <= max_iter]
        sems = [r["sem_accepted_tokens"] for r in records if r["draft_iteration"] <= max_iter]

        iters_np = np.array(iters)
        means_np = np.array(means)
        sems_np = np.array(sems)

        ax.plot(
            iters_np,
            means_np,
            color=RETENTION_COLORS[ret],
            label=RETENTION_LABELS[ret],
            linewidth=2.2,
            marker="o",
            markersize=3.8,
            alpha=0.95,
        )
        ax.fill_between(
            iters_np,
            means_np - sems_np,
            means_np + sems_np,
            color=RETENTION_COLORS[ret],
            alpha=0.15,
        )

    ax.set_title(
        "Hiệu năng Draft theo từng Iteration (MSD trên VDC-50: Retention 100%, 50%, 25%, 0%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Chỉ số Iteration Draft (Round Index $t$)", fontsize=11)
    ax.set_ylabel("Số Token Draft được Chấp nhận (Mean ± SEM)", fontsize=11)
    ax.set_xlim(-1, max_iter + 1)
    ax.set_ylim(-0.2, 5.2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-iteration performance of MSD on VDC.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl"),
        help="Path to raw MSD journal JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/6_10_26"),
        help="Directory to save CSV and plots",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data from {args.input}...")
    samples_by_ret = load_msd_data(args.input)

    print("Computing per-iteration statistics...")
    csv_rows, ret_trajectories = compute_iteration_stats(samples_by_ret)

    # Save detailed CSV
    csv_path = args.output_dir / "msd_acc_length_per_iteration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV: {csv_path} ({len(csv_rows)} rows)")

    # Save milestone summary CSV (Iter 0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60)
    milestone_iters = (0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60)
    milestone_rows = [r for r in csv_rows if r["draft_iteration"] in milestone_iters]
    milestone_csv_path = args.output_dir / "msd_iteration_milestones.csv"
    with milestone_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(milestone_rows[0].keys()))
        writer.writeheader()
        writer.writerows(milestone_rows)
    print(f"Saved milestone CSV: {milestone_csv_path} ({len(milestone_rows)} rows)")

    # Generate plots
    plot_4panel = args.output_dir / "msd_acc_length_per_iteration.png"
    plot_iteration_performance(ret_trajectories, plot_4panel)

    plot_single = args.output_dir / "msd_acc_length_detail_80_iters.png"
    plot_single_iteration_focus(ret_trajectories, plot_single, max_iter=80)

    print("Done!")


if __name__ == "__main__":
    main()
