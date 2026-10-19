#!/usr/bin/env python3
"""
Update and summarize Depth 3 vs Depth 5 benchmark results across 3 datasets:
- VidHalluc (binary QA)
- VideoChair (binary QA)
- VideoDetailCaption / VDC (detailed captioning)
Computes macro metrics, generates 5 publication-ready figures, and outputs summary tables.
"""

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_rouge_l(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 1.0 if ref_tokens == cand_tokens else 0.0
    prev = [0] * (len(cand_tokens) + 1)
    for r in ref_tokens:
        curr = [0]
        for idx, c in enumerate(cand_tokens, 1):
            curr.append(prev[idx - 1] + 1 if r == c else max(prev[idx], curr[-1]))
        prev = curr
    score = prev[-1]
    p = score / len(cand_tokens) if cand_tokens else 0
    r = score / len(ref_tokens) if ref_tokens else 0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_bleu1(reference: str, candidate: str) -> float:
    ref_t = reference.lower().split()
    cand_t = candidate.lower().split()
    if not cand_t or not ref_t:
        return 0.0
    c_counts = Counter(cand_t)
    r_counts = Counter(ref_t)
    matches = sum(min(count, r_counts.get(w, 0)) for w, count in c_counts.items())
    return matches / len(cand_t)


def compute_binary_acc(pred: str, gt_ans: str) -> float:
    p = pred.strip().lower()
    g = gt_ans.strip().lower()
    if g in ["yes", "no"]:
        return 1.0 if p.startswith(g) else 0.0
    return 0.0


def main():
    repo_root = Path("/path/to/local-resource")
    base_eval = repo_root / "results/msd_eval_qwen25vl3b"
    out_dir = repo_root / "results/6_10_26"
    fig_dir = out_dir / "figures"
    artifact_dir = Path("/path/to/local-resource")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Ground Truths
    gt_map = {}
    gt_files = [
        ("vidhalluc", repo_root / "dataset/VidHalluc/test.jsonl"),
        ("videochair", repo_root / "dataset/VideoChair/test.jsonl"),
        ("vdc", repo_root / "dataset/VideoDetailCaption/test_100.jsonl"),
    ]
    for ds_name, p in gt_files:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                sid = d.get("source_id", d.get("video_name", ""))
                gt_map[sid] = d

    # 2. Parse all jsonl outputs
    records = []
    datasets = ["vidhalluc", "videochair", "vdc"]
    depths = ["depth3", "depth5"]
    retentions = [100.0, 50.0, 25.0, 0.0]

    for depth in depths:
        for ds in datasets:
            jpath = base_eval / depth / ds / f"msd_retention_{ds}.jsonl"
            if not jpath.exists():
                print(f"Warning: {jpath} not found.")
                continue
            with open(jpath, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            df = pd.DataFrame(rows)

            for ret in retentions:
                sub = df[df["retention_percentage"] == ret]
                if len(sub) == 0:
                    continue

                tau = sub["accepted_prefix_tokens"].mean()
                dec_sp = sub["decode_speedup"].mean()
                e2e_sp = sub["end_to_end_speedup"].mean()
                lossless_rate = sub["lossless"].mean() * 100.0

                # Compute semantic metrics
                target_rouges = []
                target_bleus = []
                gt_metrics = []

                for _, r in sub.iterrows():
                    spec_txt = r.get("speculative_output_text", "")
                    tgt_txt = r.get("target_output_text", "")
                    sid = r.get("sample_id", "")

                    target_rouges.append(compute_rouge_l(tgt_txt, spec_txt))
                    target_bleus.append(compute_bleu1(tgt_txt, spec_txt))

                    # GT evaluation
                    gt_info = gt_map.get(sid, {})
                    gt_ans = gt_info.get("answer", "")
                    if ds in ["vidhalluc", "videochair"]:
                        gt_metrics.append(compute_binary_acc(spec_txt, gt_ans))
                    else:  # vdc
                        gt_metrics.append(compute_rouge_l(gt_ans, spec_txt))

                records.append({
                    "depth": depth,
                    "dataset": ds,
                    "retention_percentage": ret,
                    "count": len(sub),
                    "tau": tau,
                    "speedup_decode": dec_sp,
                    "speedup_e2e": e2e_sp,
                    "lossless_exact_rate": lossless_rate,
                    "rouge_l_to_target": np.mean(target_rouges) if target_rouges else 1.0,
                    "bleu1_to_target": np.mean(target_bleus) if target_bleus else 1.0,
                    "task_gt_performance": np.mean(gt_metrics) if gt_metrics else 0.0,
                })

    summary_df = pd.DataFrame(records)
    csv_path = out_dir / "macro_metrics_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved macro summary to {csv_path}")
    print(summary_df.to_string())

    # 3. Plotting Setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11

    # Figure 1: Mean Acceptance Length (tau) - Depth 3 vs Depth 5 (across 100% retention)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(datasets))
    width = 0.35
    d3_tau = [summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]['tau'].values[0] if len(summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]) > 0 else 0 for d in datasets]
    d5_tau = [summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]['tau'].values[0] if len(summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]) > 0 else 0 for d in datasets]

    rects1 = ax.bar(x - width/2, d3_tau, width, label='MSD Depth 3', color='#e74c3c', edgecolor='black')
    rects2 = ax.bar(x + width/2, d5_tau, width, label='MSD Depth 5', color='#2ecc71', edgecolor='black')

    ax.set_ylabel('Mean Accepted Tokens / Round (tau)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 1: Mean Acceptance Length (tau) - Depth 3 vs Depth 5 (100% Visual)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['VidHalluc', 'VideoChair', 'VDC'], fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, max(h, 0.01)),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold', color='#c0392b' if h < 0.05 else 'black')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fig1_path = fig_dir / "fig1_acceptance_length_comparison.png"
    plt.savefig(fig1_path)
    plt.close()

    # Figure 2: Decode Speedup Factor Comparison
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    d3_sp = [summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]['speedup_decode'].values[0] if len(summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]) > 0 else 0 for d in datasets]
    d5_sp = [summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]['speedup_decode'].values[0] if len(summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]) > 0 else 0 for d in datasets]

    rects1 = ax.bar(x - width/2, d3_sp, width, label='MSD Depth 3', color='#e67e22', edgecolor='black')
    rects2 = ax.bar(x + width/2, d5_sp, width, label='MSD Depth 5', color='#3498db', edgecolor='black')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='AR Baseline (1.0x)')

    ax.set_ylabel('Decode Speedup Factor (x Baseline)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Decode Speedup Factor Comparison - Depth 3 vs Depth 5 (100% Visual)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['VidHalluc', 'VideoChair', 'VDC'], fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}x', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}x', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fig2_path = fig_dir / "fig2_decode_speedup_comparison.png"
    plt.savefig(fig2_path)
    plt.close()

    # Figure 3: Semantic Lossless Metrics (ROUGE-L & BLEU-1 to Target)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    d3_rouge = [summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]['rouge_l_to_target'].values[0] if len(summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]) > 0 else 0 for d in datasets]
    d5_rouge = [summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]['rouge_l_to_target'].values[0] if len(summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == d) & (summary_df['retention_percentage'] == 100.0)]) > 0 else 0 for d in datasets]

    rects1 = ax.bar(x - width/2, d3_rouge, width, label='MSD Depth 3 (ROUGE-L to Target)', color='#9b59b6', edgecolor='black')
    rects2 = ax.bar(x + width/2, d5_rouge, width, label='MSD Depth 5 (ROUGE-L to Target)', color='#1abc9c', edgecolor='black')
    ax.axhline(1.0, color='gray', linestyle=':', label='Theoretical Lossless (1.000)')

    ax.set_ylabel('ROUGE-L Score relative to Target Output', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Semantic Lossless Preservation vs Target Model', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['VidHalluc', 'VideoChair', 'VDC'], fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fig3_path = fig_dir / "fig3_semantic_lossless_metrics.png"
    plt.savefig(fig3_path)
    plt.close()

    # Figure 4: Visual Retention Sensitivity Curve across Cuts (100% -> 50% -> 25% -> 0%)
    fig, (ax_tau, ax_sp) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    ret_levels = [100.0, 50.0, 25.0, 0.0]
    ret_labels = ['100%', '50%', '25%', '0%']
    markers = {'vidhalluc': 'o', 'videochair': 's', 'vdc': '^'}
    colors_d5 = {'vidhalluc': '#2ecc71', 'videochair': '#27ae60', 'vdc': '#16a085'}
    colors_d3 = {'vidhalluc': '#e74c3c', 'videochair': '#c0392b', 'vdc': '#d35400'}

    for ds in datasets:
        # Depth 5
        d5_vals = [summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == ds) & (summary_df['retention_percentage'] == r)]['tau'].values[0] for r in ret_levels]
        ax_tau.plot(range(4), d5_vals, marker=markers[ds], color=colors_d5[ds], label=f'D5 - {ds.upper()}', linewidth=2)
        # Depth 3
        d3_vals = [summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == ds) & (summary_df['retention_percentage'] == r)]['tau'].values[0] for r in ret_levels]
        ax_tau.plot(range(4), d3_vals, marker=markers[ds], color=colors_d3[ds], linestyle='--', label=f'D3 - {ds.upper()}', linewidth=1.8)

    ax_tau.set_xticks(range(4))
    ax_tau.set_xticklabels(ret_labels, fontweight='bold')
    ax_tau.set_xlabel('Draft Visual Retention Level', fontweight='bold')
    ax_tau.set_ylabel('Mean Acceptance Length (tau)', fontweight='bold')
    ax_tau.set_title('(A) Acceptance Length Sensitivity to Visual Retention', fontweight='bold')
    ax_tau.legend(frameon=True, fontsize=9.5)
    ax_tau.grid(True, linestyle='--', alpha=0.6)

    for ds in datasets:
        d5_sps = [summary_df[(summary_df['depth'] == 'depth5') & (summary_df['dataset'] == ds) & (summary_df['retention_percentage'] == r)]['speedup_decode'].values[0] for r in ret_levels]
        ax_sp.plot(range(4), d5_sps, marker=markers[ds], color=colors_d5[ds], label=f'D5 - {ds.upper()}', linewidth=2)
        d3_sps = [summary_df[(summary_df['depth'] == 'depth3') & (summary_df['dataset'] == ds) & (summary_df['retention_percentage'] == r)]['speedup_decode'].values[0] for r in ret_levels]
        ax_sp.plot(range(4), d3_sps, marker=markers[ds], color=colors_d3[ds], linestyle='--', label=f'D3 - {ds.upper()}', linewidth=1.8)

    ax_sp.set_xticks(range(4))
    ax_sp.set_xticklabels(ret_labels, fontweight='bold')
    ax_sp.set_xlabel('Draft Visual Retention Level', fontweight='bold')
    ax_sp.set_ylabel('Decode Speedup Factor (x AR)', fontweight='bold')
    ax_sp.set_title('(B) Decode Speedup Sensitivity to Visual Retention', fontweight='bold')
    ax_sp.legend(frameon=True, fontsize=9.5)
    ax_sp.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Figure 4: Impact of Visual Token Retention Pruning on Drafter Behavior', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig4_path = fig_dir / "fig4_retention_cut_invariance.png"
    plt.savefig(fig4_path, bbox_inches='tight')
    plt.close()

    # Figure 5: Iteration Difficulty Breakdown in Depth 5
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    diff_data = []
    for ds in datasets:
        cpath = base_eval / "depth5" / ds / "draft_iterations_detailed.csv"
        if cpath.exists():
            df_iter = pd.read_csv(cpath)
            counts = df_iter["difficulty_category"].value_counts(normalize=True)
            diff_data.append({
                "dataset": ds.upper(),
                "easy": counts.get("easy", 0.0) * 100,
                "medium": counts.get("medium", 0.0) * 100,
                "hard": counts.get("hard", 0.0) * 100,
            })
    if diff_data:
        df_diff = pd.DataFrame(diff_data)
        bar_w = 0.5
        x_idx = np.arange(len(df_diff))
        p_easy = ax.bar(x_idx, df_diff["easy"], bar_w, label='Easy (τ ≥ 1)', color='#2ecc71', edgecolor='black')
        p_med = ax.bar(x_idx, df_diff["medium"], bar_w, bottom=df_diff["easy"], label='Medium (0 < τ < 1)', color='#f1c40f', edgecolor='black')
        p_hard = ax.bar(x_idx, df_diff["hard"], bar_w, bottom=df_diff["easy"] + df_diff["medium"], label='Hard (τ = 0)', color='#e74c3c', edgecolor='black')

        ax.set_ylabel('Tỷ lệ vòng lặp (%)', fontsize=12, fontweight='bold')
        ax.set_title('Figure 5: Phân bổ Độ khó Vòng lặp Drafter trên MSD Depth 5', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(df_diff["dataset"], fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for i in range(len(df_diff)):
            ey = df_diff["easy"].iloc[i]
            my = df_diff["medium"].iloc[i]
            hy = df_diff["hard"].iloc[i]
            if ey > 5:
                ax.text(i, ey / 2, f"{ey:.1f}%", ha='center', va='center', fontweight='bold', color='white')
            if my > 5:
                ax.text(i, ey + my / 2, f"{my:.1f}%", ha='center', va='center', fontweight='bold', color='black')
            if hy > 5:
                ax.text(i, ey + my + hy / 2, f"{hy:.1f}%", ha='center', va='center', fontweight='bold', color='white')
        plt.tight_layout()
        fig5_path = fig_dir / "fig5_iteration_difficulty_depth5.png"
        plt.savefig(fig5_path, bbox_inches='tight')
        plt.close()

    # 4. Copy all figures to results/6_10_26 and artifact directory
    for f in fig_dir.glob("*.png"):
        shutil.copy(f, out_dir / f.name)
        shutil.copy(f, artifact_dir / f.name)
        (artifact_dir / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy(f, artifact_dir / "figures" / f.name)
    print("All figures generated and copied successfully!")


if __name__ == "__main__":
    main()
