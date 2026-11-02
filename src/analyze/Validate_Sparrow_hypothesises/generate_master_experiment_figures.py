#!/usr/bin/env python3
"""
Generate 5 Master Composite Figures grouping all 5 DFlash configurations by experimental theme:
Enhanced styling with large, readable fonts, prominent lines, and clear labels for publication-quality PDF export.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Styling configuration with significantly larger fonts for maximum PDF readability
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 17,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

DATA_DIR_4 = Path("results/visualcut_attention/vdc50_visualcut_attention_20261026T170825Z/summary_reports_data")
DATA_DIR_BAL = Path("results/visualcut_attention/vdc50_visualcut_balanced_20261029/summary_reports_data")
OUTPUT_DIR = Path("markdown/Baocao_DFlash_4Config_20261029/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("dflash_3layer_26_28_30_32_34", "High-Only [26,28,30,32,34]", "#4C78A8"),
    ("dflash_3layer_17_19_25_33_34", "Middle-High [17,19,25,33,34]", "#F58518"),
    ("dflash_3layer_17_28_30_32_34", "Middle-High 2 [17,28,30,32,34]", "#54A24B"),
    ("dflash_3layer_1_28_30_32_34", "Shallow-High [1,28,30,32,34]", "#E45756"),
    ("dflash_3layer_1_9_17_25_33", "Balanced [1,9,17,25,33]", "#9467BD"),
]

BUDGETS = [400, 3000, 13000, 25000]
BUDGET_LABELS = ["400", "3K", "13K", "25K"]

def load_data():
    with open(DATA_DIR_4 / "condition_stats.json") as f:
        stats = json.load(f)
    with open(DATA_DIR_4 / "paired_contrasts.json") as f:
        contrasts = json.load(f)
    with open(DATA_DIR_4 / "attention_profiles.json") as f:
        attn = json.load(f)

    with open(DATA_DIR_BAL / "condition_stats.json") as f:
        stats_bal = json.load(f)
    with open(DATA_DIR_BAL / "paired_contrasts.json") as f:
        contrasts_bal = json.load(f)
    with open(DATA_DIR_BAL / "attention_profiles.json") as f:
        attn_bal = json.load(f)

    stats.update(stats_bal)
    contrasts.update(contrasts_bal)
    attn.update(attn_bal)
    return stats, contrasts, attn

def plot_master_exp1(stats, contrasts):
    """Master Exp 1: Visual Cut across all 5 configs (2 rows x 5 cols)."""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10.5), sharex=True)

    cond_styles = [
        ("full", "Full", "#4C78A8", -0.25),
        ("cut", "Cut", "#54A24B", 0.0),
        ("zero", "Zero", "#E45756", 0.25),
    ]
    x = np.arange(len(BUDGETS))
    width = 0.22

    for col_idx, (cfg_key, cfg_name, cfg_color) in enumerate(CONFIGS):
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]

        # Row 1: Acceptance Rate (Full vs Cut vs Zero)
        for cond, label, color, offset in cond_styles:
            means = []
            yerr_low = []
            yerr_high = []
            for b in BUDGETS:
                s = stats[cfg_key].get(f"{b}_{cond}", {}).get("tau_effective", {})
                m = s.get("mean", 0)
                means.append(m)
                yerr_low.append(m - s.get("ci_low", m))
                yerr_high.append(s.get("ci_high", m) - m)

            ax_top.bar(
                x + offset, means, width, yerr=[yerr_low, yerr_high],
                capsize=4, label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.6
            )

        ax_top.set_title(f"A{col_idx+1}: {cfg_name}\nAcceptance Rate", fontsize=13, fontweight="bold")
        ax_top.set_ylim(2.0, 2.78)
        if col_idx == 0:
            ax_top.set_ylabel(r"Effective Acceptance $\tau_{\mathrm{effective}}$", fontsize=13.5, fontweight="bold")
            ax_top.legend(loc="upper right", framealpha=0.92, fontsize=10.5)

        # Row 2: Paired Contrasts vs Full Baseline
        contrast_styles = [
            ("cut_minus_full", "Cut - Full", "#54A24B", -0.15, "o"),
            ("zero_minus_full", "Zero - Full", "#E45756", 0.15, "s"),
        ]
        ax_bot.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

        for c_key, label, color, offset, marker in contrast_styles:
            diff_means = []
            err_low = []
            err_high = []
            for b in BUDGETS:
                c = contrasts[cfg_key].get(f"{b}_{c_key}", {}).get("tau_effective", {})
                m = c.get("mean", 0)
                diff_means.append(m)
                err_low.append(m - c.get("ci_low", m))
                err_high.append(c.get("ci_high", m) - m)

            ax_bot.errorbar(
                x + offset, diff_means, yerr=[err_low, err_high],
                fmt=marker, color=color, ecolor=color, elinewidth=2.0, capsize=5,
                markersize=8.5, label=label
            )

        ax_bot.set_title(f"B{col_idx+1}: {cfg_name.split('[')[0].strip()}\nPaired Effect vs Full", fontsize=13, fontweight="bold")
        ax_bot.set_ylim(-0.20, 0.15)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(BUDGET_LABELS, fontsize=12)
        ax_bot.set_xlabel("Visual Token Budget", fontsize=13, fontweight="bold")
        if col_idx == 0:
            ax_bot.set_ylabel(r"$\Delta \tau$ vs Full (Bootstrap 95% CI)", fontsize=13.5, fontweight="bold")
            ax_bot.legend(loc="upper left", framealpha=0.92, fontsize=10.5)

    fig.suptitle("Thực Nghiệm 1: Khảo Sát Visual Cut và Tách Biệt Giá Trị vs Vị Trí Trên 5 Cấu Hình DFlash", y=0.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "master_exp1_visual_cut_all_configs.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_master_exp2a(stats):
    """Master Exp 2A: Retention Sweep curves across 5 individual configs + direct 25K comparison (2 rows x 3 cols)."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    retention_pcts = [0, 25, 50, 75, 100]
    cond_order = ["cut", "retained_25", "retained_50", "retained_75", "full"]
    colors_b = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    markers_b = ["o", "s", "^", "D"]

    # 5 Individual Configs (Panels A to E)
    for idx, (cfg_key, cfg_name, _) in enumerate(CONFIGS):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]
        for b, color, marker in zip(BUDGETS, colors_b, markers_b):
            means = []
            lows = []
            highs = []
            for cond in cond_order:
                s = stats[cfg_key].get(f"{b}_{cond}", {}).get("tau_effective", {})
                m = s.get("mean", 0)
                means.append(m)
                lows.append(s.get("ci_low", m))
                highs.append(s.get("ci_high", m))
            ax.plot(retention_pcts, means, color=color, marker=marker, linewidth=2.6, markersize=8.5, label=f"Ngân sách {b} tokens")
            ax.fill_between(retention_pcts, lows, highs, color=color, alpha=0.15)

        ax.set_title(f"Panel {chr(65+idx)}: Cấu Hình {cfg_name}", fontsize=13.5, fontweight="bold")
        ax.set_xticks(retention_pcts)
        ax.set_xticklabels(["0% (Cut)", "25%", "50%", "75%", "100% (Full)"], fontsize=11.5)
        ax.set_ylim(2.08, 2.75)
        ax.set_xlabel("Tỷ Lệ Giữ Lại Visual Tokens (Retention %)", fontsize=12.5)
        ax.set_ylabel(r"Tỷ Lệ Chấp Nhận $\tau_{\mathrm{effective}}$", fontsize=12.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, framealpha=0.92, fontsize=10)

    # Panel F: Direct Comparison of All 5 Configs at 25K Budget
    ax_f = axes[1, 2]
    for cfg_key, cfg_name, color in CONFIGS:
        means_25k = []
        lows_25k = []
        highs_25k = []
        for cond in cond_order:
            s = stats[cfg_key].get(f"25000_{cond}", {}).get("tau_effective", {})
            m = s.get("mean", 0)
            means_25k.append(m)
            lows_25k.append(s.get("ci_low", m))
            highs_25k.append(s.get("ci_high", m))
        ax_f.plot(retention_pcts, means_25k, color=color, marker="o", linewidth=2.8, markersize=8.5, label=cfg_name.split("[")[0].strip())
        ax_f.fill_between(retention_pcts, lows_25k, highs_25k, color=color, alpha=0.12)

    ax_f.set_title("Panel F: So Sánh 5 Cấu Hình Tại Ngân Sách 25K Tokens\n(Toàn Bộ Đều Đạt Đỉnh Vượt Full Khi Giảm Token)", fontsize=13, fontweight="bold")
    ax_f.set_xticks(retention_pcts)
    ax_f.set_xticklabels(["0% (Cut)", "25%", "50%", "75%", "100% (Full)"], fontsize=11.5)
    ax_f.set_xlabel("Tỷ Lệ Giữ Lại Visual Tokens (Retention %)", fontsize=12.5)
    ax_f.set_ylabel(r"Tỷ Lệ Chấp Nhận $\tau_{\mathrm{effective}}$ (25K Budget)", fontsize=12.5)
    ax_f.set_ylim(2.14, 2.36)
    ax_f.legend(loc="lower left", framealpha=0.92, fontsize=10)

    fig.suptitle("Thực Nghiệm 2A: Đường Cong Visual Retention Sweep Trên 5 Cấu Hình DFlash (400 – 25K Tokens)", y=0.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "master_exp2a_retention_curves_by_config.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_master_exp2b(stats, contrasts):
    """Master Exp 2B: Sweet Spot Comprehensive Analysis across all budgets & configs (2 rows x 2 cols)."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    retention_pcts = [0, 25, 50, 75, 100]
    cond_order = ["cut", "retained_25", "retained_50", "retained_75", "full"]
    colors_b = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    markers_b = ["o", "s", "^", "D"]

    # Sub-panel A: Shift of retention curve across ALL 4 budgets (Mean of 5 configs)
    ax_a = axes[0, 0]
    for b, color, marker in zip(BUDGETS, colors_b, markers_b):
        all_means = []
        for cond in cond_order:
            cfg_m = [stats[c[0]].get(f"{b}_{cond}", {}).get("tau_effective", {}).get("mean", 0) for c in CONFIGS]
            all_means.append(np.mean(cfg_m))
        ax_a.plot(retention_pcts, all_means, color=color, marker=marker, linewidth=2.8, markersize=9.5, label=f"Ngân sách {b} tokens")

        # Highlight sweet spot
        best_idx = int(np.argmax(all_means))
        ax_a.plot(retention_pcts[best_idx], all_means[best_idx], marker="*", markersize=16, color="gold", markeredgecolor="black", markeredgewidth=1.2)

    ax_a.set_title("Panel A: Sự Dịch Chuyển Sweet Spot Theo Chiều Dài Ngân Sách\n(Trung Bình 5 Cấu Hình DFlash, Ngôi Sao Vàng: Đỉnh Tối Ưu)", fontsize=13, fontweight="bold")
    ax_a.set_xticks(retention_pcts)
    ax_a.set_xticklabels(["0% (Cut)", "25%", "50%", "75%", "100% (Full)"], fontsize=12)
    ax_a.set_xlabel("Tỷ Lệ Giữ Lại Visual Tokens (Retention %)", fontsize=13)
    ax_a.set_ylabel(r"Acceptance Rate $\tau_{\mathrm{effective}}$", fontsize=13)
    ax_a.set_ylim(2.15, 2.65)
    ax_a.legend(loc="upper right", framealpha=0.92, fontsize=11)
    ax_a.annotate("Video dài (13K, 25K):\nĐỉnh tối ưu dịch về 25% - 50%", xy=(35, 2.29), xytext=(40, 2.38),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.4, headwidth=7),
                 fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="gray", alpha=0.9))

    # Sub-panel B: Delta Gain Ret25 vs Full across ALL 4 budgets (Grouped Bar Chart with 95% CI for 5 configs)
    ax_b = axes[0, 1]
    x_b = np.arange(len(BUDGETS))
    w_bar = 0.15

    for i_cfg, (cfg_key, cfg_name, color) in enumerate(CONFIGS):
        offsets = (i_cfg - 2.0) * w_bar
        diff_means = []
        err_low = []
        err_high = []
        for b in BUDGETS:
            c = contrasts[cfg_key].get(f"{b}_retained_25_minus_full", {}).get("tau_effective", {})
            m = c.get("mean", 0)
            diff_means.append(m)
            err_low.append(m - c.get("ci_low", m))
            err_high.append(c.get("ci_high", m) - m)
        ax_b.bar(x_b + offsets, diff_means, w_bar, yerr=[err_low, err_high],
                 capsize=4.0, label=cfg_name.split("[")[0].strip(), color=color, alpha=0.9, edgecolor="black", linewidth=0.6)

    ax_b.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax_b.axhspan(0, 0.16, color="green", alpha=0.04)
    ax_b.set_title("Panel B: Mức Tăng Acceptance Khi Dùng Retained 25%\n($\\Delta \\tau = \\mathrm{Ret25} - \\mathrm{Full}$ trên Cả 5 Cấu Hình, 95% CI)", fontsize=13, fontweight="bold")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(BUDGET_LABELS, fontsize=12)
    ax_b.set_xlabel("Ngân Sách Visual Tokens", fontsize=13)
    ax_b.set_ylabel(r"Delta Gain $\Delta \tau$ (Bootstrap 95% CI)", fontsize=13)
    ax_b.set_ylim(-0.08, 0.15)
    ax_b.legend(loc="upper left", framealpha=0.92, fontsize=10)

    # Sub-panel C: Comprehensive Retention Spectrum vs Full across Budgets (Mean 5 configs)
    ax_c = axes[1, 0]
    cond_contrasts = [
        ("retained_75_minus_full", "Retained 75% - Full", "#4C78A8", "o"),
        ("retained_50_minus_full", "Retained 50% - Full", "#72B7B2", "s"),
        ("retained_25_minus_full", "Retained 25% - Full", "#F58518", "^"),
        ("cut_minus_full", "Cut (0%) - Full", "#E45756", "v"),
    ]
    x_c = np.arange(len(BUDGETS))

    for c_key, c_label, c_color, c_marker in cond_contrasts:
        mean_curve = []
        for b in BUDGETS:
            vals = [contrasts[c[0]].get(f"{b}_{c_key}", {}).get("tau_effective", {}).get("mean", 0) for c in CONFIGS]
            mean_curve.append(np.mean(vals))
        ax_c.plot(x_c, mean_curve, color=c_color, marker=c_marker, linewidth=2.6, markersize=9, label=c_label)

    ax_c.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax_c.set_title("Panel C: Phổ So Sánh Các Mức Retention vs Full\n(Trung Bình 5 Cấu Hình Theo Ngân Sách)", fontsize=13, fontweight="bold")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(BUDGET_LABELS, fontsize=12)
    ax_c.set_xlabel("Ngân Sách Visual Tokens", fontsize=13)
    ax_c.set_ylabel(r"Delta Acceptance $\Delta \tau$ so với Full", fontsize=13)
    ax_c.set_ylim(-0.07, 0.10)
    ax_c.legend(loc="lower left", framealpha=0.92, fontsize=10.5)

    # Sub-panel D: Pareto Trade-off: KV Cache Savings vs Acceptance Rate at 25K (All 5 configs)
    ax_d = axes[1, 1]
    savings_pct = [0, 25, 50, 75, 100]
    rev_cond_order = ["full", "retained_75", "retained_50", "retained_25", "cut"]

    for cfg_key, cfg_name, color in CONFIGS:
        tau_25k = [stats[cfg_key].get(f"25000_{cond}", {}).get("tau_effective", {}).get("mean", 0) for cond in rev_cond_order]
        ax_d.plot(savings_pct, tau_25k, color=color, marker="o", linewidth=2.6, markersize=8.5, label=cfg_name.split("[")[0].strip())

    ax_d.set_title("Panel D: Bản Đồ Đánh Đổi Pareto Tại Ngân Sách 25K Tokens\n(Tiết Kiệm KV Cache & FLOPs vs Acceptance Rate $\\tau$ Cho Cả 5 Cấu Hình)", fontsize=13, fontweight="bold")
    ax_d.set_xticks(savings_pct)
    ax_d.set_xticklabels(["0%\n(Full)", "25%\n(Ret75)", "50%\n(Ret50)", "75%\n(Ret25)", "100%\n(Cut)"], fontsize=11.5)
    ax_d.set_xlabel("Tỷ Lệ Cắt Giảm KV Cache & FLOPs (%)", fontsize=13)
    ax_d.set_ylabel(r"Acceptance Rate $\tau_{\mathrm{effective}}$ (25K Budget)", fontsize=13)
    ax_d.set_ylim(2.15, 2.35)
    ax_d.axvline(75, color="#F58518", linestyle=":", linewidth=1.8, alpha=0.7)
    ax_d.annotate("Điểm Vàng Pareto (Ret25% - Ret50%):\nTiết kiệm 50% - 75% KV Cache\nVÀ Acceptance cao hơn Full", xy=(75, 2.29), xytext=(25, 2.32),
                 arrowprops=dict(facecolor='#F58518', shrink=0.08, width=1.5, headwidth=7),
                 fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#F58518", alpha=0.9))
    ax_d.legend(loc="lower left", framealpha=0.92, fontsize=10)

    fig.suptitle("Thực Nghiệm 2B: Phân Tích Chuyên Sâu Điểm Tối Ưu Sweet Spot & Động Lực Học Trade-Off Toàn Diện (5 Cấu Hình)", y=0.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "master_exp2b_sweetspot_comprehensive_analysis.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_master_exp3(attn):
    """Master Exp 3: Attention Dynamics across 5 configs (3 rows x 5 cols)."""
    fig, axes = plt.subplots(3, 5, figsize=(26, 14), sharex="col")
    x = np.arange(len(BUDGETS))
    w_stack = 0.55
    w_sub = 0.25

    for col_idx, (cfg_key, cfg_name, _) in enumerate(CONFIGS):
        ax_top = axes[0, col_idx]
        ax_mid = axes[1, col_idx]
        ax_bot = axes[2, col_idx]

        vis_mass = []
        ptext_mass = []
        gen_mass = []
        curr_mass = []

        vis_dens = []
        q_dens = []
        gen_dens = []

        l0 = []
        l1 = []
        l2 = []

        for b in BUDGETS:
            f = attn[cfg_key].get(str(b), {}).get("full", {})
            vis_mass.append(f.get("vis_mass_mean", 0))
            ptext_mass.append(f.get("prompt_text_mass_mean", 0))
            gen_mass.append(f.get("gen_text_mass_mean", 0))
            curr_mass.append(f.get("curr_query_mass_mean", 0))

            vis_dens.append(f.get("vis_dens_mean", 0))
            q_dens.append(f.get("q_dens_mean", 0))
            gen_dens.append(f.get("gen_dens_mean", 0))

            layers = f.get("layer_vis_mass", {})
            l0.append(layers.get("layer_0", 0))
            l1.append(layers.get("layer_1", 0))
            l2.append(layers.get("layer_2", 0))

        vm = np.array(vis_mass)
        pm = np.array(ptext_mass)
        gm = np.array(gen_mass)
        cm = np.array(curr_mass)

        # Row 1: 100% Mass Stacked Bars
        ax_top.bar(x, vm, w_stack, label="Visual Tokens", color="#4C78A8", edgecolor="black", linewidth=0.6)
        ax_top.bar(x, pm, w_stack, bottom=vm, label="Prompt Question", color="#F58518", edgecolor="black", linewidth=0.6)
        ax_top.bar(x, gm, w_stack, bottom=vm+pm, label="Generated Text", color="#54A24B", edgecolor="black", linewidth=0.6)
        ax_top.bar(x, cm, w_stack, bottom=vm+pm+gm, label="Current Block", color="#B279A2", edgecolor="black", linewidth=0.6)
        ax_top.set_title(f"A{col_idx+1}: {cfg_name}\n100% Attention Mass", fontsize=12.5, fontweight="bold")
        ax_top.set_ylim(0, 1.35)
        if col_idx == 0:
            ax_top.set_ylabel("Attention Share (Sums to 1.0)", fontsize=13, fontweight="bold")
            ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, framealpha=0.95, fontsize=9.5, columnspacing=0.8)

        # Row 2: Per-Token Density (Log-scale)
        ax_mid.bar(x - w_sub, gen_dens, w_sub, label="Generated Text", color="#54A24B", edgecolor="black", linewidth=0.6)
        ax_mid.bar(x, q_dens, w_sub, label="Prompt Question", color="#F58518", edgecolor="black", linewidth=0.6)
        ax_mid.bar(x + w_sub, vis_dens, w_sub, label="Visual Tokens", color="#4C78A8", edgecolor="black", linewidth=0.6)
        ax_mid.set_yscale("log")
        ax_mid.set_ylim(bottom=5e-6, top=3.0)
        ax_mid.set_title(f"B{col_idx+1}: {cfg_name.split('[')[0].strip()}\nPer-Token Density (Log)", fontsize=12.5, fontweight="bold")
        if col_idx == 0:
            ax_mid.set_ylabel("Density per Token (Log)", fontsize=13, fontweight="bold")
            ax_mid.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

        # Row 3: Layer-wise Visual Mass
        ax_bot.plot(x, l0, marker="^", color="#72B7B2", linewidth=2.6, markersize=8, label="Layer 0 (Input)")
        ax_bot.plot(x, l1, marker="o", color="#E45756", linewidth=3.0, markersize=8.5, label="Layer 1 (Middle)")
        ax_bot.plot(x, l2, marker="s", color="#54A24B", linewidth=2.6, markersize=8, label="Layer 2 (Output)")
        ax_bot.set_title(f"C{col_idx+1}: {cfg_name.split('[')[0].strip()}\nLayer Visual Mass", fontsize=12.5, fontweight="bold")
        ax_bot.set_ylim(0, 1.05)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(BUDGET_LABELS, fontsize=12)
        ax_bot.set_xlabel("Visual Token Budget", fontsize=13, fontweight="bold")
        if col_idx == 0:
            ax_bot.set_ylabel("Visual Attention Mass", fontsize=13, fontweight="bold")
            ax_bot.legend(loc="upper left", framealpha=0.95, fontsize=10)

    fig.suptitle("Thực Nghiệm 3: Phân Bổ 100% Attention Mass, Mật Độ Chú Ý và Cấu Trúc Tầng Draft Trên 5 Cấu Hình", y=0.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "master_exp3_attention_dynamics_all_configs.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_master_exp4(stats):
    """Master Exp 4: Quality Invariance & Pareto Frontier Map for 5 configs (2 rows x 2 cols)."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))

    # Panel A: Pareto Frontier (TPS vs tau)
    ax_a = axes[0, 0]
    for cfg_key, cfg_name, color in CONFIGS:
        taus = [stats[cfg_key].get(f"{b}_full", {}).get("tau_effective", {}).get("mean", 0) for b in BUDGETS]
        tpss = [stats[cfg_key].get(f"{b}_full", {}).get("speculative_tps", {}).get("mean", 0) for b in BUDGETS]
        ax_a.plot(taus, tpss, marker="o", linestyle="--", linewidth=2.6, color=color, label=cfg_name, markersize=9)
    ax_a.set_title("Panel A: Bản Đồ Pareto (Tốc Độ TPS vs Acceptance Rate)", fontsize=13.5, fontweight="bold")
    ax_a.set_xlabel(r"Effective Acceptance Rate $\tau_{\mathrm{effective}}$", fontsize=13)
    ax_a.set_ylabel("Speculative Generation Speed (Tokens/Second)", fontsize=13)
    ax_a.legend(loc="upper left", framealpha=0.92, fontsize=10)
    ax_a.annotate("400 Tokens (~115 TPS)", xy=(2.52, 115), xytext=(2.38, 120),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=6), fontsize=10.5, fontweight="bold")
    ax_a.annotate("25K Tokens (~13 TPS)", xy=(2.21, 13), xytext=(2.32, 22),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.2, headwidth=6), fontsize=10.5, fontweight="bold")

    # Panel B: Lossless Rate across budgets (5 configs)
    ax_b = axes[0, 1]
    x = np.arange(len(BUDGETS))
    w = 0.16
    for idx, (cfg_key, cfg_name, color) in enumerate(CONFIGS):
        lossless = [stats[cfg_key].get(f"{b}_full", {}).get("lossless_rate", 0) * 100 for b in BUDGETS]
        ax_b.bar(x + (idx - 2.0)*w, lossless, w, label=cfg_name.split("[")[0].strip(), color=color, edgecolor="black", linewidth=0.6)
    ax_b.set_title("Panel B: Tỷ Lệ Lossless Hoàn Toàn Theo Ngân Sách Visual", fontsize=13.5, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(BUDGET_LABELS, fontsize=12)
    ax_b.set_xlabel("Visual Token Budget", fontsize=13)
    ax_b.set_ylabel("Lossless Rate (%)", fontsize=13)
    ax_b.set_ylim(0, 28)
    ax_b.legend(loc="upper right", framealpha=0.92, fontsize=10)

    # Panel C: ROUGE-L Consistency across Conditions at 3k (5 configs)
    ax_c = axes[1, 0]
    conditions = ["full", "retained_75", "retained_50", "retained_25", "cut"]
    cond_labels = ["Full", "Ret75", "Ret50", "Ret25", "Cut"]
    x_c = np.arange(len(conditions))
    for idx, (cfg_key, cfg_name, color) in enumerate(CONFIGS):
        rouges = [stats[cfg_key].get(f"3000_{c}", {}).get("spec_rouge_l", {}).get("mean", 0) for c in conditions]
        ax_c.plot(x_c, rouges, marker="s", linewidth=2.4, color=color, label=cfg_name.split("[")[0].strip(), markersize=8)

    tgt_rouge = np.mean([stats[c[0]].get("3000_full", {}).get("tgt_rouge_l", {}).get("mean", 0) for c in CONFIGS])
    ax_c.axhline(tgt_rouge, color="black", linestyle="--", linewidth=1.8, label=f"Target Baseline ({tgt_rouge:.3f})")
    ax_c.set_title("Panel C: Tính Nhất Quán ROUGE-L Qua Các Điều Kiện (3K Tokens)", fontsize=13.5, fontweight="bold")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(cond_labels, fontsize=12)
    ax_c.set_xlabel("Visual Condition in Draft", fontsize=13)
    ax_c.set_ylabel("ROUGE-L Score", fontsize=13)
    ax_c.set_ylim(0.17, 0.22)
    ax_c.legend(loc="lower left", framealpha=0.92, fontsize=10)

    # Panel D: Content Coverage Consistency across Conditions at 3k (5 configs)
    ax_d = axes[1, 1]
    for idx, (cfg_key, cfg_name, color) in enumerate(CONFIGS):
        covs = [stats[cfg_key].get(f"3000_{c}", {}).get("spec_coverage", {}).get("mean", 0) for c in conditions]
        ax_d.plot(x_c, covs, marker="^", linewidth=2.4, color=color, label=cfg_name.split("[")[0].strip(), markersize=8)

    tgt_cov = np.mean([stats[c[0]].get("3000_full", {}).get("tgt_coverage", {}).get("mean", 0) for c in CONFIGS])
    ax_d.axhline(tgt_cov, color="black", linestyle="--", linewidth=1.8, label=f"Target Baseline ({tgt_cov:.3f})")
    ax_d.set_title("Panel D: Tính Nhất Quán Content Coverage Qua Các Điều Kiện (3K Tokens)", fontsize=13.5, fontweight="bold")
    ax_d.set_xticks(x_c)
    ax_d.set_xticklabels(cond_labels, fontsize=12)
    ax_d.set_xlabel("Visual Condition in Draft", fontsize=13)
    ax_d.set_ylabel("Coverage Score", fontsize=13)
    ax_d.set_ylim(0.28, 0.36)
    ax_d.legend(loc="lower left", framealpha=0.92, fontsize=10)

    fig.suptitle("Thực Nghiệm 4: Tính Bất Biến Ngữ Nghĩa (Lossless Quality) & Bản Đồ Đánh Đổi Tốc Độ Pareto (5 Cấu Hình)", y=0.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUTPUT_DIR / "master_exp4_quality_tradeoff_all_configs.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    stats, contrasts, attn = load_data()
    print("Generating Master Experiment Figures for 5 configs (High-Legibility edition)...")
    plot_master_exp1(stats, contrasts)
    plot_master_exp2a(stats)
    plot_master_exp2b(stats, contrasts)
    plot_master_exp3(attn)
    plot_master_exp4(stats)
    print("All Master Experiment Figures generated successfully!")

if __name__ == "__main__":
    main()
