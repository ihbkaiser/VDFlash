"""Generate publication-quality figures for DFlash 4-config visualcut, retention, and attention reports."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16.5,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

DATA_DIR = Path("results/visualcut_attention/vdc50_visualcut_attention_20261026T170825Z/summary_reports_data")
OUTPUT_DIR = Path("markdown/Baocao_DFlash_4Config_20261029/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("dflash_3layer_26_28_30_32_34", "High-Only (26-34)", "#4C78A8"),
    ("dflash_3layer_17_19_25_33_34", "Middle-High (17-34)", "#F58518"),
    ("dflash_3layer_17_28_30_32_34", "Middle-High 2 (17, 28-34)", "#54A24B"),
    ("dflash_3layer_1_28_30_32_34", "Shallow-High (1, 28-34)", "#E45756"),
]
BUDGETS = [400, 3000, 13000, 25000]
BUDGET_LABELS = ["400", "3K", "13K", "25K"]

def load_data():
    with open(DATA_DIR / "condition_stats.json") as f:
        stats = json.load(f)
    with open(DATA_DIR / "paired_contrasts.json") as f:
        contrasts = json.load(f)
    with open(DATA_DIR / "attention_profiles.json") as f:
        attn = json.load(f)
    return stats, contrasts, attn

def plot_config_visual_cut(cfg_key, cfg_name, stats, contrasts):
    """Plot Fig 1: Visual Cut comparison and Paired Contrasts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(BUDGETS))
    width = 0.25

    cond_styles = [
        ("full", "Full (100% Visual)", "#4C78A8", -width),
        ("cut", "Cut (0% Visual)", "#54A24B", 0),
        ("zero", "Zero (Value Ablation)", "#E45756", width),
    ]

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

        ax1.bar(
            x + offset, means, width, yerr=[yerr_low, yerr_high],
            capsize=4, label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.5
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(BUDGET_LABELS)
    ax1.set_xlabel("Target Visual Token Budget")
    ax1.set_ylabel(r"Effective Acceptance Rate $\tau_{\mathrm{effective}}$")
    ax1.set_title(f"A: Acceptance Rate Across Visual Budgets\n({cfg_name})")
    ax1.set_ylim(2.0, 2.78)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.5)

    # Panel B: Paired contrasts against Full
    contrast_styles = [
        ("cut_minus_full", "Cut - Full", "#54A24B", -0.15, "o"),
        ("zero_minus_full", "Zero - Full", "#E45756", 0.15, "s"),
    ]

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Full Baseline (0.0)")

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

        ax2.errorbar(
            x + offset, diff_means, yerr=[err_low, err_high],
            fmt=marker, color=color, ecolor=color, elinewidth=1.8, capsize=5,
            markersize=8, label=label
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel("Target Visual Token Budget")
    ax2.set_ylabel(r"$\Delta \tau_{\mathrm{effective}}$ vs Full (Bootstrap 95% CI)")
    ax2.set_title(f"B: Paired Contrast Effect vs Full Baseline\n({cfg_name})")
    ax2.set_ylim(-0.20, 0.14)
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"fig1_visual_cut_{cfg_key}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_config_retention_sweep(cfg_key, cfg_name, stats):
    """Plot Fig 2: Visual Retention Curves."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    retention_pcts = [0, 25, 50, 75, 100]
    cond_order = ["cut", "retained_25", "retained_50", "retained_75", "full"]

    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    markers = ["o", "s", "^", "D"]

    for b, color, marker in zip(BUDGETS, colors, markers):
        means = []
        lows = []
        highs = []
        for cond in cond_order:
            s = stats[cfg_key].get(f"{b}_{cond}", {}).get("tau_effective", {})
            m = s.get("mean", 0)
            means.append(m)
            lows.append(s.get("ci_low", m))
            highs.append(s.get("ci_high", m))

        ax.plot(retention_pcts, means, color=color, marker=marker, linewidth=2, markersize=7, label=f"Budget {b} tokens")
        ax.fill_between(retention_pcts, lows, highs, color=color, alpha=0.15)

    ax.set_xlabel("Visual Tokens Retained in Draft Context (%)")
    ax.set_ylabel(r"Effective Acceptance Rate $\tau_{\mathrm{effective}}$ (Tokens/Round)")
    ax.set_title(f"Visual Retention Sweep Curves Across Budgets\n({cfg_name})")
    ax.set_xticks(retention_pcts)
    ax.set_xticklabels(["0% (Cut)", "25%", "50%", "75%", "100% (Full)"])
    ax.set_ylim(2.08, 2.78)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"fig2_retention_sweep_{cfg_key}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_config_attention_dynamics(cfg_key, cfg_name, attn):
    """Plot Fig 3: Attention allocation across 4 regions, density per token, and layer distribution."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2))

    vis_mass = []
    prompt_text_mass = []
    gen_text_mass = []
    curr_query_mass = []

    vis_dens = []
    q_dens = []
    gen_dens = []

    layer0 = []
    layer1 = []
    layer2 = []

    for b in BUDGETS:
        full = attn[cfg_key].get(str(b), {}).get("full", {})
        vis_mass.append(full.get("vis_mass_mean", 0))
        prompt_text_mass.append(full.get("prompt_text_mass_mean", 0))
        gen_text_mass.append(full.get("gen_text_mass_mean", 0))
        curr_query_mass.append(full.get("curr_query_mass_mean", 0))

        vis_dens.append(full.get("vis_dens_mean", 0))
        q_dens.append(full.get("q_dens_mean", 0))
        gen_dens.append(full.get("gen_dens_mean", 0))

        layers = full.get("layer_vis_mass", {})
        layer0.append(layers.get("layer_0", 0))
        layer1.append(layers.get("layer_1", 0))
        layer2.append(layers.get("layer_2", 0))

    x = np.arange(len(BUDGETS))
    width = 0.55

    # Panel 1: 100% Stacked Bar Chart for Attention Mass
    vm = np.array(vis_mass)
    ptm = np.array(prompt_text_mass)
    gtm = np.array(gen_text_mass)
    cqm = np.array(curr_query_mass)

    p1 = ax1.bar(x, vm, width, label="Prompt Visual Tokens", color="#4C78A8", edgecolor="black", linewidth=0.5)
    p2 = ax1.bar(x, ptm, width, bottom=vm, label="Prompt Text (Question & System)", color="#F58518", edgecolor="black", linewidth=0.5)
    p3 = ax1.bar(x, gtm, width, bottom=vm + ptm, label="Generated Text Tokens (Verified)", color="#54A24B", edgecolor="black", linewidth=0.5)
    p4 = ax1.bar(x, cqm, width, bottom=vm + ptm + gtm, label="Current Draft Block (Causal)", color="#B279A2", edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(BUDGET_LABELS)
    ax1.set_xlabel("Target Visual Budget")
    ax1.set_ylabel("Attention Mass Share (Sums to 1.0)")
    ax1.set_title("A: Complete 100% Attention Breakdown")
    ax1.set_ylim(0, 1.35)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, framealpha=0.95, fontsize=8, columnspacing=0.8)

    # Panel 2: Density per Token (Log scale with ample headroom to avoid legend overlap)
    w_sub = 0.25
    ax2.bar(x - w_sub, gen_dens, width=w_sub, label="Generated Text Token Density", color="#54A24B", edgecolor="black", linewidth=0.5)
    ax2.bar(x, q_dens, width=w_sub, label="Prompt Question Token Density", color="#F58518", edgecolor="black", linewidth=0.5)
    ax2.bar(x + w_sub, vis_dens, width=w_sub, label="Visual Token Density", color="#4C78A8", edgecolor="black", linewidth=0.5)

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel("Target Visual Budget")
    ax2.set_ylabel("Attention Density per Token (Log Scale)")
    ax2.set_title("B: Per-Token Attention Density\n(Generated Text vs Question vs Visual)")
    ax2.set_ylim(bottom=5e-6, top=3.0)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8)

    # Panel 3: Layer-wise visual mass
    ax3.plot(x, layer0, marker="^", color="#72B7B2", linewidth=2, label="Layer 0 (Input)")
    ax3.plot(x, layer1, marker="o", color="#E45756", linewidth=2.5, label="Layer 1 (Middle)")
    ax3.plot(x, layer2, marker="s", color="#54A24B", linewidth=2, label="Layer 2 (Output)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(BUDGET_LABELS)
    ax3.set_xlabel("Target Visual Budget")
    ax3.set_ylabel("Visual Attention Mass")
    ax3.set_title("C: Visual Mass by Draft Layer")
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

    fig.suptitle(f"Attention Dynamics Across Visual Budgets — {cfg_name}", y=1.02, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUTPUT_DIR / f"fig3_attention_dynamics_{cfg_key}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_config_quality_tradeoff(cfg_key, cfg_name, stats):
    """Plot Fig 4: Quality & Speedup Trade-off."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    conditions = ["full", "retained_75", "retained_50", "retained_25", "cut"]
    cond_labels = ["Full", "Ret75", "Ret50", "Ret25", "Cut"]
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#E45756"]

    # Budget 3k and 25k comparison
    for b, marker, alpha, ls in [(3000, "o", 0.9, "-"), (25000, "s", 0.7, "--")]:
        taus = [stats[cfg_key].get(f"{b}_{c}", {}).get("tau_effective", {}).get("mean", 0) for c in conditions]
        tpss = [stats[cfg_key].get(f"{b}_{c}", {}).get("speculative_tps", {}).get("mean", 0) for c in conditions]
        ax1.plot(taus, tpss, marker=marker, linestyle=ls, linewidth=2, label=f"Budget {b} tokens", markersize=8)
        for t, s, l in zip(taus, tpss, cond_labels):
            ax1.annotate(l, (t, s), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    ax1.set_xlabel(r"Effective Acceptance $\tau_{\mathrm{effective}}$")
    ax1.set_ylabel("Decode Speed (Tokens/Second)")
    ax1.set_title(f"A: Speed vs Acceptance Trade-off\n({cfg_name})")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Panel B: Quality consistency (ROUGE-L and Coverage across conditions at 3k)
    rouges = [stats[cfg_key].get(f"3000_{c}", {}).get("spec_rouge_l", {}).get("mean", 0) for c in conditions]
    covs = [stats[cfg_key].get(f"3000_{c}", {}).get("spec_coverage", {}).get("mean", 0) for c in conditions]

    tgt_rouge = stats[cfg_key].get("3000_full", {}).get("tgt_rouge_l", {}).get("mean", 0)
    tgt_cov = stats[cfg_key].get("3000_full", {}).get("tgt_coverage", {}).get("mean", 0)

    x = np.arange(len(conditions))
    ax2.bar(x - 0.18, rouges, width=0.35, label="Speculative ROUGE-L", color="#4C78A8", alpha=0.9)
    ax2.bar(x + 0.18, covs, width=0.35, label="Speculative Coverage", color="#54A24B", alpha=0.9)
    ax2.axhline(tgt_rouge, color="#4C78A8", linestyle="--", linewidth=1.2, label=f"Target ROUGE-L ({tgt_rouge:.3f})")
    ax2.axhline(tgt_cov, color="#54A24B", linestyle=":", linewidth=1.2, label=f"Target Coverage ({tgt_cov:.3f})")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels)
    ax2.set_ylabel("Quality Metric Score")
    ax2.set_title(f"B: Output Quality Invariance (3K Budget)\n({cfg_name})")
    ax2.set_ylim(0, 0.52)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=8, columnspacing=0.8)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"fig4_quality_tradeoff_{cfg_key}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_master_summary_figures(stats, contrasts, attn):
    """Plot cross-config synthesis figures."""
    # Master Fig 1: 4 Config Comparison across budgets
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (b, b_label) in enumerate(zip(BUDGETS, BUDGET_LABELS)):
        ax = axes[idx]
        x = np.arange(len(CONFIGS))
        width = 0.25

        for c_idx, (cond, label, color) in enumerate([("full", "Full", "#4C78A8"), ("retained_25", "Retained 25%", "#54A24B"), ("cut", "Cut", "#E45756")]):
            means = []
            lows = []
            highs = []
            for cfg_key, _, _ in CONFIGS:
                s = stats[cfg_key].get(f"{b}_{cond}", {}).get("tau_effective", {})
                m = s.get("mean", 0)
                means.append(m)
                lows.append(m - s.get("ci_low", m))
                highs.append(s.get("ci_high", m) - m)

            offset = (c_idx - 1) * width
            ax.bar(x + offset, means, width, yerr=[lows, highs], capsize=4, label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([name.split("(")[0].strip() for _, name, _ in CONFIGS], rotation=15, ha="right", fontsize=9.5)
        ax.set_ylabel(r"$\tau_{\mathrm{effective}}$")
        ax.set_title(f"Budget: {b_label} Visual Tokens")
        ax.set_ylim(2.1, 2.65)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Cross-Configuration Acceptance Rate Across Visual Budgets", y=0.995, fontsize=16)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "summary_fig1_config_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Master Fig 2: Retention Curves at 25K Budget
    fig, ax = plt.subplots(figsize=(9, 5.5))
    retention_pcts = [0, 25, 50, 75, 100]
    cond_order = ["cut", "retained_25", "retained_50", "retained_75", "full"]

    for cfg_key, cfg_name, color in CONFIGS:
        means = []
        lows = []
        highs = []
        for cond in cond_order:
            s = stats[cfg_key].get(f"25000_{cond}", {}).get("tau_effective", {})
            m = s.get("mean", 0)
            means.append(m)
            lows.append(s.get("ci_low", m))
            highs.append(s.get("ci_high", m))

        ax.plot(retention_pcts, means, marker="o", linewidth=2.2, color=color, label=cfg_name, markersize=7)
        ax.fill_between(retention_pcts, lows, highs, color=color, alpha=0.12)

    ax.set_xlabel("Visual Tokens Retained in Draft Context (%)", fontsize=12)
    ax.set_ylabel(r"Effective Acceptance Rate $\tau_{\mathrm{effective}}$", fontsize=12)
    ax.set_title("Retention Sweep at 25,000 Visual Tokens (All 4 DFlash Configurations)", fontsize=14)
    ax.set_xticks(retention_pcts)
    ax.set_xticklabels(["0% (Cut)", "25%", "50%", "75%", "100% (Full)"])
    ax.set_ylim(2.14, 2.40)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "summary_fig2_retention_curves_25k.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Master Fig 3: Attention Dilution Effect & Component Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))

    # Panel A: Stacked bar for 4 configs at 25K Budget
    x_cfg = np.arange(len(CONFIGS))
    w_cfg = 0.55

    vms = []
    ptms = []
    gtms = []
    cqms = []

    for cfg_key, _, _ in CONFIGS:
        f = attn[cfg_key].get("25000", {}).get("full", {})
        vms.append(f.get("vis_mass_mean", 0))
        ptms.append(f.get("prompt_text_mass_mean", 0))
        gtms.append(f.get("gen_text_mass_mean", 0))
        cqms.append(f.get("curr_query_mass_mean", 0))

    vms = np.array(vms)
    ptms = np.array(ptms)
    gtms = np.array(gtms)
    cqms = np.array(cqms)

    ax1.bar(x_cfg, vms, w_cfg, label="Prompt Visual", color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax1.bar(x_cfg, ptms, w_cfg, bottom=vms, label="Prompt Text (Question)", color="#F58518", edgecolor="black", linewidth=0.5)
    ax1.bar(x_cfg, gtms, w_cfg, bottom=vms + ptms, label="Generated Text (Answer Prefix)", color="#54A24B", edgecolor="black", linewidth=0.5)
    ax1.bar(x_cfg, cqms, w_cfg, bottom=vms + ptms + gtms, label="Current Block (Causal)", color="#B279A2", edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x_cfg)
    ax1.set_xticklabels([name.split("(")[0].strip() for _, name, _ in CONFIGS], rotation=15, ha="right", fontsize=9.5)
    ax1.set_ylabel("Attention Mass Share (Sums to 1.0)")
    ax1.set_title("A: Attention Mass Share at 25K Budget\n(All 4 DFlash Configurations)")
    ax1.set_ylim(0, 1.35)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, framealpha=0.95, fontsize=8.5, columnspacing=0.8)

    # Panel B: Density comparison across token types across budgets
    # Plot average across all 4 configs
    mean_gen_dens = [np.mean([attn[c[0]].get(str(b), {}).get("full", {}).get("gen_dens_mean", 0) for c in CONFIGS]) for b in BUDGETS]
    mean_q_dens = [np.mean([attn[c[0]].get(str(b), {}).get("full", {}).get("q_dens_mean", 0) for c in CONFIGS]) for b in BUDGETS]
    mean_vis_dens = [np.mean([attn[c[0]].get(str(b), {}).get("full", {}).get("vis_dens_mean", 0) for c in CONFIGS]) for b in BUDGETS]

    w_b = 0.25
    x_b = np.arange(len(BUDGETS))
    ax2.bar(x_b - w_b, mean_gen_dens, w_b, label="Generated Text Token Density", color="#54A24B", edgecolor="black", linewidth=0.5)
    ax2.bar(x_b, mean_q_dens, w_b, label="Prompt Question Token Density", color="#F58518", edgecolor="black", linewidth=0.5)
    ax2.bar(x_b + w_b, mean_vis_dens, w_b, label="Visual Token Density", color="#4C78A8", edgecolor="black", linewidth=0.5)

    ax2.set_yscale("log")
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel("Target Visual Budget")
    ax2.set_ylabel("Per-Token Attention Density (Log Scale)")
    ax2.set_title("B: Per-Token Attention Density Hierarchy\n(Generated Text vs Question vs Visual)")
    ax2.set_ylim(bottom=5e-6, top=3.0)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    fig.suptitle("Attention Allocation and Density Hierarchy Across Configurations", y=1.02, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUTPUT_DIR / "summary_fig3_attention_dilution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Master Fig 4: Speedup vs Acceptance Map
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg_key, cfg_name, color in CONFIGS:
        for b, alpha, sz in [(400, 0.4, 40), (3000, 0.7, 70), (13000, 0.9, 100), (25000, 1.0, 140)]:
            s_full = stats[cfg_key].get(f"{b}_full", {})
            tau = s_full.get("tau_effective", {}).get("mean", 0)
            tps = s_full.get("speculative_tps", {}).get("mean", 0)
            ax.scatter(tau, tps, color=color, s=sz, alpha=alpha, edgecolors="black", linewidths=0.5)

        # Plot line through budgets
        taus = [stats[cfg_key].get(f"{b}_full", {}).get("tau_effective", {}).get("mean", 0) for b in BUDGETS]
        tpss = [stats[cfg_key].get(f"{b}_full", {}).get("speculative_tps", {}).get("mean", 0) for b in BUDGETS]
        ax.plot(taus, tpss, color=color, linestyle="--", linewidth=1.5, alpha=0.8, label=cfg_name)

    ax.set_xlabel(r"Effective Acceptance Rate $\tau_{\mathrm{effective}}$")
    ax.set_ylabel("Speculative Generation Speed (Tokens/Second)")
    ax.set_title("Pareto Map: Acceptance Rate vs Decoding Speed (Full Visual)", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate budget points
    ax.annotate("400 Tokens (~115 TPS)", xy=(2.52, 115), xytext=(2.40, 120),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=10)
    ax.annotate("3K Tokens (~92 TPS)", xy=(2.41, 92), xytext=(2.48, 92),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=10)
    ax.annotate("13K Tokens (~30 TPS)", xy=(2.23, 30), xytext=(2.32, 40),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=10)
    ax.annotate("25K Tokens (~13 TPS)", xy=(2.21, 13), xytext=(2.30, 18),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6), fontsize=10)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "summary_fig4_speedup_vs_acceptance.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    stats, contrasts, attn = load_data()
    print("Generating figures for individual configs...")
    for cfg_key, cfg_name, _ in CONFIGS:
        plot_config_visual_cut(cfg_key, cfg_name, stats, contrasts)
        plot_config_retention_sweep(cfg_key, cfg_name, stats)
        plot_config_attention_dynamics(cfg_key, cfg_name, attn)
        plot_config_quality_tradeoff(cfg_key, cfg_name, stats)

    print("\nGenerating master summary figures...")
    plot_master_summary_figures(stats, contrasts, attn)
    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()
