"""Generate publication-quality figures for DFlash balanced [1, 9, 17, 25, 33] config."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

DATA_DIR = Path("results/visualcut_attention/vdc50_visualcut_balanced_20261029/summary_reports_data")
OUTPUT_DIR = Path("markdown/Baocao_DFlash_4Config_20261029/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFG_KEY = "dflash_3layer_1_9_17_25_33"
CFG_NAME = "Balanced [1, 9, 17, 25, 33]"

BUDGETS = [400, 3000, 13000, 25000]
BUDGET_LABELS = ["400", "3K", "13K", "25K"]

def load_data():
    with open(DATA_DIR / "condition_stats.json") as f:
        stats = json.load(f)[CFG_KEY]
    with open(DATA_DIR / "paired_contrasts.json") as f:
        contrasts = json.load(f)[CFG_KEY]
    with open(DATA_DIR / "attention_profiles.json") as f:
        attn = json.load(f)[CFG_KEY]
    return stats, contrasts, attn

def plot_visual_cut(stats, contrasts):
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
            s = stats.get(f"{b}_{cond}", {}).get("tau_effective", {})
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
    ax1.set_title(f"A: Acceptance Rate Across Visual Budgets\n({CFG_NAME})")
    ax1.set_ylim(2.0, 2.70)
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
            c = contrasts.get(f"{b}_{c_key}", {}).get("tau_effective", {})
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
    ax2.set_title(f"B: Paired Contrast Effect vs Full Baseline\n({CFG_NAME})")
    ax2.set_ylim(-0.20, 0.14)
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"fig1_visual_cut_{CFG_KEY}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_retention_sweep(stats):
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
            s = stats.get(f"{b}_{cond}", {}).get("tau_effective", {})
            m = s.get("mean", 0)
            means.append(m)
            lows.append(s.get("ci_low", m))
            highs.append(s.get("ci_high", m))

        ax.plot(retention_pcts, means, color=color, marker=marker, linewidth=2, markersize=7, label=f"Budget {b} tokens")
        ax.fill_between(retention_pcts, lows, highs, color=color, alpha=0.15)

    ax.set_xlabel("Visual Tokens Retained in Draft Context (%)")
    ax.set_ylabel(r"Effective Acceptance Rate $\tau_{\mathrm{effective}}$ (Tokens/Round)")
    ax.set_title(f"Visual Retention Sweep Curves Across Budgets\n({CFG_NAME})")
    ax.set_xticks(retention_pcts)
    ax.set_xticklabels(["0% (Cut)", "25%", "50%", "75%", "100% (Full)"])
    ax.set_ylim(2.14, 2.65)
    ax.legend(loc="lower left", framealpha=0.92, fontsize=8.5)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"fig2_retention_sweep_{CFG_KEY}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_attention_dynamics(attn):
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
        full = attn.get(str(b), {}).get("full", {})
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

    vm = np.array(vis_mass)
    ptm = np.array(prompt_text_mass)
    gtm = np.array(gen_text_mass)
    cqm = np.array(curr_query_mass)

    ax1.bar(x, vm, width, label="Prompt Visual Tokens", color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax1.bar(x, ptm, width, bottom=vm, label="Prompt Text (Question & System)", color="#F58518", edgecolor="black", linewidth=0.5)
    ax1.bar(x, gtm, width, bottom=vm + ptm, label="Generated Text Tokens (Verified)", color="#54A24B", edgecolor="black", linewidth=0.5)
    ax1.bar(x, cqm, width, bottom=vm + ptm + gtm, label="Current Draft Block (Causal)", color="#B279A2", edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(BUDGET_LABELS)
    ax1.set_xlabel("Target Visual Budget")
    ax1.set_ylabel("Attention Mass Share (Sums to 1.0)")
    ax1.set_title("A: Complete 100% Attention Breakdown")
    ax1.set_ylim(0, 1.35)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, framealpha=0.95, fontsize=8, columnspacing=0.8)

    # Panel 2: Density per Token
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

    fig.suptitle(f"Attention Dynamics Across Visual Budgets — {CFG_NAME}", y=1.02, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUTPUT_DIR / f"fig3_attention_dynamics_{CFG_KEY}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_quality_tradeoff(stats):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    conditions = ["full", "retained_75", "retained_50", "retained_25", "cut"]
    cond_labels = ["Full", "Ret75", "Ret50", "Ret25", "Cut"]

    # Budget 3k and 25k comparison
    for b, marker, alpha, ls in [(3000, "o", 0.9, "-"), (25000, "s", 0.7, "--")]:
        taus = [stats.get(f"{b}_{c}", {}).get("tau_effective", {}).get("mean", 0) for c in conditions]
        tpss = [stats.get(f"{b}_{c}", {}).get("speculative_tps", {}).get("mean", 0) for c in conditions]
        ax1.plot(taus, tpss, marker=marker, linestyle=ls, linewidth=2, label=f"Budget {b} tokens", markersize=8)
        for t, s, l in zip(taus, tpss, cond_labels):
            ax1.annotate(l, (t, s), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    ax1.set_xlabel(r"Effective Acceptance $\tau_{\mathrm{effective}}$")
    ax1.set_ylabel("Decode Speed (Tokens/Second)")
    ax1.set_title(f"A: Speed vs Acceptance Trade-off\n({CFG_NAME})")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Panel B: Quality consistency (ROUGE-L and Coverage across conditions at 3k)
    rouges = [stats.get(f"3000_{c}", {}).get("spec_rouge_l", {}).get("mean", 0) for c in conditions]
    covs = [stats.get(f"3000_{c}", {}).get("spec_coverage", {}).get("mean", 0) for c in conditions]

    tgt_rouge = stats.get("3000_full", {}).get("tgt_rouge_l", {}).get("mean", 0)
    tgt_cov = stats.get("3000_full", {}).get("tgt_coverage", {}).get("mean", 0)

    x = np.arange(len(conditions))
    ax2.bar(x - 0.18, rouges, width=0.35, label="Speculative ROUGE-L", color="#4C78A8", alpha=0.9)
    ax2.bar(x + 0.18, covs, width=0.35, label="Speculative Coverage", color="#54A24B", alpha=0.9)
    ax2.axhline(tgt_rouge, color="#4C78A8", linestyle="--", linewidth=1.2, label=f"Target ROUGE-L ({tgt_rouge:.3f})")
    ax2.axhline(tgt_cov, color="#54A24B", linestyle=":", linewidth=1.2, label=f"Target Coverage ({tgt_cov:.3f})")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels)
    ax2.set_ylabel("Quality Metric Score")
    ax2.set_title(f"B: Output Quality Invariance (3K Budget)\n({CFG_NAME})")
    ax2.set_ylim(0, 0.52)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.95, fontsize=8, columnspacing=0.8)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"fig4_quality_tradeoff_{CFG_KEY}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    stats, contrasts, attn = load_data()
    print("Generating figures for balanced config...")
    plot_visual_cut(stats, contrasts)
    plot_retention_sweep(stats)
    plot_attention_dynamics(attn)
    plot_quality_tradeoff(stats)
    print("All balanced figures generated successfully!")

if __name__ == "__main__":
    main()
