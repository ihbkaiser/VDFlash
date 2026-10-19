"""Generate publication-quality figures for the DFlash text consistency & visual context study.

Produces:
1. figure1_sensitivity_vs_retention.png: Proposal Policy Sensitivity (Assay 1)
2. figure2_speculative_acceptance_tau.png: Verified Speculative Acceptance (Assay 3)
3. figure3_free_running_repetition_collapse.png: Free-running Degenerative Repetition & Floor Effect (Assay 2)
4. figure4_evidence_synthesis_dashboard.png: Comprehensive Hypothesis Synthesis Dashboard
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set global matplotlib styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

RESULTS_DIR = Path("results/dflash_free_running_main_fp16_20261014")
OUTPUT_DIR = Path("markdown/hypothesis_text_consistency_visual_context/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_figure1_sensitivity():
    """Figure 1: Teacher-forced Proposal Policy Sensitivity across Visual Retention."""
    sens_file = RESULTS_DIR / "visual_sensitivity_summary.json"
    data = json.loads(sens_file.read_text())

    conditions = data["conditions"]
    contrasts = data["contrasts"]

    # Extract retentions and metrics
    # Reverse to plot from 100% down to 0% (or 0% to 100%)
    retentions = [100, 75, 50, 25, 0]
    labels = ["Full\n(100%)", "Retain-75\n(75%)", "Retain-50\n(50%)", "Retain-25\n(25%)", "Cut\n(0%)"]
    
    # Map data
    cond_map = {c["retention_percentage"]: c for c in conditions}
    cont_map = {c["retention_percentage"]: c for c in contrasts}
    
    agreements = [cond_map[r]["mean_proposal_agreement_with_full"] * 100 for r in retentions]
    changed = [cond_map[r]["mean_proposal_change_fraction_vs_full"] * 100 for r in retentions]
    
    changed_low = [0.0] + [cont_map[r]["changed_fraction_ci95_low"] * 100 for r in [75, 50, 25, 0]]
    changed_high = [0.0] + [cont_map[r]["changed_fraction_ci95_high"] * 100 for r in [75, 50, 25, 0]]
    
    js_div = [cond_map[r]["mean_coarse_distribution_js_vs_full"] for r in retentions]
    js_low = [0.0] + [cont_map[r]["coarse_distribution_js_ci95_low"] for r in [75, 50, 25, 0]]
    js_high = [0.0] + [cont_map[r]["coarse_distribution_js_ci95_high"] for r in [75, 50, 25, 0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figure 1: Draft Proposal Sensitivity to Visual Retention (Teacher-Forced Assay)", fontweight="bold", y=0.98)

    # Panel A: Agreement vs Changed Proposals
    x = np.arange(len(retentions))
    
    # Plot Changed Proposal % with error bands
    ax1.plot(x, changed, marker="o", color="#DC2626", linewidth=2.2, label="Proposal Changed vs Full (%)", zorder=3)
    ax1.fill_between(x, changed_low, changed_high, color="#DC2626", alpha=0.18, label="95% Bootstrap CI")
    
    # Plot Agreement %
    ax1.plot(x, agreements, marker="s", color="#1E40AF", linewidth=2.0, linestyle="--", label="Agreement with Full (%)", zorder=3)

    # Data labels on changed proposals
    for i, (val, cl, ch) in enumerate(zip(changed, changed_low, changed_high)):
        offset = 2.5 if i < 4 else -4.0
        ax1.annotate(f"{val:.1f}%", (x[i], val + offset), ha="center", fontsize=8.5, fontweight="bold", color="#991B1B")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Proposal Percentage (%)", fontweight="bold")
    ax1.set_title("(A) Proposal Deviation from Full Condition", fontweight="bold", pad=10)
    ax1.set_ylim(-2, 105)
    ax1.legend(loc="center left", framealpha=0.9)

    # Callout box in Panel A (placed in open space between curves)
    ax1.text(0.12, 0.68, "Cut flips 39.7% of greedy proposals\non fixed target prefixes (n=1,729 tokens)", 
             transform=ax1.transAxes, fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec="#F87171", lw=1))

    # Panel B: Distribution JS Divergence
    ax2.plot(x, js_div, marker="^", color="#7C3AED", linewidth=2.2, label="JS Divergence vs Full", zorder=3)
    ax2.fill_between(x, js_low, js_high, color="#7C3AED", alpha=0.18, label="95% Bootstrap CI")
    
    for i, val in enumerate(js_div):
        if val > 0:
            ax2.annotate(f"{val:.4f}", (x[i], val + 0.002), ha="center", fontsize=8.5, fontweight="bold", color="#5B21B6")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Jensen-Shannon Divergence (Coarse)", fontweight="bold")
    ax2.set_title("(B) Token Distribution Shift (JS Divergence)", fontweight="bold", pad=10)
    ax2.set_ylim(-0.003, 0.048)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure1_sensitivity_vs_retention.png", dpi=300)
    plt.close(fig)
    print("Saved figure1_sensitivity_vs_retention.png")


def plot_figure2_acceptance():
    """Figure 2: Verified Speculative Acceptance Rate across Visual Retention Conditions."""
    acc_file = RESULTS_DIR / "acceptance_summary.json"
    data = json.loads(acc_file.read_text())

    conditions = data["conditions"]
    contrasts = data["contrasts"]

    retentions = [100, 75, 50, 25, 0]
    cond_labels = ["Full\n(100%)", "Retain-75\n(75%)", "Retain-50\n(50%)", "Retain-25\n(25%)", "Cut\n(0%)"]

    cond_map = {c["retention_percentage"]: c for c in conditions}
    cont_map = {c["retention_percentage"]: c for c in contrasts}

    # Panel A: Absolute tau_effective
    tau_eff = [cond_map[r]["mean_tau_effective"] for r in retentions]
    tau_eff_low = [cond_map[r]["tau_effective_ci95_low"] for r in retentions]
    tau_eff_high = [cond_map[r]["tau_effective_ci95_high"] for r in retentions]
    err_low = [m - l for m, l in zip(tau_eff, tau_eff_low)]
    err_high = [h - m for m, h in zip(tau_eff, tau_eff_high)]

    # Panel B: Delta tau_effective vs Full
    delta_r = [75, 50, 25, 0]
    delta_labels = ["Retain-75", "Retain-50", "Retain-25", "Cut"]
    deltas = [cont_map[r]["mean_tau_effective_delta"] for r in delta_r]
    d_low = [cont_map[r]["tau_effective_delta_ci95_low"] for r in delta_r]
    d_high = [cont_map[r]["tau_effective_delta_ci95_high"] for r in delta_r]
    d_err_low = [d - l for d, l in zip(deltas, d_low)]
    d_err_high = [h - d for d, h in zip(deltas, d_high)]

    # Panel C: Wins, Ties, Losses
    wins = [cont_map[r]["wins"] for r in delta_r]
    ties = [cont_map[r]["ties"] for r in delta_r]
    losses = [cont_map[r]["losses"] for r in delta_r]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Figure 2: Verified Speculative Acceptance Evaluation (Exact Target Parity, n=24)", fontweight="bold", y=0.98)

    # Subplot A: Absolute tau_effective
    x1 = np.arange(len(retentions))
    colors_bar = ["#1E3A8A", "#0D9488", "#0284C7", "#D97706", "#DC2626"]
    bars1 = ax1.bar(x1, tau_eff, yerr=[err_low, err_high], capsize=4, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cond_labels)
    ax1.set_ylabel(r"Mean $\tau_{\mathrm{effective}}$ (tokens/round)", fontweight="bold")
    ax1.set_title(r"(A) Absolute Acceptance Rate ($\tau_{\mathrm{eff}}$)", fontweight="bold", pad=10)
    ax1.set_ylim(2.1, 2.5)
    for bar, val in zip(bars1, tau_eff):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Subplot B: Paired Delta tau_effective
    x2 = np.arange(len(delta_r))
    colors_delta = ["#0D9488" if d >= 0 else "#DC2626" for d in deltas]
    bars2 = ax2.bar(x2, deltas, yerr=[d_err_low, d_err_high], capsize=4, color=colors_delta, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(delta_labels)
    ax2.set_ylabel(r"$\Delta \tau_{\mathrm{effective}}$ vs Full", fontweight="bold")
    ax2.set_title(r"(B) Paired Difference $\Delta \tau_{\mathrm{eff}}$ (Cut − Full)", fontweight="bold", pad=10)
    ax2.set_ylim(-0.095, 0.055)
    for bar, val in zip(bars2, deltas):
        va = "bottom" if val >= 0 else "top"
        offset = 0.005 if val >= 0 else -0.005
        ax2.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.3f}", ha="center", va=va, fontsize=8.5, fontweight="bold")

    # Subplot C: Win / Tie / Loss
    x3 = np.arange(len(delta_r))
    width = 0.55
    b_win = ax3.bar(x3, wins, width, label="Wins (Better than Full)", color="#10B981", edgecolor="black", linewidth=0.8)
    b_tie = ax3.bar(x3, ties, width, bottom=wins, label="Ties", color="#9CA3AF", edgecolor="black", linewidth=0.8)
    bottom_loss = [w + t for w, t in zip(wins, ties)]
    b_loss = ax3.bar(x3, losses, width, bottom=bottom_loss, label="Losses (Worse than Full)", color="#EF4444", edgecolor="black", linewidth=0.8)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(delta_labels)
    ax3.set_ylabel("Number of Video Samples (Total = 24)", fontweight="bold")
    ax3.set_title("(C) Paired Sample Outcomes (Win / Tie / Loss)", fontweight="bold", pad=10)
    ax3.set_ylim(0, 27)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # Annotate Cut losses
    ax3.text(3, bottom_loss[3] + losses[3] / 2, f"{losses[3]}L", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax3.text(3, wins[3] / 2, f"{wins[3]}W", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure2_speculative_acceptance_tau.png", dpi=300)
    plt.close(fig)
    print("Saved figure2_speculative_acceptance_tau.png")


def plot_figure3_repetition_collapse():
    """Figure 3: Free-running Degenerative Repetition & Floor Effect."""
    raw_path = RESULTS_DIR / "raw_rollouts.jsonl"
    data = [json.loads(line) for line in raw_path.read_text().strip().split("\n") if line]

    cond_stats = {}
    for d in data:
        cond = d["condition"]
        if cond not in cond_stats:
            cond_stats[cond] = {"total": 0, "empty": 0, "top1_ratios": [], "unique_ratios": []}
        cond_stats[cond]["total"] += 1
        tokens = d.get("prediction_token_ids", [])
        if not tokens or len(tokens) <= 1:
            cond_stats[cond]["empty"] += 1
        else:
            counts = Counter(tokens)
            cond_stats[cond]["top1_ratios"].append(counts.most_common(1)[0][1] / len(tokens))
            cond_stats[cond]["unique_ratios"].append(len(counts) / len(tokens))

    order = ["full", "retain_75", "retain_50", "retain_25", "cut"]
    labels = ["Full", "Retain-75", "Retain-50", "Retain-25", "Cut"]

    empty_counts = [cond_stats[c]["empty"] for c in order]
    scorable_counts = [cond_stats[c]["total"] - cond_stats[c]["empty"] for c in order]
    top1_means = [np.mean(cond_stats[c]["top1_ratios"]) * 100 for c in order]
    uniq_means = [np.mean(cond_stats[c]["unique_ratios"]) * 100 for c in order]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Figure 3: Free-Running Draft Rollout Degeneration & Judge Floor Effect (33 Samples)", fontweight="bold", y=0.98)

    # Panel A: Output Coverage (EOS vs Scorable Text)
    x = np.arange(len(order))
    width = 0.55
    ax1.bar(x, scorable_counts, width, label="Scorable Output (Text Generated)", color="#3B82F6", edgecolor="black", linewidth=0.8)
    ax1.bar(x, empty_counts, width, bottom=scorable_counts, label="Empty Output (Immediate EOS)", color="#F97316", edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Number of Rollouts (Cohort = 33)", fontweight="bold")
    ax1.set_title("(A) Output Generation Coverage", fontweight="bold", pad=10)
    ax1.set_ylim(0, 38)
    ax1.legend(loc="upper right", framealpha=0.9)

    for i, (sc, em) in enumerate(zip(scorable_counts, empty_counts)):
        ax1.text(x[i], sc / 2, f"{sc}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if em > 0:
            ax1.text(x[i], sc + em / 2, f"{em}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    # Panel B: Repetition Collapse (Top-1 Dominance)
    ax2.bar(x, top1_means, width, color="#EF4444", alpha=0.85, edgecolor="black", linewidth=0.8, label="Top-1 Single Token Ratio (%)")
    ax2.plot(x, uniq_means, color="#1E3A8A", marker="s", linewidth=2.0, label="Unique Token Diversity (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Token Proportion (%)", fontweight="bold")
    ax2.set_title("(B) Degenerative Repetition Pathology", fontweight="bold", pad=10)
    ax2.set_ylim(0, 108)
    ax2.legend(loc="lower left", framealpha=0.95, fontsize=8)

    for i, val in enumerate(top1_means):
        ax2.text(x[i], val + 1.5, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#991B1B")

    ax2.text(0.58, 0.45, "Severe repetition loops:\nSingle token ('a', 'paint', 'image')\nmakes up 83% - 93% of all output!", 
             transform=ax2.transAxes, ha="center", fontsize=8.2, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#DC2626", lw=1.2, alpha=0.96))

    # Panel C: Radar / Bar of 5 Coherence Dimensions (All 1.0)
    dims = ["Grammar", "Reference", "Discourse", "Repetition", "Overall"]
    # All scores are exactly 1.0 across all conditions
    x3 = np.arange(len(dims))
    w3 = 0.16
    colors_dim = ["#1E3A8A", "#0D9488", "#0284C7", "#D97706", "#DC2626"]
    for idx, (cond_name, col) in enumerate(zip(labels, colors_dim)):
        ax3.bar(x3 + (idx - 2) * w3, [1.0] * 5, w3, label=cond_name, color=col, alpha=0.85, edgecolor="black", linewidth=0.7)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(dims, rotation=15)
    ax3.set_ylabel("LLM Judge Score (1 to 5)", fontweight="bold")
    ax3.set_title("(C) LLM Coherence Judge Scores (Floor Effect)", fontweight="bold", pad=10)
    ax3.set_ylim(0, 5.5)
    ax3.axhline(1.0, color="#DC2626", linestyle=":", linewidth=1.5)
    ax3.text(0.5, 0.65, "FLOOR EFFECT (Score = 1.0 / 5)\nAll 88 pairs evaluated as 1/5 & Tie\nNo discriminative signal between Full & Cut", 
             transform=ax3.transAxes, ha="center", fontsize=8.2, bbox=dict(boxstyle="round,pad=0.35", fc="#FFFBEB", ec="#F59E0B", lw=1))
    ax3.legend(loc="upper right", ncol=2, fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure3_free_running_repetition_collapse.png", dpi=300)
    plt.close(fig)
    print("Saved figure3_free_running_repetition_collapse.png")


def plot_figure4_dashboard():
    """Figure 4: Comprehensive Multi-panel Synthesis Dashboard."""
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Figure 4: Synthesis of Empirical Evidence on DFlash Text Consistency vs. Visual Context", fontweight="bold", fontsize=14, y=0.98)
    
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
    
    ax_hyp = fig.add_subplot(gs[0, 0])
    ax_sens = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[0, 2])
    ax_rep = fig.add_subplot(gs[1, 0])
    ax_judge = fig.add_subplot(gs[1, 1])
    ax_verdict = fig.add_subplot(gs[1, 2])

    # 1. Hypotheses & Protocol Box
    ax_hyp.axis("off")
    hyp_text = (
        "CORE HYPOTHESIS UNDER TEST:\n"
        "───────────────────────────────\n"
        "\"In DFlash, is text consistency truly\n"
        "independent of visual context?\n"
        "Reducing visual does NOT guarantee\n"
        "an increase in text consistency.\"\n\n"
        "THREE OPERATIONAL ASSAYS:\n"
        "• Assay 1 (H-S): Proposal Sensitivity\n"
        "  (Fixed target prefix, 10 samples, 1729 tok)\n"
        "• Assay 2 (H-C): Free-running Coherence\n"
        "  (Unverified draft rollout, 256 tok, blinded)\n"
        "• Assay 3 (H-A): Verified Speculative Tau\n"
        "  (Exact parity speculative decoding, 24 samples)\n\n"
        "MODEL & PROTOCOL:\n"
        "• Target: Qwen2.5-VL-3B-Instruct (FP16)\n"
        "• Drafter: DFlash LLaVA depth-3\n"
        "• Dataset: VDC50 (33 valid calibration rows)"
    )
    ax_hyp.text(0.02, 0.98, hyp_text, va="top", ha="left", fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.6", fc="#F8FAFC", ec="#64748B", lw=1.2))
    ax_hyp.set_title("(A) Research Framing & Setup", fontweight="bold")

    # 2. Sensitivity Curve
    sens_file = RESULTS_DIR / "visual_sensitivity_summary.json"
    sens_data = json.loads(sens_file.read_text())
    ret = [100, 75, 50, 25, 0]
    labels_s = ["100%", "75%", "50%", "25%", "0%"]
    cond_map = {c["retention_percentage"]: c for c in sens_data["conditions"]}
    chg = [cond_map[r]["mean_proposal_change_fraction_vs_full"] * 100 for r in ret]
    
    ax_sens.plot(labels_s, chg, marker="o", color="#DC2626", linewidth=2.2)
    ax_sens.set_ylabel("% Proposals Changed vs Full", fontweight="bold")
    ax_sens.set_xlabel("Visual Retention Ratio", fontweight="bold")
    ax_sens.set_title("(B) Token Policy Sensitivity (H-S)", fontweight="bold")
    for i, v in enumerate(chg):
        ax_sens.annotate(f"{v:.1f}%", (i, v + 2.5), ha="center", fontsize=8, fontweight="bold", color="#991B1B")
    ax_sens.set_ylim(-3, 48)
    ax_sens.text(0.08, 0.65, "NON-INDEPENDENT:\n39.7% of proposals change\nwhen visual is removed", 
                 transform=ax_sens.transAxes, ha="left", fontsize=8.2, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#F87171", lw=1))

    # 3. Acceptance Paired Delta
    acc_file = RESULTS_DIR / "acceptance_summary.json"
    acc_data = json.loads(acc_file.read_text())
    delta_r = [75, 50, 25, 0]
    delta_labels = ["75%", "50%", "25%", "Cut"]
    cont_map = {c["retention_percentage"]: c for c in acc_data["contrasts"]}
    deltas = [cont_map[r]["mean_tau_effective_delta"] for r in delta_r]
    d_low = [cont_map[r]["tau_effective_delta_ci95_low"] for r in delta_r]
    d_high = [cont_map[r]["tau_effective_delta_ci95_high"] for r in delta_r]
    
    colors_d = ["#0D9488" if d >= 0 else "#DC2626" for d in deltas]
    ax_acc.bar(delta_labels, deltas, yerr=[[d - l for d, l in zip(deltas, d_low)], [h - d for d, h in zip(deltas, d_high)]],
               capsize=3.5, color=colors_d, alpha=0.85, edgecolor="black")
    ax_acc.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax_acc.set_ylabel(r"$\Delta \tau_{\mathrm{eff}}$ vs Full (tokens/round)", fontweight="bold")
    ax_acc.set_xlabel("Visual Retention Condition", fontweight="bold")
    ax_acc.set_title(r"(C) Speculative Acceptance $\Delta \tau$ (H-A)", fontweight="bold")
    ax_acc.set_ylim(-0.095, 0.055)
    for i, v in enumerate(deltas):
        va = "bottom" if v >= 0 else "top"
        ax_acc.annotate(f"{v:+.3f}", (i, v + (0.004 if v >= 0 else -0.004)), ha="center", va=va, fontsize=8, fontweight="bold")

    # 4. Repetition pathology
    raw_path = RESULTS_DIR / "raw_rollouts.jsonl"
    raw_data = [json.loads(line) for line in raw_path.read_text().strip().split("\n") if line]
    cond_stats = {}
    for d in raw_data:
        cond = d["condition"]
        if cond not in cond_stats:
            cond_stats[cond] = []
        tokens = d.get("prediction_token_ids", [])
        if tokens and len(tokens) > 1:
            counts = Counter(tokens)
            cond_stats[cond].append(counts.most_common(1)[0][1] / len(tokens) * 100)
    
    order = ["full", "retain_75", "retain_50", "retain_25", "cut"]
    labels_o = ["Full", "75%", "50%", "25%", "Cut"]
    rep_vals = [np.mean(cond_stats[c]) for c in order]
    ax_rep.bar(labels_o, rep_vals, color="#EA580C", alpha=0.85, edgecolor="black")
    ax_rep.set_ylabel("Top-1 Token Dominance (%)", fontweight="bold")
    ax_rep.set_xlabel("Condition", fontweight="bold")
    ax_rep.set_title("(D) Free-running Repetition Pathology", fontweight="bold")
    ax_rep.set_ylim(0, 110)
    for i, v in enumerate(rep_vals):
        ax_rep.annotate(f"{v:.1f}%", (i, v + 2), ha="center", fontsize=8, fontweight="bold")

    # 5. Coherence Floor Effect
    ax_judge.bar(["Cut - Full\n(Full Text)", "Cut - Full\n(Prefix 128)"], [0.0, 0.0], color="#94A3B8", edgecolor="black")
    ax_judge.set_ylabel("Coherence Score Delta", fontweight="bold")
    ax_judge.set_title("(E) Coherence Floor Effect (H-C)", fontweight="bold")
    ax_judge.set_ylim(-1, 1)
    ax_judge.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax_judge.text(0.5, 0.7, "DELTA = 0.0\n100% of pairs scored 1/5\nExtreme Floor Effect due to\nDegenerative Repetition", 
                  transform=ax_judge.transAxes, ha="center", fontsize=8.2,
                  bbox=dict(boxstyle="round,pad=0.4", fc="#FFFBEB", ec="#F59E0B"))

    # 6. Overall Verdict Card
    ax_verdict.axis("off")
    verdict_text = (
        "FINAL SCIENTIFIC VERDICT\n"
        "═══════════════════════════════\n"
        "HYPOTHESIS IS CONFIRMED:\n\n"
        "1. NOT INDEPENDENT:\n"
        "   Draft token generation is deeply\n"
        "   coupled to visual context (~40%\n"
        "   proposals shift upon Cut).\n\n"
        "2. NO CONSISTENCY GAIN:\n"
        "   Reducing visual context fails to\n"
        "   prevent draft representation\n"
        "   collapse (all conditions degrade\n"
        "   into single-token loops).\n\n"
        "3. SPEEDUP DEGRADATION:\n"
        "   Cut condition decreases speculative\n"
        "   acceptance (Δtau = -0.039).\n\n"
        "IMPLICATION FOR SPECULATIVE VLM:\n"
        "Naive visual dropping harms accuracy\n"
        "and acceptance. Learned semantic\n"
        "compression is required."
    )
    ax_verdict.text(0.02, 0.98, verdict_text, va="top", ha="left", fontsize=8.2, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.6", fc="#F0FDF4", ec="#16A34A", lw=1.4))
    ax_verdict.set_title("(F) Empirical Conclusions", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure4_evidence_synthesis_dashboard.png", dpi=300)
    plt.close(fig)
    print("Saved figure4_evidence_synthesis_dashboard.png")


if __name__ == "__main__":
    plot_figure1_sensitivity()
    plot_figure2_acceptance()
    plot_figure3_repetition_collapse()
    plot_figure4_dashboard()
    print("All figures successfully generated in", OUTPUT_DIR)
