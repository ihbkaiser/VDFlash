import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set clean scientific plotting style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

FIG_DIR = Path("markdown/22_10_26/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
def load_and_group(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    from collections import defaultdict
    by_ret = defaultdict(list)
    for r in rows:
        by_ret[float(r["retention_percentage"])].append(r)
    return by_ret

dflash_data = load_and_group("results/full_suffix_retention/dflash/vdc/dflash_retention_suffix_vdc.jsonl")
msd3b_data = load_and_group("results/full_suffix_retention/msd_3b/depth3/vdc/msd_retention_suffix_vdc.jsonl")
msd7b_data = load_and_group("results/full_suffix_retention/msd_7b/msd_retention_suffix_vdc.jsonl")

ret_levels = [100.0, 50.0, 25.0, 5.0, 0.0]
ret_labels = ["100%", "50%", "25%", "5%", "0%\n(Text-only)"]

def compute_metrics(group_dict):
    dec_spds = []
    e2e_spds = []
    taus = []
    lossless_rates = []
    prefix_rates = []
    vis_tokens = []

    for ret in ret_levels:
        g = group_dict[ret]
        n = len(g)
        dec_spds.append(sum(r.get("decode_speedup", 1.0) for r in g) / n)
        e2e_spds.append(sum(r.get("end_to_end_speedup", 1.0) for r in g) / n)
        vis_tokens.append(sum(r.get("actual_visual_tokens", 0) for r in g) / n)
        
        # Tau
        t_list = []
        for r in g:
            if "tau_effective" in r:
                t_list.append(float(r["tau_effective"]))
            elif "speculative_output_ids" in r and "verification_steps" in r and r["verification_steps"] > 0:
                t_list.append(len(r["speculative_output_ids"]) / float(r["verification_steps"]))
            elif "spec_num_tokens" in r and "verification_steps" in r and r["verification_steps"] > 0:
                t_list.append(r["spec_num_tokens"] / float(r["verification_steps"]))
            else:
                t_list.append(1.0)
        taus.append(sum(t_list) / len(t_list))

        # Lossless
        lossless_count = sum(1 for r in g if r.get("lossless") is True)
        lossless_rates.append((lossless_count / n) * 100.0)

        # Prefix Match
        p_matches = []
        for r in g:
            tot = len(r.get("target_output_ids", [])) or r.get("target_num_tokens", 0) or len(r.get("speculative_output_ids", [])) or r.get("spec_num_tokens", 0) or 1
            pref = r.get("lossless_prefix_length", 0)
            if r.get("lossless") is True:
                p_matches.append(100.0)
            else:
                p_matches.append(min(100.0, (pref / max(tot, 1)) * 100.0))
        prefix_rates.append(sum(p_matches) / len(p_matches))

    return {
        "dec_spds": dec_spds,
        "e2e_spds": e2e_spds,
        "taus": taus,
        "lossless_rates": lossless_rates,
        "prefix_rates": prefix_rates,
        "vis_tokens": vis_tokens,
    }

dflash_metrics = compute_metrics(dflash_data)
msd3b_metrics = compute_metrics(msd3b_data)
msd7b_metrics = compute_metrics(msd7b_data)

x = np.arange(len(ret_levels))

# ------------------------------------------------------------------------------
# Figure 1: Speedup Comparison (Decode Speedup & E2E Speedup)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot Decode Speedup
ax1.plot(x, dflash_metrics["dec_spds"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3 (Qwen2.5-VL-3B)")
ax1.plot(x, msd7b_metrics["dec_spds"], marker="s", linewidth=2.5, markersize=8, color="#d62728", label="MSD 7B Gốc (Qwen2-VL-7B)")
ax1.plot(x, msd3b_metrics["dec_spds"], marker="^", linewidth=2.0, markersize=7, color="#ff7f0e", linestyle="--", label="MSD Depth 3 (Qwen2.5-VL-3B)")
ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Autoregressive Baseline (1.0x)")
ax1.set_xticks(x)
ax1.set_xticklabels(ret_labels)
ax1.set_xlabel("Suffix Visual Retention Level (%)")
ax1.set_ylabel("Decode Speedup (x)")
ax1.set_title("(a) Pure Decode Speedup vs. Visual Retention")
ax1.set_ylim(0.4, 2.0)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left", framealpha=0.9)

# Annotate DFlash peak and MSD 7B monotonic rise
ax1.annotate(f"{dflash_metrics['dec_spds'][0]:.2f}x", xy=(0, dflash_metrics["dec_spds"][0]), xytext=(0, dflash_metrics["dec_spds"][0] + 0.08),
             ha="center", fontweight="bold", color="#1f77b4")
ax1.annotate(f"+66% rise\n({msd7b_metrics['dec_spds'][-1]:.2f}x)", xy=(4, msd7b_metrics["dec_spds"][-1]), xytext=(4, msd7b_metrics["dec_spds"][-1] + 0.12),
             ha="center", fontweight="bold", color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))

# Plot E2E Speedup
ax2.plot(x, dflash_metrics["e2e_spds"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3 (Qwen2.5-VL-3B)")
ax2.plot(x, msd7b_metrics["e2e_spds"], marker="s", linewidth=2.5, markersize=8, color="#d62728", label="MSD 7B Gốc (Qwen2-VL-7B)")
ax2.plot(x, msd3b_metrics["e2e_spds"], marker="^", linewidth=2.0, markersize=7, color="#ff7f0e", linestyle="--", label="MSD Depth 3 (Qwen2.5-VL-3B)")
ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Autoregressive Baseline (1.0x)")
ax2.set_xticks(x)
ax2.set_xticklabels(ret_labels)
ax2.set_xlabel("Suffix Visual Retention Level (%)")
ax2.set_ylabel("End-to-End Speedup (x)")
ax2.set_title("(b) End-to-End Speedup (Prefill + Decode)")
ax2.set_ylim(0.4, 1.8)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper left", framealpha=0.9)

ax2.annotate(f"{dflash_metrics['e2e_spds'][0]:.2f}x", xy=(0, dflash_metrics["e2e_spds"][0]), xytext=(0, dflash_metrics["e2e_spds"][0] + 0.07),
             ha="center", fontweight="bold", color="#1f77b4")
ax2.annotate(f"{msd7b_metrics['e2e_spds'][-1]:.2f}x", xy=(4, msd7b_metrics["e2e_spds"][-1]), xytext=(4, msd7b_metrics["e2e_spds"][-1] + 0.07),
             ha="center", fontweight="bold", color="#d62728")

plt.tight_layout()
fig1_path = FIG_DIR / "figure5_suffix_cut_speedup_comparison.png"
plt.savefig(fig1_path)
plt.close()
print("Saved:", fig1_path)

# ------------------------------------------------------------------------------
# Figure 2: Effective Acceptance Rate (tau_eff) & Inverted Scaling
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))
plt.plot(x, msd7b_metrics["taus"], marker="s", linewidth=3.0, markersize=9, color="#d62728", label="MSD 7B Gốc (Inverted Scaling: 2.37 -> 3.74)")
plt.plot(x, dflash_metrics["taus"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3 (Robust: ~2.20 - 2.26)")
plt.plot(x, msd3b_metrics["taus"], marker="^", linewidth=2.0, markersize=7, color="#ff7f0e", linestyle="--", label="MSD Depth 3 (Flat ~1.01)")

# Annotations for MSD 7B scaling
for i, (xv, yv) in enumerate(zip(x, msd7b_metrics["taus"])):
    plt.annotate(f"{yv:.2f}", xy=(xv, yv), xytext=(xv, yv + 0.12), ha="center", fontweight="bold", color="#d62728", fontsize=10)

for i, (xv, yv) in enumerate(zip(x, dflash_metrics["taus"])):
    plt.annotate(f"{yv:.2f}", xy=(xv, yv), xytext=(xv, yv - 0.22), ha="center", fontweight="bold", color="#1f77b4", fontsize=10)

plt.xticks(x, ret_labels)
plt.xlabel("Suffix Visual Retention Level (%)")
plt.ylabel(r"Effective Accepted Tokens per Step ($\tau_{\mathrm{eff}}$)")
plt.title(r"Speculative Acceptance Rate ($\tau_{\mathrm{eff}}$) vs. Suffix Visual Cut on VDC")
plt.ylim(0.5, 4.3)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper left", framealpha=0.9)

# Highlight box on Inverted Scaling
plt.text(2.5, 1.4, "Insight: Inverted Scaling in MSD 7B\nAs visual tokens drop from 3951 to 0,\nτ_eff surges by +57.8% (2.37 -> 3.74)!\nLess visual noise = Better language drafting.",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffe6e6", edgecolor="#d62728", alpha=0.8), fontsize=10)

plt.tight_layout()
fig2_path = FIG_DIR / "figure6_suffix_cut_tau_effective.png"
plt.savefig(fig2_path)
plt.close()
print("Saved:", fig2_path)

# ------------------------------------------------------------------------------
# Figure 3: Strict Lossless Rate (%) & Prefix Match Percentage
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Strict Lossless
bar_w = 0.25
ax1.bar(x - bar_w, msd7b_metrics["lossless_rates"], width=bar_w, color="#d62728", alpha=0.85, label="MSD 7B Gốc")
ax1.bar(x, dflash_metrics["lossless_rates"], width=bar_w, color="#1f77b4", alpha=0.85, label="DFlash Depth 3")
ax1.bar(x + bar_w, msd3b_metrics["lossless_rates"], width=bar_w, color="#ff7f0e", alpha=0.85, label="MSD Depth 3")

for i, val in enumerate(msd7b_metrics["lossless_rates"]):
    ax1.text(x[i] - bar_w, val + 1.5, f"{val:.0f}%", ha="center", fontweight="bold", fontsize=9, color="#900")
for i, val in enumerate(dflash_metrics["lossless_rates"]):
    ax1.text(x[i], val + 1.5, f"{val:.0f}%", ha="center", fontweight="bold", fontsize=9, color="#005580")

ax1.set_xticks(x)
ax1.set_xticklabels(ret_labels)
ax1.set_xlabel("Suffix Visual Retention Level (%)")
ax1.set_ylabel("Strict Lossless Exact Match Rate (%)")
ax1.set_title("(a) Strict Lossless Rate (100% Sequence Exact Match)")
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", framealpha=0.9)

# Prefix Match %
ax2.plot(x, msd7b_metrics["prefix_rates"], marker="s", linewidth=2.5, markersize=8, color="#d62728", label="MSD 7B Gốc")
ax2.plot(x, dflash_metrics["prefix_rates"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3")
ax2.plot(x, msd3b_metrics["prefix_rates"], marker="^", linewidth=2.0, markersize=7, color="#ff7f0e", linestyle="--", label="MSD Depth 3")

for i, val in enumerate(msd7b_metrics["prefix_rates"]):
    ax2.annotate(f"{val:.1f}%", xy=(x[i], val), xytext=(x[i], val + 3), ha="center", fontweight="bold", color="#d62728")
for i, val in enumerate(dflash_metrics["prefix_rates"]):
    ax2.annotate(f"{val:.1f}%", xy=(x[i], val), xytext=(x[i], val - 6), ha="center", fontweight="bold", color="#1f77b4")

ax2.set_xticks(x)
ax2.set_xticklabels(ret_labels)
ax2.set_xlabel("Suffix Visual Retention Level (%)")
ax2.set_ylabel("Average Lossless Prefix Match (%)")
ax2.set_title("(b) Average Lossless Prefix Length / Sequence Length")
ax2.set_ylim(0, 110)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="center right", framealpha=0.9)

plt.tight_layout()
fig3_path = FIG_DIR / "figure7_suffix_cut_lossless_accuracy.png"
plt.savefig(fig3_path)
plt.close()
print("Saved:", fig3_path)

# ------------------------------------------------------------------------------
# Figure 4: Efficiency Frontier: Tau vs. Visual Tokens in KV Cache
# ------------------------------------------------------------------------------
plt.figure(figsize=(9, 5.5))

# Plot MSD 7B trajectory
plt.plot(msd7b_metrics["vis_tokens"], msd7b_metrics["taus"], marker="s", markersize=10, linewidth=2.5, color="#d62728", label="MSD 7B Gốc Trajectory")
for ret, vt, tau in zip(ret_levels, msd7b_metrics["vis_tokens"], msd7b_metrics["taus"]):
    offset_y = 0.12 if ret in [100.0, 50.0, 0.0] else -0.22
    plt.annotate(f"{int(ret)}% ({int(vt)} toks)\nτ={tau:.2f}", xy=(vt, tau), xytext=(vt, tau + offset_y),
                 ha="center", fontsize=9, fontweight="bold", color="#d62728",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d62728", alpha=0.8))

# Plot DFlash trajectory
plt.plot(dflash_metrics["vis_tokens"], dflash_metrics["taus"], marker="o", markersize=9, linewidth=2.5, color="#1f77b4", label="DFlash Depth 3 Trajectory")
for ret, vt, tau in zip([100.0, 0.0], [dflash_metrics["vis_tokens"][0], dflash_metrics["vis_tokens"][-1]], [dflash_metrics["taus"][0], dflash_metrics["taus"][-1]]):
    plt.annotate(f"{int(ret)}% ({int(vt)} toks)\nτ={tau:.2f}", xy=(vt, tau), xytext=(vt, tau - 0.28),
                 ha="center", fontsize=9, fontweight="bold", color="#1f77b4",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1f77b4", alpha=0.8))

plt.xlabel("Number of Visual Tokens in Draft KV Cache")
plt.ylabel(r"Effective Speculative Rate ($\tau_{\mathrm{eff}}$)")
plt.title(r"Efficiency Frontier: Pruning Visual KV Footprint vs. $\tau_{\mathrm{eff}}$")
plt.xlim(-200, 4300)
plt.ylim(1.5, 4.3)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", framealpha=0.9)

# Arrow indicating compression direction
plt.annotate("Visual Cache Pruning Direction\n(Zero Memory Overhead at 0%)",
             xy=(200, 3.8), xytext=(2200, 3.9),
             arrowprops=dict(facecolor="black", shrink=0.05, width=2, headwidth=8),
             fontsize=10, fontweight="bold", ha="center")

plt.tight_layout()
fig4_path = FIG_DIR / "figure8_suffix_cut_tradeoff_frontier.png"
plt.savefig(fig4_path)
plt.close()
print("Saved:", fig4_path)
print("All 4 figures generated successfully!")
