#!/usr/bin/env python3
"""Generate scientific comparison figures for Left-to-Right Text Cut (Zero Visual) on VDC dataset.

Plots:
- Figure 1: Speedup Comparison (Decode Speedup & E2E Speedup)
- Figure 2: tau_effective Comparison (Normal Scaling vs Architecture Invariance)
- Figure 3: Strict Lossless Accuracy & Prefix Match Length
- Figure 4: Trade-off Frontier (Text Tokens in Context vs tau_effective)
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid")
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

FIG_DIR = Path("markdown/23_10_26/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
def load_and_group(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    by_ret = defaultdict(list)
    for r in rows:
        by_ret[float(r["retention_percentage"])].append(r)
    return by_ret

dflash_data = load_and_group("results/text_cut_zero_visual/dflash/dflash_text_cut_vdc.jsonl")
msd7b_data = load_and_group("results/text_cut_zero_visual/msd_7b/msd_text_cut_vdc.jsonl")

ret_levels = [100.0, 75.0, 50.0, 25.0, 0.0]
ret_labels = ["100%\n(Full Text)", "75%\nSuffix", "50%\nSuffix", "25%\nSuffix", "0%\n(1 Token)"]

def compute_metrics(group_dict):
    dec_spds = []
    e2e_spds = []
    taus = []
    lossless_rates = []
    prefix_rates = []
    txt_tokens = []

    for ret in ret_levels:
        g = group_dict[ret]
        n = len(g)
        dec_spds.append(sum(r.get("decode_speedup", 1.0) for r in g) / n)
        e2e_spds.append(sum(r.get("end_to_end_speedup", 1.0) for r in g) / n)
        txt_tokens.append(sum(r.get("actual_text_tokens", 0) for r in g) / n)
        
        # Tau
        t_list = []
        for r in g:
            if "tau_effective" in r and r["tau_effective"] is not None:
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
        "txt_tokens": txt_tokens,
    }

dflash_metrics = compute_metrics(dflash_data)
msd7b_metrics = compute_metrics(msd7b_data)

x = np.arange(len(ret_levels))

# ------------------------------------------------------------------------------
# Figure 1: Speedup Comparison
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot Decode Speedup
ax1.plot(x, dflash_metrics["dec_spds"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3 (Qwen2.5-VL-3B)")
ax1.plot(x, msd7b_metrics["dec_spds"], marker="s", linewidth=2.5, markersize=8, color="#d62728", label="MSD 7B Gốc (Qwen2-VL-7B)")
ax1.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Autoregressive Baseline (1.0x)")
ax1.set_xticks(x)
ax1.set_xticklabels(ret_labels)
ax1.set_xlabel("Text Retention Level (Visual = 0%)")
ax1.set_ylabel("Decode Speedup (x)")
ax1.set_title("(a) Pure Decode Speedup vs. Text Suffix Retention")
ax1.set_ylim(0.4, 2.0)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left", framealpha=0.9)

ax1.annotate(f"{dflash_metrics['dec_spds'][0]:.2f}x (Stable)", xy=(0, dflash_metrics["dec_spds"][0]), xytext=(0.5, dflash_metrics["dec_spds"][0] + 0.12),
             ha="center", fontweight="bold", color="#1f77b4")
ax1.annotate(f"{msd7b_metrics['dec_spds'][0]:.2f}x", xy=(0, msd7b_metrics["dec_spds"][0]), xytext=(0, msd7b_metrics["dec_spds"][0] + 0.08),
             ha="center", fontweight="bold", color="#d62728")
ax1.annotate(f"{msd7b_metrics['dec_spds'][-1]:.2f}x", xy=(4, msd7b_metrics["dec_spds"][-1]), xytext=(4, msd7b_metrics["dec_spds"][-1] - 0.09),
             ha="center", fontweight="bold", color="#d62728")

# Plot E2E Speedup
ax2.plot(x, dflash_metrics["e2e_spds"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3 (Qwen2.5-VL-3B)")
ax2.plot(x, msd7b_metrics["e2e_spds"], marker="s", linewidth=2.5, markersize=8, color="#d62728", label="MSD 7B Gốc (Qwen2-VL-7B)")
ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Autoregressive Baseline (1.0x)")
ax2.set_xticks(x)
ax2.set_xticklabels(ret_labels)
ax2.set_xlabel("Text Retention Level (Visual = 0%)")
ax2.set_ylabel("End-to-End Speedup (x)")
ax2.set_title("(b) End-to-End Speedup (Prefill + Decode)")
ax2.set_ylim(0.4, 1.8)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper left", framealpha=0.9)

ax2.annotate(f"{dflash_metrics['e2e_spds'][0]:.2f}x", xy=(0, dflash_metrics["e2e_spds"][0]), xytext=(0, dflash_metrics["e2e_spds"][0] + 0.07),
             ha="center", fontweight="bold", color="#1f77b4")
ax2.annotate(f"{msd7b_metrics['e2e_spds'][0]:.2f}x", xy=(0, msd7b_metrics["e2e_spds"][0]), xytext=(0, msd7b_metrics["e2e_spds"][0] + 0.07),
             ha="center", fontweight="bold", color="#d62728")

plt.tight_layout()
fig1_path = FIG_DIR / "figure1_text_cut_speedup_comparison.png"
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# ------------------------------------------------------------------------------
# Figure 2: tau_effective Comparison
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(x, dflash_metrics["taus"], marker="o", linewidth=3.0, markersize=9, color="#1f77b4", label="DFlash Depth 3 (Parallel Draft Block)")
ax.plot(x, msd7b_metrics["taus"], marker="s", linewidth=3.0, markersize=9, color="#d62728", label="MSD 7B Gốc (Tree-based Draft Head)")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Autoregressive Baseline (tau = 1.0)")

ax.set_xticks(x)
ax.set_xticklabels(ret_labels)
ax.set_xlabel("Text Retention Level (Visual = 0%)")
ax.set_ylabel(r"Effective Acceptance Rate $\tau_{\mathrm{eff}}$ (tokens/round)")
ax.set_title(r"Speculative Acceptance Rate $\tau_{\mathrm{eff}}$ under Text Trimming")
ax.set_ylim(0.8, 4.2)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", framealpha=0.95)

# Annotate values
for i in range(len(x)):
    ax.annotate(f"{dflash_metrics['taus'][i]:.2f}", xy=(x[i], dflash_metrics["taus"][i]), 
                xytext=(x[i], dflash_metrics["taus"][i] - 0.22), ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
    ax.annotate(f"{msd7b_metrics['taus'][i]:.2f}", xy=(x[i], msd7b_metrics["taus"][i]), 
                xytext=(x[i], msd7b_metrics["taus"][i] + 0.15), ha="center", fontsize=10, fontweight="bold", color="#d62728")

plt.tight_layout()
fig2_path = FIG_DIR / "figure2_text_cut_tau_effective.png"
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")

# ------------------------------------------------------------------------------
# Figure 3: Strict Lossless & Prefix Match Accuracy
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Strict Lossless
bar_width = 0.35
ax1.bar(x - bar_width/2, dflash_metrics["lossless_rates"], width=bar_width, color="#1f77b4", alpha=0.85, label="DFlash Depth 3")
ax1.bar(x + bar_width/2, msd7b_metrics["lossless_rates"], width=bar_width, color="#d62728", alpha=0.85, label="MSD 7B Gốc")
ax1.set_xticks(x)
ax1.set_xticklabels(ret_labels)
ax1.set_xlabel("Text Retention Level (Visual = 0%)")
ax1.set_ylabel("Strict Lossless Accuracy (%)")
ax1.set_title("(a) Strict Lossless Rate (Exact 100% Token Match)")
ax1.set_ylim(0, 105)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left")

for i in range(len(x)):
    ax1.annotate(f"{dflash_metrics['lossless_rates'][i]:.1f}%", xy=(x[i] - bar_width/2, dflash_metrics["lossless_rates"][i]), 
                 xytext=(x[i] - bar_width/2, dflash_metrics["lossless_rates"][i] + 2), ha="center", fontsize=9, fontweight="bold")
    ax1.annotate(f"{msd7b_metrics['lossless_rates'][i]:.1f}%", xy=(x[i] + bar_width/2, msd7b_metrics["lossless_rates"][i]), 
                 xytext=(x[i] + bar_width/2, msd7b_metrics["lossless_rates"][i] + 2), ha="center", fontsize=9, fontweight="bold", color="#d62728")

# Prefix Match
ax2.plot(x, dflash_metrics["prefix_rates"], marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="DFlash Depth 3")
ax2.plot(x, msd7b_metrics["prefix_rates"], marker="s", linewidth=2.5, markersize=8, color="#d62728", label="MSD 7B Gốc")
ax2.set_xticks(x)
ax2.set_xticklabels(ret_labels)
ax2.set_xlabel("Text Retention Level (Visual = 0%)")
ax2.set_ylabel("Mean Common Prefix Match (%)")
ax2.set_title("(b) Lossless Prefix Match Length vs. Target")
ax2.set_ylim(30, 105)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="lower left")

for i in range(len(x)):
    ax2.annotate(f"{dflash_metrics['prefix_rates'][i]:.1f}%", xy=(x[i], dflash_metrics["prefix_rates"][i]), 
                 xytext=(x[i], dflash_metrics["prefix_rates"][i] - 4), ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
    ax2.annotate(f"{msd7b_metrics['prefix_rates'][i]:.1f}%", xy=(x[i], msd7b_metrics["prefix_rates"][i]), 
                 xytext=(x[i], msd7b_metrics["prefix_rates"][i] + 2), ha="center", fontsize=9, fontweight="bold", color="#d62728")

plt.tight_layout()
fig3_path = FIG_DIR / "figure3_text_cut_lossless_accuracy.png"
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")

# ------------------------------------------------------------------------------
# Figure 4: Trade-off Frontier (Text Tokens vs tau_effective)
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(dflash_metrics["txt_tokens"], dflash_metrics["taus"], marker="o", linewidth=2.5, markersize=9, color="#1f77b4", label="DFlash Depth 3")
ax.plot(msd7b_metrics["txt_tokens"], msd7b_metrics["taus"], marker="s", linewidth=2.5, markersize=9, color="#d62728", label="MSD 7B Gốc")

for i in range(len(ret_levels)):
    ax.annotate(f"{int(ret_levels[i])}% ({dflash_metrics['txt_tokens'][i]:.0f} tok)", 
                xy=(dflash_metrics["txt_tokens"][i], dflash_metrics["taus"][i]),
                xytext=(dflash_metrics["txt_tokens"][i], dflash_metrics["taus"][i] - 0.12),
                ha="center", fontsize=9, color="#1f77b4")
    ax.annotate(f"{int(ret_levels[i])}% ({msd7b_metrics['txt_tokens'][i]:.0f} tok)", 
                xy=(msd7b_metrics["txt_tokens"][i], msd7b_metrics["taus"][i]),
                xytext=(msd7b_metrics["txt_tokens"][i], msd7b_metrics["taus"][i] + 0.10),
                ha="center", fontsize=9, color="#d62728", fontweight="bold")

ax.set_xlabel("Number of Kept Text Tokens in Drafter KV Cache (Visual = 0)")
ax.set_ylabel(r"Effective Acceptance Rate $\tau_{\mathrm{eff}}$ (tokens/round)")
ax.set_title("Context Efficiency Frontier: Text Token Footprint vs. Drafting Quality")
ax.set_ylim(1.5, 4.2)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="center right", framealpha=0.95)

plt.tight_layout()
fig4_path = FIG_DIR / "figure4_text_cut_tradeoff_frontier.png"
plt.savefig(fig4_path)
plt.close()
print(f"Saved: {fig4_path}")

print("All 4 figures generated successfully!")
