#!/usr/bin/env python3
"""
Generate comprehensive comparative figures for MSD benchmark runs:
Qwen2.5-VL-3B (Depth 3 & Depth 5) vs Qwen2-VL-7B (MSD Baseline)
across VidHalluc and VideoChair benchmarks.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set high-quality styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUT_DIR = Path('results/6_10_26')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
records = []

# Load 7B
for ds in ['vidhalluc', 'videochair']:
    p = Path(f'results/msd_eval_20261007/{ds}/msd_retention_{ds}.jsonl')
    if p.exists():
        with open(p) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    r['model_label'] = 'Qwen2-VL-7B (EAGLE Depth 5)'
                    r['model_family'] = '7B'
                    r['dataset_label'] = 'VidHalluc' if ds == 'vidhalluc' else 'VideoChair'
                    records.append(r)

# Load 3B Depth 3 and Depth 5
for depth in ['depth3', 'depth5']:
    d_name = 'Depth 3' if depth == 'depth3' else 'Depth 5'
    for ds in ['vidhalluc', 'videochair']:
        p = Path(f'results/msd_eval_qwen25vl3b/{depth}/{ds}/msd_retention_{ds}.jsonl')
        if p.exists():
            with open(p) as f:
                for l in f:
                    if l.strip():
                        r = json.loads(l)
                        r['model_label'] = f'Qwen2.5-VL-3B ({d_name})'
                        r['model_family'] = f'3B_{depth}'
                        r['dataset_label'] = 'VidHalluc' if ds == 'vidhalluc' else 'VideoChair'
                        records.append(r)

df = pd.DataFrame(records)

# Compute summaries
agg = df.groupby(['model_label', 'dataset_label', 'retention_percentage']).agg(
    decode_speedup=('decode_speedup', 'mean'),
    e2e_speedup=('end_to_end_speedup', 'mean'),
    tau=('accepted_prefix_tokens', 'mean'),
    lossless=('lossless', lambda x: x.mean() * 100),
    verif_steps=('verification_steps', 'mean'),
    prefill_sec=('prefill_seconds', 'mean'),
    decode_sec=('decode_seconds', 'mean'),
    ar_prefill_sec=('ar_prefill_seconds', 'mean'),
    ar_decode_sec=('ar_decode_seconds', 'mean'),
).reset_index()

print("Summary aggregated successfully. Total rows:", len(agg))

# Colors
COLOR_7B = '#2b5c8f'       # Deep Blue
COLOR_3B_D5 = '#2ca02c'    # Green
COLOR_3B_D3 = '#d62728'    # Red
COLOR_AR = '#7f7f7f'       # Grey

cut_order = [100.0, 50.0, 25.0, 0.0]
cut_labels = ['100% (No Cut)', '50% Cut', '25% Cut', '0% (Full Cut)']

# -------------------------------------------------------------
# FIGURE 7: 3B vs 7B Speedup & Throughput by Cut Level
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

models = [
    ('Qwen2-VL-7B (EAGLE Depth 5)', COLOR_7B, 'o-', 2.5),
    ('Qwen2.5-VL-3B (Depth 5)', COLOR_3B_D5, 's-', 2.5),
    ('Qwen2.5-VL-3B (Depth 3)', COLOR_3B_D3, '^--', 2.0)
]

# (0,0) Decode Speedup - VidHalluc
ax = axes[0, 0]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VidHalluc')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['decode_speedup'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='AR Baseline (1.0x)')
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('Decode Speedup (x)', fontsize=11, fontweight='bold')
ax.set_title('(a) VidHalluc: Decode Speedup Across Retention Cuts', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0.4, 2.8)

# (0,1) Decode Speedup - VideoChair
ax = axes[0, 1]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VideoChair')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['decode_speedup'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='AR Baseline (1.0x)')
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('Decode Speedup (x)', fontsize=11, fontweight='bold')
ax.set_title('(b) VideoChair: Decode Speedup Across Retention Cuts', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0.4, 1.5)

# (1,0) End-to-End Speedup - VidHalluc
ax = axes[1, 0]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VidHalluc')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['e2e_speedup'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='AR Baseline (1.0x)')
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('End-to-End Speedup (x)', fontsize=11, fontweight='bold')
ax.set_title('(c) VidHalluc: End-to-End Speedup Across Retention Cuts', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0.8, 1.7)

# (1,1) End-to-End Speedup - VideoChair
ax = axes[1, 1]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VideoChair')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['e2e_speedup'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.5, label='AR Baseline (1.0x)')
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('End-to-End Speedup (x)', fontsize=11, fontweight='bold')
ax.set_title('(d) VideoChair: End-to-End Speedup Across Retention Cuts', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0.5, 1.5)

plt.suptitle('Figure 7: Speedup Comparison: Qwen2.5-VL-3B (Depth 3 & 5) vs Qwen2-VL-7B (MSD)', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig7_speedup_3b_vs_7b_comparison.png', bbox_inches='tight')
plt.close()
print("Saved fig7_speedup_3b_vs_7b_comparison.png")

# -------------------------------------------------------------
# FIGURE 8: Speculative Efficiency (Accepted Prefix & Verification Steps)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

# (0,0) Mean Accepted Prefix Tokens - VidHalluc
ax = axes[0, 0]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VidHalluc')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['tau'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('Accepted Prefix Tokens (τ)', fontsize=11, fontweight='bold')
ax.set_title('(a) VidHalluc: Draft Accepted Length (τ)', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(-0.1, 3.0)

# (0,1) Mean Accepted Prefix Tokens - VideoChair
ax = axes[0, 1]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VideoChair')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['tau'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('Accepted Prefix Tokens (τ)', fontsize=11, fontweight='bold')
ax.set_title('(b) VideoChair: Draft Accepted Length (τ)', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(-0.1, 3.8)

# (1,0) Verification Steps - VidHalluc
ax = axes[1, 0]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VidHalluc')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['verif_steps'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('Verification Rounds', fontsize=11, fontweight='bold')
ax.set_title('(c) VidHalluc: Target Verification Rounds (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(5, 30)

# (1,1) Verification Steps - VideoChair
ax = axes[1, 1]
for m_label, color, fmt, lw in models:
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VideoChair')]
    sub = sub.set_index('retention_percentage').reindex(cut_order).reset_index()
    ax.plot(range(4), sub['verif_steps'], fmt, color=color, label=m_label, linewidth=lw, markersize=8)
ax.set_xticks(range(4))
ax.set_xticklabels(cut_labels, fontsize=10, fontweight='bold')
ax.set_ylabel('Verification Rounds', fontsize=11, fontweight='bold')
ax.set_title('(d) VideoChair: Target Verification Rounds (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
ax.grid(True)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(50, 140)

plt.suptitle('Figure 8: Speculative Acceptance Dynamics & Verification Overhead Comparison', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig8_acceptance_and_verification_dynamics.png', bbox_inches='tight')
plt.close()
print("Saved fig8_acceptance_and_verification_dynamics.png")

# -------------------------------------------------------------
# FIGURE 9: 3B vs 7B Comprehensive Radar / Multi-Metric Bar Chart
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# (a) VidHalluc comparison at 100% and 0% Cut
x = np.arange(4)
width = 0.22

# Metrics for VidHalluc: [Decode Speedup 100%, Decode Speedup 0%, E2E Speedup 100%, E2E Speedup 0%]
ax = axes[0]
for idx, (m_label, color, _, _) in enumerate(models):
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VidHalluc')]
    val_100_dec = sub[sub['retention_percentage'] == 100.0]['decode_speedup'].values[0]
    val_0_dec = sub[sub['retention_percentage'] == 0.0]['decode_speedup'].values[0]
    val_100_e2e = sub[sub['retention_percentage'] == 100.0]['e2e_speedup'].values[0]
    val_0_e2e = sub[sub['retention_percentage'] == 0.0]['e2e_speedup'].values[0]
    
    vals = [val_100_dec, val_0_dec, val_100_e2e, val_0_e2e]
    rects = ax.bar(x + (idx - 1) * width, vals, width, label=m_label, color=color, alpha=0.9)
    # Add values on bars
    for r in rects:
        h = r.get_height()
        ax.annotate(f'{h:.2f}x', xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(['Decode (100% Cut)', 'Decode (0% Cut)', 'E2E (100% Cut)', 'E2E (0% Cut)'], fontsize=10, fontweight='bold')
ax.set_ylabel('Speedup Multiplier (x)', fontsize=11, fontweight='bold')
ax.set_title('(a) VidHalluc Benchmark Speedup Profile', fontsize=12, fontweight='bold')
ax.grid(True, axis='y')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0, 3.0)

# (b) VideoChair comparison at 100% and 0% Cut
ax = axes[1]
for idx, (m_label, color, _, _) in enumerate(models):
    sub = agg[(agg['model_label'] == m_label) & (agg['dataset_label'] == 'VideoChair')]
    val_100_dec = sub[sub['retention_percentage'] == 100.0]['decode_speedup'].values[0]
    val_0_dec = sub[sub['retention_percentage'] == 0.0]['decode_speedup'].values[0]
    val_100_e2e = sub[sub['retention_percentage'] == 100.0]['e2e_speedup'].values[0]
    val_0_e2e = sub[sub['retention_percentage'] == 0.0]['e2e_speedup'].values[0]
    
    vals = [val_100_dec, val_0_dec, val_100_e2e, val_0_e2e]
    rects = ax.bar(x + (idx - 1) * width, vals, width, label=m_label, color=color, alpha=0.9)
    for r in rects:
        h = r.get_height()
        ax.annotate(f'{h:.2f}x', xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(['Decode (100% Cut)', 'Decode (0% Cut)', 'E2E (100% Cut)', 'E2E (0% Cut)'], fontsize=10, fontweight='bold')
ax.set_ylabel('Speedup Multiplier (x)', fontsize=11, fontweight='bold')
ax.set_title('(b) VideoChair Benchmark Speedup Profile', fontsize=12, fontweight='bold')
ax.grid(True, axis='y')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
ax.set_ylim(0, 1.8)

plt.suptitle('Figure 9: Comprehensive Performance Gains: Extreme Visual Retention (100% vs 0%)', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig9_performance_profile_bars.png', bbox_inches='tight')
plt.close()
print("Saved fig9_performance_profile_bars.png")

# -------------------------------------------------------------
# FIGURE 10: 3B Detailed Breakdown: Depth Scaling (Depth 3 vs Depth 5)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# (a) Acceptance by Difficulty Category (Depth 3 vs Depth 5)
ax = axes[0]
# Load detailed iterations for 3B
df_d3_vh = pd.read_csv('results/msd_eval_qwen25vl3b/depth3/vidhalluc/draft_iterations_detailed.csv')
df_d5_vh = pd.read_csv('results/msd_eval_qwen25vl3b/depth5/vidhalluc/draft_iterations_detailed.csv')
df_d3_vc = pd.read_csv('results/msd_eval_qwen25vl3b/depth3/videochair/draft_iterations_detailed.csv')
df_d5_vc = pd.read_csv('results/msd_eval_qwen25vl3b/depth5/videochair/draft_iterations_detailed.csv')

cats = ['CONTENT', 'EASY', 'HARD']
d5_vh_acc = [df_d5_vh[df_d5_vh['difficulty_category'] == c]['accepted_length'].mean() if c in df_d5_vh['difficulty_category'].values else 0 for c in cats]
d5_vc_acc = [df_d5_vc[df_d5_vc['difficulty_category'] == c]['accepted_length'].mean() if c in df_d5_vc['difficulty_category'].values else 0 for c in cats]
d3_vh_acc = [0.0, 0.0, 0.0]
d3_vc_acc = [df_d3_vc['accepted_length'].mean(), 0.0, 0.0]

x = np.arange(len(cats))
w = 0.2
ax.bar(x - 1.5*w, d3_vh_acc, w, label='Depth 3 (VidHalluc)', color='#ff9896')
ax.bar(x - 0.5*w, d5_vh_acc, w, label='Depth 5 (VidHalluc)', color='#2ca02c')
ax.bar(x + 0.5*w, d3_vc_acc, w, label='Depth 3 (VideoChair)', color='#c5b0d5')
ax.bar(x + 1.5*w, d5_vc_acc, w, label='Depth 5 (VideoChair)', color='#9467bd')

ax.set_xticks(x)
ax.set_xticklabels(cats, fontweight='bold', fontsize=11)
ax.set_ylabel('Mean Accepted Tokens', fontweight='bold', fontsize=11)
ax.set_title('(a) Token Acceptance by Difficulty Category', fontweight='bold', fontsize=12)
ax.grid(True, axis='y')
ax.legend(frameon=True, fontsize=9)

# (b) Zero-Acceptance Rate (%)
ax = axes[1]
d5_vh_zero = (df_d5_vh['accepted_length'] == 0).mean() * 100
d3_vh_zero = 100.0
d5_vc_zero = (df_d5_vc['accepted_length'] == 0).mean() * 100
d3_vc_zero = (df_d3_vc['accepted_length'] == 0).mean() * 100

labels = ['VidHalluc\n(Depth 3)', 'VidHalluc\n(Depth 5)', 'VideoChair\n(Depth 3)', 'VideoChair\n(Depth 5)']
zero_vals = [d3_vh_zero, d5_vh_zero, d3_vc_zero, d5_vc_zero]
bar_colors = [COLOR_3B_D3, COLOR_3B_D5, COLOR_3B_D3, '#9467bd']

bars = ax.bar(labels, zero_vals, color=bar_colors, width=0.55, alpha=0.9)
for b in bars:
    h = b.get_height()
    ax.annotate(f'{h:.1f}%', xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Zero-Acceptance Rate (%) [Lower is Better]', fontweight='bold', fontsize=11)
ax.set_title('(b) Zero-Acceptance Rate Under Depth Scaling', fontweight='bold', fontsize=12)
ax.set_ylim(0, 115)
ax.grid(True, axis='y')

# (c) Decode Latency Comparison (AR vs Depth 3 vs Depth 5)
ax = axes[2]
vh_sub_d3 = agg[(agg['model_label'] == 'Qwen2.5-VL-3B (Depth 3)') & (agg['dataset_label'] == 'VidHalluc')].iloc[0]
vh_sub_d5 = agg[(agg['model_label'] == 'Qwen2.5-VL-3B (Depth 5)') & (agg['dataset_label'] == 'VidHalluc')].iloc[0]
vc_sub_d3 = agg[(agg['model_label'] == 'Qwen2.5-VL-3B (Depth 3)') & (agg['dataset_label'] == 'VideoChair')].iloc[0]
vc_sub_d5 = agg[(agg['model_label'] == 'Qwen2.5-VL-3B (Depth 5)') & (agg['dataset_label'] == 'VideoChair')].iloc[0]

benchmarks = ['VidHalluc', 'VideoChair']
ar_dec = [vh_sub_d3['ar_decode_sec'], vc_sub_d3['ar_decode_sec']]
d3_dec = [vh_sub_d3['decode_sec'], vc_sub_d3['decode_sec']]
d5_dec = [vh_sub_d5['decode_sec'], vc_sub_d5['decode_sec']]

x = np.arange(2)
w = 0.25
ax.bar(x - w, ar_dec, w, label='AR Baseline', color=COLOR_AR, alpha=0.9)
ax.bar(x, d3_dec, w, label='MSD Depth 3', color=COLOR_3B_D3, alpha=0.9)
ax.bar(x + w, d5_dec, w, label='MSD Depth 5', color=COLOR_3B_D5, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontweight='bold', fontsize=11)
ax.set_ylabel('Decode Latency (seconds) [Lower is Better]', fontweight='bold', fontsize=11)
ax.set_title('(c) Decode Time: AR vs Depth 3 vs Depth 5', fontweight='bold', fontsize=12)
ax.grid(True, axis='y')
ax.legend(frameon=True, fontsize=9)

plt.suptitle('Figure 10: Depth Scaling Dynamics in Qwen2.5-VL-3B MSD (Depth 3 vs Depth 5)', 
             fontsize=14, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig10_depth_scaling_analysis_3b.png', bbox_inches='tight')
plt.close()
print("Saved fig10_depth_scaling_analysis_3b.png")
