import matplotlib.pyplot as plt
import numpy as np

# Set overall style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
plt.subplots_adjust(hspace=0.28, wspace=0.22)

# Colors
c_depth3 = '#2b5c8f'
c_depth5 = '#d95f02'
c_depth1 = '#7570b3'
c_full = '#1b9e77'
c_zero = '#d95f02'
c_cut = '#7570b3'

# ==============================================================================
# SUBPLOT 1: Model Layer Scaling & Saturation (H3.2 VDC LLaVA68K)
# ==============================================================================
ax1 = axs[0, 0]
depths = [1, 3, 5]
tau_full = [1.987, 2.326, 2.328]
tau_zero = [1.960, 2.172, 2.221]
tau_cut = [1.957, 2.189, 2.231]

ci_full = [[1.987 - 1.915, 2.326 - 2.227, 2.328 - 2.229], [2.067 - 1.987, 2.435 - 2.326, 2.440 - 2.328]]
ci_zero = [[1.960 - 1.890, 2.172 - 2.079, 2.221 - 2.135], [2.027 - 1.960, 2.276 - 2.172, 2.315 - 2.221]]

ax1.errorbar(depths, tau_full, yerr=ci_full, fmt='-o', color=c_full, linewidth=2.5, markersize=8, capsize=5, label='Full Visual (τ)')
ax1.errorbar(depths, tau_zero, yerr=ci_zero, fmt='--s', color=c_zero, linewidth=2.2, markersize=8, capsize=5, label='Zero Visual (τ)')
ax1.plot(depths, tau_cut, ':^', color=c_cut, linewidth=2.0, markersize=7, label='Cut Visual (τ)')

ax1.annotate('Bão hòa tuyệt đối:\n+0.002 tokens (+0.08%)\nGiữa Depth 3 và 5', 
             xy=(4.0, 2.327), xytext=(3.2, 2.44),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
             fontsize=10.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="gray", lw=1))

ax1.annotate('Bước nhảy lớn (+17.1%):\n+0.339 tokens', 
             xy=(2.0, 2.15), xytext=(1.2, 2.25),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
             fontsize=10.5, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f2ff", ec="gray", lw=1))

ax1.set_xticks(depths)
ax1.set_xticklabels(['Depth 1\n(98M params)', 'Depth 3\n(252M params)', 'Depth 5\n(~400M params)'], fontsize=11, fontweight='bold')
ax1.set_ylabel('Chiều dài chấp nhận hiệu dụng (τ_eff)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Quy luật Bão hòa Kiến trúc Lớp Draft (H3.2 VDC LLaVA68K)\nDiminishing Returns từ Depth 3 lên Depth 5', fontsize=12, fontweight='bold', pad=10)
ax1.legend(loc='lower right', frameon=True, fontsize=10.5)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_ylim(1.85, 2.52)

# ==============================================================================
# SUBPLOT 2: Visual Sensitivity Contrast (Full - Zero) & Interaction
# ==============================================================================
ax2 = axs[0, 1]
x = np.arange(len(depths))
width = 0.35

diff_llava = [0.0275, 0.1536, 0.1075]
err_llava_low = [0.0275 - 0.0001, 0.1536 - 0.1036, 0.1075 - 0.0448]
err_llava_high = [0.0545 - 0.0275, 0.2059 - 0.1536, 0.1690 - 0.1075]

diff_sharegpt = [-0.0173, -0.0713, -0.0540]
err_sg_low = [-0.0173 - (-0.0318), -0.0713 - (-0.0997), -0.0540 - (-0.0826)]
err_sg_high = [-0.0043 - (-0.0173), -0.0412 - (-0.0713), -0.0261 - (-0.0540)]

rects1 = ax2.bar(x - width/2, diff_llava, width, yerr=[err_llava_low, err_llava_high], capsize=5, 
                 label='LLaVA68K (Multimodal Draft)', color='#2b5c8f', alpha=0.9, edgecolor='black')
rects2 = ax2.bar(x + width/2, diff_sharegpt, width, yerr=[err_sg_low, err_sg_high], capsize=5, 
                 label='ShareGPT68K (Text Draft)', color='#d95f02', alpha=0.9, edgecolor='black')

ax2.axhline(0, color='black', linewidth=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels(['Depth 1', 'Depth 3 (Đỉnh độ nhạy)', 'Depth 5 (Bão hòa)'], fontsize=11, fontweight='bold')
ax2.set_ylabel('Độ nhạy Visual Context: Full − Zero (tokens)', fontsize=11, fontweight='bold')
ax2.set_title('(B) Độ nhạy Visual Hidden States theo Số Lớp Draft (95% CI)\nDepth 3 đạt cực đại độ nhạy thị giác', fontsize=12, fontweight='bold', pad=10)
ax2.legend(loc='lower left', frameon=True, fontsize=10.5)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_ylim(-0.12, 0.25)

# Add values above/below bars
for rect in rects1:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02 if h >= 0 else h - 0.02,
             f'{h:+.3f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=9.5, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width()/2., h - 0.025,
             f'{h:+.3f}', ha='center', va='top', fontsize=9.5, fontweight='bold')

# ==============================================================================
# SUBPLOT 3: Speculative Collapse vs Recovery (3B MSD on VidHalluc & VideoChair)
# ==============================================================================
ax3 = axs[1, 0]
categories = ['VidHalluc\nDecode Speedup', 'VidHalluc\nE2E Speedup', 'VideoChair\nDecode Speedup', 'VideoChair\nE2E Speedup']
d3_vals = [1.625, 1.207, 0.567, 0.610]
d5_vals = [2.514, 1.406, 0.713, 0.745]

x_cat = np.arange(len(categories))
w_cat = 0.35

r1 = ax3.bar(x_cat - w_cat/2, d3_vals, w_cat, label='3B Draft Depth 3 (Sụp đổ τ ≈ 0)', color='#a6bddb', edgecolor='black')
r2 = ax3.bar(x_cat + w_cat/2, d5_vals, w_cat, label='3B Draft Depth 5 (Phục hồi τ > 0)', color='#02818a', edgecolor='black')

ax3.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Mốc cơ sở AR (1.0x)')
ax3.set_xticks(x_cat)
ax3.set_xticklabels(categories, fontsize=10.5, fontweight='bold')
ax3.set_ylabel('Hệ số tăng tốc (Speedup Factor)', fontsize=11, fontweight='bold')
ax3.set_title('(C) Hiệu năng Thực nghiệm 3B Decoupled MSD: Depth 3 vs Depth 5\nDepth 5 vượt trội về tốc độ và phá vỡ sụp đổ suy đoán', fontsize=12, fontweight='bold', pad=10)
ax3.legend(loc='upper right', frameon=True, fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.set_ylim(0, 3.0)

for rect in r1:
    h = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2., h + 0.05, f'{h:.2f}x', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
for rect in r2:
    h = rect.get_height()
    ax3.text(rect.get_x() + rect.get_width()/2., h + 0.05, f'{h:.2f}x', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#016450')

# ==============================================================================
# SUBPLOT 4: Tree Depth Survival Decay Curve (VidHalluc & VideoChair MSD)
# ==============================================================================
ax4 = axs[1, 1]
tree_depths = [1, 2, 3, 4, 5]

# Data from our 42,062 iterations
vid_100 = [65.20, 27.37, 7.27, 3.41, 0.94]
vid_0 = [93.83, 75.82, 47.30, 27.17, 18.69]
chair_100 = [82.46, 49.34, 16.18, 3.17, 0.44]
chair_0 = [96.50, 86.40, 69.02, 52.21, 42.19]

ax4.plot(tree_depths, vid_100, '-o', color='#e41a1c', linewidth=2.5, markersize=8, label='VidHalluc (100% Visual Context)')
ax4.plot(tree_depths, chair_100, '-s', color='#984ea3', linewidth=2.5, markersize=8, label='VideoChair (100% Visual Context)')
ax4.plot(tree_depths, vid_0, '--o', color='#377eb8', linewidth=2.0, markersize=7, alpha=0.7, label='VidHalluc (0% Visual - Pure Text)')
ax4.plot(tree_depths, chair_0, '--s', color='#4daf4a', linewidth=2.0, markersize=7, alpha=0.7, label='VideoChair (0% Visual - Pure Text)')

ax4.axvspan(3, 5, color='#fee0d2', alpha=0.4, label='Vùng lãng phí ở 100% Visual (Sống sót < 1%)')
ax4.annotate('Chỉ 0.44% - 0.94%\nsống sót đến Depth 5!\n(Lãng phí 99% tree mask FLOPs)', 
             xy=(5, 0.7), xytext=(3.6, 12),
             arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color='#990000',
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffcccc", ec="red", lw=1))

ax4.set_xticks(tree_depths)
ax4.set_xticklabels(['Depth 1', 'Depth 2', 'Depth 3', 'Depth 4', 'Depth 5'], fontsize=11, fontweight='bold')
ax4.set_ylabel('Tỷ lệ Sống sót của Chuỗi Token (Depth Survival Rate %)', fontsize=11, fontweight='bold')
ax4.set_title('(D) Động học Sống sót Cây Suy đoán EAGLE/MSD theo Độ sâu Tầng\nTại 100% Visual: Cây suy đoán bị cắt sớm trước tầng 4-5', fontsize=12, fontweight='bold', pad=10)
ax4.legend(loc='upper right', frameon=True, fontsize=9.5)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.set_ylim(-2, 105)

plt.tight_layout()
output_path = 'results/6_10_26/fig11_msd_depth3_vs_depth5_detailed_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Chart successfully saved to {output_path}")
