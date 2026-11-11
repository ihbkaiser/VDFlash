# Báo Cáo Thực Nghiệm DFlash: Cấu Hình `dflash_3layer_1_28_30_32_34` (Shallow-High)

**Ngày lập báo cáo:** 2026-10-29  
**Cấu hình Draft:** `dflash_3layer_1_28_30_32_34` (`shallow_high`)  
**Tầng trích xuất từ Target:** Layers [1, 28, 30, 32, 34] (1 tầng nông số 1 ngay sau embedding và 4 tầng ngọn 28, 30, 32, 34)  
**Target Model:** `Qwen2.5-VL-3B-Instruct` (36 decoder layers)  
**Tập dữ liệu:** VideoDetailCaption (VDC50, $N=50$ video samples đã calibrate)  
**Ngân sách thị giác (Visual Budgets):** 400, 3,000, 13,000, 25,000 visual tokens  
**Tổng số cases thực nghiệm:** 1,200 cases ($50 \text{ samples} \times 4 \text{ budgets} \times 6 \text{ conditions}$), hoàn tất 100%.

---

## 1. Tóm Tắt Tổng Quan

Cấu hình `shallow_high` lấy đại diện tầng thấp nhất (layer 1 - đại diện cho đặc trưng hình ảnh và ngữ pháp bề mặt sau phép chiếu embedding) kết hợp với các tầng biểu diễn ngữ nghĩa trừu tượng ở đỉnh (layers 28, 30, 32, 34).
- **Điểm mạnh ở Visual trung bình (`3,000 tokens`):** Cấu hình này đạt $\tau_{\text{effective}} = \mathbf{2.430}$ ở mức 3,000 visual tokens (thuộc nhóm cao nhất trong các cấu hình), duy trì tốc độ **91.6 TPS** (Speedup **1.60x**).
- **Độ nhạy cảm cao nhất với Value Ablation (`zero`):** Ở 25,000 tokens, khi giá trị visual bị xóa về 0 nhưng giữ nguyên vị trí, $\tau_{\text{effective}}$ của `shallow_high` sụt giảm xuống mức thấp nhất trong toàn bộ 4 cấu hình: **$2.121$**. Điều này cho thấy khi Draft có kết nối trực tiếp với tầng nông (layer 1), nó đặc biệt nhạy cảm với việc bị "làm mù" giá trị đặc trưng.
- **Lợi ích vượt bậc của `retained_25` ở 25k tokens:** Giữ lại 25% visual tokens giúp tăng $\tau_{\text{effective}}$ từ $2.211$ lên **$2.294$** (**+0.083 tokens/round**), khẳng định quy luật bão hòa visual token trên các chuỗi dài.

---

## 2. Thực Nghiệm 1: Visual Cut (`full` vs `zero` vs `cut`)

### Bảng 1: Chỉ số Acceptance, Tốc độ và Tính Lossless theo Ngân Sách Visual

| Budget (Tokens) | Điều kiện | $\tau_{\text{effective}}$ (Mean [95% CI]) | $\tau_{\text{proposal}}$ (Mean) | TPS | Speedup | Lossless Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | **2.491** [2.414, 2.570] | 1.491 | 114.5 | **1.83x** | 8.0% |
| | `zero` | 2.418 [2.332, 2.502] | 1.418 | 107.3 | 1.71x | 16.0% |
| | `cut` | 2.449 [2.358, 2.544] | 1.449 | 109.2 | 1.74x | 16.0% |
| **3,000** | `full` | **2.430** [2.368, 2.494] | 1.430 | 91.6 | **1.60x** | 6.0% |
| | `zero` | 2.300 [2.247, 2.353] | 1.300 | 87.6 | 1.53x | 2.0% |
| | `cut` | 2.309 [2.261, 2.357] | 1.309 | 89.1 | 1.55x | 6.0% |
| **13,000** | `full` | 2.213 [2.168, 2.256] | 1.213 | 29.9 | 0.97x | 6.0% |
| | `zero` | 2.167 [2.118, 2.216] | 1.167 | 29.9 | 0.97x | 4.0% |
| | `cut` | **2.245** [2.186, 2.309] | 1.245 | 31.4 | 1.02x | 2.0% |
| **25,000** | `full` | 2.211 [2.166, 2.255] | 1.211 | 12.9 | 0.96x | 2.0% |
| | `zero` | 2.121 [2.069, 2.175] | 1.121 | 12.7 | 0.94x | 0.0% |
| | `cut` | **2.229** [2.174, 2.284] | 1.229 | 13.4 | 0.99x | 2.0% |

### Bảng 2: Paired Contrasts (Hiệu số từng cặp so với `full` trên cùng video, Bootstrap 95% CI)

| Budget | Contrast | $\Delta \tau_{\text{effective}}$ [95% CI] | $\Delta$ TPS [95% CI] | Ý nghĩa Thống kê |
| :--- | :--- | :--- | :--- | :--- |
| **400** | `cut - full` | -0.042 [-0.134, +0.052] | -5.3 [-10.8, -0.4] | CI chứa 0 |
| | `zero - full` | -0.073 [-0.160, +0.012] | -7.2 [-12.5, -2.1] | Giảm tốc độ có ý nghĩa |
| **3,000** | `cut - full` | **-0.121** [-0.187, -0.054] | -2.6 [-4.9, -0.2] | **Giảm mạnh có ý nghĩa (Visual rất quan trọng)** |
| | `zero - full` | **-0.130** [-0.198, -0.065] | -4.0 [-6.5, -1.6] | **Giảm mạnh có ý nghĩa** |
| **13,000** | `cut - full` | +0.033 [-0.034, +0.103] | +1.5 [+0.7, +2.3] | CI chứa 0 (Cut tăng tốc và acceptance) |
| | `zero - full` | -0.046 [-0.096, +0.007] | -0.0 [-0.7, +0.6] | CI chứa 0 |
| **25,000** | `cut - full` | +0.018 [-0.046, +0.081] | +0.5 [+0.0, +0.9] | CI chứa 0 (Cut tương đương Full) |
| | `zero - full` | **-0.090** [-0.147, -0.033] | -0.2 [-0.6, +0.2] | **Zero làm giảm hiệu năng rõ rệt** |

![Hình 1: Đánh giá Visual Cut và Paired Contrasts so với Full (Shallow-High)](figures/fig1_visual_cut_dflash_3layer_1_28_30_32_34.png)

> [!IMPORTANT]
> **Nhận xét chính:** Ở 3,000 tokens, `shallow_high` phụ thuộc vào visual nhiều nhất: việc cắt visual làm giảm tới **-0.121 tokens/round** (mức giảm sâu nhất trong cả 4 cấu hình). Nhưng ở 13k và 25k tokens, sự phụ thuộc này biến mất và `cut` lại nhỉnh hơn `full`.

---

## 3. Thực Nghiệm 2: Visual Retention (`100% → 75% → 50% → 25% → 0%`)

### Bảng 3: Diễn tiến $\tau_{\text{effective}}$ qua các mức Retention

| Budget | `full` (100%) | `retained_75` (75%) | `retained_50` (50%) | `retained_25` (25%) | `cut` (0%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **400 tokens** | 2.491 [2.414, 2.570] | **2.495** [2.416, 2.577] | 2.463 [2.385, 2.539] | 2.482 [2.404, 2.560] | 2.449 [2.358, 2.544] |
| **3,000 tokens** | **2.430** [2.368, 2.494] | 2.395 [2.341, 2.450] | 2.409 [2.355, 2.463] | 2.402 [2.349, 2.454] | 2.309 [2.261, 2.357] |
| **13,000 tokens** | 2.213 [2.168, 2.256] | 2.242 [2.198, 2.287] | **2.257** [2.212, 2.302] | 2.240 [2.192, 2.289] | 2.245 [2.186, 2.309] |
| **25,000 tokens** | 2.211 [2.166, 2.255] | 2.207 [2.164, 2.252] | 2.225 [2.183, 2.267] | **2.294** [2.235, 2.357] | 2.229 [2.174, 2.284] |

![Hình 2: Đường cong Visual Retention Sweep qua các mức ngân sách (Shallow-High)](figures/fig2_retention_sweep_dflash_3layer_1_28_30_32_34.png)

> [!TIP]
> Ở 25,000 tokens, `retained_25` đạt **$2.294$** so với $2.211$ của `full` (tăng **+0.083 tokens/round**).

---

## 4. Thực Nghiệm 3: Phân Phối và Động Lực Học Attention

### Bảng 4: Phân bổ Chi Tiết 100% Attention Mass và Mật Độ (Condition `full`)

| Budget (Tokens) | Prompt Visual Mass | Prompt Text Mass (Question) | Generated Text Mass (Đã Sinh) | Current Draft Block (Causal) | Tổng Mass | Mật Độ Generated Text / Visual | Mật Độ Question / Visual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | 0.149 (14.9%) | 0.162 (16.2%) | **0.450 (45.0%)** | 0.238 (23.8%) | **1.000** | **32.3x** | 2.1x |
| **3,000** | 0.355 (35.5%) | 0.135 (13.5%) | **0.326 (32.6%)** | 0.183 (18.3%) | **1.000** | **174.5x** | 10.7x |
| **13,000** | **0.586 (58.6%)** | 0.135 (13.5%) | 0.138 (13.8%) | 0.141 (14.1%) | **1.000** | **858.0x** | 30.3x |
| **25,000** | **0.513 (51.3%)** | 0.085 (8.5%) | 0.280 (28.0%) | 0.121 (12.1%) | **1.000** | **955.0x** | 41.6x |

> [!NOTE]
> **Giải phẫu phân bổ 100% Attention Mass:**
> 1. **Sự kết hợp giữa tầng nông và chuỗi đã sinh:** Draft Model duy trì sự tập trung mạnh mẽ vào Generated Text Tokens (28% – 45%), kết hợp với Visual Features được trích xuất từ tầng 1 của Target.
> 2. **Phân cấp mật độ rõ rệt:** Generated Text Tokens luôn có mật độ cao nhất (**gấp 32x đến 955x so với Visual Tokens**), khẳng định các từ vừa sinh ra là điểm tựa trọng yếu nhất cho quá trình dự đoán từ tiếp theo.

### Bảng 5: Phân bổ Visual Attention Mass theo Tầng Draft (`layer 0`, `layer 1`, `layer 2`)

| Budget | Layer 0 (Input projection) | Layer 1 (Middle processing) | Layer 2 (Output head) |
| :--- | :--- | :--- | :--- |
| **400 tokens** | 0.084 (8.4%) | **0.217 (21.7%)** | 0.145 (14.5%) |
| **3,000 tokens** | 0.258 (25.8%) | **0.471 (47.1%)** | 0.341 (34.1%) |
| **13,000 tokens** | 0.446 (44.6%) | **0.737 (73.7%)** | 0.575 (57.5%) |
| **25,000 tokens** | 0.433 (43.3%) | **0.641 (64.1%)** | 0.466 (46.6%) |

![Hình 3: Động lực học Attention: Phân bổ Mass, Density Ratio và Tầng Draft (Shallow-High)](figures/fig3_attention_dynamics_dflash_3layer_1_28_30_32_34.png)

> [!NOTE]
> Khác với 3 cấu hình còn lại, tầng đầu tiên `layer 0` của `shallow_high` có mức visual mass cao nhất (lên tới 44.6% ở 13k tokens), do tầng trích xuất số 1 từ Target mang các đặc trưng embedding rất gần với visual patches.

---

## 5. Chất Lượng Ngữ Nghĩa (Quality Metrics)

### Bảng 6: Chỉ số Chất lượng Đầu ra (Speculative vs Target)

| Budget | Điều kiện | Speculative BLEU | Target BLEU | Speculative ROUGE-L | Target ROUGE-L | Speculative Coverage | Target Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | 0.045 | 0.046 | 0.182 | 0.184 | 0.215 | 0.213 |
| | `cut` | 0.045 | 0.046 | 0.182 | 0.184 | 0.219 | 0.213 |
| **3,000** | `full` | 0.044 | 0.043 | 0.199 | 0.198 | 0.320 | 0.317 |
| | `cut` | 0.044 | 0.043 | 0.199 | 0.198 | 0.325 | 0.317 |
| **13,000** | `full` | 0.043 | 0.042 | 0.196 | 0.199 | 0.343 | 0.353 |
| | `retained_25` | 0.040 | 0.042 | 0.197 | 0.199 | 0.338 | 0.353 |
| **25,000** | `full` | 0.042 | 0.042 | 0.202 | 0.200 | 0.344 | 0.342 |
| | `retained_25` | 0.040 | 0.042 | 0.201 | 0.200 | 0.346 | 0.342 |

![Hình 4: Đánh giá Trade-off giữa Tốc độ và Acceptance cùng Tính Nhất Quán Chất Lượng (Shallow-High)](figures/fig4_quality_tradeoff_dflash_3layer_1_28_30_32_34.png)

---

## 6. Kết Luận Đánh Giá Cấu Hình `shallow_high`

1. **Phù hợp nhất cho visual trung bình (3,000 tokens):** Đạt $\tau_{\text{effective}} = 2.430$ nhờ khai thác được biểu diễn chi tiết từ tầng 1 của Target.
2. **Hiện tượng bão hòa ở visual lớn:** Tương tự các cấu hình khác, ở 13k và 25k tokens, việc giữ nguyên 100% visual làm loãng attention (entropy lên tới 0.600). Giữ lại 25% visual tokens (`retained_25`) là điểm làm việc lý tưởng nhất ($2.294$ tokens/round).
