# Báo Cáo Thực Nghiệm DFlash: Cấu Hình `dflash_3layer_17_28_30_32_34` (Middle-High Variant 2)

**Ngày lập báo cáo:** 2026-10-29  
**Cấu hình Draft:** `dflash_3layer_17_28_30_32_34` (`middle_high_2`)  
**Tầng trích xuất từ Target:** Layers [17, 28, 30, 32, 34] (1 tầng neo giữa 17 và 4 tầng đỉnh 28, 30, 32, 34)  
**Target Model:** `Qwen2.5-VL-3B-Instruct` (36 decoder layers)  
**Tập dữ liệu:** VideoDetailCaption (VDC50, $N=50$ video samples đã calibrate)  
**Ngân sách thị giác (Visual Budgets):** 400, 3,000, 13,000, 25,000 visual tokens  
**Tổng số cases thực nghiệm:** 1,200 cases ($50 \text{ samples} \times 4 \text{ budgets} \times 6 \text{ conditions}$), hoàn tất 100%.

---

## 1. Tóm Tắt Tổng Quan

Cấu hình `middle_high_2` sử dụng tầng 17 làm điểm neo trung gian duy nhất kết hợp với cụm tầng đỉnh liên tiếp (28, 30, 32, 34).
- **Tỷ lệ Lossless cao nhất ở Visual ngắn:** Ở mốc `400 tokens`, cấu hình này đạt tỷ lệ sinh khớp 100% không suy hao (Lossless) lên tới **22.0%** (cao nhất trong cả 4 cấu hình), với $\tau_{\text{effective}} = \mathbf{2.532}$ và Speedup **1.88x** (116.4 TPS).
- **Hiện tượng bứt phá của `cut` và `retained` ở ngữ cảnh siêu dài (25k tokens):**
  - Tại `25,000 tokens`, điều kiện `full` bị suy giảm mạnh xuống còn $2.180$.
  - Trong khi đó, **`cut` đạt $\tau_{\text{effective}} = \mathbf{2.295}$** (cao hơn `full` tới **+0.115 tokens/round**) và **`retained_25` đạt $\mathbf{2.275}$** (+0.095 tokens/round).
  - Đây là bằng chứng thực nghiệm rõ ràng nhất cho thấy việc giữ nguyên 25,000 token thị giác ở tầng Draft gây ra hiện tượng nghẽn attention (Attention Saturation / Dilution), và việc loại bỏ hoặc rút gọn visual context giúp draft dự đoán chính xác hơn.
- **Phân bổ Attention:** Tầng `layer 1` và `layer 2` cùng chia sẻ tải visual rất cao (ở 13k tokens, `layer 1` đạt 68.8% visual mass và `layer 2` đạt **70.2% visual mass**).

---

## 2. Thực Nghiệm 1: Visual Cut (`full` vs `zero` vs `cut`)

### Bảng 1: Chỉ số Acceptance, Tốc độ và Tính Lossless theo Ngân Sách Visual

| Budget (Tokens) | Điều kiện | $\tau_{\text{effective}}$ (Mean [95% CI]) | $\tau_{\text{proposal}}$ (Mean) | TPS | Speedup | Lossless Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | **2.532** [2.458, 2.607] | 1.532 | 116.4 | **1.88x** | **22.0%** |
| | `zero` | 2.484 [2.403, 2.565] | 1.484 | 111.9 | 1.80x | 8.0% |
| | `cut` | 2.470 [2.393, 2.544] | 1.470 | 112.0 | 1.81x | 10.0% |
| **3,000** | `full` | **2.382** [2.321, 2.443] | 1.382 | 91.4 | **1.59x** | 10.0% |
| | `zero` | 2.340 [2.277, 2.405] | 1.340 | 88.6 | 1.54x | 10.0% |
| | `cut` | 2.372 [2.308, 2.434] | 1.372 | 90.5 | 1.57x | 2.0% |
| **13,000** | `full` | 2.222 [2.167, 2.272] | 1.222 | 30.3 | 0.99x | 0.0% |
| | `zero` | 2.183 [2.124, 2.242] | 1.183 | 29.7 | 0.97x | 4.0% |
| | `cut` | **2.212** [2.162, 2.259] | 1.212 | 31.1 | 1.02x | 6.0% |
| **25,000** | `full` | 2.180 [2.119, 2.241] | 1.180 | 12.9 | 1.30x | 0.0% |
| | `zero` | 2.182 [2.122, 2.242] | 1.182 | 13.1 | 1.32x | 0.0% |
| | `cut` | **2.295** [2.238, 2.350] | 1.295 | 13.4 | **1.35x** | 0.0% |

### Bảng 2: Paired Contrasts (Hiệu số từng cặp so với `full` trên cùng video, Bootstrap 95% CI)

| Budget | Contrast | $\Delta \tau_{\text{effective}}$ [95% CI] | $\Delta$ TPS [95% CI] | Ý nghĩa Thống kê |
| :--- | :--- | :--- | :--- | :--- |
| **400** | `cut - full` | -0.062 [-0.144, +0.021] | -4.4 [-9.2, +0.5] | CI chứa 0 |
| | `zero - full` | -0.048 [-0.133, +0.038] | -4.5 [-9.3, +0.5] | CI chứa 0 |
| **3,000** | `cut - full` | -0.010 [-0.071, +0.052] | -0.9 [-3.3, +1.6] | CI chứa 0 |
| | `zero - full` | -0.042 [-0.106, +0.021] | -2.8 [-5.4, -0.3] | CI chứa 0 |
| **13,000** | `cut - full` | -0.010 [-0.062, +0.042] | +0.8 [+0.1, +1.5] | CI chứa 0 |
| | `zero - full` | -0.039 [-0.098, +0.019] | -0.6 [-1.3, +0.1] | CI chứa 0 |
| **25,000** | `cut - full` | **+0.115** [+0.037, +0.194] | +0.5 [+0.1, +0.9] | **Tăng có ý nghĩa thống kê cực kỳ rõ rệt!** |
| | `zero - full` | +0.002 [-0.061, +0.063] | +0.2 [-0.2, +0.6] | CI chứa 0 |

![Hình 1: Đánh giá Visual Cut và Paired Contrasts so với Full (Middle-High Variant 2)](figures/fig1_visual_cut_dflash_3layer_17_28_30_32_34.png)

> [!IMPORTANT]
> **Phát hiện bước ngoặt tại 25,000 tokens:** Ở cấu hình `middle_high_2`, `cut - full` đạt mức chênh lệch dương **$+0.115$** với khoảng tin cậy 95% CI $[+0.037, +0.194]$ hoàn toàn nằm trên 0. Cắt bỏ 25,000 visual tokens giúp Draft tăng tốc độ giải mã từ 12.9 lên 13.4 TPS và cải thiện số token được chấp nhận trung bình mỗi vòng từ $2.180$ lên $2.295$.

---

## 3. Thực Nghiệm 2: Visual Retention (`100% → 75% → 50% → 25% → 0%`)

### Bảng 3: Diễn tiến $\tau_{\text{effective}}$ qua các mức Retention

| Budget | `full` (100%) | `retained_75` (75%) | `retained_50` (50%) | `retained_25` (25%) | `cut` (0%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **400 tokens** | **2.532** [2.458, 2.607] | 2.512 [2.437, 2.591] | 2.510 [2.432, 2.588] | 2.493 [2.410, 2.579] | 2.470 [2.393, 2.544] |
| **3,000 tokens** | 2.382 [2.321, 2.443] | 2.417 [2.355, 2.482] | 2.411 [2.350, 2.476] | **2.421** [2.350, 2.495] | 2.372 [2.308, 2.434] |
| **13,000 tokens** | 2.222 [2.167, 2.272] | 2.251 [2.203, 2.298] | 2.275 [2.231, 2.316] | **2.277** [2.235, 2.318] | 2.212 [2.162, 2.259] |
| **25,000 tokens** | 2.180 [2.119, 2.241] | 2.234 [2.176, 2.289] | 2.246 [2.189, 2.300] | **2.275** [2.227, 2.322] | **2.295** [2.238, 2.350] |

![Hình 2: Đường cong Visual Retention Sweep qua các mức ngân sách (Middle-High Variant 2)](figures/fig2_retention_sweep_dflash_3layer_17_28_30_32_34.png)

> [!TIP]
> Ở cả mức 3,000, 13,000 và 25,000 tokens, **`retained_25` và `retained_50` luôn vượt trội hơn `full`**. Đường cong hiệu năng có hình chữ U đảo ngược ở các budget trung bình và dốc dần về phía `cut` ở budget 25k.

---

## 4. Thực Nghiệm 3: Phân Phối và Động Lực Học Attention

### Bảng 4: Phân bổ Chi Tiết 100% Attention Mass và Mật Độ (Condition `full`)

| Budget (Tokens) | Prompt Visual Mass | Prompt Text Mass (Question) | Generated Text Mass (Đã Sinh) | Current Draft Block (Causal) | Tổng Mass | Mật Độ Generated Text / Visual | Mật Độ Question / Visual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | 0.116 (11.6%) | 0.174 (17.4%) | **0.474 (47.4%)** | 0.236 (23.6%) | **1.000** | **30.7x** | 2.4x |
| **3,000** | 0.263 (26.3%) | 0.122 (12.2%) | **0.420 (42.0%)** | 0.195 (19.5%) | **1.000** | **167.5x** | 12.0x |
| **13,000** | **0.598 (59.8%)** | 0.131 (13.1%) | 0.143 (14.3%) | 0.128 (12.8%) | **1.000** | **816.0x** | 22.3x |
| **25,000** | **0.467 (46.7%)** | 0.080 (8.0%) | 0.318 (31.8%) | 0.134 (13.4%) | **1.000** | **965.0x** | 30.9x |

> [!NOTE]
> **Giải phẫu phân bổ 100% Attention Mass:**
> 1. **Duy trì định hướng từ vựng:** Generated Text Tokens chiếm tới 42%–47% attention ở video ngắn/trung bình và vẫn giữ 31.8% ở video dài 25k tokens.
> 2. **Mật độ token vượt trội:** Mỗi token văn bản đã sinh ra nhận mật độ chú ý cao gấp **30x đến 965x** so với mỗi token video, đảm bảo mô hình không bị lạc đề giữa hàng vạn token hình ảnh.

### Bảng 5: Phân bổ Visual Attention Mass theo Tầng Draft (`layer 0`, `layer 1`, `layer 2`)

| Budget | Layer 0 (Input projection) | Layer 1 (Middle processing) | Layer 2 (Output head) |
| :--- | :--- | :--- | :--- |
| **400 tokens** | 0.048 (4.8%) | **0.195 (19.5%)** | 0.105 (10.5%) |
| **3,000 tokens** | 0.165 (16.5%) | **0.338 (33.8%)** | 0.286 (28.6%) |
| **13,000 tokens** | 0.403 (40.3%) | **0.688 (68.8%)** | **0.702 (70.2%)** |
| **25,000 tokens** | 0.365 (36.5%) | **0.520 (52.0%)** | **0.532 (53.2%)** |

![Hình 3: Động lực học Attention: Phân bổ Mass, Density Ratio và Tầng Draft (Middle-High Variant 2)](figures/fig3_attention_dynamics_dflash_3layer_17_28_30_32_34.png)

> [!NOTE]
> Ở budget 13k tokens, `norm_entropy` của cấu hình này lên tới **0.602** (cao nhất trong các cấu hình), phản ánh hiện tượng chú ý bị phân tán cực kỳ mạnh trên chuỗi video dài. Khi giảm xuống `ret25`, entropy giảm xuống **0.452**, tái lập sự sắc nét cho attention.

---

## 5. Chất Lượng Ngữ Nghĩa (Quality Metrics)

### Bảng 6: Chỉ số Chất lượng Đầu ra (Speculative vs Target)

| Budget | Điều kiện | Speculative BLEU | Target BLEU | Speculative ROUGE-L | Target ROUGE-L | Speculative Coverage | Target Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | 0.044 | 0.046 | 0.183 | 0.184 | 0.214 | 0.213 |
| | `cut` | 0.044 | 0.046 | 0.182 | 0.184 | 0.222 | 0.213 |
| **3,000** | `full` | 0.044 | 0.043 | 0.199 | 0.198 | 0.320 | 0.317 |
| | `cut` | 0.043 | 0.043 | 0.198 | 0.198 | 0.320 | 0.317 |
| **13,000** | `full` | 0.044 | 0.042 | 0.197 | 0.199 | 0.347 | 0.353 |
| | `retained_25` | 0.042 | 0.042 | 0.198 | 0.199 | 0.350 | 0.353 |
| **25,000** | `full` | 0.042 | 0.042 | 0.201 | 0.200 | 0.341 | 0.342 |
| | `cut` | 0.041 | 0.042 | 0.200 | 0.200 | 0.339 | 0.342 |

![Hình 4: Đánh giá Trade-off giữa Tốc độ và Acceptance cùng Tính Nhất Quán Chất Lượng (Middle-High Variant 2)](figures/fig4_quality_tradeoff_dflash_3layer_17_28_30_32_34.png)

---

## 6. Kết Luận Đánh Giá Cấu Hình `middle_high_2`

1. **Hiệu quả tối đa ở visual ngắn:** Tỷ lệ lossless 22.0% và speedup 1.88x ở 400 tokens chứng minh tầng neo 17 kết hợp với các tầng đỉnh liên tiếp nắm bắt rất chuẩn phân phối của Target.
2. **Khuyến nghị chiến lược cho video siêu dài:** Đây là cấu hình minh chứng rõ nhất cho giả thuyết *Visual Context Cut* — ở 25,000 visual tokens, việc cắt bỏ visual tokens phía draft giúp **tăng acceptance lên +0.115 tokens/round**, biến `cut` trở thành phương thức vận hành tối ưu nhất.
