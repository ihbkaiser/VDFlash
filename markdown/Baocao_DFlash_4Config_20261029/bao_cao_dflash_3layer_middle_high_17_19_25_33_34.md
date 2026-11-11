# Báo Cáo Thực Nghiệm DFlash: Cấu Hình `dflash_3layer_17_19_25_33_34` (Middle-High)

**Ngày lập báo cáo:** 2026-10-29  
**Cấu hình Draft:** `dflash_3layer_17_19_25_33_34` (`middle_high`)  
**Tầng trích xuất từ Target:** Layers [17, 19, 25, 33, 34] (trải rộng từ tầng giữa đến tầng ngọn)  
**Target Model:** `Qwen2.5-VL-3B-Instruct` (36 decoder layers)  
**Tập dữ liệu:** VideoDetailCaption (VDC50, $N=50$ video samples đã calibrate)  
**Ngân sách thị giác (Visual Budgets):** 400, 3,000, 13,000, 25,000 visual tokens  
**Tổng số cases thực nghiệm:** 1,200 cases ($50 \text{ samples} \times 4 \text{ budgets} \times 6 \text{ conditions}$), hoàn tất 100%.

---

## 1. Tóm Tắt Tổng Quan

Cấu hình `middle_high` kết hợp cả tầng biểu diễn trung gian (layers 17, 19, 25) và các tầng quyết định ngôn ngữ cấp cao (layers 33, 34).
- **Tốc độ sinh và Speculative Speedup:** Đạt đỉnh ở mức visual nhỏ (`400 tokens`) với **117.7 TPS** (Speedup **1.85x**) và giảm dần xuống **91.7 TPS** (Speedup **1.61x**) ở `3,000 tokens`, **30.2 TPS** ở `13,000 tokens`, và **10.2 TPS** ở `25,000 tokens`.
- **Hành vi Visual Cut & Retention:**
  - Ở `400 tokens` và `3k tokens`, việc giữ đầy đủ visual (`full`) mang lại acceptance tốt hơn (`full` = $2.508$ vs `cut` = $2.413$ ở 400; `full` = $2.407$ vs `cut` = $2.342$ ở 3k).
  - Ở `13k tokens`, `full`, `cut` và các mức retention đạt acceptance gần như nhau ($\tau \approx 2.24$).
  - Ở `25k tokens`, **`retained_25` đạt acceptance cao nhất** ($2.271$ vs $2.209$ ở `full`).
- **Phân bổ Attention theo tầng:** Cấu hình này phân tán attention vào visual đều hơn giữa `layer 1` và `layer 2`: ở 25k tokens, `layer 1` có 66.2% visual mass và `layer 2` có 62.3% visual mass.
- **Mật độ Attention (Density Ratio):** Token câu hỏi nhận mật độ attention cao gấp **11.1x** (ở 3k tokens), **31.7x** (ở 13k tokens) và lên tới **55.0x** (ở 25k tokens) so với mỗi token visual.

---

## 2. Thực Nghiệm 1: Visual Cut (`full` vs `zero` vs `cut`)

### Bảng 1: Chỉ số Acceptance, Tốc độ và Tính Lossless theo Ngân Sách Visual

| Budget (Tokens) | Điều kiện | $\tau_{\text{effective}}$ (Mean [95% CI]) | $\tau_{\text{proposal}}$ (Mean) | TPS | Speedup | Lossless Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | **2.508** [2.433, 2.584] | 1.508 | 117.7 | **1.85x** | 10.0% |
| | `zero` | 2.457 [2.372, 2.540] | 1.457 | 112.1 | 1.77x | 14.0% |
| | `cut` | 2.413 [2.335, 2.490] | 1.413 | 110.8 | 1.75x | 16.0% |
| **3,000** | `full` | **2.407** [2.344, 2.471] | 1.407 | 91.7 | **1.61x** | 0.0% |
| | `zero` | 2.315 [2.253, 2.378] | 1.315 | 87.1 | 1.53x | 2.0% |
| | `cut` | 2.342 [2.280, 2.406] | 1.342 | 90.3 | 1.59x | 0.0% |
| **13,000** | `full` | **2.242** [2.193, 2.290] | 1.242 | 30.2 | 0.99x | 6.0% |
| | `zero` | 2.181 [2.132, 2.232] | 1.181 | 29.7 | 0.97x | 0.0% |
| | `cut` | 2.240 [2.184, 2.299] | 1.240 | 31.2 | 1.02x | 0.0% |
| **25,000** | `full` | 2.209 [2.162, 2.261] | 1.209 | 10.2 | 1.26x | 0.0% |
| | `zero` | 2.183 [2.131, 2.235] | 1.183 | 10.1 | 1.31x | 4.0% |
| | `cut` | **2.229** [2.169, 2.293] | 1.229 | 10.1 | 1.35x | 0.0% |

### Bảng 2: Paired Contrasts (Hiệu số từng cặp so với `full` trên cùng video, Bootstrap 95% CI)

| Budget | Contrast | $\Delta \tau_{\text{effective}}$ [95% CI] | $\Delta$ TPS [95% CI] | Ý nghĩa Thống kê |
| :--- | :--- | :--- | :--- | :--- |
| **400** | `cut - full` | **-0.096** [-0.174, -0.016] | -6.9 [-11.8, -2.1] | **Giảm có ý nghĩa (Visual có ích)** |
| | `zero - full` | -0.051 [-0.141, +0.038] | -5.6 [-11.0, -0.5] | CI chứa 0 |
| **3,000** | `cut - full` | **-0.065** [-0.126, -0.003] | -1.5 [-3.9, +1.0] | **Giảm nhẹ có ý nghĩa** |
| | `zero - full` | **-0.092** [-0.155, -0.029] | -4.6 [-7.3, -1.9] | **Zero tệ hơn Cut** |
| **13,000** | `cut - full` | -0.002 [-0.062, +0.057] | +1.0 [+0.2, +1.7] | CI chứa 0 (Bão hòa visual) |
| | `zero - full` | **-0.061** [-0.117, -0.006] | -0.5 [-1.2, +0.1] | **Zero làm giảm hiệu năng** |
| **25,000** | `cut - full` | +0.020 [-0.046, +0.088] | -0.1 [-0.6, +0.4] | CI chứa 0 (Cut $\ge$ Full) |
| | `zero - full` | -0.027 [-0.086, +0.032] | -0.1 [-0.6, +0.4] | CI chứa 0 |

![Hình 1: Đánh giá Visual Cut và Paired Contrasts so với Full (Middle-High)](figures/fig1_visual_cut_dflash_3layer_17_19_25_33_34.png)

---

## 3. Thực Nghiệm 2: Visual Retention (`100% → 75% → 50% → 25% → 0%`)

### Bảng 3: Diễn tiến $\tau_{\text{effective}}$ qua các mức Retention

| Budget | `full` (100%) | `retained_75` (75%) | `retained_50` (50%) | `retained_25` (25%) | `cut` (0%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **400 tokens** | 2.508 [2.433, 2.584] | 2.474 [2.400, 2.547] | **2.520** [2.440, 2.603] | 2.469 [2.392, 2.553] | 2.413 [2.335, 2.490] |
| **3,000 tokens** | **2.407** [2.344, 2.471] | 2.403 [2.337, 2.470] | 2.404 [2.339, 2.468] | 2.391 [2.329, 2.451] | 2.342 [2.280, 2.406] |
| **13,000 tokens** | 2.242 [2.193, 2.290] | 2.244 [2.193, 2.295] | **2.244** [2.195, 2.290] | 2.231 [2.190, 2.273] | 2.240 [2.184, 2.299] |
| **25,000 tokens** | 2.209 [2.162, 2.261] | 2.265 [2.201, 2.334] | 2.257 [2.205, 2.311] | **2.271** [2.211, 2.335] | 2.229 [2.169, 2.293] |

![Hình 2: Đường cong Visual Retention Sweep qua các mức ngân sách (Middle-High)](figures/fig2_retention_sweep_dflash_3layer_17_19_25_33_34.png)

> [!TIP]
> Tương tự như cấu hình `high_only`, ở mức 25,000 tokens, việc cắt giảm xuống `retained_25` giúp cải thiện $\tau_{\text{effective}}$ từ $2.209$ lên **$2.271$**.

---

## 4. Thực Nghiệm 3: Phân Phối và Động Lực Học Attention

### Bảng 4: Phân bổ Chi Tiết 100% Attention Mass và Mật Độ (Condition `full`)

| Budget (Tokens) | Prompt Visual Mass | Prompt Text Mass (Question) | Generated Text Mass (Đã Sinh) | Current Draft Block (Causal) | Tổng Mass | Mật Độ Generated Text / Visual | Mật Độ Question / Visual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | 0.143 (14.3%) | 0.122 (12.2%) | **0.467 (46.7%)** | 0.268 (26.8%) | **1.000** | **31.6x** | 1.7x |
| **3,000** | 0.246 (24.6%) | 0.086 (8.6%) | **0.449 (44.9%)** | 0.220 (22.0%) | **1.000** | **125.0x** | 11.1x |
| **13,000** | **0.467 (46.7%)** | 0.092 (9.2%) | 0.273 (27.3%) | 0.168 (16.8%) | **1.000** | **602.5x** | 31.7x |
| **25,000** | **0.532 (53.2%)** | 0.088 (8.8%) | 0.227 (22.7%) | 0.153 (15.3%) | **1.000** | **1,120.0x** | 55.0x |

> [!NOTE]
> **Giải phẫu phân bổ 100% Attention Mass:**
> 1. **Cân bằng giữa Video và Ngữ cảnh sinh:** Ở visual 400 và 3k tokens, Generated Text Tokens chiếm tới 45%–47% attention. Khi chuỗi video dài lên 13k–25k tokens, Visual Mass tăng lên ~53%, nhưng Generated Text vẫn duy trì ổn định 23%–27% để neo giữ mạch văn.
> 2. **Mật độ chú ý vượt bậc:** Mỗi token đã sinh ra nhận mật độ chú ý cao gấp **31x đến 1,120x** so với mỗi token video, chứng tỏ mô hình ưu tiên cực cao việc bám sát các từ vừa xuất hiện trong câu trả lời.

### Bảng 5: Phân bổ Visual Attention Mass theo Tầng Draft (`layer 0`, `layer 1`, `layer 2`)

| Budget | Layer 0 (Input projection) | Layer 1 (Middle processing) | Layer 2 (Output head) |
| :--- | :--- | :--- | :--- |
| **400 tokens** | 0.046 (4.6%) | **0.182 (18.2%)** | **0.201 (20.1%)** |
| **3,000 tokens** | 0.124 (12.4%) | **0.321 (32.1%)** | **0.292 (29.2%)** |
| **13,000 tokens** | 0.260 (26.0%) | **0.616 (61.6%)** | **0.525 (52.5%)** |
| **25,000 tokens** | 0.312 (31.2%) | **0.662 (66.2%)** | **0.623 (62.3%)** |

![Hình 3: Động lực học Attention: Phân bổ Mass, Density Ratio và Tầng Draft (Middle-High)](figures/fig3_attention_dynamics_dflash_3layer_17_19_25_33_34.png)

> [!NOTE]
> So với `high_only`, cấu hình `middle_high` có tầng cuối (`layer 2`) tham gia mạnh hơn vào visual context (62.3% visual mass ở 25k tokens), cho thấy các đặc trưng từ tầng 17-25 của Target được truyền sâu hơn tới đầu ra của Draft.

---

## 5. Chất Lượng Ngữ Nghĩa (Quality Metrics)

### Bảng 6: Chỉ số Chất lượng Đầu ra (Speculative vs Target)

| Budget | Điều kiện | Speculative BLEU | Target BLEU | Speculative ROUGE-L | Target ROUGE-L | Speculative Coverage | Target Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | 0.044 | 0.046 | 0.181 | 0.184 | 0.212 | 0.213 |
| | `cut` | 0.044 | 0.046 | 0.184 | 0.184 | 0.221 | 0.213 |
| **3,000** | `full` | 0.045 | 0.043 | 0.201 | 0.198 | 0.328 | 0.317 |
| | `cut` | 0.046 | 0.043 | 0.201 | 0.198 | 0.318 | 0.317 |
| **13,000** | `full` | 0.040 | 0.042 | 0.196 | 0.199 | 0.338 | 0.353 |
| | `retained_25` | 0.043 | 0.042 | 0.201 | 0.199 | 0.355 | 0.353 |
| **25,000** | `full` | 0.040 | 0.042 | 0.200 | 0.200 | 0.338 | 0.342 |
| | `retained_25` | 0.042 | 0.042 | 0.196 | 0.200 | 0.343 | 0.342 |

![Hình 4: Đánh giá Trade-off giữa Tốc độ và Acceptance cùng Tính Nhất Quán Chất Lượng (Middle-High)](figures/fig4_quality_tradeoff_dflash_3layer_17_19_25_33_34.png)

---

## 6. Kết Luận Đánh Giá Cấu Hình `middle_high`

1. Tốc độ sinh ở visual 400 đạt **117.7 TPS** (nhanh nhất trong 4 cấu hình ở mức ngân sách này).
2. Sự kết hợp tầng giữa (17, 19, 25) giúp phân bổ biểu diễn thị giác đều qua các tầng draft, nhưng ở ngữ cảnh dài vẫn chịu hiện tượng dilution tương tự, với mật độ câu hỏi cao gấp **55x** so với visual tokens ở 25k.
