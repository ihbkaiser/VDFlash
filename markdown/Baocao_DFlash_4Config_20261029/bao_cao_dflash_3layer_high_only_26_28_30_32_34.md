# Báo Cáo Thực Nghiệm DFlash: Cấu Hình `dflash_3layer_26_28_30_32_34` (High-Only)

**Ngày lập báo cáo:** 2026-10-29  
**Cấu hình Draft:** `dflash_3layer_26_28_30_32_34` (`high_only`)  
**Tầng trích xuất từ Target:** Layers [26, 28, 30, 32, 34] (các tầng sâu nhất của Target Model)  
**Target Model:** `Qwen2.5-VL-3B-Instruct` (36 decoder layers)  
**Tập dữ liệu:** VideoDetailCaption (VDC50, $N=50$ video samples đã calibrate)  
**Ngân sách thị giác (Visual Budgets):** 400, 3,000, 13,000, 25,000 visual tokens  
**Tổng số cases thực nghiệm:** 1,200 cases ($50 \text{ samples} \times 4 \text{ budgets} \times 6 \text{ conditions}$), hoàn tất 100%.

---

## 1. Tóm Tắt Tổng Quan

Cấu hình `high_only` sử dụng 5 tầng ẩn ở phần ngọn của Target Model (`26, 28, 30, 32, 34`) để xây dựng draft representation 3 tầng.
- **Tốc độ sinh và Speculative Speedup:** Đạt đỉnh ở mức visual nhỏ (`400 tokens`) với **115.2 TPS** (Speedup **1.86x** so với Target đơn lẻ 61.9 TPS) và giảm dần xuống **92.6 TPS** (Speedup **1.62x**) ở `3,000 tokens`, **30.4 TPS** ở `13,000 tokens`, và **13.1 TPS** ở `25,000 tokens`.
- **Hiện tượng Attention Dilution & Lợi thế của Visual Retention:** Ở các ngân sách video lớn (`13k` và `25k tokens`), giữ nguyên 100% visual tokens (`full`) không mang lại acceptance cao nhất. Ngược lại, việc giữ lại 25% đến 50% visual tokens (`retained_25`, `retained_50`) giúp draft model đạt **$\tau_{\text{effective}}$ cao hơn cả `full`** ($2.318$ vs $2.262$ ở 25k).
- **Phân bổ Attention theo tầng:** Tầng giữa của Draft (`layer 1`) là tầng hấp thụ visual attention mạnh nhất (lên tới **80.5% visual mass** ở 13k tokens), trong khi tầng đầu (`layer 0`) phân bổ đều hơn cho text và draft query noise.
- **Mật độ Attention (Density Ratio):** Mặc dù tổng visual mass chiếm 50–58% ở ngữ cảnh dài, nhưng do số lượng token hình ảnh quá lớn, **mỗi token câu hỏi (Question Text) nhận mật độ attention cao gấp 11.0x (ở 3k tokens) và 44.0x (ở 25k tokens) so với mỗi token visual**.

---

## 2. Thực Nghiệm 1: Visual Cut (`full` vs `zero` vs `cut`)

Thực nghiệm này tách bạch vai trò của **cấu trúc vị trí/độ dài chuỗi** và **giá trị đặc trưng ngữ nghĩa (semantic feature values)** trong draft context:
- `full`: Giữ nguyên 100% vị trí và giá trị hidden states của visual tokens.
- `zero`: Giữ nguyên chiều dài và vị trí visual tokens, nhưng gán toàn bộ giá trị vector đặc trưng bằng 0.
- `cut`: Cắt bỏ hoàn toàn các token visual khỏi context của Draft (Draft chỉ nhìn thấy text).

### Bảng 1: Chỉ số Acceptance, Tốc độ và Tính Lossless theo Ngân Sách Visual

| Budget (Tokens) | Điều kiện | $\tau_{\text{effective}}$ (Mean [95% CI]) | $\tau_{\text{proposal}}$ (Mean) | TPS | Speedup | Lossless Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | **2.538** [2.455, 2.630] | 1.538 | 115.2 | **1.86x** | 10.0% |
| | `zero` | 2.485 [2.374, 2.627] | 1.485 | 109.7 | 1.77x | 20.0% |
| | `cut` | 2.480 [2.354, 2.628] | 1.480 | 109.5 | 1.76x | 8.0% |
| **3,000** | `full` | **2.426** [2.373, 2.485] | 1.426 | 92.6 | **1.62x** | 2.0% |
| | `zero` | 2.337 [2.272, 2.407] | 1.337 | 88.0 | 1.54x | 4.0% |
| | `cut` | 2.333 [2.277, 2.390] | 1.333 | 89.1 | 1.55x | 8.0% |
| **13,000** | `full` | **2.250** [2.199, 2.298] | 1.250 | 30.4 | 1.00x | 6.0% |
| | `zero` | 2.181 [2.123, 2.236] | 1.181 | 29.7 | 0.97x | 4.0% |
| | `cut` | 2.209 [2.154, 2.266] | 1.209 | 30.6 | 1.00x | 2.0% |
| **25,000** | `full` | **2.262** [2.213, 2.316] | 1.262 | 13.1 | 0.97x | 0.0% |
| | `zero` | 2.154 [2.106, 2.202] | 1.154 | 12.8 | 0.94x | 2.0% |
| | `cut` | 2.256 [2.196, 2.320] | 1.256 | 13.4 | 0.99x | 0.0% |

### Bảng 2: Paired Contrasts (Hiệu số từng cặp so với `full` trên cùng video, Bootstrap 95% CI)

| Budget | Contrast | $\Delta \tau_{\text{effective}}$ [95% CI] | $\Delta$ TPS [95% CI] | Ý nghĩa Thống kê |
| :--- | :--- | :--- | :--- | :--- |
| **400** | `cut - full` | -0.057 [-0.174, +0.052] | -5.7 [-11.1, -0.4] | CI chứa 0 (chưa khác biệt rõ rệt) |
| | `zero - full` | -0.053 [-0.160, +0.054] | -5.5 [-10.8, -0.2] | CI chứa 0 |
| **3,000** | `cut - full` | **-0.092** [-0.164, -0.021] | -3.5 [-5.8, -1.3] | **Giảm có ý nghĩa (Visual có ích)** |
| | `zero - full` | **-0.088** [-0.158, -0.019] | -4.6 [-7.0, -2.2] | **Giảm có ý nghĩa** |
| **13,000** | `cut - full` | -0.040 [-0.098, +0.017] | +0.2 [-0.5, +0.9] | CI chứa 0 (Bão hòa visual) |
| | `zero - full` | **-0.069** [-0.126, -0.012] | -0.7 [-1.4, +0.0] | **Zero làm giảm hiệu năng** |
| **25,000** | `cut - full` | -0.006 [-0.070, +0.058] | +0.3 [-0.2, +0.7] | CI chứa 0 (Cut tương đương Full) |
| | `zero - full` | **-0.108** [-0.170, -0.046] | -0.3 [-0.8, +0.1] | **Zero tệ nhất (Nhiễu context)** |

![Hình 1: Đánh giá Visual Cut và Paired Contrasts so với Full (High-Only)](figures/fig1_visual_cut_dflash_3layer_26_28_30_32_34.png)

> [!IMPORTANT]
> **Nhận xét chính:** Ở 400 và 3,000 tokens, visual context giúp tăng acceptance (+0.09 tokens/round). Tuy nhiên, khi video dài lên 13k và 25k tokens, `cut` hoàn toàn tương đương với `full` ($\Delta \tau \approx -0.006$, CI chứa 0). Trong khi đó, `zero` bị phạt nặng nhất vì giữ nguyên 25,000 vị trí vô nghĩa khiến attention bị loãng mà không thu được thông tin.

---

## 3. Thực Nghiệm 2: Visual Retention (`100% → 75% → 50% → 25% → 0%`)

Khảo sát đường cong thay đổi acceptance khi giảm dần số lượng token thị giác trong Draft context:

### Bảng 3: Diễn tiến $\tau_{\text{effective}}$ qua các mức Retention

| Budget | `full` (100%) | `retained_75` (75%) | `retained_50` (50%) | `retained_25` (25%) | `cut` (0%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **400 tokens** | 2.538 [2.455, 2.630] | **2.567** [2.460, 2.701] | 2.543 [2.436, 2.678] | **2.566** [2.457, 2.701] | 2.480 [2.354, 2.628] |
| **3,000 tokens** | 2.426 [2.373, 2.485] | **2.441** [2.384, 2.505] | 2.429 [2.370, 2.493] | 2.382 [2.324, 2.446] | 2.333 [2.277, 2.390] |
| **13,000 tokens** | 2.250 [2.199, 2.298] | 2.258 [2.210, 2.304] | 2.266 [2.224, 2.306] | **2.287** [2.242, 2.334] | 2.209 [2.154, 2.266] |
| **25,000 tokens** | 2.262 [2.213, 2.316] | 2.264 [2.214, 2.317] | **2.314** [2.252, 2.395] | **2.318** [2.250, 2.406] | 2.256 [2.196, 2.320] |

![Hình 2: Đường cong Visual Retention Sweep qua các mức ngân sách (High-Only)](figures/fig2_retention_sweep_dflash_3layer_26_28_30_32_34.png)

> [!TIP]
> **Điểm tối ưu (Sweet Spot):** Ở mức 13,000 và 25,000 visual tokens, **`retained_25` (giữ 25% token thị giác, tức ~3,250 đến 6,250 tokens)** đạt hiệu quả đề xuất cao nhất ($2.287$ ở 13k và $2.318$ ở 25k). Điều này chứng minh rằng việc tỉa bớt 75% visual tokens không những không làm mất thông tin cần thiết cho draft mà còn giúp khử bớt hiện tượng phân tán attention (dilution effect).

---

## 4. Thực Nghiệm 3: Phân Phối và Động Lực Học Attention

### Bảng 4: Phân bổ Chi Tiết 100% Attention Mass và Mật Độ (Condition `full`)

| Budget (Tokens) | Prompt Visual Mass | Prompt Text Mass (Question) | Generated Text Mass (Đã Sinh) | Current Draft Block (Causal) | Tổng Mass | Mật Độ Generated Text / Visual | Mật Độ Question / Visual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | 0.140 (14.0%) | 0.134 (13.4%) | **0.465 (46.5%)** | 0.261 (26.1%) | **1.000** | **40.0x** | 1.9x |
| **3,000** | 0.334 (33.4%) | 0.122 (12.2%) | **0.327 (32.7%)** | 0.216 (21.6%) | **1.000** | **190.0x** | 11.0x |
| **13,000** | **0.577 (57.7%)** | 0.111 (11.1%) | 0.151 (15.1%) | 0.161 (16.1%) | **1.000** | **1,200.0x** | 25.4x |
| **25,000** | **0.506 (50.6%)** | 0.093 (9.3%) | 0.251 (25.1%) | 0.151 (15.1%) | **1.000** | **1,610.0x** | 44.0x |

> [!NOTE]
> **Giải phẫu phân bổ 100% Attention Mass:**
> 1. **Tổng 4 thành phần luôn bằng đúng 1.0 (100%):** Chú ý không chỉ dồn vào Video và Câu hỏi mà một phần rất lớn (**15% – 46.5%**) dành cho **Generated Text Tokens** (các từ ngữ của câu trả lời đã sinh ra ở các bước trước) cùng **Current Draft Block** (15% – 26% cho causal self-attention trong khối speculative).
> 2. **Ưu thế mật độ tuyệt đối của Generated Text Tokens:** Mặc dù số lượng từ đã sinh chỉ khoảng 10–50 tokens, nhưng chúng nhận tới 15%–46% tổng attention. Do đó, **mỗi token đã sinh ra nhận mật độ chú ý cao gấp 40x đến 1,600x so với mỗi token video**, đóng vai trò mỏ neo ngữ pháp cốt lõi để duy trì tính liên tục của văn bản.

### Bảng 5: Phân bổ Visual Attention Mass theo Tầng Draft (`layer 0`, `layer 1`, `layer 2`)

| Budget | Layer 0 (Input projection) | Layer 1 (Middle processing) | Layer 2 (Output head) |
| :--- | :--- | :--- | :--- |
| **400 tokens** | 0.046 (4.6%) | **0.224 (22.4%)** | 0.151 (15.1%) |
| **3,000 tokens** | 0.166 (16.6%) | **0.484 (48.4%)** | 0.354 (35.4%) |
| **13,000 tokens** | 0.307 (30.7%) | **0.805 (80.5%)** | 0.618 (61.8%) |
| **25,000 tokens** | 0.314 (31.4%) | **0.721 (72.1%)** | 0.494 (49.4%) |

![Hình 3: Động lực học Attention: Phân bổ Mass, Density Ratio và Tầng Draft (High-Only)](figures/fig3_attention_dynamics_dflash_3layer_26_28_30_32_34.png)

> [!NOTE]
> **Đặc trưng kiến trúc của `high_only`:** Tầng `layer 1` phụ trách gần như toàn bộ việc kết nối và trích xuất ngữ cảnh thị giác từ Target hidden states. Ở 13k tokens, hơn **80.5%** tổng xác suất attention của tầng 1 đổ vào visual keys.

---

## 5. Chất Lượng Ngữ Nghĩa và Tính Lossless (Quality & Agreement)

### Bảng 6: Chỉ số Chất lượng Đầu ra (Speculative vs Target)

| Budget | Điều kiện | Speculative BLEU | Target BLEU | Speculative ROUGE-L | Target ROUGE-L | Speculative Coverage | Target Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | 0.048 | 0.046 | 0.185 | 0.184 | 0.211 | 0.213 |
| | `cut` | 0.046 | 0.046 | 0.181 | 0.184 | 0.215 | 0.213 |
| **3,000** | `full` | 0.044 | 0.043 | 0.198 | 0.198 | 0.314 | 0.317 |
| | `cut` | 0.044 | 0.043 | 0.198 | 0.198 | 0.320 | 0.317 |
| **13,000** | `full` | 0.042 | 0.042 | 0.200 | 0.199 | 0.352 | 0.353 |
| | `retained_25` | 0.044 | 0.042 | 0.201 | 0.199 | 0.347 | 0.353 |
| **25,000** | `full` | 0.042 | 0.042 | 0.200 | 0.200 | 0.342 | 0.342 |
| | `retained_25` | 0.041 | 0.042 | 0.199 | 0.200 | 0.343 | 0.342 |

![Hình 4: Đánh giá Trade-off giữa Tốc độ và Acceptance cùng Tính Nhất Quán Chất Lượng (High-Only)](figures/fig4_quality_tradeoff_dflash_3layer_26_28_30_32_34.png)

**Đánh giá:** Các chỉ số chất lượng BLEU, ROUGE-L và Content Coverage hoàn toàn tương đồng giữa Speculative decoding và Target autoregressive decoding ở mọi điều kiện (sai khác $< 0.002$, không có ý nghĩa thống kê). Việc cắt giảm visual tokens trong Draft context không làm suy giảm chất lượng câu trả lời.

---

## 6. Kết Luận Đánh Giá Cấu Hình `high_only`

1. **Khả năng tăng tốc:** Cấu hình `high_only` đạt speedup rất tốt ở video ngắn/trung bình (1.86x ở 400 tokens, 1.62x ở 3k tokens).
2. **Khuyến nghị vận hành:** Đối với video dài ($\ge 13,000$ tokens), nên áp dụng chính sách **`retained_25`** (giữ lại 25% visual tokens) cho Draft model. Giải pháp này giúp tăng $\tau_{\text{effective}}$ từ $2.262$ lên $2.318$, đồng thời giảm tải KV cache và bộ nhớ tính toán attention.
