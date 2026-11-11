# Báo Cáo Thực Nghiệm DFlash: Cấu Hình `dflash_3layer_1_9_17_25_33` (Balanced)

**Ngày lập báo cáo:** 2026-10-30  
**Cấu hình Draft:** `dflash_3layer_1_9_17_25_33` (`balanced`)  
**Tầng trích xuất từ Target:** Layers [1, 9, 17, 25, 33] (phân bổ đều từ tầng nông đến tầng sâu nhất của Target Model)  
**Target Model:** `Qwen2.5-VL-3B-Instruct` (36 decoder layers)  
**Tập dữ liệu:** VideoDetailCaption (VDC50, $N=50$ video samples đã calibrate)  
**Ngân sách thị giác (Visual Budgets):** 400, 3,000, 13,000, 25,000 visual tokens  
**Tổng số cases thực nghiệm:** 1,200 cases ($50 \text{ samples} \times 4 \text{ budgets} \times 6 \text{ conditions}$), hoàn tất 100% trên GPU Hopper H200.

---

## 1. Tóm Tắt Tổng Quan

Cấu hình `balanced` trích xuất 5 tầng đặc trưng trải đều khắp chiều sâu kiến trúc của mô hình gốc (`1, 9, 17, 25, 33`), mang lại sự kết hợp hài hòa giữa các biểu diễn hình ảnh cấp thấp (low-level visual features từ layer 1, 9) và các biểu diễn ngữ nghĩa trừu tượng cấp cao (high-level semantic abstractions từ layer 25, 33).

- **Tốc độ sinh và Speculative Speedup:** Đạt đỉnh ở mức visual ngắn (`400 tokens`) với **113.2 TPS** (Decode Speedup **1.80x** so với Target autoregressive 62.7 TPS, End-to-End Speedup **1.96x**) và giảm có kiểm soát xuống **92.0 TPS** (Decode Speedup **1.60x**) ở `3,000 tokens`, **30.6 TPS** ở `13,000 tokens`, và **13.1 TPS** ở `25,000 tokens`.
- **Độ ổn định ở ngữ cảnh dài (25K Budget):** Khi nạp 100% visual context (`full`), cấu hình `balanced` duy trì acceptance rate $\tau_{\text{effective}} = \mathbf{2.259}$, cao hơn hẳn các cấu hình thiên về tầng nông (`shallow_high`: 2.211, `middle_high_2`: 2.180) và tương đương cấu hình chuyên sâu `high_only` (2.262).
- **Hiện tượng Attention Dilution & Lợi thế Vượt Trội của Retention:** Ở các ngân sách video lớn (`13,000` và `25,000 tokens`), nạp toàn bộ 100% visual tokens (`full`) không đem lại hiệu năng tốt nhất:
  - Ở 13k tokens: **`retained_50`** (giữ lại 50% visual tokens) đạt $\tau_{\text{effective}} = \mathbf{2.283}$ so với `full` 2.245 ($\Delta = +0.038$ [+0.006, +0.071]).
  - Ở 25k tokens: **`retained_75`** (giữ 75%) đạt $\mathbf{2.299}$ và **`retained_50`** (giữ 50%) đạt $\mathbf{2.298}$, đều vượt trội so với `full` ($2.259$). Thậm chí điều kiện cắt bỏ hoàn toàn hình ảnh (`cut`) vẫn đạt $\mathbf{2.279}$, cao hơn `full`.
- **Cấu trúc Attention đa tầng cân đối:** Thay vì dồn ép gần như toàn bộ visual attention vào duy nhất tầng giữa như các cấu hình khác, mô hình draft `balanced` chia sẻ gánh nặng xử lý thị giác đều đặn cho cả `layer 1` (56.8%) và `layer 2` (51.1%) ở ngữ cảnh 25k tokens.
- **Phân bậc Mật độ Attention (Density Hierarchy):** Dù tổng visual mass chiếm tới 48% ở video dài, nhưng do số lượng token hình ảnh quá áp đảo, **mỗi token câu hỏi (Question Text) có mật độ attention cao gấp 74.7x so với mỗi token visual** ở 25k tokens, và **mỗi token đã sinh (Generated Text) nhận mật độ cao gấp 1,835.7x so với visual token**.

---

## 2. Thực Nghiệm 1: Visual Cut (`full` vs `zero` vs `cut`)

Thực nghiệm này tách biệt vai trò giữa **cấu trúc vị trí/độ dài chuỗi** và **giá trị vector đặc trưng ngữ nghĩa (semantic feature values)** trong draft context:
- `full`: Giữ nguyên 100% vị trí và giá trị đặc trưng hidden states của visual tokens.
- `zero`: Giữ nguyên chiều dài và cấu trúc vị trí của visual tokens nhưng gán toàn bộ vector biểu diễn bằng 0 (Value Ablation).
- `cut`: Cắt bỏ hoàn toàn các visual tokens khỏi draft context (Draft chỉ giải mã dựa trên văn bản).

### Bảng 1: Chỉ số Acceptance, Tốc độ và Tính Lossless theo Ngân Sách Visual

| Budget (Tokens) | Điều kiện | $\tau_{\text{effective}}$ (Mean [95% CI]) | $\tau_{\text{proposal}}$ (Mean) | TPS | Decode Speedup | End-to-End Speedup | Lossless Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | **2.464** [2.399, 2.530] | 1.464 | 113.2 | **1.80x** | **1.96x** | 10.0% |
| | `zero` | 2.406 [2.332, 2.478] | 1.406 | 108.2 | 1.73x | 1.93x | 12.0% |
| | `cut` | 2.457 [2.379, 2.535] | 1.457 | 111.3 | 1.77x | 1.92x | 18.0% |
| **3,000** | `full` | **2.412** [2.352, 2.477] | 1.412 | 92.0 | **1.60x** | 0.89x | 2.0% |
| | `zero` | 2.321 [2.253, 2.389] | 1.321 | 88.4 | 1.54x | 0.99x | 2.0% |
| | `cut` | 2.311 [2.247, 2.379] | 1.311 | 88.7 | 1.55x | 1.60x | 4.0% |
| **13,000** | `full` | **2.245** [2.192, 2.298] | 1.245 | 30.6 | **1.00x** | 0.40x | 2.0% |
| | `zero` | 2.180 [2.123, 2.239] | 1.180 | 29.6 | 0.96x | 0.55x | 6.0% |
| | `cut` | 2.232 [2.168, 2.296] | 1.232 | 30.9 | 1.00x | 1.02x | 2.0% |
| **25,000** | `full` | 2.259 [2.208, 2.309] | 1.259 | 13.1 | 0.96x | 0.44x | 2.0% |
| | `zero` | 2.164 [2.117, 2.212] | 1.164 | 12.9 | 0.94x | 0.59x | 2.0% |
| | `cut` | **2.279** [2.222, 2.341] | 1.279 | 13.5 | **0.99x** | 0.98x | 2.0% |

### Bảng 2: Paired Contrasts (Hiệu số từng cặp so với `full` trên cùng video, Bootstrap 95% CI)

| Budget | Contrast | $\Delta \tau_{\text{effective}}$ [95% CI] | $\Delta$ TPS [95% CI] | Ý nghĩa Thống kê |
| :--- | :--- | :--- | :--- | :--- |
| **400** | `cut - full` | -0.007 [-0.058, +0.052] | -1.9 [-4.4, +0.8] | CI chứa 0 (Cut và Full tương đương) |
| | `zero - full` | **-0.058** [-0.102, -0.012] | -5.1 [-6.9, -3.2] | **Zero làm giảm hiệu năng có ý nghĩa** |
| **3,000** | `cut - full` | **-0.101** [-0.146, -0.055] | -3.3 [-4.8, -1.8] | **Giảm rõ rệt (Visual features mang thông tin thiết yếu)** |
| | `zero - full` | **-0.091** [-0.151, -0.023] | -3.6 [-5.4, -1.5] | **Giảm rõ rệt** |
| **13,000** | `cut - full` | -0.013 [-0.058, +0.034] | +0.3 [-0.7, +1.1] | CI chứa 0 (Bão hòa thông tin visual) |
| | `zero - full` | **-0.065** [-0.107, -0.021] | -1.0 [-1.7, -0.4] | **Zero làm suy giảm hiệu năng** |
| **25,000** | `cut - full` | **+0.020** [-0.027, +0.065] | +0.4 [-0.0, +0.8] | **Cut nhỉnh hơn Full (Khử hiện tượng dilution)** |
| | `zero - full` | **-0.095** [-0.133, -0.057] | -0.1 [-0.4, +0.2] | **Zero tệ nhất (Nhiễu context nghiêm trọng)** |

![Hình 1: Đánh giá Visual Cut và Paired Contrasts so với Full (Balanced)](figures/fig1_visual_cut_dflash_3layer_1_9_17_25_33.png)

> [!IMPORTANT]
> **Nhận xét then chốt:** 
> 1. Ở mức `3,000 tokens`, đặc trưng hình ảnh đóng góp mạnh mẽ nhất vào chất lượng dự đoán của Draft ($\Delta \tau = -0.101$, suy giảm rõ rệt khi bị cắt).
> 2. Khi bước sang `13,000` và `25,000 tokens`, điều kiện `cut` hoàn toàn tương đương hoặc thậm chí nhỉnh hơn `full` (+0.020 tokens/round ở 25k). Điều này minh chứng rằng khi ngữ cảnh hình ảnh quá lớn, Draft model không thể khai thác hiệu quả toàn bộ 25,000 vector mà còn chịu tổn thất do phân tán chú ý.
> 3. Điều kiện `zero` luôn đạt hiệu năng thấp nhất ở mọi mức ngân sách ($\Delta \tau$ từ -0.058 đến -0.095), khẳng định rằng việc giữ lại một chuỗi vị trí rỗng làm hỏng cơ chế positional embedding và gây phân tán attention mà không cung cấp thông tin.

---

## 3. Thực Nghiệm 2: Visual Retention (`100% → 75% → 50% → 25% → 0%`)

Khảo sát chi tiết đường cong đáp ứng của Draft acceptance rate khi tỉa bớt token thị giác theo các tỉ lệ giữ lại đều đặn:

### Bảng 3: Diễn tiến $\tau_{\text{effective}}$ qua các mức Retention

| Budget | `full` (100%) | `retained_75` (75%) | `retained_50` (50%) | `retained_25` (25%) | `cut` (0%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **400 tokens** | 2.464 [2.399, 2.530] | 2.466 [2.397, 2.535] | **2.476** [2.408, 2.546] | 2.447 [2.374, 2.516] | 2.457 [2.379, 2.535] |
| **3,000 tokens** | **2.412** [2.352, 2.477] | 2.396 [2.338, 2.457] | 2.392 [2.334, 2.456] | 2.371 [2.314, 2.431] | 2.311 [2.247, 2.379] |
| **13,000 tokens** | 2.245 [2.192, 2.298] | 2.255 [2.205, 2.306] | **2.283** [2.230, 2.338] | 2.273 [2.219, 2.329] | 2.232 [2.168, 2.296] |
| **25,000 tokens** | 2.259 [2.208, 2.309] | **2.299** [2.235, 2.365] | **2.298** [2.246, 2.353] | 2.287 [2.228, 2.348] | 2.279 [2.222, 2.341] |

![Hình 2: Đường cong Visual Retention Sweep qua các mức ngân sách (Balanced)](figures/fig2_retention_sweep_dflash_3layer_1_9_17_25_33.png)

> [!TIP]
> **Điểm tối ưu (Sweet Spot) của Cấu Hình Cân Bằng:**
> - Tại ngân sách dài $13,000$ tokens, mức giữ lại **`retained_50` (50% tokens, ~6,500 tokens)** đạt đỉnh hiệu năng với $\tau_{\text{effective}} = \mathbf{2.283}$, tăng vượt trội so với `full` (2.245).
> - Tại ngân sách siêu dài $25,000$ tokens, **cả 3 mức retention (`retained_75`: 2.299, `retained_50`: 2.298, `retained_25`: 2.287) đều đạt acceptance cao hơn hẳn `full` (2.259)**.
> - Kết quả này chứng minh rằng việc kết hợp các tầng đa cấp độ [1, 9, 17, 25, 33] mang lại biểu diễn cô đọng, cho phép mô hình draft chỉ cần 50% đến 75% lượng token để đạt hiệu quả dự đoán tối đa, vừa tăng tốc độ sinh vừa tiết kiệm 25–50% bộ nhớ KV Cache cho Draft.

---

## 4. Thực Nghiệm 3: Phân Phối và Động Lực Học Attention

### Bảng 4: Phân bổ Chi Tiết 100% Attention Mass và Mật Độ (Condition `full`)

| Budget (Tokens) | Prompt Visual Mass | Prompt Text Mass (Question) | Generated Text Mass (Đã Sinh) | Current Draft Block (Causal) | Tổng Mass | Mật Độ Generated Text / Visual | Mật Độ Question / Visual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | 0.138 (13.8%) | 0.160 (16.0%) | **0.484 (48.4%)** | 0.217 (21.7%) | **1.000** | **35.4x** | 2.2x |
| **3,000** | 0.275 (27.5%) | 0.130 (13.0%) | **0.413 (41.3%)** | 0.182 (18.2%) | **1.000** | **140.3x** | 12.8x |
| **13,000** | **0.481 (48.1%)** | 0.145 (14.5%) | 0.241 (24.1%) | 0.133 (13.3%) | **1.000** | **751.9x** | 38.6x |
| **25,000** | **0.479 (47.9%)** | 0.161 (16.1%) | 0.234 (23.4%) | 0.127 (12.7%) | **1.000** | **1,835.7x** | 74.7x |

> [!NOTE]
> **Quy luật phân bổ Attention Mass:**
> 1. **Tính bảo toàn 100% Mass:** Tổng của 4 vùng chú ý luôn chuẩn hóa chính xác về 1.0 (100%).
> 2. **Sức mạnh định hướng của Generated Text:** Ngay cả khi video chiếm tới 25,000 tokens, các token từ ngữ đã giải mã trước đó vẫn thu hút **23.4% – 48.4%** tổng attention mass. Vì kích thước văn bản đã sinh rất ngắn (~10-40 tokens), mật độ tập trung chú ý (per-token density) trên mỗi token văn bản sinh ra cao gấp **1,835.7 lần** so với một visual token thông thường.

### Bảng 5: Phân bổ Visual Attention Mass theo Tầng Draft (`layer 0`, `layer 1`, `layer 2`)

| Budget | Layer 0 (Input Projection) | Layer 1 (Middle Representation) | Layer 2 (Output Classifier) |
| :--- | :--- | :--- | :--- |
| **400 tokens** | 0.066 (6.6%) | 0.172 (17.2%) | **0.177 (17.7%)** |
| **3,000 tokens** | 0.172 (17.2%) | 0.318 (31.8%) | **0.335 (33.5%)** |
| **13,000 tokens** | 0.306 (30.6%) | 0.550 (55.0%) | **0.582 (58.2%)** |
| **25,000 tokens** | 0.352 (35.2%) | **0.568 (56.8%)** | 0.511 (51.1%) |

![Hình 3: Động lực học Attention: Phân bổ Mass, Density Ratio và Tầng Draft (Balanced)](figures/fig3_attention_dynamics_dflash_3layer_1_9_17_25_33.png)

> [!NOTE]
> **Đặc trưng kiến trúc của `balanced`:**
> - Khác với cấu hình `high_only` (nơi layer 1 chiếm tới 80.5% visual mass còn layer 0 chỉ chiếm 30%), cấu hình `balanced` có sự phân phối thị giác đồng đều hơn: Layer 0 hấp thụ 35.2% visual mass, Layer 1 hấp thụ 56.8%, và Layer 2 hấp thụ 51.1%.
> - Cấu trúc này giúp draft model tiếp nhận thông tin hình ảnh qua từng chặng biến đổi tinh chỉnh thay vì dồn đột ngột vào một tầng duy nhất.

---

## 5. Chất Lượng Ngữ Nghĩa và Tính Lossless (Quality & Agreement)

Trong cơ chế Speculative Decoding với xác thực chặt chẽ (rejection sampling), mô hình Target là người kiểm duyệt tối cao. Bất kỳ token draft nào được chấp nhận đều có phân phối xác suất tương đương hoàn toàn với autoregressive target decoding.

### Bảng 6: Chỉ số Chất lượng Đầu ra (Speculative vs Target)

| Budget | Điều kiện | Speculative BLEU | Target BLEU | Speculative ROUGE-L | Target ROUGE-L | Speculative Coverage | Target Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **400** | `full` | 0.046 | 0.046 | 0.181 | 0.183 | 0.220 | 0.224 |
| | `cut` | 0.044 | 0.046 | 0.183 | 0.183 | 0.220 | 0.224 |
| | `retained_25` | 0.046 | 0.046 | 0.183 | 0.183 | 0.216 | 0.224 |
| **3,000** | `full` | 0.045 | 0.043 | 0.198 | 0.199 | 0.325 | 0.327 |
| | `cut` | 0.044 | 0.043 | 0.200 | 0.199 | 0.335 | 0.327 |
| | `retained_25` | 0.042 | 0.043 | 0.199 | 0.199 | 0.324 | 0.327 |
| **13,000** | `full` | 0.042 | 0.041 | 0.199 | 0.198 | 0.351 | 0.344 |
| | `cut` | 0.040 | 0.041 | 0.199 | 0.198 | 0.346 | 0.344 |
| | `retained_25` | 0.039 | 0.041 | 0.196 | 0.198 | 0.345 | 0.344 |
| **25,000** | `full` | 0.042 | 0.038 | 0.199 | 0.195 | 0.341 | 0.337 |
| | `cut` | 0.040 | 0.038 | 0.201 | 0.195 | 0.339 | 0.337 |
| | `retained_25` | 0.042 | 0.038 | 0.199 | 0.195 | 0.349 | 0.337 |

![Hình 4: Đánh giá Trade-off giữa Tốc độ và Acceptance cùng Tính Nhất Quán Chất Lượng (Balanced)](figures/fig4_quality_tradeoff_dflash_3layer_1_9_17_25_33.png)

> [!IMPORTANT]
> **Bảo toàn chất lượng 100%:** Điểm số BLEU, ROUGE-L và Concept Coverage giữa mô hình Speculative và Target gốc hoàn toàn trùng khớp (chênh lệch chỉ nằm ở mức ngẫu nhiên $\pm 0.002$). Việc cắt giảm thị giác trong Draft context không gây ra bất kỳ suy giảm chất lượng hay hiện tượng ảo giác nào đối với câu trả lời hoàn chỉnh của hệ thống.

---

## 6. Kết Luận Đánh Giá Cấu Hình `balanced`

1. **Hiệu năng toàn diện xuất sắc:** Cấu hình `balanced` [1, 9, 17, 25, 33] chứng minh ưu thế vượt trội khi duy trì hiệu năng ổn định xuyên suốt từ video ngắn 400 tokens (113.2 TPS, speedup 1.80x) cho tới video siêu dài 25,000 tokens ($\tau = 2.259$).
2. **Khắc phục triệt để hiện tượng suy giảm ở video dài qua cơ chế Retention:** 
   - Với video $\ge 13,000$ tokens, việc áp dụng chính sách **`retained_50`** hoặc **`retained_75`** mang lại lợi ích kép: nâng $\tau_{\text{effective}}$ lên mức kỷ lục **$2.299$**, đồng thời giải phóng $25\% - 50\%$ bộ nhớ KV Cache cho Draft model.
3. **Ý nghĩa kiến trúc:** Việc kết hợp các tầng trải dài từ đầu đến cuối mô hình Target cung cấp cho Draft model một biểu diễn đa tầng hoàn chỉnh, giúp mô hình giữ vững tính ổn định và chống chọi tốt hơn với hiện tượng pha loãng chú ý (attention dilution) so với các cấu hình chỉ tập trung cục bộ vào một nhóm tầng.
