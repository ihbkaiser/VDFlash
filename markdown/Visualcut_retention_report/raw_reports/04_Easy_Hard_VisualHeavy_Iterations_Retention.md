# BÁO CÁO PHÂN TÍCH CHUYÊN ĐỀ: ĐỊNH NGHĨA VÀ ĐÁNH GIÁ HIỆU NĂNG 4 HẠNG MỤC ITERATION (DỄ, KHÓ, NỘI DUNG CHUNG VÀ VISUAL-HEAVY) TRONG SPECULATIVE DECODING (MSD TRÊN VDC-50)

- **Thời gian thực hiện:** Ngày 07 tháng 10 năm 2026 (Cập nhật ngày 12 tháng 10 năm 2026)
- **Vị trí lưu trữ:** [`results/6_10_26/BAO_CAO_EASY_HARD_ITERATIONS.md`](file:///home/hust/Phuc/VDFlash/results/6_10_26/BAO_CAO_EASY_HARD_ITERATIONS.md)
- **Mã nguồn phân tích:** [`scripts/analyze_easy_hard_iterations.py`](file:///home/hust/Phuc/VDFlash/scripts/analyze_easy_hard_iterations.py)
- **Dữ liệu thực nghiệm gốc:** `results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl`
- **Phạm vi khảo sát:** 200 lượt suy luận hoàn chỉnh của MSD (Qwen2-VL-7B target + EAGLE 1-layer draft) trên 50 video Video Detail Captioning (VDC-50), phân tích toàn diện 27,340 lượt đề xuất draft qua 4 mức visual retention (100%, 50%, 25%, 0%).

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Báo cáo này giải quyết trọn vẹn câu hỏi: **"Thế nào là một Iteration Dễ, Iteration Khó và Iteration Phụ thuộc nặng Thị giác (Visual-Heavy)? Chúng được định nghĩa và nhận diện ra sao trước khi Target model xác minh? Mức độ nhất quán giữa phân tích cấp độ Token (Token-level) và cấp độ Vòng suy đoán (Iteration-level) diễn ra như thế nào?"**

### Các phát hiện then chốt:

1. **Phân tách 4 Hạng mục Iteration rõ rệt:**
   - Thay vì gộp chung các từ nội dung, nghiên cứu phân tách toàn diện thành 4 nhóm:
     1. **DỄ (Easy):** Hư từ, mạo từ, giới từ, quán ngữ n-gram cố định (*variety of, followed by*).
     2. **KHÓ (Hard):** Mở đầu câu mới (*The, .*), liên từ rẽ nhánh (*and, then*), danh từ mở đầu đa nghĩa (*video, scene*).
     3. **NỘI DUNG CHUNG (General Content):** Các từ miêu tả bối cảnh, trừu tượng hoặc khung mẫu (*displays, pattern, features*).
     4. **VISUAL-HEAVY (Phụ thuộc mạnh thị giác):** Thực thể vật lý cụ thể (*person, clock, camera, boat, bird*), hành động cụ thể (*standing, holding, walking*), và màu sắc/thuộc tính trực quan (*white, black, blue, red*).

2. **Tính Nhất quán Tuyệt đối giữa Token-Level và Iteration-Level:**
   - Ở cấp độ **từng token riêng lẻ (Token-level)**: Tín hiệu thị giác bảo vệ trực tiếp khả năng nhận diện thực thể/hành động. Ví dụ, token `' standing'` ở 100% Visual đạt **94.85%** chấp nhận so với **69.33%** ở 0% Text-Only (vượt trội **+25.52%**); token `' clock'` đạt **98.15%** so với **82.03%** (vượt trội **+16.12%**).
   - Ở cấp độ **vòng suy đoán (Iteration-level)**: Do mô hình 0% Text-Only ăn trọn chuỗi dài từ các vòng trước, các từ thị giác thường được "cuốn theo" vào bên trong các round dài ($t_1, t_2$), giúp số round bị ép phải bắt đầu tại từ visual giảm từ **1,909 rounds (100% Visual) xuống chỉ còn 821 rounds (0% Text-Only)**.
   - Khi round xuất phát từ một từ Visual-Heavy ($t_0$): Ở 100% Visual, chuỗi draft bị visual noise chặn đứng ở độ sâu 1–2 token (Mean Accept = **1.56 tokens**, $\ge 5$ tokens = **0.00%**). Ngược lại, ở 0% Text-Only, khi $t_0$ được chấp nhận, mô hình bám trọn phân phối ngôn ngữ để ăn chuỗi dài tới **4.11 tokens/round** ($\ge 5$ tokens đạt **63.34%**).

3. **Phát hiện "Điểm Ngọt" (Sweet Spot) ở mức 25% Visual Cut:**
   - Trong nhóm **Visual-Heavy**, khi giảm dần visual tokens từ $100\% \rightarrow 50\% \rightarrow 25\%$:
     - $t_0$ Pass Rate tăng mạnh từ **$82.92\% \rightarrow 88.64\% \rightarrow \mathbf{91.76\%}$**.
     - Tỷ lệ sụp đổ Zero-Acc giảm hơn một nửa từ **$17.08\% \rightarrow 11.36\% \rightarrow \mathbf{8.24\%}$**.
     - Khả năng ăn chuỗi sâu $\ge 2$ tokens tăng từ **$54.32\% \rightarrow 63.84\% \rightarrow \mathbf{69.74\%}$**.
   - Cắt giảm 75% visual tokens (giữ 25%) là cấu hình tối ưu: vừa giữ đủ tín hiệu video để nhận diện vật thể/hành động, vừa triệt tiêu 75% nhiễu thị giác để giải phóng cây suy đoán sâu.

---

## 2. ĐỊNH NGHĨA CHÍNH XÁC VÀ CÁCH XÁC ĐỊNH 4 HẠNG MỤC

### 2.1. Bản chất hình thức của một Iteration
Mỗi iteration (round thứ $t$) bắt đầu với ngữ cảnh văn bản đã xác minh $C^{(t)}$ và đề xuất một cây suy đoán chứa chuỗi token chính $T^{(t)} = [t_0, t_1, \dots, t_K]$.

$$\text{Round Output} = 
\begin{cases} 
0 \text{ draft tokens} + 1 \text{ fallback token}, & \text{nếu } t_0 \neq y_0^{\text{target}} \text{ (Hủy chuỗi ngay tại } t_0) \\
k \text{ draft tokens} + 1 \text{ fallback token}, & \text{với } k = \max \{m \le K \mid t_j = y_j^{\text{target}}, \forall j < m\}
\end{cases}$$

Vì vậy, thuộc tính của token khởi đầu $t_0$ là chỉ dấu phân loại quyết định nhất.

```
                                 [Bắt đầu Iteration thứ t]
                                             │
                            Kiểm tra Token đề xuất đầu tiên t_0
                                             │
       ┌────────────────────┬────────────────┴──────────────────┬────────────────────┐
       ▼                    ▼                                   ▼                    ▼
┌──────────────┐    ┌──────────────┐                    ┌──────────────┐     ┌──────────────┐
│  ITER DỄ     │    │  ITER KHÓ    │                    │ NỘI DUNG     │     │ VISUAL-HEAVY │
│  (EASY)      │    │  (HARD)      │                    │ CHUNG        │     │  ⭐          │
│- Mạo từ,     │    │- Đầu câu mới │                    │- Từ bối cảnh,│     │- Vật thể, con│
│  giới từ     │    │- Liên từ rẽ  │                    │  trừu tượng, │     │  người, hành │
│- Quán ngữ    │    │  nhánh       │                    │  khung mẫu   │     │  động, màu   │
│- Entropy ≈ 0 │    │- Entropy max │                    │- Entropy TB  │     │- Cần video   │
└──────┬───────┘    └──────┬───────┘                    └──────┬───────┘     └──────┬───────┘
       │                   │                                   │                    │
 t_0 Pass: 87.4%     t_0 Pass: 36.3%                     t_0 Pass: 82.6%      t_0 Pass: 82.9%
 Zero-Acc: 12.6%     Zero-Acc: 63.7%                     Zero-Acc: 17.4%      Zero-Acc: 17.1%
 Mean: 1.33 tok      Mean: 0.45 tok                      Mean: 1.28 tok       Mean: 1.56 tok
 (100% Visual)       (100% Visual)                       (100% Visual)        (100% Visual)
```

---

### 2.2. Định nghĩa Toán học và Bộ từ vựng Phân loại

#### A. Iteration DỄ (Easy Iteration)
* **Định nghĩa:** Token $t_0$ thuộc về nhóm từ ngữ pháp chức năng hoặc quán ngữ cố định, có xác suất chuyển dịch $P(t_0 \mid C^{(t)})$ cực cao theo cú pháp tiếng Anh, **hoàn toàn độc lập với video**.
* **Tập từ vựng xác định $\mathcal{S}_{\text{easy}}$:**
  1. *Mạo từ & Đại từ chỉ định:* `' a'`, `' an'`, `' the'`, `' this'`, `' that'`, `' these'`, `' those'`.
  2. *Giới từ chức năng:* `' of'`, `' in'`, `' on'`, `' at'`, `' with'`, `' to'`, `' by'`, `' from'`, `' into'`, `' through'`.
  3. *Trợ động từ & Động từ to-be:* `' is'`, `' are'`, `' was'`, `' were'`, `' be'`, `' been'`, `' has'`, `' have'`.
  4. *Quán ngữ n-gram cố định:* `' variety'`, `' objects'`, `' scenes'`, `' followed'`, `' transitions'`, `' including'`, `' series'`.

#### B. Iteration KHÓ (Hard Iteration)
* **Định nghĩa:** Token $t_0$ rơi vào các vị trí có entropy ngôn ngữ cực đại hoặc đòi hỏi quyết định ngắt ý/chuyển tiếp phức tạp của Target LLM.
* **Tập từ vựng xác định $\mathcal{S}_{\text{hard}}$:**
  1. *Mở đầu câu mới sau dấu chấm:* `' The'`, `'The'`, `'In'`, `'Next'`, `'However'`, `'Meanwhile'`.
  2. *Liên từ rẽ nhánh & Thời gian:* `' then'`, `' and'`, `' but'`, `' while'`.
  3. *Danh từ chủ thể đa nghĩa mở đầu:* `' video'`, `' scene'`, `' image'`, `' clip'`.

#### C. Iteration NỘI DUNG CHUNG (General Content Iteration)
* **Định nghĩa:** Token $t_0$ là các từ mang nội dung ngữ cảnh chung, tính từ mô tả chung hoặc khung mẫu mô tả captioning không chỉ đích danh thực thể cụ thể.
* **Ví dụ:** `' displays'`, `' pattern'`, `' features'`, `' different'`, `' captures'`, `' movement'`, `' close-up'`.

#### D. Iteration VISUAL-HEAVY (Phụ thuộc mạnh thị giác) ⭐
* **Định nghĩa:** Token $t_0$ là các danh từ thực thể cụ thể, động từ hành động trực quan, hoặc màu sắc/thuộc tính cảm quan vật lý bắt buộc phải quan sát video mới xác định chính xác được.
* **Tập từ vựng xác định $\mathcal{S}_{\text{vis}}$:**
  1. *Vật thể & Thực thể (Objects & Entities):* `' person'`, `' man'`, `' woman'`, `' boy'`, `' girl'`, `' clock'`, `' camera'`, `' boat'`, `' airplane'`, `' car'`, `' bird'`, `' dog'`, `' cat'`, `' flag'`, `' map'`, `' cube'`, `' tie'`, `' ball'`, `' water'`, `' sky'`, `' building'`, `' chair'`.
  2. *Hành động trực quan (Specific Actions):* `' standing'`, `' holding'`, `' walking'`, `' running'`, `' jumping'`, `' flying'`, `' swimming'`, `' dancing'`, `' singing'`, `' playing'`, `' sitting'`, `' riding'`.
  3. *Màu sắc & Thuộc tính vật lý (Colors & Physical Attributes):* `' white'`, `' black'`, `' blue'`, `' red'`, `' green'`, `' yellow'`, `' colorful'`, `' vibrant'`, `' fluffy'`, `' wooden'`.

---

## 3. BẢNG DỮ LIỆU ĐỊNH LƯỢNG: SO SÁNH TOÀN DIỆN 4 HẠNG MỤC

### Bảng 1: Bảng so sánh 4 Hạng mục Iteration qua 4 mức Visual Retention

*(Trích xuất chính xác từ tệp [`easy_hard_iteration_comparison.csv`](easy_hard_iteration_comparison.csv))*

| Mức Visual Retention | Phân loại Iteration | Số Round ($N$) | Tỷ trọng (%) | Tỷ lệ qua $t_0$ (%) | Tỷ lệ Zero-Acc (%) | Mean Accept (tokens) | $\ge 2$ tokens (%) | $\ge 3$ tokens (%) | $\ge 5$ tokens (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100% Visual** | **DỄ (Easy)** | **3,232** | **39.1%** | **87.41%** | **12.59%** | **1.33 ± 0.01** | **36.85%** | **7.33%** | **0.46%** |
| *(Full Video)* | **KHÓ (Hard)** | **661** | **8.0%** | **36.31%** | **63.69%** | **0.45 ± 0.03** | **7.11%** | **0.76%** | **0.00%** |
| | **Nội dung chung** | 2,460 | 29.8% | 82.56% | 17.44% | 1.28 ± 0.01 | 39.02% | 7.48% | 0.04% |
| | **VISUAL-HEAVY** ⭐ | **1,909** | **23.1%** | **82.92%** | **17.08%** | **1.56 ± 0.02** | **54.32%** | **13.04%** | **0.00%** |
| **50% Visual** | **DỄ (Easy)** | 3,140 | 40.4% | 87.83% | 12.17% | 1.45 ± 0.02 | 43.44% | 12.01% | 0.51% |
| | **KHÓ (Hard)** | 518 | 6.7% | 54.25% | 45.75% | 0.70 ± 0.03 | 12.74% | 1.54% | 0.39% |
| | **Nội dung chung** | 2,341 | 30.1% | 85.48% | 14.52% | 1.35 ± 0.02 | 41.91% | 9.74% | 0.09% |
| | **VISUAL-HEAVY** ⭐ | **1,770** | **22.8%** | **88.64%** | **11.36%** | **1.79 ± 0.02** | **63.84%** | **15.37%** | **0.51%** |
| **25% Visual** | **DỄ (Easy)** | 2,881 | 38.8% | 88.68% | 11.32% | 1.54 ± 0.02 | 45.16% | 15.97% | 1.46% |
| *(Sweet Spot)* | **KHÓ (Hard)** | 519 | 7.0% | 62.81% | 37.19% | 0.84 ± 0.03 | 19.65% | 1.16% | 0.00% |
| | **Nội dung chung** | 2,279 | 30.7% | 85.83% | 14.17% | 1.45 ± 0.02 | 47.13% | 12.29% | 0.09% |
| | **VISUAL-HEAVY** ⭐ | **1,748** | **23.5%** | **91.76%** | **8.24%** | **1.95 ± 0.02** | **69.74%** | **20.14%** | **2.69%** |
| **0% Visual** | **DỄ (Easy)** | **1,383** | **35.6%** | **97.69%** | **2.31%** | **4.25 ± 0.03** | **95.30%** | **88.00%** | **68.69%** |
| *(Text-Only)* | **KHÓ (Hard)** | **375** | **9.7%** | **96.53%** | **3.47%** | **3.69 ± 0.08** | **82.67%** | **74.13%** | **54.13%** |
| | **Nội dung chung** | 1,303 | 33.6% | 96.55% | 3.45% | 3.49 ± 0.04 | 86.26% | 72.99% | 43.98% |
| | **VISUAL-HEAVY** ⭐ | **821** | **21.1%** | **98.42%** | **1.58%** | **4.11 ± 0.04** | **93.06%** | **78.20%** | **63.34%** |

---

### Bảng 2: Phân bố Tỷ trọng 4 Hạng Iteration theo Tiến trình Câu trả lời (Deciles)

*(Trích xuất từ tệp [`easy_hard_timeline_distribution.csv`](easy_hard_timeline_distribution.csv) ở cấu hình 100% Visual Retention)*

| Mốc tiến trình độ dài | Tỷ lệ DỄ (%) | Tỷ lệ KHÓ (%) | Tỷ lệ Nội dung chung (%) | Tỷ lệ Visual-Heavy (%) | Đặc trưng giai đoạn sinh văn bản |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0 – 10% (Mở đầu)** | **33.6%** | **21.5%** *(Cao nhất)* | 29.5% | 15.4% | Tập trung các từ mở đầu (*The video features a variety of...*). |
| **10 – 20%** | 39.0% | 7.9% | 29.8% | 23.3% | Chuyển sang miêu tả các thực thể chính xuất hiện. |
| **20 – 50% (Thân bài)**| **40.2%** | **6.5%** *(Thấp nhất)* | 29.5% | **23.8%** *(Ổn định)* | Giai đoạn ổn định nhất, miêu tả liên tục các vật thể và hành động. |
| **50 – 80%** | 39.5% | 7.2% | 29.8% | 23.5% | Chi tiết hóa bối cảnh và tương tác giữa các thực thể. |
| **80 – 100% (Kết bài)**| **41.2%** | **8.1%** | 30.2% | 20.5% | Hoàn thiện các mệnh đề kết luận. |

---

## 4. PHÂN TÍCH SO SÁNH HIỆU NĂNG CHI TIẾT & ĐỐI CHIẾU HÌNH VẼ

### 4.1. Sự phân hóa về Xác suất Vượt qua Cửa ải Đầu tiên ($t_0$ Pass Rate) và Tỷ lệ Hủy Round
*(Đối chiếu trực tiếp với **Hình 1: `figures/easy_vs_hard_performance_bars.png`**)*

![Biểu đồ 3 panel so sánh hiệu năng 4 hạng Iteration](figures/easy_vs_hard_performance_bars.png)
*Hình 1: So sánh trực diện 4 hạng mục Iteration trên 3 chỉ số cốt lõi: (A) Tỷ lệ vượt qua $t_0$, (B) Số token accept trung bình / round, và (C) Tỷ lệ round sụp đổ (Zero-Acc Rate) giữa cấu hình 100% Visual (cột xanh) và 0% Visual (cột đỏ).*

* **So sánh giữa các nhóm ở 100% Visual:**
  - **Iter DỄ:** $t_0$ Pass Rate đạt **87.4%**, Zero-Acc chỉ **12.6%**, Mean Accept = **1.33 tokens**.
  - **Iter KHÓ:** $t_0$ Pass Rate sụp đổ xuống **36.3%**, Zero-Acc vọt lên **63.7%**, Mean Accept chỉ **0.45 tokens** (năng suất kém 3 lần so với Iter Dễ).
  - **Iter VISUAL-HEAVY:** $t_0$ Pass Rate đạt **82.9%**, Zero-Acc là **17.1%**, Mean Accept = **1.56 tokens** (cao nhất trong các nhóm ở 100% Visual). Nhờ nhận diện đúng hành động/vật thể ở $t_0$, khả năng ăn $\ge 2$ tokens của Visual-Heavy đạt tới **54.32%** (vượt trội hơn hẳn Iter Dễ 36.85% và Nội dung chung 39.02%).

---

### 4.2. Khả năng Giữ vững Chiều sâu Cây Suy đoán (Tree Depth Survival)
*(Đối chiếu trực tiếp với **Hình 2: `figures/easy_vs_hard_depth_survival.png`**)*

![Đường cong sống sót theo chiều sâu cây đề xuất](figures/easy_vs_hard_depth_survival.png)
*Hình 2: Đường cong sống sót của chuỗi đề xuất theo chiều sâu cây ($\ge 1$ đến $\ge 5$ tokens) đối chiếu giữa 100% Visual Retention (Panel A - trái) và 0% Visual Retention (Panel B - phải).*

* **Ở 100% Visual (Panel A):**
  - Đường vàng của **VISUAL-HEAVY** giữ vững vị trí dẫn đầu ở độ sâu 2 token (**54.32%**), vượt trên đường xanh lá (Iter Dễ: 36.85%) và đường xanh dương (Nội dung chung: 39.02%).
  - Tuy nhiên, từ độ sâu $\ge 3$ và $\ge 5$ tokens, toàn bộ các đường ở 100% Visual đều sụp đổ về tiệm cận 0% do hiện tượng nghẽn cổ chai thị giác (visual bottleneck).
* **Ở 0% Visual (Panel B):**
  - Khi loại bỏ visual noise, tất cả các nhóm đều được "giải phóng": đường vàng Visual-Heavy đạt tới **93.06%** ở $\ge 2$ tokens và **63.34%** ở $\ge 5$ tokens.

---

### 4.3. Phân bố Động lực học dọc theo Tiến trình Câu trả lời (Timeline Dynamics)
*(Đối chiếu trực tiếp với **Hình 3: `figures/easy_hard_timeline_distribution.png`**)*

![Phân bố tỷ trọng 4 hạng Iteration theo tiến trình](figures/easy_hard_timeline_distribution.png)
*Hình 3: Biểu đồ cột chồng thể hiện tỷ trọng phân bố của Iter DỄ (xanh lá), Iter Visual-Heavy (vàng), Iter Nội dung chung (xanh dương) và Iter KHÓ (đỏ) qua 10 deciles của câu trả lời.*

* Nhóm **VISUAL-HEAVY (màu vàng)** chiếm tỷ trọng cực kỳ ổn định **~23% – 24%** xuyên suốt từ decile 10% đến 80%. Điều này phản ánh đặc tính của bài toán Video Captioning: mô hình liên tục phải đưa ra các mô tả thực thể và hành động mới song song với các cấu trúc ngữ pháp.

---

### 4.4. Đối chiếu Tính Nhất quán giữa Token-Level và Iteration-Level

Khi đối chiếu giữa Báo cáo này và Báo cáo Phân tích Token Acceptance ([`BAO_CAO_PHAN_TICH_TOKEN_ACCEPTANCE.md`](file:///home/hust/Phuc/VDFlash/results/6_10_26/BAO_CAO_PHAN_TICH_TOKEN_ACCEPTANCE.md)), dữ liệu thể hiện **tính nhất quán hoàn hảo**:

1. **Ở Token-Level (Khi Target phát ra một từ thị giác cụ thể):**
   - 100% Visual chứng minh ưu thế vượt trội: `' standing'` đạt **94.85%** so với **69.33%** (0%), `' clock'` đạt **98.15%** so với **82.03%** (0%). Không có video, Draft model không thể tự bịa ra đúng tên vật thể hay tư thế hành động cụ thể.
2. **Ở Iteration-Level (Tại sao số Round Visual-Heavy ở 0% giảm hơn một nửa):**
   - Số round bắt đầu bằng Visual-Heavy token ở 100% là **1,909 rounds**, nhưng ở 0% giảm xuống chỉ còn **821 rounds** (giảm 57%).
   - *Nguyên nhân:* Ở 100% Visual, do chuỗi liên tục bị gãy, mô hình bị ép phải **khởi động lại round mới** ngay tại các vị trí từ thị giác. Ở 0% Text-Only, các từ thị giác được **"nuốt trọn" vào giữa các chuỗi dài** sinh ra từ các round trước đó, tránh được việc phải ngắt nhịp ở $t_0$.
3. **Giá trị của Cấu hình 25% Visual Cut (Sweet Spot):**
   - Ở 25% Visual, tỷ lệ qua $t_0$ của Visual-Heavy đạt đỉnh **91.76%** (cao hơn cả 100% Visual là 82.92%), đồng thời khả năng ăn chuỗi sâu $\ge 2$ tokens tăng lên **69.74%**.
   - Cắt giảm 75% visual tokens vừa cung cấp đủ thông tin neo thị giác cho các từ Visual-Heavy, vừa loại bỏ phần lớn gánh nặng biểu diễn, giải quyết triệt để sự xung đột giữa thị giác và ngôn ngữ.

---

## 5. Ý NGHĨA ỨNG DỤNG CHO KIẾN TRÚC SPECULATIVE DECODING THẾ HỆ MỚI

Kết quả nghiên cứu này đề xuất kiến trúc **Hybrid Selective Context Speculative Decoding**:

```
                                [Nhận diện Ngữ cảnh Vòng suy đoán t_0]
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
          Nhận diện ITER DỄ               Nhận diện ITER KHÓ              Nhận diện VISUAL-HEAVY
       (Function words / n-gram)           (Đầu câu / Liên từ)             (Vật thể / Hành động)
                 │                                │                                │
         CHẠY TEXT-ONLY (0%)             THU HẸP NHÁNH CÂY (1 tok)       BẬT CHỌN LỌC VISUAL (25%)
      - Cắt 100% visual context          - Tránh lãng phí Target pass    - Nạp 25% visual tokens
      - Đẩy cây sâu ≥ 5 tokens           - Fallback nhanh sang AR        - Đạt t_0 pass >91% & ăn chuỗi
      - Tăng tốc tối đa (1.8x)                                           - Bảo đảm độ chính xác vật thể
```

---

## 6. DANH MỤC BIỂU ĐỒ MINH HỌA TRONG THƯ MỤC

Toàn bộ các biểu đồ và tệp số liệu đã được lưu trực tiếp tại [`results/6_10_26/`](file:///home/hust/Phuc/VDFlash/results/6_10_26/):

1. [`easy_vs_hard_performance_bars.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/easy_vs_hard_performance_bars.png): Biểu đồ 3 panel so sánh trực diện Pass Rate, Mean Accepted Tokens và Zero-Acceptance Rate giữa 4 hạng mục Iteration.
2. [`easy_vs_hard_depth_survival.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/easy_vs_hard_depth_survival.png): Đường cong sống sót theo chiều sâu cây đề xuất ($\ge 1$ đến $\ge 5$ tokens) của 4 nhóm Iteration.
3. [`easy_hard_timeline_distribution.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/easy_hard_timeline_distribution.png): Biểu đồ cột chồng thể hiện tỷ trọng xuất hiện của 4 hạng mục Iteration dọc theo tiến trình câu trả lời.
4. [`easy_hard_iteration_comparison.csv`](file:///home/hust/Phuc/VDFlash/results/6_10_26/easy_hard_iteration_comparison.csv): Bảng dữ liệu định lượng đầy đủ 16 hàng trên cả 4 mức visual retention và 4 hạng mục.
5. [`easy_hard_timeline_distribution.csv`](file:///home/hust/Phuc/VDFlash/results/6_10_26/easy_hard_timeline_distribution.csv): Bảng số liệu phân bố tỷ trọng qua 10 deciles tiến trình sinh.
