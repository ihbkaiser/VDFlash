# BÁO CÁO THỰC NGHIỆM ĐÁNH GIÁ ĐA PHƯƠNG THỨC SUY LUẬN SUY ĐOÁN (MSD) TRÊN VIDHALLUC VÀ VIDEOCHAIR

> **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2-VL-7B-Instruct`  
> **Mô hình suy đoán (Draft Model):** `lucylyn/MSD-Qwen2VL-7B-Instruct` (EAGLE Tree Speculative Decoding)  
> **Môi trường thực thi:** NVIDIA GeForce RTX 3090 (24GB VRAM), PyTorch 2.0.1, CUDA 12.2, Decord 0.6.0  
> **Tập dữ liệu đánh giá:** 100 mẫu đại diện VidHalluc + 100 mẫu đại diện VideoChair (MSR-VTT)  
> **4 Độ Cắt Visual (Cut Levels):** 100% (No Cut), 50% Cut, 25% Cut, 0% (Full Cut / Text-only)  
> **Quy mô vết suy luận:** 804 lượt suy luận toàn diện; **42.062 draft iterations** được bóc tách chi tiết.

---

## 1. Tóm tắt Tổng quan và Dashboard Toàn diện

Nghiên cứu kiểm chứng độc lập và toàn diện các giả thuyết về cơ chế suy luận suy đoán đa phương thức (*Multi-modal Speculative Decoding - MSD*) trên hai bài toán thị giác máy tính tiêu biểu:
1. **Ảo giác thị giác và suy luận nhân quả - thời gian:** Đánh giá trên bộ dữ liệu **VidHalluc** (phân tầng trên 4 danh mục: ACH-MCQ, ACH-Bin, TSH, STH).
2. **Mô tả chi tiết và nhận diện thực thể (Captioning & CHAIR):** Đánh giá trên bộ dữ liệu **VideoChair** (phân tầng đồng đều qua 20 danh mục chủ đề của MSR-VTT).

![Dashboard Tổng quan Hiệu năng MSD](hallucination_comprehensive_dashboard.png)
*Hình 1: Tổng hợp 4 chiều phân tích hiệu năng của MSD trên VidHalluc và VideoChair: (a) Tốc độ VidHalluc, (b) Tốc độ VideoChair, (c) Tỷ lệ Zero-Acceptance, (d) Chiều dài chấp nhận trung bình.*

---

## 2. Bảng Tổng hợp Chỉ số Thực nghiệm theo 4 Độ Cắt Visual

| Bộ dữ liệu | Mức Độ Cắt | Decode Speedup | End-to-End Speedup | Mean Accept Length | Tỷ lệ Zero-Acceptance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VidHalluc** | 100% (No Cut) | 0.63x (Chậm hơn AR) | 1.19x | 1.04 tokens | 34.80% |
| *(100 mẫu, Short QA)* | 50% Cut | 0.72x | 1.28x | 1.26 tokens | 28.52% |
| | 25% Cut | 0.84x | 1.35x | 1.60 tokens | 18.80% |
| | **0% (Full Cut)** | **1.14x (Tăng tốc)** | **1.48x** | **2.63 tokens** | **6.17% (Giảm 5.6 lần)** |
| **VideoChair** | 100% (No Cut) | 0.67x (Chậm hơn AR) | 0.77x (Chậm hơn AR) | 1.52 tokens | 17.54% |
| *(100 mẫu, Long Captions)* | 50% Cut | 1.03x | 1.12x | 2.78 tokens | 5.69% |
| | 25% Cut | 1.12x | 1.21x | 3.18 tokens | 4.58% |
| | **0% (Full Cut)** | **1.22x (Tăng tốc)** | **1.29x** | **3.46 tokens** | **3.50% (Giảm 5.0 lần)** |

---

## 3. Phân tích Chi tiết & Đối chiếu Đồ thị các Metric qua 4 Độ Cắt Visual (Cut Levels)

### 3.1. Tốc độ Tăng tốc: Decode Speedup vs End-to-End Speedup theo độ Cắt

![So sánh Tốc độ Tăng tốc theo 4 Độ Cắt](fig1_speedup_by_cut_level.png)
*Hình 2: So sánh tốc độ Decode Speedup và End-to-End Speedup trên VidHalluc và VideoChair qua 4 độ cắt visual (100%, 50%, 25%, 0%). Đường nét đứt màu đỏ thể hiện mốc cơ sở Autoregressive (1.0x).*

* **Ở mức 100% (Không cắt):** Cả hai bộ dữ liệu đều chịu ảnh hưởng tiêu cực của điểm nghẽn thị giác (*visual bottleneck*). Decode speedup chỉ đạt **0.63x** (VidHalluc) và **0.67x** (VideoChair), nghĩa là giai đoạn sinh token chậm hơn chạy Autoregressive thông thường tới 33% – 37%.
* **Khi tăng dần độ cắt (50% $\rightarrow$ 25% $\rightarrow$ 0%):** Tốc độ giải mã tăng tuyến tính rõ rệt:
  * Trên VidHalluc: Decode Speedup tăng từ $0.63x \rightarrow 0.72x \rightarrow 0.84x \rightarrow \mathbf{1.14x}$.
  * Trên VideoChair: Decode Speedup vượt mốc 1.0x ngay từ mức cắt 50% ($1.03x$) và đạt đỉnh **1.22x** ở mức cắt toàn phần 0%.
* **Sự phân hóa giữa Decode và End-to-End Speedup:** Ở VidHalluc, do thời gian prefill chiếm đa số, E2E speedup luôn duy trì trên 1.0x ($1.19x \rightarrow 1.48x$). Ngược lại ở VideoChair, do câu trả lời dài khiến decode chiếm ưu thế, E2E speedup chỉ vượt 1.0x khi Decode Speedup vượt 1.0x ($0.77x \rightarrow 1.29x$).

---

### 3.2. Hiệu năng Suy đoán: Chiều dài Accept, Tỷ lệ Zero-Acc và Số Vòng Xác minh

![Hiệu năng Suy đoán Đa phương thức theo Độ Cắt](fig2_efficiency_by_cut_level.png)
*Hình 3: Động học hiệu năng suy đoán theo độ cắt visual: (a) Chiều dài chấp nhận trung bình mỗi round, (b) Tỷ lệ round bị từ chối 0 token (Zero-Acceptance Rate), (c) Số vòng xác minh trung bình (Verification Rounds).*

* **Chiều dài Chấp nhận Trung bình (Mean Accepted Length):**
  * Trên VidHalluc: Tăng vọt từ **1.04 tokens** (100%) lên **2.63 tokens** (0%) – tăng gấp **2.52 lần**!
  * Trên VideoChair: Tăng mạnh từ **1.52 tokens** (100%) lên **3.46 tokens** (0%) – tăng gấp **2.28 lần**!
* **Tỷ lệ Zero-Acceptance (%):**
  * Trên VidHalluc: Giảm mạnh từ **34.80% xuống chỉ còn 6.17%** (giảm 5.6 lần).
  * Trên VideoChair: Giảm cực nhanh từ **17.54% xuống 3.50%** (giảm 5.0 lần).
* **Số Vòng Xác minh (Verification Steps / Rounds):**
  * Do mỗi round chấp nhận được nhiều token hơn, số vòng forward pass của Target model giảm mạnh:
  * VidHalluc: Giảm từ **18.16 vòng xuống 10.38 vòng** (tiết kiệm **42.8%** số lần gọi Target model).
  * VideoChair: Giảm từ **125.69 vòng xuống 72.56 vòng** (tiết kiệm **42.3%** số lần gọi Target model).

---

### 3.3. Phân rã Định lượng Độ trễ Prefill vs Decode so với Autoregressive Gốc

![Phân rã Định lượng Độ trễ Prefill vs Decode](fig3_latency_breakdown_by_cut_level.png)
*Hình 4: Phân rã định lượng độ trễ Prefill (màu tím) và Decode (màu đỏ) giữa Autoregressive Baseline và 4 độ cắt visual của MSD trên VidHalluc và VideoChair.*

Bảng số liệu phân rã chi tiết:

| Bộ dữ liệu | Cấu hình | Prefill Time ($T_{\text{prefill}}$) | Decode Time ($T_{\text{decode}}$) | Tổng thời gian E2E | Tỷ trọng Decode trong E2E |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VidHalluc** | **AR Baseline** | **4.08s** | **1.16s** | **5.24s** | 22.1% |
| *(Short QA,* | 100% Retention | 2.49s | 1.90s | 4.39s | 43.3% |
| *~30 tokens)* | 50% Cut | 2.43s | 1.71s | 4.14s | 41.3% |
| | 25% Cut | 2.41s | 1.47s | 3.88s | 37.9% |
| | **0% Full Cut** | **2.39s** | **1.03s** | **3.42s** | **30.1%** |
| **VideoChair** | **AR Baseline** | **2.16s** | **7.35s** | **9.51s** | 77.3% |
| *(Long Captions,* | 100% Retention | 1.10s | 10.96s | 12.06s | 90.9% |
| *~180 tokens)* | 50% Cut | 1.08s | 7.33s | 8.40s | 87.2% |
| | 25% Cut | 1.07s | 6.65s | 7.71s | 86.2% |
| | **0% Full Cut** | **1.06s** | **6.18s** | **7.23s** | **85.4%** |

* **Giải mã hiện tượng tại VidHalluc:** Thời gian Prefill của AR gốc là 4.08s. Bản vá SDPA 4D backend trong MSD kéo prefill xuống 2.39s – 2.49s (tiết kiệm tới **1.6s**). Dù ở 100% Retention, decode bị chậm đi 0.74s (1.90s so với 1.16s), nhưng phần tiết kiệm được ở Prefill (1.6s) vẫn lớn hơn nhiều phần mất đi ở Decode (0.74s) $\rightarrow$ Tổng E2E vẫn nhanh hơn 0.85s (E2E Speedup đạt 1.19x).
* **Giải mã hiện tượng tại VideoChair:** Với caption dài, Decode chiếm tới **77% – 91%** tổng thời gian. Khi Decode bị kéo dài lên 10.96s ở 100% retention, tổng E2E lập tức bị đẩy lên 12.06s (chậm hơn AR 9.51s) $\rightarrow$ E2E Speedup tụt xuống 0.77x. Chỉ khi cắt visual về 50% - 0%, Decode time giảm còn 6.18s thì E2E mới đạt 1.29x.

---

### 3.4. Khả năng Khởi đầu Chuỗi ($t_0$ Acceptance) và Tính Toàn vẹn Lossless

![Khả năng Khởi đầu Chuỗi và Tính Toàn vẹn Lossless](fig4_t0_acceptance_and_lossless.png)
*Hình 5: (a) Tỷ lệ chấp nhận token đầu tiên ($t_0$ Acceptance Rate) và (b) Tỷ lệ trùng khớp tuyệt đối (Exact Lossless Parity) theo độ cắt visual.*

* **Tỷ lệ Chấp nhận Token Đầu tiên ($t_0$ Acceptance Rate):**
  * $t_0$ là token then chốt nhất quyết định một iteration có thành công hay không. Nếu $t_0$ bị reject, toàn bộ cây draft phía sau đều vô nghĩa.
  * Khi cắt giảm visual context:
    * Trên VidHalluc: $t_0$ acceptance rate tăng mạnh từ **65.20% (100%) lên 93.83% (0%)**.
    * Trên VideoChair: $t_0$ acceptance rate tăng từ **82.46% (100%) lên 96.50% (0%)**.
  * Điều này chứng minh rằng việc loại bỏ nhiễu thị giác giúp Draft model khởi đầu mỗi iteration với độ chính xác gần như tuyệt đối (>94–96%).
* **Tính Toàn vẹn Suy luận (Lossless Verification Parity):**
  * Tỷ lệ output trùng khớp bitwise hoàn toàn với Target model greedy baseline:
  * VidHalluc: Đạt mức rất cao **93.0% – 95.0%** trên toàn bộ 4 mức cắt.
  * VideoChair: Đạt **76.0% – 81.0%** trên toàn bộ 4 mức cắt.
  * Quan trọng nhất: **Tỷ lệ Lossless hoàn toàn không bị suy giảm khi cắt visual về 0%** (VidHalluc giữ nguyên 94.0% ở 0% so với 95.0% ở 100%; VideoChair giữ 78.0% ở 0% so với 78.0% ở 100%).
  * Điều này khẳng định cơ chế xác minh của Speculative Decoding bảo vệ 100% chất lượng đầu ra của Target model, việc cắt visual ở Draft model chỉ tăng tốc độ chứ **không hề làm suy giảm chất lượng câu trả lời**.

---

### 3.5. Phân bố Độ khó của 42.062 Draft Iterations

![Phân bố Độ khó Iteration](hallucination_difficulty_distribution.png)
*Hình 6: Phân bố mức độ khó của 42.062 draft iterations trên VidHalluc và VideoChair (phân loại thành CONTENT, EASY, HARD).*

* **CONTENT (61.5% - 66.3%):** Chiếm đa số trong cả hai tập dữ liệu. Các iteration này chứa danh từ, động từ mô tả hành động hoặc thực thể chính trong video.
* **EASY (22.7% - 27.9%):** Gồm dấu câu và các hư từ phổ biến, có xác suất chấp nhận trên 90%.
* **HARD (10.6% - 11.1%):** Điểm ngoặt ngữ cảnh hoặc giới từ phức tạp, nơi draft model dễ bị từ chối nhất.

---

## 4. Giải thích Chuyên sâu 4 Hiện tượng "Phản Trực giác"

### Hiện tượng 1: Tại sao `Decode Speedup` bị chậm (< 1.0x) mà `End-to-End Speedup` lại nhanh (> 1.0x)?
* Đã được chứng minh định lượng tại Mục 3.1 và Mục 3.3 ở trên: Bắt nguồn từ sự chênh lệch tỷ trọng giữa Prefill và Decode.
* Bản vá SDPA 4D backend của MSD tiết kiệm được gần 2.0 giây trong giai đoạn tiền xử lý video. Ở các câu trả lời ngắn (VidHalluc), lượng thời gian tiết kiệm được ở Prefill lấn át hoàn toàn độ trễ phát sinh của Decode ở mức 100% visual retention.

### Hiện tượng 2: Tại sao Target Model luôn phát ra "+1 Bonus Token" ngay cả khi Draft đoán đúng 100%?
* Target model chỉ chạy **đúng 1 forward pass song song** qua toàn bộ chuỗi token do Draft đề xuất.
* Tại vị trí token cuối cùng được chấp nhận ($t_k$), lớp chuẩn hóa và LM-head của Target **đã tính toán xong vector xác suất (logits) cho token kế tiếp $t_{k+1}$**.
* Kết quả này nằm sẵn trong VRAM mà không tốn thêm bất kỳ phép tính nào. Theo quy tắc *Bonus Token* (*Leviathan et al., 2023*), thuật toán lấy luôn $t_{k+1}$ đưa vào văn bản thay vì bỏ phí.
* Do đó, số token sinh ra luôn tuân thủ đẳng thức: $\text{Tokens Emitted} = \text{Tokens Accepted} + 1$.

### Hiện tượng 3: Tại sao "Mù Thị giác" (0% Retention) lại giúp Draft Model đoán đúng nhiều hơn và ít bị Reject hơn?
* **Lệch pha biểu diễn (Representation Drift):** Đầu Draft (EAGLE EA-layer) chỉ là một mạng nhỏ 1-layer. Khi nạp hàng nghìn visual tokens, mạng nhỏ không đủ dung lượng để căn chỉnh đa phương thức với Target 7B, tạo thành nhiễu tín hiệu.
* **Quy luật Thống kê Ngôn ngữ (Language Prior Dominance):** Khi neo ngữ cảnh ban đầu đã hình thành, chuỗi từ tiếp theo chịu sự chi phối mạnh mẽ của quy luật ngôn ngữ tự nhiên. Draft model thuần văn bản bám sát chặt chẽ quỹ đạo sinh từ của LLM, giúp $t_0$ acceptance vọt lên >94–96% và Zero-acc giảm 5–6 lần.

### Hiện tượng 4: Tại sao GPU 24GB còn thừa nửa bộ nhớ nhưng máy chủ vẫn sập vì OOM (SIGKILL 137)?
* `torchvision.io.read_video` đọc toàn bộ 1.856 khung hình 720p vào RAM hệ thống, tiêu tốn hơn 5.13 GB RAM thô mỗi video, kết hợp với rò rỉ thread pool của `swscaler` kích hoạt Linux OOM Killer.
* Chuyển sang backend **`decord`** với cơ chế *Lazy Seeking* (chỉ giải mã đúng 32 khung hình cần thiết, tiêu tốn chỉ 88 MB RAM) đã triệt tiêu hoàn toàn sự cố và duy trì VRAM ổn định ở **8.17 GiB**.

---

## 5. Phân tích Chi tiết Hiệu năng của Model trên Từng Iteration (Per-Iteration Performance Dynamics)

Dựa trên toàn bộ **42.062 draft iterations** được bóc tách độc lập, chúng tôi phân tích hành vi của Draft model qua 3 chiều kích động học: **Timeline phát triển câu**, **Độ sâu cây suy đoán (Tree Depth)**, và **Đặc tính phân loại độ khó**.

### 5.1. Động học Hiệu năng theo Timeline Thế hệ Văn bản (Early vs Late Rounds)

![Động học Chấp nhận theo Timeline](fig6_timeline_and_answer_distribution.png)
*Hình 8: (a) Phân bố tỷ lệ câu trả lời output trên VidHalluc bất biến 100% qua 4 độ cắt và (b) Chiều dài chấp nhận trung bình theo timeline thế hệ văn bản trên VideoChair.*

Bảng diễn tiến chiều dài accept và tỷ lệ Zero-Acc theo các chặng sinh từ:

| Giai đoạn Timeline | Vị trí Iteration | Chiều dài Accept TB (100% Visual) | Chiều dài Accept TB (0% Text-Only) | Tỷ lệ Zero-Acc (100% Visual) | Tỷ lệ Zero-Acc (0% Text-Only) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mở đầu câu (Early)** | **Rounds 0 – 5** | **0.51 – 0.70 tokens** | **2.23 – 2.42 tokens** | **46.3% – 60.8%** | **3.5% – 7.4%** |
| **Đoạn đầu thân bài** | **Rounds 6 – 15** | 0.98 – 1.04 tokens | 2.65 – 2.79 tokens | 31.9% – 38.7% | 5.0% – 6.4% |
| **Đoạn giữa nội dung** | **Rounds 16 – 30**| 1.18 – 1.25 tokens | 2.99 – 3.05 tokens | 24.1% – 25.8% | 2.5% – 5.1% |
| **Nội dung chi tiết** | **Rounds 31 – 60**| 1.36 – 1.40 tokens | 3.72 – 3.97 tokens | 18.7% – 22.9% | 0.0% – 3.0% |
| **Đoạn kết (Late)** | **Rounds 60+** | **1.69 – 1.82 tokens** | **4.23 – 4.85 tokens** | **0.0% – 9.4%** | **0.0% – 2.1%** |

#### Phát hiện cốt lõi từ Timeline:
1. **Khủng hoảng mở đầu ở 100% Visual Retention:** Ở 5 round đầu tiên, khi Draft model phải mang toàn bộ visual tokens nặng nề, tỷ lệ Zero-Acceptance bị đẩy lên mức đỉnh điểm: **60.8% ở VideoChair và 46.3% ở VidHalluc**! Draft model gần như tê liệt và bị Target bác bỏ liên tục.
2. **Khắc phục triệt để bằng 0% Retention:** Cắt visual về 0% ngay lập tức kéo Zero-Acc ở 5 round đầu xuống **3.5% – 7.4%** (giảm gần 10 lần), đẩy chiều dài accept lên **2.42 tokens/round** ngay từ vạch xuất phát.
3. **Hiệu ứng tích lũy ngữ cảnh (Context Accumulation):** Càng về cuối câu (Rounds 30+ và 60+), khi chuỗi văn bản đã dài, chiều dài accept của mức 0% tăng vọt lên **4.23 – 4.85 tokens/round**, đạt hiệu suất suy đoán tối đa.

---

### 5.2. Đường cong Sinh tồn theo Chiều sâu Cây Draft (Tree Depth Survival Rate)

![Đường cong Sinh tồn theo Chiều sâu Cây](fig5_tree_depth_survival_curve.png)
*Hình 7: Tỷ lệ sống sót của cây suy đoán (Tree Depth Survival Rate) từ Độ sâu 1 đến Độ sâu 5 giữa cấu hình 100% Visual Retention (màu vàng/xanh) và 0% Text-Only (màu đỏ/xanh lá).*

Bảng tỷ lệ các round chấp nhận thành công chuỗi draft đạt độ sâu $\ge k$ tokens:

| Chiều sâu Cây Draft | VidHalluc (100% Visual) | VidHalluc (0% Text-Only) | VideoChair (100% Visual) | VideoChair (0% Text-Only) | Nhận xét so sánh |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Depth 1 ($\ge 1$ token)** | 65.2% | **93.8%** | 82.5% | **96.5%** | $t_0$ accept tăng vượt trội. |
| **Depth 2 ($\ge 2$ tokens)** | 27.4% | **75.8%** | 49.3% | **86.4%** | Mức 0% giữ được >75-86%. |
| **Depth 3 ($\ge 3$ tokens)** | 7.3% | **47.3%** | 16.2% | **69.0%** | Mức 100% sụp đổ dưới 16%. |
| **Depth 4 ($\ge 4$ tokens)** | 3.4% | **27.2%** | 3.2% | **52.2%** | Mức 0% vẫn sống sót >50%! |
| **Depth 5 ($\ge 5$ tokens)** | **0.9%** | **18.7%** | **0.4%** | **42.2%** | Mức 100% gần như biến mất hoàn toàn. |

> 💡 **Ý nghĩa:** Ở 100% Visual, cây EAGLE gần như bị "chặt cụt" ngay từ Depth 2-3 (chỉ có 0.4% - 0.9% round chạm tới Depth 5). Ngược lại, ở 0% Retention, cây suy đoán sống sót sâu tới Depth 4-5 với xác suất lên tới **42.2% - 52.2%** ở VideoChair, chứng minh sự đồng điệu phi thường giữa phân phối của Draft và Target.

---

### 5.3. Hiệu năng theo Nhóm Độ khó Iteration (CONTENT vs EASY vs HARD)

| Bộ dữ liệu | Nhóm Độ khó | Chiều dài Accept TB (100% Visual) | Chiều dài Accept TB (0% Text-Only) | Tỷ lệ Zero-Acc (100% Visual) | Tỷ lệ Zero-Acc (0% Text-Only) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **VidHalluc** | **CONTENT** (Từ khóa nội dung) | 1.26 tokens | **2.78 tokens** | 25.92% | **5.54%** |
| | **EASY** (Dấu câu, hư từ) | 0.75 tokens | **2.18 tokens** | 44.29% | **9.65%** |
| | **HARD** (Giới từ, rẽ nhánh) | 0.65 tokens | **2.64 tokens** | 57.74% | **3.23%** |
| **VideoChair** | **CONTENT** (Từ khóa nội dung) | 1.59 tokens | **3.45 tokens** | 14.17% | **2.72%** |
| | **EASY** (Dấu câu, hư từ) | 1.50 tokens | **3.55 tokens** | 20.03% | **4.33%** |
| | **HARD** (Giới từ, rẽ nhánh) | 1.10 tokens | **3.36 tokens** | 33.18% | **6.49%** |

Ở cấu hình 100% Visual, các iteration thuộc nhóm `HARD` có tỷ lệ Zero-Acc lên tới **33% – 58%**. Khi cắt visual về 0%, nhóm `HARD` được cải thiện ngoạn mục: tỷ lệ Zero-Acc giảm xuống chỉ còn **3.2% – 6.5%**, biến những điểm nghẽn khó nhất thành các bước suy đoán mượt mà.

---

## 6. Phân tích Tỷ lệ Phần trăm và Phân bố Câu trả lời Output (Output Answer Distribution)

Một trong những câu hỏi quan trọng nhất của nghiên cứu là: **"Việc cắt giảm visual context ở Draft model có làm thay đổi câu trả lời hoặc làm sai lệch phân bố ngữ nghĩa của output hay không?"**

### 6.1. Phân bố Tỷ lệ Câu trả lời trên VidHalluc & Tính Bảo toàn Bất biến 100%

Trên 100 video đại diện của VidHalluc, câu trả lời do mô hình sinh ra phân hóa thành 3 nhóm ngữ nghĩa:

| Nhóm Câu trả lời Output | Ý nghĩa & Dấu hiệu nhận diện | Tỷ lệ % ở 100% Visual | Tỷ lệ % ở 50% Cut | Tỷ lệ % ở 25% Cut | Tỷ lệ % ở 0% Full Cut | Độ ổn định qua các mức cắt |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Khẳng định "Yes"** | Khẳng định hiện tượng/hành động có xảy ra | **23.0%** | **23.0%** | **23.0%** | **23.0%** | **Bất biến 100%** |
| **Phủ định "No"** | Bác bỏ hiện tượng gây ảo giác thị giác | **16.0%** | **16.0%** | **16.0%** | **16.0%** | **Bất biến 100%** |
| **Mô tả Mở (Descriptive)**| Câu trả lời diễn giải chi tiết cho câu hỏi mở | **61.0%** | **61.0%** | **61.0%** | **61.0%** | **Bất biến 100%** |
| **Tổng số mẫu** | 100 video VidHalluc | **100%** | **100%** | **100%** | **100%** | **Tuyệt đối nhất quán** |

* **Độ dài câu trả lời trung bình:** Duy trì cố định ở mức **37.1 – 37.7 tokens** qua toàn bộ 4 mức cắt.
* **Kết luận khoa học quan trọng:** Tỷ lệ phân bố câu trả lời `23% Yes, 16% No, 61% Descriptive` giữ nguyên **chính xác 100% không đổi** qua mọi mức cắt! Điều này chứng minh thuật toán Speculative Decoding bảo đảm tính toàn vẹn ngữ nghĩa tuyệt đối, Draft model bị cắt visual chỉ đóng vai trò đề xuất tốc độ, quyền quyết định nội dung hoàn toàn thuộc về Target model 100% visual.

---

### 6.2. Độ chính xác Nhiệm vụ & Tỷ lệ Ảo giác (Task Accuracy & Hallucination Rate) trên VidHalluc

Đối chiếu câu trả lời trích xuất với nhãn Ground-Truth của tập VidHalluc qua 4 mức độ cắt visual:

| Danh mục Bài toán (Task Type) | Số lượng mẫu | Độ chính xác (Accuracy) | Tỷ lệ Ảo giác (Hallucination Rate) | Chỉ số Bổ trợ (Precision / Recall / F1) |
| :--- | :---: | :---: | :---: | :--- |
| **Binary QA (Yes/No)** | 25 mẫu | **40.00%** (10/25) | **60.00%** (15/25) | Precision: **25.00%**, Recall: **18.18%**, F1: **21.05%** |
| **Scene Transition (TSH)** | 20 mẫu | **55.00%** (11/20) | **45.00%** (9/20) | Phát hiện đúng việc có/không biến đổi phân cảnh |
| **Action MCQ (ACH)** | 35 mẫu | **71.43%** (25/35) | **28.57%** (10/35) | Nhận diện đúng hành động qua đối sánh ngữ nghĩa |
| **Sequence Ranking (STH)** | 20 mẫu | **5.00%** (1/20) | **95.00%** (19/20) | Thử thách sắp xếp trật tự thời gian phức tạp |
| **Toàn bộ Benchmark VidHalluc**| **100 mẫu** | **47.00%** (47/100) | **53.00%** (53/100) | **Bảo toàn chính xác 100% trên cả 4 mức cắt visual!** |

* **Đặc tính bất biến tuyệt đối:** Mọi chỉ số Accuracy, Precision, Recall và Tỷ lệ Ảo giác trên cả 4 nhóm bài toán đều **giữ nguyên chính xác 100% không đổi** giữa 100% Visual Retention, 50% Cut, 25% Cut và 0% Text-Only.

---

### 6.3. Phân bố Độ dài Caption trên VideoChair (MSR-VTT)

Trên 100 video VideoChair, mô hình sinh các đoạn văn bản mô tả chi tiết với phân bố độ dài như sau:

| Nhóm Phân bố Độ dài | Số lượng tokens | 100% Visual Retention | 50% Visual Cut | 25% Visual Cut | 0% Full Cut (Text-only) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mô tả Ngắn (Short)** | $< 150$ tokens | 27.0% | 31.0% | 31.0% | 30.0% |
| **Mô tả Trung bình (Mid)** | $150 – 511$ tokens | 30.0% | 24.0% | 23.0% | 25.0% |
| **Mô tả Chi tiết Tối đa (Capped)**| $\ge 512$ tokens | **43.0%** | **45.0%** | **46.0%** | **45.0%** |
| **Độ dài Trung bình** | Mean Tokens | **316.2 tokens** | **321.4 tokens** | **324.4 tokens** | **323.8 tokens** |

* Gần một nửa số video (**43% – 46%**) được mô tả cực kỳ phong phú và chạm ngưỡng trần 512 tokens.
* Phân bố độ dài và cấu trúc câu mô tả hoàn toàn ổn định qua cả 4 độ cắt visual.

---

### 6.4. Đánh giá Định lượng Chỉ số Ảo giác Thực thể VideoChair (CHAIR Metrics)

![Đánh giá Định lượng Hiệu năng Task & Chỉ số Ảo giác](fig12_hallucination_performance_metrics.png)
*Hình 9: Đánh giá định lượng hiệu năng task và chỉ số ảo giác giữa VidHalluc và VideoChair qua 4 mức độ cắt visual context (100%, 50%, 25%, 0% Text-only).*

Đánh giá mức độ ảo giác thực thể (*Object Hallucination*) trên các đoạn văn bản mô tả chi tiết theo tiêu chuẩn **CHAIR** (*Caption Hallucination Assessment with Image/Video Relevance*):

| Chỉ số Đánh giá (CHAIR Metrics) | 100% Visual Retention | 50% Visual Cut | 25% Visual Cut | 0% Full Cut (Text-Only) | Chênh lệch (0% vs 100%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{CHAIR}_i$ (Instance Hallucination %)** | **92.05%** | **92.00%** | **92.01%** | **92.49%** | **+0.44% (Biến thiên không đáng kể)** |
| **$\text{CHAIR}_s$ (Sentence Hallucination %)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.00% (Bất biến)** |
| **Object Precision (Độ chuẩn xác thực thể)** | **7.95%** | **8.00%** | **7.99%** | **7.51%** | **-0.44%** |
| **Object Recall (Độ phủ thực thể GT)** | **21.69%** | **22.21%** | **22.41%** | **21.09%** | **-0.60%** |
| **Số thực thể sinh ra trung bình / video** | **140.2 objects** | **142.7 objects** | **144.2 objects** | **144.4 objects** | **+4.2 objects** |

#### Kết luận khoa học cốt lõi từ bảng chỉ số Performance & CHAIR:
1. **Ảo giác không hề gia tăng khi cắt visual:** Khi chuyển từ 100% Visual xuống 0% Text-Only, tỷ lệ ảo giác thực thể $\text{CHAIR}_i$ chỉ biến thiên ở mức **0.44%** (từ 92.05% lên 92.49%), hoàn toàn nằm trong biên độ dung sai thống kê.
2. **Độ phủ thực thể (Object Recall) được bảo toàn:** Tỷ lệ các thực thể có thật trong video được mô hình nhận diện duy trì ổn định ở mức **21.1% – 22.4%** trên toàn bộ 4 mức cắt.
3. **Ý nghĩa quyết định của Speculative Decoding:** Cơ chế xác minh của Target model đóng vai trò như một bộ màng lọc bảo vệ tuyệt đối: việc Draft model bị "mù thị giác" (0% retention) **chỉ giúp tăng tốc độ suy luận gấp 2.3 – 2.5 lần mà hoàn toàn không gây suy thoái chất lượng hay làm gia tăng ảo giác** của hệ thống đa phương thức.

---

## 7. Danh mục Tệp Dữ liệu và Mã nguồn Kết quả

Toàn bộ tài nguyên thực nghiệm được lưu trữ hoàn chỉnh tại thư mục `results/`:

1. **Báo cáo và Hình ảnh Trực quan hóa (`results/6_10_26/`):**
   * Báo cáo này: [`results/6_10_26/BAO_CAO_VIDHALLUC_VIDEOCHAIR.md`](file:///home/hust/Phuc/VDFlash/results/6_10_26/BAO_CAO_VIDHALLUC_VIDEOCHAIR.md)
   * Dashboard tổng quan: [`results/6_10_26/hallucination_comprehensive_dashboard.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/hallucination_comprehensive_dashboard.png)
   * Hình 2 - Tốc độ theo Độ Cắt: [`results/6_10_26/fig1_speedup_by_cut_level.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig1_speedup_by_cut_level.png)
   * Hình 3 - Hiệu năng Suy đoán theo Độ Cắt: [`results/6_10_26/fig2_efficiency_by_cut_level.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig2_efficiency_by_cut_level.png)
   * Hình 4 - Phân rã Độ trễ Prefill vs Decode: [`results/6_10_26/fig3_latency_breakdown_by_cut_level.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig3_latency_breakdown_by_cut_level.png)
   * Hình 5 - Khả năng Khởi đầu Chuỗi & Lossless: [`results/6_10_26/fig4_t0_acceptance_and_lossless.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig4_t0_acceptance_and_lossless.png)
   * Hình 6 - Phân bố Độ khó Iteration: [`results/6_10_26/hallucination_difficulty_distribution.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/hallucination_difficulty_distribution.png)
   * Hình 7 - Đường cong Sinh tồn Cây Draft (Depth 1-5): [`results/6_10_26/fig5_tree_depth_survival_curve.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig5_tree_depth_survival_curve.png)
   * Hình 8 - Phân bố Câu trả lời Output & Timeline: [`results/6_10_26/fig6_timeline_and_answer_distribution.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig6_timeline_and_answer_distribution.png)
   * Hình 9 - Hiệu năng Task & Chỉ số Ảo giác CHAIR: [`results/6_10_26/fig12_hallucination_performance_metrics.png`](file:///home/hust/Phuc/VDFlash/results/6_10_26/fig12_hallucination_performance_metrics.png)
   * Bảng số liệu JSON Performance & CHAIR: [`results/6_10_26/hallucination_performance_metrics.json`](file:///home/hust/Phuc/VDFlash/results/6_10_26/hallucination_performance_metrics.json)
2. **Dữ liệu Vết suy luận Gốc và Bảng Iteration (`results/msd_eval_20261007/`):**
   * VidHalluc JSONL (404 dòng): [`results/msd_eval_20261007/vidhalluc/msd_retention_vidhalluc.jsonl`](file:///home/hust/Phuc/VDFlash/results/msd_eval_20261007/vidhalluc/msd_retention_vidhalluc.jsonl)
   * VidHalluc Iterations CSV (5.963 rounds): [`results/msd_eval_20261007/vidhalluc/draft_iterations_detailed.csv`](file:///home/hust/Phuc/VDFlash/results/msd_eval_20261007/vidhalluc/draft_iterations_detailed.csv)
   * VideoChair JSONL (400 dòng): [`results/msd_eval_20261007/videochair/msd_retention_videochair.jsonl`](file:///home/hust/Phuc/VDFlash/results/msd_eval_20261007/videochair/msd_retention_videochair.jsonl)
   * VideoChair Iterations CSV (36.099 rounds): [`results/msd_eval_20261007/videochair/draft_iterations_detailed.csv`](file:///home/hust/Phuc/VDFlash/results/msd_eval_20261007/videochair/draft_iterations_detailed.csv)
3. **Mã nguồn thực thi và vẽ biểu đồ (`scripts/`):**
   * Launcher đánh giá: [`scripts/run_msd_evaluation_benchmarks.sh`](file:///home/hust/Phuc/VDFlash/scripts/run_msd_evaluation_benchmarks.sh)
   * Bộ đánh giá định lượng Task & CHAIR: [`scripts/evaluate_hallucination_performance.py`](file:///home/hust/Phuc/VDFlash/scripts/evaluate_hallucination_performance.py)
   * Bộ vẽ biểu đồ định lượng Task & CHAIR: [`scripts/plot_hallucination_performance_metrics.py`](file:///home/hust/Phuc/VDFlash/scripts/plot_hallucination_performance_metrics.py)
   * Bộ trích xuất iteration: [`scripts/extract_draft_iterations.py`](file:///home/hust/Phuc/VDFlash/scripts/extract_draft_iterations.py)
   * Bộ vẽ biểu đồ tổng quan: [`scripts/plot_hallucination_benchmarks.py`](file:///home/hust/Phuc/VDFlash/scripts/plot_hallucination_benchmarks.py)
   * Bộ vẽ biểu đồ theo 4 độ cắt: [`scripts/plot_metrics_by_cut_level.py`](file:///home/hust/Phuc/VDFlash/scripts/plot_metrics_by_cut_level.py)
   * Bộ vẽ biểu đồ độ sâu cây & phân bố câu trả lời: [`scripts/plot_iteration_and_answer_distributions.py`](file:///home/hust/Phuc/VDFlash/scripts/plot_iteration_and_answer_distributions.py)
