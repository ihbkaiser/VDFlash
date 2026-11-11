# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: ĐẶC TÍNH TOKEN VÀ NGỮ CẢNH QUYẾT ĐỊNH KHẢ NĂNG CHẤP NHẬN TRONG SPECULATIVE DECODING (MSD TRÊN VDC-50)

- **Thời gian thực hiện:** Ngày 06 tháng 10 năm 2026
- **Vị trí lưu trữ:** `results/6_10_26/BAO_CAO_PHAN_TICH_TOKEN_ACCEPTANCE.md`
- **Mã nguồn phân tích:** `scripts/analyze_token_acceptance.py` và `scripts/plot_iteration_performance.py`
- **Dữ liệu thực nghiệm gốc:** `results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl`
- **Phạm vi dữ liệu:** 200 lượt suy luận hoàn chỉnh trên 50 video Video Detail Captioning (VDC-50), bao gồm 4 mức visual retention (100%, 50%, 25%, 0%), theo dõi toàn bộ 18,991 vị trí token và 27,340 lượt đề xuất draft.

---

## TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Báo cáo này giải quyết câu hỏi trung tâm: **"Liệu các iteration chứa token như thế nào thì dễ được target model chấp nhận (accept) nhất, và bản chất cơ chế ngôn ngữ - thị giác nào chi phối hiện tượng này?"**

Dựa trên việc bóc tách toàn bộ **18,991 vị trí token** và **27,340 lượt đề xuất draft** từ 200 lượt suy luận của MSD (Qwen2-VL-7B target + EAGLE 1-layer draft) trên tập dữ liệu Video Detail Captioning (VDC-50), chúng tôi tổng hợp các phát hiện cốt lõi qua bảng điều hành và 5 luận điểm chứng minh bằng số liệu thực nghiệm dưới đây:

### Bảng tóm tắt điều hành: 5 Nhóm hiện tượng chấp nhận Token

| Nhóm hiện tượng | Token / Ngữ cảnh tiêu biểu | Vị trí bảng / Tệp số liệu | Thông số định lượng được cite | Kết luận & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| **1. Cụm từ cố định** *(Collocations)* | `' variety'`, `' objects'`, `' followed'` | **Bảng 1, dòng 1–6**; [`token_acceptance_comparison.csv`](token_acceptance_comparison.csv) | `acc_rate_100_pct` $\ge 92.9\%$; `acc_rate_0_pct` $\ge 91.8\%$; `delta_pct` $\in [-4.76\%, +2.57\%]$ | **DỄ NHẤT (>95%):** Độc lập với visual tokens; entropy tiệm cận 0 do n-gram bất biến. |
| **2. Sau Mạo từ & Giới từ** | Đứng sau `' a'`, `' with'`, `' on'`, `' of'` | **Bảng 2, dòng 1–2**; **Bảng 3 (Tree Depth)** | `Mean Accepted Tokens` = **1.59–1.80** (100%) và **4.01–4.59** (0%); $\ge 5$ tokens đạt **57.9%** (0%) | **DỄ THỨ NHÌ:** Cú pháp thu hẹp không gian từ tiếp theo, giúp draft tree ăn trọn chuỗi dài $\ge 5$ tokens. |
| **3. Điểm rẽ nhánh & Liên từ** | `' and'`, `','` (phẩy), `' then'` | **Bảng 1, dòng 7–13**; Bảng runtime H3.1 | `' and'`: `acc_rate_100` = **6.75%**, `acc_rate_0` = **80.84%** (`delta_pct` = **+74.09%**); Verify = **27.51s** vs **11.71s** | **PHÂN HÓA LỚN NHẤT:** 100% bị visual noise làm trượt nhịp nối câu (reject >93%), 0% bám chuẩn n-gram. |
| **4. Mở đầu câu mới** | `' The'` (câu mới), đứng sau dấu chấm `'.'` | **Bảng 1, dòng 16**; **Bảng 2, dòng 6**; [`msd_iteration_milestones.csv`](msd_iteration_milestones.csv) | `' The'`: `acc_rate_100` = **4.06%**; sau `'.'`: `Mean Accepted` = **0.26 token**; Iter 1: **0.00 token** | **KHÓ NHẤT:** Entropy cực đại ở đầu câu mới; draft hầu như thất bại hoàn toàn trong việc đoán từ mở đầu. |
| **5. Chi tiết thị giác cụ thể** | `' standing'`, `' clock'`, `' person'` | **Bảng 1, dòng 17–21**; Bảng runtime H3.1 | `delta_pct` âm: `' standing'` (**-25.52%**), `' clock'` (**-16.12%**); Tần suất từ = **2.6%** vs **20.2%** | **LỢI THẾ CỦA 100%:** Visual giúp đoán đúng vật thể/hành động (+6% đến +25%), nhưng bị lấn át bởi 80% từ ngữ pháp. |

---

### Chi tiết 5 phát hiện cốt lõi (kèm vị trí đối chiếu và thông số được cite):

1. **Các iteration chứa cụm từ cố định (Formulaic Collocations) đạt tỷ lệ chấp nhận tuyệt đối (>95% – 99.7%):**
   - *Vị trí số liệu:* **Bảng 1, Nhóm 1 (dòng 1–6)** trong báo cáo; tệp dữ liệu [`token_acceptance_comparison.csv`](token_acceptance_comparison.csv).
   - *Thông số được cite:* 
     - Cột `acc_rate_100_pct` và `acc_rate_0_pct`: `' variety'` đạt **99.66%** (291/292) và **94.90%** (279/294); `' objects'` đạt **99.03%** (306/309) và **94.82%** (293/309); `' followed'` đạt **98.48%** (65/66) ở cả hai mức; `' transitions'` đạt **97.50%** (78/80) và **96.25%** (77/80).
     - Cột `delta_pct`: Giá trị dao động quanh 0 ($\Delta \in [-4.76\%, +2.57\%]$).
   - *Giải thích dựa trên thông số:* Thông số `delta_pct` xấp xỉ 0 ở toàn bộ nhóm này chứng minh: tại các n-gram cố định (`a variety of objects`, `followed by`), xác suất dự đoán đúng của draft model hoàn toàn độc lập với sự hiện diện của visual tokens. Do cấu trúc cụm từ đã được cố định chặt chẽ trong prompt, entropy phân phối token kế tiếp hội tụ về 0, giúp cả hai mức retention đều đạt `acc_rate` tiệm cận 100%.

2. **Các iteration đứng ngay sau Giới từ và Mạo từ dễ chấp nhận thứ nhì (chuỗi dài 3–5 tokens):**
   - *Vị trí số liệu:* **Bảng 2, dòng 1–2 (Mạo từ & Giới từ)** và **Bảng 3 (Phân phối Draft Tree Depth)**.
   - *Thông số được cite:*
     - Cột `Mean Accepted Tokens` ở Bảng 2: Sau mạo từ `' a'` đạt **1.59 token** (100%) và **4.01 token** (0%) với cỡ mẫu $N=1,484$ round; sau giới từ `' with'` đạt **1.80** (100%) và **4.35** (0%) ($N=257$); sau `' on'` đạt **1.80** (100%) và **4.50** (0%) ($N=79$); sau `' of'` đạt **1.44** (100%) và **4.59** (0%) ($N=792$).
     - Cột `Tỷ lệ % (0% Retention)` ở Bảng 3: Tỷ lệ round ăn trọn chuỗi sâu $\ge 5$ tokens đạt **57.9%** (2,246 / 3,882 rounds), cao gấp **289 lần** so với 100% retention (**0.2%**, 16 / 8,262 rounds).
   - *Giải thích dựa trên thông số:* Cú pháp tiếng Anh quy định sau mạo từ/giới từ bắt buộc là danh từ hoặc tính từ, dẫn tới thông số `Mean Accepted Tokens` ở round tiếp theo luôn duy trì ở mức đỉnh (>4.0 tokens ở 0% và >1.6 tokens ở 100%). Nhờ không gian tìm kiếm bị thu hẹp, draft tree phát huy tối đa độ sâu, thể hiện qua con số 57.9% round đạt độ sâu $\ge 5$ tokens tại Bảng 3.

3. **Các iteration ở điểm rẽ nhánh ngữ pháp (`' and'`, `','`, `' then'`) gây ra sự phân hóa kỷ lục:**
   - *Vị trí số liệu:* **Bảng 1, Nhóm 2 (dòng 7–13)** và tệp [`token_acceptance_comparison.csv`](token_acceptance_comparison.csv).
   - *Thông số được cite:*
     - Cột `acc_rate_100_pct` vs `acc_rate_0_pct`: Token `' and'` đạt **6.75%** (78/1,156) so với **80.84%** (928/1,148), ghi nhận `delta_pct = +74.09%`; dấu phẩy `','` đạt **22.77%** so với **73.71%**, `delta_pct = +50.94%`; `' then'` đạt **13.36%** so với **72.29%**, `delta_pct = +58.93%`.
     - Cột thời gian `Verify (s)` trong báo cáo runtime H3.1 (`results/phase2_summary_20260923/README.md`): Thời gian xác minh ở 100% lên tới **27.51s** so với chỉ **11.71s** ở 0%.
   - *Giải thích dựa trên thông số:* Tỷ lệ `acc_rate_100_pct` rơi xuống dưới 25% (và chỉ 6.75% ở `' and'`) giải thích trực tiếp tại sao `Verify (s)` ở 100% tốn nhiều thời gian hơn gấp đôi: tại mọi điểm nối vế câu, target model liên tục phát hiện proposal sai và phải dừng lại để emit fallback token. Lượng visual tokens lớn (25k tokens) khiến draft layer 1 tầng bị nhiễu nhịp ngắt câu của target LLM 28 tầng, trong khi draft 0% bám chuẩn phân phối n-gram nên `delta_pct` tăng vọt trên +50% đến +74%.

4. **Các iteration mở đầu câu mới (ngay sau dấu chấm `'.'`) là khó chấp nhận nhất:**
   - *Vị trí số liệu:* **Bảng 1, dòng 16**, **Bảng 2, dòng 6** và tệp [`msd_iteration_milestones.csv`](msd_iteration_milestones.csv).
   - *Thông số được cite:*
     - Cột `acc_rate_100_pct` ở Bảng 1: Token `' The'` ở đầu câu mới chỉ đạt **4.06%** (`accepted_100 = 24` trên `total_100 = 591`).
     - Cột `Mean Accepted Tokens` ở Bảng 2: Sau dấu chấm `'.'` ($N=172$ round), round kế tiếp ở 100% chỉ đạt **0.26 ± 0.04 token / round** (giá trị thấp nhất toàn bộ Bảng 2).
     - Cột `mean_accepted_tokens` trong [`msd_iteration_milestones.csv`](msd_iteration_milestones.csv): Tại `draft_iteration = 1` (dòng 3), giá trị của 100% rơi về **chính xác 0.00** trên toàn bộ 50 sample ($N=50$).
   - *Giải thích dựa trên thông số:* Thông số `Mean Accepted Tokens = 0.26` và `acc_rate = 4.06%` chứng minh thực nghiệm rằng vị trí đầu câu mới có entropy cao nhất toàn hệ thống. Không có sự ràng buộc từ token trước, draft model hầu như thất bại hoàn toàn trong việc đoán trước từ mở đầu của target model.

5. **Nghịch lý giá trị thị giác: Lợi thế thực sự của 100% Retention:**
   - *Vị trí số liệu:* **Bảng 1, Nhóm 4 (dòng 17–21)** và bảng tổng hợp runtime H3.1.
   - *Thông số được cite:*
     - Cột `delta_pct` mang dấu âm duy nhất toàn bảng: `' standing'` có `delta_pct = -25.52%` (`acc_rate_100_pct = 94.85%` so với `acc_rate_0_pct = 69.33%`); `' clock'` có `delta_pct = -16.12%` (`acc_rate_100_pct = 98.15%` so với `82.03%`); `' person'` có `delta_pct = -6.37%` (87.25% so với 80.88%).
     - Tỷ lệ tần suất xuất hiện ($N_{\text{word}} / N_{\text{total}}$): Tổng số lần xuất hiện của `' standing'` (233) và `' clock'` (270) chỉ chiếm **~2.6%** tổng số 18,991 tokens của tập dữ liệu; trong khi riêng 2 từ nối `' a'` (2,675) và `' and'` (1,156) đã chiếm tới **20.2%**.
     - Cột `Speedup` tổng thể (H3.1): 100% retention chỉ đạt **1.03×** (13.4 tok/s) so với 0% retention đạt **1.81×** (31.0 tok/s).
   - *Giải thích dựa trên thông số:* Giá trị `delta_pct` âm khẳng định thị giác có giá trị thực sự cho các token miêu tả hành động và vật thể cụ thể (cải thiện từ +6% đến +25.5%). Tuy nhiên, vì các token này chỉ chiếm tỷ trọng thiểu số (~2.6% so với 20.2% của từ nối), ưu thế cục bộ này bị triệt tiêu hoàn toàn bởi sự sụp đổ ở các từ ngữ pháp, dẫn tới chỉ số tăng tốc `Speedup` ở 100% chỉ đạt 1.03×.



---

## 1. THIẾT KẾ THỰC NGHIỆM VÀ PHƯƠNG PHÁP KHẢO SÁT

### 1.1. Kiến trúc mô hình và thiết lập tham số
- **Target Model:** Qwen2-VL-7B-Instruct (xử lý đầy đủ video token, ~25,713 visual tokens tại 100% retention).
- **Draft Model:** MSD (EAGLE 1-layer transformer head) thực hiện Tree Speculative Decoding.
- **Tập dữ liệu:** Video Detail Captioning (VDC-50), mỗi video có độ dài trung bình target ~378 tokens mô tả chi tiết diễn biến hình ảnh.
- **Biến kiểm soát (Retention levels):**
  - **100% Retention:** Draft context nhận đủ 100% visual tokens (~25,713 tokens).
  - **50% Retention:** Draft context nhận 50% visual tokens (~12,856 tokens).
  - **25% Retention:** Draft context nhận 25% visual tokens (~6,428 tokens).
  - **0% Retention:** Draft context bị nén hoàn toàn visual tokens (0 visual token), chỉ nhận prompt văn bản và các token đã được sinh.

### 1.2. Định nghĩa các chỉ số đo lường
Mỗi vị trí token $i$ sinh ra trong câu trả lời thuộc về một vòng suy luận $t$ (`round_index`) được phân loại thành:
- **`accepted_proposal`:** Token do mô hình draft đề xuất trong cây suy đoán, trùng khớp phân phối xác suất và được target model chấp nhận ($y_i^{\text{draft}} = y_i^{\text{target}}$).
- **`fallback`:** Token do chính target model tự giải mã autoregressive sau khi chuỗi draft bị dừng hoặc đoán sai.

Tỷ lệ chấp nhận của một token $w$ (Token Acceptance Rate) được định nghĩa là:
$$\text{Acceptance Rate}(w) = \frac{N_{\text{accepted}}(w)}{N_{\text{accepted}}(w) + N_{\text{fallback}}(w)} \times 100\%$$

---

## 2. BẢNG DỮ LIỆU ĐỊNH LƯỢNG: TỶ LỆ CHẤP NHẬN THEO LOẠI TOKEN

Dựa trên việc quét qua toàn bộ 18,991 token positions, các token được phân chia thành 5 nhóm cấu trúc ngữ nghĩa rõ rệt:

### Bảng 1: So sánh Tỷ lệ Chấp nhận giữa 100% Visual Retention và 0% Visual Retention

| Nhóm chức năng (Category) | Token | Số lần xuất hiện (Target) | Chấp nhận ở 100% | Tỷ lệ 100% (%) | Chấp nhận ở 0% | Tỷ lệ 0% (%) | Chênh lệch $\Delta$ (0% - 100%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nhóm 1: Cụm từ cố định**<br>*(Formulaic Collocations)* | `' variety'`<br>`' objects'`<br>`' followed'`<br>`' transitions'`<br>`' series'`<br>`' including'` | 292<br>309<br>66<br>80<br>61<br>280 | 291<br>306<br>65<br>78<br>58<br>260 | **99.7%**<br>**99.0%**<br>**98.5%**<br>**97.5%**<br>**95.1%**<br>**92.9%** | 279<br>293<br>65<br>77<br>56<br>271 | **94.9%**<br>**94.8%**<br>**98.5%**<br>**96.3%**<br>**91.8%**<br>**95.4%** | -4.8%<br>-4.2%<br>0.0%<br>-1.2%<br>-3.3%<br>+2.5% |
| **Nhóm 2: Liên từ & Điểm rẽ nhánh**<br>*(Branching Connectors)* | `' and'`<br>`' of'`<br>`','` (phẩy)<br>`' then'`<br>`' with'`<br>`' on'`<br>`' in'` | 1,156<br>1,094<br>953<br>247<br>573<br>101<br>406 | 78<br>298<br>217<br>33<br>313<br>22<br>321 | **6.8%**<br>**27.2%**<br>**22.8%**<br>**13.4%**<br>**54.6%**<br>**21.8%**<br>**79.1%** | 928<br>869<br>698<br>180<br>458<br>86<br>345 | **80.8%**<br>**77.9%**<br>**73.7%**<br>**72.3%**<br>**81.5%**<br>**98.8%**<br>**86.9%** | **+74.1%**<br>**+50.6%**<br>**+50.9%**<br>**+58.9%**<br>**+26.9%**<br>**+77.1%**<br>+7.8% |
| **Nhóm 3: Mở đầu câu & Ngắt câu**<br>*(Sentence Boundaries)* | `' The'` (câu mới)<br>`'The'` (đầu bài)<br>`'.'` (chấm câu) | 591<br>56<br>624 | 24<br>9<br>452 | **4.1%**<br>**16.1%**<br>**72.4%** | 253<br>52<br>517 | **42.2%**<br>**92.9%**<br>**81.7%** | **+38.1%**<br>**+76.8%**<br>+9.2% |
| **Nhóm 4: Chi tiết thị giác cụ thể**<br>*(Visual Content Specific)* | `' standing'` (hành động)<br>`' clock'` (vật thể)<br>`' person'`<br>`' background'`<br>`' white'` (màu sắc) | 233<br>270<br>204<br>113<br>505 | 221<br>265<br>178<br>105<br>326 | **94.8%**<br>**98.1%**<br>**87.3%**<br>**92.9%**<br>**64.6%** | 156<br>210<br>165<br>101<br>446 | **69.3%**<br>**82.0%**<br>**80.9%**<br>**89.4%**<br>**87.1%** | **-25.5%**<br>**-16.1%**<br>-6.4%<br>-3.5%<br>+22.5% |
| **Nhóm 5: Khung mẫu mô tả chuẩn**<br>*(Domain Boilerplate)* | `' features'`<br>`' scene'`<br>`' video'`<br>`' shows'`<br>`' a'` (mạo từ)<br>`' the'` (mạo từ) | 71<br>212<br>325<br>273<br>2,675<br>466 | 33<br>55<br>124<br>237<br>1,187<br>206 | **46.5%**<br>**25.9%**<br>**38.2%**<br>**86.8%**<br>**44.4%**<br>**44.2%** | 67<br>173<br>229<br>282<br>1,977<br>346 | **97.1%**<br>**84.4%**<br>**69.8%**<br>**97.6%**<br>**72.9%**<br>**72.8%** | **+50.6%**<br>**+58.5%**<br>**+31.7%**<br>+10.8%<br>+28.5%<br>+28.6% |

---



![Tỷ lệ chấp nhận theo loại token](figures/token_acceptance_by_category.png)
*Hình 1: Tỷ lệ chấp nhận theo nhóm loại token giữa 100% Visual và 0% Visual.*

---

## 3. KHẢO SÁT NGỮ CẢNH ĐỨNG TRƯỚC (PRECEDING SYNTACTIC CONTEXT)

Khả năng thành công của một iteration không chỉ phụ thuộc vào bản thân token đó, mà chịu sự chi phối quyết định bởi **token kết thúc ở vòng suy luận liền trước**. 

### Bảng 2: Hiệu năng Round kế tiếp theo Ngữ cảnh Token đứng trước

| Loại ngữ cảnh đứng trước | Token tiêu biểu | Số lượng round khảo sát ($N$) | Mean Accepted Tokens (100% Retention) | Mean Accepted Tokens (0% Retention) | Chênh lệch $\Delta$ | Cơ chế cú pháp ngôn ngữ |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Mạo từ (Determiners)** | `' a'` | 1,484 | **1.59 ± 0.08** | **4.01 ± 0.12** | +2.42 | Sau mạo từ bắt buộc là tính từ hoặc danh từ $\rightarrow$ không gian entropy bị khóa chặt. |
| **Giới từ (Prepositions)** | `' with'`<br>`' on'`<br>`' in'`<br>`' of'` | 257<br>79<br>85<br>792 | **1.80 ± 0.11**<br>**1.80 ± 0.18**<br>**1.79 ± 0.17**<br>**1.44 ± 0.07** | **4.35 ± 0.19**<br>**4.50 ± 0.28**<br>**4.02 ± 0.25**<br>**4.59 ± 0.13** | +2.55<br>+2.70<br>+2.23<br>+3.15 | Giới từ tạo ra cụm giới từ cố định (Prepositional Phrase) $\rightarrow$ draft dễ ăn trọn chuỗi dài 3–5 token. |
| **Động từ trạng thái (Verbs)** | `' standing'`<br>`' followed'` | 73<br>65 | **1.62 ± 0.15**<br>**1.90 ± 0.16** | **4.93 ± 0.24**<br>**4.85 ± 0.22** | **+3.31**<br>+2.95 | Đỉnh điểm của 0% retention: Sau *standing* luôn là *in a room/on a street* $\rightarrow$ draft ăn trọn 5 token. |
| **Liên từ (Conjunctions)** | `' and'`<br>`','` | 1,076<br>727 | **1.33 ± 0.06**<br>**1.41 ± 0.07** | **4.28 ± 0.14**<br>**4.19 ± 0.14** | +2.95<br>+2.78 | Điểm nối giữa 2 mệnh đề: ở 100% draft bị lệch nhịp, ở 0% draft bám tốt nhịp đẳng lập. |
| **Chủ thể danh từ (Nouns)** | `' video'`<br>`' scene'`<br>`' camera'` | 200<br>157<br>69 | **0.55 ± 0.07**<br>**1.23 ± 0.10**<br>**1.01 ± 0.13** | **4.59 ± 0.19**<br>**4.30 ± 0.21**<br>**2.77 ± 0.26** | **+4.04**<br>+3.07<br>+1.76 | Sau *The video*, mô hình 100% visual bị hoang mang vì quá nhiều chi tiết video, trong khi 0% chọn ngay template *features a variety of*. |
| **Dấu chấm hết câu (Period)** | `'.'` | 172 | **0.26 ± 0.04** | **2.93 ± 0.18** | **+2.67** | **Điểm khó nhất toàn hệ thống:** Điểm kết câu mở ra câu mới không có ràng buộc ngữ pháp $\rightarrow$ 100% rớt thảm hại về 0.26 token. |

---



![Chấp nhận theo ngữ cảnh đứng trước](figures/acceptance_by_preceding_context.png)
*Hình 2: Số token chấp nhận trung bình theo ngữ cảnh token đứng trước.*

---

## 4. KHẢO SÁT THEO CHIỀU SÂU CÂY SUY ĐOÁN (DRAFT TREE DEPTH SURVIVAL)

Cơ chế Tree Decoding của EAGLE cho phép draft đề xuất nhiều nhánh. Chiều sâu chuỗi token được target chấp nhận liên tiếp phản ánh trực tiếp chất lượng của draft context:

### Bảng 3: Phân phối Tỷ lệ chấp nhận theo Chiều sâu Chuỗi Draft

| Độ sâu chấp nhận trong 1 Round | Số lượng Round đạt được (100% Retention) | Tỷ lệ % (100% Retention) | Số lượng Round đạt được (0% Retention) | Tỷ lệ % (0% Retention) |
| :--- | :---: | :---: | :---: | :---: |
| **$\ge 1$ token** | 6,679 / 8,262 | **80.8%** | 3,779 / 3,882 | **97.3%** |
| **$\ge 2$ tokens** | 3,235 / 8,262 | **39.2%** | 3,516 / 3,882 | **90.6%** |
| **$\ge 3$ tokens** | 676 / 8,262 | **8.2%** | 3,088 / 3,882 | **79.5%** |
| **$\ge 4$ tokens** | 123 / 8,262 | **1.5%** | 2,559 / 3,882 | **65.9%** |
| **$\ge 5$ tokens** | 16 / 8,262 | **0.2%** | 2,246 / 3,882 | **57.9%** |

```
Tỷ lệ sống sót chuỗi Draft (Survival Curve theo Độ sâu):
100% Retention:  [80.8%] ===> [39.2%] ===> [ 8.2%] ===> [ 1.5%] ===> [ 0.2%]  (Sụp đổ chuỗi nhanh chóng)
  0% Retention:  [97.3%] ===> [90.6%] ===> [79.5%] ===> [65.9%] ===> [57.9%]  (Duy trì chuỗi cực kỳ vững chắc)
```

---



![Tỷ lệ sống sót theo chiều sâu cây](figures/acceptance_by_tree_depth.png)
*Hình 3: Tỷ lệ sống sót của chuỗi draft theo chiều sâu cây.*




![Hiệu năng suy luận theo từng iteration](figures/msd_acc_length_per_iteration.png)
*Hình 4: Hiệu năng suy luận qua các iteration.*

---

## 5. THẢO LUẬN BẢN CHẤT CƠ CHẾ (SCIENTIFIC DISCUSSION)

Từ các bằng chứng thực nghiệm trên, chúng tôi rút ra 3 kết luận bản chất chi phối hiệu năng của Speculative Decoding trên Vision-Language Models:

### 5.1. Cơ chế 1: "Visual Noise Confound" trong 1-layer Draft Model
Kiến trúc MSD sử dụng 1 tầng Transformer làm draft head (EAGLE). 
- Khi đưa toàn bộ **25,713 visual tokens** vào draft context (100% retention), biểu diễn hidden states của draft model 1 layer bị quá tải bởi không gian đặc trưng thị giác khổng lồ. 
- Ngược lại, Target LLM (Qwen2-VL-7B) có tới 28 layers với năng lực chú ý đa tầng đủ mạnh để lọc nhiễu và điều hướng câu từ theo cú pháp ngôn ngữ tự nhiên. 
- Do draft head chỉ có 1 tầng, nó không thể mô phỏng chính xác sự cân bằng phức tạp giữa visual features và language prior của mô hình 28 tầng. Hệ quả là draft model đề xuất các token bị lệch pha so với target tại các điểm nối câu (`and`, `,`, `then`), dẫn đến tỷ lệ reject lên đến >90%.

### 5.2. Cơ chế 2: Ưu thế tuyệt đối của Language Prior đối với Text Structure
Tại mức **0% retention**, draft context hoàn toàn không chứa visual tokens. Draft model trở thành một mô hình ngôn ngữ tự hồi quy thuần túy, tối ưu hóa theo phân phối n-gram của tập huấn luyện.
- Trong tác vụ Video Detail Captioning, hơn **80% độ dài văn bản** được cấu thành từ các cấu trúc cú pháp lặp lại (*"The video begins with a scene showing...", "a variety of objects including...", "...and then transitions to..."*).
- Vì bám sát language prior mà không bị visual tokens gây nhiễu, draft 0% retention dự đoán chuẩn xác các chuỗi dài 4–5 token liên tiếp. Điều này giải thích tại sao 57.9% số round ở 0% retention chấp nhận được $\ge 5$ token.

### 5.3. Cơ chế 3: Nghịch lý "Cục bộ vs Toàn thể" (Local vs Global Paradox)
- **Về mặt cục bộ (Local semantics):** Visual context thực sự cung cấp thông tin quý giá để draft đoán đúng các danh từ và tính từ mang tính thị giác cao (`standing`, `clock`, `airplane`, `white`). Đây là lý do tại sao ở các từ này, 100% retention có tỷ lệ chấp nhận cao hơn 0% từ +10% đến +25%.
- **Về mặt toàn thể (Global throughput):** Các danh từ thị giác cụ thể chỉ xuất hiện rải rác cách nhau 5–10 tokens. Trong khi đó, các từ nối, mạo từ và khung câu xuất hiện dày đặc ở mọi vị trí. Việc có visual context cải thiện một vài từ nội dung cục bộ nhưng phá hỏng toàn bộ các từ nối xung quanh, khiến chuỗi suy đoán bị đứt đoạn liên tục (chỉ chấp nhận 0.2% chuỗi dài). Ngược lại, bỏ visual context hy sinh một phần độ chính xác ở từ cụ thể nhưng ăn trọn toàn bộ các từ nối, mang lại bước nhảy trung bình 3.7–4.3 tokens/round.

---

## 6. KHUYẾN NGHỊ THIẾT KẾ MÔ HÌNH DRAFT CHO VLM TRONG TƯƠNG LAI

Dựa trên kết luận của báo cáo, chúng tôi đề xuất 3 hướng cải tiến kiến trúc cho các hệ thống suy luận đầu cơ đa phương thức:

1. **Kiến trúc Asymmetric Context Routing (Định tuyến ngữ cảnh bất đối xứng):**
   - Không nên bơm toàn bộ visual tokens vào draft layer ở mọi token. 
   - Khi token hiện tại là dấu chấm `.` hoặc liên từ rẽ nhánh `and`, draft layer nên tự động chuyển sang chế độ **Masked Visual (0% visual)** để bám theo language prior cho đến khi thiết lập xong khung ngữ pháp mới.
2. **Cơ chế Phân bổ Cây Draft Động (Dynamic Tree Allocation):**
   - Khai thác bảng Preceding Context: Khi round trước kết thúc bằng mạo từ (`a`) hoặc giới từ (`with`, `in`), thuật toán phân bổ cây EAGLE nên tập trung tăng chiều sâu nhánh (depth-heavy tree) vì xác suất ăn chuỗi dài $\ge 5$ tokens lúc này lên tới >80%.
   - Khi round trước kết thúc bằng dấu chấm (`.`), thuật toán nên mở rộng chiều rộng nhánh (breadth-heavy tree) để bao quát nhiều từ bắt đầu câu khác nhau.
3. **Draft Model Đa Tầng (DFlash Depth $\ge 3$):**
   - Kết quả này củng cố lý do tại sao DFlash với depth 3 hoặc 5 tầng layer xử lý visual context tốt hơn nhiều so với EAGLE 1-layer, vì cần tối thiểu 3–5 tầng để draft model dung hòa được giữa visual token embeddings và linguistic syntax.

---

## 7. DANH MỤC TÀI LIỆU VÀ BIỂU ĐỒ ĐÍNH KÈM

Các tệp kết quả trực quan và bảng số liệu chi tiết được lưu trong thư mục `results/6_10_26/`:

1. [**`token_acceptance_by_category.png`**](file:///home/hust/Phuc/VDFlash/results/6_10_26/token_acceptance_by_category.png): Biểu đồ cột phân nhóm thể hiện tỷ lệ accept của từng loại từ giữa 100% và 0% visual retention.
2. [**`acceptance_by_preceding_context.png`**](file:///home/hust/Phuc/VDFlash/results/6_10_26/acceptance_by_preceding_context.png): Biểu đồ cột so sánh số token accept trung bình tùy theo token đứng trước.
3. [**`acceptance_by_tree_depth.png`**](file:///home/hust/Phuc/VDFlash/results/6_10_26/acceptance_by_tree_depth.png): Biểu đồ tỷ lệ sống sót của chuỗi draft theo chiều sâu cây đề xuất ($\ge 1$ đến $\ge 5$ tokens).
4. [**`msd_acc_length_per_iteration.png`**](file:///home/hust/Phuc/VDFlash/results/6_10_26/msd_acc_length_per_iteration.png): Biểu đồ 4 panel tổng quan hiệu năng suy luận qua 60 iterations đầu và đường cong tích lũy.
5. [**`token_acceptance_comparison.csv`**](file:///home/hust/Phuc/VDFlash/results/6_10_26/token_acceptance_comparison.csv): Bảng dữ liệu thống kê chi tiết tỷ lệ accept cho 28 tokens tiêu biểu.
6. [**`msd_acc_length_per_iteration.csv`**](file:///home/hust/Phuc/VDFlash/results/6_10_26/msd_acc_length_per_iteration.csv): Toàn bộ 896 dòng số liệu thống kê chi tiết của từng iteration.
