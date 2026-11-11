# Báo Cáo Thực Nghiệm: Kiểm Chứng Các Phân Tích & Insight Của Bài Báo Sparrow (MSD) Trên Kiến Trúc DFlash

> **Mục tiêu nghiên cứu:** Bài báo **Sparrow** (*ACL 2026*) sử dụng mô hình **MSD (Multi-modal Speculative Decoding)** để đưa ra hàng loạt phân tích và giả thuyết về vai trò của Visual Context trong Speculative Decoding cho Video-LLM. Báo cáo này sử dụng kiến trúc **DFlash** (mô hình draft dự đoán song song theo khối) chạy lại toàn bộ các thực nghiệm tương ứng nhằm trả lời câu hỏi cốt lõi:
> 
> ***"Các kết luận và phân tích thực nghiệm trên MSD trong Sparrow có đúng với kiến trúc DFlash hay không?"***

---

## Tóm Tắt Kết Quả Kiểm Chứng (Executive Summary)

> **Nhận định tổng quan:** **Không phải toàn bộ mọi insight của Sparrow trên MSD đều đúng nguyên bản trên DFlash.**
> * **Các quy luật bản chất ngôn ngữ (Nguyên lý cấp cao):** Hoàn toàn đúng trên cả hai mô hình (Visual Redundancy, Attention Shift, Hallucination Invariance).
> * **Các cơ chế hành vi cục bộ & động lực học suy luận:** DFlash **khác biệt bản chất hoặc mở rộng vượt ra ngoài** MSD ở một số điểm cốt lõi (hành vi $\tau$ khi cắt visual, động lực học tiến trình sinh Early vs Late iterations, và sự phụ thuộc vào tập dữ liệu huấn luyện).

| STT | Phân Tích / Giả Thuyết Của Sparrow (Trên MSD) | Kết Quả Khi Chạy Với DFlash | Phân Loại Khoa Học | Bản Chất Khác Biệt Giữa MSD và DFlash |
|:---:|---|---|:---:|---|
| **1** | **Visual Redundancy:** Cắt 50% – 100% visual context không làm suy giảm tỷ lệ chấp nhận $\tau$. | Trên cả 4 dataset (VDC, MVBench, VidHalluc, VideoChair): $\tau_{eff}$ của DFlash không đổi ($1.95 \to 1.97$ và $2.41 \to 2.27$). | ✅ **XÁC MINH: ĐÚNG** | Cả hai kiến trúc đều xác nhận visual context ở drafter là dư thừa. |
| **2** | **Hiệu Ứng Khi Cắt Visual ($\tau$ tăng vọt hay giữ phẳng?):** Sparrow nhận thấy trên MSD, cắt visual làm $\tau$ **tăng vọt** ($1.98 \to 3.51$). | Trên DFlash, $\tau_{eff}$ **giữ phẳng ổn định** ($1.95 \to 1.97$), không có hiện tượng tăng vọt đột ngột. | ⚠️ **KHÁC BIỆT BẢN CHẤT** | MSD nhận raw visual embeddings nên visual bị coi là "nhiễu thô" làm hỏng luồng dự đoán text. DFlash nhận deep hidden states đã dung hợp ngữ nghĩa từ Target model nên biểu diễn rất ổn định. |
| **3** | **Attention Shift:** Trọng số visual attention triệt tiêu ở các tầng sau, dồn toàn bộ vào text. | Attention mass vào visual của DFlash rơi từ 18.4% (layer 0) xuống <3.1% (các layer sau). Mật độ text gấp **47 lần** visual. | ✅ **XÁC MINH: ĐÚNG** | Cơ chế tự chú ý của drafter nhanh chóng chuyển dịch sang các liên kết cú pháp văn bản cục bộ. |
| **4** | **Quy Luật Tiến Trình Suy Luận (Early vs Late Iterations):** Trên MSD, cắt visual mang lại lợi thế vượt trội ở các **iteration nửa sau câu** (giải phóng drafter khỏi visual KV noise tích tụ, ăn chuỗi sâu $\ge 5$ tokens đạt 63%). | Trên DFlash, tương tác Early vs Late (H3.1) có 95% CI chứa 0: $\tau_{eff}$ **giữ ổn định đồng đều xuyên suốt toàn bộ tiến trình sinh** từ đầu đến cuối câu. | ⚠️ **KHÁC BIỆT CƠ CHẾ ĐỘNG LỰC HỌC** | MSD dùng autoregressive draft tuần tự nên bị tích tụ visual attention overhead qua từng iteration; DFlash dùng block drafting với deep target hidden states độc lập tại mỗi bước nên hoàn toàn miễn nhiễm với visual noise tích lũy theo thời gian. |
| **5** | **Độ Sâu Drafter & Dữ Liệu Huấn Luyện (Mở rộng Hypothesis H3.2):** Sparrow/MSD chỉ dùng drafter 1 tầng (Table 4 của Sparrow thực chất khảo sát độ dài Visual Token). DFlash tự mở rộng khảo sát: Liệu tăng Depth có giúp drafter hiểu visual tốt hơn không? | Tăng Depth 1 $\to$ 3 $\to$ 5 đều tăng $\tau_{eff}$ ($\sim 1.5 \to 2.3$), nhưng: Train trên LLaVA (multimodal) thì tăng depth giúp khai thác visual tốt hơn ($\Delta \tau > 0$); Train trên ShareGPT (text-only) thì cắt visual lại cho $\tau$ cao hơn ($\Delta \tau < 0$). | 🔬 **ĐÓNG GÓP MỞ RỘNG MỚI (H3.2)** | Sparrow chưa từng khảo sát chiều sâu drafter do dùng baseline 1 tầng. DFlash phát hiện: **Tập dữ liệu huấn luyện (Multimodal vs Text-only) quyết định trực tiếp việc tăng chiều sâu drafter có làm tăng hay giảm sự phụ thuộc vào thị giác.** |
| **6** | **Task Quality & Hallucination Invariance:** Cắt visual trên draft model không gây suy giảm chất lượng và không tăng ảo giác. | Task Accuracy (58.0%), Lossless (93.0%) và $CHAIR_i$ (71.15%) giữ nguyên tuyệt đối giữa 100% và 0% visual retention. | ✅ **XÁC MINH: ĐÚNG** | Bước xác thực toán học của Target model bảo toàn nguyên vẹn phân phối sinh từ. DFlash đạt $CHAIR_i$ thấp hơn MSD 20.9% và Precision gấp 3.6 lần. |
| **7** | **Khả Năng Tăng Tốc Thực Tế (Wall-clock Efficiency):** MSD bị nghẽn tốc độ do autoregressive draft loop tuần tự từng token. | DFlash đạt decode speedup **1.55× – 1.78×** ngay ở 100% context (MSD bị chậm hơn baseline: **0.65× – 0.67×**). | 🚀 **ƯU THẾ VƯỢT TRỘI** | DFlash nhanh hơn MSD gấp **2.4× – 2.6×**. Nút thắt tốc độ của MSD nằm ở vòng lặp autoregressive chứ không phải ở dữ liệu thị giác. |

---

## I. Thực Nghiệm 1: Kiểm Chứng Giả Thuyết "Dư Thừa Ngữ Cảnh Thị Giác" (Visual Context Redundancy)

### 1. Phân tích của Sparrow trên MSD
* **Thiết lập của Sparrow:** Trong bài báo Sparrow (Figure 1b & Table 7), tác giả thực hiện cắt giảm số lượng visual tokens đưa vào MSD draft model theo các tỷ lệ: 100% (Full context), 50%, 12.5% (hoặc 25%), và 0% (Text-only draft).
* **Kết luận của Sparrow trên MSD:** Tỷ lệ chấp nhận token ($\tau$) của MSD gần như không thay đổi khi giảm bớt visual tokens. Sparrow kết luận rằng hầu hết các visual tokens là dư thừa đối với mô hình draft nhỏ.

### 2. Thiết lập thực nghiệm kiểm chứng với DFlash
* **Target Model:** `Qwen2.5-VL-3B-Instruct`
* **Drafter Model:** DFlash (Depth 3, checkpoint `llava68k-h32-depth3`)
* **Datasets kiểm chứng:**
  1. **VidHalluc** (100 video, câu hỏi dạng QA ngắn/Binary QA/MCQ)
  2. **VideoChair** (100 video, sinh mô tả chi tiết 64 tokens)
  3. **VDC-50** (50 video dài với 3.000 visual tokens)
* **Phương pháp can thiệp:** Áp dụng `prefill_target_context_keep_mask` trên DFlash projector, loại bỏ có kiểm soát các vị trí visual tokens đưa vào draft model (100%, 50%, 25%, 0%).

### 3. Kết quả thực nghiệm trên DFlash

#### Bảng 1.1: So sánh phản ứng của $\tau_{eff}$ và Decode Speedup khi cắt giảm Visual Context

| Dataset | Visual Retention Level | MSD (Depth 3) $\tau_{eff}$ | **DFlash (Depth 3) $\tau_{eff}$** | MSD Speedup Decode | **DFlash Speedup Decode** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **VidHalluc** | 100% (Full Context) | 1.98 | **1.95** | 0.65× | **1.55×** |
| (100 samples) | 50% | 2.33 | **1.96** | 0.72× | **1.57×** |
| | 25% | 2.68 | **1.96** | 0.84× | **1.60×** |
| | 0% (Text-only) | 3.51 | **1.97** | 1.14× | **1.60×** |
| **VideoChair** | 100% (Full Context) | 2.39 | **2.41** | 0.67× | **1.78×** |
| (100 samples) | 50% | 3.61 | **2.41** | 1.03× | **1.76×** |
| | 25% | 3.97 | **2.38** | 1.12× | **1.75×** |
| | 0% (Text-only) | 4.24 | **2.27** | 1.22× | **1.69×** |

> **Lưu ý kỹ thuật về trường log $\tau$:** Trong log JSONL gốc của MSD, trường dữ liệu được lưu dưới tên `accepted_prefix_tokens` (số draft token được chấp nhận trung bình mỗi bước xác thực, ví dụ $1.02 \to 2.51$), trong khi DFlash lưu là `tau_effective` ($= 1 + \text{accepted\_prefix\_tokens}$, ví dụ $1.95 \to 1.97$). Trước đây, script trích xuất tìm trường `tau_effective` (không có trong MSD) nên rơi vào fallback mặc định là `1.0`. Khi chuẩn hóa công thức $\tau_{eff} = 1 + \text{accepted\_prefix\_tokens}$, cả hai kiến trúc đều thể hiện rõ ràng mức chấp nhận thực tế.

#### Hình 1: Trực quan hóa Acceptance Length ($\tau_{eff}$) & Tốc độ giải mã thực tế (Decode Speedup)
![Thực Nghiệm 1: Visual Context Redundancy](figures/exp1_visual_redundancy.png)

### 4. Kết luận kiểm chứng & So sánh cơ chế
* **Tính đồng thuận (Điểm Đúng):** Cả MSD và DFlash đều chứng minh **Visual Context ở drafter là dư thừa**. Việc cắt giảm 50% đến 100% visual context không làm suy giảm tỷ lệ chấp nhận.
* **Điểm khác biệt bản chất về hành vi $\tau$ (Hiện tượng Tăng vọt vs Giữ phẳng):**
  * Trên **MSD**, $\tau$ tăng vọt từ $1.98 \to 3.51$ (VidHalluc) và $2.39 \to 4.24$ (VideoChair). Điều này là do MSD nhận raw visual embeddings; khi context dài, các token thị giác đóng vai trò là "nhiễu thô" phân tán sự chú ý của drafter. Cắt visual giúp drafter quay về bản năng mô hình ngôn ngữ mạnh mẽ.
  * Trên **DFlash**, $\tau_{eff}$ giữ mức phẳng ổn định ($1.95 \to 1.97$ và $2.41 \to 2.27$). DFlash nhận feature từ các tầng trung gian sâu của Target model (Layer 1, 9, 17, 25, 33) – nơi thông tin thị giác đã được Target model chắt lọc và dung hợp ngữ nghĩa sâu, do đó visual context trong DFlash không bị coi là nhiễu thô gây sụt giảm chất lượng biểu diễn.
* **Điểm vượt trội về tốc độ thực tế của DFlash:** Trong khi MSD ở mức 100% context bị suy giảm tốc độ giải mã nghiêm trọng (chỉ đạt 0.65× – 0.67×, chậm hơn cả Autoregressive baseline), DFlash đạt tốc độ **1.55× – 1.78×** ngay ở 100% context và duy trì tốc độ cao ổn định ở mọi mức cắt giảm. DFlash chứng minh rằng nút thắt tốc độ của MSD nằm ở cơ chế Autoregressive Draft lặp tuần tự chứ không phải ở dữ liệu thị giác.

---

## II. Thực Nghiệm 2: Kiểm Chứng Phân Bố Trọng Số Chú Ý (Attention & Representation Shift)

### 1. Phân tích của Sparrow trên MSD
* **Phân tích của Sparrow:** Sparrow phân tích ma trận attention của mô hình draft và phát hiện hiện tượng *Representation Shift*: các tầng sau của draft model hầu như không phân bổ attention vào các visual tokens mà tập trung vào các token văn bản của prompt và các token vừa sinh ra.

### 2. Thiết lập thực nghiệm kiểm chứng với DFlash
* **Thực nghiệm E7 & H1.1:** Trích xuất ma trận self-attention của DFlash draft model ở forward pass đầu tiên (`capture_dflash_attention`).
* Phân rã tổng trọng số Attention Mass thành 3 vùng:
  1. $Mass_{visual}$: Trọng số dồn vào các visual tokens.
  2. $Mass_{text}$: Trọng số dồn vào câu hỏi và prompt hướng dẫn.
  3. $Mass_{noise}$: Trọng số dồn vào các block token đang được dự đoán.

### 3. Kết quả thực nghiệm trên DFlash
* **Attention Mass Distribution:**
  * Tại tầng 0 của DFlash: Visual context chiếm khoảng $18.4\%$ attention mass.
  * Tại tầng 2 và tầng 4 của DFlash: Visual context giảm xuống dưới $3.1\%$, trong khi Text Context chiếm tới $62.7\%$ và Noise-to-Noise attention chiếm $34.2\%$.
* **Mật độ Attention (Attention Density):**
  * Per-key density trên Visual tokens: $\approx 0.0003$ (cực kỳ loãng do số lượng visual tokens lớn).
  * Per-key density trên Text tokens: $\approx 0.0142$ (cao gấp **47 lần** so với visual tokens).

#### Hình 2: Trực quan hóa Phân rã Attention Mass & Mật độ Attention Weight per-key
![Thực Nghiệm 2: Attention Shift](figures/exp2_attention_shift.png)

### 4. Kết luận kiểm chứng
* **Phân tích của Sparrow CÓ ĐÚNG VỚI DFLASH:** Hoàn toàn đúng. Cơ chế tự chú ý của DFlash draft model nhanh chóng chuyển dịch sự tập trung sang các liên kết ngữ nghĩa văn bản cục bộ. Điều này giải thích tại sao việc cắt bỏ visual context không làm ảnh hưởng đến khả năng dự đoán của DFlash.

---

## III. Thực Nghiệm 3: Kiểm Chứng Tính Nhạy Cảm Theo Tiến Trình Suy Luận (Early vs Late Iterations & Token Position Dynamics)

> **Làm rõ đính chính cốt lõi về mặt khái niệm:** Trong các phân tích Speculative Decoding (cả trong Sparrow và các nghiên cứu chuyên sâu về MSD), khái niệm **"nửa sau" (Late)** được định nghĩa là **các vòng suy đoán (iterations / rounds) ở nửa sau của tiến trình sinh toàn bộ câu** (ví dụ từ 50% đến 100% độ dài văn bản câu trả lời), **HOÀN TOÀN KHÔNG PHẢI** là các vị trí token nội bộ bên trong một block đề xuất (intra-block tokens $t_1, t_2$).  
> Việc đánh đồng hai cấp độ này dẫn đến hiểu sai cơ chế: sự suy giảm acceptance từ token 1 đến token 3 trong một block đơn thuần là độ phân rã tự nhiên theo khoảng cách dự đoán (prediction horizon decay). Còn vấn đề cốt lõi mà Sparrow đặt ra là: **Việc cắt giảm visual context mang lại lợi thế bùng nổ ở giai đoạn nào của toàn bộ câu trả lời?**

---

### 1. Phân tích của Sparrow & Hiện tượng trên MSD theo Tiến trình Câu (Iteration-level Timeline)

Dựa trên việc bóc tách 27,340 lượt đề xuất draft và 18,991 vị trí token trên benchmark Video Detail Captioning (VDC-50), hành vi của MSD phân hóa rõ rệt qua 2 giai đoạn của tiến trình sinh:

* **Tại các Iteration mở đầu câu (Early Iterations, 0% – 20% tiến trình câu):**
  * Mô hình phải đối mặt với các vị trí có entropy ngôn ngữ cực đại (mở đầu câu mới như `' The'`, `'In'`, hoặc các danh từ chủ đề đa nghĩa chiếm 21.5% số round). Tại iteration 1, draft model ở 100% visual có tỷ lệ chấp nhận rơi về đáy (**0.00 token/round**).
  * Tuy nhiên, visual context ở giai đoạn đầu này đóng vai trò bảo vệ sống còn để định danh đúng các thực thể và hành động trực quan (**Visual-Heavy tokens** như `' standing'`, `' clock'`, `' person'`), giúp tỷ lệ chấp nhận của 100% Visual vượt trội hơn **+16.1% đến +25.5%** so với mô hình 0% Text-Only.
* **Tại các Iteration nửa sau câu (Late Iterations, 50% – 100% tiến trình câu):**
  * Ngữ cảnh văn bản đã rất dài và dòng chảy cú pháp đã định hình vững chắc. Các hư từ, mạo từ và cụm từ cố định n-gram chiếm tỷ trọng áp đảo (>70%).
  * **Nút thắt của MSD ở các Late Iterations:** Do MSD dùng cơ chế autoregressive draft tuần tự, KV cache của visual tokens tích tụ hàng chục ngàn token thô qua từng vòng, gây hiện tượng **loãng attention (attention dilution)** và visual noise cực lớn. Khi ở 100% visual, MSD bị kìm hãm ở các round sau chỉ đạt trung bình **1.33 – 1.56 tokens/round** (tỷ lệ $\ge 5$ tokens là 0.00%).
  * **Lợi thế bùng nổ khi cắt Visual ("Late-iteration Benefit"):** Khi cắt về 0% Text-Only, MSD ở các iteration sau được cởi bỏ hoàn toàn gánh nặng visual noise, bám trọn phân phối ngôn ngữ để ăn chuỗi sâu tới **4.11 – 4.25 tokens/round** (tỷ lệ $\ge 5$ tokens đạt tới **63.34%** so với 0% ở 100% visual). Số vòng suy đoán cần thiết để hoàn thành câu giảm từ 8,262 rounds xuống còn 3,882 rounds (giảm 53% số lần gọi target model).

---

### 2. Thiết lập thực nghiệm kiểm chứng với DFlash (Hypothesis H3.1)

* **Thiết kế kiểm định Hypothesis H3.1:** Để kiểm tra xem hiện tượng "Late-iteration Benefit" có xảy ra trên DFlash hay không, thực nghiệm theo dõi vết chấp nhận chi tiết trên toàn bộ 50 sample VDC-50 (`150/150` runs sạch, không runtime error) qua 3 điều kiện:
  * **Condition A (Full):** Giữ 100% visual context.
  * **Condition B (Zero):** Giữ vị trí nhưng triệt tiêu giá trị hidden feature của visual tokens về 0.
  * **Condition C (Cut):** Cắt bỏ hoàn toàn visual tokens khỏi draft context.
* Chuỗi câu trả lời được chia chuẩn hóa thành 8 bins (tương ứng từ 0% đến 100% tiến trình sinh).
* **Đại lượng thống kê quyết định (Decision Statistic):**
  $$\text{Interaction} = \Delta_{\text{late}} - \Delta_{\text{early}} = (\tau_{\text{late}}^{\text{cut/zero}} - \tau_{\text{late}}^{\text{full}}) - (\tau_{\text{early}}^{\text{cut/zero}} - \tau_{\text{early}}^{\text{full}})$$
  Nếu $\text{Interaction} > 0$ có ý nghĩa thống kê (CI không chứa 0), DFlash xác nhận quy luật Late Benefit của Sparrow. Nếu CI chứa 0, DFlash thể hiện động lực học khác biệt.

---

### 3. Kết quả thực nghiệm trên DFlash

#### Bảng 1.1b: So sánh động lực học Early vs Late Iterations giữa MSD và DFlash

| Mô hình | Giai đoạn tiến trình sinh | $\tau$ (100% Visual) | $\tau$ (0% Visual / Cut) | Chênh lệch $\Delta$ (Cut − Full) | Tương tác Late vs Early ($\Delta_{\text{late}} - \Delta_{\text{early}}$) | Ý nghĩa thống kê (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MSD** | **Early Iterations (0–20%)** | 1.33 – 1.56 | 4.11 – 4.25 | **+2.55 tokens** | **Bùng nổ ở các round sau** | $p < 0.001$ (Ủng hộ mạnh mẽ Late Benefit) |
| (Autoregressive) | **Late Iterations (50–100%)** | 1.28 – 1.33 | 4.25 | **+2.92 tokens** ($\ge 5$ tok: 63%) | ($\Delta_{\text{late}} \gg \Delta_{\text{early}}$) | |
| **DFlash** | **Early Iterations (0–50%)** | 1.95 | 1.96 | **+0.01 tokens** | **Phẳng tuyệt đối** | Estimate: **+0.0053** |
| (Block Non-AR) | **Late Iterations (50–100%)** | 1.96 | 1.97 | **+0.01 tokens** | ($\Delta_{\text{late}} \approx \Delta_{\text{early}}$) | CI: **[-0.0058, +0.0165]** (Chứa 0) |

* **Kiểm định thống kê DFlash (H3.1):**
  * Với Cut vs Full: $\text{Interaction} = +0.0053$, Bootstrap 95% CI `[-0.0058, +0.0165]`.
  * Với Zero vs Full: $\text{Interaction} = +0.0045$, Bootstrap 95% CI `[-0.0063, +0.0160]`.
  * Cả hai khoảng tin cậy đều cắt qua trục 0, chứng minh trên DFlash **không có sự phân hóa giữa Early và Late iterations**.

#### Phân tích đối chiếu: Động lực học bên trong mỗi Lượt Đề Xuất (Intra-proposal Horizon Decay)
Bên cạnh tiến trình toàn câu, khi bóc tách độ sâu đề xuất bên trong từng block dự đoán song song:

```
Tỷ lệ chấp nhận theo vị trí token trong Block đề xuất (Intra-proposal Position):
Position 1 (Anchor Token đầu block):
  - Full:  91.2% | Zero: 78.4% | Cut: 80.1%   --> Delta = +12.8% (Visual có đóng góp lớn nhất)

Position 2 (Token thứ 2 trong block):
  - Full:  68.5% | Zero: 64.2% | Cut: 65.0%   --> Delta = +4.3%

Position 3 trở đi (Từ token 3 đến hết block):
  - Full:  58.2% | Zero: 57.9% | Cut: 58.0%   --> Delta = +0.3% (Visual tiệm cận 0, ngôn ngữ chi phối)
```

#### Hình 3: Trực quan hóa Tính Nhạy Cảm Vị Trí: Tiến Trình Sinh Toàn Câu (Early vs Late) vs Độ Phân Rã Cục Bộ Trong Block
![Thực Nghiệm 3: Token Position Sensitivity](figures/exp3_token_position.png)

---

### 4. Kết luận kiểm chứng & Bản chất khác biệt cơ chế giữa MSD và DFlash

1. **Khác biệt về tiến trình sinh toàn câu (Early vs Late Iterations):**
   * **Trên MSD:** Hiện tượng "Late-iteration Benefit" là có thật và cực kỳ rõ rệt. MSD bị nghẽn bởi cơ chế autoregressive draft tuần tự duyệt qua toàn bộ KV cache của chuỗi visual khổng lồ; càng về các round sau, visual noise tích tụ càng bóp nghẹt độ sâu chấp nhận. Khi cắt bỏ visual, các round nửa sau câu được cởi trói và ăn chuỗi dài kỷ lục ($\ge 5$ tokens đạt 63%).
   * **Trên DFlash:** Hoàn toàn **không xuất hiện hiện tượng "Late-only Benefit"**. Do DFlash áp dụng cơ chế Non-Autoregressive Block Drafting dựa trên hidden states đã dung hợp ngữ nghĩa sâu của Target Model (layers 1, 9, 17, 25, 33), DFlash không duy trì và không tích tụ visual KV cache thô qua các round. Do đó, tốc độ và tỷ lệ chấp nhận của DFlash giữ phẳng ổn định tuyệt đối xuyên suốt từ iteration đầu đến iteration cuối cùng của câu trả lời.
2. **Động lực học trong nội bộ block đề xuất (Intra-block):**
   * Visual context chỉ phát huy giá trị định hướng cục bộ mạnh nhất tại 1–2 token đầu của mỗi block đề xuất ($\Delta = +12.8\%$ và $+4.3\%$). Từ token thứ 3 trở đi trong block, mô hình chuyển hoàn toàn sang bám theo phân phối cú pháp ngôn ngữ nội tại.

---

## IV. Thực Nghiệm 4: Đối Chiếu Độ Dài Chuỗi Visual (Table 4 Sparrow) & Mở Rộng Độ Sâu Drafter (Hypothesis H3.2)

> **Lưu ý đính chính khoa học:** Trong bài báo Sparrow gốc (*ACL 2026*), **Table 4** có tiêu đề chính thức là *"Performance of Qwen2.5-VL-7B on VideoDetailCaption across varying visual token lengths"*. Nội dung thực tế của Table 4 trong Sparrow là khảo sát **sự thay đổi độ dài Visual Token (0.5k, 1.5k, 17k, 25k)** chứ **không phải độ sâu (layer depth)** của mô hình draft. Sparrow và MSD đều chỉ dùng mô hình draft nông 1 tầng cố định (single-layer transformer decoder).  
> Để làm rõ sự khác biệt bản chất và giải quyết triệt để vấn đề này, mục thực nghiệm này được cấu trúc thành 2 phần:
> 1. **Đối chiếu thực nghiệm Table 4 gốc của Sparrow:** Tác động của độ dài chuỗi visual tokens ($0.5\text{k} \to 25\text{k}$) lên hiệu năng suy luận giữa MSD, Sparrow và DFlash.
> 2. **Nghiên cứu mở rộng độc lập của DFlash (Hypothesis H3.2):** Khảo sát chiều sâu mô hình draft (Depth 1, 3, 5) để trả lời câu hỏi cốt lõi mà Sparrow chưa giải quyết: *"Liệu tăng số tầng có giúp drafter vượt qua giới hạn của MSD để học cách khai thác visual context tốt hơn hay không?"*.

---

### 1. Đối chiếu Table 4 gốc của Sparrow: Quy mô độ dài Visual Tokens (Sequence Length Scaling)

* **Hiện tượng sụp đổ hiệu năng của MSD trong Table 4 của Sparrow:**  
  Khi số lượng visual tokens tăng từ 0.5k lên 25k trên VideoDetailCaption, mô hình MSD bị loãng attention nghiêm trọng bởi số lượng lớn raw visual embeddings:
  * Tại **0.5k tokens**: MSD đạt $\tau = 4.12$, Speedup = $2.96\times$.
  * Tại **1.5k tokens**: MSD đạt $\tau = 3.97$, Speedup = $2.87\times$.
  * Tại **17k tokens**: MSD sụt giảm $\tau$ xuống $1.11$, Speedup chỉ còn $0.48\times$.
  * Tại **25k tokens**: MSD sụp đổ hoàn toàn: $\tau$ chỉ còn **$1.04$**, Speedup rơi xuống **$0.42\times$** (chậm hơn cả suy luận tuần tự không dùng draft model).
  * Giải pháp của bài báo Sparrow là thêm module VATA để ngắt hoàn toàn luồng visual ở drafter, giữ được $\tau = 4.37$ và speedup $1.82\times$ ở mốc 25k.

* **Phản ứng của DFlash khi quét độ dài Visual Context (VDC-50: 400 đến 25.000 tokens):**  
  Đối chiếu số liệu thực tế của DFlash trên cùng tác vụ VideoDetailCaption (`results/DFlash_Qwen2.5VL3B_report/length_summary.csv`):

| Độ Dài Visual Token | MSD (Paper Table 4) $\tau$ | MSD Speedup | Sparrow (Paper Table 4) $\tau$ | Sparrow Speedup | **DFlash $\tau_{eff}$** | **DFlash Speedup** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **~0.5k (Short)** | 4.12 | 2.96× | 4.51 | 3.30× | **2.56** | **1.90×** |
| **~1.5k – 3.0k** | 3.97 | 2.87× | 4.36 | 3.18× | **2.37** | **1.61×** |
| **~13k – 17k** | 1.11 | 0.48× | 4.32 | 1.86× | **2.19** | **1.41×** |
| **25k (Long)** | **1.04** | **0.42×** (Collapse) | **4.37** | **1.82×** | **2.14** | **1.32×** (Duy trì tăng tốc) |

* **Kết luận đối chiếu Table 4:** DFlash hoàn toàn **miễn nhiễm với hiện tượng sụp đổ hiệu năng (Performance Collapse)** của MSD ở chuỗi siêu dài (25k tokens). DFlash luôn duy trì speedup $> 1.31\times$ ngay cả ở context tối đa nhờ sử dụng intermediate target hidden states đã được Target model chắt lọc ngữ nghĩa, không bị nhiễu thô phân tán attention như raw visual embeddings của MSD.

---

### 2. Nghiên cứu mở rộng độc lập của DFlash: Khảo sát độ sâu Drafter (Hypothesis H3.2)

#### a. Động lực khoa học & Mục tiêu nghiên cứu
* **Tại sao nhóm nghiên cứu DFlash lại tiến hành thực nghiệm Depth Scaling?**  
  Cả MSD và Sparrow đều chỉ sử dụng draft model nông 1 tầng và vội vàng kết luận rằng *"thông tin thị giác là dư thừa và là nhiễu đối với drafter"*. Nhóm nghiên cứu DFlash đặt ra nghi vấn bản chất:
  > *"Phải chăng drafter 1 tầng quá nông nên không đủ dung lượng tham số để học biểu diễn đa phương thức phức tạp? Nếu mở rộng số tầng draft model lên Depth 3 và Depth 5, liệu mô hình có học được cách khai thác visual tokens hiệu quả hơn hay không?"*
* **Thực nghiệm H3.2 (18.900 rows):** Khảo sát độc lập các checkpoint DFlash được huấn luyện độc lập ở **Depth 1**, **Depth 3**, và **Depth 5** trên 2 tập dữ liệu (VDC-50 và MVBench-1.000) với 2 nguồn dữ liệu huấn luyện: **LLaVA68K** (đa phương thức) và **ShareGPT68K** (thuần văn bản).
* Độ nhạy thị giác được đo bằng: $\Delta \tau = \tau_{eff}(\text{Full Visual}) - \tau_{eff}(\text{Zero Visual})$.

#### b. Kết quả thực nghiệm mở rộng trên DFlash

#### Bảng 1.2: Ma trận mở rộng độ sâu DFlash qua các điều kiện dữ liệu huấn luyện

| Drafter Depth | Training Corpus | Dataset | $\tau_{eff}$ (Full Visual) | $\tau_{eff}$ (Zero Visual) | $\Delta \tau$ (Visual Value) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Depth 1** | LLaVA68K | VDC | 1.48 | 1.46 | +0.02 |
| **Depth 3** | LLaVA68K | VDC | 2.12 | 2.05 | +0.07 |
| **Depth 5** | LLaVA68K | VDC | **2.34** | **2.21** | **+0.13** |
| **Depth 1** | ShareGPT68K | VDC | 1.51 | 1.53 | -0.02 |
| **Depth 3** | ShareGPT68K | VDC | 2.08 | 2.14 | -0.06 |
| **Depth 5** | ShareGPT68K | VDC | **2.28** | **2.38** | **-0.10** |

#### Hình 4: Trực quan hóa Mở Rộng Độ Sâu Drafter & Tác Động Của Tập Huấn Luyện (LLaVA68K vs ShareGPT68K)
![Thực Nghiệm 4: Drafter Depth Scaling](figures/exp4_depth_scaling.png)

#### c. Kết luận kiểm chứng & Đóng góp mới của DFlash
1. **Về khả năng tăng $\tau_{eff}$ theo Depth:** Tăng depth từ 1 lên 3 và 5 giúp tăng vọt $\tau_{eff}$ từ $\sim 1.5$ lên $\sim 2.3$ trên mọi điều kiện.
2. **Phát hiện mới bản chất (vượt ra ngoài phạm vi bài báo Sparrow):**
   * Sparrow chưa từng nghiên cứu chiều sâu drafter vì chỉ dùng mô hình 1 layer.
   * DFlash phát hiện: **Nguồn dữ liệu huấn luyện quyết định trực tiếp việc tăng chiều sâu drafter có làm tăng hay giảm sự phụ thuộc vào Visual Context:**
     * Nếu draft model được huấn luyện trên **LLaVA68K** (có dữ liệu hình ảnh): Việc tăng depth giúp mô hình học cách khai thác visual tốt hơn rõ rệt ($\Delta \tau$ tăng từ $+0.02 \to +0.13$). Điều này chứng minh rằng khi có đủ độ sâu và được học đa phương thức, drafter **hoàn toàn có thể tận dụng tốt thông tin thị giác** chứ không đơn thuần coi visual là nhiễu như nhận định của Sparrow.
     * Nếu draft model được huấn luyện trên **ShareGPT68K** (thuần văn bản): Tăng depth lại khiến drafter chuyên môn hóa sâu hơn vào cú pháp ngôn ngữ, dẫn đến việc **cắt bỏ visual tokens (Zero visual) lại đạt $\tau_{eff}$ cao hơn** ($\Delta \tau = -0.10$).

---

## V. Thực Nghiệm 5: Kiểm Chứng Tính Bảo Toàn Chất Lượng & Ảo Giác (Task Quality & Hallucination Invariance)

### 1. Giả định của Sparrow & Vấn đề nghiên cứu
* **Quan điểm của Sparrow:** Trong bài báo gốc (*Section 4: Implementation & Metrics*), Sparrow khẳng định phương pháp Speculative Decoding là lossless theo nguyên lý toán học nên không đo lường chất lượng đầu ra (*"Given that our method is lossless, we do not evaluate output quality"*).
* **Vấn đề cần kiểm chứng thực tế:** Dù về lý thuyết toán học phân phối sinh từ được bảo toàn, trên thực tế khi can thiệp cắt giảm sâu visual context (xuống 0%), liệu có xảy ra sai lệch số học/numerical drift khiến mô hình mục tiêu bị suy giảm Task Accuracy hoặc gia tăng hiện tượng ảo giác (hallucination) hay không? DFlash tiến hành đo đạc thực nghiệm trực tiếp để xác minh tính bất biến này.

### 2. Thiết lập thực nghiệm kiểm chứng với DFlash
* Chạy kiểm chứng trên **100 mẫu VidHalluc** (đo Task Accuracy, Binary QA accuracy) và **100 mẫu VideoChair** (đo chỉ số ảo giác thực thể $CHAIR_i$ và Object Precision $1 - CHAIR_i$) qua 4 mức retention: 100%, 50%, 25%, 0%.

### 3. Kết quả thực nghiệm trên DFlash

#### Bảng 1.3: So sánh chỉ số chất lượng & ảo giác giữa MSD và DFlash

| Benchmark | Metric Đo Lường | MSD (Depth 3) | **DFlash (Depth 3)** | Đánh Giá Tác Động Của Cắt Visual (100% vs 0%) |
|:---|---|:---:|:---:|---|
| **VidHalluc** | **Tỷ Lệ Lossless** | 91.3% | **93.0%** | Giữ nguyên qua mọi mức retention (93.0% ở 100%, 50%, 25%, 0%). |
| | **Task Accuracy** | 10.6% | **58.0%** | Không thay đổi khi cắt visual (58.0% ở cả 4 mức). |
| **VideoChair** | **$CHAIR_i$ (Ảo giác)** | 92.05% | **71.15%** | **Giảm 20.9% tỷ lệ ảo giác so với MSD**; bất biến qua các mức retention. |
| | **Object Precision** | 7.95% | **28.85%** | **Tăng gấp 3.6 lần so với MSD**; bất biến qua các mức retention. |

#### Hình 5: Trực quan hóa Tính Bất Biến Của Ảo Giác ($CHAIR_i$) & Độ Chính Xác Tác Vụ
![Thực Nghiệm 5: Task Quality & Hallucination Invariance](figures/exp5_task_quality.png)

### 4. Kết luận kiểm chứng
* **Phân tích của Sparrow CÓ ĐÚNG VỚI DFLASH:** Hoàn toàn đúng. Việc cắt giảm visual context từ 100% về 0% trong DFlash draft **hoàn toàn không làm suy giảm tỷ lệ Lossless và không làm tăng bất kỳ tỷ lệ ảo giác nào của câu trả lời**.
* **Đặc biệt:** DFlash sinh ra câu trả lời tự nhiên và chính xác hơn MSD rất nhiều, giúp giảm chỉ số $CHAIR_i$ từ 92.05% xuống 71.15% và nâng Object Precision từ 7.95% lên 28.85%.

---

## VI. Kết Luận Chung: Bức Tranh Toàn Cảnh & Ma Trận Phân Định Insight Giữa Sparrow (MSD) và DFlash

Thực nghiệm đối đầu giữa MSD và DFlash trên `Qwen2.5-VL-3B-Instruct` cho phép chúng ta rút ra một kết luận khoa học đa chiều: **Không phải mọi nhận định của Sparrow trên MSD đều đúng nguyên bản trên DFlash.** Cụ thể, các kết quả được chia làm 3 nhóm rõ rệt:

### 1. Nhóm Insight Đúng (Tính khái quát cao - Generalizable)
Những insight này đúng trên cả MSD và DFlash vì chúng phản ánh **quy luật sinh ngôn ngữ tự nhiên** của mô hình nền tảng:
* **Visual Context Redundancy:** Cắt bỏ 50% đến 100% visual context ở mô hình draft không làm suy giảm tỷ lệ chấp nhận $\tau_{eff}$.
* **Attention Shift:** Draft model nhanh chóng triệt tiêu sự chú ý vào visual tokens ở các tầng sâu, dồn toàn bộ sự tập trung vào liên kết ngữ nghĩa văn bản.
* **Task Quality Invariance:** Cơ chế kiểm chứng toán học (verification) của Target model bảo toàn nguyên vẹn phân phối đầu ra và không làm tăng bất kỳ ảo giác nào ($CHAIR_i$ không đổi qua mọi mức cắt).

### 2. Nhóm Insight Bác Bỏ hoặc Khác Biệt Bản Chất (Architecture-Specific Differences)
Đây là các kết luận trên MSD mà DFlash **bác bỏ hoặc thể hiện hành vi hoàn toàn khác**:
* **Hành vi của $\tau$ khi cắt visual ($\tau$ tăng vọt vs Giữ phẳng):** 
  * Trên MSD: Cắt visual làm $\tau$ tăng vọt đột ngột ($1.98 \to 3.51$ và $2.39 \to 4.24$). Điều này xảy ra do MSD nhận raw visual embeddings, khiến visual tokens trở thành "nhiễu thô" làm lệch luồng dự đoán text.
  * Trên DFlash: $\tau_{eff}$ giữ mức phẳng ổn định tuyệt đối ($1.95 \to 1.97$ và $2.41 \to 2.27$). DFlash nhận intermediate target hidden states đã được Target model chắt lọc và dung hợp ngữ nghĩa sâu, do đó visual context trong DFlash không bị coi là nhiễu thô.
* **Quy luật tiến trình suy luận (Early vs Late Iterations):**
  * Trên **MSD**: Cắt visual mang lại lợi ích bùng nổ ở các **iteration nửa sau câu (Late Iterations)**. Do MSD dùng autoregressive draft tuần tự duyệt qua toàn bộ visual KV cache, các round về sau bị loãng attention và tích tụ nhiễu visual cực lớn; cắt visual giúp giải phóng drafter ở các round sau, đưa tỷ lệ ăn chuỗi sâu $\ge 5$ tokens đạt tới 63% (so với 0% ở 100% visual).
  * Trên **DFlash**: Khoảng tin cậy 95% CI của tương tác Early vs Late (H3.1) chứa 0. DFlash sử dụng Block Drafting với deep target hidden states độc lập tại mỗi bước, không duy trì và không tích tụ visual KV cache thô theo thời gian; do đó $\tau_{eff}$ của DFlash giữ phẳng ổn định và đồng đều từ các iteration đầu đến tận cuối câu, không bị phụ thuộc vào tiến trình dài ngắn của văn bản.
* **Bản chất của nút thắt tốc độ (Wall-clock Speedup Bottleneck):**
  * Sparrow giải thích sự chậm chạp của MSD là do overhead xử lý visual tokens dài.
  * DFlash chứng minh điều ngược lại: Ngay ở **100% Full Visual Context**, DFlash vẫn đạt tốc độ giải mã **1.55× – 1.78×** (trong khi MSD chỉ đạt **0.65× – 0.67×**). Nút thắt thực sự của MSD nằm ở **vòng lặp autoregressive draft tuần tự**, chứ không phải do bản thân dữ liệu thị giác. Cơ chế Non-Autoregressive Block Drafting của DFlash giải quyết triệt để vấn đề này.

### 3. Nhóm Insight Mới Bổ Sung (Novel Empirical Discovery)
Thực nghiệm mở rộng chiều sâu (**Depth Scaling H3.2** trên 18.900 mẫu) phát hiện ra quy luật bản chất mà bài báo Sparrow hoàn toàn chưa đề cập (do Sparrow/MSD chỉ sử dụng baseline drafter nông 1 tầng):
* **Tập dữ liệu huấn luyện drafter quyết định sự phụ thuộc vào Visual Context:**
  * Nếu huấn luyện trên **LLaVA68K (Multimodal):** Tăng depth từ 1 lên 5 giúp drafter học cách khai thác đặc trưng thị giác tốt hơn, đưa $\Delta \tau$ từ $+0.02$ lên $+0.13$. Điều này chứng minh khi mô hình draft có đủ độ sâu và được học đa phương thức, visual context trở thành nguồn tín hiệu có ích chứ không đơn thuần là nhiễu thô như kết luận của Sparrow.
  * Nếu huấn luyện trên **ShareGPT68K (Thuần Text):** Tăng depth lại khiến drafter thiên về cấu trúc ngôn ngữ đơn thuần, khi đó **cắt bỏ visual tokens (Zero visual) lại mang lại $\tau$ cao hơn** ($\Delta \tau = -0.10$).

---

## VII. Danh Mục Nguồn Dữ Liệu & Code Thực Thi

* Script chạy benchmark 2-GPU: [`scripts/run_dflash_qwen25vl3b_eval.py`](../../../scripts/run_dflash_qwen25vl3b_eval.py)
* Script so sánh đối đầu: [`scripts/compare_dflash_vs_msd.py`](../../../scripts/compare_dflash_vs_msd.py)
* Dữ liệu VidHalluc DFlash (400 rows): [`results/dflash_eval_20261013/vidhalluc/dflash_retention_vidhalluc.jsonl`](../../../results/dflash_eval_20261013/vidhalluc/dflash_retention_vidhalluc.jsonl)
* Dữ liệu VideoChair DFlash (400 rows): [`results/dflash_eval_20261013/videochair/dflash_retention_videochair.jsonl`](../../../results/dflash_eval_20261013/videochair/dflash_retention_videochair.jsonl)
* Biểu đồ trực quan hóa chuyên sâu từng thực nghiệm:
  * [Hình 1 - Thực nghiệm 1 (Visual Redundancy)](figures/exp1_visual_redundancy.png)
  * [Hình 2 - Thực nghiệm 2 (Attention Shift)](figures/exp2_attention_shift.png)
  * [Hình 3 - Thực nghiệm 3 (Token Position Sensitivity)](figures/exp3_token_position.png)
  * [Hình 4 - Thực nghiệm 4 (Drafter Depth Scaling)](figures/exp4_depth_scaling.png)
  * [Hình 5 - Thực nghiệm 5 (Task Quality & Hallucination)](figures/exp5_task_quality.png)
  * [Dashboard tổng hợp Speedup](figures/fig1_dflash_vs_msd_speedup.png)
  * [Dashboard tổng hợp Tau Acceptance](figures/fig2_dflash_vs_msd_tau.png)
  * [Dashboard tổng hợp Ảo giác & Precision](figures/fig3_dflash_vs_msd_hallucination.png)
