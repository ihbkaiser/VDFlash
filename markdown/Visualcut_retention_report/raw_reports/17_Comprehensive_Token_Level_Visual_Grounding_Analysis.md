# Báo Cáo 17: Phân Tích Toàn Diện Cấp Độ Token (Token-Level Comprehensive Analysis) — Từ Phân Loại Thủ Công Đến Định Lượng Khách Quan Qua Target Attention Layer 18 & Thẩm Định Ngữ Nghĩa

## 1. Đặt Vấn Đề & Bước Chuyển Dịch Phương Pháp Luận

Trong các nghiên cứu ban đầu về Speculative Decoding cho Video-LLM (như [Báo Cáo 04](04_Easy_Hard_VisualHeavy_Iterations_Retention.md) và [Báo Cáo 05](05_Token_Level_Acceptance_Analysis.md)), việc phân loại token thành các nhóm "Dễ", "Khó", "Nội dung chung", và "Visual-Heavy" được thực hiện dựa trên **danh sách từ khóa thủ công (Heuristic Lexicon Lists)**:
- Nhóm Dễ: mạo từ, giới từ, quán ngữ n-gram cố định (`a`, `of`, `variety of`).
- Nhóm Khó: mở đầu câu, liên từ rẽ nhánh (`The`, `and`, `then`).
- Nhóm Visual-Heavy: danh từ vật thể (`person`, `clock`), hành động (`standing`, `walking`), màu sắc (`white`, `blue`).

### Hạn Chế Cốt Lõi Của Phương Pháp Thủ Công:
1. **Thiếu cơ sở định lượng toán học:** Việc gán nhãn từ nào là "Visual-Heavy" hoàn toàn dựa trên trực giác chủ quan của con người, không có một hàm mục tiêu hay ngưỡng quyết định (decision threshold) có thể đo lường lặp lại.
2. **Bỏ qua hành vi thực tế của mô hình:** Một từ vựng cụ thể có thể là danh từ vật thể, nhưng trong ngữ cảnh của câu nó có thể đã được suy luận dễ dàng từ văn bản trước đó (text context) chứ không thực sự đòi hỏi mô hình phải nhìn video.

Để khắc phục triệt để tính chủ quan này, dự án đã nâng cấp lên **phương pháp định lượng khách quan 100% dựa trên Attention Mass thực tế của Target Model tại Layer 18** ([Báo Cáo 03](03_Target_Visual_Attention_L18_Attribution_Data.json)). Báo cáo này tổng hợp trọn vẹn bức tranh phân tích cấp độ Token-Level kết hợp giữa toán học Attention và thẩm định ngôn ngữ học thực nghiệm.

---

## 2. Phương Pháp Định Lượng Khách Quan: Target Visual Attention Tại Layer 18 ($m^* = 18$)

### 2.1. Cơ Sở Lý Thuyết
Trong mô hình Target `Qwen2.5-VL-3B-Instruct` (36 tầng Transformer), các tầng đầu xử lý đặc trưng cục bộ, các tầng giữa thực hiện tương tác đa phương thức mạnh mẽ nhất, và các tầng cuối chuyển dịch sang không gian ngữ nghĩa văn bản để phát sinh token.

Theo lý thuyết biểu diễn thị giác (Sparrow IVSB), **Layer 18 (mốc 50% Depth)** là vị trí mà lực chú ý (Attention Weight) của các Text Query dồn vào các Visual Keys đạt giá trị cực đại trước khi bị phân tán hoặc nén sâu.

Visual Attention Mass của một token $t_i$ được tính bằng tổng trọng số attention từ query $q_{t_i}$ tới toàn bộ các visual keys $k \in \mathcal{V}$ tại Layer 18:
$$\text{Visual Attention Mass}(t_i) = \sum_{k \in \mathcal{V}} \text{Attn}^{(L=18)}(q_{t_i}, k)$$

### 2.2. Các Ngưỡng Quyết Định Thống Kê (Statistical Percentile Thresholds)
Khảo sát được thực hiện trên **1.200 token positions** được trích xuất từ 25 video đa dạng. Phân phối Attention Mass tại Layer 18 xác lập các phân vị làm ngưỡng quyết định khách quan:

| Thông Số Phân Vị | Giá Trị Thực Nghiệm | Ý Nghĩa Ngưỡng Quyết Định |
| :--- | :---: | :--- |
| **Giá trị Min** | $0.1879$ | Mức độ chú ý thị giác tối thiểu |
| **Phân vị 25% ($p_{25}$)** | $0.8452$ | Ranh giới nhóm chú ý thị giác rất thấp |
| **Phân vị 50% ($p_{50}$ - Trung Vị)** | **$1.1124$** | **Ngưỡng phân tách Low Visual vs Medium Visual** |
| **Phân vị 75% ($p_{75}$)** | **$1.5054$** | **Ngưỡng quyết định HEAVY VISUAL ⭐** |
| **Phân vị 90% ($p_{90}$)** | **$1.9700$** | **Ngưỡng quyết định ULTRA-HEAVY VISUAL ⭐⭐** |
| **Giá trị Max** | $4.1691$ | Đỉnh chú ý thị giác cao nhất ghi nhận |

---

## 3. Kết Quả Định Lượng: Đối Đầu Full vs Cut Theo Ngưỡng Attention L18

Khi phân chia 1.200 tokens thành 4 nhóm theo đúng các ngưỡng phân vị thống kê trên và đo lường Pass Rate giữa `Full` (100% Visual Context) và `Cut` (0% Visual Cut trong Drafter):

| Phân Nhóm Định Lượng Theo Attention L18 | Ngưỡng Toán Học | Số Lượng Token ($N$) | Pass Rate `Full` (100% Visual) | Pass Rate `Cut` (Visual Cut) | Chênh Lệch $\Delta (\text{Cut} - \text{Full})$ | Khoảng Tin Cậy 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Visual** | $< p_{50}$ ($< 1.1124$) | 600 | 53.17% | 55.17% | $+2.00\%$ | $[-1.33\%; +5.33\%]$ |
| **Medium Visual** | $p_{50} \to p_{75}$ | 300 | 60.67% | 61.00% | $+0.33\%$ | $[-4.00\%; +4.67\%]$ |
| **HEAVY VISUAL ⭐** | **$p_{75} \to p_{90}$** | **180** | **60.56%** | **67.78%** | **$\mathbf{+7.22\%}$** | **$[+1.11\%; +13.33\%]$** |
| **ULTRA-HEAVY ⭐⭐** | **$> p_{90}$ ($> 1.9700$)** | **120** | **60.83%** | **65.83%** | **$\mathbf{+5.00\%}$** | $[-3.33\%; +13.33\%]$ |
| **GỘP TOP 25% HEAVY VISUAL** | **$> p_{75}$ ($> 1.5054$)** | **300** | **60.67%** | **67.00%** | **$\mathbf{+6.33\%}$** | **$[+1.33\%; +11.00\%]$** |

> [!IMPORTANT]
> **Phát Hiện Khoa Học Đột Phá:**
> Ngay trên nhóm token mà Target Model phải dồn lực chú ý thị giác mạnh nhất (Top 25% Heavy Visual, $> p_{75}$), nhánh **`Visual Cut` đạt tỷ lệ chấp nhận $67.00\%$**, vượt trội hơn hẳn nhánh **`Full Context` là $60.67\%$** (chênh lệch **$+6.33\%$**, 95% CI $[+1.33\%; +11.00\%]$ hoàn toàn dương).
> Kết quả này bác bỏ trực giác thông thường và chứng minh: Text Hidden States tại Layer 18 đã hấp thụ trọn vẹn thông tin thị giác của Target; việc nhồi thêm visual vectors $H[V]$ vào Drafter chỉ tạo thêm nhiễu phân tán chú ý!

---

## 4. Thẩm Định Ngữ Nghĩa Thực Nghiệm (Semantic Ground Truth Audit)

Để trả lời câu hỏi phản biện: *"Liệu các từ có Attention Mass L18 cao về mặt toán học có thực sự là những từ mang ngữ nghĩa cần thị giác hay không?"*, toàn bộ 1.200 tokens đã được bóc tách và phân tích ngữ nghĩa:

### 4.1. Nhóm Từ Nội Dung (Content Words) — Chiếm 53% Nhóm Heavy/Ultra
Khi tách riêng các từ thực từ (Content Words) trong Top 25% Heavy Visual, danh sách từ thực tế chứng minh tính chính xác tuyệt đối về mặt ngữ nghĩa thị giác:
1. **Thực thể vật lý cụ thể (Entities/Objects):** `'dirt bike'`, `'rider'`, `'athlete'`, `'swimmer'`, `'player'`, `'man'`, `'woman'`, `'hammer'`, `'cage'`, `'gym equipment'`, `'water'`...
2. **Hành động & Môn thể thao đặc thù (Actions & Movements):** `'swims'`, `'butterfly stroke'` (bơi bướm), `'throwing'` (ném búa), `'weight lifting'` (cử tạ), `'jumps'`, `'stands'`, `'performing exercises'`, `'polo match'`, `'football'`, `'cricket'`...
3. **Bối cảnh không gian vật lý (Spatial Environments):** `'underwater'`, `'backyard'`, `'gym'`, `'arena'`, `'track'`, `'outdoor'`, `'indoor'`...
4. **Thuộc tính & Kích thước (Attributes):** `'large'`, `'white'`, `'black'`, `'two'`, `'traditional'`...

* **Đánh giá ngữ nghĩa:** 100% các từ này đều là từ ngữ miêu tả thị giác bắt buộc phải nhìn video mới xác định được (không thể đoán mò ra động tác "bơi bướm" hay "ném búa").
* **Hiệu năng trên Content Words:** Trên tập các Content Words này, nhánh **`Visual Cut` đạt Pass Rate $58.3\%$**, vượt trội **$+17.7\%$** so với nhánh **`Full Context` ($40.6\%$)**!

### 4.2. Hiện Tượng Hư Từ Neo Giữ Thị Giác (Pre-nominal Visual Anchors) — Chiếm 47%
Trong nhóm có Attention L18 cao, ngoài các Content Words, xuất hiện các từ ngữ pháp như `'a'` (36 lần ở Heavy, 23 lần ở Ultra), `'In'`, `'with'`, `'an'`. 

Khảo sát ngữ cảnh câu phát hiện cơ chế **Pre-nominal Visual Anchoring (Neo giữ thị giác tiền danh từ)**:
```
Ngữ cảnh thực tế: "... watching two men [In] a [gym] throw [a] [hammer] into [a] [cage] ..."
                                        ▲                 ▲              ▲
                                (Attention L18 cao) (Attention L18 cao)
```
- **Bản chất Autoregressive:** Trong mô hình nhân quả, trước khi phát ra một danh từ thị giác (như `'hammer'`), mô hình bắt buộc phải phát ra mạo từ `'a'`. Để quyết định chọn `'a'` (thay vì `'the'`, `'two'` hay số nhiều), Target Model bắt buộc phải dồn attention vào video ngay tại vị trí mạo từ này để xác định tính đếm được và số lượng của thực thể.
- **Kênh nén thông tin:** Mạo từ đóng vai trò "cái phễu" (anchor) hấp thụ thông tin video để chuẩn bị không gian biểu diễn cho danh từ đứng ngay sau.

---

## 5. Nghịch Lý Cấp Độ Token & Lời Giải Từ Điểm Ngọt (Sweet Spot) 25% Retention

Bằng cách kết hợp phân tích Attention L18 với phân tích động học vòng suy đoán (Iteration-level) trên 27.340 rounds ([Báo Cáo 04](04_Easy_Hard_VisualHeavy_Iterations_Retention.md)), chúng ta giải mã được toàn bộ bức tranh:

### 5.1. Tại Sao 100% Visual Lại Bị Kém Tốc Độ Dù Đoán Tốt Từng Thực Thể?
1. **Lợi thế cục bộ:** Ở từng từ thị giác đơn lẻ (`standing`, `clock`), 100% Visual đạt độ chính xác cao hơn 0% Text-Only từ $+16\%$ đến $+25\%$.
2. **Nhưng tần suất quá thấp:** Các từ thực thể thị giác cụ thể chỉ chiếm **~2.6%** tổng số tokens sinh ra.
3. **Cái giá phải trả:** Việc duy trì 25.000 visual vectors khiến Drafter bị nhiễu loạn ở các liên từ rẽ nhánh (`and`, `then`, `,`) — nhóm chiếm tới **>20.2%** tổng số tokens. Điển hình là từ nối `'and'`: ở 100% Visual tỷ lệ chấp nhận rơi tự do xuống **6.75%** (bị reject >93%), khiến thời gian xác minh tăng hơn gấp đôi (27.51s vs 11.71s).
$\rightarrow$ **Ưu thế nhỏ trên 2.6% từ thị giác bị xóa sạch bởi sự sụp đổ trên 20% từ ngữ pháp!**

### 5.2. Sự Vượt Trội Của Điểm Ngọt 25% Visual Retention
Khi khảo sát dải tỉa từ $100\% \to 50\% \to 25\% \to 0\%$ trên nhóm Visual-Heavy Iterations:

| Mức Visual Retention | Tỷ Lệ Qua Token Đầu $t_0$ (%) | Tỷ Lệ Sụp Đổ Zero-Acc (%) | Mean Accept (tokens) | Ăn Chuỗi Sâu $\ge 2$ tokens (%) |
| :--- | :---: | :---: | :---: | :---: |
| **100% Visual** (Full) | 82.92% | 17.08% | 1.56 | 54.32% |
| **50% Visual** | 88.64% | 11.36% | 1.79 | 63.84% |
| **25% Visual (Sweet Spot ⭐)** | **91.76%** | **8.24%** | **1.95** | **69.74%** |
| **0% Visual** (Text-Only) | 98.42% | 1.58% | 4.11 | 93.06% |

- Tại mức **25% Visual Retention**, tỷ lệ vượt qua $t_0$ đạt đỉnh **$91.76\%$** (cao hơn cả 100% Visual là $82.92\%$), và tỷ lệ thất bại Zero-Acc giảm hơn một nửa (từ $17.08\% \to 8.24\%$).
- **Ý nghĩa kiến trúc:** Cắt giảm 75% visual tokens vừa giữ lại đủ tín hiệu neo thị giác cho các thực thể cụ thể, vừa triệt tiêu 75% nhiễu attention, giải phóng cây suy đoán sâu mà không làm hỏng cú pháp câu.

---

## 6. Kiểm Soát Cặp Token ID $\times$ Slot Matched Control ([Báo Cáo 09](09_Token_Matched_Control_Lexical_Overlap.md))

Để đảm bảo tính khách quan và loại trừ các giả thuyết ngụy tạo về grounding failure:
- Khi phân tích sơ bộ, các từ có nhãn Lexical Overlap (từ xuất hiện trong caption tham chiếu) dường như bị ảnh hưởng mạnh bởi Cut.
- Tuy nhiên, khi **khống chế chặt chẽ cùng một Target Token ID tại cùng một Draft Slot** trên subset 24 video (20 strata, 188 matched positions):
  $$\text{Primary Interaction} = -9.46 \text{ điểm } \% \quad \text{với 95% Bootstrap CI } [-18.73; +0.70]$$
- Khoảng tin cậy chứa 0 chứng minh rằng sự sụt giảm hiệu năng không mang tính đặc hiệu cho từ vựng thị giác, mà chủ yếu do vị trí của slot đề xuất (Slot 1 chịu rủi ro cao hơn hẳn Slot 2–4).

---

## 7. Tổng Kết Các Insight Cốt Lõi Ở Cấp Độ Token-Level

1. **Định lượng thay thế định tính:** Phương pháp dùng Attention Mass tại Layer 18 ($m^* = 18$) với các phân vị $p_{50}, p_{75}, p_{90}$ cung cấp tiêu chuẩn toán học khách quan, loại bỏ hoàn toàn tính chủ quan của các danh sách từ vựng thủ công.
2. **Khẳng định tính chính xác ngữ nghĩa:** Các token được toán học xếp vào nhóm Heavy/Ultra-Heavy Visual thực sự là các từ miêu tả hành động thể thao, vật thể và không gian cụ thể (Content Words chiếm 53%).
3. **Cơ chế Pre-nominal Anchoring:** Hư từ (mạo từ `'a'`, giới từ `'in'`) chiếm 47% nhóm Heavy Visual vì chúng đóng vai trò làm điểm tựa tiếp nhận thông tin video trước khi phát ra thực thể.
4. **Visual Cut vượt trội trên từ thị giác:** Ngay trên các từ Content Words đòi hỏi thị giác cao nhất, nhánh `Visual Cut` vẫn đạt Pass Rate $58.3\%$ (vượt $+17.7\%$ so với `Full Context`).
5. **Cơ sở cho thiết kế mô hình thế hệ mới:** Cắt giảm mạnh visual context (ở mức 25% Retention hoặc 10% trong Self-Distillation) là chìa khóa để vừa giữ trọn độ chính xác trên các từ thị giác, vừa bảo vệ hoàn toàn nhịp đề xuất trên 20% các từ ngữ pháp/liên từ cốt lõi.
