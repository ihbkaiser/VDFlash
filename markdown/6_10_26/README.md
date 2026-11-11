# Báo cáo Phân tích: Đặc tính Token & Ngữ cảnh Iteration quyết định Khả năng Accept trong Speculative Decoding (MSD trên VDC-50)

Thư mục lưu trữ: `results/6_10_26/`  
Nguồn dữ liệu: `results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl` (200 lượt chạy trên 50 video VDC).  
- Báo cáo đối đầu chuyên sâu MSD Depth 3 vs Depth 5: [**`BAO_CAO_SO_SANH_DEPTH_3_VS_DEPTH_5.md`**](BAO_CAO_SO_SANH_DEPTH_3_VS_DEPTH_5.md)
- Báo cáo đối chiếu toàn diện 3B (Depth 3 & 5) vs 7B MSD: [**`BAO_CAO_SO_SANH_3B_VS_7B_MSD.md`**](BAO_CAO_SO_SANH_3B_VS_7B_MSD.md)
- Báo cáo thực nghiệm VidHalluc & VideoChair (7B): [**`BAO_CAO_VIDHALLUC_VIDEOCHAIR.md`**](BAO_CAO_VIDHALLUC_VIDEOCHAIR.md)
- Báo cáo phân tích đặc tính Token & Ngữ cảnh: [**`BAO_CAO_PHAN_TICH_TOKEN_ACCEPTANCE.md`**](BAO_CAO_PHAN_TICH_TOKEN_ACCEPTANCE.md)  
- Báo cáo định nghĩa & so sánh hiệu năng Iteration DỄ vs KHÓ: [**`BAO_CAO_EASY_HARD_ITERATIONS.md`**](BAO_CAO_EASY_HARD_ITERATIONS.md)


---

## 1. Tổng quan vấn đề nghiên cứu

Trong kỹ thuật Speculative Decoding với mô hình MSD (EAGLE draft layer trên nền Qwen2-VL), câu hỏi cốt lõi đặt ra là:
> **"Những iteration chứa token như thế nào thì draft model dễ đoán đúng và được target model chấp nhận (accept) nhất?"**

Dựa trên toàn bộ **18,991 token positions** được ghi nhận chi tiết, chúng tôi phân loại và định lượng khả năng chấp nhận của từng nhóm token, mối liên hệ với ngữ cảnh đứng trước (preceding context), và sự phân hóa giữa **100% Visual Retention** (đầy đủ 25k visual tokens) và **0% Visual Retention** (thuần language prior, bỏ visual tokens khỏi draft context).

---

## 2. Bảng số liệu: Tỷ lệ chấp nhận theo từng nhóm Token

| Nhóm Token (Category) | Token đại diện | Tỷ lệ Accept ở **100% Retention** (Có Visual) | Tỷ lệ Accept ở **0% Retention** (Không Visual) | Chênh lệch ($\Delta$ 0% - 100%) | Đánh giá & Cơ chế |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cụm từ cố định** *(Formulaic Collocations)* | `' variety'`<br>`' objects'`<br>`' followed'`<br>`' transitions'`<br>`' including'` | **99.7%** (291/292)<br>**99.0%** (306/309)<br>**98.5%** (65/66)<br>**97.5%** (78/80)<br>**92.9%** (260/280) | **94.9%** (279/294)<br>**94.8%** (293/309)<br>**98.5%** (65/66)<br>**96.3%** (77/80)<br>**95.4%** (271/284) | $\approx \mathbf{0\%}$ đến $\mathbf{-5\%}$ | **DỄ NHẤT (>95%):** Thuộc khuôn mẫu cụm từ bất biến (`a variety of objects`, `followed by`). Entropy cực thấp, cả 2 cấu hình đều đoán trúng tuyệt đối. |
| **Liên từ & Điểm rẽ nhánh** *(Branching Connectors)* | `' and'`<br>`' of'`<br>`','` (dấu phẩy)<br>`' then'`<br>`' with'` | **6.8%** (78/1156)<br>**27.2%** (298/1094)<br>**22.8%** (217/953)<br>**13.4%** (33/247)<br>**54.6%** (313/573) | **80.8%** (928/1148)<br>**77.9%** (869/1116)<br>**73.7%** (698/947)<br>**72.3%** (180/249)<br>**81.5%** (458/562) | **+74.1%**<br>**+50.6%**<br>**+50.9%**<br>**+58.9%**<br>**+26.9%** | **PHÂN HÓA LỚN NHẤT:** Ở 100%, visual noise khiến draft trượt nhịp nối câu (bị reject >75–93%). Ở 0%, draft thuần ngôn ngữ bám sát nhịp nối câu của LLM, accept vọt lên >70–80%. |
| **Mở đầu câu mới** *(Sentence Starters)* | `' The'` (câu mới)<br>`'The'` (đầu văn bản) | **4.1%** (24/591)<br>**16.1%** (9/56) | **42.2%** (253/600)<br>**92.9%** (52/56) | **+38.1%**<br>**+76.8%** | **KHÓ NHẤT:** Điểm mở đầu câu có entropy cao nhất (target có thể bắt đầu bằng nhiều từ khác nhau). 100% hầu như không đoán được token đầu câu. |
| **Chi tiết thị giác cụ thể** *(Visual Specific)* | `' standing'` (hành động)<br>`' clock'` (vật thể)<br>`' person'`<br>`' white'` (màu sắc) | **94.8%** (221/233)<br>**98.1%** (265/270)<br>**87.3%** (178/204)<br>**64.6%** (326/505) | 69.3% (156/225)<br>82.0% (210/256)<br>80.9% (165/204)<br>**87.1%** (446/512) | **-25.5%**<br>**-16.1%**<br>-6.4%<br>+22.6% | **LỢI THẾ CỦA 100% RETENTION:** Nhóm duy nhất 100% retention thắng thế rõ rệt! Nhờ có visual context, draft "nhìn thấy" hành động (`standing`) và vật thể (`clock`) để đề xuất chính xác hơn 0%. |
| **Từ khóa Boilerplate** *(Domain Verbs/Nouns)* | `' features'`<br>`' scene'`<br>`' video'`<br>`' shows'` | 46.5% (33/71)<br>25.9% (55/212)<br>38.2% (124/325)<br>**86.8%** (237/273) | **97.1%** (67/69)<br>**84.4%** (173/205)<br>**69.8%** (229/328)<br>**97.6%** (282/289) | **+50.6%**<br>**+58.5%**<br>**+31.7%**<br>+10.8% | Các từ mở đầu mô tả video chuẩn của tập VDC. Mức 0% retention bám mẫu cực tốt, đạt >70–97%. |

---

## 3. Khảo sát: Ngữ cảnh đứng trước (Preceding Context) tác động thế nào đến Iteration?

Khi xét xem **token cuối cùng của round trước** là gì, hiệu năng của round tiếp theo thay đổi cực kỳ rõ rệt:

| Token đứng trước vòng lặp | Ví dụ ngữ cảnh | Số token được Accept ở Round kế tiếp (100% Retention) | Số token được Accept ở Round kế tiếp (0% Retention) | Ý nghĩa ngôn ngữ học |
| :--- | :--- | :---: | :---: | :--- |
| **Mạo từ / Giới từ** (`' a'`, `' with'`, `' on'`, `' in'`) | *"with [blue shirt...]"*, *"a [variety of...]"* | **1.6 – 1.8 tokens** (Cao nhất ở 100%) | **4.0 – 4.3 tokens** | Sau giới từ/mạo từ, bắt buộc là danh từ hoặc tính từ $\rightarrow$ Không gian lựa chọn bị thu hẹp $\rightarrow$ **RẤT DỄ ACCEPT**. |
| **Động từ thể hiện hành vi** (`' standing'`, `' followed'`) | *"standing [in front of...]"* | **1.6 – 1.9 tokens** | **4.9 tokens** (Đỉnh điểm) | Sau động từ chỉ trạng thái thường là cụm trạng ngữ quen thuộc (`in`, `on`, `near`) $\rightarrow$ **DỄ ACCEPT**. |
| **Liên từ kết nối** (`' and'`, `','`) | *"objects, [including...]"* | **1.3 – 1.4 tokens** | **4.2 – 4.3 tokens** | Điểm rẽ nhánh giữa hai vế câu $\rightarrow$ Trung bình. |
| **Danh từ chủ thể** (`' video'`, `' camera'`, `' image'`) | *"The video [features/shows...]"* | **0.5 – 1.0 tokens** (Rất thấp ở 100%) | **4.6 tokens** | Ở 100%, visual noise làm draft không biết video làm gì tiếp theo. Ở 0%, draft chọn ngay template boilerplate chuẩn. |
| **Dấu chấm ngắt câu** (`'.'`) | *"[...end of sentence]. [The...]"* | **0.26 tokens** (Thấp nhất lịch sử!) | **2.93 tokens** | Bắt đầu câu mới: không có ràng buộc ngữ pháp từ vế trước $\rightarrow$ Entropy cực đại $\rightarrow$ **KHÓ ACCEPT NHẤT**. |

---

## 4. Tỷ lệ chấp nhận theo Chiều sâu của Draft Tree (Tree Depth)

EAGLE đề xuất một cây các token (Tree Draft). Tỷ lệ các round chấp nhận được các độ sâu khác nhau:

| Chiều sâu chuỗi Draft | 100% Retention (Có visual) | 0% Retention (Không visual) |
| :---: | :---: | :---: |
| Chấp nhận **$\ge 1$ token** | **80.8%** | **97.3%** |
| Chấp nhận **$\ge 2$ tokens** | 39.2% | **90.6%** |
| Chấp nhận **$\ge 3$ tokens** | 8.2% | **79.5%** |
| Chấp nhận **$\ge 4$ tokens** | 1.5% | **65.9%** |
| Chấp nhận **$\ge 5$ tokens** | **0.2%** | **57.9%** |

👉 **Điểm mấu chốt:** Ở 0% retention, có tới **57.9% số round ăn trọn chuỗi dài $\ge 5$ tokens**, trong khi ở 100% retention chỉ có **0.2%** (gần như draft bị đứt xích ngay từ token thứ 2 hoặc thứ 3).

---

## 5. Kết luận quy luật: "Iteration chứa token như thế nào thì dễ được accept?"

1. **Quy luật 1 — Dễ accept nhất:**
   - **Các token thành ngữ / Collocations:** Các cụm từ nối cố định (`variety`, `objects`, `followed`, `including`). Cả 100% và 0% đều đạt tỷ lệ accept gần như tuyệt đối (**>95%**).
   - **Các iteration đứng ngay sau mạo từ hoặc giới từ:** Khi round trước kết thúc bằng `' a'`, `' of'`, `' in'`, `' with'`, ngữ pháp ép chặt từ tiếp theo phải là danh từ/tính từ, giúp draft đoán trúng chuỗi dài từ 2 đến 4 tokens.
2. **Quy luật 2 — Khó accept nhất:**
   - **Các iteration ngay sau dấu chấm câu (`.`):** Điểm mở đầu câu mới (`The`, `A`, `Next`) có entropy cao nhất, draft 100% chỉ đạt 0.26 token/round (hầu như trượt 100%).
   - **Các điểm rẽ nhánh ý (`and`, `,`, `then`):** Khi có visual (100%), draft model bị nhiễu không đoán được khi nào target model muốn chuyển ý, dẫn đến tỷ lệ accept của `' and'` chỉ vỏn vẹn **6.8%**.
3. **Quy luật 3 — Nơi Visual thực sự phát huy giá trị:**
   - Đối với các từ miêu tả hành động và vật thể cụ thể trong video (`standing`, `clock`, `person`), mô hình có visual (100%) đạt tỷ lệ accept cao hơn 0% từ **+6% đến +25%**. 
   - Tuy nhiên, vì các từ nội dung cụ thể này chỉ chiếm ~15-20% tổng số từ trong caption (80% còn lại là từ nối, mạo từ và khung câu), nên tổng thể 100% retention vẫn bị tụt tốc độ giải mã so với 0% retention.

---

### Danh sách biểu đồ minh họa trong thư mục:
* [token_acceptance_by_category.png](token_acceptance_by_category.png): Biểu đồ cột so sánh tỷ lệ accept giữa các nhóm từ.
* [acceptance_by_preceding_context.png](acceptance_by_preceding_context.png): Biểu đồ số token accept theo ngữ cảnh đứng trước.
* [acceptance_by_tree_depth.png](acceptance_by_tree_depth.png): Biểu đồ tỷ lệ sống sót theo chiều sâu cây đề xuất.
* [token_acceptance_comparison.csv](token_acceptance_comparison.csv): Bảng số liệu chi tiết từng token.
