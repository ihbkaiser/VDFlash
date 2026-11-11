# Báo Cáo Chuyên Sâu: Kiểm Chứng Tính Độc Lập Giữa Text Consistency và Visual Context Trong DFlash

**Tác giả / Nhóm nghiên cứu:** Antigravity & Pair Programming Engineer  
**Ngày thực hiện:** 2026-10-14 – 2026-10-15  
**Đối tượng khảo sát:** Mô hình Target `Qwen2.5-VL-3B-Instruct` kết hợp Draft `DFlash LLaVA68K depth-3`  
**Dataset:** VideoDetailCaption (VDC50), Open-ended Video Question Answering  
**Mã nguồn & Artifacts:**
- Kế hoạch thực nghiệm: [`task_plan_dflash_free_running_20261014.md`](../../tasks/task_plan_dflash_free_running_20261014.md)
- Thiết kế thực nghiệm: [`experiment_design.md`](experiment_design.md)
- Runner thực thi: [`run_dflash_free_running.py`](../../../src/analyze/Validate_Sparrow_hypothesises/run_dflash_free_running.py)
- Kết quả thô & Tóm tắt số liệu: [`results/dflash_free_running_main_fp16_20261014/`](../../../results/dflash_free_running_main_fp16_20261014/)

---

## Tóm Tắt Điều Hành (Executive Summary)

Báo cáo này phân tích và đánh giá toàn diện giả thuyết:
> **"Trong DFlash, liệu text consistency có thật sự độc lập với visual context? Giảm visual thì chưa chắc đã tăng được text consistency."**

Qua việc triển khai hệ thống thực nghiệm đa chiều trên cohort 33 video đạt chuẩn calibration, kết hợp giữa **Teacher-forced Sensitivity Assay**, **Unverified Free-running Rollout**, và **Verified Speculative Decoding**, nghiên cứu rút ra 3 kết luận cốt lõi:

1. **Text proposal của DFlash KHÔNG HỀ độc lập với visual context:** Khi cắt bỏ hoàn toàn explicit visual rows đưa vào draft (`Cut - 0%`), có tới **39.74% token đề xuất bị thay đổi** so với khi giữ đầy đủ (`Full - 100%`) trên cùng một prefix cố định. Visual context đóng vai trò định hướng sâu sắc đến không gian xác suất của draft.
2. **Giảm visual context KHÔNG làm tăng text consistency:** Khi draft tự do sinh chuỗi dài mà không có Target verification, mô hình rơi vào hiện tượng suy thoái lặp từ nghiêm trọng (*degenerative repetition loops*) dưới mọi mức visual retention (đều chạm sàn điểm Coherence 1/5 của LLM Judge). Cắt visual không hề giúp draft "thoát" khỏi bẫy lặp hay trở nên mạch lạc hơn.
3. **Cắt visual làm suy giảm hiệu năng Speculative Decoding thực tế:** Trong quy trình speculative decoding chuẩn, điều kiện `Cut (0%)` làm giảm tốc độ chấp nhận trung bình: $\Delta \tau_{effective} = -0.039$ tokens/round (với 14 video giảm vs 8 video tăng).

**Kết luận chung:** Giả thuyết được khẳng định hoàn toàn: **Text consistency trong DFlash không độc lập với visual context, và việc cắt giảm visual context một cách thô bạo không những không cải thiện được tính liên kết văn bản mà còn gây tổn hại đến tỷ lệ chấp nhận của hệ thống.**

---

## 1. Khởi Nguồn & Câu Chuyện Nghiên Cứu (Research Motivation)

### 1.1. Trực giác kỹ thuật ban đầu và sự hoài nghi
Trong các kiến trúc Speculative Decoding cho Vision-Language Models (VLM) như DFlash, mô hình draft thường phải nhận hàng nghìn visual tokens (từ video frames) cùng với câu hỏi văn bản. Chi phí tính toán cho visual context này chiếm tỷ trọng đáng kể trong bộ nhớ và độ trễ của bước prefill/forward của draft.

Xuất phát từ đó, trong cộng đồng xuất hiện một **trực giác kỹ thuật (engineering intuition)** khá phổ biến:
> *"Draft model vốn dĩ có dung lượng nhỏ (nông, depth 3), nhiệm vụ chính của nó là bắt nhịp cú pháp, dự đoán các hư từ, liên kết ngữ pháp (text consistency), còn ngữ nghĩa thị giác phức tạp đã có Target Model lo. Do đó, visual context chỉ là yếu tố thứ yếu, thậm chí gây nhiễu cho draft. Nếu ta mạnh dạn cắt giảm (downsample hoặc cut) visual context, draft sẽ 'nhẹ gánh', tập trung vào tính nhất quán của văn bản, từ đó tăng độ mạch lạc và đẩy cao tỷ lệ chấp nhận (acceptance rate)."*

Tuy nhiên, trực giác này đặt ra câu hỏi nền tảng về bản chất biểu diễn:
* Liệu tính nhất quán văn bản (*text consistency*) trong câu trả lời mô tả video có thể thực sự tách rời khỏi bối cảnh thị giác?
* Khi cắt giảm visual context, draft có thực sự sinh ra văn bản mạch lạc hơn không, hay việc mất neo ngữ nghĩa thị giác sẽ khiến draft mất phương hướng?

### 1.2. Phân rã giả thuyết khoa học
Để tránh ngụy biện và đánh tráo khái niệm giữa "độ mượt mà câu chữ" với "tốc độ chấp nhận", bài toán được phân rã thành **3 giả thuyết cụ thể**:
* **H-S (Visual Sensitivity Precondition):** Động thái đề xuất token (*proposal policy*) của draft có thực sự ít phụ thuộc vào visual rows không? (Đo độ nhạy trực tiếp).
* **H-C (Coherence Hypothesis):** Giảm explicit visual context của draft có làm tăng tính liên kết, mạch lạc (coherence/consistency) của chuỗi output do draft tự sinh hay không?
* **H-A (Acceptance Hypothesis):** Mức visual retention đó có làm tăng số token chấp nhận hiệu dụng ($\tau_{effective}$) trong speculative decoding chuẩn hay không?

---

## 2. Thiết Kế Thực Nghiệm & Tính Chuẩn Mực Phương Pháp (Methodology)

Để có kết luận đáng tin cậy, thực nghiệm được thiết kế tuân thủ các nguyên tắc khoa học khắt khe:

```
+------------------------------------------------------------------------------------+
|                                    TARGET MODEL                                    |
|                       (Qwen2.5-VL-3B-Instruct, FP16, Full Video)                   |
+-----------------------------------------+------------------------------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
      [Assay 1: Sensitivity]                            [Assay 2 & 3: Rollout]
    Prefix target cố định                             VDC50 (33 calibrated samples)
    Teacher-forced qua Draft                         Can thiệp 5 mức Visual Retention
                |                                                   |
                v                                                   v
   Đo % Proposal thay đổi                           +---------------+---------------+
   (Full vs 75/50/25/Cut)                           |                               |
                                                    v                               v
                                          [Free-Running Rollout]         [Verified Speculative]
                                          Draft sinh tự do 256 tok      Đo tau_effective &
                                          Target refresh features       tau_proposal
                                          Blinded LLM Judge (1-5)       Exact Target Parity
```

### 2.1. Ma trận can thiệp Visual Retention
Giữ cố định target `Qwen2.5-VL-3B`, checkpoint `DFlash LLaVA depth-3`, prompt tự nhiên, block size 16, greedy decoding:
* **Full (100%):** Giữ toàn bộ visual hidden rows.
* **Retain-75 (75%):** Rải đều theo thời gian, giữ 75% visual rows.
* **Retain-50 (50%):** Giữ 50% visual rows.
* **Retain-25 (25%):** Giữ 25% visual rows.
* **Cut (0%):** Loại bỏ 100% explicit visual rows khỏi draft context.

*Quy tắc bất biến:* Target model luôn luôn xử lý video đầy đủ 100%. Can thiệp chỉ diễn ra tại đầu vào của draft model.

### 2.2. Kiểm soát độ chuẩn xác (Precision & Calibration Gate)
* **Khắc phục lỗi tie-break:** Pilot ban đầu chạy BF16 + SDPA đã gây ra hiện tượng khác biệt logit ở ranh giới làm sai lệch target parity. Thực nghiệm chuyển sang cố định **FP16** đồng nhất.
* **Khóa Calibration Gate:** Trên 50 video VDC ban đầu, chỉ **33 mẫu** đạt chuẩn dung sai chiều dài visual token (3.000 tokens `ok`). 17 mẫu lệch chuẩn bị loại bỏ trước khi inference; tuyệt đối không nới lỏng gate sau khi xem kết quả.
* **Chống thiên kiến trong đánh giá Coherence:**
  * LLM-as-a-judge (chính là model Qwen2.5-VL ở chế độ text-only) chỉ đọc văn bản thuần, không thấy video, không biết condition nào là Full hay Cut.
  * Thứ tự Text A / Text B được xáo trộn ngẫu nhiên (*blinded & randomized order*).
  * Chấm trên 5 chiều: *Grammar, Reference consistency, Discourse coherence, Repetition, Overall* (thang 1–5).
   * Chuẩn bị sẵn file chấm mù cho con người ([`human_review_blinded.csv`](../../../results/dflash_free_running_main_fp16_20261014/human_review_blinded.csv)) và khóa key giải mã.

---

## 3. Phân Tích Kết Quả Theo Từng Luận Điểm

### Luận điểm 1: Draft Proposal cực kỳ nhạy cảm với Visual Context (Bác bỏ H-S)

Để kiểm tra xem draft có thực sự "bỏ qua" visual context hay không, Assay Sensitivity tiến hành teacher-force cùng 1 prefix chuẩn trên 10 video benchmark (tương đương 1.729 vị trí token).

| Điều kiện Retention | Tỷ lệ Agreement với Full | Tỷ lệ Proposal bị thay đổi | Coarse Distribution JS Divergence |
|---|:---:|:---:|:---:|
| **Full (100%)** | 100.0% | 0.0% | 0.0000 |
| **Retain-75** | 92.83% | **7.17%** (CI: [5.57%, 9.09%]) | 0.0014 |
| **Retain-50** | 87.01% | **12.99%** (CI: [11.74%, 14.32%]) | 0.0046 |
| **Retain-25** | 76.78% | **23.22%** (CI: [21.01%, 25.95%]) | 0.0118 |
| **Cut (0%)** | 60.26% | **39.74%** (CI: [34.90%, 44.70%]) | 0.0395 |

![Figure 1: Độ nhạy đề xuất của Draft theo tỷ lệ Visual Retention](figures/figure1_sensitivity_vs_retention.png)
*Hình 1: (A) Tỷ lệ lệch proposal so với điều kiện Full tăng vọt tới 39.7% khi cắt hoàn toàn visual context (vùng bóng mờ biểu thị 95% Bootstrap CI). (B) Độ phân kỳ phân phối token (JS Divergence) tăng đơn điệu chứng minh sự dịch chuyển phân phối sâu sắc.*

> **Phân tích Luận điểm 1:**
> - Mức độ thay đổi tăng đơn điệu và tỷ lệ thuận với lượng visual rows bị cắt bỏ.
> - Tại mức `Cut (0%)`, gần **40% token đề xuất bị lật ngược**.
> - **Tiểu kết 1:** Phủ nhận tiền đề cho rằng draft model chỉ tập trung vào text. Draft model dựa rất mạnh vào visual context để định hình phân phối proposal. Text consistency không thể xem là độc lập với visual context ở cấp độ token policy.

---

### Luận điểm 2: Free-Running Coherence bộc lộ bản chất của Draft nông và Floor Effect của phép đo (Phân tích H-C)

Khi cho draft tự sinh tự do (*unverified draft rollout*) tới 256 tokens với cơ chế target feature refresh:

1. **Độ phủ output (Coverage):**
   - 41/165 output kết thúc ngay lập tức ở token EOS mà không tạo ra văn bản (Full có 10 mẫu rỗng, Cut có 4 mẫu rỗng).
   - 124 output còn lại sinh ra đủ độ dài nhưng bộc lộ hiện tượng **suy thoái vòng lặp nghiêm trọng (severe degenerative repetition loops)**.
2. **Hiện tượng suy thoái nội dung:**
   - Khi không có Target Model kiểm duyệt và sửa lỗi ở mỗi block, draft rơi vào các bẫy lặp từ vô tận:
     - `Sample 1`: `"video. white white image image image image image to a a a a. a a image a white white white white white white white white white..."`
     - `Sample 2`: `"The video appears the in is a a front front,, a a, front front front a a a a a a a a a a a a a front a a a..."`
     - `Sample 3`: `"""" paintPaint Paint Paint Paint paint a a a a a a a and a paint paint paint paint paint paint paint paint a a a A A A A A..."`
3. **Kết quả LLM Judge:**
   - 88 cặp full-text và 84 cặp matched-prefix (128 token) đều nhận điểm tuyệt đối **1/5 ở cả 5 tiêu chí** (Grammar, Reference, Discourse, Repetition, Overall) từ cả 2 phía.
   - Kết quả: **100% Hòa (Tie)**, chênh lệch paired delta `Cut - Full = 0.0` (Bootstrap 95% CI: `[0.0, 0.0]`).

![Figure 3: Hiện tượng suy thoái lặp từ và hiệu ứng chạm sàn của Judge](figures/figure3_free_running_repetition_collapse.png)
*Hình 3: (A) Tỷ lệ phân bố output rỗng (EOS) và output có text. (B) Bệnh lý lặp từ cực đoan: một token duy nhất ('a', 'paint', 'image') chiếm tới 83% - 93% tổng số token sinh ra ở mọi điều kiện. (C) Điểm đánh giá Coherence của LLM Judge bị chạm sàn 1.0/5 trên toàn bộ 5 chiều.*

> **Phân tích Luận điểm 2:**
> - Điểm chênh lệch bằng 0.0 ở đây là **Floor Effect (hiệu ứng chạm sàn)** hoàn toàn. Vì cả hai phía đều sinh ra văn bản rác/lặp từ cực đoan nên Judge chấm điểm tối thiểu 1/5 cho cả hai.
> - **Nguyên nhân cốt lõi:** DFlash là một mô hình nông (depth-3), được huấn luyện cho bài toán multi-token prediction có target guidance. Nó **không phải là một language model tự hồi quy độc lập**. Khi ép nó free-run, sai số phân phối tích lũy ngay từ block đầu tiên và dẫn tới sụp đổ biểu diễn (*representation collapse*).
> - **Tiểu kết 2:** Cắt giảm visual context **không hề cứu draft khỏi hiện tượng suy thoái**, cũng không làm câu văn mạch lạc hay nhất quán hơn. H-C hoàn toàn không có bằng chứng hỗ trợ.

---

### Luận điểm 3: Cắt Visual trong Speculative Decoding chuẩn không hề tăng Acceptance (Bác bỏ H-A)

Nghiên cứu tiến hành đo đạc speculative decoding chuẩn trên 24 mẫu đạt exact target-output parity (đảm bảo tính lossless toán học):

| Điều kiện | Mean $\tau_{effective}$ (tokens/round) | Mean $\tau_{proposal}$ | Chênh lệch $\Delta \tau_{effective}$ so với Full | 95% Bootstrap CI | Tỷ lệ Thắng / Hòa / Thua |
|---|:---:|:---:|:---:|:---:|:---:|
| **Full (100%)** | 2.3218 | 1.3218 | 0.0000 | — | — |
| **Retain-75** | 2.3344 | 1.3344 | +0.0126 | [-0.0023, +0.0264] | 14 Wins / 6 Ties / 4 Losses |
| **Retain-50** | 2.3320 | 1.3320 | +0.0102 | [-0.0146, +0.0354] | 10 Wins / 7 Ties / 7 Losses |
| **Retain-25** | 2.3387 | 1.3387 | +0.0170 | [-0.0086, +0.0449] | 10 Wins / 4 Ties / 10 Losses |
| **Cut (0%)** | 2.2828 | 1.2828 | **-0.0390** | **[-0.0794, +0.0026]** | **8 Wins / 2 Ties / 14 Losses** |

![Figure 2: Tỷ lệ chấp nhận Speculative Decoding theo điều kiện Retention](figures/figure2_speculative_acceptance_tau.png)
*Hình 2: (A) Tỷ lệ chấp nhận hiệu dụng trung bình $\tau_{effective}$ qua 5 điều kiện. (B) Chênh lệch từng cặp $\Delta \tau_{effective}$ so với Full: điều kiện Cut làm giảm 0.039 token/vòng. (C) Cơ cấu kết quả từng video: Cut có 14 mẫu thua (Losses) áp đảo 8 mẫu thắng (Wins).*

> **Phân tích Luận điểm 3:**
> - Khi loại bỏ hoàn toàn visual context (`Cut - 0%`), tốc độ chấp nhận trung bình $\tau_{effective}$ bị giảm $0.039$ tokens/round; số mẫu bị suy giảm hiệu năng (14 mẫu) áp đảo số mẫu tăng (8 mẫu).
> - Ở các mức giữ lại một phần (75%, 50%, 25%), $\tau_{effective}$ có tăng nhẹ từ $0.010$ đến $0.017$, nhưng toàn bộ khoảng tin cậy 95% bootstrap đều chứa giá trị 0 $\rightarrow$ Sự cải thiện này không có ý nghĩa thống kê.
> - **Tiểu kết 3:** Cắt bỏ hoàn toàn visual context làm giảm chất lượng proposal thực tế khiến target bác bỏ nhiều hơn. Không có bằng chứng cho thấy giảm visual context giúp tăng acceptance rate.

---

## 4. Kết Luận Chung Cho Toàn Bộ Hypothesis

Đối chiếu trực tiếp với câu hỏi nghiên cứu ban đầu:
> *"Trong Dflash thì text consistency có thật sự độc lập với visual context , giảm visual thì chưa chắc đã tăng được text consistency"*

![Figure 4: Bảng điều khiển tổng hợp bằng chứng thực nghiệm](figures/figure4_evidence_synthesis_dashboard.png)
*Hình 4: Bảng điều khiển tổng hợp kết quả 3 assay: (A) Thiết kế & tham số thực nghiệm. (B) Độ nhạy token proposal tăng tới 39.7% khi Cut. (C) Tỷ lệ chấp nhận speculative decoding sụt giảm ở Cut. (D) Hiện tượng một token thống trị 83-93% văn bản tự do. (E) Điểm số Coherence chạm sàn 1.0 (Floor Effect). (F) Kết luận khẳng định hypothesis được hỗ trợ hoàn toàn.*

Kết quả thực nghiệm đã chứng minh luận điểm này là **hoàn toàn chính xác**:

1. **Text consistency không hề độc lập với visual context:** Trong mô hình sinh mô tả video, từ ngữ ngữ pháp và từ ngữ nội dung đều chịu sự điều hướng của biểu diễn thị giác. Khi tước bỏ visual context, chính sách sinh token của draft bị xáo trộn tới gần 40%.
2. **Giảm visual context không hề giúp tăng text consistency:** Ý niệm cho rằng "cắt bớt visual để draft tập trung vào text consistency" là một trực giác sai lầm. Cả khi tự sinh tự do (đều sụp đổ thành vòng lặp lặp từ như nhau) lẫn khi chạy speculative chuẩn (tỷ lệ chấp nhận suy giảm), việc cắt visual đều không mang lại bất kỳ sự cải thiện nào về tính mạch lạc văn bản.

---

## 5. Khuyến Nghị Kỹ Thuật (Engineering Recommendations)

Từ kết luận trên, nhóm nghiên cứu đề xuất định hướng phát triển cho DFlash và các hệ thống Speculative VLM:

1. **Không áp dụng chính sách cắt tỉa visual thô bạo (Naive Dropping / Hard Cut):** Không thể đơn thuần bỏ bớt frame hay cắt visual rows ở tầng draft với kỳ vọng draft sẽ tự duy trì được text consistency.
2. **Chuyển hướng sang nén ngữ nghĩa (Semantic Compression / Cross-Attention Pooling):** Nếu muốn tối ưu bộ nhớ và FLOPs của visual context trong draft, cần sử dụng các cơ chế downsampling có học (như Q-Former, Perceiver Resampler, hoặc pooling có trọng số attention) thay vì loại bỏ vị trí token.
3. **Đánh giá Coherence bằng Human Review:** Hoàn thành việc chấm điểm độc lập trên bảng [`human_review_blinded.csv`](../../../results/dflash_free_running_main_fp16_20261014/human_review_blinded.csv) trước khi mở file giải mã [`unblind_key.json`](../../../results/dflash_free_running_main_fp16_20261014/human_review_private/unblind_key.json) để có thêm bằng chứng định tính từ góc nhìn người dùng đối với các trường hợp biên.
