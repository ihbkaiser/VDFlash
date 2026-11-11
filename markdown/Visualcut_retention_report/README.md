# TỔNG HỢP TOÀN DIỆN THỰC NGHIỆM VISUAL CUT VÀ VISUAL RETENTION
## Repository: Video-DFlash / Speculative Decoding cho Video-LMM

> **Thư mục lưu trữ:** `markdown/Visualcut_retention_report/`  
> **Thời gian tổng hợp:** Ngày 09 tháng 11 năm 2026  
> **Phạm vi khảo sát:** Toàn bộ chuỗi thực nghiệm can thiệp thị giác (**Visual Cut, Visual Retention, Target Layer Attention Attribution, First Rejection Hazard, Verify-Attention Context Selection**) trên các kiến trúc Draft Autoregressive (MSD) và Block-Parallel Diffusion (DFlash, DSpark) ghép cặp với Target `Qwen2.5-VL-3B-Instruct` và `Qwen2-VL-7B-Instruct`.

---

## 1. BẢN ĐỒ TÀI LIỆU VÀ TRUY VẾT BÁO CÁO GỐC (RAW REPORTS)

Toàn bộ các tài liệu gốc, báo cáo chuyên sâu và tệp số liệu thực nghiệm đã được sao chép và chuẩn hóa tên tại thư mục [`raw_reports/`](raw_reports/):

| Mã | Tên tệp trong `raw_reports/` | Đối tượng & Phạm vi khảo sát | Trọng tâm & Kết luận chính |
| :---: | :--- | :--- | :--- |
| **01** | [`01_DFlash_5Configs_VisualCut_Retention_VDC50.md`](raw_reports/01_DFlash_5Configs_VisualCut_Retention_VDC50.md) | **DFlash 3-layer (5 configs)**, Target Qwen2.5-VL-3B, VDC50, H200 (6.000 cases). | Khảo sát 5 cấu hình trích xuất tầng target qua 4 budget (400 đến 25K tokens) $\times$ 6 điều kiện. `Zero` luôn tệ nhất; `Cut` bão hòa hoặc tương đương `Full` ở 25K. |
| **02** | [`02_DSpark_3Layer_VisualCut_Attention_A100.md`](raw_reports/02_DSpark_3Layer_VisualCut_Attention_A100.md) | **DSpark 3-layer Balanced `[1,9,17,25,33]`**, Target Qwen2.5-VL-3B, A100 (360 cases). | Bằng chứng speedup sụp đổ: ở 25K tokens, dù $\tau \approx 1.83 - 1.90$, E2E speedup rớt xuống **$0.86\times - 0.90\times$** do chi phí nạp/chiếu context lớn. |
| **03** | [`03_Target_Visual_Attention_L18_Attribution_Data.json`](raw_reports/03_Target_Visual_Attention_L18_Attribution_Data.json) | **Target Qwen2.5-VL-3B Layer 18 Attention Attribution**, 25 video (1.200 tokens). | Dữ liệu gốc gán nhãn token định lượng qua Attention tại Layer 18 (50% depth). Nhóm Heavy-Visual có Pass Rate ở nhánh `Cut` đạt **67.0%**, cao hơn `Full` (**60.67%**). |
| **04** | [`04_Easy_Hard_VisualHeavy_Iterations_Retention.md`](raw_reports/04_Easy_Hard_VisualHeavy_Iterations_Retention.md) | **MSD Qwen2-VL-7B**, 200 lượt suy luận VDC-50, 27.340 draft rounds. | Phân loại 4 nhóm Iteration (Dễ, Khó, Nội dung chung, Visual-Heavy). Phát hiện **Sweet Spot ở 25% Visual Retention** (Zero-Acc giảm từ 17.08% xuống 8.24%). |
| **05** | [`05_Token_Level_Acceptance_Analysis.md`](raw_reports/05_Token_Level_Acceptance_Analysis.md) | **MSD Qwen2-VL-7B**, bóc tách 18.991 token positions VDC-50. | Phân tích từ vựng: các từ thị giác cụ thể (`' standing'`, `' clock'`) bị giảm acceptance 16–25% nếu cắt sạch visual, trong khi từ ngữ pháp $>95\%$ acceptance. |
| **06** | [`06_Target_Verify_Attention_Context_Selection.md`](raw_reports/06_Target_Verify_Attention_Context_Selection.md) | **DFlash Context Pruning**, Target verify attention (M0–M5 selectors). | Prune KV cache động bằng attention của target sau khi verify block. M2 prune xuống 20% KV nhưng phát sinh overhead scoring 7–8 ms/decode, chưa đạt speedup thực tế. |
| **07** | [`07_DFlash_VisualCut_Distribution_Replay.md`](raw_reports/07_DFlash_VisualCut_Distribution_Replay.md) | **DFlash Fixed-Trajectory Replay**, 50 video $\times$ 4 budgets (200 pairs). | Cắt visual làm thay đổi phân phối xác suất mạnh (JSD tăng rõ từ 0.05 lên 0.15 ở 25K), nhưng `target-hit@1` trung bình chỉ dao động rất nhỏ. |
| **08** | [`08_Prefix_First_Rejection_RiskSet.md`](raw_reports/08_Prefix_First_Rejection_RiskSet.md) | **DFlash 17.089 Complete Blocks**, phân tích First Rejection Hazard. | Thất bại tập trung ở **Slot 1–2** của block draft 16 tokens. Tỷ lệ reject tại 2 slot đầu chiếm $>75\%$; 0/17.998 block ăn trọn 15/15 token. |
| **09** | [`09_Token_Matched_Control_Lexical_Overlap.md`](raw_reports/09_Token_Matched_Control_Lexical_Overlap.md) | **DFlash 24 video**, 2.127 blocks, khống chế `Token ID × Slot`. | Kiểm định chặt chẽ nhóm Lexical Overlap: độ lệch mismatch giữa Cut và Full là $-9.46$ pp (95% CI $[-18.73; +0.70]$ chứa 0) $\rightarrow$ chưa chứng minh có grounding failure. |
| **10** | [`10_TextState_Intervention_Pilot12.md`](raw_reports/10_TextState_Intervention_Pilot12.md) | **DFlash Cohort 12 video (3K tokens)**, can thiệp Text States ($G$, $A$, $N$). | Can thiệp visual vào text state ($A$) làm giảm prefix agreement ~32%. Tuy nhiên CI chứa 0 và phát hiện bug lệch số học pre-visual $\rightarrow$ dẫn tới Diagnostic Audit. |
| **11** | [`11_Sparrow_MSD_vs_DFlash_Comparison.md`](raw_reports/11_Sparrow_MSD_vs_DFlash_Comparison.md) | **Đối chiếu toàn diện Sparrow / MSD và DFlash** qua các mốc context. | So sánh sự khác biệt bản chất giữa Draft tuần tự (AR 1-layer) và Draft song song (Diffusion multi-layer). |
| **12** | [`12_Text_Suffix_Pruning_After_Visual_Cut.md`](raw_reports/12_Text_Suffix_Pruning_After_Visual_Cut.md) | **Pruning Text Suffix** sau khi đã cắt bỏ 100% visual context. | Khảo sát các mức giữ lại text suffix (100%, 75%, 50%, 25%, 0%); $\tau$ tụt dốc đều đặn khi cắt text từ trái sang phải, chứng minh text context là cốt lõi. |
| **13** | [`13_Thought_Mechanisms_MSD_vs_DFlash.md`](raw_reports/13_Thought_Mechanisms_MSD_vs_DFlash.md) | **Ghi chú lập luận cơ chế (02/11/2026)**. | Tổng hợp các suy luận giải thích tại sao Visual Cut có tác dụng ngược nhau giữa MSD và DFlash. |

---

## 2. TỔNG HỢP CÁC KẾT QUẢ ĐỊNH LƯỢNG THEN CHỐT

### 2.1. Hiệu năng Visual Cut trên 5 cấu hình DFlash (Báo cáo 01 — 6.000 cases trên H200)

| Cấu hình Draft | Tầng Target trích xuất | Budget (Tokens) | $\tau_{\text{eff}}$ (`full`) | $\tau_{\text{eff}}$ (`cut`) | $\tau_{\text{eff}}$ (`zero`) | $\Delta \tau$ (`cut − full`) [95% CI] | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`high_only`** | `[26, 28, 30, 32, 34]` | 400 | 2.538 | 2.480 | 2.485 | -0.057 [-0.174, +0.052] | Tương đương (CI chứa 0) |
| | | 3,000 | 2.426 | 2.333 | 2.337 | **-0.092 [-0.164, -0.021]** | Visual có ích |
| | | 13,000 | 2.250 | 2.209 | 2.181 | -0.040 [-0.098, +0.017] | Tương đương |
| | | 25,000 | 2.262 | 2.256 | 2.154 | -0.006 [-0.070, +0.058] | `Cut` $\approx$ `Full`; `Zero` tệ nhất |
| **`middle_high`** | `[17, 19, 25, 33, 34]` | 400 | 2.508 | 2.413 | 2.457 | **-0.096 [-0.174, -0.016]** | Visual có ích |
| | | 3,000 | 2.407 | 2.342 | 2.315 | **-0.065 [-0.126, -0.003]** | Visual có ích |
| | | 25,000 | 2.209 | 2.229 | 2.183 | +0.020 [-0.046, +0.088] | `Cut` $\ge$ `Full` |
| **`middle_high_2`**| `[17, 28, 30, 32, 34]` | 25,000 | 2.180 | **2.295** | 2.182 | **+0.115 [+0.037, +0.194]** | **`Cut` vượt `Full` có ý nghĩa!** |
| **`shallow_high`** | `[1, 28, 30, 32, 34]` | 3,000 | 2.430 | 2.309 | 2.300 | **-0.121 [-0.187, -0.054]** | **Nhạy cảm với visual nhất** |
| **`balanced`** | `[1, 9, 17, 25, 33]` | 400 | 2.464 | 2.457 | 2.406 | -0.007 [-0.058, +0.052] | Tương đương |

* **Quy luật rút ra**:
  1. Cấu hình có tầng nông (`shallow_high` lấy Layer 1) hoặc tầng giữa (`middle_high` lấy Layer 17) có sự phụ thuộc vào visual lớn nhất ở budget vừa (3K).
  2. Cấu hình chỉ lấy tầng sâu (`high_only`) gần như không phụ thuộc vào visual ở chuỗi dài (25K), vì text hidden states ở các tầng đỉnh đã hấp thụ đủ thông tin từ target.
  3. Điều kiện `Zero` (giữ vị trí nhưng gán giá trị 0) luôn kém nhất trên mọi cấu hình vì gây nhiễu cấu trúc vị trí cho attention.

---

### 2.2. Phân loại Token theo Target Attention tại Layer 18 (50% Depth) — (Báo cáo 03)

Thực nghiệm đo trực tiếp **Visual Attention Mass của Target Model (`Qwen2.5-VL-3B-Instruct`) tại Layer 18** ($m^* = 18$) trên 1.200 tokens:
* Phân vị ngưỡng: $p_{50} = 1.1124$, $p_{75} = 1.5054$, $p_{90} = 1.9700$.

| Phân nhóm định lượng theo Target Attention | Số lượng Token | Pass Rate `Full` (100% Visual) | Pass Rate `Cut` (0% Visual) | Độ chênh lệch $\Delta (\text{Cut} - \text{Full})$ | Khoảng tin cậy 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Low Visual** ($< p_{50}$) | 600 | 53.17% | 55.17% | $+2.00\%$ | $[-1.33\%; +5.33\%]$ |
| **Medium Visual** ($p_{50} - p_{75}$) | 300 | 60.67% | 61.00% | $+0.33\%$ | $[-4.00\%; +4.67\%]$ |
| **HEAVY VISUAL** ($p_{75} - p_{90}$) ⭐ | 180 | 60.56% | **67.78%** | **$\mathbf{+7.22\%}$** | **$[+1.11\%; +13.33\%]$** |
| **ULTRA-HEAVY** ($> p_{90}$) | 120 | 60.83% | **65.83%** | **$\mathbf{+5.00\%}$** | $[-3.33\%; +13.33\%]$ |
| **GỘP TOP 25% HEAVY VISUAL** | 300 | 60.67% | **67.00%** | **$\mathbf{+6.33\%}$** | **$[+1.33\%; +11.00\%]$** |

👉 **Ý nghĩa cốt lõi**: Ngay cả trên các token mà Target Model phải dồn sự chú ý thị giác cao nhất (Heavy/Ultra-Heavy Visual), **nhánh `Cut` visual vẫn đạt tỷ lệ chấp nhận cao hơn nhánh `Full` (+6.33% đến +7.22%)**. Điều này chứng minh text hidden states tại Layer 18 đã mang đầy đủ biểu diễn thị giác ngữ cảnh; việc đưa thêm visual tokens thô vào draft model chỉ gây thêm nhiễu attention.

---

### 2.3. Khảo sát 4 mức Retention trên Draft tuần tự MSD (Báo cáo 04 — 27.340 rounds)

| Mức Visual Retention | Số Round ($N$) | Tỷ lệ Vượt qua $t_0$ | Tỷ lệ Thất bại (Zero-Acc) | Số token accept trung bình / round | $\ge 2$ tokens (%) | $\ge 5$ tokens (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **100% Visual** *(Full Video)* | 1.909 rounds | 82.92% | **17.08%** | 1.56 tokens | 54.32% | **0.00%** |
| **50% Visual** | 1.770 rounds | 88.64% | 11.36% | 1.79 tokens | 63.84% | 0.51% |
| **25% Visual** *(Sweet Spot)* | 1.748 rounds | 91.76% | **8.24%** | **1.95 tokens** | **69.74%** | **2.69%** |
| **0% Visual** *(Full Cut)* | 821 rounds | 98.42% | **1.58%** | **4.11 tokens** | **93.06%** | **63.34%** |

* **Phát hiện Sweet Spot**: Ở mức **25% Visual Retention**, tỷ lệ thất bại (Zero-Acc) giảm hơn một nửa (từ 17.08% xuống 8.24%), trong khi khả năng đề xuất chuỗi sâu $\ge 2$ tokens tăng từ 54.3% lên 69.7%.
* **Sự cuốn theo ở 0% Cut**: Ở 0% Visual, số round bắt đầu bằng Visual-Heavy token giảm mạnh từ 1.909 xuống 821 (giảm 57%) vì mô hình draft bám trọn phân phối ngôn ngữ để ăn các chuỗi dài, đẩy các từ visual vào thân chuỗi ($t_1, t_2$).

---

### 2.4. Nghịch lý Tốc độ trên DSpark 3-layer (Báo cáo 02 — A100)

| Visual Budget | Điều kiện | $\tau_{\text{eff}}$ (tokens/round) | Decode Speedup [95% CI] | E2E Speedup [95% CI] | Trạng thái gia tốc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **3K tokens** | **Full** | 2.072 | **$1.462\times$** $[1.287; 1.664]$ | **$1.409\times$** $[1.259; 1.569]$ | Tăng tốc tốt |
| | **Cut** | 2.007 | **$1.431\times$** $[1.270; 1.610]$ | **$1.387\times$** $[1.249; 1.550]$ | Tăng tốc tốt |
| | **Retained 25%** | 2.066 | **$1.478\times$** $[1.291; 1.677]$ | **$1.424\times$** $[1.273; 1.604]$ | Tăng tốc tốt |
| **25K tokens** | **Full** | 1.883 | **$0.787\times$** $[0.733; 0.843]$ | **$0.857\times$** $[0.819; 0.895]$ | **Chậm hơn AR (< 1.0×)** |
| | **Cut** | 1.829 | **$0.821\times$** $[0.767; 0.882]$ | **$0.883\times$** $[0.845; 0.925]$ | **Chậm hơn AR (< 1.0×)** |
| | **Retained 25%** | 1.902 | **$0.853\times$** $[0.789; 0.914]$ | **$0.902\times$** $[0.859; 0.945]$ | **Chậm hơn AR (< 1.0×)** |

👉 **Cảnh báo sống còn**: Ở chuỗi video dài (25K), dù $\tau$ vẫn giữ vững quanh mức ~1.9 tokens/round, nhưng **Speedup thực tế tụt xuống dưới $1.0\times$** do chi phí nạp/chiếu hàng chục nghìn visual tokens vào KV cache của Drafter quá nặng.

---

## 3. TỔNG HỢP CÁC INSIGHT VÀ GIẢI MÃ CƠ CHẾ

1. **Bác bỏ giả thuyết "Tăng Text Consistency sẽ làm Visual Cut tăng vọt $\tau$ như Sparrow"**:
   * Cú nhảy vọt $\tau$ trong Sparrow/MSD thực chất là **sự hồi phục từ một baseline `Full` bị sụp đổ quá nặng** do mô hình draft 1-layer không chịu nổi 25K visual tokens.
   * DFlash và DSpark dùng multi-layer feature injection nên baseline `Full` vốn dĩ đã chống chịu tốt; do đó không còn "khoảng trống sụp đổ" để Visual Cut tạo ra cú bật tăng vọt $\tau$.
2. **First Rejection Hazard là nút thắt chí tử của Block Diffusion**:
   * Hơn 75% các block draft của DFlash bị từ chối ngay tại **Slot 1 hoặc Slot 2**.
   * Dù text hidden state có nhất quán đến đâu, việc thiếu phản hồi nhân quả (causal feedback) nội bộ giữa các slot tương lai trong cùng một block khiến các slot sau nhanh chóng bị lệch phân phối so với Target.
3. **Nút thắt thực sự là Conditioning (Tiền xử lý & Nạp KV), không phải Drafting**:
   * Một forward sinh block của DFlash chỉ mất **1.14 ms**, nhưng giai đoạn nạp/chiếu visual condition mất tới **27.4 ms**.
   * Muốn đạt speedup thực tế cho Video-LLM, con đường duy nhất là **cắt bỏ hoặc tái sử dụng triệt để đường truyền conditioning** (như cơ chế Visual Cut kết hợp Text-State reuse).
4. **Khoảng trống thực nghiệm hiện tại**:
   * Các bài test hiện tại trên VDC50 chủ yếu phản ánh mô tả chung (dễ bị text priors chi phối). Cần chuyển sang **Temporal tasks có annotation gắt gao (sự kiện ngắn, đếm số lần, thứ tự hành động)** để bộc lộ rõ các failure case về grounding thị giác khi Cut visual.
