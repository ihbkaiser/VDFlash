# Báo Cáo 14: Phân Tích Thực Nghiệm Visual Recall@ — Attention Mass Pareto Coverage & Draft Proposal Recall@K

## 1. Mục Tiêu & Cơ Sở Thực Nghiệm

Trong các báo cáo trước ([Báo cáo 01](file:///home/tuantb/VDFlash/markdown/Visualcut_retention_report/raw_reports/01_DFlash_5Configs_VisualCut_Retention_VDC50.md), [Báo cáo 07](file:///home/tuantb/VDFlash/markdown/Visualcut_retention_report/raw_reports/07_DFlash_VisualCut_Distribution_Replay.md)), chúng ta đã chứng kiến hiện tượng **Visual Cut Retention**: khi cắt bỏ 100% visual context tokens khỏi Drafter, tỷ lệ chấp nhận hiệu dụng $\tau_{\text{eff}}$ và các chỉ số chấp nhận token hầu như không bị suy giảm (thậm chí tăng nhẹ ở 25K tokens).

Để giải mã bản chất cơ chế của hiện tượng này, thực nghiệm này triển khai khảo sát hai câu hỏi cốt lõi:
1. **Phân phối Attention thị giác của Teacher (Target L18) có tính chất gì?** Liệu có một tập hợp nhỏ "visual hotspot" chiếm hầu hết attention mass (Pareto distribution), hay attention trải đều?
2. **Khả năng bao phủ của Drafter (Draft Proposal Recall@K) biến động ra sao giữa Full và Cut?** Đặc biệt là ở các slot then chốt (Slot 1, Slot 2) và theo các độ dài ngữ cảnh thị giác từ 400 đến 25.000 tokens?

---

## 2. Dữ Liệu & Phương Pháp Đo Lường

### 2.1. Nguồn Dữ Liệu
- **Target Attention Layer 18 Attribution:** Trích xuất từ 1.200 mẫu attention attribution đo lường trên target model Qwen2.5-VL-3B tại layer 18 (`markdown/Visualcut_retention_report/raw_reports/03_Target_Visual_Attention_L18_Attribution_Data.json`).
- **Distribution Replay Pairs:** 200 tệp replay `.jsonl.gz` chứa tổng cộng **264.773 draft positions** độc lập từ tập cohort cân bằng VDC-50 (`results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/full_run/pairs/`).

### 2.2. Định Nghĩa Chỉ Số
- **Target Attention Pareto Coverage ($P_c$):** Tỷ lệ tổng trọng số attention của Teacher dồn vào top $p\%$ visual tokens có attention cao nhất.
- **Draft Proposal Recall@K ($R@K$):** Tỷ lệ các vị trí dự đoán mà token chuẩn của Teacher (target greedy token) nằm trong top $K$ xác suất cao nhất của Drafter:
  $$\text{Recall@K} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{rank}_i \le K)$$
  trong đó $\text{rank}_i$ là thứ hạng 1-indexed của target token trong phân phối logits của Drafter.

---

## 3. Kết Quả Định Lượng

### 3.1. Attention Mass Pareto Coverage (Target Layer 18)

Đo lường trên 1.200 mẫu context thị giác:

| Tỷ lệ Visual Tokens ($p\%$) | Tỷ lệ Target Attention Mass Tích Lũy | Nhận xét Phân Phối |
| :--- | :--- | :--- |
| **Top 1.0%** | **2.48%** | Không có hiện tượng co cụm cực đoan |
| **Top 5.0%** | **10.17%** | Tương quan gần như tuyến tính |
| **Top 10.0%** | **18.66%** | Không tồn tại "hotspot" chi phối |
| **Top 20.0%** | **33.43%** | 1/5 tokens nhận 1/3 attention |
| **Top 25.0%** | **39.87%** | Phân tán rộng khắp context |
| **Top 33.0%** | **49.28%** | 1/3 tokens nhận ~50% mass |
| **Top 50.0%** | **66.52%** | Nửa số token nhận 2/3 mass |
| **Top 75.0%** | **86.44%** | Phân phối tương đối đồng đều |
| **100.0%** | **100.00%** | Toàn bộ context thị giác |

> [!NOTE]
> **Kết luận về Attention Mass:** Phân phối attention thị giác của Teacher ở Layer 18 mang tính **phân tán (diffuse)**, không tuân theo quy luật Pareto lũy thừa hẹp (80/20 thông thường). Do đó, việc cố gắng chọn lọc một tập con 1-5% visual tokens "quan trọng" để giữ lại ở Drafter sẽ không thể gom được phần lớn attention mass, mà lại tốn chi phí KV routing phức tạp. Cắt bỏ toàn bộ (Visual Cut) là giải pháp dứt khoát và hiệu quả nhất.

---

### 3.2. So Sánh Tổng Thể Draft Proposal Recall@K (Full vs Cut)

Đo lường trên toàn bộ **264.773 vị trí dự đoán**:

| Chỉ số | Full Context | Visual Cut | Chênh Lệch ($\Delta = \text{Cut} - \text{Full}$) |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **19.59%** | **19.50%** | **-0.09%** |
| **Recall@2** | **28.99%** | **28.75%** | **-0.24%** |
| **Recall@3** | **35.32%** | **34.87%** | **-0.45%** |
| **Recall@5** | **43.65%** | **42.99%** | **-0.66%** |
| **Recall@10** | **54.97%** | **53.96%** | **-1.01%** |

> [!IMPORTANT]
> **Độ suy giảm tổng thể cực nhỏ:** Cắt bỏ hoàn toàn visual context chỉ làm suy giảm Recall@1 đúng **0.09%** (từ 19.59% xuống 19.50%) và Recall@5 chỉ **0.66%**. Điều này khẳng định Drafter không dựa vào visual tokens thô để sinh ra proposal tokens.

---

### 3.3. Phân Rã Theo Draft Slot (Vị Trí Trong Block)

Độ dài block DFlash gồm 15 slots dự đoán song song. Bảng dưới đây khảo sát các slot đầu tiên — nơi quyết định phần lớn acceptance rate trong speculative decoding:

| Slot | Full R@1 | Cut R@1 | $\Delta$ R@1 | Full R@2 | Cut R@2 | $\Delta$ R@2 | Full R@5 | Cut R@5 | $\Delta$ R@5 | Median Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Slot 1** | **66.16%** | **65.00%** | -1.16% | **79.84%** | **78.72%** | -1.12% | **90.12%** | **89.19%** | -0.93% | **1.0** (cả hai) |
| **Slot 2** | **46.91%** | **45.41%** | -1.50% | **61.54%** | **59.67%** | -1.87% | **76.39%** | **74.41%** | -1.98% | **2.0** (cả hai) |
| **Slot 3** | **34.01%** | **33.41%** | -0.60% | **47.59%** | **46.50%** | -1.09% | **64.13%** | **62.36%** | -1.77% | **3.0** (cả hai) |
| **Slot 4** | **25.12%** | **24.92%** | -0.20% | **36.77%** | **35.86%** | -0.91% | **53.63%** | **52.29%** | -1.34% | **5.0** (cả hai) |
| **Slot 5** | **18.93%** | **18.93%** | **+0.00%** | **29.64%** | **29.33%** | -0.31% | **46.28%** | **45.15%** | -1.13% | **7.0** (cả hai) |

#### Nhận xét quan trọng:
1. **Slot 1 có độ chính xác áp đảo:** Ở Slot 1, Drafter đạt **65.00% Recall@1**, **78.72% Recall@2**, và **89.19% Recall@5**. Nghĩa là trong 9/10 trường hợp, token đúng của Teacher nằm trong top 5 ứng viên của Drafter.
2. **Median Rank bất biến:** Tại Slot 1, Median Rank là 1.0; Slot 2 là 2.0; Slot 3 là 3.0 cho cả Full và Cut. Không có hiện tượng phân phối thứ hạng bị trượt (rank displacement) khi bỏ visual tokens.

---

### 3.4. Bóc Tách Theo Budget Ngữ Cảnh: Phát Hiện Đột Phá Ở 25K Tokens

| Budget Visual Tokens | Full R@1 | Cut R@1 | $\Delta$ R@1 | Full R@2 | Cut R@2 | $\Delta$ R@2 | Full R@5 | Cut R@5 | $\Delta$ R@5 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **400 tokens** | 21.44% | 20.66% | -0.78% | 30.94% | 29.98% | -0.96% | 46.08% | 44.54% | -1.54% |
| **3.000 tokens** | 19.92% | 19.37% | -0.55% | 29.65% | 28.94% | -0.71% | 44.68% | 43.46% | -1.22% |
| **13.000 tokens** | 18.96% | 19.03% | **+0.07%** | 28.33% | 27.96% | -0.37% | 42.57% | 41.94% | -0.63% |
| **25.000 tokens** | **18.58%** | **19.25%** | **+0.67%** | **27.63%** | **28.47%** | **+0.84%** | **42.03%** | **42.48%** | **+0.45%** |

> [!TIP]
> **Hiện Tượng Đảo Chiều Ở Long Video (25K):**
> Ở chuỗi cực dài 25.000 tokens:
> - **Cut R@1 (19.25%) cao hơn Full R@1 (18.58%) tới +0.67%**.
> - **Cut R@2 (28.47%) cao hơn Full R@2 (27.63%) tới +0.84%**.
> - **Cut R@5 (42.48%) cao hơn Full R@5 (42.03%) tới +0.45%**.
>
> **Giải thích cơ chế:** Với drafter nhỏ gọn (3 transformer layers), sự hiện diện của 25.000 tokens thị giác gây loãng ma trận attention và tạo nhiễu phân phối xác suất. Khi thực hiện **Visual Cut**, Drafter chỉ còn tập trung vào văn bản tiền đề và các biểu diễn đặc trưng đa tầng do Teacher truyền sang qua cross-attention / feature injection, giúp chất lượng dự đoán sắc bén hơn.

---

## 4. Tổng Kết & Ý Nghĩa Thiết Kế

1. **Khẳng định tính đúng đắn của Visual Cut:** Việc cắt bỏ visual tokens không làm mất khả năng dự đoán của drafter vì thông tin thị giác cốt lõi đã được Teacher mã hóa và truyền sang qua target hidden states.
2. **Tiềm năng của Tree/Multi-Candidate Drafting:** Vì Recall@2 ở Slot 1 đạt ~79% và Recall@5 đạt ~89%, nếu mở rộng Drafter từ speculative decoding dạng chuỗi (chain) sang dạng cây (tree-based proposal như Medusa hay EAGLE-2), tỷ lệ chấp nhận ở các slot đầu có thể tiệm cận 85-90%.
3. **Mã nguồn và dữ liệu tái hiện:**
   - Script phân tích: [`src/analyze/Validate_Sparrow_hypothesises/analyze_visual_recall.py`](file:///home/tuantb/VDFlash/src/analyze/Validate_Sparrow_hypothesises/analyze_visual_recall.py)
   - Unit tests: [`tests/test_visual_recall.py`](file:///home/tuantb/VDFlash/tests/test_visual_recall.py)
   - File kết quả JSON chi tiết: [`results/visual_recall_analysis/visual_recall_summary.json`](file:///home/tuantb/VDFlash/results/visual_recall_analysis/visual_recall_summary.json)
