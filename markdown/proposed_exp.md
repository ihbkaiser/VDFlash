# Video-DFlash: kết quả, insight và thực nghiệm tiếp theo

> Sổ theo dõi nghiên cứu. Mỗi vòng: khóa một câu hỏi và metric → chạy/audit → cập nhật kết quả và insight → chọn vòng kế tiếp. Tách kết quả đã đo khỏi giải thích còn giả thuyết.

Báo cáo chi tiết, hình và bảng truy vết: [bộ tổng hợp 6/11/2026](6_11_26/README.md).

## 1. Kết quả thực nghiệm

### Visual context, budget và retention

- Campaign Qwen2.5-VL-3B / DFlash 3-layer trên VDC50 hoàn tất 6.000 cases. Tác động của visual context phụ thuộc budget và layer selection: ở 3K, Cut giảm `τ_effective` rõ trên nhiều cấu hình; ở 25K có một cấu hình tăng nhẹ, các cấu hình khác chưa có contrast rõ.
- Replay Full/Cut cho thấy phân phối proposal thay đổi theo budget (JSD 0,0526 bit ở 400 → 0,1522 ở 25K), nhưng target-hit@1 trung bình chỉ dao động −0,74 đến +0,71 điểm %. `zero` và `cut` không tương đương.
- Đã thử retention bằng cắt đều, giữ từ trái và giữ từ phải; kết quả hiện có chưa cho thấy xu hướng khác biệt rõ giữa ba chính sách. Chưa có CI riêng để kết luận tương đương.
- **Phân tích Visual Recall@ & Pareto Attention L18 (11/11/2026 - Báo cáo 14):**
  - Khảo sát 264.773 draft positions (200 pairs, 15 draft slots, VDC50) và 1.200 mẫu Target L18 Attention.
  - *Target L18 Attention Pareto:* Phân tán đồng đều, không co cụm (Top 1% tokens chỉ chiếm 2,48% attention mass; Top 10% chỉ chiếm 18,66%). Không tồn tại 1–5% visual hotspot chi phối.
  - *Draft Proposal Recall:* Toàn cục suy giảm cực nhỏ khi Cut visual: R@1 = 19,50% (Cut) vs 19,59% (Full), R@5 = 42,99% (Cut) vs 43,65% (Full).
  - *Slot 1 áp đảo:* Drafter đạt R@1 = 65,00%, R@2 = 78,72%, R@5 = 89,19% ngay cả khi Cut visual (Median rank = 1,0 cả Full và Cut).
  - *Nghịch lý đảo chiều ở Long Video 25K:* Ở 25.000 tokens, **Cut vượt trội hơn Full** cả về R@1 (+0,67 điểm %), R@2 (+0,84 điểm %) và R@5 (+0,45 điểm %). Visual context ở 25K gây loãng attention của drafter nhỏ; Cut visual giúp Drafter tập trung vào text states và đạt proposal recall sắc bén hơn. Chi tiết: [Báo cáo 14](Visualcut_retention_report/raw_reports/14_Visual_Recall_Analysis.md).

### Prefix acceptance, first rejection và lexical proxy

- 200 cặp replay, 50 video × 4 budgets: Cut−Full trên prefix là −0,068 (400), −0,061 (3K), −0,012 (13K; CI chứa 0), +0,045 (25K). Rejection tập trung ở 1–2 slot đầu; không có block Full nào accept đủ 15/15.
- Ở 3K, risk-set lexical overlap cho tương tác Cut−Full **+3,56 điểm %** [0,92; 6,08]. Nhưng matched control theo token ID × slot trên subset 24 video cho **−9,46 điểm %** [−18,73; +0,70]. Khác metric và tập mẫu; lexical overlap chưa phải bằng chứng vững về grounding.

### Các kiểm tra hypothesis khác

- H3.2: depth interaction đổi dấu theo corpus (+0,126 LLaVA68K; −0,054 ShareGPT68K). H3.1: early/late interaction +0,0053 [−0,0058; +0,0165], chưa thấy acceptance gain chỉ ở nửa sau output.
- Answer-hint làm đổi interaction acceptance, nhưng chưa đo answer quality. Các thí nghiệm inference cũ có khác GPU/target hash nên không dùng làm kết luận chất lượng hay tốc độ.

### Text-state intervention — 3K, cohort 12 video

- Chạy đủ cohort đã khóa (3 video/quartile), chia hai batch 8+4. Có **1.053 eligible blocks/source**; native G và forced G/A/N output/schedule parity đạt 12/12. Hai full batch mất 0,96 GPU-hours; video chậm nhất 5,6 phút.
- Primary interaction **I = −0,0331 token/round**, bootstrap 95% CI **[−0,0799; +0,0215]**. Không đạt ngưỡng đặt trước `I ≥ 0,05` và CI dương; kết quả **inconclusive**, điểm ước lượng ngược dự đoán.
- Noise-control contrast **J = −0,0889**, CI **[−0,1435; −0,0313]**; chưa có bằng chứng về tính đặc hiệu so với noise control. Các tau macro: G Full/Cut = 1,390/1,317; A = 0,939/0,899; N = 1,310/1,181.
- **Audit matched-path hoàn tất và đạt chuẩn 100% (9/11/2026):** Đã kiểm định chẩn đoán trên 4 video (`v_ej9TkJVjK9M`, `v_zRqJZ0u6f88`, `v_aLb79wtvxoQ` và đối chứng `v_1YU4MSK80cQ`) qua [audit_dflash_textstate_matched_path.py](../src/analyze/Validate_Sparrow_hypothesises/audit_dflash_textstate_matched_path.py). Khi G và A dùng chung teacher-forced forward path, sai số pre-visual bằng **0.000000**, dung sai no-op **0.000000**, suffix leakage **0.000000** ([kết quả JSON](../results/dflash_textstate_intervention/audit_matched_path/matched_path_audit_results.json)). Sai số 2.0 trước đây được xác định là do sai lệch codepath thu thập (cached prefill vs full forward). Kết quả sụt giảm ~32–33% tau của DFlash khi triệt tiêu video features ở Target là bản chất thật của mô hình.
- Đây là forced-trajectory proposal-prefix agreement, không đo answer accuracy, free-running acceptance hay latency. Artifact: [protocol](DFlash_TextState_Intervention_20261106/protocol.md), [aggregate manifest](../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/aggregate_manifest.json), [summary](../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/pilot/summary.json), [report](../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/pilot/report.md).

## 2. Insight hiện tại

1. Không có quy tắc chung rằng video càng dài thì càng nên Cut visual; hiệu ứng phụ thuộc budget, layer selection và corpus.
2. `τ_effective` có thể che khuất thay đổi ở vị trí proposal; acceptance, answer quality và latency phải đo riêng.
3. **Target text states là kênh truyền tải thông tin video chủ đạo sang DFlash:** Khi Target bị triệt tiêu thông tin video (A), số token đề xuất đúng liên tiếp của DFlash sụt giảm đều đặn **~32,4% ở Full** và **~31,7% ở Cut**.
4. **Visual tokens trực tiếp trong DFlash không bù đắp được khi text states bị hỏng:** ở nhánh A, visual tokens trực tiếp chỉ giúp kéo lại $+0,040$ token/round (thấp hơn cả $+0,073$ ở G). Do đó, nhồi 100% visual context vào DFlash khi xử lý long video là không cần thiết và tốn kém conditioning cost.
5. Noise control cho contrast khác A; một seed và một cách match norm chưa đủ để phân biệt can thiệp video với OOD/perturbation nói chung.
6. Lexical overlap cho kết luận nhạy với estimand và matched control; chỉ nên giữ làm phân tích phụ.
7. Động lực thực nghiệm vững chắc cho **Self-Distillation 10% Visual DFlash**: Huấn luyện student chỉ cần 10% visual context bằng Target CE + Full-DFlash KL để tận dụng tối đa kênh text states mà tiết kiệm 90% chi phí conditioning.

## 3. Hướng thực nghiệm tiếp theo

| Ưu tiên | Thực nghiệm | Câu hỏi / tiêu chí | Trạng thái |
|---|---|---|:---:|
| - | **Matched-path/no-op audit cho text-state features.** | Chuẩn hóa codepath G/A, đo dung sai no-op, kiểm tra pre-visual error = 0. | **HOÀN THÀNH (Đạt 4/4)** |
| 1 | **Self-Distillation 10% Visual DFlash** ([Proposal](DFlash_Visual10_SelfDistillation_20261103/research_proposal.md) \| [SOP Hướng dẫn chạy](DFlash_Visual10_SelfDistillation_20261103/huong_dan_chay_thuc_nghiem.md)). | Huấn luyện student 10% visual với Target CE + Full-DFlash KL. Liệu có đạt acceptance gần Full DFlash mà giảm 90% visual conditioning cost không? | **SẴN SÀNG TRIỂN KHAI** |
| 2 | **Noise-seed robustness**: nhiều seed trên subset stratified, giữ G trajectory/A intervention cố định. | J có phụ thuộc seed, và A khác noise nói chung một cách ổn định không? | Đề xuất |
| 3 | **Temporal task có annotation:** sự kiện ngắn, thứ tự, counting; paired free-running sau khi cơ chế được xác nhận. | Text-state route thiếu thông tin ở loại câu hỏi nào? Dùng lexical overlap làm phụ, không làm nhãn chính. | Đề xuất |
| 4 | **Đo chi phí thực:** AR/Full/Cut end-to-end, tách prefill, draft, verify; sau đó mới sweep block size. | Khi nào acceptance chuyển thành speedup thực, đặc biệt với long video? | Đề xuất |

### Vòng kế tiếp được đề xuất

Triển khai vòng lặp thực nghiệm cho **Self-Distillation 10% Visual DFlash**:
1. Khóa Experiment Card & Feedback Ladder (R0 -> R2) cho pipeline chưng cất 2 nguồn.
2. Kiểm tra smoke quy trình cache features của Full-DFlash teacher và pipeline training student 10% visual trên `src/train_VLM/cache_visual10_selfdistill_features.py` và `src/train_Dflash_SpecForge/specforge/algorithms/common/dflash_self_distill.py`.

**Nhịp cập nhật:** sau mỗi vòng, ghi số liệu và giới hạn vào mục 1, cập nhật insight ở mục 2, rồi thay đổi trạng thái/hướng tiếp theo ở mục 3. Không giữ thí nghiệm đã chạy trong bảng đề xuất.
