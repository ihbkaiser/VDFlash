# 04 · Khống chế token identity × slot cho lexical overlap

**Kết quả:** matched analysis chưa hỗ trợ interaction dương. Điểm ước lượng −9,46 pp, interval 95% [−18,73; +0,70] chứa 0. **Mức bằng chứng: exploratory reanalysis local**, không chạy model mới.

## Câu hỏi và thiết kế

Có phải lexical-overlap positions chịu ảnh hưởng của Cut nhiều hơn ngay cả khi khống chế **target token ID và draft slot**?

| Yếu tố | Cấu hình |
|---|---|
| Dữ liệu | 24/50 video, 6/quartile thời lượng; budget 3K, slots 1–4 |
| Mẫu | 2.127 complete blocks, 8.508 positions; 953 overlap / 7.555 other |
| Stratum | Target token ID × slot |
| Common support | ≥3 positions và ≥2 video cho mỗi label trong từng stratum |
| Giữ lại | 20 strata, 188 positions: 99 overlap / 89 other |
| Metric chính | Macro qua strata của (Cut−Full mismatch)_overlap − (Cut−Full mismatch)_other |
| Uncertainty | Stratified video-cluster Bayesian bootstrap; Exp(1) weights chuẩn hóa trong quartile; 2.000 draws, seed 20261105 |

Subset ban đầu 16 video chỉ có 2 strata đủ hỗ trợ; chọn 24 sau support preflight, dựa trên token/label counts theo protocol, không dựa trên mismatch outcomes. Phương pháp uncertainty được sửa sau audit phát hiện case bootstrap làm thay đổi số strata từng draw; lịch sử này được giữ trong protocol.

## Kết quả

![Hình 8. Matching giữ khoảng 2,21% positions ban đầu. Biểu đồ bên phải là hai mean mô tả trên common-support strata; chưa có CI riêng cho hai mean này.](figures/08_matched_support.png)

*Hình 8. Matching giữ khoảng 2,21% positions ban đầu. Biểu đồ bên phải là hai mean mô tả trên common-support strata; chưa có CI riêng cho hai mean này.*

| Đại lượng | Giá trị |
| --- | --- |
| Δ mismatch overlap | +3,25 pp |
| Δ mismatch other | +12,71 pp |
| Interaction primary | -9.46 [-18.73; +0.70] pp |
| Bootstrap support | 20 strata được giữ trong mọi draw primary |
| Case-resampling diagnostic | -9.46 [-22.02; +5.85] pp; 11–20 strata/draw |

![Hình 9. Primary interval và diagnostic interval đều chứa 0. Diagnostic không bảo toàn fixed-support estimand nên không thay thế primary uncertainty.](figures/09_matched_interaction.png)

*Hình 9. Primary interval và diagnostic interval đều chứa 0. Diagnostic không bảo toàn fixed-support estimand nên không thay thế primary uncertainty.*

## Insight

- Tín hiệu lexical dương của risk-set analysis chưa bền trước control về token identity/slot. Tuy vậy, không gọi đây là đảo chiều causal: báo cáo 03 dùng hazard có risk set trên 50 video, báo cáo này dùng mismatch mọi vị trí ở subset 24 và common support hẹp.
- Matching bỏ phần lớn positions, đặc biệt nhóm other: 89/7.555 ≈1,18%; overlap giữ 99/953 ≈10,39%. Kết luận chỉ áp dụng nhóm token/slot có hỗ trợ, không đại diện mọi content words hoặc toàn bộ video.
- Macro ngang trọng số các strata khác với pooled mismatch theo số positions. Thêm observations không đồng nghĩa thêm independent strata.
- Giữ lexical labels làm phân tích phụ. Để kiểm tra grounding cần annotation về evidence/temporal dependency và control độ khó context.

## Truy vết

- [Protocol và analysis revision](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/token_control_subset24/protocol.json); [sample selection](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/token_control_subset24/sample_selection.json); [support preflight](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/token_control_subset24/sample_size_preflight.json).
- [Primary summary](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/token_control_subset24/token_control_summary.json); [matched positions](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/token_control_subset24/token_control_positions.csv); [diagnostic](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/token_control_subset24/case_resampling_diagnostic.json).
- [Analyzer](../../src/analyze/Validate_Sparrow_hypothesises/analyze_dflash_token_control_subset.py). Interval là bootstrap uncertainty của estimand đã mô tả; chưa phải chứng cứ causal hoặc một phép replication độc lập.


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
