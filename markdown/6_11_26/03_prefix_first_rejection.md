# 03 · Prefix acceptance, first rejection và lexical risk set

**Kết quả:** sai proposal thường chặn block ngay ở slot 1–2. Cut giảm prefix tại 400/3K, chưa khác rõ ở 13K và tăng nhẹ tại 25K. Liên hệ với lexical overlap chưa chứng minh nhu cầu grounding. **Mức bằng chứng: reanalysis local**, dùng lại 200 pairs của báo cáo 02.

## Phương pháp

| Đại lượng | Định nghĩa / phạm vi |
|---|---|
| Prefix L | Số top-1 proposals đúng liên tiếp trước lỗi đầu; 0–15; không cộng correction |
| Eligible blocks | 17.089 complete blocks đủ 15 slots, có anchor Full kế tiếp |
| Audit anchors | 17.998 nonterminal Full rounds; tái dựng τ_proposal khớp 200/200 pairs, max error 0 |
| Risk-set hazard | P(reject slot j \| các slot trước trong cùng condition đều đúng); macro slots 1–4 |
| Nhãn overlap | Token trùng content term trong reference caption; không phải nhãn ngữ nghĩa mù |
| Thống kê | Paired video bootstrap 2.000 lần, seed 20261105; fixed Full anchors cho Cut |

## Prefix và first rejection

![Hình 5. Prefix trung bình của complete blocks và paired Cut−Full CI. Absolute means ở trái không có error bars; CI contrast ở phải lấy từ artifact.](figures/05_prefix.png)

*Hình 5. Prefix trung bình của complete blocks và paired Cut−Full CI. Absolute means ở trái không có error bars; CI contrast ở phải lấy từ artifact.*

| Budget | Eligible blocks | Full prefix | Cut replay prefix | Cut−Full [95% CI] |
| --- | --- | --- | --- | --- |
| 400 | 3320 | 1.574 | 1.506 | -0.068 [-0.095; -0.039] |
| 3000 | 4408 | 1.376 | 1.315 | -0.061 [-0.085; -0.037] |
| 13000 | 4698 | 1.273 | 1.260 | -0.012 [-0.058; +0.048] |
| 25000 | 4663 | 1.218 | 1.263 | +0.045 [+0.011; +0.081] |

![Hình 6. Full prefix survival và tỷ lệ block có first rejection ở từng slot; hiển thị slots 1–6. Đây là mean theo video, không phải 6 token độc lập.](figures/06_first_rejection.png)

*Hình 6. Full prefix survival và tỷ lệ block có first rejection ở từng slot; hiển thị slots 1–6. Đây là mean theo video, không phải 6 token độc lập.*

Trong 17.998 nonterminal Full rounds, số lượt accept trọn 15/15 là 0. Target thêm token correction ở mỗi lượt này; cách gọi chung token +1 là “bonus sau block đã accept hết” không phù hợp với dữ liệu này. Không ngoại suy số 0 này sang checkpoint hoặc block size khác.

## Lexical risk-set analysis

![Hình 7. Thay đổi hazard theo nhóm overlap/other và interaction (Overlap−Other), đơn vị điểm phần trăm. Error bars của interaction là CI cấp video.](figures/07_lexical_riskset.png)

*Hình 7. Thay đổi hazard theo nhóm overlap/other và interaction (Overlap−Other), đơn vị điểm phần trăm. Error bars của interaction là CI cấp video.*

| Budget | Overlap Full→Cut (%) | Other Full→Cut (%) | Δ overlap, pp | Δ other, pp | Interaction [95% CI], pp |
| --- | --- | --- | --- | --- | --- |
| 400 | 48.8→55.1 | 42.4→43.2 | 6.33 | 0.81 | +5.52 [+2.44; +9.10] |
| 3000 | 52.4→56.5 | 44.5→45.0 | 4.09 | 0.53 | +3.56 [+0.92; +6.08] |
| 13000 | 55.3→58.4 | 48.6→48.6 | 3.16 | 0.05 | +3.11 [-0.04; +6.56] |
| 25000 | 58.4→60.0 | 49.8→46.7 | 1.55 | -3.17 | +4.72 [+0.62; +8.88] |

## Insight và giới hạn

- Ở 3K, hazard tăng +4,09 pp ở overlap và +0,53 pp ở other; interaction +3,56 pp có CI dương. Tuy vậy, từ nội dung có thể hiếm/khó hơn và chưa được khống chế token ID.
- Ở 25K, interaction +4,72 pp còn đến từ hazard **other giảm −3,17 pp**. Không đọc interaction dương thành “Cut gây hại rõ cho overlap”: Δ overlap riêng ở 25K có CI chứa 0.
- Risk set được quyết định bởi các slot trước và khác giữa Full/Cut. Đây là một estimand điều kiện; chưa cô lập tác động semantic. Nhãn reference overlap bỏ sót synonym, chi tiết đúng ngoài reference và phụ thuộc caption.
- Complete-block means khác mean trên toàn bộ rounds, kể cả truncated tails. Cut theo lịch Full chưa phải free-running Cut.

## Truy vết

- [Prefix summary/audit/slot metrics](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/block_prefix_analysis/summary.json) và [CSV](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/block_prefix_analysis/summary.csv).
- [Risk-set CSV](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/grounding_proxy_analysis/grounding_proxy_riskset.csv); [label definition và counts](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/grounding_proxy_analysis/grounding_proxy_summary.json).
- [Prefix analyzer](../../src/analyze/Validate_Sparrow_hypothesises/analyze_dflash_block_prefix_replay.py); [risk-set analyzer](../../src/analyze/Validate_Sparrow_hypothesises/analyze_dflash_first_rejection_grounding.py).
- Follow-up đã làm: matching token identity × slot ở [báo cáo 04](04_token_matched_control.md).


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
