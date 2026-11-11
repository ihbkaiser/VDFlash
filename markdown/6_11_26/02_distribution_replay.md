# 02 · Full/Cut thay đổi phân phối proposal đến đâu?

**Kết quả:** Cut đổi nhiều argmax/probability hơn khi visual budget tăng, trong khi thay đổi hit@1 trung bình nhỏ hơn và đổi dấu theo budget. **Mức bằng chứng: artifact local**, 200 cặp đã có audit; đây là fixed-trajectory replay.

## Câu hỏi và thiết kế

| Thành phần | Cấu hình |
|---|---|
| Target / draft | Qwen2.5-VL-3B / DFlash 3-layer `h32-depth3`, layers [1,9,17,25,33] |
| Cohort / coverage | 50 video × 4 budgets = 200 Full/Cut pairs; 264.773 proposal positions |
| Được giữ cố định | Target trajectory, anchor, block boundary, mask layout và token positions |
| Can thiệp | Bỏ visual context rows của draft; target vẫn nhận full video |
| Thống kê | Mean theo video trước, paired bootstrap 2.000 lần, seed 20261102; A100 |

JSD full vocabulary được tính khi logits còn trong RAM rồi lưu scalar. Top-100 đã lưu không đủ tái dựng KL full vocabulary chính xác. Đây không phải một campaign độc lập với các reanalysis ở báo cáo 03–04.

## Kết quả

![Hình 3. Ba đại lượng khác nhau: JSD, tỷ lệ top-1 đổi và thay đổi hit@1. Error bars lấy nguyên CI bootstrap cấp video từ artifact.](figures/03_distribution.png)

*Hình 3. Ba đại lượng khác nhau: JSD, tỷ lệ top-1 đổi và thay đổi hit@1. Error bars lấy nguyên CI bootstrap cấp video từ artifact.*

| Budget | JSD [95% CI], bit | Top-1 đổi (%) | Full hit@1 (%) | Cut hit@1 (%) | Δ hit@1 [95% CI], pp |
| --- | --- | --- | --- | --- | --- |
| 400 | 0.0526 [0.0448; 0.0619] | 24.0 | 21.97 | 21.23 | -0.74 [-1.05; -0.45] |
| 3000 | 0.0766 [0.0713; 0.0819] | 33.7 | 19.73 | 19.24 | -0.49 [-0.71; -0.27] |
| 13000 | 0.1187 [0.1121; 0.1254] | 43.0 | 18.81 | 18.87 | +0.06 [-0.30; +0.44] |
| 25000 | 0.1522 [0.1451; 0.1590] | 49.6 | 18.31 | 19.02 | +0.71 [+0.36; +1.06] |

![Hình 4. Phân rã vị trí thành both-hit, Full-only, Cut-only và neither. Tỷ lệ được macro theo video; “hit” là argmax khớp target token, chưa phải accepted prefix.](figures/04_hit_decomposition.png)

*Hình 4. Phân rã vị trí thành both-hit, Full-only, Cut-only và neither. Tỷ lệ được macro theo video; “hit” là argmax khớp target token, chưa phải accepted prefix.*

Phân tích điều kiện trên những vị trí **Full đã hit** cho:

| Budget | P(Cut hit \| Full hit), mean (%) | JSD tại Full-hit, bit |
| --- | --- | --- |
| 400 | 87.0 | 0.0409 |
| 3000 | 81.7 | 0.0598 |
| 13000 | 77.9 | 0.0907 |
| 25000 | 74.8 | 0.1122 |

## Phân tích và insight

- Hai phân phối có thể lệch đáng kể nhưng cả hai vẫn đoán sai target tại nhiều vị trí. Số argmax đổi không bằng số accept bị mất; Full-only và Cut-only có thể bù nhau khi tính mean hit.
- Cut giữ lại phần lớn Full-hit positions, nhưng xác suất điều kiện này giảm theo budget. Đây là một estimand đã condition trên Full outcome, không phải tỷ lệ acceptance tổng thể hay hiệu quả policy.
- Budget 3K cho Δ hit@1 −0,49 pp, trong khi top-1 đổi khoảng 33,7%. Trung bình hit ít đổi không có nghĩa visual không ảnh hưởng proposal.

## Audit và giới hạn

Audit ghi 200/200 status ok, alignment/target-token positions đúng, không skip. Tuy vậy source hashes chưa định danh đầy đủ loaded bytecode; có decoder H.264 warnings. Kết quả tái tính khớp summary nhưng chưa giải quyết provenance runtime này. Không suy ra chất lượng caption, free-running acceptance, memory gain hoặc latency từ các logits metrics.

## Truy vết

- [Run config / checkpoint SHA / target revision](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/full_run/run_config.json); [runtime model load](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/full_run/model_load.json).
- [Summary CSV đã tái xác minh](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/analysis_final_verification/summary.csv); [sample means](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/analysis_final_verification/sample_means.csv); [audit gốc](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/artifact_audit.json).
- [Hit decomposition](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/accepted_subset_analysis/summary.csv); [Full-hit conditional analysis](../../results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/accepted_by_full_analysis/summary.csv).
- [Analyzer](../../src/analyze/Validate_Sparrow_hypothesises/analyze_dflash_distribution_replay.py); [báo cáo nguồn](../../markdown/DFlash_VisualCut_Distribution_20261102/final_report.md).
- Bước đã làm tiếp: thay position-hit bằng contiguous prefix và first rejection ở [báo cáo 03](03_prefix_first_rejection.md).


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
