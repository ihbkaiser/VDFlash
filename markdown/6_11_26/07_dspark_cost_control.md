# 07 · DSpark: acceptance và chi phí thực

**Kết quả:** DSpark balanced ở 25K có τ_effective khoảng 1,8–1,9 nhưng speedup dưới 1×. Đây là cảnh báo về chi phí context trên **backend DSpark riêng**, chưa phải timing kết luận cho DFlash.

## Thiết kế và coverage

| Thành phần | Thiết lập |
|---|---|
| Target / draft | Qwen2.5-VL-3B / DSpark 3-layer balanced [1,9,17,25,33] |
| Mẫu đánh giá | 15 video đầu theo shared calibrated manifest order; cohort nền là VDC50 |
| Ma trận | 15 × 4 budgets × 6 conditions =360 cases |
| Timing | Capture và timing chạy riêng; target nhận full video; A100 |
| Audit local | 360 valid, không missing/invalid; exact target output match 42/360 (11,7%) |
| Thống kê | Bootstrap 2.000 lần theo video, seed 42 trong condition/contrast audit; speedup lấy bảng timing nguồn |

400/3K dùng subset từ run 50; 13K/25K chạy giới hạn 15 theo provenance nguồn. `cohort_size=50` trong audit là cohort nền, không biến study thành N=50.

## Kết quả

![Hình 15. τ_effective, decode speedup và E2E speedup của DSpark; dải màu là CI đã báo cho speedup. Số timing trích bảng lịch sử; audit/condition statistics có bản local.](figures/15_dspark_cost.png)

*Hình 15. τ_effective, decode speedup và E2E speedup của DSpark; dải màu là CI đã báo cho speedup. Số timing trích bảng lịch sử; audit/condition statistics có bản local.*

| Budget | Condition | τ_eff | Decode speedup [95% CI] | E2E speedup [95% CI] |
| --- | --- | --- | --- | --- |
| 3000 | Full | 2.072 | 1.462 [1.287; 1.664] | 1.409 [1.259; 1.569] |
| 3000 | Cut | 2.007 | 1.431 [1.270; 1.610] | 1.387 [1.249; 1.550] |
| 3000 | Retained 25% | 2.066 | 1.478 [1.291; 1.677] | 1.424 [1.273; 1.604] |
| 25000 | Full | 1.883 | 0.787 [0.733; 0.843] | 0.857 [0.819; 0.895] |
| 25000 | Cut | 1.829 | 0.821 [0.767; 0.882] | 0.883 [0.845; 0.925] |
| 25000 | Retained 25% | 1.902 | 0.853 [0.789; 0.914] | 0.902 [0.859; 0.945] |

Tại 25K, retained25−Full = +0,019 token/round [−0,021; +0,064]; CI chứa 0. Mean retention tốt hơn chưa chứng minh policy thắng. Full/Cut/retained25 đều có CI E2E dưới 1× ở budget này.

## Insight và giới hạn

- Chi phí draft và target verify tăng theo context có thể làm mất speedup dù mỗi vòng phát hơn một token. Acceptance chỉ là một thành phần của bài toán latency.
- Exact output match chỉ 11,7% toàn campaign. Các timings này chưa bảo đảm so sánh cùng output; không diễn giải thành speedup lossless-equivalent.
- Valid attention/timing pair chỉ xác nhận repeatability giữa capture và timing, không tự xác nhận losslessness so với AR target.
- Khác checkpoint/backend/GPU với DFlash campaigns; không dùng DSpark làm đối chứng causal cho multi-layer fusion của DFlash.

## Truy vết

- [DSpark checkpoint manifest](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/checkpoint_manifest.json).
- [Audit và coverage](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/audit_and_summary.json); [condition statistics](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/condition_statistics.csv); [paired contrasts](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/paired_contrasts.csv).
- [Báo cáo nguồn, bảng Mean accepted length và speedup](../../markdown/Baocao_DSpark_20261103/bao_cao_dspark_visualcut.md); [bảng timing trích có CI](../../results/report_bundle_20261106/tables/dspark_speed_archived.csv).
- Bước cần làm cho DFlash: timing AR/Full/Cut paired cùng output contract, tách prefill/draft/verify và báo lossless rate trước khi tuyên bố tăng tốc.


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
