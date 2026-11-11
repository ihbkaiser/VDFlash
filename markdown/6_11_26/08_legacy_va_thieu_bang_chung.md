# 08 · Legacy, retention policies và các khoảng trống bằng chứng

**Mục tiêu:** giữ trace các nghiên cứu trước mà không đưa các diễn giải mạnh hoặc số liệu thiếu audit vào kết luận cơ chế hiện hành. Các bảng legacy dưới trích từ báo cáo; chưa tái kiểm tra raw runs.

## Text pruning sau khi bỏ visual — 23/10

| Thiết kế đã báo cáo | Phạm vi |
|---|---|
| Dữ liệu / output | VDC50, max 128 tokens, visual retention 0%; text suffix retention 100/75/50/25/0% |
| Backends | DFlash/Qwen2.5-VL-3B trên A4000; MSD/Qwen2-VL-7B trên RTX3090 |
| Quy mô báo cáo | 500 runs =50 video ×5 mức ×2 backends |
| Evidence | Mean đã làm tròn; không có paired CI local |

![Hình 16. Legacy text-retention means và strict target output match. Khác target/backend/GPU; không đọc độ dốc giữa hai mô hình như một causal contrast.](figures/16_legacy_text_retention.png)

*Hình 16. Legacy text-retention means và strict target output match. Khác target/backend/GPU; không đọc độ dốc giữa hai mô hình như một causal contrast.*

| Text giữ lại | DFlash τ_eff | DFlash exact match | MSD τ_eff | MSD exact match |
|---|---:|---:|---:|---:|
| 100% | 2,17 | 22% | 3,63 | 88% |
| 75% | 2,17 | 22% | 3,29 | 90% |
| 50% | 2,16 | 22% | 2,69 | 90% |
| 25% | 2,13 | 22% | 2,64 | 90% |
| 0% / tối thiểu 1 token | 2,11 | 22% | 2,17 | 90% |

DFlash mean khá phẳng trong setup này, nhưng lossless rate thấp và thiếu CI. “Một token text prompt còn lại” chưa xác nhận toàn bộ generated hidden context/KV cũng chỉ còn một token. Chưa có phép đo peak memory xác nhận tuyên bố giải phóng 99,97% KV; không dùng “single-token KV deployment” làm kết luận đã đạt.

## Các kết quả lịch sử khác

| Study | Quan sát đã lưu | Có thể dùng để kết luận gì? |
|---|---|---|
| DFlash MVBench ban đầu | 500 mẫu; accuracy khoảng 58,0–58,2% qua ablations | Mô tả setup đó; chưa xác nhận tương đương quality tổng quát |
| DFlash VDC50 ban đầu | Target exact output match 2/50 ở Full/Zero, 6/50 ở Cut; Cut đổi 47/50 output hashes | Cần làm rõ output contract; khác GPU/target hash làm nhiễu so sánh |
| Checkpoints 6e/20e | MVBench outputs không đổi; VDC target hash đổi 48/50 giữa runtime | Chưa tách được tác động epoch khỏi runtime/target |
| MSD/H1b/H2.1 | Attention/acceptance thay đổi dưới visual ablations | Context cho hypothesis; khác target/architecture, chưa chứng minh cơ chế DFlash |
| Retention đều/trái/phải | Người nghiên cứu đã chạy, chưa thấy xu hướng khác biệt rõ | Ghi nhận định tính; chưa có paired numerical artifact trong bundle để kiểm định |

## Khoảng trống cần giữ khi viết insight

- Thiếu training-state reference nên chưa xác nhận hidden-state OOD; match norm không đo được mọi dạng OOD.
- Target-output mismatch làm yếu diễn giải lossless speedup; caption similarity không thay thế factual correctness.
- Các hình external trong ghi chú Thought là đầu mối literature, không phải thí nghiệm của workspace. Không tái dùng số external như kết quả local.
- Để hoàn thiện directional retention report cần raw case IDs, policy implementation, budgets/checkpoint, per-video outcomes và paired contrasts/CI. Chưa có đủ dữ liệu để dựng thêm hình hoặc suy equivalence.

## Truy vết

- [Text pruning 23/10 — §II, Bảng 1–2](../../markdown/23_10_26/REPORT_TEXT_CUT_ZERO_VISUAL_VDC.md); [bảng trích](../../results/report_bundle_20261106/tables/legacy_text_retention.csv).
- [DFlash inference ban đầu](../../markdown/Baocao_26_8_26/Dflash_cut_report/REPORT.md); [epoch comparison](../../markdown/Baocao_26_8_26/dflash20e_vs_6e_report_2026-08-26/REPORT.md).
- [MSD/H1b/H2.1](../../markdown/bao_cao_h1b_h21_full_20260909.md); [sổ kết quả hiện hành](../../markdown/proposed_exp.md).


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
