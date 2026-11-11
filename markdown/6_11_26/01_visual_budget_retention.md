# 01 · Visual budget, layer selection và retention

**Kết quả:** hiệu ứng Cut phụ thuộc budget và checkpoint; chưa có quy tắc “context dài thì nên Cut”. **Mức bằng chứng: báo cáo lịch sử** — các số dưới được trích lại từ bảng đã làm tròn; raw campaign tương ứng chưa có trong checkout này.

## Câu hỏi và thiết kế

| Thành phần | Thiết kế đã báo cáo |
|---|---|
| Target / draft | Qwen2.5-VL-3B / DFlash 3-layer; 5 checkpoints với layer selection khác nhau |
| Dữ liệu | VDC50; cùng 50 video qua 4 budgets |
| Ma trận | 5 configs × 4 budgets × 6 conditions × 50 = 6.000 cases |
| Can thiệp | Full, Zero, Cut, retained 75/50/25%; target giữ video đầy đủ |
| Metric / thống kê | τ_effective; paired video bootstrap 2.000 lần theo báo cáo nguồn; H200 |

| Config | Target layers được ghép |
|---|---|
| high_only | [26,28,30,32,34] |
| middle_high | [17,19,25,33,34] |
| middle_high_2 | [17,28,30,32,34] |
| shallow_high | [1,28,30,32,34] |
| balanced | [1,9,17,25,33] |

Budget 400/3K/13K/25K là **ngân sách visual tokens**. Sweep này đồng thời đổi preprocessing; chưa tách chiều dài context khỏi độ chi tiết hình ảnh hoặc độ khó temporal.

## Kết quả

![Hình 1. Cut−Full và Zero−Full qua 5 checkpoints; dấu * chỉ CI 95% chưa hiệu chỉnh chứa hoàn toàn một phía của 0. Số liệu trích báo cáo.](figures/01_campaign_contrasts.png)

*Hình 1. Cut−Full và Zero−Full qua 5 checkpoints; dấu * chỉ CI 95% chưa hiệu chỉnh chứa hoàn toàn một phía của 0. Số liệu trích báo cáo.*

Bảng chọn hai budget phân biệt rõ nhất; toàn bộ 20 hàng trong [CSV trích xuất](../../results/report_bundle_20261106/tables/campaign_cut_archived.csv).

| Config | Budget | Full | Cut | Zero | Cut−Full [95% CI] |
| --- | --- | --- | --- | --- | --- |
| high_only | 3000 | 2.426 | 2.333 | 2.337 | -0.092 [-0.164; -0.021] |
| high_only | 25000 | 2.262 | 2.256 | 2.154 | -0.006 [-0.070; +0.058] |
| middle_high | 3000 | 2.407 | 2.342 | 2.315 | -0.065 [-0.126; -0.003] |
| middle_high | 25000 | 2.209 | 2.229 | 2.183 | +0.020 [-0.046; +0.088] |
| middle_high_2 | 3000 | 2.382 | 2.372 | 2.34 | -0.010 [-0.071; +0.052] |
| middle_high_2 | 25000 | 2.18 | 2.295 | 2.182 | +0.115 [+0.037; +0.194] |
| shallow_high | 3000 | 2.43 | 2.309 | 2.3 | -0.121 [-0.187; -0.054] |
| shallow_high | 25000 | 2.211 | 2.229 | 2.121 | +0.018 [-0.046; +0.081] |
| balanced | 3000 | 2.412 | 2.311 | 2.321 | -0.101 [-0.146; -0.055] |
| balanced | 25000 | 2.259 | 2.279 | 2.164 | +0.020 [-0.027; +0.065] |

![Hình 2. Mean τ_effective theo lượng visual giữ lại. Chưa vẽ CI vì bảng mean nguồn không cung cấp CI cho từng điểm này; đây không phải phép so ba chính sách chọn vị trí.](figures/02_retention_curves.png)

*Hình 2. Mean τ_effective theo lượng visual giữ lại. Chưa vẽ CI vì bảng mean nguồn không cung cấp CI cho từng điểm này; đây không phải phép so ba chính sách chọn vị trí.*

## Phân tích và insight

- Tại 3K, Cut−Full có CI âm ở 4/5 configs; `middle_high_2` chưa khác rõ. Tại 25K, contrast dương rõ nằm ở `middle_high_2` (+0,115 [0,037; 0,194]); không ngoại suy sang các configs khác.
- Zero và Cut đồng thời đổi những thuộc tính khác nhau: Zero giữ các rows rỗng, Cut đổi tập keys/context length. Kết quả khác nhau không cô lập được riêng nguyên nhân “position” hoặc “attention dilution”.
- Một số mức retention có mean cao hơn Full, nhưng “mức tốt nhất” được chọn sau khi xem nhiều điểm trên cùng cohort. Cần held-out và timing để xác nhận policy có ích; chưa gọi là điểm tối ưu triển khai.
- So checkpoints thay đổi layer selection cùng checkpoint đã train. Chưa xác nhận causal effect của một layer hoặc multi-layer fusion.

## Giới hạn và trạng thái

CI chứa 0 được đọc là **chưa rõ khác biệt**, không chứng minh tương đương. Nhiều contrasts chưa hiệu chỉnh đa kiểm định. Các câu “bão hòa”, “visual là nhiễu”, “bắt buộc nén” trong diễn giải cũ được hạ thành giả thuyết; bảng này chỉ hỗ trợ acceptance sensitivity. Báo cáo hiện không tái xác minh 6.000 raw cases. Attention và output contract của cùng campaign được tổng hợp riêng ở [báo cáo 09](09_attention_output_contract.md).

**Retention đều/trái/phải:** người nghiên cứu đã chạy và báo chưa thấy xu hướng khác; chưa tìm được artifact định lượng ghép cặp cho ba policies. Không điền số hoặc vẽ ba đường suy đoán. Xem [báo cáo 08](08_legacy_va_thieu_bang_chung.md).

## Truy vết

- [Báo cáo campaign gốc — §1.2, §2.2](../../markdown/Baocao_DFlash_4Config_20261029/tong_hop_so_sanh_4_config_dflash.md).
- [Bảng Cut/Zero đã trích](../../results/report_bundle_20261106/tables/campaign_cut_archived.csv) và [bảng retention](../../results/report_bundle_20261106/tables/retention_archived.csv); giữ độ chính xác 3 chữ số của nguồn.
- Hướng tiếp: khóa policy trên cohort phát triển, kiểm tra paired held-out và E2E; không chọn policy bằng CI của cùng tập đã dùng chọn.


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
