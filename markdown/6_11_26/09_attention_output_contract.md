# 09 · Attention và output contract của campaign 5 configs

**Kết quả:** mass phân bổ khác nhau giữa visual, question và generated context; mean caption metrics gần target nhưng exact output match thấp. **Mức bằng chứng: bảng báo cáo lịch sử**, chưa có raw campaign local để xác nhận region map/timing. Đây là cùng campaign của báo cáo 01, không phải run độc lập.

## Thiết kế và phép đo

| Thành phần | Phạm vi |
|---|---|
| Backend / cohort | DFlash 3-layer, 5 checkpoints, Qwen2.5-VL-3B, VDC50, H200 theo báo cáo |
| Attention mass | Tổng attention trên từng vùng; số rows/keys và averaging cần đọc theo implementation nguồn |
| Density | Mass / số key; density ratio không đồng nghĩa causal contribution |
| Quality | BLEU, ROUGE-L, content coverage so với reference caption |
| Output contract | Exact sequence match so với greedy target; khác caption similarity |
| Timing table | Policy được chọn khác nhau giữa config/budget; mean đã làm tròn, thiếu CI local |

## Attention: mass và density

![Hình 17. Regional mass ở 25K và density ratio generated-text/visual qua budgets; số trích bảng nguồn, không tái chuẩn hóa rounded mass thành đúng 100%.](figures/17_campaign_attention.png)

*Hình 17. Regional mass ở 25K và density ratio generated-text/visual qua budgets; số trích bảng nguồn, không tái chuẩn hóa rounded mass thành đúng 100%.*

| Config (25K) | Visual mass % | Question % | Generated text % | Draft query % | Gen/visual density ratio |
| --- | --- | --- | --- | --- | --- |
| high_only | 50.6 | 9.3 | 25.1 | 15.1 | 1565.1 |
| middle_high | 53.2 | 8.8 | 22.7 | 15.3 | 1036.3 |
| middle_high_2 | 46.7 | 8.0 | 31.8 | 13.4 | 1019.3 |
| shallow_high | 51.3 | 8.5 | 28.0 | 12.1 | 918.3 |
| balanced | 47.9 | 16.1 | 23.4 | 12.7 | 1835.7 |

Generated context có mass đáng kể dù ít keys; visual có nhiều keys nên tổng mass lớn vẫn đi cùng per-key density thấp. Density ratio tăng không tự chứng minh visual là nhiễu hoặc generated text chứa toàn bộ thông tin video. Thiếu key-map/query weighting raw nên các chú giải “visual hub”, “phân công chức năng tầng” chỉ là mô tả attention, chưa phải kết quả causal ablation theo layer.

## Caption similarity, exact match và tốc độ

![Hình 18. Speedup được báo cáo và exact target output match ở các policies được chọn. Các cell có policy khác nhau; không phải một policy cố định quét budget, chưa có CI speedup.](figures/18_campaign_output_contract.png)

*Hình 18. Speedup được báo cáo và exact target output match ở các policies được chọn. Các cell có policy khác nhau; không phải một policy cố định quét budget, chưa có CI speedup.*

| Config (25K) | Condition | Spec ROUGE-L | Target ROUGE-L | Δ ROUGE-L | Exact match % | Reported speedup |
| --- | --- | --- | --- | --- | --- | --- |
| high_only | retained_25 | 0.199 | 0.2 | -0.001 | 0.0 | 0.99 |
| middle_high | retained_25 | 0.196 | 0.2 | -0.004 | 0.0 | 1.26 |
| middle_high_2 | cut | 0.2 | 0.2 | +0.000 | 0.0 | 1.35 |
| shallow_high | retained_25 | 0.201 | 0.2 | +0.001 | 2.0 | 0.99 |
| balanced | retained_75 | 0.2 | 0.195 | +0.005 | 2.0 | 0.98 |

## Insight và điều chỉnh diễn giải cũ

- Exact target output match của 20 hàng chọn trong bảng nguồn chỉ 0–22%; riêng 25K là 0–2%. Vì thế chưa xác nhận **lossless decoding**, dù mean ROUGE-L gần nhau.
- Không có equivalence margin, quality CI hoặc factuality annotation. Mean caption similarity gần nhau chưa đủ kết luận “bảo toàn ngữ nghĩa hoàn hảo”. Riêng balanced 25K, ROUGE-L chênh +0,005 theo bảng, nên diễn đạt toàn bảng sai số <0,002 cũng không được các số này hỗ trợ.
- Speedup được báo cáo không bảo đảm đo trên cùng output contract; policy chọn hậu nghiệm còn làm nhiễu so sánh. Không dùng bảng này để chốt Pareto optimum hoặc policy triển khai.
- Giữ 25% visual positions chỉ mô tả số rows visual; chưa đo tổng peak memory/total KV giảm 75% vì text, generated context, weights và target cache vẫn tồn tại.
- Hướng đúng cần kiểm tra tiếp: paired timings cùng output và audit target parity, sau đó mới đánh giá retention/distillation trên held-out.

## Truy vết

- [Báo cáo campaign gốc — §3.2, §3.3B, §4.2](../../markdown/Baocao_DFlash_4Config_20261029/tong_hop_so_sanh_4_config_dflash.md).
- [Regional mass trích](../../results/report_bundle_20261106/tables/campaign_attention_archived.csv); [density ratios trích](../../results/report_bundle_20261106/tables/campaign_density_archived.csv); [quality/speed table trích](../../results/report_bundle_20261106/tables/campaign_quality_archived.csv).
- Nguồn gốc được giữ nguyên; bộ ngày 6/11 chỉ điều chỉnh **phạm vi kết luận** dựa trên chính các bảng đó.


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
