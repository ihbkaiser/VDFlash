# 06 · Depth, tiến trình output và answer hint

**Kết quả lịch sử:** visual-value sensitivity đổi theo corpus; chưa thấy lợi ích Cut chỉ tập trung cuối output; oracle answer hint làm đổi deletion effect. **Mức bằng chứng: archived report** — raw H3/E7 chưa có trong checkout để tái audit.

## Thiết kế và metric

| Study | Câu hỏi | Phạm vi đã báo cáo | Contrast |
|---|---|---|---|
| H3.2 | Depth tăng có đổi visual sensitivity? | VDC50, depths 1/3/5, LLaVA68K và ShareGPT68K; 900 rows | (Full−Zero)_D3 − (Full−Zero)_D1; D5 exploratory |
| H3.1 DFlash | Cut chỉ có lợi ở cuối output? | VDC50, 150 Full/Zero/Cut rows, tối đa 64 output tokens | (Cut−Full)_late − (Cut−Full)_early |
| E7-DFlash H2.1 | Hint làm đổi deletion effect? | VDC50; Natural/Hint × Full/Reduced25/Deleted; 300 rows | (Deleted−Full)_Hint − (Deleted−Full)_Natural |

Cùng tên “interaction” nhưng các studies có intervention, checkpoint và thời lượng decode khác nhau. Không gộp thành một ước lượng chung. H2.1 MSD/Qwen2-VL-7B là study khác với E7-DFlash/Qwen2.5-VL-3B.

## Kết quả depth và output progress

![Hình 13. Full−Zero theo depth và các within-study interactions. Error bars trích CI đã báo; không tái bootstrap raw data và không pooled các studies.](figures/13_historical_controls.png)

*Hình 13. Full−Zero theo depth và các within-study interactions. Error bars trích CI đã báo; không tái bootstrap raw data và không pooled các studies.*

| Corpus | Depth | Full−Zero [95% CI] |
| --- | --- | --- |
| LLaVA68K | 1 | +0.0275 [+0.0001; +0.0545] |
| LLaVA68K | 3 | +0.1536 [+0.1036; +0.2059] |
| LLaVA68K | 5 | +0.1075 [+0.0448; +0.1690] |
| ShareGPT68K | 1 | -0.0173 [-0.0318; -0.0043] |
| ShareGPT68K | 3 | -0.0713 [-0.0997; -0.0412] |
| ShareGPT68K | 5 | -0.0540 [-0.0826; -0.0261] |

| Study / contrast | Estimate [95% CI] |
| --- | --- |
| H3.1 Cut late-minus-early | +0.0053 [-0.0058; +0.0165] |
| H3.1 Zero late-minus-early | +0.0045 [-0.0063; +0.0160] |
| H3.2 LLaVA D3-minus-D1 | +0.1261 [+0.0672; +0.1888] |
| H3.2 ShareGPT D3-minus-D1 | -0.0540 [-0.0912; -0.0155] |
| H2.1 DFlash hint interaction | +0.0957 [+0.0302; +0.1542] |

D3−D1 dương ở LLaVA68K nhưng âm ở ShareGPT68K: giả thuyết “depth lớn luôn tăng khả năng tận dụng visual” chưa khái quát. Full−Zero đo acceptance sensitivity, chưa đo visual understanding. H3.1 CI chứa 0, nên “token muộn luôn ít cần visual” chưa được dữ liệu này hỗ trợ; chưa phải kết luận early/late tương đương.

## Answer hint và attention

![Hình 14. E7-DFlash: hint thay đổi mức agreement và deletion effect. Raw attention mass thay đổi khi số keys thay đổi, nên cần đọc cùng density.](figures/14_hint_context.png)

*Hình 14. E7-DFlash: hint thay đổi mức agreement và deletion effect. Raw attention mass thay đổi khi số keys thay đổi, nên cần đọc cùng density.*

| Prompt | Full prefix | Reduced25 | Deleted | Deleted−Full [95% CI] |
|---|---:|---:|---:|---|
| Natural | 0,9382 | 0,9548 | 0,8658 | −0,0724 [−0,1215; −0,0240] |
| Answer hint | 1,0978 | 1,1517 | 1,1211 | +0,0233 [−0,0331; +0,0793] |

Hint cung cấp reference answer và thay đổi prompt/target continuation, nên không cô lập việc chuyển thông tin video vào text hidden states. Attention mass cao cũng có thể đến từ nhiều keys: ở Natural Full, visual density-ratio 0,528 so với question 18,755. Ở Hint Full, hint/question density ratio 0,761 [0,702; 0,826], không chứng minh mỗi hint token được ưu tiên hơn question token.

## Insight và truy vết

Các controls lịch sử hướng tới việc kiểm tra đường text-state trực tiếp, thay vì suy cơ chế từ depth hay attention. Causal masking chỉ cho token nhìn video/prefix đã có; layer sâu không mở quyền nhìn đáp án tương lai.

- [Báo cáo gốc — §4.4, §5.1, §5.2](../../markdown/Baocao_17_9_26/bao_cao_tong_hop_thuc_nghiem_hypothesis_20260910.md).
- [Depth table trích](../../results/report_bundle_20261106/tables/historical_depth.csv); [interaction table trích](../../results/report_bundle_20261106/tables/historical_controls.csv). Số liệu làm tròn theo nguồn, không phải phép chạy mới.
- Local raw paths kỳ vọng: `results/e7_dflash_h3_20260911/`, `results/h32.../` theo báo cáo nguồn; chưa khôi phục nên không ghi trạng thái “local audit passed”.


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
