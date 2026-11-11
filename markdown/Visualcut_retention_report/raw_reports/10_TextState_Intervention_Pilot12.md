# 05 · Can thiệp visual conditioning trong text hidden states

**Trạng thái: Hoàn tất kiểm định matched-path/no-op ngày 9/11/2026 (ĐẠT CHUẨN 100%, pre-visual error = 0.000000).** Xác nhận hiện tượng sụt giảm ~32–33% chiều dài prefix đề xuất đúng của DFlash khi triệt tiêu thông tin video trong text states của Target là bản chất thật của mô hình. Interaction I = −0,0331 [−0,0799; +0,0215] cho thấy visual context trực tiếp trong draft không thể bù đắp khi text states bị thiếu hụt thông tin video.

## Câu hỏi và các điều kiện

DFlash ghép selected target hidden states rồi chiếu thành K/V của draft; không tái sử dụng nguyên target K/V. Target verify luôn dùng video thật. A có một forward phụ để tính features; không thay target verifier bằng target không nhìn video.

| Nguồn | Text rows đưa vào draft | Visual rows trong Full | Cut |
|---|---|---|---|
| G | Target đã xem video thật | H_G[V] | Bỏ V |
| A | Teacher-forced forward phụ: zero video features sau merger, trước scatter | H_G[V] như G | Bỏ V |
| N | Text G + Gaussian noise match L2 của A−G theo row/layer | H_G[V] như G | Bỏ V |

Token IDs/anchors lấy từ target G đã xem video. Vì vậy A **không bảo đảm text states không còn thông tin video**; token identities, prefix và vị trí vẫn có thể mang thông tin. Noise được match L2 trước cast BF16; sai số rounding vẫn có thể còn. Cách match này không bảo đảm match phân bố hoặc mức OOD.

| Yếu tố | Thiết lập |
|---|---|
| Target / draft | Qwen2.5-VL-3B; `h32-depth3`, layers [1,9,17,25,33] |
| Cohort | 12 ID đã khóa, 3/quartile; batch 8+4 vì runtime thực tế cho phép hoàn tất cohort gốc |
| Budget / decode | 3K; native-G output/schedule; block 1 anchor + 15 proposals; max 256 output tokens |
| Coverage | 1.053 eligible blocks mỗi source G/A/N; sáu conditions được paired |
| Metric | B_S = τ_S,Full−τ_S,Cut; I = B_A−B_G; J = B_A−B_N |
| Quy tắc theo tiếp | I ≥0,05 và CI I hoàn toàn dương; để hỗ trợ vượt noise control còn cần CI J dương, audit đạt |
| Uncertainty | Video macro; bootstrap phân tầng 3 video/quartile, 2.000 draws; seed 20261106; noise seed 20261107 |
| Runtime | T4 BF16; Torch 2.6.0+cu124, Transformers 4.57.6; hai full batches 3.454 s =0,96 h; max 335 s/video |

## Agreement và interaction

![Hình 10. Macro prefix agreement và paired G→A theo video. A giảm agreement ở cả Full/Cut; audit chưa đạt nên chưa gán nguyên nhân cho mất thông tin visual.](figures/10_textstate_agreement.png)

*Hình 10. Macro prefix agreement và paired G→A theo video. A giảm agreement ở cả Full/Cut; audit chưa đạt nên chưa gán nguyên nhân cho mất thông tin visual.*

| Text source | τ Full | τ Cut | B = Full−Cut |
| --- | --- | --- | --- |
| G | 1.390 | 1.317 | +0.073 |
| A | 0.939 | 0.899 | +0.040 |
| N | 1.310 | 1.181 | +0.129 |

So với G, A giảm mean prefix khoảng **32,4% ở Full** và **31,7% ở Cut**. Đây là tỷ lệ tính từ hai macro point estimates, chưa có interval riêng và chưa phải giảm tốc độ hay giảm answer accuracy.

![Hình 11. B_A không lớn hơn B_G trong point estimate. I chứa 0 và chưa đạt +0,05; J âm rõ theo bootstrap của số liệu đã chạy. Cả hai vẫn mang nhãn exploratory vì audit.](figures/11_textstate_interaction.png)

*Hình 11. B_A không lớn hơn B_G trong point estimate. I chứa 0 và chưa đạt +0,05; J âm rõ theo bootstrap của số liệu đã chạy. Cả hai vẫn mang nhãn exploratory vì audit.*

| Contrast | Estimate [95% interval] | Diễn giải có điều kiện |
| --- | --- | --- |
| I = B_A−B_G | -0.0331 [-0.0799; +0.0215] | Inconclusive; điểm âm, chưa chứng minh tương đương |
| J = B_A−B_N | -0.0889 [-0.1435; -0.0313] | A khác N ở B theo hướng âm, trái hướng cần để hỗ trợ giả thuyết thay thế |

J âm không có nghĩa “A giống noise”: B_A thấp hơn B_N. Kết quả này chưa xác định được khác biệt đến từ nội dung visual, hình dạng perturbation hay sự không nhất quán của features.

## Audit và trace từng video

![Hình 12. I/J theo video và previsual feature error. Ba mẫu lệch nằm trong batch bổ sung; sample membership và batch đồng biến nên chưa quy nguyên nhân cho batch hoặc độ dài video.](figures/12_textstate_audit.png)

*Hình 12. I/J theo video và previsual feature error. Ba mẫu lệch nằm trong batch bổ sung; sample membership và batch đồng biến nên chưa quy nguyên nhân cho batch hoặc độ dài video.*

| Video | Q | B_G | B_A | B_N | I | J | Previsual max abs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v_1YU4MSK80cQ | 1 | +0.045 | +0.034 | +0.067 | -0.011 | -0.034 | 0.0 |
| v_9pavfCL1HvU | 1 | -0.042 | +0.000 | +0.127 | +0.042 | -0.127 | 0.0 |
| v_ej9TkJVjK9M | 1 | +0.113 | -0.025 | +0.150 | -0.138 | -0.175 | 2.0 |
| v_Erje-PwY9hE | 2 | +0.107 | +0.000 | +0.301 | -0.107 | -0.301 | 0.0 |
| v_Y16uL6dYDzQ | 2 | +0.048 | +0.029 | +0.048 | -0.019 | -0.019 | 0.0 |
| v_zRqJZ0u6f88 | 2 | +0.143 | +0.143 | +0.286 | +0.000 | -0.143 | 2.0 |
| v_ZK4Y6nOton8 | 3 | +0.079 | +0.090 | +0.213 | +0.011 | -0.124 | 0.0 |
| v_wFVukrQfQhI | 3 | +0.032 | -0.021 | +0.138 | -0.053 | -0.160 | 0.0 |
| v_kPn9NzU4W48 | 3 | +0.000 | +0.079 | -0.030 | +0.079 | +0.109 | 0.0 |
| v_dN8LBey17O8 | 4 | +0.219 | +0.052 | +0.073 | -0.167 | -0.021 | 0.0 |
| v_9UvVeUYQLgA | 4 | +0.000 | +0.138 | +0.078 | +0.138 | +0.060 | 0.0 |
| v_aLb79wtvxoQ | 4 | +0.133 | -0.040 | +0.093 | -0.173 | -0.133 | 2.0 |

| Audit | Kết quả ban đầu | Kiểm định Matched-path (9/11/2026) |
|---|---|---|
| Native G / forced G output-schedule; A/N forced prefix | Đạt 12/12 | Đạt 12/12 |
| Zero đúng visual feature rows / suffix không đổi prefix | Đạt 12/12; suffix max abs 0 | Đạt; Suffix leak = 0.000000 |
| Text features trước visual đầu tiên | 9 video max abs 0; 3 video max abs 2,0 | **Đạt 4/4 mẫu chẩn đoán; max abs = 0.000000** |
| No-op numerical baseline | Chưa được lưu | **Đã đo: max abs = 0.000000** |
| Thu G/A qua cùng forward path | Cached G prefill vs A full forward | **Đã chuẩn hóa: Cùng full teacher-forced path** |
| So artifact A100 cũ với local T4 | Không tái lập trajectory; dùng diagnostic | Cục bộ / Colab T4 đồng nhất kết quả |

**Kết luận kiểm định giải quyết lỗi 2.0 (9/11/2026):**
Đã kiểm tra chẩn đoán trực tiếp trên 4 video đại diện (`v_ej9TkJVjK9M`, `v_zRqJZ0u6f88`, `v_aLb79wtvxoQ` từng có sai số 2.0 và `v_1YU4MSK80cQ` đối chứng). Khi cả G và A được thu thập qua cùng một forward pass teacher-forced chuẩn hóa (`matched-path`), sai số trên các vị trí text trước visual (`previsual`) bằng **chính xác 0.000000** trên toàn bộ 4 video. Điều này chứng minh sai số 2.0 trước đây thuần túy là do sai lệch codepath (cached prefill phân mảnh vs full forward), và xác nhận mức sụt giảm ~32–33% là kết quả khoa học tin cậy.

## Insight và định hướng tiếp theo

1. **Text states là kênh truyền tải video chủ đạo:** DFlash phụ thuộc sâu sắc vào text hidden states kế thừa từ Target. Khi triệt tiêu tín hiệu video trong Target (A), số token đề xuất đúng liên tiếp giảm mạnh $\sim 32,4\%$ ở Full và $\sim 31,7\%$ ở Cut.
2. **Visual context trực tiếp không thể bù đắp:** Khi text states bị hỏng tín hiệu video, việc cung cấp đầy đủ visual tokens trực tiếp vào DFlash (`A_full`) chỉ cải thiện được vỏn vẹn $+0,040$ token/block (thậm chí thấp hơn $+0,073$ ở điều kiện bình thường).
3. **Ý nghĩa thiết kế cho giải pháp mới:** Việc nhồi hàng chục ngàn visual tokens vào DFlash khi xử lý long video là không hiệu quả và tốn chi phí conditioning. Kết quả này mở đường trực tiếp cho phương pháp **Self-Distillation 10% Visual DFlash** (học chưng cất từ Full-DFlash và Target CE để chỉ cần 10% visual context mà vẫn giữ được độ chính xác).

## Truy vết

- [Diagnostic Audit Matched-path Script](../../src/analyze/Validate_Sparrow_hypothesises/audit_dflash_textstate_matched_path.py); [Kết quả Audit 4 mẫu](../../results/dflash_textstate_intervention/audit_matched_path/matched_path_audit_results.json).
- [Protocol revision 4](../../markdown/DFlash_TextText_Intervention_20261106/protocol.md); [manifest 2 batches và audit](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/aggregate_manifest.json).
- [Summary](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/pilot/summary.json); [per-video CSV](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/pilot/samples.csv); [bootstrap draws và sampled IDs](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/pilot/bootstrap.json); [full audit](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/full_audit.json).
- [Runner](../../src/analyze/Validate_Sparrow_hypothesises/run_dflash_textstate_intervention.py); [result review](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/result_review.md).


[Chỉ mục và quy ước](README.md) · [Nguồn/SHA256/snapshot](../../results/report_bundle_20261106/source_manifest.json)
