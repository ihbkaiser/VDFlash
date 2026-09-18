# Báo cáo tiến độ H3.2 — DFlash depth1/depth3 trên VDC và MVBench

**Ngày chạy:** 2026-09-15  
**Mục tiêu:** kiểm tra giả thuyết H3.2: DFlash có nhiều draft layer hơn có thể khai thác visual hidden tốt hơn DFlash một layer hay không.

## 1. Thiết kế đã khóa

Target model được giữ cố định là `Qwen/Qwen2.5-VL-3B-Instruct`. Chỉ thay đổi draft checkpoint:

- LLaVA68K: depth1 và depth3.
- ShareGPT68K: depth1 và depth3.

Mỗi sample chạy ba điều kiện:

1. **Full:** giữ visual hidden values và visual positions.
2. **Zero:** giữ nguyên visual positions nhưng đặt visual hidden values bằng 0. Đây là contrast chính để cô lập giá trị visual.
3. **Cut:** loại visual positions khỏi draft context. Đây là contrast phụ, chịu ảnh hưởng của thay đổi độ dài context.

Metric chính là `accepted_effective_tokens` (`τ_eff`). Với mỗi sample, contrast chính là:

```text
I = (τ_eff(Full) − τ_eff(Zero))_depth3
    − (τ_eff(Full) − τ_eff(Zero))_depth1
```

Bootstrap 95% CI được tính ở cấp sample và ghép cặp theo cùng sample ID; không resample từng token hoặc từng speculative round.

VDC dùng calibration 3000-token hiện có, cho phép các điểm `out_of_tolerance` và ghi lại trạng thái calibration. MVBench không dùng calibration VDC; dùng cấu hình cố định 8 frames, `min_pixels=max_pixels=200704`, đồng thời ghi `actual_visual_tokens`.

## 2. Kiểm tra checkpoint

Các checkpoint được tải từ [Hugging Face dataset repository](https://huggingface.co/datasets/Tphuc15/qwen25vl-3b-depth1-depth3). Tất cả đều materialize được đúng architecture:

- depth1: 1 draft layer, 98,050,304 parameters.
- depth3: 3 draft layers, 252,199,680 parameters.
- Cùng `target_layer_ids=(1,9,17,25,33)`, hidden size 2048 và feature dim 10240.
- LLaVA depth1/depth3: `global_step=6375`, epoch 6.
- ShareGPT depth1/depth3: `global_step=6129`, epoch 5.

Smoke test trên VDC và MVBench cho cả hai checkpoint family đều chạy được; target output hash giữ nhất quán giữa Full/Zero/Cut trong các smoke sample.

## 3. VDC — đã hoàn tất

Coverage: **600/600 rows**, gồm 50 samples × 3 conditions × 2 corpus × 2 depth; `status=error`: **0**. Target-output hash audit: không phát hiện khác biệt giữa các depth/condition.

### Kết quả paired Full–Zero

| Training corpus | Depth | n | Mean Full−Zero | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| LLaVA68K | 1 | 50 | 0.0275 | [0.0001, 0.0545] |
| LLaVA68K | 3 | 50 | 0.1536 | [0.1036, 0.2059] |
| ShareGPT68K | 1 | 50 | −0.0173 | [−0.0318, −0.0043] |
| ShareGPT68K | 3 | 50 | −0.0713 | [−0.0997, −0.0412] |

### Primary depth interaction

| Training corpus | n | `(Full−Zero)_depth3 − (Full−Zero)_depth1` | Bootstrap 95% CI |
|---|---:|---:|---:|
| LLaVA68K | 50 | **+0.1261** | **[+0.0672, +0.1888]** |
| ShareGPT68K | 50 | **−0.0540** | **[−0.0914, −0.0142]** |

Diễn giải đúng mức:

- **LLaVA68K/VDC:** dữ liệu ủng hộ H3.2 ở contrast Full–Zero: depth3 có độ nhạy với visual hidden lớn hơn depth1.
- **ShareGPT68K/VDC:** dữ liệu không ủng hộ cùng chiều; depth3 không cho thấy lợi ích visual-hidden lớn hơn depth1 trong contrast này.
- Chưa được phép kết luận rằng “nhiều layer luôn tốt hơn”; corpus training là một biến tương tác rõ ràng trong VDC.

### Hình và artifact

![H3.2 VDC — acceptance theo depth và condition](../results/h32_depth_20260915/analysis_vdc/h32_acceptance_by_depth.png)

![H3.2 VDC — visual-value sensitivity theo depth](../results/h32_depth_20260915/analysis_vdc/h32_visual_sensitivity.png)

![H3.2 VDC — primary depth interaction](../results/h32_depth_20260915/analysis_vdc/h32_primary_depth_interaction.png)

- [VDC summary JSON](../results/h32_depth_20260915/analysis_vdc/h32_summary.json)
- [VDC group statistics CSV](../results/h32_depth_20260915/analysis_vdc/h32_group_statistics.csv)
- [VDC paired statistics CSV](../results/h32_depth_20260915/analysis_vdc/h32_paired_statistics.csv)

## 4. MVBench — đang chạy, chưa kết luận

Full-run được cấu hình cho 1.000 sample trong `selected.jsonl` ở mỗi corpus×depth, gồm 3.000 condition rows/run và 12.000 rows cho toàn bộ bốn run. Các run chạy tuần tự trên GPU0/GPU1 để tránh tranh chấp VRAM.

Tại thời điểm cập nhật báo cáo:

- LLaVA68K depth1: khoảng **2.5K/3000 rows**, tất cả các rows đã ghi đều
  `status=ok`; đã đi qua hơn `800` sample.
- LLaVA68K depth3, ShareGPT68K depth1/depth3: chưa bắt đầu.
- Full MVBench coverage và paired interaction chưa được báo cáo là kết quả.

Runner và log:

- Script: `scripts/run_h32_depth_full_20260915.sh`
- Output root: `results/h32_depth_20260915/mvbench/`
- Journal hiện tại: `results/h32_depth_20260915/mvbench/llava68k_depth1/h3_2_depth_dflash.jsonl`
- Log runtime: `/tmp/vdflash_h32_full.log`

## 5. Trạng thái kết luận

### CONFIRMED

- Implementation và checkpoint loading đạt smoke rung tương đương R4 trên cả hai corpus/dataset.
- H3.2 được kiểm tra đầy đủ trên VDC ở mức 50 sample/cell; VDC LLaVA68K ủng hộ depth interaction dương, còn VDC ShareGPT68K cho interaction âm.

### EXPLORATORY

- Các số liệu MVBench hiện tại chỉ là tiến độ đầu run, chưa đủ để suy luận.
- VDC interaction là kết quả full study của riêng VDC nhưng vẫn cần đối chiếu MVBench trước khi viết kết luận tổng quát.

### INCOMPLETE

- MVBench full paired analysis chưa hoàn tất.
- Chưa có nhiều seed cho mỗi depth; kết luận hiện tại là so sánh giữa một checkpoint depth1 và một checkpoint depth3 cho mỗi corpus.

**Highest verified rung:** VDC full study đã hoàn tất; MVBench full study vẫn đang chạy. Không dùng MVBench partial rows để kết luận H3.2.
