# Báo cáo thực nghiệm DSpark 3-layer: VisualCut và Attention

**Ngày chạy:** 2026-11-03
**Target:** Qwen2.5-VL-3B-Instruct
**Tập dữ liệu:** VideoDetailCaption VDC50; tập được đánh giá ghép cặp **N=15** / 50
**GPU:** NVIDIA A100-SXM4-80GB
**Metric acceptance:** $\tau_{\mathrm{effective}}$, số token hiệu dụng mỗi vòng xác minh (đã gồm target bonus)

## Tóm tắt phạm vi

Báo cáo này chạy lại ma trận VisualCut trong PDF DFlash balanced cho checkpoint DSpark Phase 2 `balanced [1, 9, 17, 25, 33]`. Kết luận bên dưới chỉ áp dụng cho checkpoint này, target Qwen2.5-VL-3B-Instruct và tập con được chọn từ cohort VDC50 đã calibrate.

Ma trận được giới hạn ở N=15 video trong cohort VDC50, dùng chung sample ID qua 4 visual-token budget (400, 3K, 13K, 25K) và 6 điều kiện (`full`, `zero`, `cut`, `retained_75`, `retained_50`, `retained_25`). Quy tắc chọn mẫu: first 15 shared calibrated sample IDs in manifest order, paired across all four budgets. Draft nhận điều kiện visual được can thiệp; target vẫn dùng prompt đầy đủ. Mỗi ca được chạy capture và timing riêng; phân tích chỉ tính ca có hai lượt lặp khớp nhau và có attention shard hợp lệ.
Để tiết kiệm compute, campaign báo cáo N=15 thay vì chạy đủ 50 mẫu. Mỗi forward attention được tổng hợp theo cùng công thức mass/density/entropy rồi lưu thành summary record có SHA-256; pilot một mẫu vẫn lưu tensor attention thô để đối chiếu. Các kết quả 400 và 3K là subset của lần chạy 50 mẫu đã hoàn thành; 13K và 25K được chạy với sample limit N=15.

**Không đồng nhất acceptance với độ đúng câu trả lời.** Tỷ lệ khớp chuỗi greedy target được báo riêng; ROUGE-L/BLEU chỉ đo tương đồng với caption tham chiếu VDC.

## Checkpoint và môi trường

| Cấu hình                  |        Target layers | Phase 2 step | Số tham số draft | SHA-256 weight-only state (đầu…cuối) |
| --------------------------- | -------------------: | -----------: | -----------------: | ---------------------------------------- |
| Balanced [1, 9, 17, 25, 33] | `1, 9, 17, 25, 33` |         6372 |        329,993,217 | `525103…beb5f7`                       |

- SHA-256 đầy đủ của weight-only state: `525103682aa5eddf27fbcade637df6181b3351d6d0894d5fdd575ccfb5beb5f7`; đối chiếu [`checkpoint_manifest.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/checkpoint_manifest.json).
- Runtime: Python 3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]; PyTorch 2.11.0+cu130; CUDA 13.0; dtype `torch.bfloat16`; target attention `sdpa`; draft attention `eager`; capture `all_forwards_all_heads_all_queries`.
- Dataset manifest SHA-256: `ecf37cbf23226ff5eb4f5a54b187ff0d476b3f940d27dd0e52b465a93bda2a2e` (50 rows).
- Calibration: `v5_decord_matched_nframes_minmax_pixels_target_seed`, Decord, ±10% token tolerance; cohort gốc N=50; tập đánh giá N=15; [`cohort_selection.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/calibration/cohort_selection.json), [`reduced_provenance.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/reduced_provenance.json).
- Local source revision: `6fa277989d9ecfb9e9d036d738927de0660abcc5`; diff SHA-256: `6964be70f10890b5f10682bc2333283491e48b7ffa397e855fe11d77f07dc5b4`.
- Inference states chỉ giữ `draft_state_dict` cần cho loader; archive gốc không bị sửa và vẫn là nguồn đầy đủ để resume training.

## Kiểm tra độ phủ campaign

| Cấu hình                  | Expected cases | Journal rows | Valid attention cases | Missing | Invalid | Exact output match |
| --------------------------- | -------------: | -----------: | --------------------: | ------: | ------: | -----------------: |
| Balanced [1, 9, 17, 25, 33] |            360 |          360 |                   360 |       0 |       0 |     42/360 (11.7%) |

`Exact output match` là so khớp toàn bộ chuỗi speculative với greedy target. Ca không khớp vẫn có thể hợp lệ cho acceptance/attention nếu lượt capture và timing lặp lại cho cùng output và acceptance rounds; chúng không được gọi là lossless.

## Smoke inference và kiểm tra checkpoint

Pilot của checkpoint balanced gồm 12 ca: hai budget (400, 25K) × sáu điều kiện × một video. Capture/timing lặp khớp ở 12/12 ca; validator shard pass 2/2 task. Exact greedy-target output match: 0/12 ca. Pilot không gộp vào thống kê campaign N=15 phía dưới.

Artifacts pilot: [`pilot_summary.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/pilot/pilot_summary.json), [`pilot_validation_summary.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/pilot/pilot_validation_summary.json). Raw pilot tensor shards được giữ trên Colab; mỗi summary validator ghi rõ trạng thái hash/key-map.

## Thực nghiệm 1 — Giá trị và vị trí visual

Bảng dưới so sánh acceptance cho `full`, `cut` và `zero`. CI là bootstrap 95% ở cấp sample.

### Balanced [1, 9, 17, 25, 33]

| Budget | Full $\tau$ [95% CI] | Cut $\tau$ [95% CI] | Zero $\tau$ [95% CI] |    Cut − Full [95% CI] |   Zero − Full [95% CI] |
| -----: | ---------------------: | --------------------: | ---------------------: | ----------------------: | ----------------------: |
|    400 |   2.073 [1.955, 2.198] |  1.990 [1.883, 2.096] |   2.022 [1.908, 2.133] | -0.084 [-0.139, -0.035] | -0.051 [-0.104, -0.007] |
|     3K |   2.072 [1.977, 2.192] |  2.007 [1.937, 2.086] |   2.051 [1.966, 2.137] | -0.065 [-0.132, -0.003] |  -0.020 [-0.086, 0.034] |
|    13K |   1.947 [1.843, 2.050] |  1.837 [1.740, 1.929] |   1.894 [1.801, 1.987] | -0.111 [-0.186, -0.047] | -0.054 [-0.116, -0.000] |
|    25K |   1.883 [1.799, 1.970] |  1.829 [1.740, 1.923] |   1.854 [1.759, 1.947] | -0.054 [-0.106, -0.006] |  -0.030 [-0.077, 0.019] |

![Full/Zero/Cut acceptance](figures/fig1_visual_cut_tau.png)

## Thực nghiệm 2 — Quét tỷ lệ retention

Retention sweep gồm `cut` (0%), 25%, 50%, 75% và `full` (100%). Contrast chính được đọc là `retained_25 − full` tại 25K; các contrast còn lại mang tính mô tả. CI chứa 0 được diễn giải là chưa có bằng chứng rõ về khác biệt theo contrast đó.

| Cấu hình                  | Budget | Retention tốt nhất theo mean $\tau$ | $\tau$ | Retained 25% − Full [95% CI] |
| --------------------------- | -----: | --------------------------------------- | -------: | ----------------------------: |
| Balanced [1, 9, 17, 25, 33] |    400 | Full                                    |    2.073 |        -0.024 [-0.066, 0.013] |
| Balanced [1, 9, 17, 25, 33] |     3K | Retained 50%                            |    2.079 |        -0.006 [-0.037, 0.028] |
| Balanced [1, 9, 17, 25, 33] |    13K | Full                                    |    1.947 |        -0.024 [-0.060, 0.012] |
| Balanced [1, 9, 17, 25, 33] |    25K | Retained 25%                            |    1.902 |         0.019 [-0.021, 0.064] |

![Retention sweep](figures/fig2_retention_sweep.png)

Hình dưới bổ sung tốc độ đo được theo cùng các mức retention. Speedup là tỷ số thời gian `target / speculative` trên lượt timing; đường ngang 1× biểu thị ngang tốc target. CI bootstrap theo 15 video.

![Decode và E2E speedup theo visual retention](figures/fig5_retention_speedup.png)

\clearpage

## Thực nghiệm 3 — Attention mass, density và entropy

Attention được tổng hợp từ các forward của draft tại ngân sách 25K. Mass là tổng xác suất attention trên vùng; density là mass chia cho số key trong vùng. Điều kiện `cut` không có visual key nên visual density không được định nghĩa ở điều kiện đó.

![Attention mass và density tại 25K](figures/fig3_attention_25k.png)

| Cấu hình                  | Điều kiện | Visual mass | Visual density | Question mass | Generated-target-hidden mass | Normalized entropy |
| --------------------------- | ------------ | ----------: | -------------: | ------------: | ---------------------------: | -----------------: |
| Balanced [1, 9, 17, 25, 33] | Full         |       0.368 |          0.000 |         0.020 |                        0.387 |              0.455 |
| Balanced [1, 9, 17, 25, 33] | Cut          |       0.000 |             — |         0.044 |                        0.582 |              0.421 |
| Balanced [1, 9, 17, 25, 33] | Retained 25% |       0.239 |          0.000 |         0.026 |                        0.459 |              0.388 |
| Balanced [1, 9, 17, 25, 33] | Retained 50% |       0.295 |          0.000 |         0.023 |                        0.428 |              0.414 |
| Balanced [1, 9, 17, 25, 33] | Retained 75% |       0.335 |          0.000 |         0.021 |                        0.405 |              0.436 |
| Balanced [1, 9, 17, 25, 33] | Zero         |       0.278 |          0.000 |         0.027 |                        0.443 |              0.450 |

## Thực nghiệm 4 — Throughput, chất lượng tham chiếu và output agreement

Throughput là tốc độ speculative đo trong lượt timing. Tỷ lệ khớp chuỗi là exact match với greedy target; BLEU/ROUGE-L/coverage so với caption tham chiếu chỉ là độ tương đồng văn bản, không phải đánh giá sự thật của caption.

![Throughput và output agreement](figures/fig4_speed_and_agreement.png)

| Cấu hình                  | Budget | Điều kiện |  Draft token/s [95% CI] | Speculative ROUGE-L | Δ ROUGE-L so với target | Exact match (%) |
| --------------------------- | -----: | ------------ | ----------------------: | ------------------: | ------------------------: | --------------: |
| Balanced [1, 9, 17, 25, 33] |    400 | Full         | 38.754 [36.622, 40.938] |               0.185 |                    -0.000 |            6.7% |
| Balanced [1, 9, 17, 25, 33] |    400 | Cut          | 37.339 [35.451, 39.270] |               0.185 |                    -0.000 |            6.7% |
| Balanced [1, 9, 17, 25, 33] |    400 | Retained 25% | 38.435 [36.506, 40.329] |               0.185 |                    -0.000 |            6.7% |
| Balanced [1, 9, 17, 25, 33] |     3K | Full         | 36.360 [34.969, 37.949] |               0.195 |                     0.001 |            0.0% |
| Balanced [1, 9, 17, 25, 33] |     3K | Cut          | 35.817 [34.667, 36.993] |               0.195 |                     0.001 |            0.0% |
| Balanced [1, 9, 17, 25, 33] |     3K | Retained 25% | 36.723 [35.333, 38.324] |               0.195 |                     0.001 |            0.0% |
| Balanced [1, 9, 17, 25, 33] |    13K | Full         | 23.658 [22.445, 24.928] |               0.204 |                    -0.007 |           26.7% |
| Balanced [1, 9, 17, 25, 33] |    13K | Cut          | 23.697 [22.299, 25.163] |               0.204 |                    -0.007 |           26.7% |
| Balanced [1, 9, 17, 25, 33] |    13K | Retained 25% | 24.367 [23.041, 25.791] |               0.204 |                    -0.007 |           26.7% |
| Balanced [1, 9, 17, 25, 33] |    25K | Full         | 13.817 [12.903, 14.794] |               0.201 |                     0.006 |           13.3% |
| Balanced [1, 9, 17, 25, 33] |    25K | Cut          | 14.240 [13.286, 15.232] |               0.201 |                     0.006 |           13.3% |
| Balanced [1, 9, 17, 25, 33] |    25K | Retained 25% | 14.526 [13.604, 15.447] |               0.201 |                     0.006 |           13.3% |

### Mean accepted length và speedup

Metric $\tau_{\mathrm{effective}}$ là số token hiệu dụng mỗi vòng xác minh (token draft được chấp nhận cộng target bonus). Speedup được tính từ lượt timing riêng theo tỷ số `target / speculative`; E2E gồm prefill và decode, giá trị trên 1 nghĩa là nhanh hơn target greedy.

| Cấu hình | Budget | Điều kiện | $\tau_{\mathrm{effective}}$ [95% CI] | Decode speedup [95% CI] | E2E speedup [95% CI] |
| ---------- | -----: | ------------ | -------------------------------------: | ----------------------: | -------------------: |
| Balanced   |    400 | Full         |                   2.073 [1.958, 2.194] |    1.482 [1.362, 1.603] | 1.470 [1.350, 1.586] |
| Balanced   |    400 | Cut          |                   1.990 [1.885, 2.098] |    1.426 [1.313, 1.549] | 1.416 [1.301, 1.528] |
| Balanced   |    400 | Retained 25% |                   2.049 [1.932, 2.164] |    1.470 [1.347, 1.587] | 1.459 [1.343, 1.576] |
| Balanced   |     3K | Full         |                   2.072 [1.975, 2.199] |    1.462 [1.287, 1.664] | 1.409 [1.259, 1.569] |
| Balanced   |     3K | Cut          |                   2.007 [1.936, 2.082] |    1.431 [1.270, 1.610] | 1.387 [1.249, 1.550] |
| Balanced   |     3K | Retained 25% |                   2.066 [1.974, 2.187] |    1.478 [1.291, 1.677] | 1.424 [1.273, 1.604] |
| Balanced   |    13K | Full         |                   1.947 [1.838, 2.052] |    1.228 [1.160, 1.311] | 1.165 [1.111, 1.216] |
| Balanced   |    13K | Cut          |                   1.837 [1.745, 1.931] |    1.227 [1.151, 1.309] | 1.169 [1.113, 1.230] |
| Balanced   |    13K | Retained 25% |                   1.923 [1.819, 2.032] |    1.281 [1.202, 1.362] | 1.202 [1.145, 1.259] |
| Balanced   |    25K | Full         |                   1.883 [1.800, 1.965] |    0.787 [0.733, 0.843] | 0.857 [0.819, 0.895] |
| Balanced   |    25K | Cut          |                   1.829 [1.740, 1.922] |    0.821 [0.767, 0.882] | 0.883 [0.845, 0.925] |
| Balanced   |    25K | Retained 25% |                   1.902 [1.804, 1.991] |    0.853 [0.789, 0.914] | 0.902 [0.859, 0.945] |

## Diễn giải

- **Balanced [1, 9, 17, 25, 33], 25K:** contrast `retained_25 − full` là 0.019 [-0.021, 0.064] token/vòng (CI 95% chứa 0).
- **Speedup tại 25K:** decode 0.79–0.85× và E2E 0.86–0.90× ở `full`, `cut`, `retained_25`; cả hai đều dưới 1× dù $\tau_{\mathrm{effective}}$ khoảng 1.83–1.90. Đây là số đo trên N=15; cho thấy ở budget này lợi ích acceptance chưa bù chi phí draft và xác minh trên ngữ cảnh visual lớn.

Các confidence interval là mô tả, bootstrap ghép cặp theo sample cho contrasts; campaign này không áp dụng hiệu chỉnh đa kiểm định. Kết quả chỉ mô tả checkpoint balanced Phase 2, trên cohort và phần cứng Colab đã ghi nhận.

\clearpage

## Kết luận theo cấp độ bằng chứng

### CONFIRMED

- Pilot đã chạy 12 ca inference. Capture/timing lặp lại khớp 12/12 ca; kiểm tra tính toàn vẹn artifact đạt 2/2 task.
- Full campaign hiện có 360/360 ca attention hợp lệ trên 360 journal rows; exact greedy-target match là 42/360 ca. Phần này mô tả trạng thái audit được lưu trong artifacts.

### EXPLORATORY

- Các ước lượng acceptance, throughput, attention và paired contrasts mô tả cohort VDC50 đã chọn. CI dùng bootstrap theo sample; chưa hiệu chỉnh đa kiểm định.
- Kết quả chỉ bao phủ checkpoint balanced Phase 2 và target Qwen2.5-VL-3B-Instruct. Không suy rộng sang checkpoint/model/dataset khác.

### FAILED / INCOMPLETE

- Exact greedy-target output match trong pilot đạt 0/12 ca. Tỷ lệ này là kiểm tra strict agreement; acceptance `tau_effective` được báo riêng.
- Ma trận N=15 đạt coverage theo audit: 360/360 ca hợp lệ; cần 360 ca.
- Báo cáo chỉ đo checkpoint balanced được chỉ định; không suy rộng kết luận sang checkpoint DSpark khác.

### HIGHEST VERIFIED RUNG

**Rung 3 — ma trận N=15 hoàn chỉnh:** checkpoint balanced có đủ 360/360 ca hợp lệ, không thiếu/invalid/unexpected rows theo audit. Kết quả không đại diện cho toàn bộ cohort 50 mẫu; strict output agreement được báo độc lập ở bảng trên.

### EVIDENCE GAPS

- Full campaign giữ summary attention theo từng forward kèm SHA-256; raw attention tensors của campaign không được lưu. Vì vậy các metric đã định nghĩa có thể tái kiểm, còn phép phân tích mới trên tensor thô cần chạy capture lại.
- Log của một số video có warning H.264 (`mmco: unref short failure`, `Missing reference picture`), dù các case vẫn hoàn tất. Chưa kiểm chứng độc lập frame giải mã so với nguồn; kết quả phụ thuộc preprocessing/decode hiện tại.
- Báo cáo dùng N=15 video từ cohort calibrate N=50; lựa chọn theo thứ tự manifest nên chưa phải mẫu ngẫu nhiên. Chưa có lần lặp độc lập theo seed/cohort.
- Checkpoint DSpark high-only `[26, 28, 30, 32, 34]` chưa nằm trong phạm vi báo cáo này; kết quả chỉ áp dụng cho checkpoint balanced.

### RECOMMENDED NEXT

Lặp cohort/seed độc lập cho checkpoint balanced để kiểm tra độ ổn định của các paired contrasts; có thể mở rộng sang checkpoint khác khi đó trở thành mục tiêu.

## Artifacts và khả năng tái lập

- Calibration và cohort: [`calibration.jsonl`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/calibration/calibration.jsonl), [`cohort_selection.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/calibration/cohort_selection.json).
- **Balanced [1, 9, 17, 25, 33]:** [`audit_and_summary.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/audit_and_summary.json), [`condition_statistics.csv`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/condition_statistics.csv), [`paired_contrasts.csv`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/paired_contrasts.csv), [`case_attention_summary.csv`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/analysis/case_attention_summary.csv), [`run_config.json`](../../results/dspark_visualcut_attention/dspark_vdc15_20261103/full/dspark_3layer_1_9_17_25_33/run_config.json).
- Mã tạo hình và báo cáo được lưu cạnh source: [`generate_dspark_visualcut_report.py`](../../src/analyze/Validate_Sparrow_hypothesises/generate_dspark_visualcut_report.py).
- Journals giữ `attention_capture` hashes; analyzer xác minh SHA-256 các shard summary trước khi tổng hợp. Pilot raw tensors được giữ riêng trên Colab; full campaign lưu summary record per-forward, không giữ toàn bộ tensor attention.
