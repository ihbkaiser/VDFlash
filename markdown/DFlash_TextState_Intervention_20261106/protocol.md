# DFlash: kiểm tra đường truyền video qua text hidden states

Ngày thiết kế: 6/11/2026. Cập nhật thực thi: **6/11/2026, revision 4**. Pilot đã chạy đủ 12 video đã khóa, chia thành batch 8+4 theo runtime thực tế. Output và emission schedule lấy từ native DFlash G target-verifier path; forced G/A/N replay dùng cùng output/schedule. Kết quả định lượng đã tổng hợp, nhưng audit đường thu hidden states chưa đạt: 3/12 video có sai khác tối đa 2.0 ở text rows trước visual đầu tiên, và không có no-op baseline để xác định sai khác này có phải sai số số học hay không. Vì vậy I/J bên dưới chỉ là exploratory, chưa dùng để kết luận nhân quả.

## 1. Câu hỏi và giả thuyết

Text hidden states do target xem video tạo ra có làm giảm lợi ích bổ sung của visual positions trực tiếp trong draft context không?

**Giả thuyết:** khi giảm điều kiện video trong text states, lợi ích Full−Cut trên độ dài prefix accept sẽ tăng. Thí nghiệm giữ nguyên token identities, vị trí, G-native target-verifier output và visual states gốc; thay đổi nguồn text states và sự hiện diện của visual positions trong draft.

Đây là kiểm tra khả năng thay thế giữa hai đường thông tin dưới một can thiệp cụ thể. “Ghép nhiều layer là nguyên nhân”, “text states chứa toàn bộ đáp án” và “bỏ visual giúp tăng tốc” đều chưa được kiểm định bởi phép đo này.

## 2. Phạm vi và các yếu tố cố định

| Yếu tố | Cấu hình khóa |
|---|---|
| Target | Qwen2.5-VL-3B-Instruct; revision `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| Draft | `h32-depth3-llava68k-latest`; 3-layer, target layers `[1, 9, 17, 25, 33]` |
| Checkpoint SHA256 | `77cdbc44dea0e08cc1ee958e3efcef61318e756a77366f4fcef799435d5c3d41` |
| Draft config SHA256 | `a0d3f6e1cce43bc0b8b5e636b8295908afcc60bca38bf4fd9d57c843e5ed1484` |
| Dữ liệu | 12 video đã khóa ban đầu; 3 video mỗi quartile thời lượng |
| Preprocessing | Calibration 3K từng video từ campaign cũ; giữ `nframes`, min/max pixels, reader `decord`, grid và M-RoPE |
| Decode | Native DFlash G target-verifier output làm reference; block size 16: 1 anchor + 15 proposals; output tối đa 256 tokens |
| Precision/backend | BF16, target SDPA, draft eager; `eval()`, không dropout hoặc training |
| Loại phép đo | Forced replay trên cùng output và emission schedule native-G cục bộ; A/N không đổi anchor hoặc lịch tiến bước |

Thiết kế ban đầu chọn các phần tử ở index 1, 3, 5 (đếm từ 0) trong từng quartile của subset 24 đã sắp theo thời lượng. Do dự toán compute, revision 3 chạy trước hai phần tử đầu mỗi quartile. Sau khi đo runtime thực tế, revision 4 chạy tiếp bốn ID còn lại đã khóa trước đó; việc chia batch không dựa trên outcome. Đây là cohort exploratory, không phải held-out.

| Quartile | Video IDs |
|---|---|---|
| 1 | `v_1YU4MSK80cQ`, `v_9pavfCL1HvU`, `v_ej9TkJVjK9M` |
| 2 | `v_Erje-PwY9hE`, `v_Y16uL6dYDzQ`, `v_zRqJZ0u6f88` |
| 3 | `v_ZK4Y6nOton8`, `v_wFVukrQfQhI`, `v_kPn9NzU4W48` |
| 4 | `v_dN8LBey17O8`, `v_9UvVeUYQLgA`, `v_aLb79wtvxoQ` |

Artifact nguồn của đúng 12 video có `status=ok`, `full_matches_target_greedy=true`; lịch cũ gồm **1.143 block đủ 15 proposals và có anchor Full kế tiếp**, tương ứng **17.145 vị trí mỗi điều kiện**. Tuy nhiên, smoke trên local T4 (Torch 2.6.0+cu124, Transformers 4.57.6) không tái lập target token hash của artifact chạy trên A100 (Torch 2.11.0+cu130), dù prompt fingerprint và số visual tokens khớp. Local one-token greedy còn lệch với block verifier tại một token emitted. Vì vậy revision 3 giữ mẫu, target revision, checkpoint, preprocessing settings và metric, nhưng lấy output/schedule từ native DFlash G làm reference; forced G replay phải tái lập chúng, A/N dùng cùng lịch. Số block mới được log; không ghép lịch cũ với trajectory mới. Source replay cũ và one-token greedy chỉ là diagnostic.

[Cohort 12 đã khóa](../../results/dflash_textstate_intervention/pilot12_3k_20261106/design/sample_selection.json), [revision 3 (batch 8)](../../results/dflash_textstate_intervention/pilot12_3k_20261106/design/runtime_revision3_8video.json), [revision 4 (đủ 12)](../../results/dflash_textstate_intervention/pilot12_3k_20261106/design/runtime_revision4_complete12.json), [block schedule và SHA256 nguồn](../../results/dflash_textstate_intervention/pilot12_3k_20261106/design/block_schedule.json), [experiment card gộp](../../results/dflash_textstate_intervention/pilot12_3k_20261106/design/experiment_card_revision4_full12_aggregate.json).

## 3. Tạo hai nguồn text features

Gọi `V` là visual placeholder positions; `T` gồm toàn bộ vị trí còn lại, kể cả text prompt, role/separator tokens và output prefix. Không thay anchor/mask token embeddings của draft.

### G — target xem video gốc

Chạy DFlash Full với preprocessing gốc và video thật; lấy output được phát ra bởi target verifier cùng emission schedule làm reference cục bộ. Forced G replay phải tái lập output và schedule này. Thu hidden states tại đúng năm layer đã khóa, tạo `H_G`; đây là nguồn text và visual rows chuẩn.

### A — target bị giảm visual input

Chạy target với **cùng token IDs teacher-forced của G**, attention mask, grid, M-RoPE và dtype. Adapter hiện thực đặt đầu ra `get_video_features` bằng 0 **sau visual encoder/merger và ngay trước khi scatter vào placeholder**. Vì thao tác này chỉ đổi các hàng visual sẽ được scatter vào `inputs_embeds`, nó tương đương đặt các hàng `V` bằng 0 tại language-model entry; mọi hàng `T`, token IDs, grid và M-RoPE giữ nguyên. Không đặt pixel tensor bằng 0 vì video đen vẫn có thể tạo visual embeddings khác 0; không xóa placeholder hoặc đổi grid.

Adapter phải xác định đúng language-model entry của Transformers version thực chạy; đường dẫn module được ghi vào audit. Hook chỉ hoạt động trong nhánh A và được gỡ trong `finally`. Ngoài hook này, hai forward dùng cùng code path. Cache target của G và A độc lập từ prefill; không tái dùng target KV đã chứa video gốc cho A.

Thu `H_A` ở cùng năm layer. Dù target A có thể dự đoán câu khác, **không lấy argmax của A để đổi chuỗi token**: luôn teacher-force output G-native. VDC reference answer không được chèn vào prompt.

Tại mỗi vị trí `p`, hai target forward chỉ được attend tới các token `≤p`. Khi draft block có anchor tại vị trí tuyệt đối `a`, chỉ cung cấp target features ở `p<a`. Anchor thật tại `a` là token đã biết, đi vào draft qua embedding như baseline. Các slot còn lại là mask. Có thể tính cả chuỗi teacher-forced trước với causal mask, nhưng tuyệt đối không đưa feature hàng `p≥a` vào context của block đó.

### Ghép input của draft

Trước fusion, tạo mỗi source `S` bằng quy tắc:

```text
S = G: text rows = H_G[T]; visual rows = H_G[V]
S = A: text rows = H_A[T]; visual rows = H_G[V]
```

Sau đó dùng nguyên `extract_context_feature → fc → hidden_norm → draft k_proj/v_proj`. Như vậy **visual rows trong cả G_full và A_full đều là visual states từ video thật**. Tuyệt đối không đưa visual rows H_A[V] vào draft; nếu thay cả hai nguồn thì không còn tách được hai đường thông tin.

## 4. Ma trận điều kiện và control nhiễu

| Điều kiện | Text context rows | Visual context rows | Vai trò |
|---|---|---|---|
| `G_full` | H_G[T] | H_G[V] | Baseline gốc |
| `G_cut` | H_G[T] | Bỏ V | Lợi ích visual khi đường text còn nguyên |
| `A_full` | H_A[T] | H_G[V] | Visual thật khi đường text bị can thiệp |
| `A_cut` | H_A[T] | Bỏ V | Cả đường visual trực tiếp bị bỏ và text bị can thiệp |
| `N_full` | H_N[T] | H_G[V] | Control nhiễu text, giữ visual |
| `N_cut` | H_N[T] | Bỏ V | Control nhiễu text, bỏ visual |

Cut giữ nguyên absolute positions/M-RoPE của các token còn lại, không đánh lại chỉ số. Ngay cả khi hai điều kiện có text giống nhau, mỗi điều kiện phải có draft cache riêng. Ưu tiên replay không cache cho cả sáu điều kiện để tránh nhiễm chéo; cách này chỉ được dùng sau gate tái lập baseline. Nếu cần cache để đạt ngân sách, chuyển toàn bộ sáu điều kiện sang cùng một quy trình cache đã audit trước pilot.

**N kiểm tra ảnh hưởng của perturbation nói chung:** với từng text row `p` và selected layer `l`, tạo hướng Gaussian chuẩn hóa `u[l,p]`, rồi:

```text
d[l,p] = norm(H_A[l,p] - H_G[l,p], L2)
H_N[l,p] = H_G[l,p] + d[l,p] * u[l,p]
```

Noise seed `20261107`, sinh bằng stable hash của `(seed, sample_id, layer_id, absolute_position)`; không dùng Python `hash()`. Mỗi hàng nhận cùng noise ở mọi block chứa nó. Nếu `d=0`, giữ nguyên hàng. Match L2 trước fusion theo từng layer; đồng thời log độ lệch sau fc/norm vì projection có thể đổi tương quan độ lớn. N không cần một target forward thứ ba.

Cả sáu điều kiện chạy trên đủ 12 video, một realization noise đã khóa. Việc match L2 chưa bảo đảm A và N có cùng mức OOD; control này chỉ giảm một cách giải thích thay thế. Kiểm tra thêm noise seeds là vòng sau, không lựa seed thuận lợi trong pilot này.

## 5. Metric chính, thống kê và quyết định

Với video `i`, block `b`, điều kiện `c`, so greedy proposals với đúng 15 target tokens tham chiếu. `L[i,b,c]` là số token đúng liên tiếp trước lỗi đầu tiên, trong `[0,15]`. Token đúng sau lỗi đầu tiên không được cộng vào prefix. Không cộng correction/bonus token vào metric này.

```text
tau[i,c] = mean_b L[i,b,c]
B[i,S]   = tau[i,S_full] - tau[i,S_cut], S ∈ {G,A,N}
I[i]     = B[i,A] - B[i,G]
I        = mean_video I[i]                      # metric chính, kỳ vọng > 0
J        = mean_video (B[i,A] - B[i,N])           # control về tính đặc hiệu
```

Mỗi video có trọng số bằng nhau dù số block khác nhau. Dùng percentile paired cluster bootstrap **2.000 lượt, seed `20261106`**: cohort đầy đủ lấy mẫu có hoàn lại 3 video trong từng quartile, dùng chung draw cho toàn bộ sáu điều kiện. Tính cả I và J từ cùng draw. Không resample positions độc lập.

Ngưỡng tác động đáng theo tiếp của pilot: **I ≥ 0,05 token/round**, cùng lower bound của interval 95% >0. Đây là ngưỡng quyết định nghiên cứu đã đặt trước; chưa có power analysis bảo đảm N=12 phát hiện được mức này. Mốc 0,05 cùng bậc với contrast Full−Cut khoảng 0,061 tại 3K trong replay cũ.

| Kết quả | Diễn giải được phép |
|---|---|
| I đạt ngưỡng, interval I hoàn toàn dương; interval J hoàn toàn dương; audit đạt | Hỗ trợ khả năng thay thế giữa hai đường dưới can thiệp này, vượt control noise đã chọn; cần lặp lại để khái quát |
| I đạt ngưỡng nhưng J chưa rõ | Có interaction với perturbation text; chưa quy riêng cho việc giảm thông tin video |
| I chứa 0 | Pilot chưa phân biệt được hiệu ứng; không tuyên bố tương đương hoặc tăng mẫu để tìm significance |
| I hoàn toàn âm | Ngược dự đoán dưới can thiệp này; xem hai route bổ sung nhau, trạng thái không nhất quán hoặc OOD bằng diagnostics, không đổi metric |
| Interval I nằm trọn trong [−0,05; +0,05] | Ước lượng pilot tập trung quanh hiệu ứng nhỏ; chưa phải kiểm định equivalence chính thức |
| Thiếu video/điều kiện hoặc audit đỏ | Run incomplete/invalid; không công bố như pilot 12 video đã hoàn tất |

**Metrics phụ, exploratory:** sáu tau, B_G/B_A/B_N, `tau_G_cut−tau_A_cut`, top-1 mismatch theo slot, target log-probability, JSD nếu không tăng chi phí lưu trữ đáng kể; norm/cosine H_G–H_A theo layer và vùng prompt/output; tỷ lệ prefix 0 và 15. Không dùng lexical-overlap subgroup hoặc early/late subgroup để thay kết luận chính. Seed sensitivity của noise chưa được xác nhận trong vòng này.

## 6. Audit trước khi nhận kết quả

1. **Identity:** kiểm tra checkpoint/config SHA, target revision, video/input fingerprints, prompt/target-token hashes, visual token count và calibration. Ghi source snapshot/hash và versions thực load.
2. **Can thiệp đúng nơi:** prefill A có đúng số visual rows bằng 0; input embeddings ngoài V bằng bản G. No-op hook tạo output giống baseline. Feature rows text trước visual đầu tiên phải trùng trong sai số số học đã đo bằng no-op, vì chưa nhìn được visual.
3. **Không rò tương lai:** test sửa token suffix sau anchor không đổi features được phép dùng của prefix. Assert context max position `<a`, anchor/token/mask layout giống nhau cả sáu điều kiện. Không reuse H_G text qua cache A.
4. **Giữ riêng route:** visual rows H_G giữ nguyên giữa mọi điều kiện Full; Cut không còn visual keys ở bất kỳ draft layer nào. N chỉ đổi T. Log index-map và source label từng vùng, không chỉ tensor shape.
5. **Baseline parity:** native G reference và forced G replay phải khớp output, anchor, proposal count và emitted-token schedule; target verifier phải dự đoán đúng mọi token đã phát ra. A/N dùng cùng G output/schedule. So sánh với one-token greedy và artifact gốc được lưu làm diagnostic, không thay thế gate nội bộ này.
6. **Không tái dùng sai trạng thái:** target G/A cache tách riêng; draft cache tách theo condition, hoặc không cache. `NaN/Inf`, hook còn sót, stale cache hoặc alignment sai đều dừng run.

Smoke ban đầu đã cho thấy target token hash khác giữa local T4 Torch 2.6.0+cu124 và artifact A100 Torch 2.11.0+cu130; prompt fingerprint và visual count khớp nhưng không đủ chứng minh pixel tensor giống hệt vì fingerprint nguồn không hash pixel values. Local one-token greedy còn lệch với target verifier trong emitted path. Revision 3 dùng native DFlash G output làm trajectory và kiểm tra forced replay trên cùng target-verifier path. Output/schedule parity chỉ kiểm tra contract, không đánh giá answer quality.

**Kết quả audit revision 4:** native G và forced G/A/N output-prefix parity đạt 12/12; đúng số visual rows được zero, và suffix perturbation không đổi prefix features. Tuy nhiên, so sánh cached G-prefill với A full teacher-forced forward cho sai khác trước visual đầu tiên ở `v_ej9TkJVjK9M`, `v_zRqJZ0u6f88`, `v_aLb79wtvxoQ` (max absolute error 2.0); 9 video còn lại là 0. No-op baseline chưa được lưu và hai bank đang được thu qua hai đường forward khác nhau, nên không thể biết sai khác này do can thiệp hay đường tính. Audit gate **chưa đạt**; I/J chỉ được báo như exploratory cho tới khi matched-path/no-op audit được làm rõ. [Audit gộp](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/aggregate_manifest.json).

## 7. Ngân sách và trình tự thực hiện

**Giới hạn:** một checkpoint, một budget, 12 video, 6 conditions; **tối đa 2 GPU-hours và 10 phút/video trong pha xử lý video**. Hai full batches mất tổng cộng 3.454 s (0,96 GPU-hours); video chậm nhất 335 s (5,6 phút). Không train, không sweep layer/budget/block size, không chạy 50 video. A100 80GB không có sẵn; dùng T4 cục bộ BF16 và ghi rõ khác biệt stack/runtime.

1. **R0/design:** khóa protocol, IDs, block schedule, hashes và decision rule trước khi chạy.
2. **Static checks:** xác định adapter language-model entry đúng version, kiểm tra mask/indices, pairing, công thức prefix và bootstrap bằng fixtures nhỏ. Runtime preflight kiểm tra CUDA, processor/video dependencies, model/checkpoint access.
3. **Smoke hai video cố định:** `v_1YU4MSK80cQ` và `v_dN8LBey17O8`. Sinh output/schedule bằng native DFlash G; chạy forced G/A/N × Full/Cut trên các block liên quan tới 64 token đầu. G features được thu từ các forward teacher-forced có cache; A được thu bằng một forward teacher-forced causal với visual features bằng 0. Smoke kiểm tra output/schedule parity, target verifier trên emitted prefix, đúng hook/visual-row count, finite logits, coverage, feature rows trước visual, và perturbation suffix không làm đổi prefix features. Target-token hash nguồn cũ và one-token greedy được giữ làm diagnostic, không làm gate. Chỉ chạy phần còn lại nếu toàn bộ gate smoke đạt và ước lượng thời gian nằm trong cap.
4. **Pilot:** chạy toàn bộ các block đủ 15 proposals có anchor Full kế tiếp trong G-native schedule cho sáu điều kiện trên 12 video. Full 12 có **1.053 eligible blocks/source**; mỗi video dùng chung output, lịch và feature banks.
5. **Audit và phân tích:** execution hoàn tất 12/12 video, nhưng audit feature trước visual chưa đạt; vì vậy xuất I/J và bootstrap như exploratory, không kết luận chính cho giả thuyết cho tới khi sửa/đo matched-path no-op.
6. **Cập nhật sổ nghiên cứu:** ghi kết quả vào mục 1 của proposed_exp, cập nhật insight có đúng phạm vi và chọn vòng tiếp theo. Không tự động thực hiện sweep tiếp.

Raw hidden states/KV chỉ tồn tại theo từng video trong RAM/GPU, giải phóng sau video; không dump full vocabulary logits hoặc attention matrices. G bank thu từ cached prefill; A bank thu bằng full teacher-forced causal forward; khác biệt đường thu này là điểm cần kiểm tra bằng no-op trước khi diễn giải nhân quả. N lấy từ hai bank; các bank dùng lại cho sáu điều kiện. Timing pipeline này không đo speedup inference.

## 8. Artifact và công việc triển khai

Artifacts thiết kế và các lần chạy bị dừng được giữ trong `results/dflash_textstate_intervention/pilot12_3k_20261106/`. Hai batch chạy đủ nằm ở `revised_local_stack_v5/` (8 video) và `revised_local_stack_v6_remaining4/` (4 video). Bản gộp và phân tích là [native_g_full12_aggregate](../../results/dflash_textstate_intervention/pilot12_3k_20261106/native_g_full12_aggregate/): `aggregate_manifest.json`, `full_audit.json`, `pilot/blocks.jsonl`, `pilot/positions.jsonl.gz`, `pilot/samples.csv`, `pilot/blocks.csv`, `pilot/bootstrap.json`, `pilot/summary.json`, `pilot/report.md`. Không có `worker.log` hoặc `feature_diagnostics.csv` trong artifacts này.

Runner `run_dflash_textstate_intervention.py` và instrumentation trong decoder hỗ trợ absolute-position interventions, fixed emission schedule và target-verifier parity checks. Các kết quả gộp dùng cohort 12 gốc; file runtime revision 4 ghi rõ hai batch và anomaly audit.

## 9. Giới hạn diễn giải

Token identities và anchor sinh từ target video thật vẫn mang thông tin video. A giảm visual conditioning của hidden representation khi cố định text, không xóa toàn bộ thông tin video khỏi draft. Full với A ghép hai nguồn feature có thể không nhất quán với phân bố train; noise matching chưa loại hết khả năng đó. Kết quả dương không chứng minh layer fusion là nguyên nhân duy nhất; kết quả không dương không bác bỏ toàn bộ đường text.

Cohort revision 4 gồm 12 video, 3 video mỗi duration quartile; 3K là ngân sách token, không phải một sweep độ dài video. Đây là exploratory pilot với một checkpoint và một noise seed, dùng G-native output/schedule. Sai khác A100/T4 và feature-capture audit còn giới hạn diễn giải. Chưa đo held-out tasks, answer quality, free-running acceptance hoặc latency; không dùng khác biệt MSD–DFlash làm đối chứng nhân quả.

[Sổ nghiên cứu](../proposed_exp.md) · [Ghi chú cơ chế](<../Thought - 2-11-26/Thought - 2 11 2026 40c16fef729f80d283a5cee781862213.md>).
