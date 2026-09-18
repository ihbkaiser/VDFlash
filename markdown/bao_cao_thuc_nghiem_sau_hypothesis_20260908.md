# Báo cáo thực nghiệm sâu về visual-token ablation

Ngày tổng hợp: 2026-09-08

> Báo cáo này là snapshot trước khi chạy full H1b/H2.1. Trạng thái cập nhật
> xem tại [báo cáo tổng hợp 2026-09-10](bao_cao_tong_hop_thuc_nghiem_hypothesis_20260910.md:1).

## 1. Phạm vi và provenance

Báo cáo này sử dụng hai bộ test đang có trong checkout:

- `dataset/VideoDetailCaption/test.jsonl`: 50 video VDC.
- `dataset/MVBench/classified/selected.jsonl`: 1.000 record. Artifact GPU đã
  chạy trước đây dùng manifest cân bằng 500 mẫu, gồm 100 mẫu cho mỗi 5 task:
  `action_prediction`, `action_sequence`, `moving_attribute`,
  `moving_direction`, `object_interaction`.

Các kết quả DFlash Qwen2.5-VL-3B trong phần benchmark là artifact GPU đã lưu từ
23–25/08/2026, không phải một lần chạy lại trong phiên tổng hợp này. Chúng được
phân tích lại ở mức từng sample và từng acceptance round. Trong phiên hiện tại,
Qwen2.5-VL-3B không còn snapshot trong `.hf_cache`; vì vậy chưa thể claim rằng
benchmark Qwen2.5 mới đã được rerun. Ngoài ra checkout có artifact Qwen2-VL-7B +
MSD đã chạy trước đây trên các cohort VDC 23–30 mẫu cho các phép length,
retention và attention; pilot Z/answer-hint mới nhất trong phiên này vẫn chỉ có
cohort ổn định 2 mẫu.

## 2. Ma trận điều kiện

| Ký hiệu | Intervention ở draft | Mục đích |
|---|---|---|
| Full | Giữ visual positions, keys và values | Baseline |
| Zero | Giữ visual positions/chiều context nhưng zero visual values ở các layer đã chọn | Tách vai trò của value khỏi vai trò chiếm chỗ trong softmax |
| Cut | Xóa vật lý visual positions | Kiểm tra tác động của việc loại bỏ denominator positions |
| Zero-high | Zero visual ở layer cao 25, 33 | Kiểm tra visual signal ở các layer cuối |

Target model luôn giữ full visual input trong các phép so sánh DFlash. Do đó,
thay đổi ở acceptance chủ yếu phản ánh draft-target agreement; không nên diễn
giải trực tiếp là target model mất khả năng trả lời.

## 3. Kết quả benchmark paired trên hai bộ test

Các số `k` dưới đây là `mean_accepted_proposals` lấy trực tiếp từ từng sample
report; `Δk` là trung bình chênh lệch paired so với Full. Đây là acceptance
proposal factor, không đồng nhất với `tau` tổng hợp trong report cũ.

### 3.1. MVBench, n = 500

Checkpoint chính: `qwen25vl-3b-dflash-llava68k-latest`.

| Condition | k | Lossless | Accuracy | Tokens/s |
|---|---:|---:|---:|---:|
| Full | 1.0095 | 0.998 | 0.582 | 26.209 |
| Zero | 0.9960 | 0.998 | 0.582 | 26.909 |
| Cut | 0.9903 | 0.998 | 0.582 | 26.833 |
| Zero-high | 0.9717 | 1.000 | 0.580 | 22.757 |

Paired với Full:

| So sánh | Δk | Bootstrap 95% CI của Δk | Khác speculative hash | Khác target hash | Δ accuracy |
|---|---:|---:|---:|---:|---:|
| Zero − Full | −0.0135 | [−0.0320, 0.0040] | 0/500 | 0/500 | 0.000 |
| Cut − Full | −0.0192 | [−0.0392, 0.0010] | 0/500 | 0/500 | 0.000 |
| Zero-high − Full | −0.0378 | [−0.0673, −0.0097] | 11/500 | 10/500 | −0.002 |

Acceptance theo task cho Full/Zero/Cut gần như không đổi. Các giá trị `k` của
Full theo task lần lượt là 0.466, 0.392, 0.093, 0.131 và 0.256; khác biệt
đáng chú ý chỉ nằm ở task `moving_attribute` của Zero-high (accuracy 0.86 so
với 0.87 của Full).

### 3.2. VDC50, n = 50

| Condition | k | Lossless | ROUGE-L | Tokens/s |
|---|---:|---:|---:|---:|
| Full | 1.5030 | 0.04 | 0.2413 | 113.827 |
| Zero | 1.4332 | 0.04 | 0.2413 | 110.741 |
| Cut | 1.4332 | 0.12 | 0.2378 | 78.408 |
| Zero-high | 1.4891 | 0.04 | 0.2413 | 113.020 |

Paired với Full:

| So sánh | Δk | Bootstrap 95% CI của Δk | Khác speculative hash | Khác target hash | Δ ROUGE-L |
|---|---:|---:|---:|---:|---:|
| Zero − Full | −0.0698 | [−0.0894, −0.0499] | 0/50 | 0/50 | 0.0000 |
| Zero-high − Full | −0.0139 | [−0.0284, 0.0022] | 0/50 | 0/50 | 0.0000 |
| Cut − Full | −0.0697 | [−0.1495, 0.0028] | 47/50 | 48/50 | −0.0034 |

Hai checkpoint `llava68k` và `sharegpt68k` tạo cùng chuỗi output trên VDC50,
nhưng acceptance khác nhau. Với `sharegpt68k`, `k` của Full/Zero/Cut/Zero-high
lần lượt là 0.6680/0.6764/0.6498/0.6692; throughput tương ứng là
77.75/78.00/54.55/77.87 tokens/s. Điều này cho thấy checkpoint ảnh hưởng rõ
đến draft-target agreement dù output cuối cùng có thể giống nhau.

## 4. H1a — text over-attention và hidden shift

### Câu hỏi kiểm chứng

Nếu Cut làm text branch nhận quá nhiều trọng số, cần quan sát chuỗi bằng chứng:

`text mass tăng → text contribution/norm tăng → hidden/logit lệch phân phối → acceptance giảm`.

### Kết quả hiện có

Benchmark paired cho thấy Cut không làm thay đổi output hash hay accuracy trên
MVBench; trên VDC50 Cut làm thay đổi output path nhưng bị confound bởi việc batch
Cut chạy trên GPU khác và target hash khác Full ở 48/50 mẫu. Vì vậy, các số này
chưa chứng minh hidden state rơi khỏi training distribution.

Artifact Sparrow/DFlash có target hidden visual-retention diagnostic theo layer,
nhưng đó là phép đo cosine/response khi giữ visual context ở các layer khác nhau,
không phải mean/covariance hidden trên training data. Nó cho thấy visual cosine
giảm xuống dưới 0.25 vào khoảng layer 32–33 và còn khoảng 0.142 ở layer 36;
đây là bằng chứng visual representation đã biến đổi mạnh ở các layer cuối, không
phải bằng chứng OOD.

### Kết luận cho H1a

**Chưa được kiểm chứng.** Phần TODO “thu thập mean hidden trên training và so
sánh với MVBench” chưa hoàn tất vì checkout không có full training records,
multimodal LLaVA-68K source/images hoặc teacher hidden cache. Không được dùng
hidden-retention diagnostic để thay thế cho training-distribution comparison.

### Thực nghiệm còn thiếu để chốt H1a

1. Với cùng sample và cùng target input, log hidden ở last instruction, token đầu
   answer và 4–16 answer positions đầu cho Full/Zero/Cut.
2. Trên full training data, tính mean/covariance theo layer và token role; đo
   Mahalanobis distance của MVBench/VDC.
3. Log thêm `||o_text||`, `||o_visual||`, draft-target KL và top-1 logit agreement.
4. Chạy Full/Cut trên cùng GPU, cùng process settings để loại target-hash confound.

## 5. H1b — visual attention dilution

### Kết quả attention mass trực tiếp

Pilot MSD mới trên VDC cohort n = 2 đã chạy Full / Reduced-25% / Deleted-0% với
attention logging. Giá trị zero-value control được đo thêm và cho attention
weights giống real-value, đúng với cơ chế: softmax weights phụ thuộc Q/K, không
phụ thuộc V.

| Query | Condition | Visual mass | Text mass | Instruction mass |
|---|---|---:|---:|---:|
| Last instruction | Full | 0.8707 | 0.1014 | 0.0279 |
| Last instruction | Reduced-25% | 0.4282 | 0.4360 | 0.1358 |
| Last instruction | Deleted-0% | 0.0000 | 0.6943 | 0.3057 |
| All text | Full | 0.4920 | 0.4250 | 0.0580 |
| All text | Reduced-25% | 0.2565 | 0.6028 | 0.1156 |
| All text | Deleted-0% | 0.0000 | 0.7812 | 0.1924 |

Trong pilot này, `k` là 1.391/2.113/2.514 cho Full/Reduced-25%/Deleted-0% và cả
hai mẫu đều lossless. Tuy nhiên đây chỉ là n = 2; nó chưa đủ để kết luận có
trade-off hình chữ U trên toàn bộ VDC.

### Diễn giải đúng

- Visual mass cao tự nó không chứng minh dilution: cần đo visual contribution
  và KL giữa logits Full với visual-value-zero.
- Nếu visual mass cao nhưng visual-value-zero làm logits thay đổi ít, đó mới là
  mẫu hình phù hợp với “attention lãng phí”.
- Nếu visual mass cao và value ablation làm KL tăng rõ, visual attention vẫn
  hữu ích; khi đó không nên gọi là dilution.
- Pilot cho thấy zero value giữ nguyên attention weights nhưng làm `k` thay đổi,
  nên value pathway có thể ảnh hưởng output dù denominator không đổi. Đây là
exploratory evidence cho việc phải tách attention mass khỏi value contribution.

### Artifact Qwen2-VL/MSD có quy mô lớn hơn

Artifact `sparrow_validation_qwen2vl_final_20260819` có các phép đo VDC đã pass
validity gate của pipeline:

- Length sweep: 24–28 sample ở mỗi mức, với actual visual tokens khoảng 566,
  2,984, 13,020 và 26,004 cho các mốc 400/3K/13K/25K.
- Retention sweep: 23 sample cho mỗi mức retention, với cả `all_text` và
  `last_instruction`.
- Tất cả các dòng hợp lệ trong các bảng này có `lossless = 1.0` và native
  prefill parity hợp lệ.

Ở length sweep, `accepted_prefix_tokens` của nhánh giữ visual giảm từ 2.449 ở
400 tokens xuống 1.279 ở 25K, trong khi nhánh remove-all tăng từ 2.654 lên
3.869. Ở retention sweep 25K, `last_instruction` tăng từ 1.293 ở retention
100% lên 3.875 ở 0%; `all_text` có mẫu hình gần như giống hệt. Đây là bằng chứng
mạnh hơn pilot n=2 rằng việc giảm visual context có thể tăng acceptance của MSD,
nhưng vẫn chưa phân biệt hoàn toàn dilution khỏi các tác động khác của retention
selection và prefill implementation.

Attention của draft MSD tại query `last_instruction` cũng cho thấy visual mass
tăng rất mạnh theo độ dài visual context:

| Actual visual tokens | N | Draft visual mass | Draft text mass | Draft instruction mass |
|---:|---:|---:|---:|---:|
| 572 | 14 | 0.4099 | 0.4032 | 0.1869 |
| 2,912 | 14 | 0.8519 | 0.1027 | 0.0454 |
| 25,024 | 9 | 0.9962 | 0.0018 | 0.0020 |

Ở target, cùng query có visual/text/instruction mass lần lượt khoảng
0.1300/0.6195/0.2505 tại 572 tokens và 0.2127/0.5727/0.2145 tại 2,912 tokens.
Chênh lệch target–draft cho thấy không nên đồng nhất attention của MSD với
attention của target; H1b cần được đánh giá ở đúng nhánh draft vì acceptance do
draft-target agreement quyết định.

### Kết luận cho H1b

**Có bằng chứng hỗ trợ ở cohort VDC/MSD, nhưng causal value claim chưa được xác
nhận.** Phần TODO “attention mass cho text và visual” đã được đo trực tiếp ở
pilot và trên artifact Qwen2-VL/MSD 23–30 mẫu; retention/length sweep cũng cho
mẫu hình acceptance tăng khi giảm visual context. Tuy nhiên chưa có null-slot
sweep nhiều mức kèm logit-KL, `||o_visual||` và `||o_text||` trên toàn bộ hai
test set, nên chưa thể kết luận rằng visual mass cao là attention “lãng phí”
thay vì visual information hữu ích.

### Diagnostic bổ sung về độ dài visual context

Artifact Sparrow/DFlash cũ có 200 dòng length sweep trên VDC, 50 mẫu ở mỗi
milestone danh nghĩa. Acceptance factor hiệu dụng giảm khi số visual tokens tăng,
trong khi thời gian end-to-end tăng gần tuyến tính:

| Visual target | Mean visual tokens thực tế | `tau_eff` | Speculative end-to-end (s) |
|---:|---:|---:|---:|
| 400 | 519 | 2.5595 | 2.623 |
| 3,000 | 3,700 | 2.3726 | 4.275 |
| 13,000 | 16,381 | 2.1908 | 13.252 |
| 25,000 | 27,530 | 2.1354 | 21.976 |

Pearson correlation giữa visual length và `tau_eff` là −0.401; giữa visual
length và speculative end-to-end là +0.970. Retention sweep 0/1/5/10/25/100%
cho `tau_eff` lần lượt 2.3126/2.3217/2.3524/2.3714/2.3841/2.3726. Đây là
mẫu hình phù hợp với khả năng attention dilution khi giữ nhiều visual context,
nhưng calibration của artifact dùng `allow_out_of_tolerance` và audit tổng thể
chưa đạt strict coverage; vì vậy chỉ ghi nhận là exploratory, không dùng làm
claim xác nhận H1b.

## 6. H3.1 — acceptance theo vị trí câu trả lời

Từ acceptance trace của artifact DFlash, mỗi proposal được gán vào vị trí output
normalized và gom vào 5 bins: 0–10%, 10–25%, 25–50%, 50–75%, 75–100%.

### MVBench, tỷ lệ proposal được chấp nhận

| Condition | 0–10% | 10–25% | 25–50% | 50–75% | 75–100% |
|---|---:|---:|---:|---:|---:|
| Full | .377 | .251 | .318 | .319 | .201 |
| Zero | .379 | .254 | .317 | .318 | .201 |
| Cut | .376 | .249 | .319 | .318 | .202 |

### VDC50, tỷ lệ proposal được chấp nhận

| Condition | 0–10% | 10–25% | 25–50% | 50–75% | 75–100% |
|---|---:|---:|---:|---:|---:|
| Full | .577 | .634 | .577 | .586 | .604 |
| Zero | .528 | .612 | .574 | .577 | .604 |
| Cut | .536 | .610 | .590 | .569 | .588 |

Cut/Zero giảm acceptance rõ nhất ở đầu VDC: khoảng −0.041 đến −0.049 ở bin
0–10%, và giảm khoảng −0.022 đến −0.024 ở bin 10–25%. Ở các bin sau, Cut có
thể tăng cục bộ tại 25–50% (+0.013) nhưng vẫn giảm ở 50–100%; không thấy mẫu
hình đơn giản “đầu thấp, cuối cao” trên toàn câu.

### Kết luận cho H3.1

**Chưa được xác nhận; có tín hiệu không nhất quán giữa backend.** Artifact
DFlash VDC50 cho thấy Cut/Zero giảm acceptance ở đầu câu, nhưng không tăng ổn
định ở cuối. Ngược lại, pilot MSD n = 2 cho thấy Reduced/Deleted tăng acceptance
ngay từ các bin đầu (ví dụ bin 0–10%: Full .429, Reduced .786, Deleted .714).
Do đó chưa thể kết luận rằng acceptance chỉ cao hơn sau khi target đã trả lời một
nửa thông tin. Cần khóa cùng model/backend và chạy lại full cohort, đồng thời log
first-reject position và target visual-sensitivity (KL Full-target so với
visual-zero-target) theo answer position.

## 7. H2.1 — VDC cần visual information khi prompt không chứa answer

Thiết kế đúng là factorial 2 × 3:

| Prompt | Full | Reduced/Cut | Deleted |
|---|---:|---:|---:|
| Natural VDC | ✓ | ✓ | ✓ |
| Answer-augmented (25/50/100%) | ✓ | ✓ | ✓ |

Reference chỉ được dùng như một diagnostic leakage control, không dùng để claim
benchmark quality. Primary interaction là:

`[k(Deleted) − k(Full)]_augmented − [k(Deleted) − k(Full)]_natural`.

### Trạng thái

Artifact benchmark lớn hiện chưa có factorial answer-augmented Full/Cut/Deleted
trên toàn bộ VDC50. Pilot MSD answer-hint trước đây chỉ ở cohort nhỏ; không đủ
để kết luận H2.1. Do đó phần TODO “mớm answer vào prompt rồi đo lại metrics và
attention mass” **chưa hoàn tất ở quy mô yêu cầu**.

Pilot đó có kết quả:

| Prompt | Full `k` | Reduced-25% `k` | Deleted-0% `k` | Lossless Full/Reduced/Deleted |
|---|---:|---:|---:|---:|
| Natural | 1.209 | 2.258 | 2.514 | 2/2, 2/2, 2/2 |
| Answer hint 100% | 1.204 | 2.324 | 2.366 | 1/2, 2/2, 1/2 |

Với Deleted so với Full, chênh lệch `k` giảm từ `+1.305` ở Natural xuống
`+1.162` khi có answer hint; interaction theo hướng này là âm, không phải
positive interaction như H2.1 dự đoán. Tuy nhiên answer-hint làm mất strict
losslessness ở 1/2 mẫu của Full và Deleted, nên đây chỉ là tín hiệu bác bỏ sơ bộ,
chưa phải kết luận thống kê.

### Kết luận cho H2.1

**Chưa kiểm chứng.** VDC Cut có chất lượng thấp hơn Full một chút trong artifact
cũ, nhưng khác target hash và khác GPU nên không thể quy nguyên nhân cho việc
prompt thiếu answer semantics.

## 8. H3.2 — số layer của drafter và khả năng hiểu visual

Hypothesis này yêu cầu so sánh nhân quả DFlash 1/3/5 layers. Cần train tất cả
điều kiện qua cùng hai phase (text-only và multimodal), giữ seed, data order,
target features, optimizer schedule và số update cố định.

Hiện checkout chỉ có DFlash 5-layer/6e checkpoints; không có full training data,
prepared manifests, teacher cache và checkpoint 1-layer/3-layer. Vì vậy chưa thể
chạy phép so sánh depth causal. Không được suy luận “DFlash hiểu visual hơn MSD
vì nhiều layer hơn” từ acceptance hiện tại, vì khác biệt còn có thể do training,
kiến trúc, checkpoint và context implementation.

## 9. Đánh giá mức độ hoàn tất

| Hypothesis / phần việc | Trạng thái | Bằng chứng hiện có |
|---|---|---|
| H1a: over-attention gây hidden OOD | **Chưa kiểm chứng** | Chưa có training hidden mean/covariance; có hidden-retention proxy |
| H1b: attention dilution | **Có bằng chứng hỗ trợ, causal value claim chưa xác nhận** | Attention mass/retention trên Qwen2-VL/MSD cohort 23–30 mẫu + paired DFlash benchmark |
| H2.1: answer-augmented làm visual deletion ít hại hơn | **Chưa kiểm chứng** | Chưa có factorial full VDC50 |
| H3.1: Cut chỉ tốt hơn sau nửa câu | **Hỗ trợ một phần** | Position-wise trace trên MVBench500/VDC50 |
| H3.2: depth nhiều layer hiểu visual tốt hơn MSD | **Bị block** | Thiếu DFlash 1/3-layer và training assets |

## 10. Ưu tiên thực nghiệm tiếp theo

1. **Ưu tiên 1 — H1b full cohort:** chạy attention mass + logit-KL + visual/text
   contribution trên VDC50 và một MVBench cohort cân bằng; sweep
   Full/Reduced-25/10/5/Deleted và Zero-value control.
2. **Ưu tiên 2 — H2.1:** chạy factorial natural/answer-augmented trên cùng 50
   VDC, cùng GPU và cùng preprocessing; đo interaction acceptance, attention và
   target parity.
3. **Ưu tiên 3 — H3.1:** bổ sung first-reject position, token-level correctness
   và visual-sensitivity theo vị trí.
4. **Ưu tiên 4 — H1a:** khôi phục training manifest hoặc teacher hidden cache;
   nếu không có, chỉ báo cáo paired shift so với Full, không dùng từ “OOD”.
5. **Ưu tiên 5 — H3.2:** khôi phục data/cache rồi train DFlash 1/3/5 layers với
   hai phase giống nhau.
6. Khi Qwen2.5-VL-3B snapshot được khôi phục, rerun Full và Cut trên cùng GPU;
   đặc biệt VDC Cut hiện bị confound bởi target hash khác 48/50 mẫu và throughput
   thấp hơn mạnh dù giảm sequence length.

## Artifacts chính

- [VDC50 DFlash comparison](../results/infer/VDC50_COMPARISON_REPORT.md)
- [MVBench Full report](../results/infer/mvbench100_full_20260823/MVBENCH_REPORT.md)
- [Sparrow/DFlash diagnostic report](../results/sparrow_validation_dflash_qwen25vl3b_2026-08-24_flash_model_parallel_2gpu/REPORT.md)
- [MSD Z pilot summary](../results/hypothesis_validation_gpu_20260908/SUMMARY.md)
- [Kế hoạch protocol ban đầu](ke_hoach_thuc_nghiem_hypothesis_validation.md)
