# Báo cáo thực nghiệm H1b và H2.1 — 2026-09-09

## Mục tiêu

Kiểm tra hai nhóm giả thuyết còn thiếu bằng inference trên GPU:

1. H1b: khi giữ, giảm hoặc xoá visual positions, attention của drafter phân
   bổ thế nào giữa visual và text; đồng thời kiểm tra tác động riêng của việc
   zero visual values lên phân phối logit đầu ra của drafter.
2. H2.1: kiểm tra prompt tự nhiên và prompt được bổ sung reference answer trong
   cùng ba mức visual retention, nhằm xem việc đưa thông tin đáp án vào text có
   làm mất lợi thế của visual-token deletion hay không.

## Cấu hình và thiết kế

- Target: `Qwen/Qwen2-VL-7B-Instruct`.
- Drafter: `lucylyn/MSD-Qwen2VL-7B-Instruct`.
- Thiết bị: model-parallel trên RTX 3090 24 GB và RTX A4000 16 GB.
- VDC: 50 mẫu test. MVBench: 1.000 mẫu đã chọn, gồm 5 task, mỗi task 200
  mẫu.
- H1b: `Full`, `Reduced 25%`, `Deleted` visual positions × `real`, `zero`
  visual values × `last_instruction`, `all_text` query policy. Attention được
  chuẩn hoá trên các key đứng trước query. Logit diagnostic được lấy ở
  `draft_initial_next_token`, không phải target `orig` logits.
- H2.1: `Natural`, `Answer-hint` × `Full`, `Reduced 25%`, `Deleted`; greedy
  decoding, 64 token tối đa, 1 fps. Answer-hint có dạng
  `Reference information (oracle hint): <reference>` và chỉ dùng cho chẩn
  đoán alignment, không phải phép đo chất lượng benchmark.

## Kiểm tra tính đầy đủ

- H1b-VDC: 600 dòng, tương ứng 50 mẫu × 12 cấu hình; không có error row.
- H1b-MVBench: 12.000 dòng, tương ứng 1.000 mẫu × 12 cấu hình; đủ cả 5 task,
  không có error row.
- H2.1-Natural và H2.1-Answer-hint: mỗi file 150 dòng, tương ứng 50 mẫu × 3
  mức retention; 150/150 cặp khoá sample-condition khớp, không có error row.

Các mẫu VDC dài ở phần cuối phải dùng fallback `max_pixels=100352`; ba mẫu
cuối trong nhóm này thêm `max_frames=8`. MVBench dùng `max_frames=8` cho toàn
bộ mẫu. Đây là kiểm soát tài nguyên sau khi native preprocessing ở một số
video dài bị host kill, và cần được giữ như một biến giới hạn khi diễn giải.

## Kết quả H1b

Với `last_instruction` và visual values thật, attention mass trung bình:

| Dataset | Visual condition | Visual mass | Text mass |
|---|---:|---:|---:|
| VDC50 | Full | 0.8727 | 0.0953 |
| VDC50 | Reduced 25% | 0.6777 | 0.2394 |
| VDC50 | Deleted | 0.0000 | 0.6925 |
| MVBench | Full | 0.3313 | 0.4305 |
| MVBench | Reduced 25% | 0.1746 | 0.5117 |
| MVBench | Deleted | 0.0000 | 0.6390 |

Mức tăng text mass theo paired comparison:

- VDC50: `Full → Reduced` +0.1441, bootstrap 95% CI [0.1181, 0.1729];
  `Full → Deleted` +0.5972, CI [0.5553, 0.6367].
- MVBench: `Full → Reduced` +0.0811, CI [0.0796, 0.0827];
  `Full → Deleted` +0.2084, CI [0.2061, 0.2107].

Chỉ số tập trung phụ cũng tăng sau khi xoá visual: mean top-five text-token
mass tăng từ 0.0517 lên 0.3256 ở VDC50 và từ 0.2127 lên 0.3009 ở MVBench.

### Value ablation

Zero visual values không làm thay đổi attention mass: sai khác tuyệt đối lớn
nhất giữa `real` và `zero` là 0 trên toàn bộ các cặp. Điều này phù hợp với
việc intervention tác động vào V, không tác động vào Q/K.

Tuy nhiên, intervention có tác động lên logit của drafter:

| Dataset | Condition | Mean max absolute logit delta | Top-1 match |
|---|---:|---:|---:|
| VDC50 | Full | 1.1622 | 100% |
| VDC50 | Reduced 25% | 0.5830 | 100% |
| MVBench | Full | 0.2176 | 100% |
| MVBench | Reduced 25% | 0.2199 | 100% |

Forward và reverse probability KL đều bằng 0 trong các dòng full-run, kể cả
khi tính float64. Vì phân phối greedy hiện rất peaked, KL ở temperature hiện
tại đã bão hoà về số 0 và không đủ nhạy để kết luận hai phân phối giống nhau.
Trong run này, `max logit delta` và `top-1 match` là hai diagnostic có thể dùng
được; một follow-up tốt hơn cần lưu logits hoặc chạy thêm logit-temperature /
logit-margin calibration trên cohort nhỏ.

### Kết luận cho H1b

Kết quả ủng hộ cơ chế attention dilution ở cấp phân bổ attention: giảm/xoá
visual positions làm visual mass giảm và text mass tăng có hệ thống trên cả
VDC50 và MVBench. Kết quả cũng cho thấy text concentration tăng, phù hợp với
phần over-attention, đặc biệt rõ ở VDC50.

Tuy nhiên, đây chưa phải bằng chứng trực tiếp rằng output embedding rơi khỏi
phân phối gốc hoặc câu trả lời sai. Phép so sánh mean hidden trên training
distribution chưa thực hiện được vì workspace không có training examples hoặc
training hidden-state statistics tương ứng. Qwen2.5-VL-3B cũng chưa có trong
cache, nên các số liệu này thuộc cặp Qwen2-VL-7B/MSD-Qwen2VL-7B.

## Kết quả H2.1

Mean `accepted_prefix_tokens` trên VDC50:

| Prompt | Full | Reduced 25% | Deleted |
|---|---:|---:|---:|
| Natural | 1.1600 | 1.7471 | 2.6986 |
| Answer-hint | 1.1898 | 1.6345 | 2.4509 |

Gain `Deleted − Full` là +1.5386 với Natural và +1.2611 với Answer-hint.
Tương tác factorial:

`(Hint: Deleted − Full) − (Natural: Deleted − Full) = −0.2776`

với bootstrap 95% CI [−0.4094, −0.1411]. Vì vậy, kết quả không ủng hộ dự đoán
rằng thêm reference answer vào text sẽ làm mất hiệu ứng acceptance tốt hơn sau
khi xoá visual.

Kết quả này cần được hiểu là alignment/acceptance của drafter với target. Prompt
Answer-hint làm thay đổi chính target continuation: cả 150 target output hashes
đều khác Natural. Do đó, đây chưa phải kiểm định sạch về “visual có còn cần cho
độ đúng câu trả lời không”, và runner hiện chưa thu thập answer accuracy.

Các guardrail cũng không cho thấy suy giảm lossless prefix rõ ràng: với Natural,
tỷ lệ output lossless là 92%/92%/94% cho Full/Reduced/Deleted; với Answer-hint
là 88%/90%/94%. `lossless_prefix_length` trung bình lần lượt là 59.86/60.54/60.96
và 59.72/59.84/62.14 token. Đây vẫn là losslessness so với target continuation
của từng prompt, không phải correctness so với ground-truth answer.

## Các phần còn thiếu

- Thu thập training hidden-state statistics rồi so sánh với VDC/MVBench để
  kiểm tra trực tiếp giả thuyết out-of-distribution của over-attention.
- Chạy follow-up KL có nhiệt độ/logit margin phù hợp để tránh KL saturation.
- Nếu cần kết luận về độ đúng, bổ sung evaluator accuracy cho VDC và MVBench;
  acceptance length không thay thế được accuracy.

## Artifact

- H1b VDC: `results/h1b_h21_20260909/attention_kl_vdc50_part1_0_45.jsonl`,
  `attention_kl_vdc50_part2_45_47.jsonl`, `attention_kl_vdc50_part3_47_50.jsonl`.
- H1b MVBench: `results/h1b_h21_20260909/attention_kl_mvbench_000_100.jsonl`
  đến `attention_kl_mvbench_900_1000.jsonl`.
- H2.1: `results/h1b_h21_20260909/msd_natural_vdc50.jsonl` và
  `results/h1b_h21_20260909/msd_answer_hint_vdc50.jsonl`.
