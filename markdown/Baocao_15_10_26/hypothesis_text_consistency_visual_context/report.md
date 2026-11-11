# Báo cáo thực nghiệm: text consistency của DFlash theo visual context

Ngày cập nhật: 2026-10-15  
Ngày chạy cohort chính: 2026-10-14  
Trạng thái: chưa đủ bằng chứng để xác nhận hoặc bác bỏ H-C

## Tóm tắt

Thực nghiệm kiểm tra liệu giảm explicit visual context có cải thiện tính liên
kết và mạch lạc của output do DFlash tự sinh hay không. Trên 10 prefix target
cố định, Cut làm thay đổi 39.7% proposal so với Full. Tuy vậy, phép chấm
coherence tự động cho điểm sàn ở mọi cặp; acceptance cũng không tăng rõ ở Cut.
Do đó các kết quả hiện có **chưa xác nhận rằng coherence độc lập với visual
context, và cũng chưa cho thấy giảm visual làm coherence tốt hơn**.

Đây là hypothesis riêng về coherence của drafter. Nó không phải kết luận từ
các thí nghiệm trước về attention allocation, answer-hint hay depth. Xem
[thiết kế thực nghiệm](experiment_design.md) để biết protocol và decision rule.

## Thiết kế và phạm vi thực thi

Target là Qwen2.5-VL-3B-Instruct; drafter là DFlash LLaVA68K depth-3. Run dùng
FP16, greedy decoding và tối đa 256 token. Draft sinh tự do, không verification;
target vẫn nhận video đầy đủ và cập nhật features theo prefix draft. Các mức
visual retention trong draft là 100%, 75%, 50%, 25% và 0%.

Kế hoạch ban đầu gồm 50 mẫu VDC. Chỉ 33 mẫu có calibration tại 3.000 visual
tokens nằm trong ngưỡng; 17 mẫu còn lại bị loại trước inference. Không nới
ngưỡng sau khi xem kết quả. Vì thế đây là cohort một phần, không phải full
cohort 50 mẫu.

## Kết quả

| Assay | Kết quả | Ý nghĩa |
|---|---|---|
| Free-running rollout | 165/165 dòng được ghi; không có runtime error. Có text ở 124 output; 41 kết thúc EOS trước khi có text chấm được (Full: 10; Cut: 4). | Coverage không đầy đủ và empty output không đều giữa condition; `rollout_generation_complete=false`, `rollout_study_ready=false`. |
| Full-text coherence | 88/88 cặp được chấm. Tất cả đều 1/5 ở cả hai output trên cả năm chiều, winner `tie`, evidence rỗng. Cut−Full = 0.0, bootstrap 95% CI [0.0, 0.0], `n=22`. | Judge chạm sàn; contrast bằng 0 không chứng minh hai condition tương đương. |
| Matched-prefix coherence | 84/84 cặp tại prefix 128 token đều 1/5/tie. Cut−Full = 0.0, 95% CI [0.0, 0.0], `n=21`. | Kiểm soát độ dài không khắc phục được floor effect. |
| Proposal sensitivity | 50/50 probes trên 10 prefix hoàn tất. Tỷ lệ proposal đổi so với Full: Retain-75 7.2%, Retain-50 13.0%, Retain-25 23.2%, Cut 39.7% (Cut 95% CI 34.9–44.7%). | Có bằng chứng proposal policy nhạy với visual retention; chưa biết token đổi có cần thông tin visual về mặt ngữ nghĩa hay không. |
| Verified acceptance | Exact target-output parity ở 24/33 mẫu trong mỗi condition; 9 mẫu mismatch bị loại khỏi paired analysis. `tau_effective(Cut−Full)` = −0.039 token/vòng, 95% CI [−0.079, 0.003], `n=24`. | Chưa có bằng chứng Cut tăng acceptance; CI chứa 0. |
| Seed-control pilot | Hai video với target-greedy seed chung; hai cặp coherence đều hòa. | Quá nhỏ và cùng bị hạn chế bởi judge floor; chưa xác nhận độ ổn định theo cách khởi đầu. |

Các nhãn token trong sensitivity assay không phân biệt function words với
content words. Run này cũng chưa đo grounding hay answer quality. Retention 0%
chỉ xóa explicit visual rows đã chọn khỏi draft context; text hidden states và
target features vẫn có thể truyền thông tin video.

## Review kết quả

**CONFIRMED** — Các pha tự động chạy xong trên cohort calibration hợp lệ, không
có runtime hoặc lỗi parse. Proposal của drafter phản ứng rõ với retention trên
các prefix được giữ cố định.

**EXPLORATORY** — Sensitivity là chỉ dấu thay đổi policy proposal, không phải
bằng chứng về nhu cầu ngữ nghĩa của token. Ước lượng acceptance Cut−Full hơi âm,
nhưng CI chứa 0. Seed-control chỉ có hai video.

**FAILED / INCOMPLETE** — H-C chưa được xác nhận hay bác bỏ. Chỉ có 33/50 mẫu;
41/165 rollout không có text để chấm; mọi điểm coherence đều ở sàn 1/5 với
evidence rỗng. Chấm mù bằng người chưa hoàn tất.

**HIGHEST VERIFIED RUNG** — R4 smoke end-to-end là rung xanh cao nhất cho
inference workflow. Pilot R5 có chạy nhưng không tạo early coherence signal
dùng được vì judge floor. Cohort chính đã chạy một phần nhưng không đạt R6
xanh vì coverage và primary outcome chưa hợp lệ để ra quyết định.

**EVIDENCE GAPS** — Cần nhãn coherence mù của người, đánh giá calibration và
khả năng phân biệt của judge, cohort đạt đủ calibration gate, quy tắc empty
output định trước, cùng phép đo grounding/answer quality. Tính độc lập không
thể được suy ra từ một contrast không có ý nghĩa thống kê; cần định trước biên
tương đương và dùng phép đo đủ nhạy.

**RECOMMENDED NEXT** — Hoàn tất `human_review_blinded.csv` trước khi mở khóa
unblind. Đối chiếu điểm và độ đồng thuận của người với judge; nếu judge vẫn
chạm sàn, dùng human ratings hoặc sửa rubric trước run xác nhận. Sau đó khóa
cohort, xử lý empty output và biên tương đương; chạy lại paired Full/Cut cùng
các retention trung gian. Báo cáo riêng coherence, grounding và acceptance.

**REPORT LINE** — Drafter proposal nhạy với visual retention, nhưng phép đo
coherence hiện tại không đủ phân biệt điều kiện; chưa có bằng chứng rằng giảm
visual làm tăng coherence hoặc rằng text consistency độc lập với visual context.

## Artifact

- [Run summary](../../../results/dflash_free_running_main_fp16_20261014/report.md)
- [Protocol fingerprint](../../../results/dflash_free_running_main_fp16_20261014/protocol.json)
- [Raw free-running outputs](../../../results/dflash_free_running_main_fp16_20261014/raw_rollouts.jsonl)
- [Raw visual-sensitivity journal](../../../results/dflash_free_running_main_fp16_20261014/raw_sensitivity.jsonl)
- [Raw acceptance journal](../../../results/dflash_free_running_main_fp16_20261014/raw_acceptance.jsonl)
- [Full-text judge pairs](../../../results/dflash_free_running_main_fp16_20261014/judge_pairs.jsonl)
- [Matched-prefix judge pairs](../../../results/dflash_free_running_main_fp16_20261014/judge_matched_prefix_pairs.jsonl)
- [Coherence summary](../../../results/dflash_free_running_main_fp16_20261014/coherence_summary.json)
- [Matched-prefix summary](../../../results/dflash_free_running_main_fp16_20261014/coherence_matched_prefix_summary.json)
- [Visual-sensitivity summary](../../../results/dflash_free_running_main_fp16_20261014/visual_sensitivity_summary.json)
- [Acceptance summary](../../../results/dflash_free_running_main_fp16_20261014/acceptance_summary.json)
- [Blinded human-review sheet](../../../results/dflash_free_running_main_fp16_20261014/human_review_blinded.csv)
