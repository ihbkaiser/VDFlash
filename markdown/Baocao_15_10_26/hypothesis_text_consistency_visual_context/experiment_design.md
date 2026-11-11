# Thiết kế thực nghiệm: text consistency theo visual retention

Ngày khóa protocol chính: 2026-10-14  
Model: Qwen2.5-VL-3B-Instruct + DFlash LLaVA68K depth-3  
Dataset: VDC50, câu hỏi mở

## Câu hỏi và hypothesis

**Câu hỏi chính:** text consistency của chuỗi do draft model tự sinh thay đổi
thế nào khi giảm explicit visual context được đưa vào drafter?

**H-C:** giảm explicit visual context có thể cải thiện độ liên kết và mạch lạc
của output do drafter tự sinh. Đây là hypothesis về coherence, không đồng nhất
với draft–target agreement, acceptance length hay answer accuracy.

**H-A, outcome phụ:** visual retention có thể làm thay đổi acceptance trong
speculative decoding chuẩn. Đo riêng outcome này; không gọi acceptance là
coherence.

**H-S, tiền đề cần đo:** chỉ một phần nhỏ vị trí proposal có thể nhạy với visual
context. Không giả định trước rằng các vị trí đó là content words hay token có
ý nghĩa thị giác thiết yếu.

Không quan sát thấy khác biệt có ý nghĩa thống kê chưa đủ để khẳng định
coherence độc lập với visual context. Muốn kiểm tra tính tương đương/độc lập,
cần hiệu chuẩn phép đo với người chấm, định trước biên tương đương có ý nghĩa
thực tế và đủ cỡ mẫu cho kiểm định tương đương.

## Intervention và điều kiện

Giữ cố định model, checkpoint, prompt, video, preprocessing, target layers,
block size và greedy decoding. Thay đổi tỷ lệ explicit visual hidden rows mà
drafter giữ lại:

| Điều kiện | Visual rows trong draft |
|---|---:|
| Full | 100% |
| Retain-75 | 75% |
| Retain-50 | 50% |
| Retain-25 | 25% |
| Cut | 0% |

Target luôn nhận video đầy đủ. Khi drafter sinh tiếp, target xử lý prefix draft
mới để cập nhật target features; target logits không chọn hoặc sửa token của
free-running output. Vì vậy Cut loại explicit visual rows đã chọn khỏi draft
context, nhưng không đảm bảo mọi thông tin video biến mất khỏi text hidden
states hoặc target features.

## Free-running coherence study

- VDC50 dùng câu hỏi tự nhiên; không đưa reference answer vào prompt sinh hoặc
  prompt chấm coherence.
- Mỗi condition sinh tối đa 256 token bằng FP16 và greedy decoding, không
  verification, rollback hay target bonus token.
- Output tự sinh là đối tượng chấm coherence. Ghi EOS, số token, cap-hit, số
  câu và trạng thái output rỗng.
- LLM judge chỉ thấy text, không thấy video, reference, checkpoint hay retention;
  thứ tự cặp được che và đảo. Chấm grammar/continuation, referential consistency,
  discourse coherence, repetition và overall coherence (1–5). Yêu cầu evidence
  span khi có lỗi lớn.
- Chấm grounding/task relevance riêng với video. Coherence cao không đảm bảo
  câu trả lời đúng hoặc có grounding.

### Contrast và decision rule đã khóa

Contrast chính là chênh lệch paired overall coherence `Cut − Full` ở cấp video;
bootstrap 95% CI cũng lấy video làm đơn vị. H-C chỉ được hỗ trợ nếu contrast
dương, CI trên 0, coherence scoring đã được so sánh với đánh giá mù của người
và lợi ích không chỉ do output ngắn/chung chung. CI chứa 0 là chưa đủ bằng chứng.

So sánh Retain-75/50/25 với Full là phụ. Nếu dùng để quyết định hypothesis,
cần điều chỉnh nhiều so sánh hoặc xác nhận trên cohort độc lập.

## Assay phụ

### Visual sensitivity của proposal

Trên 10 prefix target-greedy cố định sinh từ Full, teacher-force cùng prefix
qua năm retention conditions. Đo tỷ lệ proposal greedy thay đổi so với Full.
Đây là phép đo độ nhạy policy proposal, không xác định token nào cần visual
evidence về mặt ngữ nghĩa. Phân loại function words và content words chưa nằm
trong protocol hiện tại.

### Acceptance

Chạy speculative decoding chuẩn trên cùng các điều kiện và cohort. Metric chính
là `tau_effective`, kèm `tau_proposal`; chỉ đưa vào paired analysis các dòng có
exact target-output parity. Kết quả acceptance không thay thế coherence score.

## Giới hạn cỡ mẫu và trạng thái thực thi

Kế hoạch là 50 video. Chỉ đưa vào run các mẫu có calibration tại 3.000 visual
tokens nằm trong ngưỡng đã định. Run ngày 2026-10-14 có 33 mẫu hợp lệ; 17 mẫu
khác bị loại trước inference, không nới calibration gate sau khi xem kết quả.
Đây là run một phần của cohort dự kiến, không phải xác nhận đủ 50 mẫu.

Pilot dùng hai video và tối đa 128 token; không pool pilot với study chính. Có
thêm seed-control dùng một token target-greedy chung ở Full/Cut; loại token seed
khỏi phần text được chấm.
