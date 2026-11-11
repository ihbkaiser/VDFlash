# Báo cáo Hypothesis 3.3: MSD trên image captioning S–V–O

**Ngày:** 2026-10-12  
**Trạng thái:** phân tích lại artifact pilot đã có; không chạy lại mô hình trong lần tổng hợp này.

## Tóm tắt kết luận

Thí nghiệm hiện có cho thấy khi drafter giảm visual retention từ 100% xuống 0%, `accepted_prefix_tokens` tăng trung bình **0,173 tokens/round** trên 50 ảnh. Khoảng bootstrap paired 95% là **[0,085; 0,262]**. Vì vậy, artifact này ủng hộ **chiều hướng** “0% visual có acceptance length cao hơn” trong cohort đã chạy, nhưng không tái hiện mức tăng **3,7–4,3 tokens/round** nêu trong giả thuyết.

Kết quả đó **không xác nhận cơ chế** “tập trung hơn vào text làm câu coherent hơn”. Bộ log không có acceptance theo token/vai trò S–V–O, không đo coherence, và mỗi mẫu chỉ yêu cầu một câu nên không thể đo coherence *giữa các câu*. Phần dự đoán visual giúp draft tốt hơn ở danh từ/tính từ cụ thể cũng chưa kiểm tra được. Ngoài ra, hai ảnh được kiểm tra thủ công có caption đầu ra sai rõ so với ảnh và caption tham chiếu.

**Phán quyết:** H3.3 được hỗ trợ một phần ở metric acceptance tổng hợp; các giải thích về local semantics, function words và coherence chưa được kiểm chứng. Không nên dùng kết quả này để kết luận rằng xóa visual cải thiện độ đúng hoặc độ coherent của caption.

## 1. Giả thuyết và câu hỏi đo lường

H3.3 gồm ba mệnh đề có thể quan sát riêng:

1. **Local semantics:** giữ visual giúp drafter dự đoán tốt hơn token mang nội dung hình ảnh như danh từ, động từ và tính từ; vì vậy acceptance ở các token đó phải cao hơn tại 100% so với 0% retention.
2. **Global acceptance:** bỏ visual có thể tăng acceptance trên toàn chuỗi nhờ draft-target agreement tốt hơn ở từ chức năng và khung câu.
3. **Cơ chế coherence:** mức tăng global acceptance xảy ra vì drafter tập trung vào text và tạo chuỗi coherent hơn.

Artifact hiện tại đo được mệnh đề thứ hai ở dạng hẹp: **số proposal token được target chấp nhận trung bình mỗi verification round**. Nó không trực tiếp đo answer accuracy, semantic accuracy theo vai trò, coherence, hay throughput thực tế theo tokens/second. Theo quy ước của repo, `tau_effective = accepted_prefix_tokens + 1` vì mỗi round còn có target bonus token; `accepted_prefix_tokens` tự nó không bao gồm bonus.

## 2. Thiết kế và dữ liệu

| Thành phần | Thiết lập trong artifact |
|---|---|
| Dữ liệu | 50 ảnh COCO-derived; mỗi mẫu có reference caption và trường `subject`, `verb`, `object` |
| Prompt | `Describe this image in one concise sentence.` |
| Điều kiện | 100% và 0% visual retention, ghép cặp trên cùng `sample_id` |
| Output cap | `max_new_tokens=32` theo mặc định của runner |
| Model mặc định của runner | Qwen2-VL-7B-Instruct làm base/target; MSD-Qwen2VL-7B-Instruct làm drafter |
| Vai trò visual context | Target verification luôn giữ full visual context; retention thay đổi draft context |
| File kết quả | 100 dòng JSONL trong hai điều kiện |

S–V–O ở đây là metadata từ caption tham chiếu. Prompt không yêu cầu mô hình phải dùng cấu trúc S–V–O cụ thể, nên không thể mặc định mọi output đều hiện thực hóa các span S/V/O đã gán.

Thiết kế target-full/draft-retained có ý nghĩa khi diễn giải: acceptance đo mức **drafter khớp với target đang thấy đầy đủ ảnh**, không đo khả năng target nhìn ảnh hay độ đúng caption cuối cùng.

## 3. Kiểm tra độ đầy đủ và tính hợp lệ

- Manifest có **50/50 `sample_id` duy nhất**; result JSONL có **100/100 cặp** `sample_id × retention`, gồm 50 dòng mỗi điều kiện. Không có cặp thiếu hoặc trùng.
- Cả 50 mẫu đều có subject, verb và object; cả 50 đường dẫn ảnh còn tồn tại trong môi trường hiện tại.
- `accepted_prefix_tokens` của từng dòng khớp với trung bình các giá trị trong `acceptance_trace` tương ứng.
- Mỗi dòng lưu `role_stats` cho S/V/O/F, nhưng **tất cả denominator `total_tokens` đều bằng 0**. Không có tỷ lệ acceptance S/V/O có thể diễn giải. Đây không có nghĩa acceptance thật bằng 0: trace vị trí/token đã không được ghi nhận vào các thống kê này.
- Result JSONL không lưu `acceptance_by_position`, `status`, `error`, model/checkpoint ID, hash của target output hay seed cấu hình. Vì vậy không thể audit lại phân loại token, provenance chính xác hoặc trạng thái lỗi từng dòng chỉ từ file này.
- Kiểm tra bổ sung cho thấy runtime có giữ trace trong bộ nhớ, nhưng runner vừa đọc nhầm khóa `position_records` (runtime chỉ trả `accepted_proposal_positions` và `fallback_position`) vừa không serialize `acceptance_by_position` vào JSONL. Vì vậy các `role_stats` bằng 0 là lỗi thu thập trace, không phải acceptance thực bằng 0.
- Runner có hàm phân vai token dùng phép kiểm tra substring (`token_text in phrase`). Ngay cả khi khôi phục được trace, cách này cần thay bằng đối sánh tokenizer/span có kiểm tra, nếu không token con hoặc chuỗi con có thể bị gán nhầm vai trò.

### Spot-check caption và ảnh

Hai trường hợp được đối chiếu trực tiếp với ảnh, reference caption và output:

| Sample | Reference và ảnh | Output MSD |
|---|---|---|
| `coco_277912` | Ảnh có người đang trượt ván; reference nói người đàn ông trượt ván trên phố. | Cả hai retention mô tả một bức tranh trừu tượng. |
| `coco_406732` | Ảnh có hươu cao cổ trong rừng; reference nói hươu cao cổ đi trong rừng. | 100% mô tả hộp trà; 0% cũng mô tả hộp trà và dừng giữa câu. |
| `coco_183939` | Ảnh có một con chó đen đang nhảy trong công viên; reference nói chó nhảy lên bắt đồ chơi. | Cả hai retention mô tả thuyền, nước và bầu trời; 100% thêm từ `floating`. |

Đây là spot-check nhỏ, không phải ước lượng lỗi trên toàn bộ 50 ảnh. Tuy nhiên, nó cho thấy output chất lượng caption không thể được giả định là đúng chỉ vì có acceptance cao. Output 0% của `coco_406732` bị cắt giữa câu; điều này phù hợp với giới hạn 32 token của runner.

## 4. Kết quả acceptance

Metric chính là mean accepted proposal tokens mỗi round. Chênh lệch được ghép theo ảnh và định nghĩa là `0% − 100%`.

| Metric | 100% retention | 0% retention | Chênh lệch paired `0% − 100%` | Bootstrap paired 95% CI |
|---|---:|---:|---:|---:|
| `accepted_prefix_tokens` | 2,359 (SD 0,496) | 2,533 (SD 0,455) | **+0,173** | **[+0,085; +0,262]** |
| `tau_effective` = proposal + target bonus | 3,359 | 3,533 | **+0,173** | **[+0,085; +0,262]** |

Trong 50 cặp, 0% retention có acceptance length cao hơn ở **29** mẫu, thấp hơn ở **8**, và bằng nhau ở **13**. Median paired difference là **+0,111 tokens/round**. Khoảng CI được tính bằng paired percentile bootstrap: lấy mẫu lại 50 chênh lệch theo ảnh 50.000 lần, seed `20261012`. Đây là khoảng bất định mô tả cho cohort này; chưa có đăng ký trước phương pháp suy luận hay lần lặp độc lập.

Mức tăng quan sát được là nhỏ: **+0,17**, không phải bước tăng **+3,7 đến +4,3 tokens/round**. Nếu khoảng 3,7–4,3 trong giả thuyết được hiểu là mức `tau_effective` tuyệt đối thay vì mức tăng, mean của điều kiện 0% ở đây là **3,53**, cũng không khớp chính xác; hai phép đo chỉ nên so sánh trực tiếp khi định nghĩa metric và cấu hình runner giống nhau.

### Phân bố số proposal được chấp nhận trong một round

`acceptance_trace` có các giá trị từ 0 đến 5. Bảng dưới đây đếm verification rounds gộp từ 50 ảnh mỗi điều kiện.

| Proposal được chấp nhận trong round | 100%: số round (% trong 430) | 0%: số round (% trong 406) |
|---:|---:|---:|
| 0 | 49 (11,4%) | 36 (8,9%) |
| 1 | 85 (19,8%) | 75 (18,5%) |
| 2 | 122 (28,4%) | 108 (26,6%) |
| 3 | 100 (23,3%) | 101 (24,9%) |
| 4 | 20 (4,7%) | 25 (6,2%) |
| 5 | 54 (12,6%) | 61 (15,0%) |

Nếu tạm định nghĩa round dài là có ít nhất 4 proposal được chấp nhận, tỷ lệ là **17,2%** ở 100% và **21,2%** ở 0%; bucket 5 token lần lượt là **12,6%** và **15,0%**. Do giả thuyết chưa định nghĩa ngưỡng “chuỗi dài”, các số này chỉ mô tả trace và không phải so sánh chính thức với phát biểu “0,2%”. Chúng cũng không cho biết các token được chấp nhận là danh từ, tính từ, từ nối hay mạo từ.

## 5. Thời gian và output cuối

| Metric | 100% retention | 0% retention | Chênh lệch paired `0% − 100%` | Bootstrap paired 95% CI |
|---|---:|---:|---:|---:|
| End-to-end seconds/sample | 0,821 (SD 0,157) | 0,778 (SD 0,139) | −0,042 s | [−0,059; −0,026] |

Đây là phép đo batch size 1, chưa đủ điều kiện để kết luận speedup hay throughput thực. Runner luôn chạy 100% trước 0% cho từng ảnh; do thứ tự cố định, hiệu ứng warm-up/cache có thể góp phần tạo chênh lệch thời gian. Không có run lặp hoặc hoán đổi thứ tự điều kiện.

Output text giống hệt giữa hai điều kiện ở **48/50** ảnh; hai cặp khác nhau là `coco_183939` và `coco_406732`. Đây là kiểm tra độ ổn định chuỗi output giữa điều kiện, không phải kiểm tra độ đúng. Artifact không ghi hash để xác nhận target continuation; hơn nữa output khác nhau ở một số cặp cần được giải thích trong một lần chạy có seed và logging đầy đủ.

## 6. Đối chiếu từng phần của H3.3

| Mệnh đề | Kết quả hiện tại | Đánh giá |
|---|---|---|
| Bỏ visual làm acceptance length tăng | +0,173 tokens/round; CI paired 95% nằm trên 0 | **Có hỗ trợ về chiều hướng** trong pilot 50 ảnh; hiệu ứng nhỏ. |
| Bỏ visual tạo bước nhảy +3,7–4,3 tokens/round | Chênh lệch đo được +0,173 | **Không được tái hiện** trong artifact này. |
| 100% visual cải thiện acceptance cho danh từ/tính từ/động từ cụ thể | Không có token positions/role denominators; không có nhãn tính từ | **Chưa kiểm chứng.** |
| 0% cải thiện từ chức năng và khung câu | Không có nội dung token được chấp nhận theo vị trí/role | **Chưa kiểm chứng.** |
| Lợi thế đến từ coherence giữa các câu | Một mẫu chỉ tạo một câu; không có coherence metric | **Không được kiểm tra bởi thiết kế hiện tại.** |
| Caption cuối chính xác hơn hoặc kém chính xác hơn | Không có COCO caption score; spot-check có sai lệch rõ | **Không thể suy ra từ acceptance.** |

### Phân tích reject-token ở hai cặp output khác nhau

Trace xác nhận được một round `k=0` ở mỗi retention của `coco_183939`, một round `k=0` ở `coco_406732` 0%, và không có round `k=0` ở `coco_406732` 100%. Đây là số reject **chắc chắn tối thiểu**, không phải tổng reject: 11 round ở mỗi điều kiện của `coco_406732` và 5 round ở mỗi điều kiện của `coco_183939` có `k=1..4`, nhưng log không cho biết nhánh được chọn còn proposal sau prefix đã nhận hay không. Các round `k=5` là nhận hết nhánh dài nhất, nên token tiếp theo là bonus chứ không phải reject.

| Sample | 100% visual retention | 0% visual retention | So với thông tin visual trong ảnh |
|---|---|---|---|
| `coco_183939` | Round 7 có reject xác nhận; target token thay thế gần từ `in` trong “in the background”. | Round 7 cũng có reject xác nhận; target token thay thế gần `in`. | Hai vị trí thay thế gần từ chức năng. Không có bằng chứng ở đây rằng một điều kiện reject danh từ/tính từ visual nhiều hơn. Proposal draft thực sự bị loại không được lưu. |
| `coco_406732` | Không có round `k=0`, nhưng 11 round còn mơ hồ; không thể kết luận reject ít hơn. | Round 6 có reject xác nhận; target token thay thế gần `pot` trong “tea pot”. | Vị trí bề mặt gần một danh từ cụ thể, nhưng proposal bị loại không biết là gì; target vẫn mô tả hộp trà trong khi ảnh có hươu cao cổ. Vì vậy không thể diễn giải đây là draft 0% dự đoán sai danh từ `pot`. |

Ở `coco_183939`, output 100% có thêm `floating`, đồng thời round 3 nhận thêm một proposal so với 0%. Đây là liên hệ bề mặt đáng kiểm tra lại, nhưng không xác định được token `floating` có phải proposal được nhận hay không nếu thiếu token IDs; hơn nữa mô tả thuyền không khớp ảnh chó đang nhảy. Chi tiết cách tính index và các round chưa phân loại nằm ở Phụ lục A.

### Kết luận cho H3.3

Kết quả phù hợp với phần dự đoán hẹp rằng **giảm visual context ở draft có thể làm tăng draft-target acceptance một chút**. Chênh lệch paired dương trong cohort này, nhưng không xác định vì sao acceptance tăng. Không có bằng chứng trực tiếp rằng draft chuyển chú ý sang text, rằng từ chức năng được chấp nhận nhiều hơn, hoặc rằng câu trở nên coherent hơn. Vì mỗi câu là một mô tả độc lập, dữ liệu không thể trả lời mệnh đề coherence giữa các câu.

Do đó, kết luận tổng hợp là **H3.3 chưa được xác nhận; chỉ có bằng chứng pilot cho chiều hướng tăng acceptance tổng hợp khi giảm visual retention**. Mệnh đề local semantics không bị bác bỏ hay xác nhận; nó chưa được đo.

## 7. Thiết kế follow-up để kiểm tra cơ chế

1. **Cố định target continuation:** dùng cùng target output hoặc teacher-forced target tokens cho các retention condition. Ghi seed, tokenizer/model revision, hash input và target output.
2. **Sửa và kiểm tra instrumentation:** ghi cho từng verification round token ID/text, target position, accepted/fallback flag và `acceptance_by_position`. Dừng run nếu role denominator bằng 0 thay vì ghi tỷ lệ 0%.
3. **Gán role theo span/tokenizer:** căn chỉnh các span S/V/O và tính từ trong reference hoặc answer target với token IDs; phân loại riêng function words. Kiểm tra thủ công một tập mẫu trước khi chạy full cohort.
4. **Kiểm định local semantics:** so paired acceptance rate theo vai trò, đặc biệt danh từ/tính từ thị giác ở 100% so với 0%; bootstrap ở cấp ảnh và báo CI cho từng nhóm. Không dùng output cuối để thay cho acceptance ở từng vị trí.
5. **Kiểm định global coherence:** nếu giả thuyết vẫn nói về coherence *giữa câu*, tạo tác vụ có 2–3 câu liên kết về cùng ảnh/sự kiện. Đo coherence độc lập với acceptance (ví dụ đánh giá mù bởi người chấm hoặc metric được định trước). Với dữ liệu một câu hiện tại, chỉ nên đặt câu hỏi về coherence bên trong câu.
6. **Đo throughput đúng nghĩa:** thêm warm-up, hoán đổi/ngẫu nhiên thứ tự điều kiện, lặp nhiều lần và báo tokens/second cùng paired latency. Giữ cùng GPU/process và tách chi phí prefill khỏi decode.
7. **Đo chất lượng caption riêng:** chấm caption theo reference/image bằng metric captioning hoặc đánh giá mù; báo cáo metric này tách biệt hoàn toàn với `accepted_prefix_tokens`.

## 8. Artifact và provenance

- [Runner](../../scripts/run_image_captioning_msd_svo.py)
- [Manifest 50 ảnh](../../dataset/image_captioning_svo_50.jsonl)
- [Kết quả raw JSONL](../image_captioning_msd_svo/msd_image_captioning_svo_50.jsonl)
- [Runtime MSD retention](../../src/analyze/Validate_Sparrow_hypothesises/runtime.py)
- [Quy ước diễn giải metric trong workflow](../../WORKFLOW_NOTES.md)

Model trong bảng là model được khai báo làm mặc định trong runner; file kết quả không ghi lại CLI arguments thực tế. Mọi so sánh với thí nghiệm video hoặc cohort/checkpoint khác cần được xem là tham chiếu ngoài phạm vi, trừ khi metric, target-output contract và preprocessing được làm đồng nhất.

## Phụ lục A. Audit vị trí reject cho hai cặp output khác nhau

Phụ lục này chỉ dùng trace đã lưu cho `coco_183939` và `coco_406732`; không chạy lại model. Chỉ số vị trí bên dưới là chỉ số token trong answer, bắt đầu từ 0, được khôi phục từ số token mỗi round. Nó không phải chỉ số từ/cụm từ trong caption.

### A.1. Diễn giải đúng trace theo vòng

`acceptance_trace[r] = k` là số proposal draft được target chấp nhận trong round `r`. Khi update, EAGLE nối vào answer một token target làm root của round và tiếp theo là `k` proposal đã được chấp nhận. Nếu round bắt đầu tại answer index `s`, các vị trí là:

- Token root target: `s`.
- Draft proposal được chấp nhận: `s+1` đến `s+k`.
- Token target kế tiếp: `s+k+1`; token này trở thành root ở round sau nếu vòng lặp tiếp tục.

Một lần reject chỉ xảy ra nếu nhánh draft được chọn còn proposal sau `k` proposal đã chấp nhận. Cấu hình runner đặt `depth=4`; EAGLE tạo lớp top-k đầu tiên rồi mở rộng thêm bốn lớp, nên nhánh dài nhất có năm proposal. Do đó `k=0` xác nhận ít nhất proposal đầu tiên bị reject; `k=5` là chấp nhận hết nhánh dài nhất và token kế tiếp là bonus. Với `k=1..4`, trace hiện tại không lưu độ dài nhánh được chọn, nên không phân biệt được reject với việc nhánh draft đã kết thúc.

| Mẫu | Retention | `acceptance_trace` theo round | Round reject xác nhận được | Vị trí token target thay thế (0-based) | Round chấp nhận đủ 5 proposal | Round chưa phân loại (`k=1..4`) |
|---|---:|---|---|---:|---|---:|
| `coco_183939` | 100% | `4, 2, 3, 5, 1, 3, 0, 5` | 7 | 25 | 4, 8 | 5 |
| `coco_183939` | 0% | `4, 2, 2, 5, 3, 1, 0, 5` | 7 | 24 | 4, 8 | 5 |
| `coco_406732` | 100% | `3, 1, 2, 1, 4, 1, 3, 1, 2, 2, 1, 5` | Không có round `k=0` | — | 12 | 11 |
| `coco_406732` | 0% | `3, 1, 2, 3, 2, 0, 4, 1, 1, 2, 1, 2` | 6 | 17 | Không có | 11 |

Đây là **số reject chắc chắn tối thiểu**: một ở mỗi điều kiện của `coco_183939`, không có vị trí chắc chắn ở `coco_406732` 100%, và một ở `coco_406732` 0%. Những round có `k=1..4` có thể bổ sung reject, nên không thể coi các con số này là tổng reject.

### A.2. Vị trí trong văn bản và vai trò visual

Không có tokenizer hoặc token ID của output trong artifact; vì vậy ghép chỉ số token sang từ dưới đây chỉ là đối chiếu gần đúng bằng cách đếm từ hiển thị. Token draft bị reject cũng không được lưu. Văn bản ở vị trí tương ứng là **token target thay thế**, không phải nội dung proposal draft đã bị reject.

| Mẫu / retention | Round | Vị trí answer | Từ hiển thị gần vị trí thay thế (ước lượng) | Diễn giải |
|---|---:|---:|---|---|
| `coco_183939` / 100% | 7 | 25 | `in` trong “in the background” | Vị trí target thay thế gần từ chức năng; identity/role của proposal bị reject không biết. |
| `coco_183939` / 0% | 7 | 24 | `in` trong “in the background” | Cũng gần từ chức năng; identity/role của proposal bị reject không biết. |
| `coco_406732` / 0% | 6 | 17 | Gần `pot` trong “tea pot” | Vị trí target thay thế gần danh từ bề mặt; proposal bị reject không được lưu, còn target fallback vẫn sai với ảnh hươu cao cổ. |
| `coco_406732` / 100% | — | — | Không có round `k=0` | Chưa thể kết luận không có reject: 11 round có `k=1..4` còn mơ hồ. |

Ở `coco_183939`, output 100% có thêm `floating`, trong khi round 3 có ba proposal được chấp nhận so với hai ở 0%. Hai quan sát này tương thích với khả năng 100% chấp nhận thêm một token quanh cụm đó, nhưng không chứng minh được ánh xạ token vì thiếu tokenizer/ID. Quan trọng hơn, “thuyền nổi trên nước” không khớp ảnh chó đang nhảy; đây không phải bằng chứng rằng visual retention cải thiện độ đúng ngữ nghĩa.

Ở `coco_406732`, cả output đều mô tả hộp trà thay vì hươu cao cổ; 0% còn bị cắt sau “different”. Vị trí reject xác nhận ở 0% gần `pot`, nhưng target fallback tại chỗ đó cũng không sửa được lỗi nhận diện cảnh. Do đó, ngay cả vị trí bề mặt có vẻ là danh từ visual cũng không đủ chứng minh draft 100% tốt hơn hoặc draft 0% tệ hơn ở local semantics.

### A.3. Giới hạn instrumentation và kết luận audit

Hàm `acceptance_round_record()` hiện gán accepted proposal bắt đầu ở `s` và fallback tại `s+k`. Theo thứ tự token thực tế của EAGLE, token ở `s` là target root, nên accepted proposals bắt đầu ở `s+1` và token target kế tiếp ở `s+k+1`. Runner hiện cũng đọc `position_records`, trong khi runtime không phát trường đó, rồi bỏ `acceptance_by_position` khỏi JSONL. Vì vậy không dùng các trường position/role hiện tại để kết luận vị trí chính xác.

Không thể chạy lại hai mẫu trong môi trường audit này: NVIDIA driver không khả dụng (`nvidia-smi` không kết nối được) và checkpoint base Qwen2-VL-7B mặc định cùng tokenizer không có trong cache. CSV bên dưới lưu phép đổi trace thành chỉ số token đã khôi phục, đồng thời đánh dấu các round chắc chắn, bonus và chưa phân loại:

- [CSV audit vị trí theo round](reject_position_audit_two_samples_20261012.csv)

**Kết luận:** dữ liệu lưu lại chỉ cho phép xác nhận ba reject tối thiểu trên bốn traces; hai vị trí của token target thay thế gần từ chức năng `in`, một vị trí gần danh từ bề mặt `pot`, nhưng không biết nội dung của bất kỳ proposal draft nào bị reject. Các round còn lại chưa đủ metadata để gọi là reject. Chưa có bằng chứng đáng tin để kết luận 100% hay 0% tốt hơn tại các từ cần thông tin visual.
