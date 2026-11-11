# DFlash VisualCut: báo cáo kết quả phân phối

## Tóm tắt

Bỏ visual positions khỏi DFlash context làm thay đổi phân phối proposal; mức thay đổi tăng theo visual budget. Tuy nhiên, target-hit@1 trên toàn bộ draft positions chỉ chênh tối đa 0,74 điểm phần trăm giữa full và cut. Trên tập vị trí full-hit, cut vẫn chọn cùng token target ở 75–87% vị trí, nhưng JSD có điều kiện cũng tăng theo budget.

Phân phối và τ-effective là hai đại lượng khác nhau. Kết quả cho thấy chúng có thể thay đổi không cùng mức; chưa chứng minh full/cut tương đương performance hay bên nào thắng tổng thể.

## Thiết kế và thống kê

So sánh hai chế độ **full visual/cut visual** trên cùng checkpoint balanced DFlash depth-3 (target layers `[1, 9, 17, 25, 33]`) và target Qwen2.5-VL-3B. Dùng cùng cohort 50 video VDC50 ở bốn nhãn visual budget 400, 3.000, 13.000, 25.000: 200 cặp mẫu–budget. Cut bỏ visual positions khỏi DFlash context; target vẫn nhận full visual context. Giới hạn sinh là 256 token.

Hai điều kiện replay trên cùng target-greedy sequence, draft positions và block layout. Tính trung bình vị trí trong từng video trước; CI 95% dùng bootstrap ghép cặp cấp 50 video, 2.000 lần, seed `20261102`. Logits full-vocabulary chỉ giữ trong RAM; artifact không lưu chúng hoặc target hidden states.

## Toàn bộ draft positions

JSD là Jensen–Shannon divergence trên softmax full-vocabulary ở T=1, log cơ số 2. JSD đối xứng và nằm trong `[0, 1]` bit; giá trị không phải phần trăm. Top-1 changed là tỷ lệ argmax full/cut khác nhau; top-100 overlap là số token chung trong hai top-100 chia 100. Entropy đo độ phân tán tổng thể của mỗi phân phối.

| Visual budget | JSD [CI 95%], bit | Top-1 changed | Top-100 overlap | Entropy full / cut, bit | Target-hit@1 full → cut; Δ [CI 95%] |
|---:|---:|---:|---:|---:|---:|
| 400 | 0,0526 [0,0448; 0,0619] | 24,0% | 85,2% | 5,294 / 5,298 | 21,97% → 21,23%; −0,74 điểm [−1,05; −0,45] |
| 3.000 | 0,0766 [0,0713; 0,0819] | 33,7% | 79,7% | 5,536 / 5,546 | 19,73% → 19,24%; −0,49 điểm [−0,71; −0,27] |
| 13.000 | 0,1187 [0,1121; 0,1254] | 43,0% | 72,1% | 5,748 / 5,766 | 18,81% → 18,87%; +0,06 điểm [−0,30; +0,44] |
| 25.000 | 0,1522 [0,1451; 0,1590] | 49,6% | 67,4% | 5,820 / 5,752 | 18,31% → 19,02%; +0,71 điểm [+0,36; +1,06] |

## Phân phối cut trên các vị trí full visual được accept

Ở đây `full accepted` được định nghĩa theo replay: argmax full bằng token greedy target tại draft position. Sau đó đo cut tại đúng các vị trí ấy. Đây là per-position target-hit, không phải accepted-prefix của decode tự do.

| Visual budget | Tỷ lệ vị trí full-hit | Cut cũng hit trong tập full-hit [CI 95%] | JSD có điều kiện [CI 95%], bit | Top-100 overlap | Xác suất target full / cut |
|---:|---:|---:|---:|---:|---:|
| 400 | 21,97% | 86,98% [85,15; 88,54] | 0,0409 [0,0350; 0,0482] | 86,9% | 0,537 / 0,516 |
| 3.000 | 19,73% | 81,65% [80,66; 82,58] | 0,0598 [0,0553; 0,0648] | 81,7% | 0,509 / 0,489 |
| 13.000 | 18,81% | 77,93% [76,72; 79,06] | 0,0907 [0,0843; 0,0972] | 74,7% | 0,492 / 0,472 |
| 25.000 | 18,31% | 74,83% [73,40; 76,27] | 0,1122 [0,1054; 0,1194] | 70,6% | 0,486 / 0,472 |

## Phân tích và insight

- **Khác biệt tăng theo budget:** JSD tăng 0,0526 → 0,1522 bit; top-1 changed tăng 24,0% → 49,6%; top-100 overlap giảm 85,2% → 67,4%.
- **Độ tập trung tổng thể khá gần, nhưng các token trong head bị sắp xếp lại.** Entropy của full/cut gần nhau ở ba budget đầu và chênh 0,068 bit ở 25.000; trong khi top-1 và top-100 overlap thay đổi rõ. Vì vậy không nên diễn giải kết quả như cut chỉ làm phân phối phẳng hơn.
- **Trong các vị trí full-hit**, cut giữ cùng token target ở 86,98% → 74,83% vị trí. JSD điều kiện thấp hơn JSD toàn bộ, nhưng vẫn tăng 0,0409 → 0,1122 bit.
### Vì sao phân phối đổi nhưng τ-effective có thể gần nhau?

Báo cáo balanced trước ghi nhận Δτ-effective (cut−full) là −0,007, −0,101, −0,013 và +0,020. Đây là số liệu của thí nghiệm trước; τ-effective đo lượng token được accept theo block, còn JSD đo chênh lệch xác suất trên toàn vocabulary. Vì vậy JSD tăng không bắt buộc τ-effective giảm tương ứng.

Replay hiện tại cho thấy các lần target-hit chỉ thuộc về một điều kiện khá ít và đổi chiều theo budget:

| Visual budget | Chỉ full-hit | Chỉ cut-hit | Δ target-hit@1 cut−full trong replay | Δτ-effective ở báo cáo trước |
|---:|---:|---:|---:|---:|
| 400 | 2,78% | 2,04% | −0,74 điểm | −0,007 |
| 3.000 | 3,61% | 3,12% | −0,49 điểm | −0,101 |
| 13.000 | 4,13% | 4,19% | +0,06 điểm | −0,013 |
| 25.000 | 4,62% | 5,33% | +0,71 điểm | +0,020 |

Đây là tỷ lệ vị trí trung bình theo video; “hit” nghĩa là argmax tại vị trí đó trùng target, không phải accepted-prefix tự do. Các chênh lệch hit nhỏ và có lúc full, lúc cut nhỉnh hơn. Điều này **giải thích được vì sao hai kết quả có thể cùng đúng**, nhưng chưa chứng minh cơ chế làm τ-effective gần nhau: replay cố định tiền tố và không đo accepted-prefix/throughput tự do. Cũng chưa có bằng chứng rằng một điều kiện thắng ở khía cạnh hiệu năng khác.
- KL là metric có hướng: `KL(full || cut)` trả lời cut lệch thế nào nếu full là teacher; đây là hướng phù hợp cho distillation full → cut. JSD được dùng ở đây vì đối xứng. Không thể tính KL full-vocabulary chính xác từ artifact hiện có vì chỉ lưu top-100 và JSD, không lưu toàn bộ logits/xác suất.

## Hướng tiếp theo: self-distillation xuống 10% visual

Kết quả tạo động lực để thử, chưa chứng minh sẽ thành công. Condition cut hiện tại bỏ 100% visual positions khỏi DFlash context; giữ 10% là điều kiện mới, không thể nội suy tuyến tính từ kết quả này.

Thiết kế đề xuất: đóng băng full-visual DFlash làm teacher; khởi tạo student từ cùng checkpoint, giữ text và 10% visual positions được lấy mẫu phân tầng theo frame. Distill logits tại cùng target prefixes/mask positions bằng `KL(full || student-10%)`, kèm loss target-token để giữ target-hit. Dùng tập train tách khỏi VDC50; đánh giá VDC50 bằng JSD, target-hit và free-running accepted-prefix/τ-effective, throughput, peak memory. Teacher logits cần được tạo on-the-fly hoặc chạy lại; top-100 artifact hiện tại không đủ cho KL chính xác. Target vẫn dùng full visual context, nên mức tiết kiệm compute nằm chủ yếu ở DFlash.

## Giới hạn thực nghiệm

Full run hoàn tất 200 cặp, không skip; analyzer tái chạy khớp summary Colab. Run dùng A100-SXM4-80GB; cảnh báo H.264 và sai khác source/bytecode runtime được ghi trong `artifact_audit.json`. Replay cố định target trajectory, nên chưa đo accepted-prefix tự do, latency hay throughput. Ngân sách là nhãn calibration; số visual positions thực tế được lưu trong pair artifacts.
