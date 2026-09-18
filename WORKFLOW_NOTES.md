# Workflow Notes

File này ghi nhớ các quy trình cần tuân thủ khi tiếp tục làm việc trong
repository VDFlash.

## 1. Nguồn checkpoint Hugging Face

- `Tphuc15/qwen25vl-3b-depth1-depth3`: checkpoint H3.2 cho DFlash depth1 và
  depth3.
- Hugging Face **dataset** `Tphuc15/vdflash-qwen25vl-3b-checkpoints`:
  checkpoint DFlash 5-layer, gồm
  các biến thể 6e/20e cho LLaVA68K và ShareGPT68K.
- Không ghi token Hugging Face vào file, log hoặc hội thoại. Nếu repository
  gated, dùng credential local đã được cấp quyền hoặc đường dẫn checkpoint
  local.

## 2. Quy trình H3.2 depth study

- Giữ cố định target Qwen2.5-VL-3B, cohort, prompt, video preprocessing và
  calibration khi so sánh depth.
- Mỗi depth chạy ba điều kiện:
  - `Full`: giữ visual positions và visual values.
  - `Zero`: giữ positions, đặt visual values bằng 0.
  - `Cut`: loại visual positions khỏi draft context.
- Metric chính là `accepted_effective_tokens` (`tau_effective`), tức số token
  hiệu dụng trung bình mỗi vòng xác minh; metric này đã bao gồm một target
  bonus token.
- Contrast chính của H3.2 là:

  ```text
  (Full - Zero)_depth3 - (Full - Zero)_depth1
  ```

- Khi thêm depth5, phải chạy lại bằng đúng runner H3.2 hiện tại trước khi đưa
  vào hình. Không ghép trực tiếp các artifact legacy vì khác preprocessing,
  target-output contract hoặc calibration.
- `accepted_prefix_tokens`/`tau_proposal` không cộng 1; nếu cần metric số token
  hiệu dụng thì dùng `tau_effective = tau_proposal + 1` trên valid rounds.

## 3. Báo cáo và hình

- Báo cáo giao cho người dùng nằm ở
  `markdown/Baocao_17_9_26/bao_cao_tong_hop_thuc_nghiem_hypothesis_20260910.md`.
- Hình giao kèm báo cáo nằm ở `markdown/Baocao_17_9_26/figures/`.
- Artifact số liệu gốc nằm trong `results/`; không sửa hoặc xóa artifact gốc
  khi chỉ cần cập nhật báo cáo.
- Mọi diễn giải phải phân biệt acceptance/alignment với answer accuracy và
  không gọi `accepted_prefix_tokens` là accuracy.
- Error bars bootstrap 95% CI ở cấp sample; CI chứa 0 nghĩa là chưa có bằng
  chứng rõ ràng rằng contrast khác 0.

## 4. Quản lý dung lượng và xóa checkpoint

- Trước khi xóa, kiểm tra chính xác đường dẫn, kích thước, nội dung và không có
  job đang dùng file.
- Không xóa checkpoint local chỉ vì muốn đổi sang repository khác nếu chưa
  xác nhận tên thư mục cụ thể.
- Khi dung lượng thấp, ưu tiên tải checkpoint cần dùng vào thư mục tạm hoặc ổ
  ngoài; chạy xong mới dọn đúng file tạm.
- Sau khi xóa, kiểm tra lại `du`, `df` và xác nhận repository Hugging Face vẫn
  là nguồn khôi phục.

## 5. Kiểm tra trước khi kết luận

- Kiểm tra coverage, `status=ok`, runtime errors, target-output hash và sample
  pairing trước khi phân tích.
- Không tuyên bố hoàn tất nếu chưa chạy verification tương ứng.
- Với thay đổi code, chạy test tập trung trước; với thay đổi report, kiểm tra
  toàn bộ link markdown và sự tồn tại của hình/artifact được tham chiếu.
