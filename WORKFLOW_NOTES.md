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

## 6. Inference tiếp theo từ checkpoint MSD DFlash

- Archive Hugging Face `ihbkaiser/qwen25vl-3b-msd-checkpoints/outputs.tar.zst`
  được lưu tại `artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints/outputs.tar.zst`
  và giải nén dưới
  `artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints/extracted/outputs/`.
- Thực nghiệm inference sắp tới dùng checkpoint `latest` của **Phase 2** cho
  cả bốn cấu hình DFlash 3-layer. Mỗi `latest` hiện trỏ tới step 6375:
  - `dflash_3layer_26_28_30_32_34` (`high_only`)
  - `dflash_3layer_17_19_25_33_34` (`middle_high`)
  - `dflash_3layer_17_28_30_32_34` (`middle_high_2`)
  - `dflash_3layer_1_28_30_32_34` (`shallow_high`)
- Đường dẫn mỗi checkpoint là
  `artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints/extracted/outputs/<config>/phase2/qwen25vl-3b-dflash-llava68k-<config>/qwen25vl-3b-dflash-llava68k-<config>-latest`;
  `latest` là symlink tới thư mục `*-step6375`.
- Phase 1 (`*-step6129`) là checkpoint ShareGPT trước khi fine-tune Phase 2;
  giữ lại cho resume hoặc phân tích, không dùng làm checkpoint cuối mặc định
  cho inference hai giai đoạn.
- Checkpoint là training state SpecForge (`training_state.pt` cùng state của
  rank 0 và rank 1), không phải thư mục export HF độc lập. Chọn loader/exporter
  phù hợp trước khi chạy inference; giữ nguyên archive gốc và thư mục đã giải
  nén.

## 7. Môi trường inference chung trên GPU Gadi

- Dùng môi trường `.venv` tại thư mục gốc repository; trong PBS script gọi
  trực tiếp `.venv/bin/python`, không tạo một venv riêng cho từng queue.
- Pin Torch `2.7.1+cu126` và torchvision `0.22.1+cu126` trong
  `requirements.txt`. CUDA 12.6 giữ hỗ trợ Volta (V100, SM 7.0) và cũng phục
  vụ Ampere (A100) cùng Hopper (H200) trong cùng môi trường.
- Không dùng wheel cu128 hiện tại cho V100: lần chạy ngày 2026-10-27 load được
  model nhưng lỗi `no kernel image is available` tại Qwen M-RoPE vì Torch wheel
  không có kernel SM 7.0. Khi thay Torch/CUDA wheel dùng chung, hold các job GPU
  đang chờ rồi release sau khi kiểm tra import và `pip check`.
- Queue Gadi dùng trong smoke inference: `gpuvolta` cho V100 (12 CPU/GPU),
  `dgxa100` cho A100 (16 CPU/GPU), và `gpuhopper` cho H200 (12 CPU/GPU). Xác
  nhận GPU thực bằng `nvidia-smi` trong log job, không chỉ dựa vào tên queue.
- Khi chạy cùng sample trên nhiều GPU, ghi report riêng theo GPU; report phải
  có `success=true`, đủ bốn config và `outputs_match=true` trước khi kết luận.
- Smoke một sample ngày 2026-10-27: H200 (`report_h200.json`) cho
  `outputs_match=true` ở cả bốn config, ESR 4.02–4.86x và DSR 1.42–1.73x.
  V100 cu126 (`report.json`) và A100 (`report_a100.json`) chạy hết decode nhưng
  trả `outputs_match=false` ở cả bốn config, nên chỉ dùng H200 cho thực nghiệm
  chính cho tới khi mismatch trên V100/A100 được giải thích. Kết quả này là
  sanity check trên một sample, không phải đánh giá chất lượng tổng quát.

## 8. VisualCut và attention campaign

- Dùng `calibrate_visualcut_vdc50` để đo số visual token từ `video_grid_thw`
  bằng cùng `nframes`, `min_pixels`, `max_pixels` như inference. Calibration
  version và sample fingerprint được lưu để resume không tái dùng phép đo cũ
  khi video hoặc quy trình thay đổi.
- Worker `run_dflash_visualcut_attention` ghi ma trận case dự kiến vào
  `run_config.json`; analyzer dùng ma trận này để kiểm tra đủ case theo số
  config, target, chunk và sample limit của campaign.
- Raw attention lưu thành shard PyTorch nén bằng chương trình dòng lệnh `zstd`.
  Cần có `zstd` trên node chạy; PBS script kiểm tra binary trước khi tải model.
  Shard và journal nằm dưới `results/`, không đưa artifact vào Git.
- Khi submit PBS, đặt `TARGET_MODEL`, `CHECKPOINT_ROOT` và `HF_HOME` trong môi
  trường job; không ghi đường dẫn checkpoint hoặc cache cá nhân vào script.
- Sau pilot, chạy `validate_dflash_visualcut_pilot` để kiểm tra sáu điều kiện,
  SHA256, số record, key map và tính lossless; khi phân tích campaign đầy đủ,
  dùng `analyze_dflash_visualcut_attention --require-complete`.

## 9. VisualCut distribution replay

- Distribution replay so sánh hai điều kiện `full` và `cut` trên cùng target
  greedy trajectory, anchor, mask layout và ranh giới block. Draft dùng kiến trúc
  3-layer với target layers `[1, 9, 17, 25, 33]`; config mẫu trỏ tới checkpoint
  `h32-depth3` cùng layer selection. Visual context chỉ bị loại khỏi DFlash.
- Dùng `run_dflash_distribution_replay` để chạy; pilot dùng config mẫu
  `configs/dflash_distribution_colab.pilot.example.json` (một mẫu, 25K), campaign
  đầy đủ dùng `configs/dflash_distribution_colab.full.example.json` (50 mẫu, bốn
  ngân sách). Sửa các đường dẫn theo mount trên Colab trước khi chạy.
- Trên Colab CLI, gọi `scripts/run_dflash_distribution_colab.sh CONFIG.json`.
  Launcher upload source bundle/config; dữ liệu và checkpoint đọc từ Google Drive.
  Giữ output trên Drive để resume và không lưu logits vocabulary đầy đủ.
- Runner nhận checkpoint qua `checkpoint` tường minh hoặc cặp
  `checkpoint_root`/`checkpoint_name`; `draft_config` phải khớp layer selection.
- Pair artifact gzip lưu target greedy IDs, visual positions, input/checkpoint
  fingerprints, metric theo draft position và top-100. Raw target hidden và raw
  full-vocabulary logits chỉ tồn tại trong RAM.
- Sau đủ 200 pair, `analyze_dflash_distribution_replay` tính trung bình vị trí
  trong video rồi bootstrap ghép cặp 2.000 lần ở cấp sample, dùng chung resample
  sample IDs qua bốn ngân sách. Artifact nằm dưới `results/` hoặc output directory
  được cấu hình và không đưa vào Git.

## 10. Text-State Feature Capture & Matched-Path Protocol

- Khi thực hiện kiểm định nhân quả hoặc can thiệp feature (như nghiên cứu
  Text-State Intervention), phải thu thập feature của cả hai nhánh $G$
  (chuẩn) và $A$ (can thiệp) qua **cùng một lượt forward teacher-forced
  duy nhất** trên chuỗi `[prompt_ids, answer_ids]` với `use_cache=False`.
- Tuyệt đối không so sánh feature trích xuất từ cached prefill phân mảnh của
  vòng lặp decode với full teacher-forced forward pass để tránh sai lệch số học
  ở các vị trí text trước visual (`previsual`).
- Hook can thiệp trên `target.model.get_video_features` phải hỗ trợ cả
  Transformers 4.x (trả về tuple tensor) và Transformers 5.x (trả về
  `BaseModelOutputWithPooling` với thuộc tính `pooler_output`).
- Khi mở rộng `input_ids` và `attention_mask` cho chuỗi token đầu ra,
  phải đồng thời mở rộng `mm_token_type_ids` (điền 0 cho các token text)
  để tránh lỗi lệch kích thước tensor trong phép tính RoPE 3D.
