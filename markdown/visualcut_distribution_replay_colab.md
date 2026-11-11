# DFlash VisualCut Distribution Replay trên Colab CLI

Runner dùng DFlash 3-layer với target layers `[1, 9, 17, 25, 33]` và ghép cặp hai
điều kiện `full`/`cut` ngay trong cùng vòng decode. Config mẫu trỏ tới checkpoint
`h32-depth3` có cùng kiến trúc và layer selection. Target sinh greedy sequence
một lần cho mỗi mẫu–ngân sách. Ở mỗi draft block, `cut` nhận cùng anchor, mask
layout, target prefix và ranh giới block mà `full` vừa dùng. Chỉ context của
DFlash bỏ visual positions; target vẫn nhận prompt visual đầy đủ.

Logits full-vocabulary của hai điều kiện chỉ nằm trong RAM cho block hiện tại.
Runner tính JSD và metric phân phối ngay sau khi có cả hai lượt, rồi chỉ ghi
metric theo vị trí cùng top-100 IDs/logits/probabilities. Pair artifact cũng ghi
target greedy IDs, visual positions, fingerprint input, checkpoint fingerprint
và trạng thái ghép cặp. Pair chưa hoàn chỉnh không được analyzer đưa vào báo cáo.

## 1. Chuẩn bị Colab CLI và dữ liệu

Cài [Google Colab CLI](https://github.com/googlecolab/google-colab-cli), rồi đặt
dataset, calibration, cohort, target model và checkpoint ở Google Drive để runtime
đọc được. CLI chạy mã cục bộ trên VM từ xa; launcher dưới đây upload source bundle
và config, trong khi dữ liệu lớn được đọc từ Drive.

Sửa một bản sao của `configs/dflash_distribution_colab.pilot.example.json` hoặc
`configs/dflash_distribution_colab.full.example.json` để các đường dẫn trỏ đúng
tệp trên Drive. `target_model` phải là thư mục model Qwen2.5-VL-3B mà runtime
Colab đọc được. Config mẫu trỏ `checkpoint` tới file DFlash H3.2 depth-3:

```text
<Drive>/VDFlash/artifacts/Tphuc15_qwen25vl-3b-depth1-depth3/
  qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt
```

Checkpoint trên máy hiện tại có cùng target-layer selection `[1, 9, 17, 25, 33]`
với `balanced.json`. Model và checkpoint cần nằm trên Drive hoặc đường dẫn mà
runtime Colab truy cập được; launcher chỉ upload source bundle và run config.

Launcher mặc định tạo session H100 high-memory, mount Drive, upload source/config,
cài các dependency đã pin trong `scripts/requirements_colab_dflash_distribution.txt`,
rồi chạy runner. Session được giữ lại để chạy pilot trước và full campaign sau.

## 2. Pilot kỹ thuật 25K

Đặt đường dẫn thực tế trong bản config pilot rồi chạy từ terminal ở repository:

```bash
scripts/run_dflash_distribution_colab.sh /path/to/dflash_distribution_pilot.json
```

Pilot dùng một mẫu ở 25.000 visual tokens và giới hạn sinh 256 tokens. Kiểm tra
`run_config.json`, pair JSONL gzip và log runtime trên Google Drive trước khi tạo
config full. Nếu session đã tồn tại và Drive đã mount, dùng lại session:

```bash
COLAB_CREATE_SESSION=0 COLAB_MOUNT_DRIVE=0 \
COLAB_SESSION=vdflash-distribution \
scripts/run_dflash_distribution_colab.sh /path/to/dflash_distribution_pilot.json
```

## 3. Campaign VDC50

Dùng config full với `budgets: [400, 3000, 13000, 25000]`, `cohort_size: 50`,
`sample_limit: null`, `max_new_tokens: 256` và một `output_dir` mới. Chạy tiếp trên
session đã dùng cho pilot:

```bash
COLAB_CREATE_SESSION=0 COLAB_MOUNT_DRIVE=0 \
COLAB_SESSION=vdflash-distribution \
scripts/run_dflash_distribution_colab.sh /path/to/dflash_distribution_full.json
```

Runner kiểm tra đủ cùng 50 sample có calibration `ok` ở tất cả ngân sách trước khi
nạp model. Các pair hoàn chỉnh được resume theo file trong `output_dir/pairs/`.
Không dùng lại thư mục output nếu thay đổi checkpoint, target, cohort, ngân sách,
calibration hoặc giới hạn sinh.

Config full tự chạy analyzer sau khi đủ 200 pair. Nếu cần chạy lại thống kê:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_distribution_replay \
  --run-dir /path/to/output_dir --bootstrap-reps 2000 --seed 20261102
```

Analyzer tính trung bình vị trí trong từng video trước, rồi bootstrap ghép cặp 95%
ở cấp sample; cùng chỉ số sample được lấy lại cho cả bốn ngân sách trong mỗi lần
bootstrap. Kết quả nằm trong `analysis/summary.json`, `summary.csv` và
`sample_means.csv`.

## 4. CLI lifecycle

Để dừng session sau khi campaign hoàn tất:

```bash
colab stop -s vdflash-distribution
```

Các lệnh `colab new`, `colab drivemount`, `colab upload`, `colab exec` và
`colab stop` theo CLI chính thức của Google. `colab exec -f` truyền source của
script bootstrap; launcher upload riêng source bundle và config để các module dự
án có mặt trên VM.


## 5. Kết quả thực nghiệm hoàn tất

Full run VDC50 × 4 ngân sách đã hoàn tất. Báo cáo cuối, summary Colab, kết quả tái phân tích và kiểm toán artifact được lưu trong [folder báo cáo DFlash VisualCut](DFlash_VisualCut_Distribution_20261102/README.md). Các pair artifacts đầy đủ và log vẫn nằm trong `results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/`.
