# Hướng dẫn sử dụng checkpoint VDFlash/DFlash từ Hugging Face

Tài liệu này hướng dẫn cách lấy lại checkpoint cho các thực nghiệm DFlash mà không cần giữ bản checkpoint lớn trong codebase.

## Kho lưu trữ

Checkpoint và dữ liệu đang nằm trong dataset repository:

<https://huggingface.co/datasets/Tphuc15/vdflash-qwen25vl-3b-checkpoints>

Vì đây là **dataset repository**, mọi lệnh Hugging Face trong tài liệu này phải có `--repo-type dataset`.

Bốn checkpoint đã được kiểm tra từ xa trước khi xoá bản local:

| Biến thể | Dataset | File trên HF | Kích thước | SHA-256 |
|---|---|---|---:|---|
| 20e | LLaVA 68k | `qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt` | 5,688,979,511 bytes | `8b3a3c0a17092f1e46fe9af4db7bd301af88fdd57e1bc4fceca49aee48695b17` |
| 20e | ShareGPT 68k | `qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt` | 5,688,979,511 bytes | `6be2ab5e2aa64820a5a44ade79eb0e220140726852036dbb025ae29b2379b758` |
| 6e | LLaVA 68k | `qwen25vl-3b-dflash-6e-llava68k-latest/training_state.pt` | 5,688,979,511 bytes | `6ec8030d00fed2a20f0be5ae249588622d18ab0777267bfcd4af816c7ddb2755` |
| 6e | ShareGPT 68k | `qwen25vl-3b-dflash-6e-sharegpt68k-latest/training_state.pt` | 5,688,979,511 bytes | `44fc2ff9e16cdd5f99bc6aa909fe679a47c36ede717b076b672f72b0a240cf4e` |

`training_state.pt` là checkpoint PyTorch/SpecForge, chứa `draft_state_dict` cùng trạng thái cần thiết cho việc resume. Bản `training_state.pt` gốc nên được dùng khi cần tiếp tục training; không nên tự đổi tên hoặc chỉ giữ một phần file.

## 1. Đăng nhập Hugging Face

Trên máy chạy thực nghiệm, kiểm tra CLI và đăng nhập bằng tài khoản có quyền đọc repository:

```bash
hf --version
hf auth login
hf auth whoami
```

Nếu repository private, token cần quyền đọc. Không ghi token vào shell script, config commit vào Git, log hoặc chat.

Tài liệu chính thức: [Hugging Face authentication](https://huggingface.co/docs/hub/security-tokens).

## 2. Tải một checkpoint cần dùng

Nên tải checkpoint vào ổ đĩa ngoài codebase, ví dụ `/mnt/fast-storage/vdflash_checkpoints`. Có thể thay đường dẫn này bằng ổ local khác còn đủ dung lượng.

```bash
export HF_REPO="Tphuc15/vdflash-qwen25vl-3b-checkpoints"
export DFLASH_CHECKPOINT_ROOT="/mnt/fast-storage/vdflash_checkpoints"

mkdir -p "$DFLASH_CHECKPOINT_ROOT"

hf download "$HF_REPO" \
  qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt \
  --repo-type dataset \
  --local-dir "$DFLASH_CHECKPOINT_ROOT"
```

File sẽ được đặt tại:

```text
/mnt/fast-storage/vdflash_checkpoints/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt
```

Đổi phần path trong lệnh để lấy biến thể khác:

```text
qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt
qwen25vl-3b-dflash-6e-llava68k-latest/training_state.pt
qwen25vl-3b-dflash-6e-sharegpt68k-latest/training_state.pt
```

Lệnh `hf download` có thể tải một file cụ thể và giữ cấu trúc thư mục từ repository; đây là cách phù hợp cho thực nghiệm chỉ cần checkpoint. Xem thêm [Hugging Face download files](https://huggingface.co/docs/huggingface_hub/guides/download).

## 3. Tải cả bốn checkpoint

Chỉ dùng lệnh này nếu cần so sánh toàn bộ ma trận thực nghiệm. Tổng dung lượng checkpoint khoảng 21.2 GiB, chưa tính cache model và dữ liệu:

```bash
hf download "$HF_REPO" \
  --include 'qwen25vl-3b-dflash-*/training_state.pt' \
  --repo-type dataset \
  --local-dir "$DFLASH_CHECKPOINT_ROOT"
```

Không dùng `hf download "$HF_REPO" --repo-type dataset --local-dir dataset` vì repository còn chứa dataset/video lớn; thao tác đó sẽ đưa toàn bộ dữ liệu trở lại codebase.

Các tuỳ chọn cache là độc lập với nơi lưu checkpoint. Nếu cần chuyển cache Hugging Face sang ổ lớn hơn, đặt biến riêng trước khi chạy:

```bash
export HF_HOME="/mnt/fast-storage/huggingface"
export HF_XET_HIGH_PERFORMANCE=1  # tuỳ chọn, hữu ích khi đường truyền/ổ đĩa đủ nhanh
```

## 4. Kiểm tra file sau khi tải

Ví dụ với checkpoint 20e/LLaVA:

```bash
CHECKPOINT="$DFLASH_CHECKPOINT_ROOT/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt"

test -f "$CHECKPOINT"
stat -c '%n %s bytes' "$CHECKPOINT"
sha256sum "$CHECKPOINT"
```

Kết quả phải có kích thước `5688979511` bytes và SHA-256 tương ứng trong bảng trên. Việc kiểm tra hash hữu ích sau khi copy checkpoint giữa các ổ đĩa.

## 5. Chạy thực nghiệm

Các script hiện tại nhận **đường dẫn local**, chưa nhận trực tiếp Hugging Face repo ID. Vì vậy cần tải file trước, sau đó truyền đường dẫn qua `CHECKPOINT` hoặc `--checkpoint`.

### Validation pipeline

Smoke test trên một mẫu:

```bash
CHECKPOINT="$DFLASH_CHECKPOINT_ROOT/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt" \
  src/analyze/Validate_Sparrow_hypothesises/run_dflash_validation_gpu.sh --limit 1
```

Bỏ `--limit 1` khi chạy toàn bộ validation. Với máy hai GPU và model-parallel:

```bash
CHECKPOINT="$DFLASH_CHECKPOINT_ROOT/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt" \
  bash src/analyze/Validate_Sparrow_hypothesises/run_dflash_model_parallel_2gpu.sh --limit 1
```

### So sánh hai checkpoint

Ví dụ so sánh 20e và 6e trên cùng dataset LLaVA:

```bash
python -m src.infer.qwen25vl_dflash_compare \
  --checkpoint "$DFLASH_CHECKPOINT_ROOT/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt" \
  --checkpoint "$DFLASH_CHECKPOINT_ROOT/qwen25vl-3b-dflash-6e-llava68k-latest/training_state.pt" \
  --limit 1
```

Ma trận tên checkpoint có ý nghĩa như sau:

| Trục | Giá trị | Ý nghĩa |
|---|---|---|
| Số bước training | `6e`, `20e` | checkpoint sau khoảng 6 hoặc 20 epoch, theo tên run |
| Dữ liệu | `sharegpt68k`, `llava68k` | nguồn/phiên bản dataset dùng cho run |

Khi so sánh, nên giữ nguyên model nền, processor, prompt, seed và tập evaluation; chỉ thay checkpoint cần kiểm tra.

## 6. Cấu hình tương thích

Checkpoint hiện được materialize bằng cấu hình draft tương ứng trong codebase:

```text
src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json
```

Không xoá hoặc thay đổi cấu hình này nếu còn cần chạy các checkpoint trên. Cấu hình bao gồm các thông tin như số draft layer, hidden size, vocabulary, mask token và M-RoPE; checkpoint không phải là một model độc lập có thể nạp tuỳ ý bằng mọi cấu hình Qwen2.5-VL.

Checkpoint PyTorch có thể thực thi cơ chế pickle khi nạp. Chỉ nạp file từ repository đáng tin cậy của nhóm; không tải và nạp `training_state.pt` từ nguồn không rõ.

## 7. Quy trình chuẩn cho mỗi thực nghiệm

```text
Chọn biến thể
    -> hf download đúng training_state.pt
    -> sha256sum kiểm tra file
    -> đặt CHECKPOINT/--checkpoint
    -> chạy smoke test --limit 1
    -> chạy full experiment
    -> lưu report/log ở thư mục results hoặc markdown
```

Sau khi thực nghiệm xong, có thể giữ checkpoint trên ổ ngoài để tái sử dụng. Nếu cần giải phóng tiếp ổ ngoài, chỉ xoá thư mục checkpoint cụ thể sau khi chắc chắn không còn job nào dùng nó; lần sau tải lại từ repository HF.

## 8. Khôi phục nhanh sau khi local checkpoint bị xoá

Ví dụ khôi phục checkpoint 20e/ShareGPT:

```bash
export DFLASH_CHECKPOINT_ROOT="/mnt/fast-storage/vdflash_checkpoints"

hf download Tphuc15/vdflash-qwen25vl-3b-checkpoints \
  qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt \
  --repo-type dataset \
  --local-dir "$DFLASH_CHECKPOINT_ROOT"
```

Sau đó dùng:

```bash
CHECKPOINT="$DFLASH_CHECKPOINT_ROOT/qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt" \
  src/analyze/Validate_Sparrow_hypothesises/run_dflash_validation_gpu.sh --limit 1
```

## Ghi chú về phiên bản

- Repository HF là nơi lưu trữ lâu dài; codebase chỉ giữ code và cấu hình cần thiết.
- Không commit checkpoint, raw dataset, token hoặc đường dẫn máy cá nhân vào Git.
- Nếu sau này đổi tên checkpoint hoặc thay đổi cấu hình, cập nhật tài liệu này cùng commit chứa thay đổi đó.
