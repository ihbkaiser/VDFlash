# Vanilla Video-DFlash cho Qwen2.5-VL

Repo này triển khai pipeline offline hai giai đoạn để train draft model
**vanilla DFlash** cho `Qwen/Qwen2.5-VL-3B-Instruct`, rồi chạy speculative
inference trên video thật. Draft nhận hidden features của target qua KV
injection, dự đoán đồng thời `block_size - 1` token mask trong một forward và
để target xác minh song song. Pipeline không dùng Sparrow attention, tree
drafting, glimpsing hay video-token selector.

Workflow mặc định:

1. Stage 1: hash-shuffle theo seed rồi lấy 68.000 hội thoại multi-turn thật từ
   [ShareGPT Vicuna unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
2. Stage 2: 68.000 mẫu thật và ảnh từ
   [LLaVA Visual Instruct Pretrain LCS-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
3. Target Qwen2.5-VL được freeze hoàn toàn. Stage 1 dùng nguyên assistant
   response cuối của ShareGPT, không append instruction và không generate lại;
   cache chứa token/position IDs cùng 5 hidden layers đã chọn. Stage 2 vẫn dùng
   raw-greedy target response theo preset multimodal riêng.
4. Chỉ projection, 5 draft decoder layers và draft norms được train.
5. Stage 2 khởi tạo trực tiếp từ final checkpoint của Stage 1.

Preset 8×B200 nằm tại:

- `src/train_VLM/config_b200_8gpu_stage1_sharegpt.json`
- `src/train_VLM/config_b200_8gpu_stage2_llava.json`

Preset chạy đồng thời 3B trên GPU 0–3 và 7B trên GPU 4–7 nằm tại:

- `src/train_VLM/config_b200_4gpu_3b_stage1_sharegpt.json`
- `src/train_VLM/config_b200_4gpu_3b_stage2_llava.json`
- `src/train_VLM/config_b200_4gpu_7b_stage1_sharegpt.json`
- `src/train_VLM/config_b200_4gpu_7b_stage2_llava.json`

Hai run dùng chung manifest/ảnh đã prepare, nhưng có teacher cache, checkpoint
và output riêng. Preset 4 GPU 3B dùng micro-batch 1 × accumulation 16; preset
4 GPU 7B dành cho B200 180 GB dùng micro-batch thật 4 × accumulation 4. Cả hai
giữ global batch 64 records. Preset 7B đồng thời chạy đủ 512 anchors mỗi
forward, pad context tới chiều dài tĩnh và tắt activation checkpointing để đổi
VRAM lấy throughput. `selected_target_layers` để `null` nhằm tự chọn đúng 5
layer theo kiến trúc local 3B hoặc 7B.

Preset bám Appendix A.1 của DFlash: BF16, 6 epochs, LR `6e-4`, cosine schedule,
warmup `0.04`, clip `1.0`, 512 anchors, block 16, loss decay 7, 5 draft layers
và 5 target features. Phase 1 dùng `max_seq_length=2048`,
`teacher_response_mode="dataset"`, `response_max_new_tokens=0`;
`micro_batch_size=1` và `gradient_accumulation_steps=8` cho global batch 64
records trên 8 GPU.

## 1. Cài môi trường

Yêu cầu khuyến nghị: Linux x86_64, Python 3.11, NVIDIA driver tương thích CUDA
12.8. PyTorch 2.7 là stable release đầu tiên có hỗ trợ NVIDIA Blackwell và
wheel CUDA 12.8; repo pin PyTorch 2.7.1, TorchVision 0.22.1 và Transformers
4.57.1 để giữ API Qwen/FlexAttention ổn định.

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Phase 1 có bộ pin riêng đã được xác nhận trên máy RTX 3090/A4000 tại
`src/train_VLM/requirements.txt`. Nếu chạy Phase 1 trên driver hiện tại, dùng:

```bash
python -m pip install -r src/train_VLM/requirements.txt
```

Xác minh môi trường và 8 B200:

```bash
python - <<'PY'
import torch, torchvision, transformers
print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

python -m pytest -q
```

Có thể đặt Hugging Face cache lên ổ dung lượng lớn trước khi download:

```bash
export HF_HOME=/mnt/fast-storage/huggingface
hf auth login  # chỉ cần nếu Hub yêu cầu token
```

## 2. Tải dataset thật

Các revision trong preset được pin để manifest tái lập được. Chạy từ root repo:

```bash
mkdir -p data/video_dflash/raw/sharegpt data/video_dflash/raw/llava

hf download anon8231489123/ShareGPT_Vicuna_unfiltered \
  ShareGPT_V3_unfiltered_cleaned_split.json \
  --repo-type dataset \
  --revision 192ab2185289094fc556ec8ce5ce1e8e587154ca \
  --local-dir data/video_dflash/raw/sharegpt

hf download liuhaotian/LLaVA-Pretrain \
  blip_laion_cc_sbu_558k.json images.zip \
  --repo-type dataset \
  --revision 70f9d1e5e1a697fe35830875cfc7de1dd590d727 \
  --local-dir data/video_dflash/raw/llava
```

`images.zip` của LLaVA khoảng 27 GB. `prepare_data` chỉ extract ảnh thuộc 68K
records đã chọn sang `image_root`; không cần giải nén toàn bộ bằng tay.

Tạo manifest deterministic cho cả hai stage:

```bash
python -m src.train_VLM.prepare_data \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json

python -m src.train_VLM.prepare_data \
  --config src/train_VLM/config_b200_8gpu_stage2_llava.json
```

Mỗi manifest có file `manifest.jsonl.meta.json` ghi Hub commit, SHA-256 của
annotation, seed và chính xác source ID/index của toàn bộ sample đã chọn.
`max_samples=68000` nghĩa là chọn 68K records hợp lệ theo hash-shuffle; đặt
`--max-samples 0` nếu muốn dùng tất cả.

## 3. Smoke test subset nhỏ

Smoke workflow dùng 32 records thật mỗi stage, chạy prepare → teacher cache →
train Stage 1 → train Stage 2 → decode một MP4 thật:

```bash
src/train_VLM/train_3090_smoke.sh
```

Để tạo lại toàn bộ smoke artifacts:

```bash
VIDEO_DFLASH_OVERWRITE=1 src/train_VLM/train_3090_smoke.sh
```

Muốn kiểm tra riêng video decoder trước khi train:

```bash
python -m src.train_VLM.smoke_video \
  --num-frames 4 --size 112 --max-new-tokens 16 --block-size 4
```

## 4. Preprocess/cache teacher features trên 8 B200

Mỗi rank load một frozen target trên GPU riêng, xử lý một phần không trùng nhau
của manifest, ghi shard riêng rồi rank 0 merge index theo thứ tự source.

Preset 3B/4×B200 tối ưu riêng bước này bằng batch 64 trên mỗi GPU, bucket 1.024
sequence theo độ dài, 8 CPU preprocessing workers/rank, shard 32 records và
hàng đợi ghi đĩa nền sâu 2. Target forward gọi thẳng backbone, dừng ngay sau
selected layer cuối và chỉ giữ 5 hidden layers cần cache; LM-head cùng các
decoder layer phía sau không được tính. Nếu batch 64 không vừa VRAM, cache tự
chia đôi batch bị OOM mà không thay đổi dữ liệu đầu ra.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.cache_teacher_features \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json

torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.cache_teacher_features \
  --config src/train_VLM/config_b200_8gpu_stage2_llava.json
```

Chạy riêng 3B trên GPU 0–3:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m src.train_VLM.cache_teacher_features \
  --config src/train_VLM/config_b200_4gpu_3b_stage1_sharegpt.json
```

Cache lưu BF16 losslessly. Với hidden size 2048, 5 features và mọi sequence đều
đạt giới hạn, upper bound xấp xỉ 2,7 TiB cho Stage 1 ở 2048 tokens và 4 TiB cho
Stage 2 ở 3072 tokens; dữ liệu thực thường thấp hơn nhưng vẫn cần dự trù storage
ở quy mô TB. Giảm `max_seq_length`, `max_samples` hoặc `num_target_features` nếu
không đủ dung lượng. Với mode generate của Stage 2, cũng có thể giảm
`response_max_new_tokens`. Những thay đổi này phải nhất quán giữa cache và train.

## 5. Full Stage 1 và exact resume

Full Stage 1:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.train_draft \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json
```

Resume Stage 1 rõ ràng từ `latest`:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.train_draft \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json \
  --resume auto
```

`auto_resume=true` là mặc định của preset, nên chạy lại nguyên lệnh “Full Stage
1” cũng tự tìm `output_dir/checkpoints/latest`. Dùng `--no-auto-resume
--overwrite` chỉ khi chủ động muốn bắt đầu lại từ đầu.

## 6. Full Stage 2 và exact resume

Trong config Stage 2:

```json
"checkpoint": "artifacts/video_dflash_b200_68k/stage1_sharegpt/checkpoint"
```

Đây là weights-only initialization trực tiếp từ final Stage 1; optimizer và
scheduler Stage 2 được tạo mới đúng chủ ý.

Full Stage 2:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.train_draft \
  --config src/train_VLM/config_b200_8gpu_stage2_llava.json
```

Resume Stage 2 dùng optimizer/scheduler/progress của chính Stage 2, không load
lại optimizer Stage 1:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.train_draft \
  --config src/train_VLM/config_b200_8gpu_stage2_llava.json \
  --resume auto
```

## 7. Resume contract và checkpoint layout

`recovery_save_every_steps` điều khiển chu kỳ ghi recovery checkpoint. Giá trị
mặc định `1` fsync sau mỗi optimizer step. Preset 4×B200 7B đặt `50` để việc ghi
model/AdamW nhiều GB không làm GPU chờ ở mọi step; SIGTERM/SIGINT, snapshot,
`max_train_steps` và step cuối vẫn luôn ép commit. Mất điện/SIGKILL có thể quay
lại tối đa 49 step với preset này. Mỗi lần commit đều ghi vào thư mục tạm,
fsync, atomic-rename rồi atomic-update symlink `checkpoints/latest`.

`trainer_state.pt` lưu:

- full draft model nằm trong `model.safetensors`;
- AdamW optimizer và cosine scheduler state;
- `global_step`, epoch và exact offset kế tiếp trong epoch;
- full shuffled sampler permutation và tổng sample đã commit;
- Python, NumPy, Torch CPU và CUDA RNG state riêng cho từng rank;
- history và immutable training contract.

Snapshot dài hạn được tạo mặc định mỗi `save_every_epochs=0.5`; `latest` được
cập nhật ở mỗi recovery commit. `keep_last_checkpoints=3` xoay vòng snapshot
nhưng luôn giữ recovery mới nhất. Batch cuối không bị pad/lặp record: các rank
có thể nhận số record khác nhau và gradient được weighted all-reduce theo số
record thật.

Resume exact sẽ từ chối nếu đổi world size, epochs, sample/cache fingerprint,
batch/accumulation, LR/optimizer/scheduler, seed, block/anchor/loss, dtype hoặc
model/layer selection. Hành vi fail-closed này ngăn reset LR hay train lại một
sample đã commit. Nếu muốn đổi hyperparameter để tạo run mới, dùng checkpoint
cũ qua `--checkpoint`, đổi `--output-dir`, và không dùng `--resume`.

## 8. Inference Video-DFlash trên video thật

Sau khi Stage 2 hoàn tất:

```bash
python -m src.train_VLM.infer_video \
  --checkpoint artifacts/video_dflash_b200_68k/stage2_llava/checkpoint \
  --video /absolute/path/to/video.mp4 \
  --prompt "Describe the important events in this video." \
  --num-frames 8 \
  --video-min-pixels 50176 \
  --video-max-pixels 50176 \
  --max-new-tokens 256 \
  --output artifacts/video_dflash_b200_68k/inference/result.json
```

CLI tự đọc model, block size, draft depth, target layers và processor contract
từ checkpoint. Target xử lý video đúng một lần ở prefill; các vòng sau chỉ dùng
text IDs, M-RoPE positions và target KV cache. Decoder hiện bảo đảm lossless so
với raw-greedy target ở temperature 0.

## 9. Override config/CLI

Mọi tham số chính đều có trong JSON/YAML và có CLI override: model/revision,
dataset/file/image paths, sample count, epochs, LR/AdamW/warmup/clip, micro
batch, gradient accumulation, dtype, sequence/block/anchor sizes, draft depth,
feature count và exact selected layers, loss decay, teacher generation length,
cache/shard/output path, checkpoint cadence, resume và device.

Ví dụ tạo một run 4K records, 2 epochs, global batch 32 trên 8 GPU và output
riêng:

```bash
python -m src.train_VLM.prepare_data \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json \
  --max-samples 4000 \
  --prepared-manifest artifacts/ablation_4k/stage1/manifest.jsonl

torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.cache_teacher_features \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json \
  --prepared-manifest artifacts/ablation_4k/stage1/manifest.jsonl \
  --teacher-cache-dir artifacts/ablation_4k/stage1/teacher_cache

torchrun --standalone --nproc_per_node=8 \
  -m src.train_VLM.train_draft \
  --config src/train_VLM/config_b200_8gpu_stage1_sharegpt.json \
  --teacher-cache-dir artifacts/ablation_4k/stage1/teacher_cache \
  --epochs 2 --micro-batch-size 1 --gradient-accumulation-steps 4 \
  --output-dir artifacts/ablation_4k/stage1/checkpoint
```

Chi tiết kiến trúc, manifest, online legacy trainer và benchmark nằm trong
`src/train_VLM/README.md`.
