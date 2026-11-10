
# Hugging Face Checkpoints & Repositories

Tổng hợp các repository và checkpoint trên Hugging Face phục vụ các thực nghiệm DFlash và MSD (Multi-modal Speculative Decoding) trên mô hình nền tảng Qwen2.5-VL / Qwen2-VL.

---

## 1. DFlash Checkpoints

### 1.1. DFlash Depth 1 & Depth 3 (Nghiên cứu H3.2 Depth Study)

* **Repository:** [`Tphuc15/qwen25vl-3b-depth1-depth3`](https://huggingface.co/datasets/Tphuc15/qwen25vl-3b-depth1-depth3)
* **Loại repo (`repo_type`):** `dataset`
* **Mô hình nền:** `Qwen/Qwen2.5-VL-3B-Instruct`
* **Các file checkpoint:**
  * `qwen25vl-3b-dflash-llava68k-h32-depth1-latest/` (Depth 1, LLaVA-68K, `step 6375`, epoch 6)
  * `qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt` (Depth 3, LLaVA-68K, `step 6375`, epoch 6)
  * `qwen25vl-3b-dflash-sharegpt68k-h32-depth1-latest/` (Depth 1, ShareGPT-68K, `step 6129`, epoch 5)
  * `qwen25vl-3b-dflash-sharegpt68k-h32-depth3-latest/training_state.pt` (Depth 3, ShareGPT-68K, `step 6129`, epoch 5)
* **Lệnh tải mẫu:**
  ```bash
  hf download Tphuc15/qwen25vl-3b-depth1-depth3 \
    qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt \
    --repo-type dataset \
    --local-dir checkpoints/
  ```

### 1.2. DFlash 5-Layer (Depth 5 - Biến thể 6e / 20e)

* **Repository:** [`Tphuc15/vdflash-qwen25vl-3b-checkpoints`](https://huggingface.co/datasets/Tphuc15/vdflash-qwen25vl-3b-checkpoints)
* **Loại repo (`repo_type`):** `dataset`
* **Mô hình nền:** `Qwen/Qwen2.5-VL-3B-Instruct`
* **Các file checkpoint:**
  * `qwen25vl-3b-dflash-6e-llava68k-latest/training_state.pt` (5 layers, 6 epochs)
  * `qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt` (5 layers, 20 epochs)
  * `qwen25vl-3b-dflash-6e-sharegpt68k-latest/training_state.pt` (5 layers, 6 epochs)
  * `qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt` (5 layers, 20 epochs)
* **Lệnh tải mẫu:**
  ```bash
  hf download Tphuc15/vdflash-qwen25vl-3b-checkpoints \
    qwen25vl-3b-dflash-6e-llava68k-latest/training_state.pt \
    --repo-type dataset \
    --local-dir checkpoints/
  ```

### 1.3. DFlash 3-Layer Layer Selection (Archive `outputs.tar.zst`)

* **Repository lưu trữ:** [`ihbkaiser/qwen25vl-3b-msd-checkpoints`](https://huggingface.co/datasets/ihbkaiser/qwen25vl-3b-msd-checkpoints)
* **Tên file archive:** `outputs.tar.zst`
* **Loại repo (`repo_type`):** `dataset`
* **Mô hình nền:** `Qwen/Qwen2.5-VL-3B-Instruct`
* **Mô tả:** Archive toàn bộ checkpoint huấn luyện cho **4 cấu hình Layer Selection của DFlash 3-layer**, gồm cả Phase 1 (`step 6129` - ShareGPT-68K) và Phase 2 (`step 6375` - LLaVA-68K):
  * `dflash_3layer_26_28_30_32_34` (`high_only`)
  * `dflash_3layer_17_19_25_33_34` (`middle_high`)
  * `dflash_3layer_17_28_30_32_34` (`middle_high_2`)
  * `dflash_3layer_1_28_30_32_34` (`shallow_high`)
* **Đường dẫn sau khi giải nén:**
  `artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints/extracted/outputs/<config>/phase2/qwen25vl-3b-dflash-llava68k-<config>/qwen25vl-3b-dflash-llava68k-<config>-latest/training_state.pt`
* **Lệnh tải archive:**
  ```bash
  hf download ihbkaiser/qwen25vl-3b-msd-checkpoints \
    outputs.tar.zst \
    --repo-type dataset \
    --local-dir artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints/
  ```

---

## 2. DSpark 3-Layer Checkpoints

* **Repository:** [`ihbkaiser/qwen25vl-3b-msd-checkpoints`](https://huggingface.co/datasets/ihbkaiser/qwen25vl-3b-msd-checkpoints)
* **Archive:** [`dspark_3layer.zst`](https://huggingface.co/datasets/ihbkaiser/qwen25vl-3b-msd-checkpoints/blob/main/dspark_3layer.zst) (`repo_type=dataset`)
* **Nội dung đã xác nhận trong archive:** hai checkpoint Phase 2, LLaVA-68K, step `6372`:
  * `balanced`: layers `[1, 9, 17, 25, 33]`
  * `high-only`: layers `[26, 28, 30, 32, 34]`
* **Tải archive:**
  ```bash
  hf download ihbkaiser/qwen25vl-3b-msd-checkpoints \
    dspark_3layer.zst \
    --repo-type dataset \
    --local-dir artifacts/ihbkaiser_qwen25vl-3b-msd-checkpoints/
  ```

## 3. MSD Checkpoints (Qwen2.5-VL-3B)

* **Repository:** [`ihbkaiser/qwen25vl-3b-msd-checkpoints`](https://huggingface.co/datasets/ihbkaiser/qwen25vl-3b-msd-checkpoints)
* **Loại repo (`repo_type`):** `dataset`
* **Mô hình nền:** `Qwen/Qwen2.5-VL-3B-Instruct`
* **Các file checkpoint drafter MSD:**
  * `qwen25vl-3b-msd-depth3-latest.zip` (MSD Depth 3 Drafter, `global_step=80000`, epoch 37 — dùng cho thực nghiệm so sánh Depth 3 vs 5 và thực nghiệm Attention Layer 18)
    * SHA256: `f14da72e8e650732f4d1bb42b412612bcae637b593539a503a1a6ac9bf96b8ac`
  * `qwen25vl-3b-msd-depth5-latest.zip` (MSD Depth 5 Drafter, `global_step=48000`, epoch 22)
    * SHA256: `24ac7b4de1d0330007e4a4d662aebc026a546d384de4f811a151c208eb2a02fc`
* **Lệnh tải / giải nén tự động:**
  ```bash
  python scripts/download_msd_checkpoints.py --output-dir dataset/qwen25vl-3b-msd-checkpoints --extract
  ```

---

## 4. Author Baseline (MSD Qwen2-VL-7B)

* **Repository:** [`lucylyn/MSD-Qwen2VL-7B-Instruct`](https://huggingface.co/lucylyn/MSD-Qwen2VL-7B-Instruct)
* **Loại repo (`repo_type`):** `model`
* **Mô hình nền:** `Qwen/Qwen2-VL-7B-Instruct`
* **Mục đích:** Checkpoint chính thức từ tác giả bài báo MSD dùng để chạy benchmark đối chứng trên Colab GPU L4.

---

## HF Authentication

Use the local Hugging Face CLI credential; do not store token values in this file.

