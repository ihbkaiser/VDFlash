#!/usr/bin/env bash
# ==============================================================================
# Pipeline Tự Động Toàn Diện: DFlash Visual-10% Self-Distillation trên B200
# 
# Quy trình tự động 1 lệnh duy nhất:
#   TẢI ASSETS -> TIỀN XỬ LÝ -> CACHE 5 LAYERS -> TEST SMOKE 15 STEPS -> TRAIN FULL -> EVAL
#
# Cách sử dụng:
#   1. Mặc định (Test xong tự động chuyển sang Train Full):
#        bash scripts/run_visual10_selfdistill_b200.sh
#
#   2. Chỉ chạy thử Smoke Test (15 steps) rồi dừng:
#        bash scripts/run_visual10_selfdistill_b200.sh --smoke-only
#
#   3. Bỏ qua Smoke Test, chạy thẳng Full Training:
#        bash scripts/run_visual10_selfdistill_b200.sh --no-smoke
#
#   4. Tùy biến tham số (số GPU hoặc đường dẫn):
#        NUM_GPUS=2 DATA_ROOT=/data/dflash bash scripts/run_visual10_selfdistill_b200.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- 1. Thiết Lập Đường Dẫn Mặc Định ---
PYTHON_BIN="${PYTHON_BIN:-$(which python3)}"
if [[ -f "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/dflash-posttraining}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${REPO_ROOT}/checkpoints}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/models}"

RAW_DIR="${DATA_ROOT}/raw"
FEATURES_DIR="${DATA_ROOT}/features"
TEACHER_DIR="${CHECKPOINTS_ROOT}/phase2-dflash"
SMOKE_OUTPUT_DIR="${CHECKPOINTS_ROOT}/dflash-visual10/smoke_test"
FULL_OUTPUT_DIR="${CHECKPOINTS_ROOT}/dflash-visual10/run_final_b200"

if [[ -d "${MODELS_ROOT}/Qwen2.5-VL-3B-Instruct" ]]; then
  TARGET_MODEL="${TARGET_MODEL:-${MODELS_ROOT}/Qwen2.5-VL-3B-Instruct}"
else
  TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
fi
TARGET_CONFIG="${REPO_ROOT}/configs/target_qwen25vl_3b.json"
DRAFT_ARCH_CONFIG="${REPO_ROOT}/configs/dflash_configs/high_only.json"
TEACHER_CKPT="${TEACHER_DIR}/qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt"

# Tự động nhận diện số lượng GPU
if command -v nvidia-smi &> /dev/null; then
  DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  DETECTED_GPUS=1
fi
NUM_GPUS="${NUM_GPUS:-${DETECTED_GPUS}}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  NUM_GPUS=1
fi

DO_SMOKE=1
DO_FULL=1
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

for arg in "$@"; do
  case "$arg" in
    --smoke|--smoke-only)
      DO_SMOKE=1
      DO_FULL=0
      ;;
    --no-smoke)
      DO_SMOKE=0
      DO_FULL=1
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      ;;
  esac
done

# Thiết lập số mẫu xử lý: Nếu chỉ chạy smoke-only thì rút gọn để test nhanh
if [[ "${DO_FULL}" -eq 0 ]]; then
  TRAIN_RECORDS=50
  VAL_RECORDS=10
  TEST_RECORDS=10
else
  TRAIN_RECORDS=3000
  VAL_RECORDS=250
  TEST_RECORDS=250
fi

echo "========================================================================"
echo ">> Video-DFlash Visual-10% Self-Distillation Automation Pipeline"
echo ">> Kế hoạch: Smoke Test [${DO_SMOKE}] -> Full Production Train [${DO_FULL}]"
echo ">> Số GPU B200: ${NUM_GPUS} | Python: ${PYTHON_BIN}"
echo ">> Data Root: ${DATA_ROOT}"
echo ">> Checkpoints Root: ${CHECKPOINTS_ROOT}"
echo "========================================================================"

mkdir -p "${DATA_ROOT}" "${RAW_DIR}" "${FEATURES_DIR}" "${TEACHER_DIR}" "${MODELS_ROOT}" "${SMOKE_OUTPUT_DIR}" "${FULL_OUTPUT_DIR}"

# Tự động đồng bộ đường dẫn target model vào config JSON
DYNAMIC_TARGET_CONFIG="${DATA_ROOT}/target_config.json"
cat <<EOF > "${DYNAMIC_TARGET_CONFIG}"
{
  "target_model": "${TARGET_MODEL}",
  "mixed_precision": "bf16",
  "max_seq_length": 32768,
  "response_max_new_tokens": 1024
}
EOF
TARGET_CONFIG="${DYNAMIC_TARGET_CONFIG}"

# --- 2. Kiểm Tra Môi Trường, Authentication & Dependencies ---
echo ""
echo ">> [1/5] Kiểm tra môi trường & cài đặt gói SpecForge..."

# Kiểm tra xác thực Hugging Face cho repo private (chỉ khi không bật --skip-download)
if [[ "${SKIP_DOWNLOAD}" != "1" && ! -f "${RAW_DIR}/download_status.json" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami &>/dev/null; then
    echo ">> [CẢNH BÁO] Repo 'Tphuc15/dflash-posttraining-raw' là dataset private."
    echo ">> Chưa phát hiện thông tin đăng nhập Hugging Face. Vui lòng đăng nhập trước bằng lệnh:"
    echo ">>    hf auth login   (hoặc: export HF_TOKEN=...)"
    echo ">> Hoặc nếu đã tải dữ liệu trước đó, hãy thêm cờ: --skip-download"
  fi
fi

if ! "${PYTHON_BIN}" -c "import specforge" &> /dev/null; then
  echo ">> Cài đặt SpecForge ở chế độ editable (dùng cờ --no-deps để tránh xung đột wheel)..."
  "${PYTHON_BIN}" -m pip install --no-deps -e src/train_Dflash_SpecForge
fi

"${PYTHON_BIN}" -c "
import torch
print('>> PyTorch:', torch.__version__, '| CUDA khả dụng:', torch.cuda.is_available())
from torch.nn.attention.flex_attention import flex_attention
print('>> FlexAttention backend: SẴN SÀNG')
"

# --- 3. Tải Dữ Liệu & Checkpoint Từ Hugging Face (Bỏ qua nếu bật --skip-download) ---
echo ""
echo ">> [2/5] Kiểm tra tài nguyên dữ liệu & checkpoints..."

if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  echo ">> [BỎ QUA TẢI TỪ HF] Đã bật cờ --skip-download (sử dụng tài nguyên đã tải trước)."
  echo ">> Kiểm tra file cục bộ:"
  echo ">>   - Teacher Checkpoint: ${TEACHER_CKPT}"
  echo ">>   - Raw Data Pool:      ${RAW_DIR}"
  
  if [[ ! -f "${TEACHER_CKPT}" ]]; then
    echo ">> [LỖI] Không tìm thấy Teacher Checkpoint tại: ${TEACHER_CKPT}"
    echo ">> Vui lòng đặt file checkpoint vào đường dẫn trên hoặc bỏ cờ --skip-download để tự tải."
    exit 1
  fi
  if [[ ! -d "${RAW_DIR}" || -z "$(ls -A "${RAW_DIR}" 2>/dev/null)" ]]; then
    echo ">> [LỖI] Thư mục raw data pool trống: ${RAW_DIR}"
    echo ">> Vui lòng đặt dữ liệu vào thư mục trên hoặc bỏ cờ --skip-download để tự tải."
    exit 1
  fi
  echo ">> [✓] Đã xác nhận đầy đủ tài nguyên cục bộ sẵn sàng."
else
  # 3.1 Checkpoint Teacher Phase 2
  if [[ ! -f "${TEACHER_CKPT}" ]]; then
    echo ">> Tải checkpoint Teacher DFlash Phase 2 từ Hugging Face..."
    hf download Tphuc15/qwen25vl-3b-depth1-depth3 \
      qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt \
      --repo-type dataset \
      --local-dir "${TEACHER_DIR}"
  else
    echo ">> Checkpoint Teacher Phase 2 đã tồn tại: ${TEACHER_CKPT}"
  fi

  # 3.2 Dataset Raw Pool
  if [[ ! -f "${RAW_DIR}/download_status.json" && -z "$(ls -A "${RAW_DIR}" 2>/dev/null)" ]]; then
    echo ">> Tải dataset raw pool từ Hugging Face (Tphuc15/dflash-posttraining-raw)..."
    hf download Tphuc15/dflash-posttraining-raw \
      --repo-type dataset \
      --local-dir "${RAW_DIR}"
  else
    echo ">> Dataset raw pool đã tồn tại tại: ${RAW_DIR}"
  fi
fi

# --- 4. Chuẩn Bị Manifest & Greedy Target Tokens ---
echo ""
echo ">> [3/5] Xử lý dữ liệu & sinh target tokens (${TRAIN_RECORDS} train / ${VAL_RECORDS} val)..."

if [[ ! -f "${DATA_ROOT}/manifests/train.jsonl" ]]; then
  echo ">> Đang giải nén video và lập manifest train/val/test..."
  "${PYTHON_BIN}" -m src.train_VLM.prepare_posttraining prepare \
    --source "${RAW_DIR}" \
    --output-dir "${DATA_ROOT}" \
    --train-records "${TRAIN_RECORDS}" \
    --validation-records "${VAL_RECORDS}" \
    --test-records "${TEST_RECORDS}" \
    --long-fraction 0.75 \
    --caption-fraction 0.30 \
    --visual-budget-mixture "3000:0.25:12,13000:0.375:50,25000:0.375:98" \
    --spatial-tokens-per-temporal-patch 512 \
    --seed 20261104
else
  echo ">> Manifests đã tồn tại tại ${DATA_ROOT}/manifests"
fi

if [[ ! -f "${DATA_ROOT}/responses/train.jsonl" ]]; then
  echo ">> Đang sinh exact greedy target tokens bằng Target model..."
  "${PYTHON_BIN}" -m src.train_VLM.prepare_posttraining targets \
    --data-dir "${DATA_ROOT}" \
    --config "${TARGET_CONFIG}" \
    --batch-size 4 \
    --splits train validation
else
  echo ">> Target responses đã tồn tại tại ${DATA_ROOT}/responses"
fi

# Thẩm định tính toàn vẹn và cách ly video giữa các split
echo ">> Kiểm định toàn vẹn dữ liệu..."
"${PYTHON_BIN}" -m src.train_VLM.prepare_posttraining validate \
  --data-dir "${DATA_ROOT}" \
  --require-target-responses

# --- 5. Trích Xuất & Lưu Trữ Target Hidden States (5 Layers) ---
echo ""
echo ">> [4/5] Trích xuất Target Hidden States (5 Layers: 26, 28, 30, 32, 34)..."
if [[ ! -d "${FEATURES_DIR}/train" ]]; then
  "${PYTHON_BIN}" -m src.train_VLM.cache_visual10_selfdistill_features \
    --data-dir "${DATA_ROOT}" \
    --target-config "${TARGET_CONFIG}" \
    --draft-config "${DRAFT_ARCH_CONFIG}" \
    --output-dir "${FEATURES_DIR}" \
    --splits train validation
else
  echo ">> Feature cache đã tồn tại tại: ${FEATURES_DIR}"
fi

# --- Hàm Tạo Config YAML Tự Động ---
generate_yaml_config() {
  local target_yaml="$1"
  local output_dir="$2"
  local epochs="$3"
  local accum_steps="$4"
  local run_id="$5"

  cat <<EOF > "${target_yaml}"
model:
  target_model_path: ${TARGET_MODEL}
  draft_model_config: ${DRAFT_ARCH_CONFIG}
  draft_checkpoint_path: ${TEACHER_CKPT}
  draft_warm_start_mode: strict
  target_backend: sglang
  input_modality: qwen2_5_vl
  torch_dtype: bfloat16

data:
  hidden_states_path: ${FEATURES_DIR}/train
  eval_hidden_states_path: ${FEATURES_DIR}/validation
  hidden_state_phase: phase2
  max_length: 32768
  dataloader_num_workers: 4

training:
  strategy: dflash
  dflash_self_distill: true
  dflash_teacher_checkpoint_path: ${TEACHER_CKPT}
  dflash_visual_retention: 0.1
  dflash_visual_selector: temporal_spatial
  dflash_distill_recipe: dual
  dflash_kd_weight: 0.5
  dflash_kd_temperature: 1.0
  num_epochs: ${epochs}
  batch_size: 1
  accumulation_steps: ${accum_steps}
  fsdp_sharding: FULL_SHARD
  learning_rate: 1.0e-5
  warmup_ratio: 0.03
  max_grad_norm: 1.0
  attention_backend: flex_attention
  num_anchors: 128
  objective_chunk_blocks: 2
  loss_decay_gamma: 7.0
  eval_interval: 100
  save_interval: 200
  max_checkpoints: 3
  log_interval: 10

tracking:
  report_to: none

output_dir: ${output_dir}
run_id: ${run_id}

deployment:
  mode: local_colocated
  trainer:
    nnodes: 1
    nproc_per_node: ${NUM_GPUS}
EOF
}

# --- 6. GIAI ĐOẠN TEST: SMOKE TEST (15 STEPS) ---
if [[ "${DO_SMOKE}" -eq 1 ]]; then
  echo ""
  echo "========================================================================"
  echo ">> [BƯỚC TEST] Khởi chạy Pre-flight Smoke Test (15 steps) trên ${NUM_GPUS} GPU..."
  echo ">> Mục đích: Kiểm tra compile Triton FlexAttention kernel & cấp phát VRAM..."
  echo "========================================================================"

  SMOKE_CONFIG="${REPO_ROOT}/configs/run_visual10_smoke.yaml"
  generate_yaml_config "${SMOKE_CONFIG}" "${SMOKE_OUTPUT_DIR}" 1 2 "smoke-visual10"

  "${PYTHON_BIN}" -m specforge.cli train \
    --config "${SMOKE_CONFIG}" \
    training.max_steps=15

  echo ""
  echo ">> [✓] SMOKE TEST 15 STEPS HOÀN TẤT THÀNH CÔNG!"
  echo ">> Môi trường GPU, FlexAttention và Dataloader hoạt động hoàn hảo."
fi

# --- 7. GIAI ĐOẠN HUẤN LUYỆN FULL: PRODUCTION TRAINING ---
if [[ "${DO_FULL}" -eq 1 ]]; then
  echo ""
  echo "========================================================================"
  echo ">> [BƯỚC TRAIN] TỰ ĐỘNG CHUYỂN TIẾP VÀO FULL PRODUCTION TRAINING..."
  echo ">> Huấn luyện 3 epochs với FSDP trên ${NUM_GPUS} GPU B200..."
  echo "========================================================================"

  FULL_CONFIG="${REPO_ROOT}/configs/run_visual10_full.yaml"
  generate_yaml_config "${FULL_CONFIG}" "${FULL_OUTPUT_DIR}" 3 8 "production-visual10"

  "${PYTHON_BIN}" -m specforge.cli train \
    --config "${FULL_CONFIG}"

  echo ""
  echo "========================================================================"
  echo ">> [✓] FULL TRAINING HOÀN TẤT THÀNH CÔNG!"
  echo ">> Checkpoint lưu tại: ${FULL_OUTPUT_DIR}"
  echo "========================================================================"

  # --- 8. TỰ ĐỘNG ĐÁNH GIÁ (EVALUATION TRÊN TEST SET) ---
  if [[ -f "${DATA_ROOT}/responses/test.jsonl" ]]; then
    echo ""
    echo ">> [EVAL] Đang tiến hành đánh giá tự động trên tập Test (250 mẫu)..."
    mkdir -p "${REPO_ROOT}/results/evaluation_visual10_selfdistill"
    "${PYTHON_BIN}" -m src.train_VLM.evaluate_visual10_selfdistill \
      --target-config "${TARGET_CONFIG}" \
      --draft-config "${DRAFT_ARCH_CONFIG}" \
      --checkpoint "${FULL_OUTPUT_DIR}/run-latest" \
      --manifest "${DATA_ROOT}/responses/test.jsonl" \
      --visual-retention 0.1 \
      --visual-selector temporal_spatial \
      --output "${REPO_ROOT}/results/evaluation_visual10_selfdistill/summary.json"
    echo ">> [✓] Báo cáo nghiệm thu đã được lưu tại: results/evaluation_visual10_selfdistill/summary.json"
  fi
fi

echo ""
echo "========================================================================"
echo ">> TOÀN BỘ QUY TRÌNH ĐÃ HOÀN TẤT 100%!"
echo "========================================================================"
