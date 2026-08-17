# Chạy lại & kiểm chứng các insight trong Sparrow paper bằng MSD

Tài liệu này mô tả cách chạy toàn bộ harness kiểm chứng paper
*Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for
Speculative Decoding in Video LLMs* (ACL 2026) bằng **MSD**
(`lucylyn/MSD-Qwen2VL-7B-Instruct`, EAGLE-based) trên subset 50 video
`dataset/VideoDetailCaption`.

## 1. Insight cần kiểm chứng và runner tương ứng

| # | Insight (paper) | Runner (GPU) | Model | Metric chính |
|---|---|---|---|---|
| I1 | Fig 1(a): accepted length suy giảm khi visual tokens tăng (0.4k→25k) | `msd --condition full` | MSD-Qwen2VL-7B (4-bit) | accepted length, decode/e2e speedup |
| I2 | Fig 1(b): negative visual gain — prune draft visual input (100→0%) làm accepted length tăng | `msd --condition retention` | MSD-Qwen2VL-7B | accepted length, lossless rate |
| I3 | Fig 2: attention dilution trong draft model | `attention` (target proxy) + **`draft_attention`** (MSD draft thật) | Qwen2-VL-7B / MSD draft | instruction/visual/text attention mass, visual entropy |
| I4 | Fig 3(a): visual KV indispensable ở layer đầu, robust sau layer ~20 | `layers` figure3 | Qwen2.5-VL-7B | prefix agreement, output ROUGE-L, VDC answer-quality delta theo layer cut |
| I5 | Fig 3(b): middle layers là arena chính của visual-text interaction | `layers` figure3_attention | Qwen2.5-VL-7B | per-layer/per-head visual mass |
| I6 | Fig 6/App D: visual semantics internalize — visual cosine < 0.25 gần layer 20 | `layers` figure6 | Qwen2.5-VL-7B | layerwise visual/text cosine |

Lưu ý kiểm chứng trung thực: paper đo attention dilution của **draft model**.
Runner `draft_attention` (mới) hook vào các `self_attn` layer của EAGLE draft
(`model.ea_layer.layers`) và chụp attention tại forward full-context đầu tiên
của `topK_genrate` (call duy nhất có KV cache rỗng) — đúng hành vi prefill của
draft. Runner `attention` giữ nguyên như proxy trên target để đối chiếu; hai
nguồn được phân biệt bằng field `attention_source` (`target` / `msd_draft`).

## 2. Môi trường GPU host (T4 16GB trở lên)

```bash
# Python 3.10, PyTorch CUDA 12.1 và Transformers 4.49.0 (xem requirements.txt của harness)
python3.10 -m venv .venv-msd && source .venv-msd/bin/activate
pip install -r src/analyze/Validate_Sparrow_hypothesises/requirements.txt
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Kích hoạt venv và CUDA runtime wheels cho bitsandbytes
source src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh
python -m bitsandbytes
```

Nếu `bitsandbytes` báo thiếu `libcusparse.so.12`, cài bổ sung:

```bash
python -m pip install nvidia-cusparse-cu12==12.1.0.106
source src/analyze/Validate_Sparrow_hypothesises/activate_msd_env.sh
```

Model cần có trong HF cache (hoặc `hf auth login` để tải khi chạy):

- `Qwen/Qwen2-VL-7B-Instruct` (target của MSD, dùng cho Fig 1/2)
- `lucylyn/MSD-Qwen2VL-7B-Instruct` (draft MSD chính thức)
- `Qwen/Qwen2.5-VL-7B-Instruct` (~20 GB, dùng cho Fig 3/6)

```bash
hf download Qwen/Qwen2-VL-7B-Instruct --local-dir ~/.cache/...  # hoặc để AutoModel tự tải
hf download lucylyn/MSD-Qwen2VL-7B-Instruct
hf download Qwen/Qwen2.5-VL-7B-Instruct
```

Dataset: `dataset/VideoDetailCaption/` (50 videos + `subset_manifest.jsonl`)
phải nằm cạnh repo (đã có sẵn trong repo này).

## 3. Cách chạy

### Cách 1 — Một lệnh duy nhất (khuyên dùng)

```bash
src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh
```

Trên cặp RTX 3090 + RTX A4000, để MSD và `draft_attention` dùng cùng placement
model-parallel, chạy:

```bash
MSD_DEVICE_MAP=model_parallel \
MSD_MAX_MEMORY=0:22GiB,1:14GiB \
src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh
```

Script tự chạy: calibration (nếu chưa có) → `all` (preflight → target/draft
attention → MSD full + remove-all + attention-guided retention → layers →
audit → report), với `--quantized` (4-bit cho T4). Chỉ điểm calibration
`ok` được đưa vào cohort paper-shaped; đặt `ALLOW_OUT_OF_TOLERANCE=1` nếu cần
diagnostic riêng. Kết quả nằm ở `results/sparrow_validation/`:

- `msd_full.jsonl`, `msd_remove_all.jsonl`,
  `msd_retention_last_instruction.jsonl`,
  `msd_retention_all_text.jsonl`, `figure2_attention.jsonl`, `figure2_draft_attention.jsonl`,
  `layer_analysis.jsonl` → merge thành `results.jsonl`
- `audit.json` (fail-closed: leak target/draft, masks chồng nhau, model sai contract…)
- `report/REPORT.md`, `report/figure*_insight_*.{png,pdf,svg}` (only when
  coverage passes; otherwise `report/diagnostic/` is watermarked),
  `report/paper_statistics.json`

Tham số hữu ích:

```bash
./run_sparrow_validation_gpu.sh --limit 5        # smoke 5 samples
./run_sparrow_validation_gpu.sh --skip-msd       # bỏ qua stage nào đó
./run_sparrow_validation_gpu.sh --layer-visual-targets 3000 13000
```

### Cách 2 — Chạy từng bước (trên GPU host)

```bash
# CPU-ish: calibration (chỉ cần processor + video, không cần model 7B)
python -m src.analyze.Validate_Sparrow_hypothesises calibrate \
  --output results/sparrow_validation/calibration.jsonl

# GPU
python -m src.analyze.Validate_Sparrow_hypothesises msd \
  --calibration results/sparrow_validation/calibration.jsonl \
  --condition both --allow-out-of-tolerance \
  --output results/sparrow_validation/msd.jsonl

python -m src.analyze.Validate_Sparrow_hypothesises attention \
  --calibration results/sparrow_validation/calibration.jsonl \
  --visual-targets 400 3000 --quantized --allow-out-of-tolerance \
  --output results/sparrow_validation/figure2_attention.jsonl

python -m src.analyze.Validate_Sparrow_hypothesises draft_attention \
  --calibration results/sparrow_validation/calibration.jsonl \
  --visual-targets 400 3000 --allow-out-of-tolerance \
  --output results/sparrow_validation/figure2_draft_attention.jsonl

python -m src.analyze.Validate_Sparrow_hypothesises layers \
  --calibration results/sparrow_validation/calibration.jsonl \
  --experiments both --visual-targets 3000 --quantized --allow-out-of-tolerance \
  --output results/sparrow_validation/layer_analysis.jsonl

# CPU: audit + report (fail-closed)
python -m src.analyze.Validate_Sparrow_hypothesises audit \
  --input results/sparrow_validation/results.jsonl \
  --output results/sparrow_validation/audit.json
python -m src.analyze.Validate_Sparrow_hypothesises report \
  --input results/sparrow_validation/results.jsonl \
  --output-dir results/sparrow_validation/report
```

Nếu đã có sẵn `calibration.jsonl` (chạy sẵn trên máy không GPU như trong repo
này), có thể copy file đó sang GPU host và dùng `--skip-calibration`.

## 4. Smoke test trước khi chạy full

```bash
python -m src.analyze.Validate_Sparrow_hypothesises msd --limit 1 \
  --output results/sparrow_validation/msd_smoke.jsonl
```

Nên bắt đầu với 1 sample để xác nhận: native prefill parity đạt, lossless
đúng (target output là prefix của speculative output), timing hợp lý. Sample
có thể dùng: `v_AwgGYaV1lT0` (3k tokens ok, 25k không đạt — dùng
`--allow-out-of-tolerance` nếu muốn chạy điểm đó).

## 5. Đọc kết quả — tiêu chí kiểm chứng định tính

So với số paper (Table 4/Fig 1/2/3/6) — máy 3090+A4000 + 4-bit không cần khớp
số tuyệt đối, chỉ cần khớp **xu hướng** trên cùng cohort VDC local:

| Insight | Paper nói | Kiểm chứng đạt khi |
|---|---|---|
| I1 | MSD accepted length 4.12@0.5k → 1.04@25k; latency tăng | accepted length giảm dần theo milestone; decode latency tăng |
| I2 | accepted length tăng khi retention giảm; 0% không tệ hơn 100% | accepted length(0%) ≥ accepted length(100%); lossless rate ổn định |
| I3 | attention bị phân tán ở context dài | visual entropy/mass cao hơn ở 3k so với 0.4k (draft rows là bằng chứng chính) |
| I4 | cắt visual KV từ layer ≥ 20 không đổi output và chất lượng | prefix agreement giữ ~1.0; VDC answer-quality delta gần 0 khi layer_cut ≥ 20 |
| I6 | visual cosine tụt < 0.25 quanh layer 20, text gần phẳng | visual_cosine < 0.25 gần layer 20; text_cosine cao hơn rõ |

Report tự phân biệt *paper-conformance* (audit fail-closed) với *numerical
reproduction* — nếu validity gate là `FALSE`, report không được coi là bằng
chứng.

## 6. Lưu ý

- `msd`/`draft_attention` chạy batch size 1; trên T4 16 GB dùng 4-bit
  (BitsAndBytes nf4 — mặc định của `load_msd_qwen2vl`).
- Milestone 25k: nhiều video VDC chỉ đạt trần ~17k tokens (video ngắn); hành
  vi mặc định là chạy điểm đo gần nhất và ghi `calibration_status:
  out_of_tolerance` — đúng tinh thần fail-closed, không giả mạo số liệu.
- Retention sweep với `--selection top_attention`/`last_instruction`/`all_text`
  cần file scores từ Figure 2 (`--selection-scores`); mặc định `uniform`.
- `run_attention.py` (target proxy) và `run_draft_attention.py` (MSD draft)
  cùng emit schema Figure 2 với `attention_source` khác nhau; plots tách riêng
  hai nguồn (`figure2_insight_attention.png` vs `..._draft.png`).
