# DFlash Qwen3.5-4B video acceptance-length experiment

Harness for the plan in
[`markdown/ke_hoach_thuc_nghiem_acceptance_qwen35_dflash.md`](../../markdown/ke_hoach_thuc_nghiem_acceptance_qwen35_dflash.md):
measure how DFlash acceptance length changes with the number of visual tokens
on Video-MME videos (short / medium / long), under several draft visual-input
ratios (`100%`, `50%`, `12.5%` and `0%`), with an online speculative decoder whose output must
match greedy target decoding token-for-token.

## Files

- `qwen35_dflash_video_decode.py` — VLM-aware online DFlash decoder (target
  M-RoPE prefill, DFlash block drafting, hybrid-cache verification, per-round
  acceptance logging).
- `qwen35_video_acceptance.py` — CLI: `prepare`, `run`, `analyze`.
- `requirements_qwen35.txt` — isolated environment.
- `../../tests/test_qwen35_video_acceptance.py` — unit + stub integration tests.

## Key technical decisions

1. **Lossless verification is mandatory.** Transformers 5.3.0's Qwen3.5 hybrid
   cache resets the linear-attention state on every cached forward with
   `seq_len > 1`, so a parallel 16-token verification block is *not* equivalent
   to greedy decoding (verified empirically). The decoder therefore verifies
   draft proposals with the target's native one-token recurrent path
   (`--verify-mode exact`), which is bit-identical to the path used by greedy
   decoding. Drafting remains fully parallel (one DFlash block per round), so
   the per-round acceptance lengths are exactly the online acceptance lengths.
   The plan's parallel block verification (`--verify-mode block`, with cache
   clone + accepted-prefix replay) is implemented for reference; the output
   equality gate rejects it for Qwen3.5 hybrid targets.
2. **`video_grid_thw` normalization.** The Qwen3.5 processor emits one vision
   span per temporal patch but returns a single grid row (e.g. `[2,14,14]` for
   4 frames). `get_rope_index` iterates one grid row per video span, so grids
   are split to one row per span (`[[1,14,14],[1,14,14]]`) for both the decoder
   and the greedy `generate` reference.
3. **Growing attention mask.** Every decode-step forward passes an attention
   mask covering the full prefix (as `generate` does); a `(1,1)` mask changes
   the 4D causal mask and breaks exactness for the full-attention layers.
4. **Draft visual ratios** keep a uniformly spread subset of visual positions
   in the draft context while retaining surrounding text. `100%` keeps the
   full prompt; `0%` removes the whole vision span (including timestamps and
   boundaries). The target always sees the complete multimodal prompt.
5. Acceptance metric: `tau_proposal = mean(k)` and
   `tau_effective = mean(k+1)` over non-terminal, full-size verification
   rounds; terminal/partial/EOS-truncated rounds are logged but excluded.

## Environment

```bash
python -m venv .venv-qwen35
source .venv-qwen35/bin/activate
pip install -r src/experiments/requirements_qwen35.txt
pip install -e externals/dflash   # editable DFlash package
```

The runner also works without the editable install (it adds
`externals/dflash` to `sys.path` automatically).

## Workflow

### 1. Prepare the dataset subset

```bash
python -m src.experiments.qwen35_video_acceptance prepare \
  --dataset lmms-eval/Video-MME \
  --output-dir data/video_mme_acceptance \
  --per-duration 24 \
  --seed 42
```

The plan's `MME-Benchmarks/Video-MME` id is not publicly accessible; the
standard `lmms-eval/Video-MME` mirror is used instead (same benchmark data).
Videos are extracted from the dataset's ZIP chunks with HTTP range requests,
so only the selected videos are downloaded. `--pilot` selects 2 short, 2
medium and 2 long videos and also emits controlled-sweep runs for the 2 long
videos.

### 2. Pilot (28 runs)

```bash
python -m src.experiments.qwen35_video_acceptance run \
  --manifest data/video_mme_acceptance/pilot_manifest.jsonl \
  --output results/qwen35_dflash_acceptance/pilot.jsonl \
  --pilot \
  --visual-percentages 100,50,12.5,0 \
  --visual-token-budgets 392,2989,13034,24990 \
  --max-new-tokens 256 \
  --temperature 0 \
  --seed 42
```

### 3. Main experiment (336 runs)

```bash
python -m src.experiments.qwen35_video_acceptance run \
  --manifest data/video_mme_acceptance/manifest.jsonl \
  --output results/qwen35_dflash_acceptance/main.jsonl \
  --experiments natural,controlled \
  --visual-percentages 100,50,12.5,0 \
  --visual-token-budgets 392,2989,13034,24990 \
  --max-new-tokens 256 \
  --temperature 0 \
  --seed 42 \
  --resume
```

### 4. Analyze

```bash
python -m src.experiments.qwen35_video_acceptance analyze \
  --input results/qwen35_dflash_acceptance/main.jsonl \
  --output-dir results/qwen35_dflash_acceptance/report \
  --bootstrap-replicates 10000 \
  --seed 42
```

Outputs: `per_run.csv`, `per_sample.csv`, `per_bucket.csv`, `summary.json`,
`bootstrap.json`, `report.md` and four PNG charts.

## Other video caption benchmarks

The decoder can also be run on local-video caption datasets through the
generic benchmark adapter:

```bash
python -m src.analyze.qwen35_dflash_benchmark \
  --dataset lmms-lab/VideoDetailCaption \
  --video-root /data/VideoDetailCaption \
  --output results/videodetailcaption/qwen35_dflash.jsonl \
  --num-frames 64 \
  --max-new-tokens 256 \
  --visual-percentages 100,50,12.5,0 \
  --resume
```

For the VDC aspect tasks, use `--dataset wchai/lmms_VDC_test` and one of
`--task detailed`, `short`, `main_object`, `camera` or `background`. The
runner writes `prediction`, `reference`, `outputs_match`, acceptance metrics,
and timing fields per sample. The official VideoDetailDescription/VDC LLM
judge remains a separate scoring step; this keeps benchmark scoring from
being mixed into the lossless DFlash decoder.

## Notes

- The experiment assumes one 48–80GB GPU; `run` refuses to start on CPU unless
  `--allow-cpu` is passed (for smoke tests only).
- Every successful run records `target_output_hash == speculative_output_hash`;
  the losslessness gate is enforced inside `run` and checked again in
  `analyze`.
- Exact visual-token budgets (392 / 2989 / 13034 / 24990) are produced at
  224×224 letterbox because Qwen3.5 emits 49 LLM tokens per temporal patch
  (ceil(frames/2) × 49).
