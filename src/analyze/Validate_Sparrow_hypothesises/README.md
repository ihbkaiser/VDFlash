# Validate Sparrow hypotheses

This directory contains a local MSD-based verification harness for the insight
experiments in `externals/Sparrow/2026.acl-long.450.pdf`.

The harness uses the local duration-representative 50-sample
`VideoDetailCaption` subset. It verifies MSD mechanisms and trends; it is not
intended to reproduce the paper's ViSpec baseline, benchmark mix, or hardware
specific absolute numbers. It does not download or commit model weights.
The official MSD/Qwen2-VL and Qwen2.5-VL checkpoints are selected in the
contract; actual inference requires a Python 3.10 CUDA environment.

The default `local_insight_vdc50` profile is deliberately strict: only rows
whose calibration status is `ok` are evidence, every required milestone and
selector curve must be present, and at least 10 samples must be shared by all
figures. A smoke run therefore produces a diagnostic report, never a falsely
complete paper figure. Use `--contract .../paper_contract.yaml` only when you
explicitly want the legacy non-enforcing schema audit.

## First checks

From the repository root:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises preflight \
  --output results/sparrow_validation/preflight.json

python -m src.analyze.Validate_Sparrow_hypothesises prepare \
  --output results/sparrow_validation/planned_manifest.jsonl
```

The first command is intentionally usable without a GPU. On the T4 machine,
run it with `--require-gpu --require-models` before loading any checkpoint.

## Measured visual-token calibration

Calibration must happen through the Qwen processor because the real token
count is determined by `video_grid_thw`, not by FPS alone:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises calibrate \
  --limit 2 \
  --output results/sparrow_validation/calibration.jsonl
```

Every row records the nearest measured point to 0.4K, 3K, 13K or 25K and is
marked `ok` or `out_of_tolerance`. No estimated count is accepted as a final
experiment result.

## Audit and report

Completed runtime rows must contain the provenance fields described by the
paper contract. The audit fails closed if a target input changed during a
draft-retention ablation, the query position is not the final instruction, a
layer intervention is not recorded, or target/speculative token IDs differ.

```bash
python -m src.analyze.Validate_Sparrow_hypothesises audit \
  --input results/sparrow_validation/results.jsonl \
  --output results/sparrow_validation/audit.json

python -m src.analyze.Validate_Sparrow_hypothesises report \
  --input results/sparrow_validation/results.jsonl \
  --output-dir results/sparrow_validation/report
```

`REPORT.md` separates row conformance, coverage, and losslessness. Paper-
shaped figures are written as PNG/PDF/SVG only when the strict coverage gate
passes; otherwise they are placed under `report/diagnostic/` with an
`INCOMPLETE DIAGNOSTIC` watermark.

## Paper-conformance scenarios

The suite treats these as gates, not optional diagnostics:

| ID | Scenario | Failure meaning |
|---|---|---|
| S0 | Contract validates 0.4K/3K/13K/25K, retention rates, tree `30-4-8`, greedy decoding and model families | The run is not configured like the paper |
| S1 | Every selected video exists, has a stable fingerprint, and every visual-token count comes from `video_grid_thw` | Dataset or token-length confound |
| S2 | Native Qwen2-VL multimodal logits match manually fused video embeddings + M-RoPE | Video adapter is incorrect |
| S3 | Target and MSD output token IDs match exactly for greedy decoding | Losslessness is not established |
| S4 | Figure 2 observes the final instruction query and reports disjoint instruction/visual/text masks | Attention-dilution plot is measuring the wrong tokens |
| S5 | Figure 3 records the layer where visual KV is masked and Figure 6 records visual/text cosine curves | Layer-wise claim is not reproducible |
| S6 | Report contains only measured rows and passes the audit before plotting | A planned or invalid run cannot become evidence |

These scenarios are covered by the CPU tests in `tests/`; S2 and S3 are
also executed as GPU smoke gates by `run_msd.py` before a result row is
written.

## Figure 1: MSD visual-length and retention runs

### Two-GPU data-parallel runner

`run_msd_2gpu.sh` runs two independent batch-size-one MSD workers concurrently,
splitting the manifest between the selected GPUs. This is the recommended way
to use heterogeneous GPUs such as an RTX 3090 and an RTX A4000; `torchrun` is
not used because the MSD runner has no DDP synchronization.

```bash
src/analyze/Validate_Sparrow_hypothesises/run_msd_2gpu.sh \
  --gpus 0,1 \
  --limit 2 \
  --visual-targets 400,3000 \
  --condition full \
  --output-dir results/sparrow_validation_2gpu
```

Use `--limit 2` for a smoke test so each GPU receives one video. The launcher
calibrates once, starts both workers, merges `msd_gpu*.jsonl`, and writes the
combined audit and report. Use `--dry-run` to inspect the generated commands.
For the Figure 1(a) `Remove All` series, pass
`--length-series remove_all --retention-percentages 0` together with
`--condition both`; this labels the zero-retention rows as Figure 1(a) while
keeping the target input unchanged.

### Memory-aware runner for 3090 + A4000

The simple launcher above splits samples round-robin. For the heterogeneous
3090 (24 GB) and A4000 (16 GB) pair, use the memory-aware launcher so splitting
happens per `(video, visual-token milestone)` job. It measures the actual token
count from calibration, routes jobs above the A4000 budget to the 3090, and
keeps one model process per GPU:

```bash
src/analyze/Validate_Sparrow_hypothesises/run_msd_memory_aware_2gpu.sh \
  --gpus 0,1 \
  --a4000-max-visual-tokens 5500 \
  --strong-max-visual-tokens 11000 \
  --visual-targets 400,3000,13000,25000 \
  --condition both \
  --output-dir results/sparrow_validation_memory_aware_2gpu
```

`5500` is a conservative cap: the A4000 probe reached 5,760 tokens with
`max_new_tokens=512` but OOMed at the next tested level. The 3090 probe reached
11,520 tokens and OOMed near 23K, so the launcher uses an 11,000-token strong
GPU cap by default. Jobs above that cap are written to `unsupported_jobs.jsonl`
instead of being launched and crashing the run. The first GPU is assumed to
be the 3090; change `--gpus` if the device order differs. Reaching 25K would
require model sharding/tensor parallelism or a substantially different memory
configuration; it is not safe with the current single-GPU MSD runtime.

### 25K model-parallel runner

For the 13K/25K milestones, use the single-process model-parallel launcher,
not the two-worker data-parallel launchers:

```bash
src/analyze/Validate_Sparrow_hypothesises/run_msd_model_parallel_2gpu.sh \
  --gpus 0,1 \
  --visual-targets 13000,25000 \
  --condition both \
  --max-new-tokens 512 \
  --output results/sparrow_validation_model_parallel_2gpu/msd.jsonl
```

To run the Figure 1(a) keep/remove-all pair at the long milestones, add
`--length-series remove_all --retention-percentages 0`. The full run then
emits `msd_keep_visual` and `msd_remove_all` rows without mixing the
zero-retention condition into Figure 1(b). If a video has no point within the
10% calibration tolerance, strict mode skips it; use
`--allow-out-of-tolerance` only for a separately labelled nearest-point
diagnostic.

This path uses explicit layer placement, chunks video vision frames, avoids
full-sequence vocabulary logits, and allocates MSD KV cache to the actual
context length. It was validated on `v_SEVVSei-r6w` at 25,168 visual tokens:
`lossless=true`, prefix `512/512`, MSD decode `51.91s`, peak 3090 allocation
`15.70 GiB`. The calibration grid includes 176/192-frame candidates so the
25K target can be represented without using an unnecessarily large per-frame
resolution.

After the T4 environment and checkpoints are available, first run one sample:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises.run_msd \
  --limit 1 \
  --output results/sparrow_validation/msd_full_smoke.jsonl
```

The complete MSD runner uses the measured calibration rows and supports both
the full-input visual-length sweep and the draft-only visual-retention sweep:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises msd \
  --calibration results/sparrow_validation/calibration.jsonl \
  --condition both \
  --output results/sparrow_validation/msd.jsonl
```

For Figure 1(b), the target always keeps the full calibrated video. Only the
draft-side embedding sequence is compacted at 100/25/10/5/1/0 percent, and
the row records separate target/draft fingerprints. The default selector is
deterministic uniform retention. To reproduce attention-guided retention after
the Figure 2 run, pass `--selection top_attention --selection-scores ...`.

The runner records separate MSD prefill/decode/end-to-end timings, AR timing,
acceptance traces, output token IDs and the native-prefill parity gate.

The report renderer produces paper-shaped outputs from measured rows:

- `figure1_insight_summary.{png,pdf,svg}`: two-panel local Figure 1 analogue with MSD keep/remove-all length series and attention-guided retention curves.
- `figure2_insight_attention.{png,pdf,svg}`: exactly the short/long MSD-draft panels with disjoint Instruction/Visual/Text regions.
- `figure3_insight_layer_analysis.{png,pdf,svg}`: local output-agreement and VDC answer-quality proxy beside an aggregated, head-sorted layer heatmap.
- `figure6_insight_retention.{png,pdf,svg}`: visual/text hidden-state retention curves with the middle-layer marker.
- `paper_statistics.json` and `figure*_statistics.csv`: per-condition N, mean, spread, and deterministic bootstrap 95% intervals.

These figures mirror the paper's visual grammar and grouping, but all plotted
values come from completed local runs. Missing experiments are not filled with
paper numbers; incomplete runs are diagnostic-only and watermarked.

## Runtime invariant

`runtime.py` constructs video embeddings and Qwen2-VL M-RoPE explicitly. The
image-only `get_input_embeds_qwen2vl` helper from the original MSD repository
is not used for video. Before speculative decoding, the runtime compares the
standard multimodal target logits with the manually fused prefill logits.

## Figure 2: attention dilution

This runner captures one attention row per decoder layer and never materializes
the full `L x L` matrix. The query is the final user-instruction token, and
the JSONL contains disjoint instruction/visual/text positions plus per-visual
token weights:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises attention \
  --calibration results/sparrow_validation/calibration.jsonl \
  --visual-targets 400 3000 \
  --quantized \
  --output results/sparrow_validation/figure2_attention.jsonl
```

The paper attributes attention dilution to the *draft* model, so an additional
runner probes the official MSD draft's own attention (EAGLE `ea_layer` layers)
during its full-context prefill — the only draft forward with an empty KV
cache inside `topK_genrate`:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises draft_attention \
  --calibration results/sparrow_validation/calibration.jsonl \
  --visual-targets 400 3000 \
  --output results/sparrow_validation/figure2_draft_attention.jsonl
```

Rows share the Figure 2 schema and are distinguished by `attention_source`
(`target` vs `msd_draft`). The strict paper-shaped plot uses the MSD-draft
`last_instruction` trace; target rows remain a diagnostic proxy. Compact
summary rows provide selector scores without requiring an O(L²) artifact.

## Figure 3 and Figure 6: layer analyses

`layers` runs Figure 3(a) visual-KV truncation, Figure 3(b) final-instruction
visual attention by layer, and Figure 6/Appendix D cosine retention. It uses
Qwen2.5-VL-7B, eager attention and bounded hooks for hidden states:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises layers \
  --calibration results/sparrow_validation/calibration.jsonl \
  --visual-targets 3000 \
  --experiments both \
  --quantized \
  --output results/sparrow_validation/layer_analysis.jsonl
```

Figure 3(a) masks visual KV columns from each requested layer onward and
compares greedy output IDs, prefix agreement, output ROUGE-L, and VDC answer
quality (ROUGE-L against the dataset answer) with the native target. The
answer-quality delta is a local task-quality proxy; it is not the paper's
original benchmark accuracy. Figure 6 computes cosine similarity to the fused
input embedding per layer without retaining all hidden-state tensors in
memory.

## Complete orchestration

The process-isolated orchestrator runs calibration, attention first, then MSD
length/retention series, layer probes, merges their JSONL rows, audits them and
writes the report:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises all \
  --output-dir results/sparrow_validation \
  --quantized
```

A single-script wrapper for a GPU host (T4-class) is also provided; it runs
calibration when missing, then `all` with `--quantized`. It excludes
out-of-tolerance points by default. Set `ALLOW_OUT_OF_TOLERANCE=1` only for a
separately-labelled diagnostic run:

```bash
src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh
```

See `RUN_ON_GPU.md` for the full environment setup and per-insight
verification criteria.

Use `--limit 1` for a smoke run, or `--skip-msd`, `--skip-attention` and
`--skip-layers` when validating one stage. The final merged file is
`results/sparrow_validation/results.jsonl`; the report is valid only when the
row, losslessness, and coverage gates all pass. Composite figures and statistics are written under
`results/sparrow_validation/report/` (or the directory passed to
`report --output-dir`).
