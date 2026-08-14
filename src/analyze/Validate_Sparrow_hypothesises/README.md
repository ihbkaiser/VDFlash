# Validate Sparrow hypotheses

This directory contains the paper-conformance harness for the insight
experiments in `externals/Sparrow/2026.acl-long.450.pdf`.

The harness uses the local duration-representative 50-sample
`VideoDetailCaption` subset. It does not download or commit model weights.
The official MSD/Qwen2-VL and Qwen2.5-VL checkpoints are selected in the
contract; actual inference requires a Python 3.10 CUDA environment.

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

`REPORT.md` separates paper-conformance from numerical reproduction. It must
not be interpreted as evidence when the validity gate is `FALSE`.

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

- `figure1_insight_summary.png`: two-panel Figure 1 analogue with accepted-length/error bars plus latency bars, and retention curves for `Last Instr.`, `All Text`, or the configured selector.
- `figure2_insight_attention.png`: short/long visual-context panels with Instruction/Visual/Text regions and attention curves.
- `figure3_insight_layer_analysis.png`: layer-cut output agreement beside the head-by-layer visual-attention heatmap.
- `figure6_insight_retention.png`: visual/text hidden-state retention curves with the middle-layer marker.
- `paper_statistics.json` and `figure*_statistics.csv`: per-condition N, mean, spread, and deterministic bootstrap 95% intervals.

These figures mirror the paper's visual grammar and grouping, but all plotted
values come from completed local runs. Missing experiments are not filled with
paper numbers; the corresponding panel/file appears only when its rows exist.

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
(`target` vs `msd_draft`); the audit accepts both, and the report renders them
as separate figures (`figure2_insight_attention.png` vs
`figure2_insight_attention_draft.png`). `--selection top_attention` in the MSD
runner consumes either file's last-instruction scores.

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
compares greedy output IDs, prefix agreement and ROUGE-L with the native
target. Figure 6 computes cosine similarity to the fused input embedding per
layer without retaining all hidden-state tensors in memory.

## Complete orchestration

The process-isolated orchestrator runs calibration, all selected GPU stages,
merges their JSONL rows, audits them and writes the report:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises all \
  --output-dir results/sparrow_validation \
  --quantized
```

A single-script wrapper for a GPU host (T4-class) is also provided; it runs
calibration when missing, then `all` with `--quantized` and
`--allow-out-of-tolerance` (several short VDC videos cannot reach the 25k
milestone, so the closest measured point is used and recorded):

```bash
src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh
```

See `RUN_ON_GPU.md` for the full environment setup and per-insight
verification criteria.

Use `--limit 1` for a smoke run, or `--skip-msd`, `--skip-attention` and
`--skip-layers` when validating one stage. The final evidence file is
`results/sparrow_validation/results.jsonl`; the report is valid only when the
audit exit code is zero. Composite figures and statistics are written under
`results/sparrow_validation/report/` (or the directory passed to
`report --output-dir`).
