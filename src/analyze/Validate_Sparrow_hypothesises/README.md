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

These scenarios are covered by the 12 CPU tests in `tests/`; S2 and S3 are
also executed as GPU smoke gates by `run_msd.py` before a result row is
written.

## Full-input MSD smoke/run

After the T4 environment and checkpoints are available, first run one sample:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises.run_msd \
  --limit 1 \
  --output results/sparrow_validation/msd_full_smoke.jsonl
```

This command currently enables only the full-input condition. It checks native
prefill parity and exact greedy losslessness before writing a row. The
retention condition is rejected explicitly until the compacted draft context
passes the isolation audit; an image-only approximation must not be used as a
substitute.

## Runtime invariant

`runtime.py` constructs video embeddings and Qwen2-VL M-RoPE explicitly. The
image-only `get_input_embeds_qwen2vl` helper from the original MSD repository
is not used for video. Before speculative decoding, the runtime compares the
standard multimodal target logits with the manually fused prefill logits.
