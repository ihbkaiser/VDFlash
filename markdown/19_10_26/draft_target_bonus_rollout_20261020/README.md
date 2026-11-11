# Draft block + target bonus experiment

This folder contains a separate follow-up to the 2026-10-14 free-running
rollouts. The treatment emits the usual 15 DFlash proposal tokens, then uses
the target's greedy next-token argmax as a sixteenth output token. The target
processes the old anchor plus all 15 proposals to produce that token and the
features for the next block. The 15 draft proposals remain unverified; none are
rejected, replaced, or rolled back.

The baseline is `results/dflash_free_running_main_fp16_20261014/`. The intended
cohort and decoding settings match that run: the exact 33 VDC sample IDs pinned
in the baseline protocol, five
visual-retention conditions, the same depth-3 checkpoint, FP16, refreshed
features, greedy decoding, and a 256-token cap. Analyze repetition inside the
first draft block separately from later blocks.

## Run

From the repository root, run the target-bonus arm on a CUDA-enabled session:

```bash
./.venv/bin/python -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_free_running \
  --phase preflight \
  --target-bonus-per-block \
  --limit 50 \
  --max-new-tokens 256 \
  --dtype fp16 \
  --feature-refresh refresh \
  --output-dir markdown/19_10_26/draft_target_bonus_rollout_20261020/results

./.venv/bin/python -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_free_running \
  --phase generate \
  --target-bonus-per-block \
  --limit 50 \
  --max-new-tokens 256 \
  --dtype fp16 \
  --feature-refresh refresh \
  --output-dir markdown/19_10_26/draft_target_bonus_rollout_20261020/results
```

The journal and immutable protocol are written to `results/`. The launcher reads
the exact 33 IDs from the baseline protocol and passes them as `--sample-id`
arguments. `--limit 50` alone would select 50 records, so it is not sufficient
to reproduce the paired cohort.

## Analyze and plot

After the journal has all 165 rows, run:

```bash
./.venv/bin/python markdown/19_10_26/draft_target_bonus_rollout_20261020/analyze_rollouts.py
```

This writes `analysis_summary.json`, `paired_repetition_metrics.csv`,
`report.md`, and `figures/repetition_by_block.png`. The analysis checks that the
two arms use the same cohort, checkpoint, model, token cap, and visual
conditions. Bootstrap intervals resample videos within each retention
condition.

## Execution status

The cohort-pinned preflight selected 33 samples and planned 165 rollouts. An
escalated CUDA check sees one RTX 3090 with about 3.15 GiB free; the host RTX
A4000 is not enumerated because NVIDIA SMI reports `Unknown Error`. The runner
and analysis are prepared, but this folder does not contain target-bonus
inference results or a result figure yet. No synthetic rows or plots are
substituted for the requested model run. See `run_attempt.log` for the latest
resource check and stop reason.
