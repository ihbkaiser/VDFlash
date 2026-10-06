# VDC-MCQ2 H2.1 experiment card — 2026-09-23

## Question and controlled change

Does requiring an explanatory answer after the selected option produce longer,
format-compliant VDC-MCQ responses, and how does that affect DFlash acceptance
and runtime compared with the existing VDC-natural and VDC-MCQ configurations?

- Existing controls: `natural` and `mcq` journals from Phase 1.
- New variant: `mcq2`; preserve the same VDC-MCQ question and four options.
  One option is the gold caption by design (so its answer text is visible in
  the MCQ prompt), but the correct label is not separately marked and the
  reference is not repeated outside the choices. Require a leading `(A)`–`(D)`
  and 1–2 complete sentences grounded in the video.
- Target: local Qwen2.5-VL-3B-Instruct.
- Draft: DFlash depth 3, separately trained LLaVA68K and ShareGPT68K checkpoints.
- Cohort: the same VDC-MCQ manifest, 50 videos per corpus.
- Conditions: Full, Zero, Cut for each sample and prompt.
- New-run cap: `max_new_tokens=128`. Existing comparison journals used a
  64-token cap; the report will explicitly mark this cap difference.

## Run matrix and artifacts

The new run contains 2 corpora × 50 samples × 3 visual conditions = 300 rows.

- LLaVA68K raw journal:
  `results/phase1_raw_20260923/h21_mcq2/llava68k/h2_1_dflash.jsonl`
- ShareGPT68K raw journal:
  `results/phase1_raw_20260923/h21_mcq2/sharegpt68k/h2_1_dflash.jsonl`
- Summary and visualizations:
  `results/phase2_summary_20260923/h21_mcq2/`
- User report:
  `results/phase2_summary_20260923/README.md`

Use the VDC-MCQ manifest, existing per-sample 3,000-visual-token calibration,
same target, depth-3 config, video preprocessing, and Full/Zero/Cut definitions
as the previous H2.1 run. Run one-sample and five-sample pilots before the
complete 50-sample corpus runs; resume from journals without overwriting the
existing Phase-1 data.

## Metrics to collect and describe

Per sample/condition retain the complete target and speculative output IDs and
text, prompt text/IDs, selected/correct options and provenance, per-position and
per-round acceptance traces, accepted-prefix/effective lengths, global
round-based acceptance rate, output lengths, lossless prefix/hash, prefill,
draft, verification, decode and end-to-end timing, throughput/speedup, visual
token counts, runtime status, and the generation cap.

Phase-2 descriptive metrics:
- Target and speculative output tokens, answer words, and sentence counts.
- Parsed choice and correct-option rate, separately from response-format rate.
- MCQ2 format proxy: leading `(A)`–`(D)`, at least five answer words, and
  one or two punctuation-terminated sentences. This is not a semantic-quality
  score.
- Accepted prefix/effective tokens, global round acceptance rate, timing,
  throughput, speedup, losslessness, cap-hit indicators, and visual context.
- Group summaries and line plots by prompt variant and Full/Zero/Cut, retaining
  per-corpus records.

Do not calculate `I_format`, early/late interactions, bootstrap intervals, or
make a hypothesis decision in this collection/update.

## Completion checks

- Pilot succeeds on GPU and produces the intended `mcq2` rendered prompt.
- Each corpus has 150/150 expected sample-condition rows, no runtime errors,
  and exact recorded provenance for the HF checkpoint/config.
- Check target-output consistency across Full/Zero/Cut and report any parity
  deviations; do not silently discard failed rows.
- Summary tables/plots resolve from the README and clearly disclose the
  128-versus-64 token-cap comparison.

## Checkpoint provenance

Restore only the exact depth-3 files from the documented HF dataset repository
`Tphuc15/qwen25vl-3b-depth1-depth3`. Their verified SHA-256 values and sizes
are recorded in `findings_vdc_mcq2_20260923.md`. Stage one corpus at a time in
`/dev/shm` because the workspace filesystem is nearly full; do not delete
unrelated model/cache artifacts.

## Runtime and calibration decisions verified during pilots

- Use the already-installed `.venv-msd` environment (Transformers 4.52.4,
  Torch 2.5.1+cu121); the default `.venv` has Transformers 4.49.0 and cannot
  import the DFlash-required attention helper.
- Place the target wholly on GPU0 and DFlash draft on GPU1. The custom
  `model_parallel` target map failed tied-parameter validation; target weights
  are about 7.1 GB and fit within GPU0's available VRAM. No shared loader code
  was changed.
- Pass `--allow-out-of-tolerance` to match legacy H2.1 behavior. Prior raw
  H2.1 has an `out_of_tolerance` sample measured at 4,576 visual tokens for a
  3,000 target; this is retained and its realized count remains logged.
- Pilot 5 is complete at 15/15 `status=ok`, with no 128-token cap hits. It
  observed mean speculative output 92.6 tokens / 81.2 answer words, 100% option
  accuracy and losslessness, and 0% compliance with the 1–2 sentence format
  proxy (the model copied multi-sentence option text). These are pilot-only
  descriptive values, not a population conclusion.
