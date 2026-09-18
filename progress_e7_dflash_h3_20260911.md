# Progress: DFlash E7 and H3.1 follow-up

## 2026-09-11

- Read the repository instructions and the experiment-design, feedback-ladder,
  planning and J-space protocols.
- Preserved the existing unrelated `task_plan.md`, `findings.md` and
  `progress.md`; created scoped planning files for this study.
- Confirmed the existing E7 artifacts are MSD-based and the current H3.1
  position CSV has only aggregate bins, with the MSD pilot documented as
  `n=2`.
- Confirmed the DFlash code path targets Qwen2.5-VL-3B and found four local
  training-state candidates.
- Confirmed host GPU access works with the approved host probe; repository-local
  cache lacks a complete Qwen2.5-VL-3B target snapshot.
- Added failing CPU tests for prompt variants, visual retention remapping,
  attention region mass/density and position-wise acceptance.
- Implemented the tested E7/H3.1 helper module, DFlash first-prefill attention
  capture, DFlash context-cut plumbing, and a fallback for the missing legacy
  caption scorer.
- Updated stale DFlash default checkpoint paths to the existing 20e LLaVA and
  ShareGPT checkpoints.
- Focused DFlash tests pass: 33 tests.
- At the initial preflight the target model was absent from the repository
  cache; it was subsequently resolved from the local HF cache and used for
  the completed GPU runs.
- Fixed a correctness bug in cut-context intervention: the draft hidden
  context is now compacted by the keep mask after any value transform. Added
  regression coverage for raw prefill-hidden compaction.
- Added DFlash round telemetry for absolute answer positions
  (`proposal_positions`, `accepted_proposal_positions`); position 0 is the
  target-generated anchor and speculative proposals start at position 1.
- Restored the full dependency-light text metric contract in `metrics.py` so
  the compatibility scorer still reports exact match, BLEU-1..4, BLEU,
  ROUGE-L, coverage and unigram metrics.
- GPU1 pilots completed: E7 1 sample × 6 rows and H3.1 1 sample × 3 rows.
  All pilot rows are `status=ok`; E7 captures 5 draft layers and H3.1 records
  non-empty position-wise proposal traces.
- H3.1 full run started in tmux session `vdflash_h31_full` on GPU1 with
  VDC50, 3 conditions, target visual tokens 3000 and max_new_tokens 64.
  GPU0 remains occupied by an unrelated user process and is intentionally not
  touched.
- H3.1-DFlash full completed with 150/150 rows, 50/50 sample IDs and 0
  runtime-error rows. Position-wise bootstrap analysis was generated under
  `results/e7_dflash_h3_20260911/h31_full/analysis`; the early/late
  interactions for Zero and Cut have CIs crossing zero, so the simple
  late-onset hypothesis is not supported by this DFlash run.
- E7-DFlash full was then started in tmux session `vdflash_e7_full` on GPU1
  with 300 planned rows (`50 × 2 prompt variants × 3 visual conditions`),
  max_new_tokens 64 and first-prefill attention capture.

## 2026-09-12

- E7-DFlash full completed with 300/300 rows and 0 runtime errors. An audit
  found 9 answer-hint rows with empty token spans in three long-answer
  samples; the cause was a BPE boundary mismatch in full-hint matching.
- Added a suffix-length fallback restricted to answer tokens, regression tests,
  and reran exactly the 9 affected rows on GPU1. All 9 repair rows have
  non-empty answer-hint spans and attention status `ok`.
- Created the corrected 300-row journal under
  `results/e7_dflash_h3_20260911/e7_full_corrected/`, preserving the original
  journal and recording replacement provenance in `merge_manifest.json`.
- Re-ran E7/H3.1 analysis and generated acceptance and attention mass/density
  figures plus CSV/JSON statistics under
  `results/e7_dflash_h3_20260911/analysis`.
- Corrected E7-DFlash statistics: Natural Deleted − Full acceptance is
  `-0.0724` (95% CI `[-0.1215, -0.0240]`), Answer-hint is `+0.0233`
  (95% CI `[-0.0331, +0.0793]`), and the interaction is `+0.0957`
  (95% CI `[+0.0302, +0.1542]`). Hint density per key remains below question
  density at Full/Reduced/Deleted, despite larger raw hint mass.
- Updated the report to group E7-DFlash under Q2/H2.1 with full numbers,
  figures, density interpretation, corrected artifact links and explicit
  oracle/accuracy limitations.
