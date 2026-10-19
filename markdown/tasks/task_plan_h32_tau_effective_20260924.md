# H3.2 tau effective comparison figure — 2026-09-24

## Goal

Create one H3.2 figure comparing tau effective across depth 1/3/5 for
MVBench and VDC, split by LLaVA68K and ShareGPT68K, with Full/Zero/Cut
visual conditions shown as separate lines. Store the PNG and exact plotting
CSV under results/phase2_summary_20260923/h32/ and link them in the supplied
Phase 2 README.

## Locked interpretation

- Use metric_summary.csv metric=tau_effective.
- Tau effective is acc length plus one target bonus token, as defined in the
  README and reflected by the runtime summary.
- X axis: draft depth 1, 3, 5.
- Four panels: MVBench/LLaVA68K, MVBench/ShareGPT68K, VDC/LLaVA68K,
  VDC/ShareGPT68K.
- Each line: Full, Zero, Cut; points show mean and error bars show ±1 SD.
- Preserve dataset/corpus/sample grouping; do not pool MVBench and VDC.

## Phases

- [complete] 1. Inspect README, metric_summary.csv, existing H3.2 plotter and manifest.
- [complete] 2. Implement the tau-effective CSV/PNG plot.
- [complete] 3. Link the figure and CSV from the H3.2 README section.
- [complete] 4. Verify all 36 groups, output files, README links and focused tests.

## Errors

| Error | Attempt | Resolution |
|---|---:|---|
| An initial check assumed every MVBench n value was 1,000 | 1 | Use N=1,000 as cohort size and validate the recorded n valid values instead. |

## Next

Add the reusable plot function and run it on the existing H3.2 metric summary.

- Initial group check correctly found 36 groups but used an invalid n-value assumption; the final check uses n valid ranges from the CSV.

## Verification evidence

- Focused pytest: 3 passed.
- Tau-effective CSV: 36/36 dataset × corpus × depth × visual-condition groups.
- Tau effective matches accepted effective exactly for every group.
- PNG: 3027 × 2097 and nonblank; manifest updated.
- README relative-link audit: 31 links checked, 0 missing.
- py_compile and git diff --check passed.
