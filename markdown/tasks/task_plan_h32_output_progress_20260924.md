# H3.2 acc length by output progress — 2026-09-24

## Goal

Create separate H3.2 figures for draft depth 1, 3, and 5. Each figure shows
mean acc length at output-progress points 10% through 100%, with four panels
for MVBench/VDC × LLaVA68K/ShareGPT68K and Full/Zero/Cut lines.

## Locked interpretation

- Use the H3.2 metric_records.csv target output length and
  acceptance_round_records.jsonl round trace.
- Acc length is matched_proposals per valid speculative round.
- Exclude partial and terminal rounds, matching the existing H2.1 progress
  definition; at 100%, use all valid non-terminal/non-partial rounds.
- Average round values within each sample first, then average across samples.
- Keep dataset, corpus, depth and visual condition separate.
- Report mean, sample SD, n valid samples, n valid rounds and n total samples
  in one detailed CSV.

## Phases

- [complete] 1. Inspect H3.2 trace schema and output-length coverage.
- [complete] 2. Implement depth-specific progress summarizer and figures.
- [complete] 3. Link the three figures and CSV from the H3.2 README section.
- [complete] 4. Verify coverage, 100% reconstruction, files, links and tests.

## Errors

| Error | Attempt | Resolution |
|---|---:|---|
| H3.2 row_id repeats across dataset/corpus/depth | 1 | Use composite key dataset + corpus + depth + visual condition + row_id. |

## Next

Add the H3.2 progress summarizer/plotter and run it over the Phase 2 traces.

## Verification evidence

- Output CSV: 360 rows, 36 groups × 10 progress points; all means present.
- Three PNGs generated for depth 1, 3 and 5, each 3027 × 2097.
- The 100% point matches metric_summary accepted_prefix_tokens for all 36 groups.
- Focused pytest, py_compile, manifest validation and README link audit passed after the README update.


Final verification: 360 CSV rows across 36 groups and 10 progress points; all 36 100% values match accepted_prefix_tokens within tolerance; all three PNGs are nonblank; focused pytest passes 3/3; README relative links and manifest entries pass.
