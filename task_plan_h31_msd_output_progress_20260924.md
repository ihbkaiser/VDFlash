# H3.1 MSD output-progress figure — 2026-09-24

## Goal

Create a Phase 2 H3.1 MSD figure showing acc length as a function of
cumulative target-output progress, with points every 10% from 10% through
100%, and one series for each visual retention level (100%, 50%, 25%, 0%).
Preserve existing user work and write derived artifacts under
`results/phase2_summary_20260923/h31/`.

## Interpretation locked from source artifacts

- README H3.1 scope: MSD, 200 valid rows, 50 samples × retention 100/50/25/0.
- Acc length is the accepted draft proposal count per speculative round,
  represented by `accepted_prefix_tokens` / `tau_proposal`.
- MSD rounds use `accepted_proposal_count` and `answer_position_start`; they
  do not use DFlash's `matched_proposals` field.
- At progress p, include rounds with
  `answer_position_start < target_output_tokens * p / 100`.
- First average eligible round values within each sample, then average across
  samples in each retention × progress group. Report sample SD and valid-sample
  count in the CSV; the plot shows mean lines with ±1 sample SD error bars.
- The 100% point will be checked against each row's recorded
  `accepted_prefix_tokens`.

## Phases

- [complete] 1. Read README/workflow, inspect dirty worktree, and locate H3.1 traces.
- [complete] 2. Implement the MSD output-progress summarizer/plotter.
- [complete] 3. Generate CSV/PNG artifacts in the Phase 2 H3.1 directory.
- [complete] 4. Verify coverage, 100% reconstruction, image readability, and git diff.

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Shell sandbox failed before command execution with `mountinfo path is not absolute` | 1 | Used approved escalated local shell for read-only workspace inspection and artifact generation. |
| Initial skill alias path included nonexistent `/r0` directory | 1 | Resolved the installed filesystem path to `/home/hust/.codex/skills/<skill>`. |
| Generic H2.1 progress code expects DFlash `matched_proposals` | 1 | Use MSD-specific `accepted_proposal_count` from `acceptance_rounds`. |

## Next

Run focused tests, validate README links and output files, then record the final
verification evidence.

- The first plotter run raised NameError: math; added the missing import and reran successfully.

## Verification evidence

- Focused pytest: 3 passed in src/analyze/Validate_Sparrow_hypothesises/tests/test_phase2_overview.py.
- Output CSV: 40 rows, four retentions, ten progress points, 50 valid samples per point.
- 100% means match metric_summary.csv exactly for all four retentions.
- README local-link audit: 29 relative links checked, 0 missing.
- PNG type/size: 2547 × 1572, 213157 bytes; py_compile and JSON validation passed.

- J-space ship cannot decode the binary PNG as UTF-8; CSV and README were shipped successfully, while PNG integrity was verified with file metadata and PIL.
