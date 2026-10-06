# Findings — H3.1 MSD output progress

## Source and scope

- Source README: `results/phase2_summary_20260923/README.md`.
- H3.1 Phase 2 directory: `results/phase2_summary_20260923/h31/`.
- `metric_records.csv` contains 200 rows, 50 samples for each retention
  `100.0`, `50.0`, `25.0`, and `0.0`.
- `acceptance_round_records.jsonl` was generated from the MSD journal, but its
  generic DFlash fields `matched_proposals` and `proposal_count` are null.

## MSD trace contract

- Raw journal: `results/phase1_raw_20260923/h31/msd_retention_vdc50_25k_final.jsonl`.
- Each row has `acceptance_rounds`; each round has
  `answer_position_start` and `accepted_proposal_count`.
- `accepted_prefix_tokens` equals the mean of `accepted_proposal_count` over
  all recorded rounds. A sample check gives 0.84375 from 64 rounds.
- Target output length is the number in `target_output_tokens`; the same sample
  has 116 target tokens and 64 recorded rounds.
- The last MSD round may accept proposals near/end beyond the target token
  boundary; the progress rule uses the round's start position, matching the
  cumulative-prefix semantics used by the H2.1 progress figure.

## Intended output

- `results/phase2_summary_20260923/h31/acc_length_output_progress.csv`
- `results/phase2_summary_20260923/h31/acc_length_output_progress.png`


## Generated result

- The plotter writes 40 rows: 4 retention levels × 10 progress points
  10, 20, ..., 100.
- Every point has 50 valid samples and a sample-level SD.
- At 100%, the terminal MSD round is included so the plotted mean exactly
  reproduces metric_summary.csv:
  100%=1.2194582069, 50%=1.3629824474,
  25%=1.4705107912, 0%=3.6689325554.
- The figure is
  results/phase2_summary_20260923/h31/acc_length_output_progress.png;
  the detailed CSV is next to it.
