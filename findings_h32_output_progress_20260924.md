# Findings — H3.2 acc length by output progress

- H3.2 has 18,900 metric rows: MVBench 1,000 and VDC 50 per experimental
  group before trace-validity filtering.
- Target output lengths range from 2 to 64 tokens.
- acceptance_round_records.jsonl exposes matched_proposals,
  answer_position_start, is_terminal and is_partial_block for DFlash rounds.
- The existing H2.1 progress method is the relevant aggregation template:
  sample-balanced means over valid rounds inside cumulative output prefixes.

## Generated result

- The output CSV contains 360 rows: 36 groups × 10 progress points.
- Three figures were generated, one each for depth 1, 3 and 5. Each figure
  has four dataset × corpus panels and Full/Zero/Cut lines.
- Every 100% group mean matches the corresponding accepted_prefix_tokens
  summary exactly.
- n valid is smallest at early progress points for short MVBench outputs;
  this is expected after excluding terminal and partial rounds.
- Outputs:
  results/phase2_summary_20260923/h32/acc_length_output_progress_h32.csv
  and the three depth-specific PNGs in the same directory.
