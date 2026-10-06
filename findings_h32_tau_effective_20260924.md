# Findings — H3.2 tau effective figure

- Phase 2 H3.2 summary: results/phase2_summary_20260923/h32/metric_summary.csv.
- The summary contains 36 tau_effective groups: 2 datasets × 2 training
  corpora × 3 depths × 3 visual conditions.
- The README defines tau_effective in H2.1/H3.2 as Acc length + 1.
- tau_effective and accepted_effective_tokens match exactly for every group.
- H3.2 sample sizes are 1,000 per MVBench group and 50 per VDC group,
  subject to the valid n recorded in the CSV.

## Generated result

- The figure has four panels: MVBench/VDC × LLaVA68K/ShareGPT68K.
- Each panel has depth 1/3/5 on the x axis and Full/Zero/Cut lines.
- The output CSV has 36 rows, one for every dataset × corpus × depth ×
  visual-condition group.
- MVBench cohort N is 1,000 per group but recorded n valid ranges from 769 to
  963; VDC has n=50 in each group.
- The generated files are results/phase2_summary_20260923/h32/tau_effective_h32.png
  and results/phase2_summary_20260923/h32/tau_effective.csv.
