# Progress — H3.2 tau effective figure

## 2026-09-24

- Read the Phase 2 README and identified the H3.2 depth/corpus/dataset
  comparison contract.
- Confirmed metric_summary.csv contains all 36 tau_effective groups and that
  it matches accepted_effective_tokens exactly.
- Next: implement and run the four-panel tau-effective plotter.

- Added the four-panel tau-effective plotter and CLI entry to
  scripts/plot_phase2_overview.py.
- Generated tau_effective_h32.png and tau_effective.csv.
- Added the figure and CSV link to the H3.2 section of the Phase 2 README.
- Updated the H3.2 manifest with both derived outputs.
- Initial n check was corrected to distinguish MVBench N=1,000 from recorded
  n valid; all 36 metric groups are present and tau effective matches accepted
  effective exactly.
- Final verification passed: focused pytest 3/3, 36-group parity audit, PNG/manifest checks, README link audit 31/31, py_compile, and git diff --check.
