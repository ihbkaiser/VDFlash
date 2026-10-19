# Progress — H3.2 acc length by output progress

## 2026-09-24

- Confirmed the requested output is three depth-specific figures.
- Confirmed H3.2 round trace fields support output-progress aggregation.
- Next: implement, generate and verify the three figures plus detailed CSV.

- Added composite sample keys after the first run exposed row_id reuse across
  dataset/corpus/depth.
- Generated the depth 1/3/5 PNGs and the combined 360-row CSV.
- Added all three images and the detailed CSV to the H3.2 README.
- Updated the H3.2 manifest with the four derived outputs.
- Fresh data checks pass: 360 rows, 36 groups, no missing means, and 100%
  matches metric_summary for all groups.
- Final verification passed: focused pytest 3/3; CSV has 360 rows across 36 groups and 10 progress points; all 36 100% values match accepted_prefix_tokens within tolerance; three PNGs are nonblank; README relative links and manifest entries are valid.
