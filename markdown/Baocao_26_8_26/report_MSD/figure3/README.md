# Figure 3 — Qwen2.5-VL-3B

Canonical Figure 3 combines the corrected MVBench visual-KV ablation and the
completed all-layer visual-attention measurement for the same model/cohort.

```text
figure3/
├── figure3_insight_layer_analysis.{png,pdf,svg}  composite used by reports
├── figure3a_qwen25vl3b_full_20260821.png         panel (a) presentation copy
├── figure3b_qwen25vl3b_visual_attention_20260822.png  panel (b) presentation copy
├── figure3a_statistics.csv                       panel (a) statistics
├── figure3b_statistics.csv                       current canonical 36-layer CSV
├── figure3b_qwen25vl3b_statistics_20260822.csv   dated copy of panel (b) CSV
└── legacy_qwen2vl_20260819/                      superseded composite/statistics provenance
```

Panel (a) contains 1,000 MVBench samples and 10,000 scored rows. Panel (b)
contains 1,000 samples × 36 native decoder layers = 36,000 successful rows,
with 16 heads and zero error rows. The raw panel (b) JSONL and summary remain
in [`../../figure3b_qwen25vl3b_visual_attention_20260822/`](../../figure3b_qwen25vl3b_visual_attention_20260822/).
