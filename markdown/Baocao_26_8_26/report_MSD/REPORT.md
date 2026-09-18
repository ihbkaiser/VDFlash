# Sparrow insight validation report

**Validity gate:** `TRUE`

The canonical folder uses the homogeneous Figure 2 cohort: 14 paired samples with exact measured visual-token counts 572 at the 400 target and 2912 at the 3K target. Assets are grouped under their corresponding Figure folders.

This report distinguishes paper-conformance from numerical reproduction. The local run uses the VDC-50 subset and may use 4-bit inference on a 3090+A4000 model-parallel setup.

## Paper traceability

| Figure | Claim | Model | Metric |
|---|---|---|---|
| Figure 1(a) | MSD acceptance and latency degrade as visual length grows | Qwen/Qwen2-VL-7B-Instruct | accepted_length, end_to_end_seconds |
| Figure 1(b) | video acceptance is robust or improves when draft visual input is reduced | Qwen/Qwen2-VL-7B-Instruct | accepted_length, lossless_output_match |
| Figure 2 | long visual context dilutes draft attention | Qwen/Qwen2-VL-7B-Instruct | query_only_attention_mass, normalized_visual_entropy |
| Figure 3 | Qwen2.5-VL-3B visual KV flow and all-layer visual attention | Qwen/Qwen2.5-VL-3B-Instruct | MVBench accuracy, per-head visual attention mass |
| Figure 6 / Appendix D | visual hidden states internalize into text representations | Qwen/Qwen2-VL-7B-Instruct | layerwise_cosine_similarity_to_input_embedding |

## Aggregate summaries

The large aggregate table below is retained for the legacy VDC-50 validation
archive used by Figures 1/2/6. Its historical rows labelled `Figure 3` are not
the updated Qwen2.5-VL-3B evidence. For the canonical Figure 3 result, use the
36-layer statistics and composite in the Figure 3 section below.

| Condition | N | Accepted length | Lossless rate |
|---|---:|---:|---:|
| Figure 1(a) / 0 / 0.0 / unknown | 104 | 3.348 | 100.0% |
| Figure 1(a) / 11968 / 100.0 / unknown | 1 | 1.137 | 100.0% |
| Figure 1(a) / 12121 / 100.0 / unknown | 3 | 1.290 | 100.0% |
| Figure 1(a) / 12288 / 100.0 / unknown | 1 | 1.515 | 100.0% |
| Figure 1(a) / 12512 / 100.0 / unknown | 9 | 1.483 | 100.0% |
| Figure 1(a) / 12740 / 100.0 / unknown | 1 | 1.384 | 100.0% |
| Figure 1(a) / 13294 / 100.0 / unknown | 1 | 0.911 | 100.0% |
| Figure 1(a) / 13728 / 100.0 / unknown | 11 | 1.434 | 100.0% |
| Figure 1(a) / 14300 / 100.0 / unknown | 1 | 1.452 | 100.0% |
| Figure 1(a) / 23936 / 100.0 / unknown | 2 | 1.494 | 100.0% |
| Figure 1(a) / 24576 / 100.0 / unknown | 2 | 1.072 | 100.0% |
| Figure 1(a) / 24633 / 100.0 / unknown | 2 | 1.071 | 100.0% |
| Figure 1(a) / 25024 / 100.0 / unknown | 7 | 1.223 | 100.0% |
| Figure 1(a) / 25200 / 100.0 / unknown | 1 | 1.622 | 100.0% |
| Figure 1(a) / 2737 / 100.0 / unknown | 3 | 1.425 | 100.0% |
| Figure 1(a) / 27456 / 100.0 / unknown | 10 | 1.333 | 100.0% |
| Figure 1(a) / 28886 / 100.0 / unknown | 1 | 1.183 | 100.0% |
| Figure 1(a) / 2912 / 100.0 / unknown | 10 | 1.562 | 100.0% |
| Figure 1(a) / 3072 / 100.0 / unknown | 2 | 1.759 | 100.0% |
| Figure 1(a) / 3128 / 100.0 / unknown | 9 | 1.696 | 100.0% |
| Figure 1(a) / 560 / 100.0 / unknown | 11 | 2.439 | 100.0% |
| Figure 1(a) / 572 / 100.0 / unknown | 13 | 2.457 | 100.0% |
| Figure 1(b) / 0 / 0.0 / unknown | 44 | 3.872 | 100.0% |
| Figure 1(b) / 1197 / 5.0 / unknown | 4 | 3.911 | 100.0% |
| Figure 1(b) / 1229 / 5.0 / unknown | 4 | 3.547 | 100.0% |
| Figure 1(b) / 1232 / 5.0 / unknown | 4 | 3.054 | 100.0% |
| Figure 1(b) / 1251 / 5.0 / unknown | 14 | 3.300 | 100.0% |
| Figure 1(b) / 1260 / 5.0 / unknown | 1 | 4.609 | 100.0% |
| Figure 1(b) / 1373 / 5.0 / unknown | 15 | 3.794 | 100.0% |
| Figure 1(b) / 1444 / 5.0 / unknown | 2 | 3.707 | 100.0% |
| Figure 1(b) / 239 / 1.0 / unknown | 4 | 3.941 | 100.0% |
| Figure 1(b) / 23936 / 100.0 / unknown | 4 | 1.494 | 100.0% |
| Figure 1(b) / 2394 / 10.0 / unknown | 4 | 2.773 | 100.0% |
| Figure 1(b) / 24576 / 100.0 / unknown | 4 | 1.072 | 100.0% |
| Figure 1(b) / 2458 / 10.0 / unknown | 4 | 1.891 | 100.0% |
| Figure 1(b) / 246 / 1.0 / unknown | 8 | 3.604 | 100.0% |
| Figure 1(b) / 2463 / 10.0 / unknown | 4 | 1.909 | 100.0% |
| Figure 1(b) / 24633 / 100.0 / unknown | 4 | 1.071 | 100.0% |
| Figure 1(b) / 250 / 1.0 / unknown | 14 | 3.602 | 100.0% |
| Figure 1(b) / 2502 / 10.0 / unknown | 14 | 2.060 | 100.0% |
| Figure 1(b) / 25024 / 100.0 / unknown | 14 | 1.223 | 100.0% |
| Figure 1(b) / 252 / 1.0 / unknown | 1 | 4.609 | 100.0% |
| Figure 1(b) / 2520 / 10.0 / unknown | 1 | 2.894 | 100.0% |
| Figure 1(b) / 25200 / 100.0 / unknown | 1 | 1.622 | 100.0% |
| Figure 1(b) / 27456 / 100.0 / unknown | 15 | 1.411 | 100.0% |
| Figure 1(b) / 2746 / 10.0 / unknown | 15 | 2.181 | 100.0% |
| Figure 1(b) / 275 / 1.0 / unknown | 15 | 4.010 | 100.0% |
| Figure 1(b) / 28886 / 100.0 / unknown | 2 | 1.183 | 100.0% |
| Figure 1(b) / 2889 / 10.0 / unknown | 2 | 1.743 | 100.0% |
| Figure 1(b) / 289 / 1.0 / unknown | 2 | 3.810 | 100.0% |
| Figure 1(b) / 5984 / 25.0 / unknown | 4 | 1.886 | 100.0% |
| Figure 1(b) / 6144 / 25.0 / unknown | 4 | 1.182 | 100.0% |
| Figure 1(b) / 6158 / 25.0 / unknown | 4 | 1.444 | 100.0% |
| Figure 1(b) / 6256 / 25.0 / unknown | 14 | 1.400 | 100.0% |
| Figure 1(b) / 6300 / 25.0 / unknown | 1 | 2.274 | 100.0% |
| Figure 1(b) / 6864 / 25.0 / unknown | 15 | 1.761 | 100.0% |
| Figure 1(b) / 7222 / 25.0 / unknown | 2 | 1.502 | 100.0% |
| Figure 3 / 2737 / unknown / 0 | 3 | n/a | 100.0% |
| Figure 3 / 2737 / unknown / 12 | 3 | n/a | 100.0% |
| Figure 3 / 2737 / unknown / 16 | 3 | n/a | 100.0% |
| Figure 3 / 2737 / unknown / 20 | 3 | n/a | 100.0% |
| Figure 3 / 2737 / unknown / 24 | 3 | n/a | 100.0% |
| Figure 3 / 2737 / unknown / 4 | 3 | n/a | 100.0% |
| Figure 3 / 2737 / unknown / 8 | 3 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 0 | 14 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 12 | 14 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 16 | 14 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 20 | 14 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 24 | 14 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 4 | 14 | n/a | 100.0% |
| Figure 3 / 2912 / unknown / 8 | 14 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 0 | 1 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 12 | 1 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 16 | 1 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 20 | 1 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 24 | 1 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 4 | 1 | n/a | 100.0% |
| Figure 3 / 2992 / unknown / 8 | 1 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 0 | 2 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 12 | 2 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 16 | 2 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 20 | 2 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 24 | 2 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 4 | 2 | n/a | 100.0% |
| Figure 3 / 3072 / unknown / 8 | 2 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 0 | 10 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 12 | 10 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 16 | 10 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 20 | 10 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 24 | 10 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 4 | 10 | n/a | 100.0% |
| Figure 3 / 3128 / unknown / 8 | 10 | n/a | 100.0% |

## Diagnostic row counts

| Figure | Rows |
|---|---:|
| Figure 1(a) | 205 |
| Figure 1(b) | 264 |
| Figure 2 | 300 |
| Figure 3(a) | 10,000 |
| Figure 3(b) | 36,000 |
| Figure 6 / Appendix D | 840 |

## Paper-shaped statistics

The following aggregates are computed only from measured rows. Each metric
includes N, mean, spread, and a 95% interval. The new Figure 3(b) CSV labels
its interval method explicitly as `normal_approximation_1.96_SE`; legacy CSVs
retain their deterministic bootstrap contract.

[paper_statistics.json](metadata/paper_statistics.json) · [figure1a_statistics.csv](figure1/figure1a_statistics.csv) · [figure1b_statistics.csv](figure1/figure1b_statistics.csv) · [figure2_statistics.csv](figure2/figure2_statistics.csv) · [figure3a_statistics.csv](figure3/figure3a_statistics.csv) · [figure3b_statistics.csv](figure3/figure3b_statistics.csv) · [dated Figure 3(b) CSV](figure3/figure3b_qwen25vl3b_statistics_20260822.csv) · [figure6_statistics.csv](figure6/figure6_statistics.csv)

## Paper-style figures

### Figure1 Insight Summary

![Figure1 Insight Summary](./figure1/figure1_insight_summary.png)

### Figure2 Insight Attention

![Figure2 Insight Attention](./figure2/figure2_insight_attention.png)

### Figure 3 — Qwen2.5-VL-3B visual flow and attention

![Figure 3 — Qwen2.5-VL-3B visual KV ablation and all-layer attention](./figure3/figure3_insight_layer_analysis.png)

The composite now combines the corrected Qwen2.5-VL-3B Figure 3(a) run
(1,000 MVBench samples, 10,000 scored rows) with the completed Figure 3(b)
all-layer attention run (1,000 samples × 36 layers = 36,000 rows, 16 heads,
zero errors). The legacy Qwen2-VL composite is preserved under
`figure3/legacy_qwen2vl_20260819/` for provenance only.

- [Panel (a) PNG](figure3/figure3a_qwen25vl3b_full_20260821.png)
- [Panel (b) PNG](figure3/figure3b_qwen25vl3b_visual_attention_20260822.png)
- [Figure 3(b) summary](../figure3b_qwen25vl3b_visual_attention_20260822/visual_attention.summary.json)
- [Figure 3(b) raw JSONL](../figure3b_qwen25vl3b_visual_attention_20260822/visual_attention.jsonl)

### Figure6 Insight Retention

![Figure6 Insight Retention](./figure6/figure6_insight_retention.png)

Download links for all generated formats:
- [diagnostic/figure1a_acceptance_vs_visual_length.png](diagnostic/figure1a_acceptance_vs_visual_length.png)
- [diagnostic/figure1a_latency_vs_visual_length.png](diagnostic/figure1a_latency_vs_visual_length.png)
- [diagnostic/figure1a_speedup_vs_visual_length.png](diagnostic/figure1a_speedup_vs_visual_length.png)
- [diagnostic/figure1b_acceptance_vs_retention.png](diagnostic/figure1b_acceptance_vs_retention.png)
- [diagnostic/figure1b_lossless_rate_vs_retention.png](diagnostic/figure1b_lossless_rate_vs_retention.png)
- [diagnostic/figure3b_visual_attention_by_layer.png](diagnostic/figure3b_visual_attention_by_layer.png)
- [diagnostic/figure3_layerwise_visual_ablation.png](diagnostic/figure3_layerwise_visual_ablation.png)
- [diagnostic/figure6_information_retention.png](diagnostic/figure6_information_retention.png)
- [figure1/figure1_insight_summary.png](figure1/figure1_insight_summary.png)
- [figure1/figure1_insight_summary.pdf](figure1/figure1_insight_summary.pdf)
- [figure1/figure1_insight_summary.svg](figure1/figure1_insight_summary.svg)
- [figure2/figure2_insight_attention.png](figure2/figure2_insight_attention.png)
- [figure2/figure2_insight_attention.pdf](figure2/figure2_insight_attention.pdf)
- [figure2/figure2_insight_attention.svg](figure2/figure2_insight_attention.svg)
- [figure3/figure3_insight_layer_analysis.png](figure3/figure3_insight_layer_analysis.png)
- [figure3/figure3_insight_layer_analysis.pdf](figure3/figure3_insight_layer_analysis.pdf)
- [figure3/figure3_insight_layer_analysis.svg](figure3/figure3_insight_layer_analysis.svg)
- [figure3/figure3a_qwen25vl3b_full_20260821.png](figure3/figure3a_qwen25vl3b_full_20260821.png)
- [figure3/figure3b_qwen25vl3b_visual_attention_20260822.png](figure3/figure3b_qwen25vl3b_visual_attention_20260822.png)
- [figure3/figure3b_qwen25vl3b_statistics_20260822.csv](figure3/figure3b_qwen25vl3b_statistics_20260822.csv)
- [figure6/figure6_insight_retention.png](figure6/figure6_insight_retention.png)
- [figure6/figure6_insight_retention.pdf](figure6/figure6_insight_retention.pdf)
- [figure6/figure6_insight_retention.svg](figure6/figure6_insight_retention.svg)

## Audit issues

No paper-conformance issues were found.

## Coverage gate

Coverage valid: `TRUE`; paired samples: `10`.

## Losslessness

{
  "valid": true,
  "checked_rows": 2659,
  "valid_rows": 469,
  "issues": []
}
