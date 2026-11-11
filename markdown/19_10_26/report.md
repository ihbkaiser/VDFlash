# DDTree video visual-cut sweep (19 Oct 2026)

## Scope and protocol

Local manifests: VDC, VidHalluc and VideoCHAIR. The run compares target autoregressive decoding, DFlash and DDTree (depth 3, DDTree node budget 16) at 100%, 50%, 25% and 0% visual-token retention in the drafter context. Target AR always sees the complete video. Captions use a 256-token cap; VidHalluc short answers stop at EOS. The target is Qwen2.5-VL-3B-Instruct, eager attention, 16 sampled frames at 1 fps, with the same prompt/preprocessing for paired rows.

DDTree follows the paper's fixed-budget branching verification idea ([paper](https://arxiv.org/abs/2604.12989)); local adaptations and runner code are retained in this repository. Dataset references: [Video Detailed Caption](https://huggingface.co/datasets/wchai/Video-Detailed-Caption/blob/main/README.md) and [VidHalluc](https://people-robots.github.io/vidhalluc/).

Retention follows the legacy prefix policy: the first retained fraction of visual-token positions is supplied to the drafter; the target still receives full visual context. Results are local subsets (50/100/100 records), not the full published datasets.

## Coverage and integrity

| Dataset | Manifest n | AR baselines | Expected conditions | Successful conditions | Runtime errors | Missing | Duplicate IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| vdc | 50 | 50 | 400 | 400 | 0 | 0 | 0 |
| vidhalluc | 100 | 100 | 800 | 800 | 0 | 0 | 0 |
| videochair | 100 | 100 | 800 | 799 | 0 | 1 | 0 |

Successful-row integrity errors (prompt/target/output hashes and token lengths): **0**.

## Run completeness

The requested sweep is incomplete: **1 of 2,000 speculative conditions are missing**. Statistics use successful rows only. Only the listed condition is under-sampled; its mean and paired contrast are provisional, while the other cells have their planned sample counts.

Missing row IDs: `video9691:ddtree:depth3:retention-0:budget-16`.

## Performance and exactness

Speedups compare total synchronized generation wall time (prefill plus decode) with the paired AR baseline. They are not quality-adjusted: a speculative answer that stops early can appear faster, so read them with output length and task scores. Error bars use 10,000 sample-level bootstrap resamples (95% CI; seed 20261019). Tau values average valid, full non-terminal rounds; tau-effective includes the target bonus token.

| Dataset | Method | Retain | n | Exact % (95% CI) | E2E speedup mean (95% CI) | Median speedup | Decode speedup | tok/s | mean tau eff. | mean output tokens | mean length / AR | peak VRAM GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vdc | dflash | 100% | 50/50 | 16.0 (6.0–26.0) | 1.488 (1.403–1.593) | 1.425 | 1.882 | 42.4 | 2.365 | 193.6 | 0.988 | 15.93 |
| vdc | dflash | 50% | 50/50 | 4.0 (0.0–10.0) | 1.462 (1.371–1.576) | 1.396 | 1.807 | 42.2 | 2.362 | 196.6 | 1.003 | 15.93 |
| vdc | dflash | 25% | 50/50 | 6.0 (0.0–14.0) | 1.456 (1.376–1.545) | 1.401 | 1.802 | 42.6 | 2.329 | 197.1 | 1.006 | 15.93 |
| vdc | dflash | 0% | 50/50 | 4.0 (0.0–10.0) | 1.445 (1.365–1.534) | 1.388 | 1.764 | 43.3 | 2.294 | 199.2 | 1.030 | 15.93 |
| vdc | ddtree | 100% | 50/50 | 8.0 (2.0–16.0) | 1.642 (1.532–1.784) | 1.559 | 2.198 | 47.3 | 3.020 | 196.1 | 1.010 | 15.93 |
| vdc | ddtree | 50% | 50/50 | 4.0 (0.0–10.0) | 1.652 (1.519–1.828) | 1.552 | 2.186 | 47.0 | 3.024 | 194.9 | 1.015 | 15.93 |
| vdc | ddtree | 25% | 50/50 | 8.0 (2.0–16.0) | 1.657 (1.527–1.828) | 1.585 | 2.261 | 48.1 | 3.083 | 198.8 | 1.045 | 15.93 |
| vdc | ddtree | 0% | 50/50 | 4.0 (0.0–10.0) | 1.639 (1.506–1.821) | 1.521 | 2.172 | 47.9 | 2.954 | 200.4 | 1.042 | 15.93 |
| vidhalluc | dflash | 100% | 100/100 | 92.0 (86.0–97.0) | 1.053 (1.040–1.068) | 1.031 | 1.564 | 7.4 | 2.023 | 13.1 | 1.012 | 15.10 |
| vidhalluc | dflash | 50% | 100/100 | 92.0 (86.0–97.0) | 1.052 (1.040–1.066) | 1.035 | 1.591 | 7.5 | 2.048 | 13.2 | 1.015 | 15.10 |
| vidhalluc | dflash | 25% | 100/100 | 92.0 (86.0–97.0) | 1.053 (1.041–1.067) | 1.033 | 1.619 | 7.6 | 2.026 | 13.3 | 1.016 | 15.10 |
| vidhalluc | dflash | 0% | 100/100 | 91.0 (85.0–96.0) | 1.055 (1.042–1.070) | 1.035 | 1.627 | 7.6 | 2.016 | 13.3 | 1.017 | 15.10 |
| vidhalluc | ddtree | 100% | 100/100 | 91.0 (85.0–96.0) | 1.074 (1.055–1.096) | 1.044 | 1.729 | 7.7 | 2.445 | 13.0 | 1.014 | 15.10 |
| vidhalluc | ddtree | 50% | 100/100 | 90.0 (84.0–96.0) | 1.076 (1.056–1.099) | 1.045 | 1.769 | 7.7 | 2.505 | 13.0 | 1.014 | 15.10 |
| vidhalluc | ddtree | 25% | 100/100 | 92.0 (86.0–97.0) | 1.076 (1.057–1.098) | 1.047 | 1.789 | 7.7 | 2.484 | 13.0 | 1.014 | 15.10 |
| vidhalluc | ddtree | 0% | 100/100 | 93.0 (88.0–98.0) | 1.076 (1.058–1.098) | 1.047 | 1.817 | 7.8 | 2.439 | 13.1 | 1.014 | 15.10 |
| videochair | dflash | 100% | 100/100 | 27.0 (19.0–36.0) | 1.865 (1.733–2.018) | 1.687 | 2.291 | 68.3 | 2.926 | 163.3 | 1.118 | 9.86 |
| videochair | dflash | 50% | 100/100 | 28.0 (19.0–37.0) | 1.831 (1.706–1.978) | 1.674 | 2.236 | 65.1 | 2.804 | 157.8 | 1.050 | 9.86 |
| videochair | dflash | 25% | 100/100 | 30.0 (21.0–39.0) | 1.806 (1.698–1.923) | 1.649 | 2.171 | 68.5 | 2.911 | 164.4 | 1.092 | 9.86 |
| videochair | dflash | 0% | 100/100 | 27.0 (19.0–36.0) | 1.823 (1.706–1.952) | 1.696 | 2.198 | 66.8 | 2.801 | 161.2 | 1.063 | 9.86 |
| videochair | ddtree | 100% | 100/100 | 30.0 (21.0–39.0) | 2.016 (1.893–2.147) | 1.866 | 2.570 | 76.7 | 3.716 | 168.1 | 1.130 | 9.86 |
| videochair | ddtree | 50% | 100/100 | 27.0 (19.0–36.0) | 2.014 (1.893–2.142) | 1.835 | 2.573 | 75.3 | 3.689 | 164.4 | 1.098 | 9.86 |
| videochair | ddtree | 25% | 100/100 | 25.0 (17.0–34.0) | 2.044 (1.909–2.201) | 1.831 | 2.635 | 74.7 | 3.625 | 162.7 | 1.138 | 9.86 |
| videochair | ddtree | 0% | 99/100 | 19.2 (12.1–27.3) | 1.968 (1.854–2.089) | 1.837 | 2.477 | 74.6 | 3.547 | 165.7 | 1.103 | 9.87 |

![Speedup by dataset, method and visual retention](figures/speedup.png)

![Exact output match with target AR](figures/exact_match.png)

![Effective accepted tokens per verification round](figures/tau_effective.png)

## Proposal and tree efficiency

Proposal acceptance rate is total accepted draft proposals divided by total proposed positions across a condition. DDTree node-budget utilization is proposed nodes divided by tree budget × verification rounds. The per-depth proposed/accepted node counts are in `ddtree_depth_acceptance.csv`; DFlash has no branching-tree depth dimension.

| Dataset | Method | Retain | Mean proposals / round | Mean accepted proposals / round | Proposal acceptance | Mean emitted / round | Budget utilization | Mean max tree depth |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vdc | dflash | 100% | 15.000 | 1.389 | 0.093 | 2.389 | — | 0.000 |
| vdc | dflash | 50% | 15.000 | 1.387 | 0.092 | 2.387 | — | 0.000 |
| vdc | dflash | 25% | 15.000 | 1.347 | 0.090 | 2.347 | — | 0.000 |
| vdc | dflash | 0% | 15.000 | 1.323 | 0.088 | 2.323 | — | 0.000 |
| vdc | ddtree | 100% | 16.000 | 2.047 | 0.128 | 3.047 | 1.000 | 7.620 |
| vdc | ddtree | 50% | 16.000 | 2.042 | 0.128 | 3.042 | 1.000 | 8.100 |
| vdc | ddtree | 25% | 16.000 | 2.119 | 0.132 | 3.119 | 1.000 | 8.100 |
| vdc | ddtree | 0% | 16.000 | 1.988 | 0.124 | 2.988 | 1.000 | 8.160 |
| vidhalluc | dflash | 100% | 15.000 | 1.260 | 0.084 | 2.260 | — | 0.000 |
| vidhalluc | dflash | 50% | 15.000 | 1.265 | 0.084 | 2.265 | — | 0.000 |
| vidhalluc | dflash | 25% | 15.000 | 1.230 | 0.082 | 2.230 | — | 0.000 |
| vidhalluc | dflash | 0% | 15.000 | 1.177 | 0.078 | 2.177 | — | 0.000 |
| vidhalluc | ddtree | 100% | 16.000 | 1.804 | 0.113 | 2.804 | 1.000 | 6.500 |
| vidhalluc | ddtree | 50% | 16.000 | 1.815 | 0.113 | 2.815 | 1.000 | 6.650 |
| vidhalluc | ddtree | 25% | 16.000 | 1.773 | 0.111 | 2.773 | 1.000 | 6.670 |
| vidhalluc | ddtree | 0% | 16.000 | 1.768 | 0.110 | 2.768 | 1.000 | 6.670 |
| videochair | dflash | 100% | 15.000 | 1.973 | 0.132 | 2.973 | — | 0.000 |
| videochair | dflash | 50% | 15.000 | 1.841 | 0.123 | 2.841 | — | 0.000 |
| videochair | dflash | 25% | 15.000 | 1.962 | 0.131 | 2.962 | — | 0.000 |
| videochair | dflash | 0% | 15.000 | 1.853 | 0.124 | 2.853 | — | 0.000 |
| videochair | ddtree | 100% | 16.000 | 2.778 | 0.174 | 3.778 | 1.000 | 8.960 |
| videochair | ddtree | 50% | 16.000 | 2.748 | 0.172 | 3.748 | 1.000 | 8.850 |
| videochair | ddtree | 25% | 16.000 | 2.680 | 0.167 | 3.680 | 1.000 | 8.810 |
| videochair | ddtree | 0% | 16.000 | 2.608 | 0.163 | 3.608 | 1.000 | 8.859 |

## Task quality

Full metric-level means and sample-bootstrap intervals are in `results/ddtree_eval_20261019/analysis/quality_metrics.csv`. VDC uses dependency-light BLEU/ROUGE/overlap metrics against its local reference. VidHalluc accuracy is also split by task type. Sequence_Ranking scoring reads the first two distinct explicit Action A/B labels in generated prose (or a compact AB/BA label); one action label alone is unscorable. This scorer correction was applied to saved AR/speculative task annotations without changing inference outputs. VideoCHAIR caption overlap averages its local references; its `chair_i_reference_vocab` and `chair_s_reference_vocab` are lexical proxies based on the manifest's reference-caption vocabulary, not official human visual-object annotations.

| Dataset | Method | Retain | Task type | Metric | n | Mean | 95% CI |
|---|---|---:|---|---|---:|---:|---:|
| vdc | target_ar | AR baseline | all | bleu | 50 | 0.049 | 0.040–0.060 |
| vdc | target_ar | AR baseline | all | coverage | 50 | 0.316 | 0.288–0.345 |
| vdc | target_ar | AR baseline | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | target_ar | AR baseline | all | rouge_l | 50 | 0.193 | 0.183–0.203 |
| vdc | target_ar | AR baseline | all | unigram_f1 | 50 | 0.397 | 0.378–0.414 |
| vdc | dflash | 100% | all | bleu | 50 | 0.051 | 0.043–0.059 |
| vdc | dflash | 100% | all | coverage | 50 | 0.314 | 0.285–0.341 |
| vdc | dflash | 100% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | dflash | 100% | all | rouge_l | 50 | 0.197 | 0.188–0.206 |
| vdc | dflash | 100% | all | unigram_f1 | 50 | 0.405 | 0.388–0.420 |
| vdc | dflash | 50% | all | bleu | 50 | 0.050 | 0.043–0.058 |
| vdc | dflash | 50% | all | coverage | 50 | 0.311 | 0.284–0.338 |
| vdc | dflash | 50% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | dflash | 50% | all | rouge_l | 50 | 0.198 | 0.189–0.208 |
| vdc | dflash | 50% | all | unigram_f1 | 50 | 0.402 | 0.386–0.418 |
| vdc | dflash | 25% | all | bleu | 50 | 0.052 | 0.044–0.062 |
| vdc | dflash | 25% | all | coverage | 50 | 0.321 | 0.293–0.350 |
| vdc | dflash | 25% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | dflash | 25% | all | rouge_l | 50 | 0.199 | 0.188–0.209 |
| vdc | dflash | 25% | all | unigram_f1 | 50 | 0.404 | 0.386–0.420 |
| vdc | dflash | 0% | all | bleu | 50 | 0.049 | 0.042–0.056 |
| vdc | dflash | 0% | all | coverage | 50 | 0.319 | 0.292–0.346 |
| vdc | dflash | 0% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | dflash | 0% | all | rouge_l | 50 | 0.197 | 0.188–0.205 |
| vdc | dflash | 0% | all | unigram_f1 | 50 | 0.403 | 0.387–0.419 |
| vdc | ddtree | 100% | all | bleu | 50 | 0.051 | 0.043–0.059 |
| vdc | ddtree | 100% | all | coverage | 50 | 0.320 | 0.295–0.345 |
| vdc | ddtree | 100% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | ddtree | 100% | all | rouge_l | 50 | 0.197 | 0.187–0.206 |
| vdc | ddtree | 100% | all | unigram_f1 | 50 | 0.408 | 0.393–0.423 |
| vdc | ddtree | 50% | all | bleu | 50 | 0.048 | 0.041–0.056 |
| vdc | ddtree | 50% | all | coverage | 50 | 0.318 | 0.290–0.346 |
| vdc | ddtree | 50% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | ddtree | 50% | all | rouge_l | 50 | 0.198 | 0.189–0.208 |
| vdc | ddtree | 50% | all | unigram_f1 | 50 | 0.407 | 0.390–0.423 |
| vdc | ddtree | 25% | all | bleu | 50 | 0.047 | 0.040–0.055 |
| vdc | ddtree | 25% | all | coverage | 50 | 0.314 | 0.287–0.341 |
| vdc | ddtree | 25% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | ddtree | 25% | all | rouge_l | 50 | 0.197 | 0.188–0.206 |
| vdc | ddtree | 25% | all | unigram_f1 | 50 | 0.400 | 0.386–0.414 |
| vdc | ddtree | 0% | all | bleu | 50 | 0.048 | 0.041–0.055 |
| vdc | ddtree | 0% | all | coverage | 50 | 0.322 | 0.295–0.349 |
| vdc | ddtree | 0% | all | exact_match | 50 | 0.000 | 0.000–0.000 |
| vdc | ddtree | 0% | all | rouge_l | 50 | 0.201 | 0.192–0.210 |
| vdc | ddtree | 0% | all | unigram_f1 | 50 | 0.411 | 0.395–0.427 |
| vidhalluc | target_ar | AR baseline | all | task_correct | 100 | 0.780 | 0.690–0.860 |
| vidhalluc | target_ar | AR baseline | BinaryQA | task_correct | 25 | 0.680 | 0.480–0.840 |
| vidhalluc | target_ar | AR baseline | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | target_ar | AR baseline | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | target_ar | AR baseline | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | dflash | 100% | all | task_correct | 100 | 0.770 | 0.680–0.850 |
| vidhalluc | dflash | 100% | BinaryQA | task_correct | 25 | 0.640 | 0.440–0.800 |
| vidhalluc | dflash | 100% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | dflash | 100% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | dflash | 100% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | dflash | 50% | all | task_correct | 100 | 0.770 | 0.690–0.850 |
| vidhalluc | dflash | 50% | BinaryQA | task_correct | 25 | 0.640 | 0.440–0.800 |
| vidhalluc | dflash | 50% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | dflash | 50% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | dflash | 50% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | dflash | 25% | all | task_correct | 100 | 0.770 | 0.690–0.850 |
| vidhalluc | dflash | 25% | BinaryQA | task_correct | 25 | 0.640 | 0.440–0.840 |
| vidhalluc | dflash | 25% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | dflash | 25% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | dflash | 25% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | dflash | 0% | all | task_correct | 100 | 0.770 | 0.680–0.850 |
| vidhalluc | dflash | 0% | BinaryQA | task_correct | 25 | 0.640 | 0.440–0.840 |
| vidhalluc | dflash | 0% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | dflash | 0% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | dflash | 0% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | ddtree | 100% | all | task_correct | 100 | 0.780 | 0.700–0.860 |
| vidhalluc | ddtree | 100% | BinaryQA | task_correct | 25 | 0.680 | 0.480–0.840 |
| vidhalluc | ddtree | 100% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | ddtree | 100% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | ddtree | 100% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | ddtree | 50% | all | task_correct | 100 | 0.780 | 0.690–0.860 |
| vidhalluc | ddtree | 50% | BinaryQA | task_correct | 25 | 0.680 | 0.480–0.840 |
| vidhalluc | ddtree | 50% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | ddtree | 50% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | ddtree | 50% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | ddtree | 25% | all | task_correct | 100 | 0.780 | 0.700–0.860 |
| vidhalluc | ddtree | 25% | BinaryQA | task_correct | 25 | 0.680 | 0.480–0.840 |
| vidhalluc | ddtree | 25% | MCQ | task_correct | 35 | 0.914 | 0.800–1.000 |
| vidhalluc | ddtree | 25% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | ddtree | 25% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| vidhalluc | ddtree | 0% | all | task_correct | 100 | 0.780 | 0.700–0.860 |
| vidhalluc | ddtree | 0% | BinaryQA | task_correct | 25 | 0.680 | 0.480–0.840 |
| vidhalluc | ddtree | 0% | MCQ | task_correct | 35 | 0.914 | 0.828–1.000 |
| vidhalluc | ddtree | 0% | Scene_Transition | task_correct | 20 | 0.650 | 0.450–0.850 |
| vidhalluc | ddtree | 0% | Sequence_Ranking | task_correct | 20 | 0.800 | 0.600–0.950 |
| videochair | target_ar | AR baseline | all | chair_i_reference_vocab | 100 | 0.734 | 0.707–0.759 |
| videochair | target_ar | AR baseline | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | dflash | 100% | all | chair_i_reference_vocab | 100 | 0.734 | 0.706–0.761 |
| videochair | dflash | 100% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | dflash | 50% | all | chair_i_reference_vocab | 100 | 0.730 | 0.703–0.756 |
| videochair | dflash | 50% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | dflash | 25% | all | chair_i_reference_vocab | 100 | 0.737 | 0.709–0.763 |
| videochair | dflash | 25% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | dflash | 0% | all | chair_i_reference_vocab | 100 | 0.733 | 0.705–0.760 |
| videochair | dflash | 0% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | ddtree | 100% | all | chair_i_reference_vocab | 100 | 0.726 | 0.697–0.752 |
| videochair | ddtree | 100% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | ddtree | 50% | all | chair_i_reference_vocab | 100 | 0.735 | 0.708–0.760 |
| videochair | ddtree | 50% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | ddtree | 25% | all | chair_i_reference_vocab | 100 | 0.742 | 0.716–0.767 |
| videochair | ddtree | 25% | all | chair_s_reference_vocab | 100 | 1.000 | 1.000–1.000 |
| videochair | ddtree | 0% | all | chair_i_reference_vocab | 99 | 0.731 | 0.702–0.758 |
| videochair | ddtree | 0% | all | chair_s_reference_vocab | 99 | 1.000 | 1.000–1.000 |

## Paired quality change vs target AR

Positive values mean the speculative condition's metric is higher than paired AR; for hallucination proxies, lower values are preferable. The full per-sample paired deltas are in `paired_quality_deltas_vs_ar.csv`.

| Dataset | Method | Retain | Task type | Metric | n paired | Δ vs AR (95% CI) |
|---|---|---:|---|---|---:|---:|
| vdc | dflash | 100% | all | bleu | 50 | 0.001 (-0.005–0.007) |
| vdc | dflash | 100% | all | coverage | 50 | -0.003 (-0.019–0.015) |
| vdc | dflash | 100% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | dflash | 100% | all | rouge_l | 50 | 0.004 (-0.000–0.008) |
| vdc | dflash | 100% | all | unigram_f1 | 50 | 0.008 (-0.001–0.019) |
| vdc | dflash | 50% | all | bleu | 50 | 0.001 (-0.005–0.006) |
| vdc | dflash | 50% | all | coverage | 50 | -0.005 (-0.022–0.011) |
| vdc | dflash | 50% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | dflash | 50% | all | rouge_l | 50 | 0.005 (0.000–0.010) |
| vdc | dflash | 50% | all | unigram_f1 | 50 | 0.006 (-0.004–0.017) |
| vdc | dflash | 25% | all | bleu | 50 | 0.003 (-0.002–0.008) |
| vdc | dflash | 25% | all | coverage | 50 | 0.005 (-0.014–0.025) |
| vdc | dflash | 25% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | dflash | 25% | all | rouge_l | 50 | 0.005 (0.000–0.011) |
| vdc | dflash | 25% | all | unigram_f1 | 50 | 0.007 (-0.001–0.017) |
| vdc | dflash | 0% | all | bleu | 50 | -0.001 (-0.007–0.004) |
| vdc | dflash | 0% | all | coverage | 50 | 0.003 (-0.013–0.019) |
| vdc | dflash | 0% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | dflash | 0% | all | rouge_l | 50 | 0.003 (-0.002–0.009) |
| vdc | dflash | 0% | all | unigram_f1 | 50 | 0.007 (-0.001–0.015) |
| vdc | ddtree | 100% | all | bleu | 50 | 0.001 (-0.004–0.006) |
| vdc | ddtree | 100% | all | coverage | 50 | 0.004 (-0.012–0.020) |
| vdc | ddtree | 100% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | ddtree | 100% | all | rouge_l | 50 | 0.003 (-0.002–0.009) |
| vdc | ddtree | 100% | all | unigram_f1 | 50 | 0.012 (0.002–0.023) |
| vdc | ddtree | 50% | all | bleu | 50 | -0.001 (-0.008–0.006) |
| vdc | ddtree | 50% | all | coverage | 50 | 0.001 (-0.013–0.017) |
| vdc | ddtree | 50% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | ddtree | 50% | all | rouge_l | 50 | 0.005 (-0.000–0.010) |
| vdc | ddtree | 50% | all | unigram_f1 | 50 | 0.010 (-0.000–0.023) |
| vdc | ddtree | 25% | all | bleu | 50 | -0.002 (-0.010–0.004) |
| vdc | ddtree | 25% | all | coverage | 50 | -0.002 (-0.016–0.012) |
| vdc | ddtree | 25% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | ddtree | 25% | all | rouge_l | 50 | 0.003 (-0.002–0.009) |
| vdc | ddtree | 25% | all | unigram_f1 | 50 | 0.004 (-0.008–0.017) |
| vdc | ddtree | 0% | all | bleu | 50 | -0.001 (-0.007–0.005) |
| vdc | ddtree | 0% | all | coverage | 50 | 0.006 (-0.011–0.023) |
| vdc | ddtree | 0% | all | exact_match | 50 | 0.000 (0.000–0.000) |
| vdc | ddtree | 0% | all | rouge_l | 50 | 0.008 (0.003–0.013) |
| vdc | ddtree | 0% | all | unigram_f1 | 50 | 0.015 (0.004–0.028) |
| vidhalluc | dflash | 100% | BinaryQA | task_correct | 25 | -0.040 (-0.120–0.000) |
| vidhalluc | dflash | 100% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 100% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 100% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 50% | BinaryQA | task_correct | 25 | -0.040 (-0.120–0.000) |
| vidhalluc | dflash | 50% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 50% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 50% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 25% | BinaryQA | task_correct | 25 | -0.040 (-0.120–0.000) |
| vidhalluc | dflash | 25% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 25% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 25% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 0% | BinaryQA | task_correct | 25 | -0.040 (-0.120–0.000) |
| vidhalluc | dflash | 0% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 0% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | dflash | 0% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 100% | BinaryQA | task_correct | 25 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 100% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 100% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 100% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 50% | BinaryQA | task_correct | 25 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 50% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 50% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 50% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 25% | BinaryQA | task_correct | 25 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 25% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 25% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 25% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 0% | BinaryQA | task_correct | 25 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 0% | MCQ | task_correct | 35 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 0% | Scene_Transition | task_correct | 20 | 0.000 (0.000–0.000) |
| vidhalluc | ddtree | 0% | Sequence_Ranking | task_correct | 20 | 0.000 (0.000–0.000) |
| videochair | dflash | 100% | all | chair_i_reference_vocab | 100 | 0.000 (-0.013–0.013) |
| videochair | dflash | 100% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | dflash | 50% | all | chair_i_reference_vocab | 100 | -0.004 (-0.017–0.009) |
| videochair | dflash | 50% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | dflash | 25% | all | chair_i_reference_vocab | 100 | 0.003 (-0.007–0.014) |
| videochair | dflash | 25% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | dflash | 0% | all | chair_i_reference_vocab | 100 | -0.001 (-0.012–0.012) |
| videochair | dflash | 0% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | ddtree | 100% | all | chair_i_reference_vocab | 100 | -0.008 (-0.020–0.003) |
| videochair | ddtree | 100% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | ddtree | 50% | all | chair_i_reference_vocab | 100 | 0.001 (-0.011–0.012) |
| videochair | ddtree | 50% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | ddtree | 25% | all | chair_i_reference_vocab | 100 | 0.009 (-0.003–0.021) |
| videochair | ddtree | 25% | all | chair_s_reference_vocab | 100 | 0.000 (0.000–0.000) |
| videochair | ddtree | 0% | all | chair_i_reference_vocab | 99 | -0.002 (-0.013–0.009) |
| videochair | ddtree | 0% | all | chair_s_reference_vocab | 99 | 0.000 (0.000–0.000) |

## Paired DDTree minus DFlash contrasts

Each contrast uses only shared successful sample IDs at the same visual cut. The interval is bootstrapped across paired samples; a CI containing zero gives no clear evidence of a difference for that metric.

| Dataset | Retain | n paired | Δ E2E speedup (95% CI) | Δ decode speedup (95% CI) | Δ tau eff. (95% CI) | Δ exact-match rate (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| vdc | 100% | 50 | 0.154 (0.094–0.216) | 0.316 (0.172–0.442) | 0.655 (0.592–0.713) | -0.080 (-0.180–0.000) |
| vdc | 50% | 50 | 0.189 (0.120–0.271) | 0.378 (0.230–0.530) | 0.663 (0.584–0.746) | 0.000 (0.000–0.000) |
| vdc | 25% | 50 | 0.202 (0.096–0.329) | 0.459 (0.255–0.689) | 0.754 (0.648–0.866) | 0.020 (-0.060–0.100) |
| vdc | 0% | 50 | 0.194 (0.119–0.312) | 0.407 (0.277–0.600) | 0.660 (0.572–0.742) | 0.000 (-0.060–0.060) |
| vidhalluc | 100% | 100 | 0.021 (0.013–0.032) | 0.165 (0.087–0.246) | 0.427 (0.317–0.539) | -0.010 (-0.050–0.030) |
| vidhalluc | 50% | 100 | 0.024 (0.014–0.035) | 0.178 (0.096–0.267) | 0.462 (0.344–0.586) | -0.020 (-0.060–0.020) |
| vidhalluc | 25% | 100 | 0.023 (0.013–0.036) | 0.170 (0.090–0.255) | 0.463 (0.347–0.586) | 0.000 (-0.040–0.040) |
| vidhalluc | 0% | 100 | 0.022 (0.013–0.032) | 0.190 (0.108–0.273) | 0.428 (0.331–0.525) | 0.020 (-0.030–0.080) |
| videochair | 100% | 100 | 0.151 (0.087–0.214) | 0.279 (0.144–0.397) | 0.790 (0.659–0.950) | 0.030 (-0.060–0.120) |
| videochair | 50% | 100 | 0.183 (0.115–0.250) | 0.337 (0.193–0.463) | 0.885 (0.735–1.060) | -0.010 (-0.110–0.090) |
| videochair | 25% | 100 | 0.238 (0.180–0.300) | 0.464 (0.347–0.597) | 0.714 (0.541–0.864) | -0.050 (-0.150–0.050) |
| videochair | 0% | 99 | 0.161 (0.111–0.208) | 0.300 (0.213–0.386) | 0.771 (0.694–0.855) | -0.071 (-0.162–0.020) |

## Timing and resource interpretation

Per-condition component means (prefill, draft, tree build/compile, verification, commit, decode and total wall time), per-device peak allocated VRAM, raw output IDs/text, all round traces, and AR outputs are retained in the results directory. Eager attention was selected after the pilot showed improved parity in some cases; it remains the same backend for all full-run conditions. Peak memory is allocated tensor memory, summed across visible GPUs for the aggregate column; it is not total device reservation. Retention only changes which visual-context hidden states are supplied to the drafter: the target still prefills the full video and retains its full cache, so peak VRAM is expected to stay nearly constant. The VideoCHAIR DDTree 0% mean is 9.874 GiB versus 9.864 GiB at 25% because that cell is missing `video9691`, whose 25% peak was 8.873 GiB; across the 99 paired samples, the 0%-minus-25% mean peak-memory difference is effectively zero (-2.4e-8 GiB). The displayed 0.01 GiB increase is a sample-composition effect, not a retention effect.

The token-exactness pilot did not pass the locked 100% parity guardrail on captioning workloads. Full-run quality and exactness are therefore reported as measured. A positive speedup alone does not establish an equivalent answer.

## Machine-readable artifacts

- Raw AR baselines: `results/ddtree_eval_20261019/<dataset>/target_ar.jsonl`.
- Raw DFlash/DDTree outputs, token IDs, timing, task scores and per-GPU memory: `results/ddtree_eval_20261019/<dataset>/speculative.jsonl`.
- Round acceptance/tree traces: `results/ddtree_eval_20261019/<dataset>/round_traces.jsonl`.
- Aggregated tables/integrity report: `results/ddtree_eval_20261019/analysis/`.
- Run arguments and checkpoint/model provenance: `results/ddtree_eval_20261019/run_summary.json`.
