# Reframed H2.1/H3.1/H3.2 experiment plan — updated 2026-09-22

## Current phase boundary: Phase 1 is raw-data collection only

This run must only construct the required datasets, execute the locked
conditions, and preserve complete per-sample/per-position raw records. It must
not compute higher-level metrics or compare hypotheses. In particular, Phase 1
does **not** compute `I_format`, early/late contrasts, depth interactions,
bootstrap confidence intervals, plots, or hypothesis decisions. Those belong to
Phase 2 after raw-data coverage and schema audits pass.

Phase 1 may record runtime-native quantities such as accepted token counts,
proposal counts, token IDs/text, timing, hashes, accuracy fields, and attention
arrays when available. Recording a field is different from aggregating or
interpreting it.

## Purpose

This plan incorporates the revised interpretation:

1. MVBench is multiple-choice and exposes candidate answer text, so visual
   deletion may improve draft acceptance through text-side answer leakage.
2. VDC is free-form captioning, so deleting visual tokens may remove information
   needed to answer rather than merely changing attention allocation.
3. MSD's apparent Cut gain may be position-dependent, with the already-generated
   answer text carrying the second half of the output.
4. DFlash depth 1/3/5 is an indirect test of whether draft depth determines
   whether visual values are useful to the drafter.

## Experiment card A — H2.1 answer leakage and VDC format

**Question:** Does converting VDC to a controlled multiple-choice format make its
visual-ablation acceptance effect resemble MVBench?

**Hypothesis:** VDC-MCQ will have a larger/non-negative Cut-or-Zero gain than
natural free-form VDC because the option text supplies answer candidates. The
natural VDC caption prompt should show a smaller or negative visual-ablation
effect.

**Baseline:** Existing VDC50 free-form prompt, target Qwen2.5-VL-3B-Instruct,
the selected DFlash checkpoint, and the current Full/Zero/Cut draft-side
interventions.

**Variant:** Replace only the VDC task/prompt with a VDC-MCQ manifest using the
same canonical `Question: ... Option: (A)...` format used by MVBench. The
rendered template is fixed as:

```text
Question:<question>
Option:
(A) <candidate A>
(B) <candidate B>
(C) <candidate C>
(D) <candidate D>
Only give the best option.
```

Do not use the existing Answer-hint prompt as the VDC-MCQ variant. The
reference caption is metadata used to construct/audit the correct option; it is
never appended to the inference prompt.

**Phase-1 raw collection:** Do not select or compute a primary comparison yet.
For every sample/format/condition, record the runtime-native acceptance and
performance fields, including `tau_effective`, `accepted_effective_tokens`,
accepted prefix, proposal/accepted counts, output-position traces, per-round
records, output token IDs/text, prompt token counts, visual-token counts,
prefill/decode/verification/end-to-end timing, throughput, losslessness,
target/speculative hashes, and runtime errors. Preserve the option metadata and
target/draft parsed answers for VDC-MCQ.

**Deferred to Phase 2:** acceptance-rate summaries, format interactions,
`I_format`, confidence intervals, plots, and any answer-leakage conclusion.

**Dataset:** The existing 50 VDC videos, with a new versioned VDC-MCQ JSONL.
Every row must contain `sample_id`, `video`, `question`, `candidates`,
`correct_option`, source reference answer, distractor provenance, and an audit
status. The correct option must be unambiguous; distractors should be hard
enough that the task cannot be solved by length or lexical artifacts.

**Phase-1 stop condition:** Stop only for data/schema problems, missing raw
fields, target-output non-parity, invalid option annotations, or runtime
errors. Do not decide whether answer leakage is supported during this phase.

### VDC-MCQ construction gate

Before GPU inference:

1. Choose a fixed question template that asks one objectively answerable event,
   action, or scene fact per VDC video.
2. Create one correct option from the verified VDC reference caption, while
   keeping that caption out of the prompt.
3. Create three hard distractors from the same semantic category, avoiding
   answer-length, punctuation, and lexical-overlap shortcuts.
4. Audit every row for exactly one correct option and record the provenance.
5. Add loader and schema tests before a one-sample parity run.

If a reliable VDC-MCQ annotation cannot be produced, stop this branch and report
that the format-confound hypothesis is not testable with the current VDC labels;
do not substitute the old oracle Answer-hint experiment.

## Experiment card B — H3.1 MSD position-wise gain

**Question:** Is the MSD `retention-0` acceptance gain negative in the first
half of the answer and positive only after the target has generated half of it?

**Hypothesis:** `Delta_early < 0`, `Delta_late > 0`, and
`Delta_late − Delta_early > 0` for `retention-0` versus `retention-100` on MSD.

**Baseline/variant:** Same MSD/Qwen2-VL-7B runtime, VDC50 cohort, natural VDC
prompt, preprocessing, and target full-visual context. The primary MSD sweep
uses draft visual retention exactly at `100, 50, 25, 0%`; `100%` is the full
visual baseline and `0%` is the no-visual endpoint. Do not replace this sweep
with the DFlash `Zero/Cut` conditions.

**Phase-1 raw collection:** For every sample and retention level, record the
complete acceptance trace with absolute target output positions: round ID,
round start, proposal token IDs/text, target token IDs/text, proposal count,
accepted count, accepted prefix, first reject position, effective emitted
tokens, verification steps, and `acceptance_by_position`. Also record output
IDs/text, losslessness, target/speculative hashes, visual-token counts, target
fingerprints, timing, throughput, and runtime errors.

**Deferred to Phase 2:** absolute/normalized position summaries, early/late
contrasts, bootstrap intervals, attention follow-up, and the H3.1 decision.

**Phase-1 stop condition:** Stop for incomplete traces, missing token/text
records, target fingerprint/output mismatch, invalid retention conditions, or
runtime errors; do not label the hypothesis true/false from the collection run.

**Gated follow-up:** Only after the position result shows a reliable first/second
half difference, run a small detailed attention probe to inspect whether later
queries attend more to generated text. This probe is explanatory, not the
primary H3.1 test.

## Experiment card C — H3.2 DFlash depth

**Question:** Does visual-value sensitivity increase when DFlash depth increases
from 1 to 3 to 5 layers, within the same target/task/corpus protocol?

**Hypothesis:** `Full−Zero` increases with depth if deeper DFlash drafts use
visual values more effectively. A Cut gain at depth 1 but a Cut penalty at depth
3/5 is consistent with the MSD-versus-DFlash explanation, but is not sufficient
without the paired depth interaction.

**Baseline/variants:** Same target, corpus, VDC/MVBench protocol, visual token
calibration, seed policy, target layers, and intervention; only DFlash draft
depth/checkpoint changes: 1, 3, 5. This is a within-DFlash depth test; it does
not by itself prove that MSD is worse because its architecture has one layer.

**Phase-1 raw collection:** For every dataset/corpus/depth/sample/condition,
record the full raw journal: `Full/Zero/Cut` condition and masks, draft depth
and checkpoint identity, prompt/input fingerprints, visual-token positions and
counts, target/speculative output IDs/text, per-round acceptance records,
`acceptance_by_position`, accepted/proposed counts, `tau` fields, timing,
throughput, memory/parameter metadata when available, losslessness, output
hashes, target parity, and runtime errors.

**Deferred to Phase 2:** `Full−Zero`, `Full−Cut`, depth interactions, corpus
pooling, bootstrap intervals, plots, and the interpretation of visual use versus
general language-modeling quality.

**Phase-1 stop condition:** Stop for missing depth/checkpoint metadata,
condition leakage, target-output mismatch, incomplete raw traces, or runtime
errors. Do not decide the depth hypothesis during this collection phase.

## Promotion ladder

### R0 — protocol and manifest freeze

Freeze VDC-MCQ schema, distractor audit, visual conditions, target model,
checkpoint, sample IDs, raw-field schema, retention grid, and output directory.
Do not freeze or compute higher-level comparison metrics in Phase 1.

### R1 — static and schema checks

Validate JSONL fields, unique IDs, one correct option, candidate count, video
paths, prompt rendering, and output schema. Add/update focused loader tests.

### R2 — one-sample parity

Run natural VDC, VDC-MCQ, MSD retention `100/50/25/0`, and DFlash depth smoke
paths on one sample. Require target-prefill parity, finite metrics, correct
option parsing, complete raw token/text traces, and valid position traces. This
is a schema/runtime check, not a metric comparison.

### R3 — five-sample pilot

Check raw-record completeness: option metadata, acceptance traces, target
hashes, visual-token counts, output text/IDs, timing fields, and no-error
coverage. Stop if distractors are invalid or prompt rendering changes
unexpectedly; do not rank conditions.

### R4 — VDC50 H2.1/H3.1 cohort

Run VDC natural and VDC-MCQ with Full/Zero/Cut, plus MSD H3.1 retention
`100/50/25/0` on the same VDC50 sample IDs. Keep the backend/model artifacts
separate. Write raw JSONL journals and manifests only; defer all summaries and
contrasts.

### R5 — H3.2 full raw-data collection

Complete all `2 corpora × 3 depths × 2 datasets` cells with the fixed
Full/Zero/Cut protocol: 6 VDC cells and 6 MVBench cells. The planned row count
is 900 VDC rows (`50 × 3 × 2 × 3`) plus 18,000 MVBench rows
(`1,000 × 3 × 2 × 3`) = 18,900 rows. Require full planned raw coverage and
schema audits; do not compute pooled or cross-cell conclusions in Phase 1.

### R6 — raw-data audit handoff

Produce raw-data manifests, coverage/completeness reports, schema validation,
and a handoff note listing missing fields or failed rows. Higher-level CSV
summaries, figures, bootstrap analysis, and hypothesis decisions belong to
Phase 2.

## Historical evidence carried into this plan

The following values are historical artifacts only. Phase 1 does not recompute,
pool, compare, or reinterpret them; they are retained as provenance for why the
raw fields below are needed.

- Existing MSD H2.1 Natural/Answer-hint is an oracle alignment diagnostic, not
  VDC-MCQ. Its interaction was `−0.2776` with CI
  `[−0.4094, −0.1411]`; it is not evidence for the corrected VDC-MCQ study.
- Existing DFlash E7 interaction was `+0.0957` with CI
  `[+0.0302, +0.1542]`, also alignment-only and target-prompt-changing.
- Existing DFlash H3.1 VDC50 did not show late-only gain: Cut interaction
  `+0.0053`, CI `[-0.0058,+0.0165]`.
- Existing H3.2 VDC depth interaction is corpus-dependent: LLaVA D3−D1
  `+0.1261`, ShareGPT D3−D1 `−0.0540`; this is why MVBench and corpus-specific
  analysis remain necessary.

## Files and artifacts to update when execution begins

- Dataset/schema: `src/analyze/Validate_Sparrow_hypothesises/dataset.py` and a
  versioned VDC-MCQ manifest under `results/` or the configured data directory.
- Runtime: `src/analyze/Validate_Sparrow_hypothesises/run_dflash_h21_h31.py`
  and `run_msd.py`, only after the R0 design is approved. The DFlash runner
  still needs a VDC-MCQ prompt-format path, and the MSD launch must pass only
  `--retention-percentages 100 50 25 0` for the primary H3.1 sweep.
- Analysis: `analyze_dflash_h21_h31.py`, `analyze_h32_depth.py`, plus a focused
  VDC-MCQ/H3.1 statistics module if the existing schemas cannot represent it.
- Tests: loader/prompt schema, option correctness, position pairing, target hash
  consistency, and depth interaction fixtures.
- Report: update the hypothesis report only after the new artifacts pass the
  coverage and guardrail audits.
