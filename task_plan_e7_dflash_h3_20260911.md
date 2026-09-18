# Plan: DFlash E7 and H3.1 follow-up

## Goal

Run a controlled DFlash version of E7 for H2.1, add DFlash attention
allocation logs for Natural versus Answer-hint prompts, and replace the H3.1
MSD `n=2` position pilot with a full-cohort DFlash position study.

## Experiment cards

### E7-DFlash / H2.1

- Question: Does adding reference-answer semantics to the prompt reduce the
  apparent acceptance gain caused by reducing/removing visual context?
- Hypothesis: Answer-hint makes the `Deleted - Full` acceptance gap smaller
  than Natural; answer-hint attention mass should concentrate on the appended
  answer text if the drafter uses it.
- Baseline: Qwen2.5-VL-3B target + the selected local DFlash checkpoint,
  Natural prompt, draft visual retention Full.
- Variants: prompt `{Natural, Answer-hint}` crossed with draft visual
  retention `{100%, 25%, 0%}`. Target input remains full visual context.
- Primary metric: paired factorial interaction on DFlash `tau_effective` or
  equivalent mean accepted proposals, with `Deleted - Full` computed within
  each prompt variant.
- Guardrails: target/draft exact-token parity, lossless rate, target output
  hash, output length, latency, and no error/unsupported rows.
- Dataset: full local VDC50 test manifest; reference answer is used only as
  an explicitly labelled oracle diagnostic and never as benchmark accuracy.
- Seed: greedy decoding, temperature 0; no stochastic seed sweep.
- Budget: pilot 2 samples first; full 50-sample factorial after pilot passes;
  reserve roughly 2 hours wall-clock on the current two-GPU host.
- Success criterion: interaction is negative with a paired/bootstrap CI that
  excludes zero and the answer-hint attention mass is measurably above the
  Natural answer-region mass. This confirms only the alignment mechanism;
  answer quality remains a separate metric.
- Exploratory-only: attention mass alone cannot prove semantic use or answer
  correctness.

### H3.1-DFlash position study

- Question: Does the acceptance gain from visual cut appear only after the
  target has generated roughly half of the answer?
- Hypothesis: early-position acceptance for Cut/Zero is below Full, while the
  late-position gain is positive; the late-minus-early difference should be
  positive and stable.
- Baseline/variants: same DFlash target/checkpoint, same prompt and full
  target visual context, with draft conditions Full, Zero-value and Cut.
- Primary metric: paired position-wise acceptance delta relative to Full,
  plus a pre-registered early (0--25%) versus late (50--100%) interaction.
- Guardrails: full acceptance traces, first-reject positions, lossless rate,
  target output hashes and exact sample/condition coverage.
- Dataset: full local VDC50 test manifest; add MVBench only if the DFlash
  prompt/loader path passes a separate smoke test.
- Seed: greedy decoding, temperature 0; no stochastic seed sweep.
- Budget: 2-sample smoke, then full VDC50; position bins are reported with
  bootstrap confidence intervals over samples.
- Success criterion: H3.1 is supported only if the early/late interaction is
  positive with CI excluding zero and the direction is consistent across
  neighboring bins. Any gain already present in the first bin rejects the
  simple late-onset story.

## Feedback ladder

- R0: this protocol and decision rules written down.
- R1: imports, model/checkpoint/config/manifest preflight.
- R2: one-batch DFlash decode plus attention hook; finite metrics and valid
  answer-token position mapping.
- R4/R5: 2-sample pilot for E7 and H3.1; stop on mismatch, unsupported rows,
  or missing attention records.
- R6: full VDC50 runs for both studies.
- R7: paired statistics, figures, report reconstruction, and explicit status
  for each hypothesis.

## Current status

- [x] Existing MSD E7 and MSD H3.1 pilot audited.
- [x] Confirm the intended local DFlash checkpoint and GPU visibility.
- [x] Implement prompt-variant, position-trace and attention-region logging.
- [x] Run smoke/pilot.
- [x] Run full DFlash E7 and H3.1.
- [x] Analyze, visualize and update the report.
- [x] Audit and repair the 9 E7 rows with empty answer-hint spans caused by
  BPE boundary matching; preserve provenance in the corrected journal.

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Skill alias path initially used as a literal directory | 1 | Re-read skills using their resolved filesystem paths |
| GPU visibility unavailable in the current tool-side probe | 1 | Recheck through the approved host command before any GPU launch |
