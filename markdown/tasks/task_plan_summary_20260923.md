# Current task: prepare the corrected VDC-MCQ H2.1 study (2026-09-22)

## Phase boundary

The current run is **Phase 1 raw-data construction and collection only**. It
must preserve complete per-sample, per-round, per-position and output text/token
records for H2.1, H3.1 and H3.2. It must not compute higher-level metrics,
format/depth/early-late interactions, bootstrap intervals, plots, or hypothesis
decisions. Those are Phase 2 work after raw-data completeness audits.

## Execution status — 2026-09-22

- H2.1 raw collection is complete for both local depth-3 DFlash corpora:
  300 `ok` rows each, covering 50 VDC samples × natural/MCQ × Full/Zero/Cut.
- H3.2 existing VDC/MVBench journals were copied and enriched with decoded
  token text in `results/phase1_raw_20260922/h32_enriched/`.
- H3.1 full raw collection is complete on MSD with the official Qwen2-VL-7B
  target/draft pair, VDC50, visual target 25K, and retention `100/50/25/0`:
  200 measured rows across 50 samples. H3.2 depth-5 MVBench collection is
  complete for both LLaVA68K and ShareGPT68K, using the validated 6e
  checkpoints; each cell has 3,000 `ok` rows.
- Final raw-only coverage is recorded in
  `results/phase1_raw_20260922/phase1_raw_collection_audit.json`.

## Required correction

H2.1 must use a **VDC multiple-choice dataset with the same prompt structure as
MVBench**. The previous `answer_hint` branch, which appended the reference
caption to the prompt, is not the H2.1 experiment and must not be used as its
replacement.

## H2.1 execution plan

- Build a versioned VDC-MCQ manifest from the same VDC50 videos.
- Each row contains the video, a closed question, exactly four candidates
  `(A)`--`(D)`, one verified `correct_option`, the source reference caption,
  distractor provenance, and an audit status.
- Render the prompt exactly in the MVBench shape:

  ```text
  Question:<question>
  Option:
  (A) <candidate A>
  (B) <candidate B>
  (C) <candidate C>
  (D) <candidate D>
  Only give the best option.
  ```

- Do not append the reference caption or any oracle answer text to the prompt.
- Keep question, option text, option order, and target visual input identical
  across `Full`, `Zero`, and `Cut`; only the draft-side visual intervention
  changes.
- Run the paired VDC-MCQ `Full/Zero/Cut` experiment first. Retain natural VDC
  with the same videos as a recommended format control, but do not call the old
  `answer_hint` branch VDC-MCQ.

## H2.1 recorded outputs

For every sample and condition, preserve the rendered prompt metadata, option
labels/text, correct option provenance, target/speculative output IDs/text,
acceptance fields emitted by the runtime, per-position acceptance,
proposal/target text, timing, throughput, visual-token counts, target output
hash, and runtime status/error. Answer parsing, accuracy, summaries and plots
are generated only in Phase 2 after coverage and target-parity audits pass.

## H2.1 files to add or update

- Add a VDC-MCQ manifest builder and audit report; never overwrite the original
  VDC manifest.
- Extend `src/analyze/Validate_Sparrow_hypothesises/dataset.py` with the
  MVBench-compatible VDC-MCQ prompt renderer while retaining VDC video
  preprocessing/calibration.
- Extend `src/analyze/Validate_Sparrow_hypothesises/run_dflash_h21_h31.py` to
  load the VDC-MCQ schema and record answer/format metadata.
- Add focused tests for schema validation, exact prompt rendering, option
  uniqueness, answer parsing, and target-output parity.
- Only after these checks pass, run the one-sample, five-sample, and VDC50
  pilots. The one-sample schema smoke and both VDC50 raw runs are now complete;
  remaining blocked cells are recorded above.

## H2.1 decision boundary

Compare VDC-MCQ against natural VDC using the same visual conditions. A
MVBench-like sign or reduced penalty in VDC-MCQ supports answer leakage as a
contributor; otherwise answer leakage alone is insufficient. Acceptance length
is not interpreted as task quality without correct-option accuracy and target
parity.

---

# Current task: complete H3.1/H3.2 report and figures (2026-09-15)

## Goal

Update `markdown/bao_cao_tong_hop_thuc_nghiem_hypothesis_20260910.md` so H3.1
and H3.2 are organized by hypothesis, backed by the correct full-run artifacts,
and accompanied by figures that directly answer their decision statistics.
Keep incomplete MVBench H3.2 progress separate from confirmed VDC results and
exclude all MSD position pilot figures with `n=2` from the main evidence.

## Phases

- [complete] 1. Audit H3.1/H3.2 artifacts and current MVBench coverage.
- [complete] 2. Generate direct hypothesis figures and verify their statistics.
- [complete] 3. Patch the main report, links, provenance, and conclusion status.
- [complete] 4. Run focused verification and report exact confirmed/incomplete scope.

## Decisions / constraints

- H3.1 DFlash full: VDC50, `n=50`, 150/150 valid rows; no pooling with MSD `n=2`.
- H3.2 primary metric: sample-level `accepted_effective_tokens`; primary contrast
  is `(Full−Zero)_depth3 − (Full−Zero)_depth1`.
- H3.2 VDC is complete at 600/600 rows; MVBench is not reportable until all four
  corpus×depth runs have full 12,000-row coverage.
- Generated figures remain under `results/`; do not modify or delete user artifacts.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Existing planning files described an unrelated baseline-smoke task | 1 | Preserve historical content below; prepend this task-specific plan |

---

# Baseline Inference Smoke/Debug Plan

## Goal

Use only the existing `.venv` and the server's available GPU to make every
baseline inference launcher in scope runnable, then verify each with a one-
sample smoke test.

## Constraints

- Do not create or select `.venv-msd` or any other environment.
- Do not overwrite unrelated user changes already present in the worktree.
- Do not run full datasets or training; use one sample for runtime smoke tests.
- Keep generated outputs under `results/` and do not commit model/data artifacts.

## Phases

- [completed] 1. Verify workspace routing, `.venv`, CUDA, models/data, and dirty worktree.
- [completed] 2. Inventory baseline inference launchers and run static/CLI preflight.
- [pending] 3. Reproduce runtime failures with one sample and record component-boundary evidence.
- [completed] 4. Add failing regression tests and apply minimal root-cause fixes one at a time.
- [blocked] 5. Re-run focused tests, all repository tests, and one-sample GPU smoke checks.
- [in_progress] 6. Report verified baselines, artifacts, remaining blockers, and exact commands.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Current tool container reported no NVIDIA driver during initial inspection | 1 | Host-level `nvidia-smi` later confirmed two GPUs (3090 + A4000) |
| Required `./.venv` is absent on both workspace and host; requested project path `~/fast_infer_text_sum` is also not mounted here | 2 | Added fail-fast shared routing; runtime pytest/inference waits for the requested workspace mount |

## Current blocker

The tool host is `/path/to/local-resource`, while the supplied shell prompt is
for `tuantb@teslaT4:~/fast_infer_text_sum`. Host privilege exposes RTX 3090 and
RTX A4000, not the reported T4, and the required `.venv` is absent. Code routing
and static checks are complete; no model process or pytest run has been started
because doing so would use an unauthorized interpreter.
# Current task: relocate report images and repair markdown links (2026-09-17)

## Goal

Complete the report delivery folder `markdown/Baocao_17_9_26` by copying the
referenced figures into that folder and changing report-relative image/data
links so they resolve from the relocated markdown file.

## Phases

- [complete] 1. Inventory report links and available source artifacts.
- [complete] 2. Create the delivery folder and copy required images.
- [complete] 3. Patch links in the copied report and validate every local link.
- [complete] 4. Verify the final folder contents and summarize changes.

## Constraints

- Preserve the original report and all unrelated user changes.
- Copy figures; do not delete or move source artifacts under `results/`.
- Keep non-image data links valid from the relocated report when possible.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| Checksum audit used an incorrect report filename | 1 | Re-run with the actual delivery report filename |
| Four PNG basenames occur in multiple result directories | 2 | Validate checksum against at least one source and retain the explicit source mapping |
| Final checker matched `../results/` as a substring inside `../../results/` | 3 | Check stale links as full markdown targets, not raw substrings |

## Current follow-up: add H3.2 depth5 (2026-09-18)

### Experiment card

- Question: does adding a depth5 draft checkpoint extend the H3.2 depth trend on
  the same VDC50 protocol used for depth1/depth3?
- Baseline: existing H3.2 VDC, Qwen2.5-VL-3B target, 50 samples per corpus,
  Full/Zero/Cut, `accepted_effective_tokens`, paired sample bootstrap.
- Variant: one independently trained five-layer DFlash checkpoint per corpus;
  no target, prompt, preprocessing, or intervention change.
- Primary metric: `(Full−Zero)_depth5 − (Full−Zero)_depth1`; depth3 remains the
  registered comparison and depth5 is an extension until separately registered.
- Guardrails: 150 valid rows per corpus, target-output hash consistency, zero
  runtime errors, and the same H3.2 visual/target protocol.
- Dataset/budget: VDC50, 300 rows total for depth5; no MVBench claim from
  partial coverage.
- Success criterion: report depth5 point estimates and bootstrap CIs; do not
  claim a universal depth benefit unless both corpora agree in direction.
- Exploratory-only: reusing old 6e/20e artifacts without rerun is not valid
  evidence because their preprocessing/target contract differs.

### Current status

- Local workspace has only H3.2 depth1/depth3 checkpoints and configs.
- The H3.2 Hugging Face repository has no `depth5` files.
- A separate repository exposes old 6e/20e five-layer checkpoints, but they
  were produced under the older VDC protocol.
- The current GPU probe cannot communicate with the NVIDIA driver; free disk is
  approximately 9.3 GB, below the size needed to keep both old checkpoints
  locally while preserving workspace headroom.
- Do not modify H3.2 figures or insert placeholder depth5 values until a valid
  depth5 rerun is available.

### Resolution (2026-09-18)

- Local target model was found at `/path/to/local-resource`; no target-model download was needed.
- The 5-layer checkpoint was downloaded from the documented Hugging Face
  **dataset** repo `Tphuc15/vdflash-qwen25vl-3b-checkpoints` using the 6e
  LLaVA/ShareGPT artifacts. Temporary checkpoints were removed after each run
  to preserve disk space; depth1/depth3 checkpoints were not deleted.
- Both depth5 VDC runs completed with 150/150 `status=ok` rows and 50 samples
  per corpus. The analyzer was extended and rerun on 900/900 rows across
  depth1/3/5; figures and the delivery report now include depth5.
# Current task: reframe H2.1/H3.1/H3.2 and plan the next validation study (2026-09-21)

## Goal

Test the revised explanation that MVBench's multiple-choice answer leakage can
make visual ablation improve draft acceptance, while free-form VDC captioning
does not contain the same answer information. In parallel, resolve H3.1 with
position-wise MSD evidence and test the indirect depth explanation with DFlash
depth 1/3/5. No new inference job is launched until the VDC-MCQ manifest and
the metric/decision rules below are frozen.

## Revised hypotheses

- **H2.1 — answer leakage / task format:** MVBench exposes candidate answers in
  the text prompt; therefore Cut/Zero can increase draft acceptance by making
  the draft attend more to text. VDC's natural caption prompt has no candidate
  answer and should not show the same direction. A VDC-MCQ branch is needed to
  separate answer leakage from the dataset/video distribution.
- **H3.1 — late-only acceptance gain:** if MSD's Cut gain is caused by the
  already-generated text answer, Cut should be worse than Full in the first
  half of the answer and better in the second half. The primary unit is an
  answer-position acceptance trace, not a mean over speculative rounds.
- **H3.2 — draft depth and visual use:** if DFlash's deeper draft can use visual
  values while MSD's shallow draft cannot, the visual-value sensitivity
  `Full−Zero` should increase with DFlash depth. `Full−Cut` is secondary because
  it also changes context length and the attention denominator.

## Locked protocol

- Interventions: `Full`, `Zero` (visual positions kept, values zeroed), and
  `Cut` (visual positions removed). Keep the target's visual input unchanged.
- H2.1 primary metric: interaction in `accepted_effective_tokens` or
  `tau_effective`, with `accepted_prefix_tokens`/acceptance rate reported as
  secondary diagnostics. Task accuracy/correct-option rate is a separate
  guardrail for VDC-MCQ and MVBench.
- H3.1 primary metric: paired position-wise acceptance-rate interaction
  `Delta_late − Delta_early`, with absolute and normalized answer-position bins.
- H3.2 primary metric: paired depth interaction
  `(Full−Zero)_higher_depth − (Full−Zero)_lower_depth`; registered comparison
  remains D3−D1, with D5−D1 and D5−D3 exploratory extensions.
- Bootstrap at sample level; never treat tokens or speculative rounds as
  independent observations. Audit `status=ok`, sample pairing, target-output
  hashes, losslessness, and runtime errors before interpreting a contrast.

## Phases

- [complete] 1. Record the revised causal interpretation and separate format,
  answer leakage, visual information, and draft-depth effects.
- [in_progress] 2. Freeze the VDC-MCQ construction and manifest audit contract.
- [pending] 3. Run VDC natural versus VDC-MCQ Full/Zero/Cut with performance and
  correct-option guardrails.
- [pending] 4. Run the same position-wise protocol on MSD VDC50; use the
  existing DFlash H3.1 full run as a separate backend reference.
- [pending] 5. Complete the DFlash depth 1/3/5 comparison on MVBench and add
  position-wise/quality/cost diagnostics.
- [pending] 6. Produce one decision memo that distinguishes confirmed,
  conditional, exploratory, and unresolved claims.

## Design constraints and open decision

- VDC currently contains one generic caption question and a free-form reference
  caption; it has no native candidate options. The VDC-MCQ manifest must add a
  question, one verified correct option, and hard distractors, then use the same
  canonical `Question/Option/(A)...` prompt shape as MVBench. A generated or
  cross-sample distractor set must be provenance-tracked and manually/audited
  for one unambiguous correct option.
- Do not call the old `Answer-hint` experiment a VDC-MCQ result. It appended the
  reference caption to the prompt and changed the target continuation; it is an
  oracle leakage diagnostic only.
- If VDC-MCQ changes the sign of the Cut/Zero acceptance effect toward the
  MVBench direction, this supports answer leakage as an explanation. If it does
  not, answer leakage alone is insufficient. Neither outcome by itself proves
  answer quality.
- Existing H3.2 VDC results are corpus-dependent: LLaVA supports D3−D1, while
  ShareGPT is opposite. This is preliminary evidence against a universal
  depth-benefit claim, not a reason to pool the corpora.

## Errors / risks to log

| Risk | Control |
|---|---|
| VDC-MCQ distractors are trivial or ambiguous | hard-negative selection, answer audit, and per-option counts before inference |
| MCQ format changes output length/tokenization | keep prompt template fixed across visual conditions and report prompt/token statistics |
| Acceptance increase is mistaken for quality improvement | report correct-option accuracy and target parity separately |
| `Cut` mixes visual removal with context-length change | use `Full−Zero` as the H3.2 visual-value contrast; label `Full−Cut` secondary |
| H3.1 backend/protocol mixing | run MSD full separately; never pool MSD with DFlash |

## Execution errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| H3.2 plot failed because some groups had no `actual_visual_tokens` mean | 1 | Keep missing values in the descriptive CSV and render them as omitted `NaN` points in the figure |

See `task_plan_h21_h31_h32_reframed_20260921.md` for the experiment cards,
promotion ladder, and concrete analysis outputs.
# Current task: prepare the H2.1/H3.1/H3.2 experiment runs (2026-09-22)

## Goal

Prepare, validate, and then run the three hypothesis studies with the correct
backend and checkpoint provenance. H3.1 uses MSD, not DFlash. The first run
stage records raw acceptance/output/timing evidence; hypothesis contrasts are
deferred until the journals pass coverage and hash audits.

## Locked experiment matrix

### H2.1 — answer-hint diagnostic

- Target: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Draft: DFlash 20e LLaVA68K checkpoint from
  `Tphuc15/vdflash-qwen25vl-3b-checkpoints`.
- Dataset: VDC `dataset/VideoDetailCaption/test.jsonl`, 50 samples.
- Prompt variants: `natural`, `answer_hint`.
- Draft visual conditions: `full`, `reduced` (25%), `deleted` (0%).
- Expected matrix: 300 rows.
- This is the existing answer-hint diagnostic, not the future VDC-MCQ
  manifest. VDC-MCQ remains a separate study until its options are audited.

### H3.1 — MSD visual-retention sweep

- Target: `Qwen/Qwen2-VL-7B-Instruct`.
- Draft: `lucylyn/MSD-Qwen2VL-7B-Instruct`.
- Dataset: VDC `dataset/VideoDetailCaption/subset_manifest.jsonl`, 50 samples.
- Prompt: natural caption prompt; no answer hint.
- Target keeps the full visual input; only MSD draft visual context changes.
- Retention levels: `100`, `50`, `25`, `0` percent exactly.
- Main visual anchor: approximately 25K target visual tokens; calibration
  coverage is audited before committing to the full cohort. A 3K fallback is
  a separately labelled pilot, never pooled with the 25K cohort.
- Expected retention matrix: 200 rows (50 samples × 4 levels). If native Full
  and retention-100 are both retained for runtime-path parity, the extra
  native-control rows are reported separately.
- Primary raw diagnostic: position-wise `acceptance_by_position`; aggregate
  `accepted_prefix_tokens` is secondary. No first/second-half contrast is
  calculated in the collection stage.

### H3.2 — DFlash depth 1/3/5

- Target: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Draft corpora: `llava68k`, `sharegpt68k`.
- Depths: 1, 3, 5.
- Conditions: `full`, `zero`, `cut`; target visual input remains full.
- VDC: 50 samples/cell → 900 rows.
- MVBench: 1000 samples/cell → 18,000 rows.
- Expected total: 18,900 rows.
- Depth1/3 checkpoints come from `Tphuc15/qwen25vl-3b-depth1-depth3`.
- Depth5 uses the previously validated `6e` checkpoints from
  `Tphuc15/vdflash-qwen25vl-3b-checkpoints`; do not substitute 20e in this
  matrix without registering a separate training-step variant.

## Files and commands to prepare

- H2.1/H3.2 runner:
  `src/analyze/Validate_Sparrow_hypothesises/run_dflash_h21_h31.py`.
- H3.2 batch wrapper:
  `scripts/run_h32_depth_full_20260915.sh`; extend its depth loop to include
  depth5 and pass the downloaded 6e checkpoint paths.
- H3.1 runner:
  `src/analyze/Validate_Sparrow_hypothesises/run_msd.py`.
- H3.1 model-parallel wrapper:
  `src/analyze/Validate_Sparrow_hypothesises/run_msd_model_parallel_2gpu.sh`.
  The current wrapper expects `.venv`; use the resolved CUDA environment only
  after its environment path is fixed or invoke the module directly with the
  approved workspace environment.
- Raw/detail collector:
  `src/analyze/Validate_Sparrow_hypothesises/analyze_run_metrics.py`; extend
  its normalization for MSD fields (`acceptance_trace`, flat timing fields)
  before using it as the H3.1 analyzer.

## Information to persist

Every successful or error row must preserve:

- sample/condition/model/checkpoint/config/provenance;
- prompt variant and target/draft visual-token counts;
- calibration candidate, tolerance/status, and actual visual count;
- target/speculative output IDs, decoded texts, hashes, lossless flag and
  common-prefix length;
- aggregate acceptance length and raw acceptance trace;
- position-wise proposed/accepted counts and rates;
- per-round emitted/accepted lengths and corresponding text/token IDs;
- prefill, draft, verify, decode, end-to-end time, throughput, speedup and
  peak memory when emitted;
- runtime errors, retry state, and row status.

Derived artifacts per experiment:

- `raw_journal.jsonl`;
- `output_detail_records.jsonl`;
- `output_position_records.jsonl`;
- `acceptance_round_records.jsonl`;
- `metric_records.csv`, `metric_summary.csv`, `metric_coverage.csv`;
- position-wise CSV and PNG plots;
- `manifest.json` with source files, counts, tokenizer, checkpoint hashes and
  missing-field coverage.

## Preflight gates

1. Confirm GPU visibility, CUDA environment, disk space, and no conflicting
   process.
2. Verify all target/draft model paths or authorized HF access without placing
   tokens in scripts/logs.
3. Download only missing checkpoints; verify file size and SHA-256.
4. Verify manifests, video paths, calibration coverage, and sample counts.
5. Run one-sample smoke for each backend/condition family.
6. Verify target-output hash invariance across visual conditions where the
   target is intentionally unchanged.
7. Run full journals with resume-safe output directories.
8. Run detail extraction, coverage audit, and plots.
9. Only after coverage passes, compute H2.1/H3.1/H3.2 hypothesis contrasts.

## Decision rules for collection

- Do not pool MSD H3.1 with DFlash H3.1.
- Do not pool H3.2 depth5 6e with a 20e depth variant.
- Do not replace missing acceptance values with zero.
- Do not interpret partial MVBench coverage as a result.
- Do not compute `I_format`, early/late contrasts, or depth interactions during
  raw collection.

## Status

- [complete] Correct H3.1 backend to MSD and retention grid to 100/50/25/0.
- [complete] Identify H3.2 depth5 checkpoint source and exact filenames.
- [in_progress] Extend MSD metric normalization and run preflight.
- [pending] Obtain missing models/checkpoints and verify hashes.
- [pending] Smoke H2.1, H3.1, and H3.2.
- [pending] Run full matrices and detailed output extraction.
- [pending] Produce hypothesis-level contrasts and decision memo.

## Phase 2 status — 2026-09-23

The raw collection phase is complete and the first descriptive aggregation
phase is also complete. The outputs are under
`results/phase2_summary_20260923/` and are summarized in its `README.md`.

- [complete] Normalize and summarize H2.1, H3.1, and H3.2 runtime-native fields.
- [complete] Preserve/link output detail, per-position, per-round, and coverage
  artifacts.
- [complete] Generate overview and detailed plots.
- [pending] Parse H2.1 correct-option accuracy from the audited VDC-MCQ fields.
- [pending] Compute H3.1 normalized first-half/second-half position contrasts.
- [pending] Compute paired H3.2 Full−Zero/Full−Cut depth contrasts with uncertainty.
- [pending] Write the final hypothesis decision memo after the locked metrics
  are available.

---
