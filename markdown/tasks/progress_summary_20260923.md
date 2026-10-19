# Progress Log

## 2026-09-21 — Reframed hypothesis plan

- Recorded the revised interpretation that MVBench answer leakage may explain
  why visual ablation improves acceptance, while free-form VDC lacks candidate
  answer text.
- Created `task_plan_h21_h31_h32_reframed_20260921.md` with experiment cards,
  metrics, decision rules, promotion gates, and the VDC-MCQ manifest audit.
- No inference or code implementation was started. The remaining design gate is
  the construction and audit protocol for one correct VDC-MCQ option plus hard
  distractors.

## 2026-09-17 — Report delivery folder

- Started inventory of the source report, image references, and available
  result figures before making any writes.
- Inventory complete: 18 unique PNG references are present and all source files
  exist under `results/`; the nested report's current `../results/...` links do
  not resolve.
- Created `markdown/Baocao_17_9_26/figures` and copied all 18 referenced PNGs;
  source artifacts under `results/` were left unchanged.
- Initial link audit passed with `0` missing links; checksum audit was corrected
  after using the wrong report filename in its first attempt.
- The first corrected checksum audit found four duplicate basenames in results;
  this is a validation-script assumption issue, not a copied-file mismatch.
- Rewrote all delivery-report image links to `figures/...`, repaired result and
  script links to the correct nested paths, and repaired the two sibling-report
  links to `../...`.
- Explicit source-to-copy checksum verification passed for all 18 images.
- The first combined final-check attempt was a checker false positive on the
  nested `../../results/` prefix; the link audit itself reported 0 missing links.
- Final verification passed: 102 markdown links scanned, 101 relative links,
  0 missing/stale links, and 18 valid local PNGs.

## 2026-08-27

- Received explicit approval to implement the agreed smoke/debug design.
- Created persistent plan, findings, and progress files.
- Host GPU visibility was confirmed (RTX 3090 + RTX A4000), but the required
  `./.venv` is absent; `.venv-msd` remains unused.
- Next: run static launcher/preflight checks and locate any hard-coded or
  implicit environment routing while waiting for the required `.venv` path.
- Static preflight found hard-coded `.venv-msd` references and implicit `python`
  calls in baseline launchers. Runtime debugging is blocked until the actual
  `tuantb@teslaT4:~/fast_infer_text_sum` workspace or its `.venv` is exposed to
  this agent.
- Added `scripts/resolve_workspace.sh` and `src/workspace.py`. Launchers now
  derive the checkout from their own file location (or `VD_FLASH_WORKSPACE`)
  and hard-stop unless `<workspace>/.venv/bin/python` exists.
- Routed MSD, DFlash, Sparrow, Qwen3.5, MVBench, and external Sparrow VDC
  baseline launchers through the shared resolver. Removed host-specific paths
  from the runnable VDC baseline adapter and made its local VDC JSONL/video
  loading portable.
- Static verification passes: workspace-routing shell contract, `bash -n` for
  all scoped launchers, `git diff --check`, and resolver override fixture.
- Added root-relative handling to the direct MSD/attention/layer/figure3
  runners, the SpecForge HF smoke entry point, ViSpec baseline, and SpecVLM
  wrappers. `LIMIT=1` is now supported by the MVBench batch wrapper.
- Additional verification passes: resolver auto-location fixture, portable
  launcher fixture from outside the repository, all scoped shell scripts via
  `bash -n`, routing contract, and no remaining host-path hits in scoped
  runnable launchers.
- Runtime pytest/GPU smoke remains pending because this checkout still lacks
  `.venv/bin/python`; `.venv-msd` has not been invoked.

## 2026-09-15 — H3.1/H3.2 report completion

- Audited H3.1 DFlash full VDC50: 150/150 valid rows, 50 samples, no errors.
- Re-ran H3.1 analysis and added paired per-position delta statistics plus
  `h31_dflash_position_delta.png/.pdf`.
- Audited H3.2 VDC: 600/600 valid rows across two corpora, two depths and three
  visual conditions; target hash audit is consistent.
- Re-ran H3.2 VDC analysis and added `h32_visual_sensitivity.png` alongside the
  existing acceptance-by-depth and primary-interaction figures.
- Updated the main hypothesis report and the H3.2 progress report. The main
  report now separates confirmed VDC evidence from incomplete MVBench progress;
  MSD H3.1 `n=2` figures remain excluded from the main evidence.
- Verification: markdown link audit 104/104 passed; focused pytest 11/11 passed;
  py_compile passed; `git diff --check` passed.
- MVBench H3.2 remains an active GPU job. At the latest audit it had roughly
  2.5K valid rows in LLaVA68K/depth1 and the other three cells had not started;
  no MVBench conclusion is written.

## 2026-09-18 — H3.2 depth5 follow-up

- Read the H3.2 runner/analyzer and locked the extension as a VDC50 rerun with
  the existing Full/Zero/Cut intervention and `accepted_effective_tokens`.
- Checked local checkpoints, configs, result journals, available disk, and GPU
  visibility. Only depth1/depth3 H3.2 checkpoints are local; GPU driver probe
  failed; free disk is about 9.3 GB.
- Checked the H3.2 Hugging Face repository for `*depth5*`; no matching files.
- Checked the separate checkpoint repository metadata with approved network
  access; it contains old 6e/20e five-layer checkpoints, about 5.69 GB each.
- Compared old VDC artifacts with current H3.2 rows and found differing target
  hashes/protocol, so no old values were inserted into the figures.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| `huggingface.co` DNS failure in sandbox | 1 | Re-ran metadata query with approved external connection |
| H3.2 repo has no depth5 artifact | 1 | Checked separate checkpoint repository for compatible five-layer artifacts |
| NVIDIA driver unavailable in current probe | 1 | Did not launch GPU experiment; preserved existing figures |
| First checkpoint download used repo-local HF cache and returned 401 | 1 | Retried with the machine credential cache |
| Machine credential is not authorized for gated 5-layer repository (403) | 1 | Did not bypass access control; waiting for an authorized local checkpoint or HF access |
| Retried 5-layer download as authenticated user `VietNguyen865`; repository still returns 403 | 2 | No checkpoint was downloaded and no H3.2 depth5 run was started |

## 2026-09-18 — H3.2 depth5 follow-up resolved

- Found the local target model and reran the depth5 smoke tests on the two
  CUDA GPUs; both LLaVA and ShareGPT smoke tests wrote 3/3 valid conditions.
- Ran full VDC50 depth5 for both corpora. Each output has 150/150 rows,
  `status=ok` for every row, complete Full/Zero/Cut coverage, and consistent
  target hashes.
- Corrected the Hugging Face source type to `--repo-type dataset` and used the
  documented 6e depth5 artifacts. Temporary LLaVA/ShareGPT checkpoints were
  deleted after their respective runs; existing depth1/depth3 checkpoints were
  preserved.
- Generalized the H3.2 analyzer with a failing-then-passing regression test;
  reran it on all six VDC journals. Final analysis: 900/900 rows valid, with
  depth1/3/5 plotted in all three figures.
- Updated `markdown/Baocao_17_9_26/figures/` and the H3.2 section of the
  delivery report with the new depth5 estimates and the D5−D1 exploratory
  interaction. MVBench remains separate and incomplete.

## 2026-09-19 — H3.2 pairwise D5−D3 extension

- Added a failing regression assertion for direct `D5−D3`, then generalized
  the analyzer to emit all pairwise depth interactions.
- Recomputed the direct paired bootstrap contrasts from the same 50 samples:
  LLaVA `−0.0461` `[−0.1129,+0.0206]`; ShareGPT `+0.0173`
  `[−0.0126,+0.0471]`.
- Updated H3.2c in the report package to show `D3−D1`, `D5−D1`, and `D5−D3`.
  Both D5−D3 CIs cross zero, so the report does not claim a significant
  depth5-versus-depth3 difference.

## 2026-09-22 — H2.1/H3.1/H3.2 raw-metric pilot

- Added `analyze_run_metrics.py`, a descriptive-only collector for fields
  already emitted by the runtime: acceptance/tau, position-wise acceptance,
  speculative and target timings/speed, output lengths/losslessness/hashes,
  visual-token counts, and existing H2.1 attention summaries.
- Added `metric_coverage.csv` and manifest coverage fields. Missing runtime
  values remain missing; no `I_format`, early/late contrast, or depth
  interaction is computed here.
- H2.1: 300/300 valid rows; H3.1: 150/150; H3.2: 12,900/12,900 from six VDC
  and four MVBench journals. Position traces are present after preserving
  `acceptance_by_position` during flattening.
- H2.1/H3.1 key acceptance, timing, and losslessness fields have 100%
  coverage. H3.2 `accepted_prefix_tokens` coverage is 94.1% (12,139/12,900);
  acceptance rate, timing, and losslessness are complete.
- Reviewed representative acceptance, timing, and position plots; all contain
  plotted condition/group data. Focused regression and existing H2/H3 analyzer
  tests pass: 14/14.
- Repository-wide pytest was attempted with `.venv-msd`; collection is blocked
  by two pre-existing imports of the absent module
  `src.analyze.Whether_they_are_appliable_for_dDrafter`, while the scoped
  experiment tests pass.
- Added decoded output artifacts using the local Qwen tokenizer: per-sample
  output IDs/text, per-position acceptance plus target/speculative token text,
  and per-round accepted/emitted lengths with block/proposal text. Reran all
  three pilots with `text_decoded=true`; scoped tests now pass 15/15.

## 2026-09-22 — experiment preparation plan

- Wrote the locked preparation plan for the corrected studies. H3.1 uses MSD
  with retention 100/50/25/0; H3.2 expands to depth1/3/5 for both corpora and
  both VDC/MVBench; H2.1 was initially described with the prior answer-hint
  diagnostic.
- No new GPU experiment was launched in this planning phase.

## 2026-09-22 — H2.1 correction to VDC-MCQ

- Corrected the active plan: H2.1 must create a derived VDC-MCQ manifest and
  render the exact MVBench-style `Question/Option/(A)-(D)` prompt.
- The reference caption is retained only as provenance/audit metadata and is
  never inserted into the prompt. The old `answer_hint` branch is not used as
  the H2.1 VDC-MCQ condition.
- No GPU experiment has been launched yet; manifest construction, schema tests,
  prompt parity, and answer parsing are required first.

## 2026-09-22 — Phase 1 raw collection executed

- Built `dataset/VideoDetailCaption/vdc_mcq_manifest.jsonl` with 50 VDC
  samples, four auditable options per sample, and source/distractor provenance.
- H2.1 completed with the Qwen2.5-VL-3B target and local DFlash depth-3
  checkpoints for both LLaVA68K and ShareGPT68K. Each corpus has 300 rows:
  50 samples × natural/MCQ prompt × Full/Zero/Cut; every row is `status=ok`.
- H2.1 rows preserve prompt IDs/text, target/speculative output IDs/text,
  per-position acceptance IDs/text, per-round proposal/target IDs/text,
  visual positions, attention summary, timing, hashes, and checkpoint metadata.
- Existing H3.2 journals were copied to
  `results/phase1_raw_20260922/h32_enriched/` with decoded output and
  position/round token text. The original journals were not modified.
- H3.1 was not launched because the Qwen2-VL-7B and MSD model files are not
  local and the current filesystem cannot hold the required weights without
  deleting user artifacts. MVBench depth-5 H3.2 was not launched because the
  checked checkpoint repository has no depth-5 file; existing VDC depth-5 raw
  journals remain available and were enriched.
- Raw-only audit is saved at
  `results/phase1_raw_20260922/phase1_raw_collection_audit.json`. No higher
  metrics, hypothesis contrasts, or plots were generated in this phase.

## 2026-09-22 — plan consistency audit

- Reviewed the user's H2.1/H3.1/H3.2 formulation against the runners and
  reports. Corrected the active plan and legacy experiment plan so H2.1 uses
  VDC-MCQ, H3.1 uses MSD retention `100/50/25/0`, and H3.2 covers the full
  depth-1/3/5 matrix without claiming a direct MSD-vs-DFlash causal test.
- Identified that the current DFlash runner still exposes the old E7
  `natural/answer_hint` path and its H3.1/H3.2 condition helper uses
  `Full/Zero/Cut`; the VDC-MCQ prompt path and the MSD retention run must be
  handled explicitly before GPU execution.
- No new inference job was launched during this audit.

## 2026-09-22 — Phase-1 boundary correction

- Reframed the active plans so the next run only builds data and collects raw
  records. Higher metrics, aggregation, plots, bootstrap comparisons and
  hypothesis decisions are explicitly deferred to Phase 2.
- Added raw collection requirements for H2.1 VDC natural/MCQ, H3.1 MSD
  retention `100/50/25/0`, and H3.2 DFlash depth/corpus/dataset/condition
  cells.
- No new inference job was launched after this correction.
## 2026-09-22 — H3.2 depth-5 raw collection completed

- MVBench × LLaVA68K × depth-5 completed with 3,000/3,000 `ok` rows across
  1,000 samples and `full/zero/cut`.
- MVBench × ShareGPT68K × depth-5 completed with 3,000/3,000 `ok` rows across
  1,000 samples and `full/zero/cut`; the interrupted run was resumed without
  rerunning completed rows.
- Both journals preserve prompt IDs/text, target/speculative output IDs/text,
  per-position and per-round acceptance traces, visual positions, timing, and
  runtime metadata. No higher metrics or plots were computed.
- H3.1 remains not run because the required Qwen2-VL-7B and MSD weights cannot
  fit in the currently available storage without deleting user artifacts.

## 2026-09-23 — H3.1 MSD full-length raw collection active

- The local DFlash training-state checkpoints were removed only after their
  exact Hugging Face paths were verified; H2.1/H3.2 raw journals remain intact.
- H3.1 is running on the official Qwen2-VL-7B/MSD pair at the 25K visual
  target with retention `100/50/25/0`, VDC50, natural prompt, and
  `max_new_tokens=512`.
- The run uses the verified two-GPU placement (vision on GPU1; decoder/EAGLE
  on GPU0), persists one JSONL record per sample/retention job, and keeps full
  output IDs/text, acceptance rounds/positions, timings, visual positions, and
  hashes. At the latest checkpoint, 2 samples × 4 retention levels were
  written successfully. No Phase-2 metrics or plots have been computed.

## 2026-09-23 — H3.1 MSD full-length raw collection completed

- H3.1 now has 200 measured rows: 50 VDC samples × retention
  `100/50/25/0`, at the 25K visual target, with complete prompt/output,
  position/round trace and timing fields.
- One error-marker row from an earlier OOM attempt is retained for provenance;
  the corresponding sample was subsequently completed at all four retention
  levels. The audit therefore reports 201 JSONL lines but 200 measured rows.
- The final high-memory sample was completed with decoder layers split across
  both GPUs after repairing MSD EAGLE tree-state placement; the remaining
  samples used the earlier stable map. No higher metrics, plots, or hypothesis
  decisions were computed.

## 2026-09-23 — Phase 2 descriptive aggregation and visualization

- Aggregated the completed raw journals with
  `src/analyze/Validate_Sparrow_hypothesises/analyze_run_metrics.py`:
  H2.1 `600` rows, H3.1 `200` measured rows, and H3.2 `18,900` rows.
- Generated per-experiment CSV/JSONL summaries and the acceptance, timing,
  output, visual-token, position, and H2 attention plots.
- Added `scripts/plot_phase2_overview.py` and generated compact overview plots;
  H3.1 is ordered `100/50/25/0`, and H2/H3.2 use
  `draft_context_visual_tokens` to distinguish `full/zero` from `cut`.
- Wrote the consolidated report at
  `results/phase2_summary_20260923/README.md`.
- Descriptive findings show an MCQ-vs-natural difference in H2.1, a strong
  MSD accepted-prefix increase at 0% retention in H3.1, and corpus/dataset
  dependent depth effects in H3.2. No hypothesis decision was claimed because
  H2.1 accuracy, H3.1 early/late contrast, and H3.2 paired depth contrasts
  remain separate analysis steps.
- Verification passed: focused experiment tests `13 passed`, Python compile,
  `git diff --check`, and non-empty overview PNG checks.

## 2026-09-23 — Acceptance-length line visualizations

- Added `scripts/plot_phase2_overview.py` support for explicit
  `accepted_prefix_tokens = k` and effective `acc length = k+1` series.
- Generated polished line plots for H2.1, H3.1 retention, and H3.2 MVBench/VDC
  depth/corpus panels under `results/phase2_summary_20260923/`.
- Updated the consolidated README with links to the new line plots.
- Added two regression tests for effective-length derivation and PNG output;
  the focused suite now passes `15 tests`.

## 2026-09-23 — H2.1 acceptance-rate correction

- Diagnosed the large MCQ/Natural acceptance-rate gap as denominator
  truncation: `acceptance_by_position` drops proposals after a short MCQ
  output, although each DFlash round proposed the full block.
- Added a round-based `acceptance_rate` using
  `sum(matched_proposals) / sum(proposal_count)` when complete DFlash round
  telemetry is available; retained the position fallback for MSD because its
  current trace lacks `proposal_count`.
- Recomputed H2.1 and regenerated H2.1 plots/report values. Corrected rates are
  MCQ `0.149–0.154` and Natural `0.056–0.059`.
- Recomputed H3.2 and H3.1 summaries/plots so generated artifacts match the
  current analyzer semantics; raw journals were not modified.
- Verification passed: focused experiment suite `26 passed`, corrected H2.1
  CSV values checked, old rate values absent from README, updated PNGs nonempty,
  and `git diff --check` passed.
