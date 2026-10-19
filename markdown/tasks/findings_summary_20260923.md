# Findings

## 2026-09-21 — Revised H2.1/H3.1/H3.2 interpretation

- The user's revised H2.1 is a format/answer-leakage hypothesis: MVBench
  exposes candidate answer text through multiple-choice options, while the
  current VDC manifest is free-form captioning. Therefore a VDC-MCQ branch is
  required to test whether the sign of the visual-ablation acceptance effect
  follows the task format rather than the video dataset alone.
- Existing VDC rows have one generic caption question plus a reference caption;
  they do not contain native candidates. A valid VDC-MCQ manifest must add one
  auditable correct option and hard distractors with provenance. The old
  Natural/Answer-hint run is not equivalent because it changes the target
  continuation and is only an oracle alignment diagnostic.
- H2.1 primary follow-up contrast is the format interaction
  `[Zero/Cut−Full]_VDC-MCQ − [Zero/Cut−Full]_VDC-natural`; acceptance remains
  separate from MCQ correct-option accuracy.
- H3.1 should be rerun on MSD with position-wise acceptance rates and a primary
  `late_delta−early_delta` interaction. The existing DFlash VDC50 result is a
  separate backend reference and did not show late-only gain.
- H3.2 remains an indirect depth test: within each corpus, compare
  `Full−Zero` across DFlash depths 1/3/5. Existing VDC results are corpus
  dependent (LLaVA D3−D1 positive; ShareGPT negative), so no universal depth
  claim is currently justified.
- Detailed plan: `task_plan_h21_h31_h32_reframed_20260921.md`.

## 2026-09-17 — Report delivery folder

- User requested a completed report package under `markdown/Baocao_17_9_26`.
- The source report remains `markdown/bao_cao_tong_hop_thuc_nghiem_hypothesis_20260910.md`.
- Need distinguish image links, which should point to copied local images, from
  CSV/JSON artifact links, which should remain linked to `results/` unless a
  report-local copy is explicitly needed.
- The delivery report currently has 18 PNG link occurrences (18 unique files),
  all sourced from `results/`; the relative links are broken from the nested
  delivery folder because they still use the original `../results/...` depth.
- The delivery package already contains the report but no figures subfolder.
- The source PNGs for all 18 references exist. Plan: copy them flat into a
  `figures/` subfolder and rewrite image links to `figures/<filename>.png`.
- The report also contains relative links to result CSV/JSON/MD artifacts and
  one runner script; these need two-level upward traversal from the delivery
  report (`../../results/...` and `../../scripts/...`).
- The first checksum command referenced a shortened filename that does not
  exist; the report's actual filename includes `_tong_hop_`.
- Four image basenames also exist in multiple result directories; uniqueness by
  basename is not a valid integrity criterion. The audit will require each
  copied image to match at least one source checksum and separately check the
  explicit source paths used for the copy.
- The final checker had a false positive because `../../results/` contains the
  raw substring `../results/`; stale-link validation must inspect target tokens
  or use a negative lookbehind.

## Initial repository state

- Repository: `/path/to/local-resource`.
- Existing worktree changes are present in DFlash audit/report/orchestration and
  Qwen2.5-VL inference files, plus new report/parallel files. These belong to
  the user and must be preserved.
- The active shell previously resolved `python` to `.venv-msd/bin/python`
  (Python 3.10.12), which is explicitly out of scope for this run.
- The initial tool-side CUDA probe reported `torch.cuda.is_available() == False`
  and NVIDIA driver communication failure. This must be revalidated on the
  requested server path before claiming a GPU smoke result.
- Host-level GPU probe succeeded and exposed `NVIDIA GeForce RTX 3090` and
  `NVIDIA RTX A4000`, with driver `535.309.01`.
- Host-level checks still found no `/path/to/local-resource` and no
  `/path/to/local-resource/**/.venv/bin/python` within the checked tree. The only Python
  environment in the repository is `.venv-msd`, which is explicitly forbidden
  for this task.

## Scope interpretation

“All baseline infer” means existing baseline/runtime inference launchers under
`src/infer`, `src/analyze/Validate_Sparrow_hypothesises`, the Qwen35 analysis
launcher, and the external Sparrow video baseline launcher where its required
model/data are available. Long-running training launchers are excluded.

## 2026-09-18 — H3.2 depth5 follow-up

- H3.2 VDC currently contains exactly two draft depths: 1 and 3; local
  checkpoint/config inventory found no depth5 artifact.
- `src/train_Dflash_SpecForge/configs/qwen2.5-vl-3b-dflash.json` is a generic
  five-layer DFlash config, but a config alone is not a trained depth5
  checkpoint and must not be used to fabricate H3.2 values.
- The H3.2-specific Hugging Face dataset repository
  `Tphuc15/qwen25vl-3b-depth1-depth3` has no files matching `depth5`.
- The separate repository documented in
  `markdown/huong_dan_su_dung_dflash_checkpoints_huggingface.md` contains old
  five-layer `6e` and `20e` checkpoints for both corpora. They are about
  5.69 GB each and were generated by the older validation path.
- Existing old VDC Full/Zero/Cut artifacts cannot be merged into H3.2: their
  preprocessing/target-output contract differs from current H3.2 rows. For
  sample `v_AwgGYaV1lT0`, current H3.2 has a target hash beginning `b9861dc7`,
  while the old run has `8d534b2a` or `c197de82`.
- Metadata probing confirmed the old repository has four expected 6e/20e
  checkpoint paths; no depth5 artifact was downloaded.

## 2026-09-18 — H3.2 depth5 completed

- The target model is available locally at
  `/path/to/local-resource`.
- The correct checkpoint source is the Hugging Face **dataset** repo
  `Tphuc15/vdflash-qwen25vl-3b-checkpoints`; using `repo_type=model` gives a
  misleading 404. The 6e LLaVA and ShareGPT five-layer artifacts were run
  separately so only one 5.69GB temporary checkpoint occupied disk at a time.
- LLaVA68K depth5 and ShareGPT68K depth5 each completed with 150/150 rows
  `ok`, 50 samples × Full/Zero/Cut, and 50 target hashes.
- Combined VDC analyzer input is 900/900 valid rows: 2 corpora × 3 depths ×
  50 samples × 3 conditions; target hashes are consistent within every sample.
- Depth5 `Full−Zero`: LLaVA `+0.1075` CI `[+0.0448,+0.1690]`; ShareGPT
  `−0.0540` CI `[−0.0826,−0.0261]`.
- Relative depth interactions: LLaVA `D3−D1=+0.1261`
  `[+0.0672,+0.1888]`, `D5−D1=+0.0799` `[+0.0105,+0.1532]`; ShareGPT
  `D3−D1=−0.0540` `[−0.0912,−0.0155]`, `D5−D1=−0.0367`
  `[−0.0707,−0.0039]`. These remain corpus-dependent, not universal depth
  evidence.
- Added a regression test and generalized `analyze_h32_depth.py` to plot all
  observed depths while preserving D3−D1 as the registered primary contrast.

## 2026-09-19 — H3.2 pairwise D5−D3 extension

- Extended the analyzer and H3.2c figure to include the direct pairwise
  contrast `D5−D3` in visual-value sensitivity.
- LLaVA68K: `D5−D3 = −0.0461`, bootstrap CI
  `[−0.1129,+0.0206]`; ShareGPT68K: `+0.0173`, CI
  `[−0.0126,+0.0471]`. Both CIs contain zero, so the current data do not
  establish a significant difference between depth5 and depth3.
- The figure now displays `D3−D1`, `D5−D1`, and `D5−D3`; D3−D1 remains the
  registered primary comparison.

## 2026-09-22 — raw metric collection pilot

- The existing journals contain enough data to plot acceptance prefix and
  effective tokens, tau proposal/effective, aggregate position-wise
  acceptance, speculative/target prefill/decode/verify/end-to-end timing,
  throughput, speedup, output lengths, losslessness, output-hash match,
  visual-token counts, and (where emitted) H2.1 attention summaries.
- The pilot intentionally reports raw condition summaries only. It does not
  decide H2.1 `I_format`, H3.1 first/second-half behavior, or H3.2 depth
  interactions.
- H3.2 has 761 missing `accepted_prefix_tokens` values. This is recorded in
  coverage metadata and not interpreted as zero; the other headline runtime
  fields used in the pilot are complete.
- The original journals contain token IDs but not reliable decoded text in
  every `acceptance_rounds` entry. The new detail pass decodes from the local
  Qwen tokenizer and stores both IDs and text, while retaining any original
  recorded text separately for audit.

## 2026-09-22 — experiment preparation plan

- H3.1 is now explicitly an MSD study: Qwen2-VL-7B target with the official
  MSD-Qwen2VL-7B draft, VDC50, natural prompt, and retention levels 100/50/25/0.
- H3.2 full three-depth matrix requires 18,900 rows: 900 VDC and 18,000
  MVBench. Depth5 must use the previously validated 6e LLaVA/ShareGPT
  checkpoints; depth1/3 are the local H3.2 checkpoints.
- The DFlash metric collector cannot be applied blindly to MSD because MSD
  writes flat timing fields and `acceptance_trace`; an MSD normalization step
  is required before H3.1 detail plots.

## 2026-09-22 — H2.1 format correction

- H2.1 is corrected to a VDC-MCQ study: VDC must be converted to the same
  multiple-choice prompt shape as MVBench, with four options and one audited
  correct option.
- The old `answer_hint` experiment is not a VDC-MCQ experiment because it
  appends the reference caption/oracle text to the prompt; it is excluded from
  the corrected H2.1 claim.
- The VDC-MCQ manifest must preserve source-caption and distractor provenance
  as audit metadata while keeping all reference text out of the model prompt.
- The VDC video preprocessing/calibration remains VDC-specific; only the task
  prompt format changes to MVBench-compatible `Question/Option/(A)-(D)`.

## 2026-09-22 — hypothesis-plan consistency audit

- The legacy aggregate values map to draft corpora: MVBench LLaVA
  `2.248→2.276`, ShareGPT `2.495→2.568`; VDC LLaVA `2.474→2.405`, ShareGPT
  `1.650→1.658` for the cited EXP1-zero rows. Therefore VDC is not uniformly
  lower under Zero: ShareGPT has a small `+0.008` change. These means are
  descriptive only and do not establish a paired hypothesis result.
- H2.1 must compare Natural VDC and VDC-MCQ on the same samples and report
  `Full/Zero/Cut` separately. Raw acceptance length across free-form captioning
  and MCQ is not a quality comparison because output lengths and tokenization
  differ; use within-format contrasts and correct-option guardrails.
- H3.1 is an MSD retention study, not a DFlash Zero/Cut study. The locked
  primary levels are exactly `100, 50, 25, 0%`, with target full visual input
  and draft retention intervention. The primary position contrast is
  `retention-0 − retention-100` by normalized answer position.
- H3.2 tests only the within-DFlash depth mechanism. DFlash depth `1/3/5`
  cannot by itself causally prove that MSD gains from cutting visual tokens
  because MSD has one layer; model family, training, target and runtime also
  differ. The registered primary contrast remains `Full−Zero` by depth, with
  `Full−Cut` secondary.

## 2026-09-22 — Phase-1 collection boundary

- The current run is raw-data construction/collection only. It may record
  runtime-native values and complete token/text traces, but it must not compute
  mean acc length/rate summaries, `I_format`, early/late statistics, depth
  interactions, confidence intervals, plots, or hypothesis decisions.
- Phase 1 must preserve enough information to reconstruct later analyses:
  prompt/manifest metadata, visual masks and positions, target/speculative
  token IDs and decoded text, per-round acceptance, absolute output positions,
  timings, output hashes, parity/losslessness, option answers, checkpoint and
  runtime metadata.

## 2026-09-22 — Phase-1 execution status

- H2.1 now has two complete raw journals under
  `results/phase1_raw_20260922/h21/`: LLaVA68K depth-3 and ShareGPT68K
  depth-3, each with 300/300 `ok` rows and balanced natural/MCQ × Full/Zero/Cut
  coverage.
- A failed first H2.1 attempt stopped at calibration out-of-tolerance; it was
  not used as data. The complete reruns explicitly used
  `--allow-out-of-tolerance`, and the first partial journal was retained only
  as a diagnostic artifact.
- H3.2 raw trace enrichment is lossless with respect to the original journal:
  it derives text only from recorded token IDs and writes a separate output
  tree. It does not calculate acceptance summaries or alter the source data.
- H3.1 remains an environment-blocked run, not a negative result. The locked
  retention levels remain 100/50/25/0 and must be run after model storage is
  available. MVBench depth-5 H3.2 likewise remains pending a compatible
  trained checkpoint.
## 2026-09-22 — H3.2 depth-5 collection status

- The missing MVBench depth-5 cells are now collected with the validated 6e
  DFlash checkpoints: LLaVA68K and ShareGPT68K, each with 1,000 samples ×
  `full/zero/cut` = 3,000 raw rows.
- Both completed cells have complete prompt/output and acceptance traces:
  prompt IDs/text, target and speculative IDs/text, per-position token text,
  per-round proposal/target text, visual positions, timing, and runtime fields.
- The ShareGPT cell was resumed from its partial journal after an interrupted
  process; final coverage is 3,000 rows with all statuses `ok`.
- This is a collection result only. No acc-length/rate aggregation, early/late
  analysis, depth interaction, plot, or hypothesis decision was performed.

## 2026-09-23 — H3.1 runtime placement

- A full-length MSD run at the 25K visual target is feasible with the current
  3090 + A4000 host when the visual encoder is placed on GPU1 and the decoder
  layers plus EAGLE draft are kept on GPU0. The balanced decoder split caused
  cross-device KV/BMM mismatches, so the validated collection layout is the
  asymmetric placement above.
- The run records all four locked retention levels for each VDC50 sample and
  is intentionally left as raw evidence; no aggregated acceptance or
  first/second-half conclusion is allowed in Phase 1.

## 2026-09-23 — H3.1 completion and two-GPU fix

- The H3.1 journal is complete at 200 measured rows. A single retained error
  marker documents the initial OOM and is excluded from the measured-row
  count; the sample was later successfully rerun.
- Splitting decoder layers exposed a second MSD-specific issue: EAGLE's
  `tree_mask_init` and `position_ids` are plain tensor attributes and are not
  moved by `nn.Module.to()`. Rebuilding tree state after placing EAGLE on the
  embedding device fixed the CUDA BMM mismatch and allowed the final 25K
  sample to run with `max_new_tokens=512`.
- This is still Phase-1 raw evidence only. The observed losslessness/error
  fields remain per-row data and have not been aggregated into hypothesis
  metrics.

## 2026-09-23 — Phase 2 descriptive summary

- H2.1 MCQ accepted prefix is `0.956/1.112/1.491` for full/zero/cut, while
  natural VDC is `0.870/0.823/0.846`; MCQ acceptance rate is about `0.72–0.76`
  versus `0.062–0.066` for natural. These are within-format runtime
  observations, not answer accuracy.
- H3.1 MSD accepted prefix is `1.219/1.363/1.471/3.669` at retention
  `100/50/25/0`; output length remains about `380` tokens. The MSD acceptance
  rate field is a normalization artifact because fallback positions have
  `proposed=0`.
- H3.2 MVBench benefits from visual removal in several ShareGPT depth-3/5
  cells but not uniformly across LLaVA. VDC shows opposite trends for LLaVA
  versus ShareGPT, so the depth effect is corpus/dataset dependent.
- `actual_visual_tokens` is invariant across intervention conditions because it
  describes visual positions in the model layout. `draft_context_visual_tokens`
  is the appropriate condition-sensitive field: full/zero retain positions,
  cut has zero draft visual tokens.
- The consolidated descriptive report is
  `results/phase2_summary_20260923/README.md`; it records the missing metrics
  and why no H2.1/H3.1/H3.2 hypothesis decision is claimed yet.

## 2026-09-23 — Acceptance-length visualization

- The new line plots show both quantities explicitly: dashed markers are
  `accepted_prefix = k`; solid markers are `acc length = k+1`.
- H3.1 is plotted in the requested retention order `100/50/25/0`; H3.2 uses
  depth `1/3/5` with separate Full/Zero/Cut lines for each corpus.
- The visualization confirms that the earlier table's “Accepted prefix” was
  not the effective acc length; the effective curve is exactly one token above
  it whenever both are computed on the same round set.

## 2026-09-23 — H2.1 acceptance-rate bug and correction

- The original H2.1 `acceptance_rate` was computed from `acceptance_by_position`.
  This is a position diagnostic, not a global round denominator: MCQ had 92/100
  outputs of length 3 while DFlash still proposed blocks of about 15 tokens,
  so most terminal-round proposals were omitted from the denominator.
- The old rates (`MCQ ~0.72–0.76`, Natural `~0.062–0.066`) were therefore not
  comparable. Recomputing from complete round records gives MCQ
  `0.149/0.149/0.154` and Natural `0.059/0.056/0.057` for Full/Zero/Cut.
- The analyzer now prefers round-level `matched_proposals/proposal_count` for
  DFlash. Position-level rates remain available in position artifacts. MSD
  still uses its legacy fallback until its runner records a complete proposal
  denominator.
