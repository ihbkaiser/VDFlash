# Findings: DFlash E7 and H3.1 follow-up

## Existing evidence

- Existing E7 is MSD-only: Qwen2-VL-7B target plus MSD-Qwen2VL-7B draft.
- Existing H3.1 position artifact combines legacy DFlash data with an MSD
  position pilot of `n=2`; it is exploratory and has no confidence intervals.
- Existing DFlash validation path is implemented for Qwen2.5-VL-3B and uses
  `InstrumentedDFlashDecoder` with target-side full visual context.

## Local DFlash candidates

- `dataset/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt`
- `dataset/qwen25vl-3b-dflash-20e-sharegpt68k-latest/training_state.pt`
- `dataset/qwen25vl-3b-dflash-6e-llava68k-latest/training_state.pt`
- `dataset/qwen25vl-3b-dflash-6e-sharegpt68k-latest/training_state.pt`

The existing DFlash launcher defaults to the historical unsuffixed
`qwen25vl-3b-dflash-llava68k-latest` path, which is absent in this checkout.
The intended checkpoint must therefore be selected explicitly; the 20e
LLaVA-68K checkpoint is the current primary candidate because it is the
checkpoint used by the recent DFlash report path.

The repository-local `.hf_cache` contains Qwen2-VL-7B and MSD, but not a
complete Qwen2.5-VL-3B snapshot. The default user HF cache also did not expose
the Qwen2.5-VL-3B files during the read-only check. A model preflight must
therefore either resolve an existing external cache or download the target
model before the DFlash pilot.

## Required implementation distinction

The generic DFlash orchestrator currently supports length, retention,
attention and layer stages, but not the H2.1 Natural/Answer-hint factorial.
`prepare_video_prompt` currently reads `record["question"]` and therefore
needs a controlled prompt-variant hook. DFlash attention capture currently
summarizes context versus noise, but does not split the target context into
question/instruction/answer-hint regions. E7 needs that region split and raw
per-layer/per-head mass or density logs.

## Interpretation constraints

- DFlash `tau_effective` includes the target bonus token; raw matched proposal
  counts and per-round traces must also be retained.
- Answer-hint changes the target prompt and therefore changes the target
  continuation. E7 is an oracle alignment diagnostic, not a clean answer
  quality comparison.
- Attention mass must be normalized both as total mass and density/mass per
  eligible key so that a larger number of text tokens is not mistaken for
  stronger concentration.

## New DFlash pilot findings

- The selected local model/checkpoint pair is
  `Qwen/Qwen2.5-VL-3B-Instruct` with
  `dataset/qwen25vl-3b-dflash-20e-llava68k-latest/training_state.pt`.
- The target model remains on the full visual prompt in every intervention;
  reduced/deleted/cut changes only the DFlash-side target-hidden context, and
  zero changes only visual hidden values while retaining visual positions.
- E7 attention capture works on all five DFlash layers. On the first pilot
  sample, the answer-hint region had 125 tokens and its density ratio versus
  context-uniform was about 7.28/2.82/0.64 for full/reduced/deleted context;
  these are pilot diagnostics only, not cohort conclusions.
- A first implementation attempt exposed and fixed two measurement bugs:
  cut-context hidden states were not compacted after building the keep mask,
  and acceptance rounds did not contain absolute proposal positions. The
  corrected decoder now records both proposal and accepted positions.
- The H3.1 position convention is explicit: generated answer position 0 is
  the target anchor before the first speculative block, so its proposal rate
  is undefined; speculative proposal rates begin at position 1.
- GPU1 pilot results are complete and error-free for E7 (6 rows) and H3.1 (3
  rows). Full H3.1 is now running on GPU1; GPU0 is occupied by an unrelated
  user process and has not been interrupted.

## H3.1-DFlash full result

- The full DFlash journal contains 150/150 rows (`50 samples ×
  full/zero/cut`), 150 unique row IDs and zero runtime-error rows. The
  experiment uses the local Qwen2.5-VL-3B target and 20e LLaVA-68K DFlash
  checkpoint, with `max_new_tokens=64`.
- Mean accepted proposal tokens are Full `0.9395`, Zero `0.8249` and Cut
  `0.8657`. Position-wise curves show the same broad ordering rather than a
  stable late-only Cut gain.
- Paired early-vs-late interaction is `+0.0045` for Zero versus Full (95% CI
  `[-0.0063, +0.0160]`) and `+0.0053` for Cut versus Full (95% CI
  `[-0.0058, +0.0165]`). Because both intervals include zero, this run does
  not support the simple H3.1 claim that the visual-cut gain appears only
  after half of the answer.
- These conclusions are for the new DFlash backend. The historical MSD
  position pilot remains a separate `n=2` artifact until a same-backend MSD
  full run is completed.

## E7-DFlash corrected full result

- The corrected E7 journal has 300/300 valid rows: VDC50 × Natural/Answer-hint
  × Full/Reduced/Deleted. The selected pair is Qwen2.5-VL-3B-Instruct with
  the local 20e LLaVA-68K DFlash checkpoint; the target retains full visual
  context and the intervention is draft-side.
- Nine rows from three long-answer samples were rerun after the audit found
  empty answer-hint spans caused by BPE boundary matching. The final journal
  has valid answer-only spans for all 150 Answer-hint rows; the original
  journal remains unchanged and the correction is recorded in
  `e7_full_corrected/merge_manifest.json`.
- Mean `accepted_prefix_tokens` is Natural Full/Reduced/Deleted
  `0.9382/0.9548/0.8658` and Answer-hint
  `1.0978/1.1517/1.1211`. Deleted − Full is `-0.0724` for Natural and
  `+0.0233` for Answer-hint; the paired interaction is `+0.0957`, CI
  `[+0.0302,+0.1542]`. This is alignment evidence only, not ground-truth
  answer quality.
- Attention raw mass and density diverge. In Answer-hint Full, hint mass is
  `0.2187` versus question mass `0.0504`, but density-ratio versus
  context-uniform is `7.531` for hint versus `10.011` for question; the
  hint/question density ratio is `0.761`. At Deleted it is `0.957` with CI
  `[0.880,1.037]`. This supports the token-count confound interpretation and
  does not establish semantic prioritization of the hint.
- Corrected figures are `analysis/e7_dflash_acceptance.png` and
  `analysis/e7_dflash_attention_mass_density.png`; the full statistics are in
  `analysis/summary.json` and the accompanying CSV files.

## Runtime checks

- Host GPU probe: RTX 3090 24 GiB and RTX A4000 16 GiB are visible when the
  command is run with host GPU access.
- `.venv-msd` has CUDA, Transformers 4.52.4 and decord 0.6.0; `.venv` has
  CUDA/Transformers but lacks decord. The previous DFlash run path therefore
  uses `.venv-msd`.
- Existing DFlash unit tests currently expose two unrelated historical issues:
  a missing legacy caption-scoring module and a default checkpoint path that is
  no longer present. The first is needed to import the decoder and will get a
  minimal fallback test/fix; the second is handled by explicit checkpoint CLI
  arguments for this study.
