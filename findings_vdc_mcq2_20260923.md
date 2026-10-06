# Findings — VDC-MCQ2 H2.1 — 2026-09-23

- The legacy H2.1 rows contain `natural` and `mcq`, 50 samples per corpus,
  three visual conditions; prior report records the output cap as 64.
- Each VDC-MCQ manifest example has four candidate captions; exactly one
  equals `reference_answer`. Its gold text appears as one unmarked option in
  the prompt, while `correct_option` is metadata and is not separately shown.
  This is intentional MCQ answer-text leakage, distinct from disclosing a
  correct label outside the candidate list.
- The MCQ outputs are often option-only, which is why MCQ2 separates choice
  correctness from explanatory-answer length and format compliance.
- The correct checkpoint source is the **dataset** repo
  `Tphuc15/qwen25vl-3b-depth1-depth3`; the model-type repo with the same name
  only exposes metadata.
- Exact depth-3 artifacts:
  - LLaVA68K:
    `qwen25vl-3b-dflash-llava68k-h32-depth3-latest/training_state.pt`,
    3,530,854,875 bytes, SHA-256
    `77cdbc44dea0e08cc1ee958e3efcef61318e756a77366f4fcef799435d5c3d41`.
  - ShareGPT68K:
    `qwen25vl-3b-dflash-sharegpt68k-h32-depth3-latest/training_state.pt`,
    3,530,854,875 bytes, SHA-256
    `56cfd3a12bf01a50cf0391d33be3e9d4a50c3e60be676abb7a42bf89e912dc6a`.
- The expected local checkpoint files are absent. No replacement with the older
  6e/20e artifacts is valid for this comparison.
- GPU probe: RTX 3090 and RTX A4000 are available; both currently have enough
  free VRAM for the 3B target/draft pair. `/dev/shm` has about 15 GB free.
- The MCQ2 format-compliance metric is a mechanical proxy, not answer-quality
  scoring; option accuracy remains a separate metric.

## Pilot and runtime findings

- The one-sample diagnostic with the first wording produced a 122-token,
  109-word full selected caption, correct option D, and target/speculative
  equality. It used nine sentences and therefore failed the predeclared 1–2
  sentence format proxy.
- The prompt was tightened to request a concise one-to-two-sentence
  paraphrase and explicitly forbid copying an entire option. On the first
  five samples under this prompt, all three conditions per sample were valid:
  15/15 rows `ok`, mean speculative length 92.6 tokens (median 92), 81.2
  answer words, no output at the 128-token cap, 100% choice accuracy,
  100% lossless output, and 0% format-proxy compliance. The model still
  returned the multi-sentence option caption. Treat these as pilot-only.
- Historical H2.1 records confirm calibration tolerance deviations were
  allowed: sample `v_C7yd6yEkxXE` had relative error 0.28 and realized visual
  count 4,576 for target 3,000, yet was a successful row. MCQ2 therefore uses
  the same explicit allow flag and keeps actual token counts.
- The first pilot exposed two environment issues before any sample was written:
  a custom two-GPU target map fails tied-parameter validation, and
  `.venv` Transformers 4.49.0 lacks the DFlash flex-attention symbol.
  `.venv-msd` (Transformers 4.52.4, Torch 2.5.1+cu121) imports the DFlash
  model correctly; target-on-GPU0/draft-on-GPU1 then completed inference.
- The temporary LLaVA checkpoint was verified against HF metadata:
  3,530,854,875 bytes and SHA-256
  `77cdbc44dea0e08cc1ee958e3efcef61318e756a77366f4fcef799435d5c3d41`.

## Full runs and collected metrics

- LLaVA68K and ShareGPT68K each completed 50 samples × Full/Zero/Cut = 150
  rows; all 300 new rows are `status=ok`, with no missing target/speculative
  text or acceptance-by-position/round trace arrays.
- Within every sample and corpus, target-output hashes match across Full/Zero/Cut
  (50/50 samples per corpus). All 300 MCQ2 rows have lossless output.
- Combined descriptive analyzer input: 600 legacy rows + 300 MCQ2 rows = 900;
  `metric_records.csv`, `metric_summary.csv`, `metric_coverage.csv`, per-token/
  per-round JSONL details, and seven PNG metric plots were generated under
  `results/phase2_summary_20260923/h21_mcq2/`.
- Pooled across both draft corpora (n=100 per visual condition), MCQ2 means:
  - Full: accepted prefix 3.676 (n=96), global round acceptance 0.241,
    speculative time 3.556 s, speedup 1.310×.
  - Zero: accepted prefix 3.742 (n=96), global round acceptance 0.245,
    speculative time 3.537 s, speedup 1.317×.
  - Cut: accepted prefix 3.847 (n=96), global round acceptance 0.251,
    speculative time 3.507 s, speedup 1.328×.
- Response metrics are identical across visual conditions because target
  output is paired and all drafts are lossless: mean 112.78 output tokens,
  96.24 answer words, 7.84 sentences, 96% correct-option match, 0% strict
  format-proxy compliance, 58% reaching max_new_tokens=128, and 100% lossless.
- Prefix metric has n=96 because two 3-token outputs in each corpus have no
  defined accepted prefix; acceptance-rate denominator still covers all 100.
- Legacy natural/mcq outputs use a documented 64-token cap; MCQ2 uses 128.
  Thus output length/time across prompt variants are not same-cap comparisons.
- The scoped `/dev/shm` staging directories were deleted after runs finished.
- Root-cause review found `VideoSample`/`load_vdc_manifest` carried only
  `distractor_provenance`, not the manifest's four-option provenance. Added
  the missing typed field and loader mapping; the new real-loader regression
  test was observed failing before the fix and passing afterward.
- Backfilled `option_provenance` in all 339 new MCQ2 raw rows (300 final and
  39 pilot/diagnostic rows) from the canonical manifest. Successful-row
  candidates and labels matched; all other raw fields were verified unchanged.
  Nine pre-calibration error rows had no candidate payload and were mapped by
  sample ID only.
- Fresh verification: focused suite 31 passed; Python compilation passed; all
  87 local Markdown links in the README resolve. No I_format, early/late
  interaction, or hypothesis decision was computed.
