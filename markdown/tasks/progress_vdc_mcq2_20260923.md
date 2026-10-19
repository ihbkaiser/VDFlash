# Progress — VDC-MCQ2 H2.1 — 2026-09-23

## Completed

- Read the workspace experiment/checkpoint/report protocol.
- Confirmed old H2.1 journals contain 50 VDC-MCQ samples per corpus with
  Full/Zero/Cut and the two existing prompt variants.
- Added the independent `mcq2` prompt renderer and runner CLI support; the
  previous `mcq` ending remains unchanged.
- Added response parser metrics and raw per-row `max_new_tokens` metadata.
- Added analyzer fields for answer words/sentences, choice correctness,
  expected-format compliance, cap hits, and a response-metrics plot.
- Test-first verification: the new prompt and analyzer tests were observed
  failing before implementation.
- Fresh focused verification after provenance fix: 31 tests passed.

## Completed full runs and summary

- Ran a one-sample ShareGPT pilot and full 50-video runs for both checkpoints;
  each final journal contains 150/150 `status=ok` rows, 50 per condition.
- Verified target-output hash parity across Full/Zero/Cut for all 50 samples
  per corpus. Raw rows retain output text/IDs, acceptance traces, timing,
  provenance, cap metadata, and lossless flags.
- Aggregated the old 600 rows plus 300 MCQ2 rows (900 total) and generated
  seven plot files, including response/choice/format/cap visualization.
- Added pooled and per-corpus tables and embedded MCQ2 charts to the Phase-2 README.
- Review found the VDC loader dropped `option_provenance`; added the field and
  loader mapping with a regression test. Backfilled all 339 MCQ2 full/pilot
  rows from the canonical manifest; candidate/label and all non-provenance
  fields were validated unchanged.

## Environment/security notes

- Exact checkpoint copies were staged under task-scoped `/dev/shm` paths and
  removed after both GPU runners finished.
- Downloads used `token=False`; the credential-looking value in `HUGGING_FACE.md`
  was neither used nor copied. Keep the previously communicated rotation advice.

## Pilot findings / full-run settings

- Initial pilot established the target/draft placement and compatible
  environment; the first calibration-blocked pilot is retained separately as a
  diagnostic, not combined with data.
- A corrected five-sample pilot using `--allow-out-of-tolerance` completed
  15/15 `status=ok` rows over Full/Zero/Cut. No output reached the 128-token
  cap; target hashes were constant within sample.
- The response parser separately reports choice correctness (100% in this
  small pilot) and the stricter MCQ2 format proxy (0%: model repeated the
  multi-sentence caption option). Keep the prompt and metric fixed for the
  full run; report this behavior transparently.
- Final runtime: existing `.venv-msd`, target GPU0, draft GPU1, and the
  legacy calibration-tolerance flag. No repository-wide dependency change.
