# Findings

- Baseline experiment: `results/dflash_free_running_main_fp16_20261014/`.
- Baseline protocol: 33 calibrated videos, five retention conditions (100, 75,
  50, 25, 0%), DFlash depth 3, FP16, greedy proposals, refreshed target
  features, maximum 256 generated tokens.
- Baseline raw journal has 165 rows: 124 non-empty outputs and 41 immediate EOS
  outputs. Prior token-ID audit found first adjacent repeats in the first
  15-token draft block for the non-empty rows; repeated runs also cross block
  boundaries. This motivates reporting first-block and later-block metrics
  separately.
- Target-bonus arm definition: each complete 15-token draft proposal block is
  followed by one greedy target token selected after processing that block.
  The draft tokens remain unverified. The bonus token is a new target-selected
  output token and may serve as the next draft anchor.
- Target-bonus arm does not test standard speculative decoding. A separate
  exact-verification condition would be needed to study that behavior.
- The project `.venv` contains PyTorch 2.1.2+cu121. The normal sandbox hides
  `/dev/nvidia*`; escalated access exposes the RTX 3090, but PyTorch sees only
  one device. The RTX A4000 is present at PCI `02:00.0` but NVIDIA SMI reports
  `Unknown Error` for it.
- Latest RTX 3090 memory check: 20,742 MiB used of 24,576 MiB; PyTorch reports
  about 3.15 GiB free. No model load or inference was started with this amount
  of free memory.
- The filesystem has about 7.7 GB free and reports 100% use. The model
  checkpoint is already local; avoid downloading or copying model weights.
- Existing `markdown/19_10_26/report.md` and its figures are a separate verified
  speculative benchmark; new artifacts must stay in this task's subdirectory.
- A preflight using only `--limit 50` selected 50 records, not the baseline's 33.
  No inference was run with this mismatched cohort. The new arm must explicitly
  use the baseline protocol's 33 `cohort_sample_ids`.
- Re-running preflight with all 33 baseline `--sample-id` values under elevated
  GPU access succeeds and reports `selected_samples=33`, `planned_rollouts=165`,
  and `cuda_available=true`. This is a metadata-only preflight and does not
  establish that the models fit the currently available VRAM. The exact IDs
  are read from the baseline protocol by `run_experiment.sh`.
- Decoder alignment for a target-bonus block: refresh the target on the previous
  anchor plus all 15 draft proposals; use its last-position greedy logits for
  the bonus token. The new bonus becomes the next draft anchor and is processed
  by the target during the following refresh.
