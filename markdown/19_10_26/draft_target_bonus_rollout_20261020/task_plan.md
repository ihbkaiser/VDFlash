# Task plan: draft 15 + target bonus rollout

## Goal

Run a separate unverified rollout arm that emits one greedy target token after
each complete 15-token DFlash proposal block. Preserve the earlier cohort,
checkpoint, visual-retention conditions, prompt, FP16 setting, feature refresh,
and 256-token cap. Save raw logs, block-level repetition analysis, plots, and a
report under this directory. Keep the existing `markdown/19_10_26` artifacts
unchanged.

## Design decisions

- The target bonus is selected from target logits after the target processes the
  old anchor and all 15 draft proposals.
- The 15 proposals remain unverified and are never rolled back. Verification
  calls remain zero; this arm is not standard speculative decoding.
- A target bonus is emitted only when a full draft block was emitted without
  EOS and the token budget has room for it.
- Compare with the paired 2026-10-14 draft-only raw rollouts. Analyze the first
  draft block separately from later blocks because repetition can begin within
  a parallel block before a bonus token is added.

## Phases

1. Inspect prior protocol, outputs, runner, environment, and workspace state —
   complete.
2. Add the target-bonus rollout mode and record block provenance — complete.
3. Run focused code validation and a cohort-pinned preflight — complete.
4. Run preflight pinned to the baseline's exact 33 sample IDs — complete; run
   the full paired cohort — blocked by unavailable second GPU and insufficient
   free VRAM on the one visible GPU.
5. Analyze paired repetition, plot results, and write the report — blocked on
   generation output.

## External blocker

The ordinary sandbox hides NVIDIA devices, while an escalated CUDA check exposes
one RTX 3090. PyTorch reports one visible GPU, with about 3.15 GiB free out of
23.66 GiB at the latest check. The host's RTX A4000 at PCI address `02:00.0`
returns `Unknown Error` from NVIDIA SMI and is not enumerated by PyTorch. The
3B target plus DFlash draft cannot be loaded safely with the current available
VRAM. Do not substitute synthetic results for the requested inference results.
Resume phase 4 when both required GPUs are healthy/visible or enough VRAM is
freed on a supported device.

The first preflight with `--limit 50` selected 50 records rather than the
baseline's 33. No inference was started. The launcher must read and pass the
exact baseline `cohort_sample_ids` as repeated `--sample-id` arguments.

## Errors encountered

| Issue | Evidence | Resolution |
|---|---|---|
| Preflight selected 50 instead of the paired 33 | `preflight.json` from `--limit 50` | Launcher now reads all 33 IDs from the baseline protocol; corrected preflight reports 165 rollouts |
| Analyzer could not import `src` | `analyze_rollouts.py --help` raised `ModuleNotFoundError` | Corrected repository-root path; `--help` and compilation now succeed |
| CUDA generation unavailable | No `/dev/nvidia*`; `torch.cuda.is_available()` is false; launcher exits before model load | Requires a CUDA-enabled session; no inference data or plot was fabricated |
| GPU resources insufficient in elevated session | Only RTX 3090 is enumerated; 3.15 GiB free; RTX A4000 reports `Unknown Error` | Preflight completed for 165 rows; generation withheld to avoid a predictable model-load OOM |

## Isolation note

`markdown/19_10_26` already contains another report and five figures dated
2026-10-20. Those files describe a separate verified speculative benchmark and
will not be edited by this task.
