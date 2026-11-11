# DDTree video cut-sweep — plan

## Objective

Run a paired, reproducible DDTree video inference study across the locally
available VideoDetailCaption (VDC), VidHalluc, and VideoChair manifests. Keep
the target model and video preprocessing fixed while varying draft-context
visual retention. Save raw rows under `results/ddtree_eval_20261019/` and the
analysis/report in this folder.

## Locked protocol

- Target: local Qwen2.5-VL-3B-Instruct; drafter: local Qwen2.5-VL-3B DFlash,
  depth 3, block size 16.
- Datasets/manifests: VDC 50 records, VidHalluc 100, VideoChair 100.
- Visual retention in draft context: 100%, 50%, 25%, 0%; use the existing
  DFlash runner's prefix-retention selection for direct comparability. The
  target always receives the complete video.
- Video processing: 16 frames, 1 fps, max visual pixels `256 * 28 * 28`;
  greedy decoding, maximum 256 generated tokens.
- Compare target autoregressive decoding, linear DFlash, and DDTree at a fixed
  tree-node budget of 16. Reuse one target baseline per sample and keep all
  methods/cuts paired to that baseline.
- Primary performance metric: end-to-end speedup against target AR. Report
  paired sample bootstrap 95% CIs. Guardrail: generated token IDs must exactly
  match the target AR output (100% lossless); report task quality separately.
- Secondary metrics: prefill/decode/end-to-end latency, tokens/s, decode and
  end-to-end speedup, tau proposal/effective, accepted nodes/proposals,
  verification rounds and calls, draft/tree-build/tree-compile/verify/commit
  time, peak allocated VRAM, output lengths/hashes, task metrics, runtime error
  and coverage counts.

## Feedback ladder

| Rung | Gate | State |
|---|---|---|
| R0 | Protocol and experiment card fixed | Complete |
| R1 | Imports/config/local paths/device visibility | Complete |
| R2 | One VDC sample, all four cuts, exact output parity | Complete; parity gate fails for long captions |
| R4/R5 | Full runner smoke/pilot and data coverage check | Complete; eager pilot has no runtime failures, exactness is not universal |
| R6 | Full 250-sample × 4-retention sweep, all planned methods | Partial: 1,999/2,000 speculative conditions; one final VideoCHAIR DDTree 0% row is missing after a GPU/NVML fault |
| R7 | Paired analysis, figures, and result memo | Analysis/report complete for successful rows; the affected VideoCHAIR 0% cell remains provisional |

## Errors and decisions

| Date | Finding | Action |
|---|---|---|
| 2026-10-19 | Sandbox process has no `/dev/nvidia*`; elevated read-only check sees RTX 3090 + RTX A4000. | Run inference commands with hardware access outside the sandbox; keep files in the writable workspace. |
| 2026-10-19 | Existing `run_dflash_qwen25vl3b_eval.py` is linear DFlash, not DDTree. | Add an isolated DDTree decoder/runner; do not relabel historical DFlash rows. |
| 2026-10-19 | VDC local manifest is a 50-record subset; VidHalluc and VideoChair are 100 records each. | Report this as local subset coverage, not full official benchmark coverage. |
| 2026-10-19 | First DDTree smoke exposed incorrect target loader return ordering and replayed video kwargs during cached target verification. | Corrected the local runner/decoder and reran smoke successfully. |
| 2026-10-19 | SDPA and eager pilot outputs are not universally identical to target AR; short binary VidHalluc answers match, longer captions can diverge. | Keep token-level exact match as a reported guardrail, but complete the requested sweep as an empirical quality/performance study and do not claim losslessness. |
| 2026-10-19 | Eager attention improves some parity cases but can run slower; measured outer wall time now synchronizes both GPUs and records per-GPU peak allocated VRAM. | Use eager for the diagnostic/full run and retain backend in summary metadata. |
| 2026-10-19 | The full run completed VDC 400/400, VidHalluc 800/800, and VideoCHAIR 799/800 conditions with no runtime-error rows. The last VideoCHAIR DDTree/0% condition stalled while NVML reported an unknown RTX A4000 error; resetting the GPU was avoided because the RTX 3090 also had active desktop/Xorg processes. | Preserve the missing condition ID and hardware interruption in run metadata; report successful-row analyses and mark only the affected cell provisional. |
| 2026-10-19 | The original VidHalluc Sequence_Ranking scorer did not parse natural-language Action A/B ordering and marked all such rows incorrect. | Add regression tests; parse explicit labels in order; recompute task annotations for 100 AR and 800 speculative saved rows without changing inference outputs; regenerate all analysis with 10,000 paired/sample bootstrap resamples. |
