# Progress — 2026-10-19

1. Inspected `WORKFLOW_NOTES.md`, prior VDC ablations, current DFlash runner,
   local manifests, video files, target/checkpoint paths, and existing raw
   results. No existing files were edited.
2. Confirmed the local full-data scope: VDC 50, VidHalluc 100, VideoChair 100;
   all corresponding local video files are present.
3. Confirmed sandbox CUDA visibility is absent. An elevated read-only `nvidia-smi
   -L` check confirmed RTX 3090 and RTX A4000 are available outside the
   sandbox.
4. Started an isolated DDTree decoder/runner so the repository's current
   DFlash/MSD edits and historical artifacts remain untouched.
5. Fixed loader ordering and cached multimodal verification inputs found by the
   smoke run. Added synchronized end-to-end timing, per-device peak allocated
   VRAM, prompt hashes, speculative token IDs, and attention-backend metadata.
6. Passed smoke and eager pilot runs on two videos from each dataset (6 total;
   48 speculative conditions, zero runtime failures). Short VidHalluc binary
   answers were exact; longer caption outputs were not consistently token
   identical to target AR. The full study therefore measures quality changes
   alongside speed and acceptance, without treating DDTree as lossless.
7. Began the requested full sweep on local VDC-50, VidHalluc-100, and
   VideoChair-100 manifests, four visual-retention levels, target AR, DFlash,
   and DDTree.

## Validation log

| Check | Result |
|---|---|
| `git status` and local changes | Existing user changes present; preserve them. |
| Local paths and data coverage | Pass: target/checkpoint/config and all 250 manifest video paths exist. |
| PyTorch/Transformers import | Pass in `.venv`: torch 2.1.2+cu121, Transformers 4.52.4. |
| Sandbox CUDA | Unavailable: `torch.cuda.is_available() == False`, device count 0. |
| Hardware outside sandbox | Pass: RTX 3090 and RTX A4000 visible. |
| Python syntax / CLI import | Pass: decoder and runner compile; `--help` loads. |
| Smoke/pilot | Pass: 6 samples × 8 speculative conditions, eager backend, no runtime errors. |
| Exact-output pilot guardrail | Fail by design evidence: long-caption rows diverge; report rates and quality metrics. |

8. The full inference pass reached 1,999/2,000 speculative conditions: VDC
   400/400, VidHalluc 800/800, and VideoCHAIR 799/800. Runtime-error rows are
   zero; the only missing ID is
   `video9691:ddtree:depth3:retention-0:budget-16`. The last condition stalled
   with the RTX A4000 reporting an NVML unknown error. The RTX 3090 still had
   desktop/Xorg processes, so no disruptive GPU reset was attempted.
9. Added regression tests for natural-language VidHalluc Sequence_Ranking
   outputs. They failed against the original compact-label-only parser, then
   passed after parsing explicit Action A/B mentions in order. Recomputed the
   task-metric annotations for all 100 AR and 800 speculative VidHalluc rows;
   verified that inference outputs, hashes, timings and traces were unchanged.
10. Regenerated all tables, paired sample-bootstrap CIs, integrity results and
    figures with 10,000 bootstrap resamples. Coverage is 24/24 condition cells,
    with one row missing only from VideoCHAIR DDTree at 0% retention; successful
    row integrity errors remain zero. Sequence_Ranking is now 16/20 (80%) for
    AR, DFlash and DDTree at every retention level.

## Final verification

| Check | Result |
|---|---|
| `tests/test_ddtree_video_benchmark_metrics.py` | Pass: 3 tests, including both orderings and incomplete-answer handling. |
| Python compilation | Pass: DDTree runner and analysis script compile. |
| Analysis regeneration | Pass: VDC 400/400, VidHalluc 800/800, VideoCHAIR 799/800; 0 runtime-error rows, 0 integrity errors. |
| Figures and report artifacts | Pass: report, five figures, six analysis files, and run metadata present. |
| Full `pytest -q` | Collection blocked by two existing tests importing the absent `src.analyze.Whether_they_are_appliable_for_dDrafter` package; the focused DDTree metric tests pass. |
| End-to-end sweep | Incomplete by one condition due to GPU/NVML stall; no reset attempted. |
