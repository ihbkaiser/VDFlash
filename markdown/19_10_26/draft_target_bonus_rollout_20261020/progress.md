# Progress

## 2026-10-20

- Confirmed the user authorized the separate `15 draft + 1 greedy target bonus`
  arm without proposal verification.
- Created this isolated output directory to preserve the existing
  `markdown/19_10_26` report and figures.
- Reviewed the existing free-running decoder and baseline protocol.
- Confirmed `markdown/19_10_26` already has an unrelated report and figures;
  this run is isolated in `draft_target_bonus_rollout_20261020/`.
- Specified alignment: target refresh consumes the prior anchor and all 15 draft
  proposals, selects the next token from the final target logits, then leaves
  that bonus token as the next draft anchor.
- First preflight completed but exposed cohort mismatch: `--limit 50` selects 50
  samples while the paired baseline protocol has 33. No inference was started.
  Launcher is being changed to pin the exact 33 baseline IDs before rerunning
  preflight.
- Target-bonus decoder and runner implemented; focused CPU contract test passed
  after first failing on the missing mode. The existing DFlash free-running
  integrity tests also pass (23 total across both test files), and all modified
  Python files compile.
- Cohort-pinned preflight now reports 33 selected samples and 165 planned
  rollouts. CUDA is still unavailable, so no model inference has run.
- Analyzer import path fixed after its first `--help` check failed; the corrected
  CLI help and syntax compilation pass.
- Ran the experiment launcher once; its CUDA guard stopped before model loading
  or raw-journal creation, as intended for this environment.
- Retried `run_experiment.sh` on 2026-10-20 (Australia/Sydney); it exited with
  status 1 and `CUDA is not available`. Captured the attempt in
  `run_attempt.log`; no model was loaded and no rollout row was written.
- Initial ordinary-sandbox check: project venv has PyTorch, but CUDA is
  unavailable and no NVIDIA device nodes are exposed. No inference process was
  started from that sandbox context.
- Rechecked through escalated GPU access after the user reported two cards:
  PyTorch sees one RTX 3090; the RTX A4000 at PCI `02:00.0` returns
  `Unknown Error`. The 3090 had 20,742 MiB used and only about 3.15 GiB free.
- Re-ran the cohort-pinned preflight with `cuda_available=true`; it confirms
  33 samples and 165 planned rollouts. Did not load models because available
  VRAM is insufficient and the second card is not usable from this process.
- Next: restore visibility/health of the A4000 or free sufficient VRAM, then
  run the 165-row journal and generate the paired figure and report.
