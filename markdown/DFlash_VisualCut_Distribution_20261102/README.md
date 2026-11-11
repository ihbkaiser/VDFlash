# DFlash VisualCut distribution experiment (2026-11-02)

This folder contains the final report and compact analysis/provenance documents for run `vdc50_balanced_distribution_replay_full_forced_trajectory_20261102`.

## Files

- [Final report](final_report.md): study design, results, interpretation, debugging notes, and evidence limits.
- [Colab summary tables](colab_analysis/summary.csv), [summary JSON](colab_analysis/summary.json), and [sample means](colab_analysis/sample_means.csv): outputs produced on Colab.
- [Verified local reanalysis](verified_local_reanalysis/summary.csv), [summary JSON](verified_local_reanalysis/summary.json), and [sample means](verified_local_reanalysis/sample_means.csv): rerun against downloaded artifacts; the 136 summary CSV rows match Colab exactly.
- [Artifact audit](artifact_audit.json), [run config](run_config.json), and [model-load record](model_load.json): verification and provenance.
- [Full-accepted-position analysis](accepted_by_full_analysis/summary.csv) and [metadata](accepted_by_full_analysis/summary.json): compares full/cut distributions only where full top-1 matches the target; it is a fixed-replay per-position analysis.
- [Four-way match-state breakdown](accepted_subset_analysis/summary.csv): both-match, full-only, cut-only, and neither-match groups.
- [Source snapshot note](source_snapshot_README.md): source and runtime bytecode provenance detail.
- [Colab CLI procedure](../visualcut_distribution_replay_colab.md): preparation, pilot, and full-run instructions.

## Full artifacts

The complete run artifacts, including all 200 paired per-position gzip files, logs, calibration records, and the independently rechecked outputs remain under:

`results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/`

The large pair artifacts are kept there to avoid duplicating them in this documentation folder. No full-vocabulary logits or target hidden states were saved.
