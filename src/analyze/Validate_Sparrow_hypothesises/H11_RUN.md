# H1.1: Full/Cut versus the Phase 2 training representation

Use the **same completed Phase 2 draft checkpoint** for both commands. Point
`--draft-config` to the resolved config that accompanied that checkpoint, and
`--feature-root` to its Phase 2 LLaVA teacher-feature `.ckpt` directory. The
reference reader verifies the target-layer/phase metadata. Neither command
trains a model or writes into the checkpoint/cache.

## One-command launcher

The supplied checkpoint, cache, model paths and two-GPU settings (target on
`cuda:0`, draft on `cuda:1`) are already filled in
`scripts/h11_3b_20e_cache.env`. From the repository root, run:

```bash
bash scripts/run_h11_3b_20e.sh
```

This profile captures 1,000 Full training references and 200 paired Full/Cut
cache examples. The earlier 200/2 setting was only a smoke test. Allow a new
timestamped output directory for each run; an already-running job keeps the
sample counts with which it started.

The prefilled profile uses `scripts/h11_full_dflash_3b.json` (five trained
draft layers). Before loading the target model, the runner checks that the
Phase 2 checkpoint actually has layers `0..4`. A checkpoint trained with only
three layers cannot be converted to five by editing its JSON; the check stops
and requests a matching checkpoint.

Copy `scripts/h11.env.example` to a private `.env` file, replace its absolute
checkpoint, Phase 2 feature-cache, config, and model paths, then run
from the repository root:

```bash
bash scripts/run_dflash_h11.sh --env-file /absolute/path/to/h11.env
```

The default `H11_EVAL_MODE=cache` runs Full/Cut on **a separate group of cached
LLaVA training examples**, disjoint from the reference group. The model has
already seen both groups during Phase 2 training. This mode needs no video or
original image files. It compares first-block draft proposals to stored teacher
tokens, without a fresh target verification pass. This is an **in-domain
representation-shift test**, not evidence about VDC or external image
captioning; acceptance here is a first-block proxy, not runtime tau.

Set `H11_EVAL_MODE=video` and the three VDC paths in the env file when video is
available. Both modes print a layer-by-layer table and save
`run.log`, `summary.txt`, `analysis/summary.json`, and
`analysis/paired_distances.jsonl` under a new timestamped `results/h11_*`
directory. It stops on any failed stage and prints the log path.

## Individual commands for the video mode

Run from the repository root on the CUDA machine, in the project environment:

```bash
python -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_h11 reference \
  --checkpoint /path/to/phase2/training_state.pt \
  --draft-config /path/to/draft_config_phase2.json \
  --feature-root /path/to/phase2/hidden_states \
  --target-model /path/to/qwen25-vl-3b \
  --reference-samples 200 --train-max-length 3072 \
  --output-dir results/h11_pilot

python -m src.analyze.Validate_Sparrow_hypothesises.run_dflash_h11 evaluate \
  --checkpoint /path/to/phase2/training_state.pt \
  --draft-config /path/to/draft_config_phase2.json \
  --target-model /path/to/qwen25-vl-3b \
  --manifest /path/to/VideoDetailCaption/test.jsonl \
  --video-root /path/to/VideoDetailCaption \
  --calibration /path/to/VideoDetailCaption/calibration.jsonl \
  --limit 2 --max-new-tokens 32 \
  --output-dir results/h11_pilot

python -m src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11 \
  --reference results/h11_pilot/train_full.npz \
  --evaluation results/h11_pilot/test_full_cut.npz \
  --reports results/h11_pilot/test_full_cut.jsonl \
  --output-dir results/h11_pilot/analysis
```

The defaults for target and draft devices are `cuda:0` and `cuda:1`; override
`--device`, `--draft-device`, `--device-map`, and `--max-memory` when necessary.
For the full VDC50 run, first increase `--reference-samples` (for example to
1000, if that many cache records are available), then set `--limit 50` in
`evaluate`, using a new output directory. The 2-sample pilot checks plumbing;
its confidence interval and correlation have no inferential value.

`train_full.npz` holds one first-supervised-anchor Full vector per randomly
selected training file. `test_full_cut.npz` contains paired first-block query
vectors after attention and after each draft layer. The analyzer fits a
regularized PCA distance on 75% of the training vectors, calibrates a 95th
percentile threshold on the other 25%, and reports Full/Cut paired distances,
bootstrap intervals, OOD rates, first-block acceptance, and tau. The target
always sees the full video and paired rows must share its input/output hashes.
Acceptance measures draft–target agreement, not caption quality.
Set `--train-max-length` to the Phase 2 training config's `data.max_length` if
it differs from 3072. Files without a supervised anchor inside that boundary
are skipped and recorded by the reference job's progress count.

## Follow-up: representation shifts outside the 32 PCA directions

After the H1.1 cache run finishes, compare the old PCA distance with the
distance **outside** those training directions, using its existing NPZ/JSONL
files. This is CPU-only and does not load either model or rerun inference:

```bash
python3 -m src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11_deep \
  --run-dir results/h11_20260923T192807Z
```

The command prints a layer-by-layer table and writes `analysis_deep/summary.txt`,
`summary.json`, and `paired_residuals.jsonl` under that run directory. Positive
`Residual delta` means Cut deviates from the Full training reference in
directions the original PCA score omitted. `rho` compares each sample's
residual shift with its loss of first-block teacher-token matches; `adjusted
rho` also accounts for visual-token count and anchor position. Bootstrapped
95% intervals show sampling variation. PCA and residual distance use different
units, so compare their signs and OOD rates, not the magnitudes of their deltas.
These associations across layers are exploratory and do not establish
causality, and the cache-mode outcome is not runtime video acceptance.
