# Repository Guidelines

## Project Structure & Module Organization

- `src/train_VLM/` contains the primary Video-DFlash pipeline: data preparation, teacher-feature caching, draft training, evaluation, and video inference. Presets and launchers live beside the modules.
- `src/analyze/` contains benchmark and hypothesis-validation experiments; `src/dataset/` contains dataset preparation utilities.
- `tests/` contains the repository-level pytest suite; focused experiment tests are under `src/analyze/Validate_Sparrow_hypothesises/tests/`.
- `externals/` holds vendored or adapted upstream projects; keep local changes isolated and documented.
- Research notes are in `markdown/`. Generated datasets, caches, checkpoints, and reports should remain in configured artifact/data directories and must not be committed.

## Build, Test, and Development Commands

From the root, create a Python 3.11 environment and install `requirements.txt`. Useful commands are:

```bash
python -m pytest -q                                      # full test suite
python -m pytest -q tests/test_visual_context.py         # focused regression test
python -m src.train_VLM.prepare_data --config CONFIG.json # deterministic manifest
src/train_VLM/train_3090_smoke.sh                        # end-to-end GPU smoke run
```

The smoke script requires CUDA and model/data access. Full caching and training use `torchrun`; follow `README.md` for config and GPU count.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8-compatible Python, `snake_case` for functions, variables, and modules, and `PascalCase` for classes. Prefer type hints and short docstrings for public APIs and CLI entry points. Keep modules runnable with `python -m ...`; use explicit config arguments. Shell scripts should be Bash-compatible, quote paths, and expose tunable values through named environment variables.

## Testing Guidelines

Tests use pytest and follow `test_*.py` naming. Add regression tests beside the affected area, mock or skip model/GPU-dependent work where practical, and run the focused test before the full suite. CUDA integration tests are skipped when CUDA is unavailable; no coverage threshold is configured.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects such as `Add ...`, `Fix ...`, or `Update ...`; history also includes concise Vietnamese summaries. Keep commits focused. Pull requests should explain the motivation, affected configs/modules, datasets or checkpoints, and validation results. Include benchmark tables or screenshots for output changes, link the relevant issue or experiment, and call out GPU, storage, or Hugging Face requirements.

## Security & Configuration Tips

Never commit access tokens, local model paths, raw datasets, checkpoints, or large generated caches. Use Hugging Face CLI authentication and variables such as `HF_HOME` for credentials and storage configuration. Before sharing results, check logs and configs for secrets or private dataset locations.
