# Hugging Face checkpoint cleanup plan

## Goal

Verify that the four local DFlash checkpoints are present in
`Tphuc15/vdflash-qwen25vl-3b-checkpoints`, document how future experiments can
retrieve and use them, then remove only the verified local checkpoint files.

## Scope and safety

- Keep all dataset, image, video, calibration, and source files.
- Do not overwrite unrelated dirty-worktree changes.
- Do not delete local checkpoints until remote verification and local reference
  checks are complete.
- Preserve the original `training_state.pt` files so future training resume
  remains possible.

## Phases

- [complete] 1. Inventory local files and verify the HF upload.
- [complete] 2. Write and review the reusable checkpoint usage guide.
- [complete] 3. Remove only the four verified local checkpoint files.
- [complete] 4. Verify deletion, documentation, and remaining code/data state.

## Errors encountered

| Error | Attempt | Resolution |
|---|---:|---|
| None for this task | 0 | — |

## Current decisions

- Use the existing HF dataset repo rather than create another repository.
- Use one remote subdirectory per checkpoint to preserve the four experiment
  variants.
- Update the guide with the current local runner limitation: it currently
  expects a local checkpoint path, so remote use requires `hf download` or an
  equivalent local cache step.
- The remote tree contains all four checkpoint paths with the same byte sizes
  as local files; expanded metadata also supplied exact SHA-256 values.
- Added `markdown/huong_dan_su_dung_dflash_checkpoints_huggingface.md` with
  download, hash-check, runner, compatibility, and recovery instructions.
