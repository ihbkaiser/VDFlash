# Findings: HF checkpoint cleanup

- HF destination supplied by the user: `Tphuc15/vdflash-qwen25vl-3b-checkpoints`.
- User's upload output reports 1,348 files and 25.3 GB processed, including the
  four DFlash checkpoints and dataset/video files.
- Local checkpoint candidates are four untracked directories under `dataset/`.
- Each local `training_state.pt` is 5,688,979,511 bytes (about 5.30 GiB); total
  local checkpoint storage is about 21.2 GiB.
- The four checkpoint files are not Git-tracked, so removing them frees local
  disk space without rewriting Git history.
- Current DFlash loaders resolve local `training_state.pt` paths; remote HF
  download integration is not currently implemented in the loader.
- Remote verification at commit `634a96fdbabfd9131ab2912bf1f24638c6f09296`
  found all four files at the repository root, each with size
  `5,688,979,511` bytes.
- Local SHA-256 values matched the remote LFS SHA-256 values exactly:
  `8b3a3c0a...95b17` (20e/LLaVA), `6be2ab5e...b758` (20e/ShareGPT),
  `6ec8030d...2755` (6e/LLaVA), and `44fc2ff9...cf4e` (6e/ShareGPT).
- The usage guide is at
  `markdown/huong_dan_su_dung_dflash_checkpoints_huggingface.md`.
- User-confirmed repository mapping for future experiments:
  - `Tphuc15/qwen25vl-3b-depth1-depth3` → H3.2 depth1/depth3 checkpoints.
  - `Tphuc15/vdflash-qwen25vl-3b-checkpoints` → five-layer DFlash checkpoints
    (6e/20e variants for LLaVA68K and ShareGPT68K).
- The currently materialized H3.2 depth1/depth3 directory is
  `dataset/qwen25vl-3b-depth1-depth3` and occupies about 9.2 GB; it is a
  separate deletion candidate from the older four-checkpoint cleanup recorded
  above.
- A retry on 2026-09-18 with authenticated user `VietNguyen865` still returned
  `403 Forbidden` for the gated five-layer repository; no five-layer file was
  downloaded.
- The four local `training_state.pt` files and their now-empty parent
  directories were removed after the pre-delete checks.
- Final checks confirmed all four remote files remain available with the same
  sizes and SHA-256 values; local `dataset/` remains present with data files
  and is about 2.4G after cleanup.
- The two documented validation shell scripts pass `bash -n`; the guide has
  no trailing whitespace.
