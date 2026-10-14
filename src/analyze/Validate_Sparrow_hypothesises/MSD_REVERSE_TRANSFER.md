# MSD reverse text-to-visual transfer

This experiment tests whether the trained MSD draft can score answer labels
when a multimodal prompt is ordered `video -> text` versus `text -> video`.
It is separate from the older target-hidden/DFlash diagnostics in this
directory.

The primary prediction path is:

```text
Qwen2-VL target prefill
  -> target final hidden states + target visual input embeddings
  -> trained model.ea_layer (MSD/EAGLE draft)
  -> draft hidden state at the final prompt position
  -> frozen model.base_model.lm_head
  -> restricted A/B/C/... candidate probabilities
```

Target vocabulary logits are not used for the primary scores. The target
hidden states are conditioning features only. The current compatible trained
checkpoint is `lucylyn/MSD-Qwen2VL-7B-Instruct` paired with
`Qwen/Qwen2-VL-7B-Instruct`; this is the vendored Qwen2-VL MSD path, not a
Qwen2.5-VL checkpoint.

## Conditions

| ID | Order | MSD-side conditioning |
|---|---|---|
| E1 | Video -> Text | Full |
| E2 | Text -> Video | Full |
| E3 | Video -> Text | Visual-only |
| E4 | Text -> Video | Visual-only |
| E5 | Video -> Text | Text-only |
| E6 | Text -> Video | Text-only |

For visual-only and text-only conditions, the target still receives the full
multimodal prompt. Rows not belonging to the retained modality are zeroed at
both the target-hidden and input-embedding interfaces while sequence length
and positions are preserved. This means contextual target hidden states on
retained visual rows can carry information transferred during the full target
prefill; the report records that the draft visual embedding rows themselves
remain the original target visual-tower embeddings.

The implementation passes `position_ids=None` to the vendored EAGLE draft so
that `cnets.Model` uses its native 2-D sequential-position contract and
expands it internally. Passing the target Qwen M-RoPE tensor directly would
flatten its three channels into the batch dimension and is invalid for this
MSD implementation.

## Local checks

```powershell
py -3.11 -m pytest -q `
  src/analyze/Validate_Sparrow_hypothesises/tests/test_reverse_transfer.py `
  src/analyze/Validate_Sparrow_hypothesises/tests/test_reverse_transfer_runtime.py `
  src/analyze/Validate_Sparrow_hypothesises/tests/test_reverse_transfer_cli.py `
  src/analyze/Validate_Sparrow_hypothesises/tests/test_reverse_transfer_report.py
```

## GPU smoke on Modal

The launcher mounts the HF cache, data volume, and results volume. It fails if
CUDA or the compatible MSD checkpoint is unavailable; it never substitutes a
target-only or random draft.

```powershell
$env:PYTHONIOENCODING = "utf-8"
modal run src/analyze/Validate_Sparrow_hypothesises/modal_reverse_transfer.py `
  --args "--dataset mvbench `
    --dataset-path /data/visual_probing_sparrow_mvbench_hard_l4/manifest/mvbench_hard_mcq.jsonl `
    --video-root /data/visual_probing_sparrow_mvbench_hard_l4/videos/mvbench `
    --num-samples 1 --num-frames 4 --max-pixels 200704 `
    --conditions E1 E2 E3 E4 E5 E6 --permutations --include-d4 `
    --output-dir /results/reverse_transfer/msd_smoke"
```

Results contain `raw_results.jsonl`, `d4_video_only.jsonl`,
`run_config.json`, and a generated `report/` directory. D4 is diagnostic only
and is not included in primary VQA accuracy. Resume with `--resume` and use a
new or existing output directory as appropriate.

For a real evaluation, provide the supported MVBench or VidHalluc manifest,
keep the same `--num-frames`, pixel budget, checkpoint names, and output a
separate artifact directory for each benchmark. The report includes condition
summaries, paired contrasts, bootstrap intervals, and label-permutation
consistency. A smoke result is not evidence for the full benchmark.
