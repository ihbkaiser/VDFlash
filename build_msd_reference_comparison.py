"""Build a reference-comparison JSONL from the MSD 2-sample run.

Reads results/sparrow_validation/msd_full_2samples.jsonl and the VDC manifest,
decodes AR/Spec token IDs to text, and writes rows with 'prediction' and
'reference' for the existing compare_dflash_reference.py scorer.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoProcessor

MSD_ROWS = "results/sparrow_validation/msd_full_2samples.jsonl"
MANIFEST = "dataset/VideoDetailCaption/subset_manifest.jsonl"
OUT = "results/sparrow_validation/msd_vs_reference.jsonl"
MODEL = "Qwen/Qwen2-VL-7B-Instruct"


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    proc = AutoProcessor.from_pretrained(MODEL)
    msd_rows = load_jsonl(MSD_ROWS)
    manifest = {r.get("video_name") or r.get("sample_id"): r for r in load_jsonl(MANIFEST)}

    out_rows = []
    for row in msd_rows:
        sample_id = row["sample_id"]
        ref = manifest[sample_id]["answer"]
        ar_text = proc.batch_decode([torch.tensor(row["target_output_ids"])])[0]
        spec_text = proc.batch_decode([torch.tensor(row["speculative_output_ids"])])[0]
        out_rows.append(
            {
                "sample_id": sample_id,
                "visual_percentage": row.get("actual_visual_tokens"),
                "status": "ok",
                "outputs_match": row["lossless"],
                "num_output_tokens": len(row["speculative_output_ids"]),
                "prediction": spec_text,
                "prediction_ar": ar_text,
                "reference": ref,
            }
        )

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        for r in out_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in out_rows:
        print("=" * 70)
        print(f"sample: {r['sample_id']}  lossless={r['outputs_match']}")
        print(f"PRED (MSD): {r['prediction']}")
        print(f"REF       : {r['reference']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
