"""Paired equal-budget visual/text deletion on existing H1.1 cache examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.infer.qwen25vl_dflash_compare import _load_draft, _load_target, _resolve_dtype
from .analyze_dflash_h11_matched import choose_positions
from .run_dflash_h11 import (
    _cached_forward, _ensure_specforge_importable, _visual_token_ids,
    attention_summary, first_valid_anchor,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--draft-config", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--train-max-length", type=int, default=3072)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="cuda")
    parser.add_argument("--max-memory", default="0:22GiB,1:14GiB")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    args = parser.parse_args(argv)
    if args.budget < 1 or args.train_max_length < 2:
        parser.error("budget must be positive and train-max-length must be at least two")
    if not torch.cuda.is_available():
        raise RuntimeError("matched H1.1 capture requires CUDA")
    _ensure_specforge_importable()
    from specforge.runtime.data_plane.feature_store import load_feature_file

    with np.load(args.source_run / "test_full_cut.npz", allow_pickle=False) as existing:
        source_ids = [str(value) for value in existing["sample_ids"].tolist()]
        original_checkpoint = str(existing["checkpoint"].item())
    if original_checkpoint != str(args.checkpoint):
        raise ValueError("matched control must use the same draft checkpoint as the original H1.1 run")
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise ValueError("missing or duplicate source sample IDs")
    original_full = {}
    with (args.source_run / "test_full_cut.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("visual_condition") == "full":
                original_full[str(row["sample_id"])] = row
    if set(original_full) != set(source_ids):
        raise ValueError("original Full reports do not match source sample IDs")
    dtype = _resolve_dtype(args.dtype, torch.device(args.device))
    print(f"[H1.1 matched] Loading target on {args.device}", flush=True)
    processor, target, _ = _load_target(
        args.target_model, device=torch.device(args.device), dtype=dtype,
        attention="sdpa", device_map=args.device_map, max_memory=args.max_memory,
    )
    print(f"[H1.1 matched] Loading draft on {args.draft_device}", flush=True)
    draft, _ = _load_draft(
        str(args.checkpoint), str(args.draft_config),
        device=torch.device(args.draft_device), dtype=dtype,
    )
    draft.config._attn_implementation = "eager"
    visual_ids = _visual_token_ids(target)
    special_ids = set(int(value) for value in processor.tokenizer.all_special_ids)
    pairs, reports, sample_ids = [], [], []
    skipped = 0
    feature_root = args.feature_root.resolve()
    for ordinal, sample_id in enumerate(source_ids, 1):
        path = (feature_root / sample_id).resolve()
        if not path.is_relative_to(feature_root) or not path.is_file():
            raise FileNotFoundError(f"source feature file unavailable: {sample_id}")
        raw = load_feature_file(str(path))
        ids = raw["input_ids"].reshape(-1)[:args.train_max_length].tolist()
        try:
            anchor = first_valid_anchor(raw["loss_mask"].reshape(-1)[:args.train_max_length])
        except ValueError:
            raise ValueError(f"previously valid H1.1 sample lacks an anchor: {sample_id}") from None
        positions, n_visual, n_text = choose_positions(
            ids, anchor, visual_ids=visual_ids, special_ids=special_ids,
            budget=args.budget, seed=args.seed, sample_id=sample_id,
        )
        del raw
        if positions is None:
            skipped += 1
            print(f"[H1.1 matched] skip {ordinal}/{len(source_ids)}: {n_visual} visual, "
                  f"{n_text} eligible text; budget {args.budget}", flush=True)
            continue
        pair = {}
        for condition, dropped in positions.items():
            captured = _cached_forward(
                path, draft=draft, target=target, device=torch.device(args.draft_device),
                max_length=args.train_max_length, visual_token_ids=visual_ids,
                drop_positions=dropped,
            )
            if len(captured["weights"]) != len(draft.layers):
                raise RuntimeError("eager attention did not return weights for every draft layer")
            full = original_full[sample_id]
            if (captured["target_input_fingerprint"] != full["target_input_fingerprint"]
                    or captured["target_output_hash"] != full["target_output_hash"]
                    or captured["anchor"] != full["anchor"]):
                raise ValueError(f"cached target changed for sample {sample_id}")
            pair[condition] = captured["vectors"]
            reports.append({
                "sample_id": sample_id, "matched_condition": condition,
                "budget": args.budget, "dropped_positions": dropped,
                "visual_count": n_visual, "eligible_text_count": n_text,
                "context_length": anchor - args.budget,
                "target_input_fingerprint": captured["target_input_fingerprint"],
                "target_output_hash": captured["target_output_hash"],
                "matched_proposals": captured["accepted"],
                "proposal_count": captured["proposal_count"],
                "h11_attention": attention_summary(
                    captured["weights"], captured["visual_positions"], int(draft.block_size)
                ),
            })
        pairs.append(pair)
        sample_ids.append(sample_id)
        if len(pairs) == 1 or len(pairs) % 10 == 0:
            print(f"[H1.1 matched] {len(pairs)} eligible pairs / {ordinal} source samples", flush=True)
    if not pairs:
        raise ValueError(f"no eligible pairs at budget {args.budget}; retry with a smaller budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "matched_vectors.npz",
        visual=np.stack([row["visual_k"] for row in pairs]),
        text=np.stack([row["text_k"] for row in pairs]),
        sample_ids=np.asarray(sample_ids), checkpoint=original_checkpoint, budget=args.budget,
    )
    with (args.output_dir / "matched_reports.jsonl").open("w", encoding="utf-8") as handle:
        for row in reports:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[H1.1 matched] Saved {len(pairs)} paired samples; skipped {skipped} ineligible samples", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
