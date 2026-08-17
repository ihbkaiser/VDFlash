#!/usr/bin/env python3
"""Tokenize a LLaVA manifest and validate feature-cache capacity."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tqdm.auto import tqdm

from specforge.qwen25vl import prepare_training_example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--expected-records", type=int, default=68000)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--allow-low-disk", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoConfig, AutoProcessor

    records = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.expected_records and len(records) != args.expected_records:
        raise ValueError(f"expected {args.expected_records} records, found {len(records)}")
    target_config = AutoConfig.from_pretrained(args.target_model_path)
    processor = AutoProcessor.from_pretrained(args.target_model_path)
    with Path(args.draft_model_config).open(encoding="utf-8") as handle:
        draft_config = json.load(handle)
    text_config = getattr(target_config, "text_config", target_config)
    hidden_size = int(getattr(text_config, "hidden_size"))
    feature_count = len(draft_config["dflash_config"]["target_layer_ids"])

    total_tokens = 0
    truncated = 0
    minimum_response = None
    progress = tqdm(
        records,
        desc="Preflight LLaVA",
        unit="sample",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for record in progress:
        prepared = prepare_training_example(
            processor,
            target_config,
            record,
            image_root=args.image_root,
            max_length=args.max_length,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        length = int(prepared["input_ids"].shape[-1])
        total_tokens += length
        truncated += int(bool(prepared["response_truncated"]))
        response_tokens = int(prepared["loss_mask"].sum().item())
        minimum_response = (
            response_tokens
            if minimum_response is None
            else min(minimum_response, response_tokens)
        )
        progress.set_postfix(truncated=truncated, refresh=False)

    raw_bytes = total_tokens * feature_count * hidden_size * 2
    required_bytes = int(raw_bytes * 1.2)
    output_parent = Path(args.output_path).expanduser().resolve().parent
    free_bytes = shutil.disk_usage(output_parent).free
    stats = {
        "records": len(records),
        "total_tokens": total_tokens,
        "truncated_responses": truncated,
        "minimum_response_tokens": minimum_response,
        "hidden_size": hidden_size,
        "target_feature_count": feature_count,
        "raw_hidden_state_bytes": raw_bytes,
        "required_free_bytes_with_margin": required_bytes,
        "free_bytes": free_bytes,
    }
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    if not args.allow_low_disk and free_bytes < required_bytes:
        raise OSError(
            f"feature cache requires about {required_bytes} bytes but only "
            f"{free_bytes} bytes are free under {output_parent}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
