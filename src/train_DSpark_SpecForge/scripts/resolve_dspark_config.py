#!/usr/bin/env python3
"""Materialize a phase-specific Qwen2.5-VL DSpark draft configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from specforge.hidden_state import resolve_dflash_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="source")
    parser.add_argument("--output", required=True, dest="destination")
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument("--target-layer-ids", required=True)
    parser.add_argument("--draft-num-hidden-layers", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args(argv)

    source_path = Path(args.source)
    with source_path.open(encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("architectures") != ["DSparkDraftModel"]:
        raise ValueError("source config must declare DSparkDraftModel")
    method = source.get("dflash_config") or {}
    if method.get("projector_type") != "dspark":
        raise ValueError("source config must set dflash_config.projector_type=dspark")
    if method.get("position_encoding") != "qwen2_5_vl_mrope":
        raise ValueError("Qwen2.5-VL DSpark config must enable three-axis MRoPE")
    if args.block_size is not None and args.block_size < 2:
        raise ValueError("--block-size must be at least 2")

    payload = resolve_dflash_config(
        source_path,
        args.destination,
        target_layer_ids=args.target_layer_ids,
        draft_num_hidden_layers=args.draft_num_hidden_layers,
        phase=args.phase,
    )
    if args.block_size is not None:
        payload["block_size"] = args.block_size
        destination = Path(args.destination)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)

    print(
        json.dumps(
            {
                "phase": args.phase,
                "draft_config": args.destination,
                "draft_num_hidden_layers": payload["num_hidden_layers"],
                "block_size": payload["block_size"],
                "target_layer_ids": payload["dflash_config"]["target_layer_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
