#!/usr/bin/env python3
"""Materialize a phase-specific DFlash draft config."""

from __future__ import annotations

import argparse
import json

from specforge.hidden_state import resolve_dflash_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="source")
    parser.add_argument("--output", required=True, dest="destination")
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument(
        "--target-layer-ids",
        default=None,
        help="ordered comma-separated target decoder IDs; defaults to the source config",
    )
    args = parser.parse_args(argv)
    payload = resolve_dflash_config(
        args.source,
        args.destination,
        target_layer_ids=args.target_layer_ids,
        phase=args.phase,
    )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "draft_config": args.destination,
                "target_layer_ids": payload["dflash_config"]["target_layer_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
