#!/usr/bin/env python3
"""Validate a hidden-state cache against one phase's resolved DFlash config."""

from __future__ import annotations

import argparse
import json

from specforge.hidden_state import validate_hidden_state_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    args = parser.parse_args(argv)

    with open(args.draft_model_config, encoding="utf-8") as handle:
        draft_config = json.load(handle)
    method_config = draft_config.get("dflash_config") or {}
    metadata = validate_hidden_state_metadata(
        args.feature_root,
        target_layer_ids=method_config.get("target_layer_ids", []),
        hidden_size=int(draft_config["hidden_size"]),
        expected_phase=args.phase,
    )
    print(
        json.dumps(
            {
                "feature_root": args.feature_root,
                "phase": args.phase,
                "target_layer_ids": metadata["target_layer_ids"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
