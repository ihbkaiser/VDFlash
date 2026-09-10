#!/usr/bin/env python3
"""Add provenance metadata to a legacy DFlash hidden-state cache.

This migration does not rewrite feature rows. It checks one representative
row's required keys and hidden-state width, then writes the metadata consumed
by the current fail-closed cache validator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from specforge.hidden_state import (
    build_hidden_state_metadata,
    write_hidden_state_metadata,
)
from specforge.runtime.data_plane.feature_store import load_feature_file


def _first_feature_file(root: Path) -> str | None:
    """Find one deterministic feature row without walking the whole cache."""

    if root.is_file() and root.name.endswith((".ckpt", ".ckpt.gz")):
        return str(root)
    if not root.is_dir():
        return None
    for directory, directories, filenames in os.walk(root):
        directories.sort()
        for name in sorted(filenames):
            if name.endswith((".ckpt", ".ckpt.gz")):
                return str(Path(directory) / name)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument("--target-model", default=None)
    args = parser.parse_args(argv)

    feature_root = Path(args.feature_root).expanduser()
    first_file = _first_feature_file(feature_root)
    if first_file is None:
        raise ValueError(f"no .ckpt or .ckpt.gz files found under {feature_root}")

    with Path(args.draft_model_config).expanduser().open(encoding="utf-8") as handle:
        draft_config = json.load(handle)
    method_config = draft_config.get("dflash_config") or {}
    layer_ids = method_config.get("target_layer_ids")
    if not isinstance(layer_ids, list):
        raise ValueError("draft config must define dflash_config.target_layer_ids")
    hidden_size = draft_config.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"draft config hidden_size must be positive, got {hidden_size!r}")

    raw = load_feature_file(first_file)
    required = {"input_ids", "loss_mask", "hidden_states"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"legacy feature row {first_file} is missing keys: {missing}")
    hidden_states = raw["hidden_states"]
    if hidden_states.ndim not in (2, 3):
        raise ValueError(
            f"legacy hidden_states must have 2 or 3 dimensions, got {tuple(hidden_states.shape)}"
        )
    feature_width = int(hidden_states.shape[-1])
    expected_width = len(layer_ids) * hidden_size
    if feature_width != expected_width:
        raise ValueError(
            "legacy hidden-state width mismatch: "
            f"row={feature_width}, expected={expected_width} "
            f"({len(layer_ids)} target layers x hidden_size {hidden_size})"
        )
    if args.phase == "phase2" and "position_ids" not in raw:
        raise ValueError(f"Phase 2 feature row {first_file} is missing position_ids")

    dtype = draft_config.get("dtype")
    metadata = build_hidden_state_metadata(
        target_layer_ids=layer_ids,
        hidden_size=hidden_size,
        phase=args.phase,
        target_model=args.target_model,
        dtype=str(dtype) if dtype is not None else None,
        layer_indexing="qwen25vl_hf_decoder_zero_based",
        expected_count=5,
    )
    output = write_hidden_state_metadata(feature_root, metadata)
    print(
        json.dumps(
            {
                "feature_root": str(feature_root),
                "metadata": str(output),
                "checked_row": first_file,
                "feature_width": feature_width,
                "target_layer_ids": layer_ids,
                "phase": args.phase,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
