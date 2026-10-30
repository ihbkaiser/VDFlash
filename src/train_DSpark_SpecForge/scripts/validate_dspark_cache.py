#!/usr/bin/env python3
"""Check that an existing offline cache contains DSpark Qwen2.5-VL features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from specforge.hidden_state import validate_hidden_state_metadata
from specforge.runtime.data_plane.feature_store import load_feature_file
from specforge.runtime.data_plane.offline_reader import list_feature_files


REQUIRED_KEYS = (
    "input_ids",
    "loss_mask",
    "hidden_states",
    "target_last_hidden_states",
    "position_ids",
)


def _sequence_length(value: torch.Tensor, key: str) -> int:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a tensor, got {type(value)!r}")
    if key == "position_ids":
        if value.ndim == 2 and value.shape[0] == 3:
            return int(value.shape[-1])
        if value.ndim == 3 and value.shape[0] == 3 and value.shape[1] == 1:
            return int(value.shape[-1])
        raise ValueError(
            "position_ids must have shape [3, sequence] or [3, 1, sequence], "
            f"got {tuple(value.shape)}"
        )
    if key in {"input_ids", "loss_mask"}:
        if value.ndim == 1:
            return int(value.shape[0])
        if value.ndim == 2 and value.shape[0] == 1:
            return int(value.shape[1])
        raise ValueError(f"{key} must have shape [sequence] or [1, sequence]")
    if value.ndim == 2:
        return int(value.shape[0])
    if value.ndim == 3 and value.shape[0] == 1:
        return int(value.shape[1])
    raise ValueError(
        f"{key} must have shape [sequence, width] or [1, sequence, width], "
        f"got {tuple(value.shape)}"
    )


def _check_sample(path: str, *, hidden_size: int, target_layer_count: int) -> None:
    raw = load_feature_file(path)
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f"missing required tensors: {missing}")
    lengths = {key: _sequence_length(raw[key], key) for key in REQUIRED_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"sequence lengths differ: {lengths}")

    hidden = raw["hidden_states"]
    target = raw["target_last_hidden_states"]
    if int(hidden.shape[-1]) != hidden_size * target_layer_count:
        raise ValueError(
            "hidden_states width must equal hidden_size * selected target layers "
            f"({hidden_size} * {target_layer_count}), got {int(hidden.shape[-1])}"
        )
    if int(target.shape[-1]) != hidden_size:
        raise ValueError(
            f"target_last_hidden_states width must be {hidden_size}, "
            f"got {int(target.shape[-1])}"
        )


def _sample_indices(count: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= count:
        return list(range(count))
    if limit == 1:
        return [0]
    return sorted({round(index * (count - 1) / (limit - 1)) for index in range(limit)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--phase", required=True, choices=("phase1", "phase2"))
    parser.add_argument("--expected-samples", type=int, default=68000)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=32,
        help="feature files to inspect; use 0 to inspect every file",
    )
    args = parser.parse_args(argv)
    if args.expected_samples < -1:
        parser.error("--expected-samples must be -1 or non-negative")
    if args.sample_limit < 0:
        parser.error("--sample-limit must be non-negative")

    with open(args.draft_model_config, encoding="utf-8") as handle:
        draft = json.load(handle)
    method = draft.get("dflash_config") or {}
    layer_ids = method.get("target_layer_ids", [])
    files = list_feature_files(args.feature_root)
    if not files:
        raise FileNotFoundError(f"no .ckpt or .ckpt.gz feature files in {args.feature_root}")
    if args.expected_samples >= 0 and len(files) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} cached samples, found {len(files)} "
            f"under {args.feature_root}"
        )

    metadata_path = Path(args.feature_root) / "hidden_state_metadata.json"
    metadata_status = "missing"
    if metadata_path.is_file():
        metadata = validate_hidden_state_metadata(
            args.feature_root,
            target_layer_ids=layer_ids,
            hidden_size=int(draft["hidden_size"]),
            expected_phase=args.phase,
        )
        metadata_status = str(metadata.get("capture_method", "present"))

    indices = _sample_indices(len(files), args.sample_limit)
    failures: list[str] = []
    for index in indices:
        try:
            _check_sample(
                files[index],
                hidden_size=int(draft["hidden_size"]),
                target_layer_count=len(layer_ids),
            )
        except Exception as exc:  # report path context, then stop before training
            failures.append(f"{files[index]}: {type(exc).__name__}: {exc}")
            if len(failures) >= 10:
                break
    if failures:
        raise ValueError("DSpark cache validation failed:\n" + "\n".join(failures))

    print(
        json.dumps(
            {
                "feature_root": str(Path(args.feature_root).resolve()),
                "phase": args.phase,
                "cached_samples": len(files),
                "inspected_samples": len(indices),
                "target_layer_ids": layer_ids,
                "metadata_capture_method": metadata_status,
                "required_features": REQUIRED_KEYS,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
