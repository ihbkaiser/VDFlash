"""Configuration and provenance helpers for offline hidden-state features."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HIDDEN_STATE_METADATA_FILENAME = "hidden_state_metadata.json"
HIDDEN_STATE_SCHEMA_VERSION = 1


def parse_target_layer_ids(
    value: str | Sequence[int],
    *,
    expected_count: int | None = 5,
    num_target_layers: int | None = None,
) -> tuple[int, ...]:
    """Parse and validate ordered target decoder layer IDs.

    Qwen2.5-VL DFlash recipes use five IDs. ``expected_count=None`` is kept for
    shared metadata code that also serves other DFlash-family configurations.
    The logical IDs are zero-based and must leave the following decoder
    boundary available for the Qwen2.5-VL SGLang fallback.
    """

    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",")]
        if not value.strip() or any(not item for item in raw_values):
            raise ValueError("target_layer_ids must be comma-separated integers")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raise ValueError(
            "target_layer_ids must be a comma-separated string or a sequence"
        )

    layer_ids: list[int] = []
    for raw in raw_values:
        if isinstance(raw, bool):
            raise ValueError("target_layer_ids must contain integers")
        try:
            layer_id = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"target_layer_ids must contain integers, got {raw!r}"
            ) from exc
        if isinstance(raw, float) and raw != layer_id:
            raise ValueError(f"target_layer_ids must contain integers, got {raw!r}")
        layer_ids.append(layer_id)

    if expected_count is not None and len(layer_ids) != expected_count:
        raise ValueError(
            f"target_layer_ids must contain exactly {expected_count} IDs, "
            f"got {len(layer_ids)}"
        )
    if any(layer_id < 0 for layer_id in layer_ids):
        raise ValueError(
            f"target_layer_ids must be non-negative, got {layer_ids!r}"
        )
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"target_layer_ids must be unique, got {layer_ids!r}")
    if num_target_layers is not None:
        if num_target_layers < 1:
            raise ValueError(
                f"num_target_layers must be positive, got {num_target_layers!r}"
            )
        if any(layer_id + 1 >= num_target_layers for layer_id in layer_ids):
            raise ValueError(
                "target_layer_ids must be valid target decoder layers and leave "
                f"a following decoder boundary (depth={num_target_layers}), "
                f"got {layer_ids!r}"
            )
    return tuple(layer_ids)


def _draft_layer_ids(payload: dict[str, Any]) -> tuple[int, ...]:
    method_config = payload.get("dflash_config") or {}
    if not isinstance(method_config, dict):
        raise ValueError("draft config dflash_config must be an object")
    if "target_layer_ids" not in method_config:
        raise ValueError("draft config does not define target_layer_ids")
    return parse_target_layer_ids(
        method_config["target_layer_ids"],
        num_target_layers=payload.get("num_target_layers"),
    )


def resolve_dflash_config(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    target_layer_ids: str | Sequence[int] | None = None,
    draft_num_hidden_layers: int | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    """Materialize one phase's DFlash config without mutating the source file.

    ``draft_num_hidden_layers`` changes only the DFlash decoder depth.  It does
    not derive a new target-layer list: this is important for depth ablations
    where every draft must consume the same cached target hidden states.
    """

    source_path = Path(source).expanduser()
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"draft config {source_path} must contain an object")

    method_config = payload.get("dflash_config")
    if not isinstance(method_config, dict):
        raise ValueError("draft config dflash_config must be an object")
    num_target_layers = payload.get("num_target_layers")
    requested = (
        _draft_layer_ids(payload)
        if target_layer_ids is None
        else parse_target_layer_ids(
            target_layer_ids,
            num_target_layers=num_target_layers,
        )
    )
    resolved = dict(payload)
    resolved_method_config = dict(method_config)
    resolved_method_config["target_layer_ids"] = list(requested)
    resolved["dflash_config"] = resolved_method_config
    if draft_num_hidden_layers is not None:
        if (
            isinstance(draft_num_hidden_layers, bool)
            or not isinstance(draft_num_hidden_layers, int)
            or draft_num_hidden_layers <= 0
        ):
            raise ValueError(
                "draft_num_hidden_layers must be a positive integer, got "
                f"{draft_num_hidden_layers!r}"
            )
        layer_types = list(payload.get("layer_types") or [])
        if layer_types and len(set(layer_types)) > 1:
            raise ValueError(
                "draft_num_hidden_layers cannot resize a mixed layer_types "
                "layout; provide a homogeneous DFlash config"
            )
        layer_type = layer_types[0] if layer_types else "full_attention"
        resolved["num_hidden_layers"] = draft_num_hidden_layers
        resolved["layer_types"] = [layer_type] * draft_num_hidden_layers
    if phase is not None and (not isinstance(phase, str) or not phase.strip()):
        raise ValueError("phase must be a non-empty string when provided")

    destination_path = Path(destination).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(resolved, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def build_hidden_state_metadata(
    *,
    target_layer_ids: str | Sequence[int],
    hidden_size: int,
    phase: str | None = None,
    target_model: str | None = None,
    target_model_revision: str | None = None,
    capture_method: str = "dflash",
    dtype: str | None = None,
    layer_indexing: str = "target_decoder_zero_based",
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Build JSON-safe hidden-state provenance for a cache or checkpoint."""

    layer_ids = parse_target_layer_ids(
        target_layer_ids,
        expected_count=expected_count,
    )
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size <= 0
    ):
        raise ValueError(
            f"hidden_size must be a positive integer, got {hidden_size!r}"
        )
    return {
        "target_layer_ids": list(layer_ids),
        "layer_indexing": layer_indexing,
        "num_layers": len(layer_ids),
        "hidden_size": hidden_size,
        "feature_dim": len(layer_ids) * hidden_size,
        "phase": phase or "unspecified",
        "capture_method": capture_method,
        "target_model": target_model,
        "target_model_revision": target_model_revision,
        "dtype": dtype,
        "schema_version": HIDDEN_STATE_SCHEMA_VERSION,
    }


def _metadata_path(root: str | os.PathLike[str]) -> Path:
    return Path(root).expanduser() / HIDDEN_STATE_METADATA_FILENAME


def write_hidden_state_metadata(
    root: str | os.PathLike[str], metadata: dict[str, Any]
) -> Path:
    """Write cache metadata atomically and return its path."""

    path = _metadata_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_hidden_state_metadata(root: str | os.PathLike[str]) -> dict[str, Any]:
    path = _metadata_path(root)
    try:
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(
            f"hidden-state metadata is missing at {path}; rebuild the feature cache"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid hidden-state metadata at {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"hidden-state metadata at {path} must be an object")
    return metadata


def validate_hidden_state_metadata(
    root: str | os.PathLike[str],
    *,
    target_layer_ids: str | Sequence[int],
    hidden_size: int | None = None,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    """Fail closed when a feature cache does not match the requested run."""

    metadata = load_hidden_state_metadata(root)
    expected_ids = parse_target_layer_ids(target_layer_ids, expected_count=None)
    saved_ids = parse_target_layer_ids(
        metadata.get("target_layer_ids", []),
        expected_count=None,
    )
    if saved_ids != expected_ids:
        raise ValueError(
            "hidden-state metadata target_layer_ids mismatch: "
            f"saved={list(saved_ids)!r}, requested={list(expected_ids)!r}"
        )
    if metadata.get("num_layers") != len(saved_ids):
        raise ValueError(
            "hidden-state metadata num_layers does not match target_layer_ids"
        )
    if hidden_size is not None:
        if metadata.get("hidden_size") != hidden_size:
            raise ValueError(
                "hidden-state metadata hidden_size mismatch: "
                f"saved={metadata.get('hidden_size')!r}, requested={hidden_size!r}"
            )
        expected_feature_dim = len(saved_ids) * hidden_size
        if metadata.get("feature_dim") != expected_feature_dim:
            raise ValueError(
                "hidden-state metadata feature_dim mismatch: "
                f"saved={metadata.get('feature_dim')!r}, "
                f"requested={expected_feature_dim!r}"
            )
    if expected_phase is not None and metadata.get("phase") != expected_phase:
        raise ValueError(
            "hidden-state metadata phase mismatch: "
            f"saved={metadata.get('phase')!r}, requested={expected_phase!r}"
        )
    if metadata.get("schema_version") != HIDDEN_STATE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported hidden-state metadata schema_version: "
            f"{metadata.get('schema_version')!r}"
        )
    return metadata


__all__ = [
    "HIDDEN_STATE_METADATA_FILENAME",
    "HIDDEN_STATE_SCHEMA_VERSION",
    "build_hidden_state_metadata",
    "load_hidden_state_metadata",
    "parse_target_layer_ids",
    "resolve_dflash_config",
    "validate_hidden_state_metadata",
    "write_hidden_state_metadata",
]
