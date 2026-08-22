"""Audit helpers for DFlash evidence; intentionally independent from MSD."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .dflash_contract import validate_dflash_row


def audit_dflash_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_grid: Mapping[str, Iterable[int]] | None = None,
) -> dict[str, Any]:
    """Validate DFlash rows and summarize semantic coverage/losslessness."""

    invalid_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    valid_rows = 0
    lossless_rows = 0
    semantic_status_counts: dict[str, int] = defaultdict(int)
    experiment_counts: dict[str, int] = defaultdict(int)
    retention_fingerprints: dict[str, set[str]] = defaultdict(set)
    valid_materialized: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        errors = validate_dflash_row(row)
        if errors:
            invalid_rows.append(
                {"index": index, "sample_id": row.get("sample_id"), "errors": errors}
            )
            continue
        if row.get("status") in {"error", "unsupported"}:
            error_rows.append(
                {
                    "index": index,
                    "sample_id": row.get("sample_id"),
                    "experiment": row.get("experiment"),
                    "error": row.get("error", row.get("status")),
                }
            )
            continue

        valid_rows += 1
        valid_materialized.append(dict(row))
        status = str(row["semantic_status"])
        experiment = str(row["experiment"])
        semantic_status_counts[status] += 1
        experiment_counts[experiment] += 1
        if row.get("target_output_ids") is not None and row.get("speculative_output_ids") is not None:
            if row["target_output_ids"] == row["speculative_output_ids"]:
                lossless_rows += 1
        if experiment == "target_hidden_visual_retention":
            retention_fingerprints[str(row["sample_id"])].add(
                str(row.get("full_target_input_fingerprint", ""))
            )

    retention_fingerprint_errors = sorted(
        sample_id for sample_id, fingerprints in retention_fingerprints.items() if len(fingerprints) > 1
    )
    coverage_gaps: dict[str, list[int]] = {}
    grid_bindings = {
        "length_targets": ("length_sweep", "length_target"),
        "retention_percentages": (
            "target_hidden_visual_retention",
            "retention_percentage",
        ),
        "attention_targets": ("dflash_context_attention", "target_visual_tokens"),
        "layer_cuts": ("qwen25vl_target_visual_kv", "layer_index"),
    }
    for key, expected in (expected_grid or {}).items():
        binding = grid_bindings.get(key)
        if binding is None:
            continue
        experiment, field = binding
        observed = {
            int(row[field])
            for row in valid_materialized
            if row.get("experiment") == experiment and row.get(field) is not None
        }
        missing = sorted({int(value) for value in expected} - observed)
        if missing:
            coverage_gaps[key] = missing
    return {
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "error_rows": error_rows,
        "lossless_rows": lossless_rows,
        "semantic_status_counts": dict(semantic_status_counts),
        "experiment_counts": dict(experiment_counts),
        "retention_fingerprint_errors": retention_fingerprint_errors,
        "coverage_gaps": coverage_gaps,
        "coverage_valid": (
            not invalid_rows
            and not error_rows
            and not retention_fingerprint_errors
            and not coverage_gaps
        ),
    }


__all__ = ["audit_dflash_rows"]
