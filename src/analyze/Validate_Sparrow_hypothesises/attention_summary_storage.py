"""Compact per-forward attention records without changing reported metrics."""

from __future__ import annotations

from typing import Any, Mapping


SUMMARY_RECORD_SCHEMA = "vdflash_attention_summary_exact_v1"


def compact_attention_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Replace a raw attention matrix with the analyzer's exact aggregates.

    The full attention tensor and key map are consumed immediately by the same
    summarizer used for raw shard analysis. The resulting record preserves the
    per-forward outputs and aggregate sums used by ``case_attention_summary``.
    """

    # Lazy import avoids coupling the raw writer to the analyzer at module load
    # time; both paths still share one implementation of the metric formula.
    from .analyze_dflash_visualcut_attention import _summarize_attention_record

    row = {
        "row_id": record["row_id"],
        "checkpoint_name": record["checkpoint_name"],
        "sample_id": record["sample_id"],
        "target_visual_tokens": record["visual_token_target"],
        "visual_condition": record["visual_condition"],
        "prompt_regions_original": record["prompt_regions_original"],
    }
    summary, aggregate = _summarize_attention_record(row, dict(record))
    return {
        "schema": SUMMARY_RECORD_SCHEMA,
        "case_id": str(record["case_id"]),
        "row_id": str(record["row_id"]),
        "layer_index": int(record["layer_index"]),
        "round_index": int(record["round_index"]),
        "summary": summary,
        "aggregate": aggregate,
    }
