"""Figure 2 DFlash context-attention evidence builders."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .dflash_contract import DFlashExperiment, DFlashSemanticStatus


def run_dflash_context_attention(
    samples: Iterable[Mapping[str, Any]],
    probe: Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    *,
    metadata: Mapping[str, Any],
    limit: int | None = None,
    row_sink: Callable[[Mapping[str, Any]], None] | None = None,
    skip: Callable[[Mapping[str, Any]], bool] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Build one adapted row per captured DFlash layer/query summary."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        if skip is not None and skip(sample):
            continue
        base = dict(metadata)
        base.update(
            {
                "experiment": DFlashExperiment.CONTEXT_ATTENTION.value,
                "semantic_status": DFlashSemanticStatus.ADAPTED.value,
                "sample_id": str(sample.get("sample_id") or sample.get("id") or index),
                "input_fingerprint": str(sample.get("input_fingerprint", "")),
                "query_policy": "draft_block",
            }
        )
        try:
            summaries = list(probe(sample))
        except Exception as exc:  # pragma: no cover - exercised by GPU runs
            summaries = [{"status": "error", "error": f"{type(exc).__name__}: {exc}"}]
        finally:
            if cleanup is not None:
                cleanup()
        if not summaries:
            summaries = [{"status": "unsupported", "error": "no attention weights captured"}]
        for summary in summaries:
            row = dict(base)
            row.update(
                {
                    "attention_source": "dflash_context",
                    "metrics": dict(summary.get("metrics", {})),
                }
            )
            row.update(summary)
            row.setdefault("query_policy", "draft_block")
            row.setdefault("attention_source", "dflash_context")
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    return rows


__all__ = ["run_dflash_context_attention"]
