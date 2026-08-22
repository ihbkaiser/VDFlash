"""Qwen2.5-VL target-side Figure 3/3(b)/6 diagnostic builders."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from typing import Any

from .dflash_contract import (
    DFlashExperiment,
    DFlashSemanticStatus,
)

_TARGET_DIAGNOSTIC_NAMES = {
    item.value
    for item in (
        DFlashExperiment.TARGET_VISUAL_KV,
        DFlashExperiment.TARGET_ATTENTION,
        DFlashExperiment.TARGET_HIDDEN_COSINE,
    )
}


@contextmanager
def eager_target_attention(model: Any):
    """Force inspectable target attention and restore the caller's backend."""

    candidates = [
        getattr(model, "config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ]
    configs = []
    seen = set()
    for config in candidates:
        if config is None or id(config) in seen or not hasattr(config, "_attn_implementation"):
            continue
        seen.add(id(config))
        configs.append(config)
    previous = [getattr(config, "_attn_implementation") for config in configs]
    try:
        for config in configs:
            config._attn_implementation = "eager"
        yield
    finally:
        for config, value in zip(configs, previous):
            config._attn_implementation = value


def run_qwen25vl_layer_diagnostics(
    samples: Iterable[Mapping[str, Any]],
    probe: Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    *,
    metadata: Mapping[str, Any],
    limit: int | None = None,
    row_sink: Callable[[Mapping[str, Any]], None] | None = None,
    skip: Callable[[Mapping[str, Any]], bool] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Build target-side diagnostic rows with no speculative output claims."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        if skip is not None and skip(sample):
            continue
        base = dict(metadata)
        base.update(
            {
                "semantic_status": DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value,
                "sample_id": str(sample.get("sample_id") or sample.get("id") or index),
                "input_fingerprint": str(sample.get("input_fingerprint", "")),
                "target_output_ids": None,
                "speculative_output_ids": None,
            }
        )
        try:
            diagnostics = list(probe(sample))
            if not diagnostics:
                diagnostics = [{
                    "experiment": DFlashExperiment.TARGET_HIDDEN_COSINE.value,
                    "status": "unsupported",
                    "error": "no target-side diagnostic rows produced",
                }]
        except Exception as exc:  # pragma: no cover - exercised by GPU runs
            diagnostics = [{
                "experiment": DFlashExperiment.TARGET_HIDDEN_COSINE.value,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }]
        finally:
            if cleanup is not None:
                cleanup()
        for diagnostic in diagnostics:
            experiment = str(diagnostic.get("experiment", ""))
            row = dict(base)
            row.update(
                {
                    "experiment": experiment,
                    "metrics": dict(diagnostic.get("metrics", {})),
                }
            )
            row.update(diagnostic)
            if experiment not in _TARGET_DIAGNOSTIC_NAMES:
                row["status"] = "error"
                row["error"] = f"unsupported target diagnostic experiment: {experiment}"
            row["semantic_status"] = DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value
            row["target_output_ids"] = None
            row["speculative_output_ids"] = None
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    return rows


__all__ = ["eager_target_attention", "run_qwen25vl_layer_diagnostics"]
