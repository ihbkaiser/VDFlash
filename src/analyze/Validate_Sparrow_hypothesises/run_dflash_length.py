"""Figure 1(a)/(b) evidence builders for the DFlash validation matrix."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .dflash_contract import (
    DFLASH_LENGTH_TARGETS,
    DFLASH_RETENTION_PERCENTAGES,
    DFlashExperiment,
    DFlashSemanticStatus,
)
from .dflash_runtime import build_visual_retention_mask


def _sample_id(sample: Mapping[str, Any], index: int) -> str:
    return str(sample.get("sample_id") or sample.get("id") or sample.get("video_name") or index)


def _run_condition(
    decode: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    sample: Mapping[str, Any],
    condition: Mapping[str, Any],
    *,
    cleanup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    try:
        return dict(decode(sample, condition))
    except Exception as exc:  # pragma: no cover - exercised by GPU runs
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if cleanup is not None:
            cleanup()


def _row(
    *,
    metadata: Mapping[str, Any],
    sample: Mapping[str, Any],
    index: int,
    result: Mapping[str, Any],
    condition: Mapping[str, Any],
    experiment: DFlashExperiment,
    semantic_status: DFlashSemanticStatus,
) -> dict[str, Any]:
    row = dict(metadata)
    row.update(
        {
            "experiment": experiment.value,
            "semantic_status": semantic_status.value,
            "sample_id": _sample_id(sample, index),
            "input_fingerprint": str(
                result.get("input_fingerprint", sample.get("input_fingerprint", ""))
            ),
            "metrics": dict(result.get("metrics", {})),
        }
    )
    row.update(condition)
    row.update(result)
    return row


def run_length_sweep(
    samples: Iterable[Mapping[str, Any]],
    decode: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
    length_targets: Iterable[int] = DFLASH_LENGTH_TARGETS,
    limit: int | None = None,
    row_sink: Callable[[Mapping[str, Any]], None] | None = None,
    skip: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Run/build one direct length-sweep row per sample and target length."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        for length_target in length_targets:
            condition = {
                "length_target": int(length_target),
                "target_visual_tokens": int(length_target),
            }
            if skip is not None and skip(sample, condition):
                continue
            row = _row(
                metadata=metadata,
                sample=sample,
                index=index,
                result=_run_condition(decode, sample, condition, cleanup=cleanup),
                condition=condition,
                experiment=DFlashExperiment.LENGTH_SWEEP,
                semantic_status=DFlashSemanticStatus.DIRECT,
            )
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    return rows


def run_hidden_visual_retention(
    samples: Iterable[Mapping[str, Any]],
    decode: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
    retention_percentages: Iterable[int] = DFLASH_RETENTION_PERCENTAGES,
    limit: int | None = None,
    row_sink: Callable[[Mapping[str, Any]], None] | None = None,
    skip: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Run/build adapted hidden-context visual-retention rows."""

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if limit is not None and index >= limit:
            break
        total_length = int(sample["context_length"])
        visual_positions = sample.get("visual_positions", ())
        full_fingerprint = str(sample.get("full_target_input_fingerprint", ""))
        for percentage in retention_percentages:
            mask = build_visual_retention_mask(
                total_length=total_length,
                visual_positions=visual_positions,
                retention_percentage=int(percentage),
            )
            condition = {
                "retention_percentage": int(percentage),
                "target_visual_tokens": len(visual_positions),
                "hidden_context_mask": mask.tolist(),
                "full_target_input_fingerprint": full_fingerprint,
            }
            if skip is not None and skip(sample, condition):
                continue
            row = _row(
                metadata=metadata,
                sample=sample,
                index=index,
                result=_run_condition(decode, sample, condition, cleanup=cleanup),
                condition=condition,
                experiment=DFlashExperiment.TARGET_HIDDEN_VISUAL_RETENTION,
                semantic_status=DFlashSemanticStatus.ADAPTED,
            )
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    return rows


__all__ = ["run_hidden_visual_retention", "run_length_sweep"]
