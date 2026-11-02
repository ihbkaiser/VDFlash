"""Shared row validation and metrics for the VisualCut report scripts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bootstrap_ci(values: list[Any], n_reps: int = 2000, seed: int = 42) -> dict[str, float | int]:
    """Return a percentile bootstrap interval for finite numeric values."""

    if n_reps <= 0:
        raise ValueError("n_reps must be positive")
    finite = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict) and "end_to_end" in value:
            value = value["end_to_end"]
        if isinstance(value, (int, float)) and math.isfinite(value):
            finite.append(float(value))
    if not finite:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0, "n": 0}
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if len(finite) == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "std": 0.0, "n": 1}
    rng = np.random.default_rng(seed)
    boot_means = [
        np.mean(rng.choice(finite, size=len(finite), replace=True))
        for _ in range(n_reps)
    ]
    return {
        "mean": mean,
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "std": std,
        "n": len(finite),
    }


def is_usable_case(row: dict[str, Any]) -> bool:
    """Exclude failed cases and capture/timing repeats that disagree."""

    return (
        row.get("status") in {"ok", "mismatch"}
        and row.get("capture_and_timing_outputs_match") is not False
    )


def extract_case_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten one case row into the metrics consumed by the report figures."""

    metrics = row.get("metrics") or {}
    timing = row.get("timing") or {}
    speculative_timing = timing.get("speculative") or {}
    target_timing = timing.get("target") or {}
    speedup = row.get("speedup")
    end_to_end_speedup = (
        speedup.get("end_to_end") if isinstance(speedup, dict) else speedup
    )
    speculative_tps = speculative_timing.get("tokens_per_second")
    target_tps = target_timing.get("tokens_per_second")
    decode_speedup = (
        speculative_tps / target_tps
        if speculative_tps is not None and target_tps is not None and target_tps > 0
        else None
    )
    quality = row.get("quality_metrics") or {}
    speculative_quality = quality.get("speculative") or {}
    target_quality = quality.get("target") or {}

    def delta(metric: str) -> float | None:
        speculative_value = speculative_quality.get(metric)
        target_value = target_quality.get(metric)
        if speculative_value is None or target_value is None:
            return None
        return float(speculative_value) - float(target_value)

    return {
        "tau_effective": metrics.get("tau_effective"),
        "tau_proposal": metrics.get("tau_proposal"),
        "speculative_tps": speculative_tps,
        "target_tps": target_tps,
        "speedup": end_to_end_speedup,
        "decode_speedup": decode_speedup,
        "prefill_s": speculative_timing.get("prefill_s"),
        "decode_s": speculative_timing.get("decode_s"),
        "draft_s": speculative_timing.get("draft_s"),
        "verify_s": speculative_timing.get("verify_s"),
        "lossless": bool(row.get("lossless")),
        "lossless_prefix_length": row.get("lossless_prefix_length"),
        "spec_bleu": speculative_quality.get("bleu"),
        "tgt_bleu": target_quality.get("bleu"),
        "bleu_delta": delta("bleu"),
        "spec_rouge_l": speculative_quality.get("rouge_l"),
        "tgt_rouge_l": target_quality.get("rouge_l"),
        "rouge_l_delta": delta("rouge_l"),
        "spec_coverage": speculative_quality.get("coverage"),
        "tgt_coverage": target_quality.get("coverage"),
        "coverage_delta": delta("coverage"),
        "spec_unigram_f1": speculative_quality.get("unigram_f1"),
        "tgt_unigram_f1": target_quality.get("unigram_f1"),
    }


def summarize_cases(
    all_data: dict[tuple[str, int, str, str], dict[str, Any]],
    config_names: tuple[str, ...] | list[str],
    budgets: tuple[int, ...],
    conditions: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build per-condition summaries and sample-paired contrasts."""

    metric_names = (
        "tau_effective",
        "tau_proposal",
        "speculative_tps",
        "target_tps",
        "speedup",
        "decode_speedup",
        "prefill_s",
        "decode_s",
        "draft_s",
        "verify_s",
        "spec_bleu",
        "tgt_bleu",
        "bleu_delta",
        "spec_rouge_l",
        "tgt_rouge_l",
        "rouge_l_delta",
        "spec_coverage",
        "tgt_coverage",
        "coverage_delta",
        "spec_unigram_f1",
        "tgt_unigram_f1",
        "lossless_prefix_length",
    )
    contrast_metrics = (
        "tau_effective",
        "tau_proposal",
        "speculative_tps",
        "speedup",
        "spec_bleu",
        "spec_rouge_l",
        "spec_coverage",
    )
    condition_stats: dict[str, Any] = {}
    paired_contrasts: dict[str, Any] = {}
    for config_name in config_names:
        condition_stats[config_name] = {}
        paired_contrasts[config_name] = {}
        for budget in budgets:
            for condition in conditions:
                rows = [
                    row
                    for (config, row_budget, row_condition, _sample_id), row in all_data.items()
                    if config == config_name
                    and row_budget == budget
                    and row_condition == condition
                ]
                stat = {
                    "sample_count": len(rows),
                    "lossless_count": sum(row["lossless"] for row in rows),
                    "lossless_rate": (
                        sum(row["lossless"] for row in rows) / len(rows) if rows else 0.0
                    ),
                }
                for metric in metric_names:
                    stat[metric] = bootstrap_ci(
                        [row[metric] for row in rows if row.get(metric) is not None]
                    )
                condition_stats[config_name][f"{budget}_{condition}"] = stat

            full_rows = {
                sample_id: row
                for (config, row_budget, condition, sample_id), row in all_data.items()
                if config == config_name and row_budget == budget and condition == "full"
            }
            for condition in conditions:
                if condition == "full":
                    continue
                candidate_rows = {
                    sample_id: row
                    for (config, row_budget, row_condition, sample_id), row in all_data.items()
                    if config == config_name
                    and row_budget == budget
                    and row_condition == condition
                }
                metric_differences = {}
                for metric in contrast_metrics:
                    differences = [
                        float(candidate_rows[sample_id][metric]) - float(full_row[metric])
                        for sample_id, full_row in full_rows.items()
                        if sample_id in candidate_rows
                        and candidate_rows[sample_id].get(metric) is not None
                        and full_row.get(metric) is not None
                    ]
                    metric_differences[metric] = bootstrap_ci(differences)
                paired_contrasts[config_name][
                    f"{budget}_{condition}_minus_full"
                ] = metric_differences
    return condition_stats, paired_contrasts


def sample_evenly(items: list[Any], max_items: int) -> list[Any]:
    """Select deterministic, approximately evenly spaced items."""

    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if len(items) <= max_items:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[index] for index in indices]


__all__ = [
    "bootstrap_ci",
    "extract_case_metrics",
    "is_usable_case",
    "sample_evenly",
    "summarize_cases",
]
