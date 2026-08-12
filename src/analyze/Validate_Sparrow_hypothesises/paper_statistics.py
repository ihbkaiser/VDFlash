"""Paper-shaped descriptive statistics for the Sparrow insight figures.

The paper reports averages, but a local reproduction should also expose N and
uncertainty.  The functions here keep the raw measured rows untouched and
produce deterministic bootstrap intervals over completed sample rows.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def _finite(values: Iterable[Any]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_ci(values: Sequence[float], replicates: int = 2000, seed: int = 42) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, replicates)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def summarize(values: Iterable[Any], *, replicates: int = 2000, seed: int = 42) -> dict[str, Any]:
    clean = _finite(values)
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p25": None,
            "p75": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    low, high = _bootstrap_ci(clean, replicates=replicates, seed=seed)
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "p25": _quantile(clean, 0.25),
        "p75": _quantile(clean, 0.75),
        "ci95_low": low,
        "ci95_high": high,
    }


def _group_value(row: dict[str, Any], primary: str, fallback: str | None = None) -> Any:
    value = row.get(primary)
    if value is None and fallback:
        value = row.get(fallback)
    return value


def _group_rows(rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        grouped[key].append(row)
    return grouped


def _metric_row(
    key: tuple[Any, ...],
    fields: Sequence[str],
    group: Sequence[dict[str, Any]],
    metrics: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result = {field: value for field, value in zip(fields, key)}
    result["rows"] = len(group)
    result["sample_count"] = len({row.get("sample_id") for row in group if row.get("sample_id") is not None})
    for offset, metric in enumerate(metrics):
        values = [row.get(metric) for row in group]
        result[metric] = summarize(values, replicates=replicates, seed=seed + offset)
    return result


def _attention_summary_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("paper_figure") == "Figure 2" and row.get("modality") == "summary"
    ]


def build_paper_statistics(
    rows: Iterable[dict[str, Any]],
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Return grouped statistics for every figure/table-shaped output."""

    rows = list(rows)
    f1a = [row for row in rows if row.get("paper_figure") == "Figure 1(a)"]
    f1b = [row for row in rows if row.get("paper_figure") == "Figure 1(b)"]
    f2 = _attention_summary_rows(rows)
    f3a = [row for row in rows if row.get("paper_figure") == "Figure 3"]
    f3b = [row for row in rows if row.get("paper_figure") == "Figure 3(b)"]
    f6 = [row for row in rows if row.get("paper_figure") == "Figure 6 / Appendix D"]

    f1a_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in f1a:
        target = _group_value(row, "calibration_target_visual_tokens", "actual_visual_tokens")
        f1a_groups[(target,)].append(row)
    f1a_stats = [
        _metric_row(
            (target,),
            ["visual_tokens"],
            group,
            [
                "accepted_prefix_tokens",
                "prefill_seconds",
                "decode_seconds",
                "end_to_end_seconds",
                "end_to_end_speedup",
                "lossless",
            ],
            replicates=replicates,
            seed=seed,
        )
        for (target,), group in sorted(f1a_groups.items(), key=lambda item: (float(item[0][0]) if item[0][0] is not None else float("inf"),))
    ]
    for row in f1a_stats:
        row["actual_visual_tokens"] = summarize(
            [item.get("actual_visual_tokens") for item in f1a_groups[(row["visual_tokens"],)]],
            replicates=replicates,
            seed=seed + 100,
        )

    f1b_stats = [
        _metric_row(
            key,
            ["retention_percentage", "selection_policy"],
            group,
            ["accepted_prefix_tokens", "prefill_seconds", "decode_seconds", "end_to_end_seconds", "lossless", "end_to_end_speedup"],
            replicates=replicates,
            seed=seed,
        )
        for key, group in sorted(
            _group_rows(f1b, ["retention_percentage", "selection_policy"]).items(),
            key=lambda item: (str(item[0][1]), -float(item[0][0] or 0)),
        )
    ]
    f2_stats = [
        _metric_row(
            key,
            ["visual_tokens", "attention_policy"],
            group,
            ["visual_mass", "text_mass", "instruction_mass", "visual_entropy"],
            replicates=replicates,
            seed=seed,
        )
        for key, group in sorted(
            _group_rows(f2, ["target_visual_tokens", "attention_policy"]).items(),
            key=lambda item: (float(item[0][0] or 0), str(item[0][1])),
        )
    ]
    f3a_stats = [
        _metric_row(key, ["layer_cut"], group, ["rouge_l", "prefix_agreement", "lossless"], replicates=replicates, seed=seed)
        for key, group in sorted(_group_rows(f3a, ["layer_cut"]).items(), key=lambda item: float(item[0][0] or 0))
    ]
    f3b_stats = [
        _metric_row(key, ["layer"], group, ["visual_mass"], replicates=replicates, seed=seed)
        for key, group in sorted(_group_rows(f3b, ["layer"]).items(), key=lambda item: float(item[0][0] or 0))
    ]
    f6_stats = [
        _metric_row(key, ["layer"], group, ["visual_cosine", "text_cosine"], replicates=replicates, seed=seed)
        for key, group in sorted(_group_rows(f6, ["layer"]).items(), key=lambda item: float(item[0][0] or 0))
    ]
    return {
        "figure1a": f1a_stats,
        "figure1b": f1b_stats,
        "figure2": f2_stats,
        "figure3a": f3a_stats,
        "figure3b": f3b_stats,
        "figure6": f6_stats,
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            result.update(_flatten(child, child_prefix))
        else:
            result[child_prefix] = child
    return result


def write_statistics(
    statistics_by_figure: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
) -> list[str]:
    """Write JSON plus one flat CSV per figure for paper/table use."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "paper_statistics.json"
    json_path.write_text(json.dumps(statistics_by_figure, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    files = [json_path.name]
    for figure, entries in statistics_by_figure.items():
        flattened = [_flatten(entry) for entry in entries]
        columns = sorted({key for entry in flattened for key in entry})
        path = output / f"{figure}_statistics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({column: entry.get(column) for column in columns} for entry in flattened)
        files.append(path.name)
    return files
