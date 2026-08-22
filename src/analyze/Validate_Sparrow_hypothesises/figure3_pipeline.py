"""Process-isolated orchestration and reporting helpers for current Figure 3.

The measured panel implementations remain standalone modules so they can be
run independently.  This module owns the shared configuration, fail-closed
cross-panel validation, bundle metadata, and canonical Figure 3 rendering.
None of the helpers below imports Torch or loads a checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_TASKS = (
    "action_prediction",
    "action_sequence",
    "moving_attribute",
    "moving_direction",
    "object_interaction",
)
DEFAULT_PREPROCESSING = {
    "fps": 8.0,
    "max_frames": 8,
    "min_pixels": 256 * 28 * 28,
    "max_pixels": 360 * 420,
    "dtype": "bfloat16",
    "device_map": "auto",
    "quantized": False,
}


def build_panel_commands(
    *,
    python: str,
    model: str = DEFAULT_MODEL,
    manifest: str | Path,
    output_dir: str | Path,
    tasks: Sequence[str] = DEFAULT_TASKS,
    limit_per_task: int | None = None,
    fps: float = 8.0,
    max_frames: int = 8,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 360 * 420,
    max_new_tokens: int = 16,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    quantized: bool = False,
) -> dict[str, list[str]]:
    """Build complete commands for the two standalone current Figure 3 CLIs."""

    output = Path(output_dir)
    manifest_text = str(manifest)
    common = [
        "--model",
        str(model),
        "--manifest",
        manifest_text,
        "--tasks",
        *[str(task) for task in tasks],
        "--fps",
        str(float(fps)),
        "--max-frames",
        str(int(max_frames)),
        "--min-pixels",
        str(int(min_pixels)),
        "--max-pixels",
        str(int(max_pixels)),
        "--device-map",
        str(device_map),
        "--dtype",
        str(dtype),
    ]
    if limit_per_task is not None:
        common.extend(("--limit-per-task", str(int(limit_per_task))))
    if quantized:
        common.append("--quantized")

    command_a = [
        str(python),
        "-m",
        "src.analyze.figure3a_qwen25vl3b",
        *common,
        "--output",
        str(output / "figure3a.jsonl"),
        "--summary-output",
        str(output / "figure3a.summary.json"),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--plot-output",
        str(output / "figure3a.png"),
    ]
    command_b = [
        str(python),
        "-m",
        "src.analyze.figure3b_qwen25vl3b",
        *common,
        "--output",
        str(output / "figure3b.jsonl"),
        "--summary-output",
        str(output / "figure3b.summary.json"),
        "--plot-output",
        str(output / "figure3b.png"),
    ]
    return {"a": command_a, "b": command_b}


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _same_field(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    field: str,
    issues: list[dict[str, str]],
    *,
    code: str = "preprocessing_mismatch",
) -> None:
    if left.get(field) != right.get(field):
        issues.append(
            _issue(
                code,
                f"{field} differs between Figure 3(a) and Figure 3(b): "
                f"{left.get(field)!r} != {right.get(field)!r}",
            )
        )


def validate_figure3_summaries(
    a_summary: Mapping[str, Any],
    b_summary: Mapping[str, Any],
    *,
    expected_model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Validate that two panel summaries describe one complete experiment."""

    issues: list[dict[str, str]] = []
    if a_summary.get("paper_figure") != "Figure 3(a)":
        issues.append(_issue("panel_figure_mismatch", "Figure 3(a) summary has the wrong paper_figure"))
    if b_summary.get("paper_figure") != "Figure 3(b)":
        issues.append(_issue("panel_figure_mismatch", "Figure 3(b) summary has the wrong paper_figure"))

    for summary, label in ((a_summary, "Figure 3(a)"), (b_summary, "Figure 3(b)")):
        if summary.get("model") != expected_model:
            issues.append(
                _issue(
                    "model_mismatch",
                    f"{label} uses {summary.get('model')!r}; expected {expected_model!r}",
                )
            )
        if int(summary.get("num_error_records", 0) or 0) != 0:
            issues.append(_issue("panel_errors", f"{label} contains error records"))
        if float(summary.get("coverage", 0.0) or 0.0) < 1.0:
            issues.append(_issue("incomplete_coverage", f"{label} coverage is below 1.0"))

    if a_summary.get("manifest_sha256") != b_summary.get("manifest_sha256"):
        issues.append(
            _issue(
                "manifest_mismatch",
                "Figure 3(a) and Figure 3(b) do not use the same manifest hash",
            )
        )
    if not a_summary.get("manifest_sha256") or not b_summary.get("manifest_sha256"):
        issues.append(_issue("missing_manifest_hash", "both panel summaries must include manifest_sha256"))
    if a_summary.get("tasks") != b_summary.get("tasks"):
        issues.append(_issue("tasks_mismatch", "Figure 3 panels use different task lists"))

    for field in ("fps", "max_frames", "min_pixels", "max_pixels", "dtype", "device_map", "quantized"):
        _same_field(a_summary, b_summary, field, issues)

    a_layers = int(a_summary.get("layer_count", 0) or 0)
    b_layers = int(b_summary.get("layer_count", 0) or 0)
    if a_layers <= 0 or b_layers <= 0:
        issues.append(_issue("missing_layer_count", "both panels must report a positive layer_count"))
    elif a_layers != b_layers:
        issues.append(_issue("layer_count_mismatch", f"layer counts differ: {a_layers} != {b_layers}"))

    a_expected = int(a_summary.get("expected_rows_if_complete", 0) or 0)
    a_observed = int(a_summary.get("num_scored_rows", 0) or 0)
    if a_expected <= 0 or a_observed != a_expected:
        issues.append(
            _issue(
                "figure3a_incomplete",
                f"Figure 3(a) rows are incomplete: observed {a_observed}, expected {a_expected}",
            )
        )
    b_expected = int(b_summary.get("expected_rows_if_complete", 0) or 0)
    b_observed = int(b_summary.get("num_success_rows", 0) or 0)
    if b_expected <= 0 or b_observed != b_expected:
        issues.append(
            _issue(
                "figure3b_incomplete",
                f"Figure 3(b) rows are incomplete: observed {b_observed}, expected {b_expected}",
            )
        )
    expected_layers = list(range(b_layers)) if b_layers > 0 else []
    if list(b_summary.get("layer_indices", [])) != expected_layers:
        issues.append(
            _issue(
                "layer_coverage_mismatch",
                f"Figure 3(b) must cover zero-based layers {expected_layers}",
            )
        )

    shared = {
        "model": expected_model,
        "manifest": a_summary.get("manifest"),
        "manifest_sha256": a_summary.get("manifest_sha256"),
        "tasks": list(a_summary.get("tasks", [])),
        "layer_count": a_layers if a_layers == b_layers else None,
        "preprocessing": {field: a_summary.get(field) for field in DEFAULT_PREPROCESSING},
    }
    return {"valid": not issues, "issues": issues, "shared": shared}


def _finite(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result.append(numeric)
    return result


def _mean(values: Iterable[Any]) -> float | None:
    data = _finite(values)
    return sum(data) / len(data) if data else None


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [bool(row.get(field)) for row in rows if field in row]
    return sum(values) / len(values) if values else None


def build_figure3_statistics(
    a_rows: Iterable[Mapping[str, Any]],
    b_rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic task/cut and layer aggregates from current rows."""

    a_groups: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in a_rows:
        if row.get("condition") == "error" or "correct" not in row:
            continue
        cutoff = row.get("layer_cut")
        cutoff_value = int(cutoff) if cutoff is not None else None
        a_groups[(str(row.get("task", "unknown")), cutoff_value)].append(row)
    a_stats: list[dict[str, Any]] = []
    for (task, cutoff), group in sorted(a_groups.items(), key=lambda item: (item[0][0], 10**9 if item[0][1] is None else item[0][1])):
        a_stats.append(
            {
                "task": task,
                "layer_cut": cutoff,
                "num_samples": len(group),
                "accuracy": _mean(1.0 if row.get("correct") else 0.0 for row in group),
                "mean_prefix_agreement": _mean(row.get("prefix_agreement") for row in group),
                "lossless_rate": _rate(group, "lossless"),
            }
        )

    b_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in b_rows:
        if row.get("condition") == "error" or row.get("layer") is None:
            continue
        try:
            layer = int(row["layer"])
        except (TypeError, ValueError):
            continue
        b_groups[layer].append(row)
    b_stats: list[dict[str, Any]] = []
    for layer, group in sorted(b_groups.items()):
        masses = _finite(row.get("visual_mass") for row in group)
        mean_mass = sum(masses) / len(masses) if masses else None
        ci_low = ci_high = mean_mass
        if len(masses) > 1:
            margin = 1.96 * statistics.pstdev(masses) / math.sqrt(len(masses))
            ci_low, ci_high = mean_mass - margin, mean_mass + margin
        head_arrays = [
            _finite(row.get("per_head_visual_mass", []))
            for row in group
            if isinstance(row.get("per_head_visual_mass"), list)
        ]
        head_count = max((len(values) for values in head_arrays), default=0)
        mean_heads = [
            _mean(values[index] for values in head_arrays if index < len(values))
            for index in range(head_count)
        ]
        b_stats.append(
            {
                "layer": layer,
                "num_samples": len(group),
                "mean_visual_mass": mean_mass,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "head_count": head_count,
                "mean_per_head_visual_mass": mean_heads,
            }
        )
    return {"figure3a": a_stats, "figure3b": b_stats}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    if not columns:
        columns = ["num_samples"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "paper_figure",
        "model",
        "model_source",
        "manifest",
        "manifest_sha256",
        "tasks",
        "num_records",
        "num_scored_rows",
        "num_success_rows",
        "num_error_records",
        "expected_rows_if_complete",
        "coverage",
        "layer_count",
        "layer_cut_points",
        "layer_indices",
        "layer_index_convention",
        "attention_query",
        "metric",
        *DEFAULT_PREPROCESSING,
    )
    return {field: summary.get(field) for field in fields if field in summary}


def write_figure3_metadata(
    output_dir: str | Path,
    a_summary: Mapping[str, Any],
    b_summary: Mapping[str, Any],
    a_rows: Iterable[Mapping[str, Any]],
    b_rows: Iterable[Mapping[str, Any]],
    *,
    audit: Mapping[str, Any] | None = None,
    expected_model: str = DEFAULT_MODEL,
) -> list[str]:
    """Write current Figure 3 CSV, JSON metadata, and fail-closed audit."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    a_rows_list = [dict(row) for row in a_rows]
    b_rows_list = [dict(row) for row in b_rows]
    statistics_by_panel = build_figure3_statistics(a_rows_list, b_rows_list)
    validation = validate_figure3_summaries(
        a_summary,
        b_summary,
        expected_model=expected_model,
    )
    if audit is None:
        audit_payload: dict[str, Any] = validation
    else:
        audit_payload = dict(audit)
        audit_payload.setdefault("summary_validation", validation)
        audit_payload["valid"] = bool(audit_payload.get("valid", False)) and validation["valid"]
    _write_csv(output / "figure3a_statistics.csv", statistics_by_panel["figure3a"])
    _write_csv(output / "figure3b_statistics.csv", statistics_by_panel["figure3b"])
    (output / "figure3_statistics.json").write_text(
        json.dumps(statistics_by_panel, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "figure3_audit.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_files = {
        name: {"path": str(output / name), "sha256": _sha256(output / name)}
        for name in (
            "figure3a.jsonl",
            "figure3a.summary.json",
            "figure3a.png",
            "figure3b.jsonl",
            "figure3b.summary.json",
            "figure3b.png",
            "figure3_insight_layer_analysis.png",
            "figure3_insight_layer_analysis.pdf",
            "figure3_insight_layer_analysis.svg",
        )
    }
    metadata = {
        "bundle_version": 1,
        "paper_figure": "Figure 3",
        "model": a_summary.get("model"),
        "manifest": a_summary.get("manifest"),
        "manifest_sha256": a_summary.get("manifest_sha256"),
        "tasks": list(a_summary.get("tasks", [])),
        "preprocessing": {field: a_summary.get(field) for field in DEFAULT_PREPROCESSING},
        "layer_count": a_summary.get("layer_count"),
        "layer_cut_points": a_summary.get("layer_cut_points"),
        "layer_index_convention": b_summary.get("layer_index_convention"),
        "panels": {
            "figure3a": _compact_summary(a_summary),
            "figure3b": _compact_summary(b_summary),
        },
        "statistics": statistics_by_panel,
        "audit": audit_payload,
        "source_files": source_files,
    }
    (output / "figure3_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return [
        "figure3a_statistics.csv",
        "figure3b_statistics.csv",
        "figure3_statistics.json",
        "figure3_audit.json",
        "figure3_metadata.json",
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_figure3_bundle(output_dir: str | Path) -> dict[str, Any]:
    """Load a composed current Figure 3 bundle from disk."""

    output = Path(output_dir)
    required = (
        "figure3a.jsonl",
        "figure3a.summary.json",
        "figure3b.jsonl",
        "figure3b.summary.json",
        "figure3_metadata.json",
        "figure3_audit.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"current Figure 3 bundle is missing: {', '.join(missing)}")
    return {
        "output_dir": str(output.resolve()),
        "a_rows": _read_jsonl(output / "figure3a.jsonl"),
        "b_rows": _read_jsonl(output / "figure3b.jsonl"),
        "a_summary": json.loads((output / "figure3a.summary.json").read_text(encoding="utf-8")),
        "b_summary": json.loads((output / "figure3b.summary.json").read_text(encoding="utf-8")),
        "metadata": json.loads((output / "figure3_metadata.json").read_text(encoding="utf-8")),
        "audit": json.loads((output / "figure3_audit.json").read_text(encoding="utf-8")),
    }


def compose_figure3_plot(
    output_dir: str | Path,
    a_summary: Mapping[str, Any],
    b_summary: Mapping[str, Any],
    *,
    formats: Sequence[str] = ("png", "pdf", "svg"),
) -> list[str]:
    """Render a compact current Figure 3(a)/(b) composite."""

    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    axis_a, axis_b = axes
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for point in a_summary.get("summary", []):
        if point.get("layer_cut") is not None:
            grouped[str(point.get("task", "task"))].append(point)
    for task, points in sorted(grouped.items()):
        points = sorted(points, key=lambda point: int(point["layer_cut"]))
        axis_a.plot(
            [int(point["layer_cut"]) for point in points],
            [float(point.get("accuracy", 0.0)) for point in points],
            marker="o",
            linewidth=1.7,
            markersize=4,
            label=task,
        )
        baselines = [
            point
            for point in a_summary.get("summary", [])
            if point.get("task") == task and point.get("layer_cut") is None
        ]
        if baselines:
            axis_a.axhline(float(baselines[0].get("accuracy", 0.0)), linestyle="--", alpha=0.3)
    axis_a.set_title("(a) Visual KV ablation")
    axis_a.set_xlabel("Visual KV masked from decoder layer x")
    axis_a.set_ylabel("MVBench accuracy")
    axis_a.set_ylim(0.0, 1.0)
    axis_a.grid(alpha=0.25)
    if grouped:
        axis_a.legend(fontsize=7, loc="best")

    aggregate = b_summary.get("aggregate") or {}
    layer_stats = aggregate.get("layer_stats") or []
    if not layer_stats:
        axis_b.text(0.5, 0.5, "No complete Figure 3(b) rows", ha="center", va="center")
        axis_b.set_axis_off()
    else:
        layers = [int(point["layer"]) for point in layer_stats]
        means = [float(point.get("mean_visual_mass", 0.0)) for point in layer_stats]
        lows = [float(point.get("ci95_low", mean)) for point, mean in zip(layer_stats, means)]
        highs = [float(point.get("ci95_high", mean)) for point, mean in zip(layer_stats, means)]
        axis_b.plot(layers, means, color="#2c7fb8", marker="o", linewidth=1.6, markersize=3.5)
        axis_b.fill_between(layers, lows, highs, color="#2c7fb8", alpha=0.18)
        axis_b.set_title("(b) All-layer visual attention")
        axis_b.set_xlabel("Native decoder layer (zero-based)")
        axis_b.set_ylabel("Mean visual attention mass")
        axis_b.set_xlim(min(layers), max(layers))
        axis_b.grid(alpha=0.25)

    figure.suptitle(f"Sparrow Figure 3: {a_summary.get('model', DEFAULT_MODEL)}", fontsize=13)
    files: list[str] = []
    for file_format in formats:
        filename = f"figure3_insight_layer_analysis.{file_format}"
        figure.savefig(output / filename, dpi=180 if file_format == "png" else None)
        files.append(filename)
    plt.close(figure)
    return files


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TASKS",
    "build_figure3_statistics",
    "build_panel_commands",
    "compose_figure3_plot",
    "load_figure3_bundle",
    "validate_figure3_summaries",
    "write_figure3_metadata",
]
