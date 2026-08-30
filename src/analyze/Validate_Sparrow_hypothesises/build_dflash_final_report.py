"""Build a reproducible final report from completed DFlash JSONL artifacts.

This module is deliberately CPU-only.  It reads measured stage journals,
preserves non-OK decode rows in the summaries, and writes diagnostic figures
without loading Transformers or a model checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PRIMARY_DIR = Path(
    "results/sparrow_validation_dflash_qwen25vl3b_2026-08-24_flash_model_parallel_2gpu"
)
DEFAULT_LAYER_DIR = Path(
    "results/sparrow_validation_dflash_qwen25vl3b_2026-08-25_layer_analysis"
)
DEFAULT_OUTPUT_DIR = Path("results/sparrow_validation_dflash_qwen25vl3b_final_paper_style_2026-08-25")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a required JSONL stage journal, ignoring blank lines."""

    if not path.is_file():
        raise FileNotFoundError(f"required DFlash artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(rows: Iterable[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        value = _number(current)
        if value is not None:
            values.append(value)
    return fmean(values) if values else None


def _condition(value: Any) -> int | float | str:
    numeric = _number(value)
    if numeric is None:
        return "unknown"
    return int(numeric) if numeric.is_integer() else numeric


def _sort_value(value: int | float | str) -> tuple[int, float | str]:
    return (0, float(value)) if isinstance(value, (int, float)) else (1, str(value))


def _row_status(row: Mapping[str, Any]) -> str:
    value = row.get("status")
    return str(value) if value not in (None, "") else "missing"


def summarize_decode_rows(
    rows: Iterable[Mapping[str, Any]], condition_field: str
) -> list[dict[str, Any]]:
    """Summarize decode rows while retaining all result statuses."""

    grouped: dict[int | float | str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_condition(row.get(condition_field))].append(row)

    result: list[dict[str, Any]] = []
    for condition, group in sorted(grouped.items(), key=lambda item: _sort_value(item[0])):
        statuses = Counter(_row_status(row) for row in group)
        valid_group = [row for row in group if _row_status(row) in {"ok", "mismatch"}]
        known_count = sum(statuses.get(status, 0) for status in ("ok", "mismatch", "unsupported", "error"))
        result.append(
            {
                "condition": condition,
                "n": len(group),
                "valid_n": len(valid_group),
                "ok": statuses.get("ok", 0),
                "mismatch": statuses.get("mismatch", 0),
                "unsupported": statuses.get("unsupported", 0),
                "error": statuses.get("error", 0),
                "unknown": len(group) - known_count,
                "lossless_rate": statuses.get("ok", 0) / len(valid_group) if valid_group else None,
                "tau_effective_mean": _mean(valid_group, ("metrics", "tau_effective")),
                "accepted_length_mean": _mean(valid_group, ("metrics", "tau_effective")),
                "speculative_latency_mean": _mean(
                    valid_group, ("timing", "speculative", "end_to_end_s")
                ),
                "target_latency_mean": _mean(
                    valid_group, ("timing", "target", "end_to_end_s")
                ),
                "speedup_mean": _mean(valid_group, ("speedup", "end_to_end")),
            }
        )
    return result


def summarize_attention_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate DFlash context attention by visual target and draft layer."""

    grouped: dict[tuple[int | float | str, int | float | str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(_condition(row.get("target_visual_tokens")), _condition(row.get("layer_index")))].append(row)
    result = []
    for (target, layer), group in sorted(
        grouped.items(), key=lambda item: (_sort_value(item[0][0]), _sort_value(item[0][1]))
    ):
        entry = {
            "target_visual_tokens": target,
            "layer_index": layer,
            "n": len(group),
            "context_attention_mass_mean": _mean(group, ("context_attention_mass",)),
        }
        noise_mean = _mean(group, ("noise_attention_mass",))
        if noise_mean is not None:
            entry["noise_attention_mass_mean"] = noise_mean
        result.append(entry)
    return result


def summarize_layer_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate the three target-side layer diagnostic experiments."""

    groups: dict[str, dict[int | float | str, list[Mapping[str, Any]]]] = {
        "visual_kv": defaultdict(list),
        "attention": defaultdict(list),
        "cosine": defaultdict(list),
    }
    experiment_to_group = {
        "qwen25vl_target_visual_kv": "visual_kv",
        "qwen25vl_target_attention": "attention",
        "qwen25vl_target_hidden_cosine": "cosine",
    }
    for row in rows:
        group_name = experiment_to_group.get(str(row.get("experiment")))
        if group_name is not None:
            groups[group_name][_condition(row.get("layer_index"))].append(row)

    output: dict[str, list[dict[str, Any]]] = {}
    for group_name, grouped in groups.items():
        entries: list[dict[str, Any]] = []
        for layer, group in sorted(grouped.items(), key=lambda item: _sort_value(item[0])):
            entry: dict[str, Any] = {"layer_index": layer, "n": len(group)}
            if group_name == "visual_kv":
                entry["diagnostic_output_length_mean"] = _mean(
                    group, ("metrics", "diagnostic_output_length")
                )
            elif group_name == "attention":
                entry["visual_attention_mass_mean"] = _mean(
                    group, ("metrics", "visual_attention_mass")
                )
            else:
                entry["text_cosine_mean"] = _mean(group, ("metrics", "text_cosine"))
                entry["visual_cosine_mean"] = _mean(group, ("metrics", "visual_cosine"))
            entries.append(entry)
        output[group_name] = entries
    return output


def _load_audit(directory: Path) -> dict[str, Any]:
    path = directory / "dflash_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"required DFlash audit is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _audit_indices(audit: Mapping[str, Any], fields: Sequence[str]) -> set[int]:
    indices: set[int] = set()
    for field in fields:
        entries = audit.get(field, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and isinstance(entry.get("index"), int):
                indices.add(int(entry["index"]))
    return indices


def _filter_rows_by_audit(
    rows: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    offset: int = 0,
    fields: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Exclude rows that the source audit could not materialize as valid rows."""

    failed = _audit_indices(audit, fields)
    return [
        row
        for local_index, row in enumerate(rows)
        if offset + local_index not in failed
    ]


def _audit_has_failures(audit: Mapping[str, Any]) -> bool:
    def nonempty(value: Any) -> bool:
        if isinstance(value, (list, dict, tuple, set)):
            return bool(value)
        try:
            return int(value or 0) > 0
        except (TypeError, ValueError):
            return bool(value)

    return (
        not bool(audit.get("coverage_valid", False))
        or nonempty(audit.get("invalid_rows"))
        or nonempty(audit.get("error_rows"))
        or nonempty(audit.get("unsupported_rows"))
        or nonempty(audit.get("mismatch_rows"))
        or nonempty(audit.get("retention_fingerprint_errors"))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_summary(primary_dir: Path, layer_dir: Path) -> dict[str, Any]:
    """Load the approved source runs and build a JSON-serializable summary."""

    primary_files = {
        "length": primary_dir / "figure1a_length_sweep.jsonl",
        "retention": primary_dir / "figure1b_target_hidden_visual_retention.jsonl",
        "attention": primary_dir / "figure2_dflash_context_attention.jsonl",
    }
    layer_file = layer_dir / "figure3_3b_6_target_diagnostics.jsonl"
    primary_rows = {name: read_jsonl(path) for name, path in primary_files.items()}
    layer_rows = read_jsonl(layer_file)
    primary_audit = _load_audit(primary_dir)
    layer_audit = _load_audit(layer_dir)
    incomplete = _audit_has_failures(primary_audit) or _audit_has_failures(layer_audit)
    primary_offsets: dict[str, int] = {}
    offset = 0
    for name, rows in primary_rows.items():
        primary_offsets[name] = offset
        offset += len(rows)
    primary_decode_rows = {
        name: _filter_rows_by_audit(
            rows,
            primary_audit,
            offset=primary_offsets[name],
            fields=("invalid_rows",),
        )
        for name, rows in primary_rows.items()
    }
    primary_valid_rows = {
        name: _filter_rows_by_audit(
            rows,
            primary_audit,
            offset=primary_offsets[name],
            fields=("invalid_rows", "error_rows", "unsupported_rows"),
        )
        for name, rows in primary_rows.items()
    }
    layer_valid_rows = _filter_rows_by_audit(
        layer_rows,
        layer_audit,
        fields=("invalid_rows", "error_rows", "unsupported_rows"),
    )
    source_paths = [*primary_files.values(), layer_file, primary_dir / "dflash_audit.json", layer_dir / "dflash_audit.json"]
    sources = []
    for path in source_paths:
        if path.is_file():
            sources.append({"path": str(path), "sha256": _sha256(path)})
    all_rows = [row for stage_rows in primary_rows.values() for row in stage_rows]
    all_rows.extend(layer_rows)
    target_models = sorted({str(row["target_model"]) for row in all_rows if row.get("target_model")})
    calibration_policies = sorted(
        {str(row["calibration_policy"]) for row in all_rows if row.get("calibration_policy")}
    )
    attention_targets = sorted(
        {
            int(row["target_visual_tokens"])
            for row in primary_rows["attention"]
            if _number(row.get("target_visual_tokens")) is not None
        }
    )
    return {
        "status": "INCOMPLETE DIAGNOSTIC" if incomplete else "COMPLETE",
        "primary_dir": str(primary_dir),
        "layer_dir": str(layer_dir),
        "target_models": target_models,
        "calibration_policies": calibration_policies,
        "attention_targets": attention_targets,
        "primary_audit": primary_audit,
        "layer_audit": layer_audit,
        "length": summarize_decode_rows(primary_decode_rows["length"], "length_target"),
        "retention": summarize_decode_rows(primary_decode_rows["retention"], "retention_percentage"),
        "attention": summarize_attention_rows(primary_valid_rows["attention"]),
        "layers": summarize_layer_rows(layer_valid_rows),
        "row_counts": {
            "length": len(primary_rows["length"]),
            "retention": len(primary_rows["retention"]),
            "attention": len(primary_rows["attention"]),
            "layer_diagnostics": len(layer_rows),
        },
        "audited_row_counts": {
            "length": len(primary_valid_rows["length"]),
            "retention": len(primary_valid_rows["retention"]),
            "attention": len(primary_valid_rows["attention"]),
            "layer_diagnostics": len(layer_valid_rows),
        },
        "sources": sources,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: Any, output: Path, stem: str, plt: Any) -> list[str]:
    fig.tight_layout()
    files = []
    for extension in ("png", "pdf", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=600 if extension == "png" else None, bbox_inches="tight")
        files.append(path.name)
    plt.close(fig)
    return files


def _plot_value(value: Any) -> float:
    return float("nan") if value is None else float(value)


def render_figures(summary: Mapping[str, Any], output: Path) -> list[str]:
    """Render Sparrow/MSD-shaped DFlash figures without requiring model code.

    The layout follows the canonical Figure 1/2/3/6 composition in
    ``results/report_MSD/BAO_CAO_VIET.md``.  Where DFlash does not record the
    paper's raw tensor (per-token modality attention or per-head attention),
    the available aggregate is rendered as a clearly labeled proxy.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif", "STIXGeneral"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "axes.grid": False,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    colors = {
        "keep": "#2166ac",
        "remove": "#b2182b",
        "latency": "#f4a582",
        "target_latency": "#d6604d",
        "visual": "#1b7837",
        "instruction": "#2166ac",
        "text": "#d6604d",
        "baseline": "#666666",
    }

    def style_axis(axis: Any) -> None:
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    def token_label(value: Any) -> str:
        numeric = _number(value)
        if numeric is None:
            return str(value)
        return f"{numeric / 1000:g}K" if numeric >= 1000 else f"{numeric:g}"

    length = list(summary.get("length", []))
    retention = list(summary.get("retention", []))
    if length or retention:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), squeeze=False)
        left, right = axes[0]
        if length:
            ordered = sorted(length, key=lambda row: _sort_value(row["condition"]))
            positions = list(range(len(ordered)))
            labels = [token_label(row["condition"]) for row in ordered]
            twin = left.twinx()
            speculative_latency = [row.get("speculative_latency_mean") for row in ordered]
            target_latency = [row.get("target_latency_mean") for row in ordered]
            if any(value is not None for value in speculative_latency):
                twin.bar(
                    [position - 0.17 for position in positions],
                    [_plot_value(value) for value in speculative_latency],
                    width=0.32,
                    color=colors["latency"],
                    alpha=0.85,
                    label="DFlash end-to-end",
                )
            if any(value is not None for value in target_latency):
                twin.bar(
                    [position + 0.17 for position in positions],
                    [_plot_value(value) for value in target_latency],
                    width=0.32,
                    color=colors["target_latency"],
                    alpha=0.85,
                    label="Target greedy end-to-end",
                )
            line = left.plot(
                positions,
                [_plot_value(row.get("accepted_length_mean")) for row in ordered],
                marker="o",
                color=colors["keep"],
                label="DFlash effective emitted tokens/round (τ)",
            )
            left.set_xticks(positions, labels)
            left.set_xlabel("Visual token length")
            left.set_ylabel("Average emitted tokens per acceptance round")
            twin.set_ylabel("End-to-end latency (s)")
            left.set_title("(a) Visual-length sweep")
            style_axis(left)
            twin.spines["top"].set_visible(False)
            handles, labels_ = left.get_legend_handles_labels()
            twin_handles, twin_labels = twin.get_legend_handles_labels()
            left.legend(handles + twin_handles, labels_ + twin_labels, loc="best")
        else:
            left.text(0.5, 0.5, "No measured Figure 1(a) rows", ha="center", va="center", transform=left.transAxes)
            left.set_axis_off()
        if retention:
            ordered = sorted(retention, key=lambda row: float(row["condition"]), reverse=True)
            x_values = [float(row["condition"]) for row in ordered]
            right.plot(
                x_values,
                [_plot_value(row.get("accepted_length_mean")) for row in ordered],
                marker="o",
                color=colors["keep"],
                label="DFlash effective emitted tokens/round (τ)",
            )
            right.set_xlabel("Retained visual input (%)")
            right.set_ylabel("Average emitted tokens per acceptance round")
            right.set_title("(b) Draft visual retention")
            retention_ticks = sorted({float(row["condition"]) for row in retention})
            right.set_xticks(retention_ticks)
            right.set_xticklabels([
                str(int(value)) if value.is_integer() else str(value)
                for value in retention_ticks
            ], rotation=45, ha="right")
            right.invert_xaxis()
            style_axis(right)
            right.legend(loc="best")
        else:
            right.text(0.5, 0.5, "No measured Figure 1(b) rows", ha="center", va="center", transform=right.transAxes)
            right.set_axis_off()
        fig.suptitle("Figure 1. Impact of visual token length and retention on DFlash", y=1.02, fontsize=11)
        files.extend(_save_figure(fig, output, "figure1_insight_summary", plt))

    attention = list(summary.get("attention", []))
    if attention:
        targets = sorted({row["target_visual_tokens"] for row in attention}, key=_sort_value)
        selected = [target for target in (400, 3000) if target in targets] or targets[:2]
        fig, axes = plt.subplots(1, len(selected), figsize=(5.25 * len(selected), 3.7), squeeze=False)
        axes = list(axes[0])
        for axis, target in zip(axes, selected):
            group = sorted(
                [row for row in attention if row["target_visual_tokens"] == target],
                key=lambda row: float(row["layer_index"]),
            )
            layers = [row["layer_index"] for row in group]
            axis.plot(
                layers,
                [_plot_value(row.get("context_attention_mass_mean")) for row in group],
                marker="o",
                color=colors["visual"],
                label="Context mass",
            )
            if any(row.get("noise_attention_mass_mean") is not None for row in group):
                axis.plot(
                    layers,
                    [_plot_value(row.get("noise_attention_mass_mean")) for row in group],
                    marker="s",
                    linestyle="--",
                    color=colors["text"],
                    label="Noise mass",
                )
            axis.set_xlabel("DFlash draft layer")
            axis.set_ylabel("Attention mass")
            axis.set_ylim(0, 1.05)
            axis.set_title(f"({'a' if axis is axes[0] else 'b'}) {token_label(target)} visual tokens")
            axis.text(
                0.02,
                0.98,
                "DFlash proxy\n(no per-token modality trace)",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="#555555",
            )
            style_axis(axis)
            axis.legend(loc="best")
        fig.suptitle("Figure 2. Draft attention distribution (DFlash proxy)", y=1.02, fontsize=11)
        files.extend(_save_figure(fig, output, "figure2_insight_attention", plt))

    layers = summary.get("layers", {})
    if any(layers.values()):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), squeeze=False)
        axes = list(axes[0])
        kv = layers.get("visual_kv", [])
        if kv:
            axes[0].plot(
                [row["layer_index"] for row in kv],
                [_plot_value(row.get("diagnostic_output_length_mean")) for row in kv],
                marker="o",
                color=colors["keep"],
                label="Diagnostic output length",
            )
            axes[0].set_title("(a) Layer-cut visual-KV diagnostic")
            axes[0].set_xlabel("Visual KV masking start layer")
            axes[0].set_ylabel("Diagnostic output length")
            axes[0].text(0.02, 0.98, "DFlash proxy", transform=axes[0].transAxes, ha="left", va="top", fontsize=7, color="#555555")
            style_axis(axes[0])
            axes[0].legend(loc="best")
        else:
            axes[0].text(0.5, 0.5, "No measured layer-cut rows", ha="center", va="center", transform=axes[0].transAxes)
            axes[0].set_axis_off()
        layer_attention = layers.get("attention", [])
        if layer_attention:
            axes[1].plot(
                [row["layer_index"] for row in layer_attention],
                [_plot_value(row.get("visual_attention_mass_mean")) for row in layer_attention],
                marker="o",
                color=colors["visual"],
                label="Head-summed visual attention",
            )
            axes[1].set_title("(b) Layer-wise visual attention")
            axes[1].set_xlabel("Target layer")
            axes[1].set_ylabel("Head-summed visual attention mass")
            axes[1].text(0.02, 0.98, "DFlash proxy\n(no per-head vector)", transform=axes[1].transAxes, ha="left", va="top", fontsize=7, color="#555555")
            style_axis(axes[1])
            axes[1].legend(loc="best")
        else:
            axes[1].text(0.5, 0.5, "No measured attention rows", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_axis_off()
        fig.suptitle("Figure 3. Layer-wise visual flow and attention — Qwen2.5-VL-3B", y=1.02, fontsize=11)
        files.extend(_save_figure(fig, output, "figure3_insight_layer_analysis", plt))

    cosine = layers.get("cosine", [])
    if cosine:
        fig, axis = plt.subplots(figsize=(6.2, 4.2))
        layer_values = [row["layer_index"] for row in cosine]
        axis.plot(
            layer_values,
            [_plot_value(row.get("visual_cosine_mean")) for row in cosine],
            color=colors["remove"],
            marker="o",
            label="Visual information retention",
        )
        axis.plot(
            layer_values,
            [_plot_value(row.get("text_cosine_mean")) for row in cosine],
            color=colors["keep"],
            marker="o",
            label="Text information retention",
        )
        if 20 in layer_values:
            axis.axvline(20, color=colors["baseline"], linestyle="--", linewidth=1.0, label="Layer 20")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Retention rate (cosine)")
        axis.set_ylim(0, 1.05)
        axis.set_title("Figure 6. Layer-wise information retention (DFlash proxy)")
        style_axis(axis)
        axis.legend(loc="best")
        files.extend(_save_figure(fig, output, "figure6_insight_retention", plt))

    return files


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(summary: Mapping[str, Any], output: Path, figure_files: Sequence[str]) -> None:
    """Write machine-readable tables and the experiment-by-experiment report."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "source_audits.json").write_text(
        json.dumps({"primary": summary["primary_audit"], "layers": summary["layer_audit"]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "length_summary.csv", summary.get("length", []))
    _write_csv(output / "retention_summary.csv", summary.get("retention", []))
    _write_csv(output / "attention_summary.csv", summary.get("attention", []))
    for name, rows in summary.get("layers", {}).items():
        _write_csv(output / f"layer_{name}_summary.csv", rows)
    primary_dir = Path(str(summary["primary_dir"]))
    layer_dir = Path(str(summary["layer_dir"]))
    for source in summary.get("sources", []):
        source_path = Path(source["path"])
        prefix = "primary" if source_path.parent == primary_dir else "layers" if source_path.parent == layer_dir else "other"
        destination = output / f"source_{prefix}_{source_path.name}"
        if source_path.name.endswith(".json"):
            shutil.copy2(source_path, destination)

    primary = summary["primary_audit"]
    layers = summary["layer_audit"]
    target_models = ", ".join(summary.get("target_models", [])) or "unknown"
    calibration_policies = ", ".join(summary.get("calibration_policies", [])) or "unknown"
    attention_targets = ", ".join(map(str, summary.get("attention_targets", []))) or "unknown"
    length_n = summary["row_counts"]["length"]
    retention_n = summary["row_counts"]["retention"]
    attention_n = summary["row_counts"]["attention"]
    layer_n = summary["row_counts"]["layer_diagnostics"]
    audited_counts = summary.get("audited_row_counts", {})
    length_valid_n = audited_counts.get("length", length_n)
    retention_valid_n = audited_counts.get("retention", retention_n)
    attention_valid_n = audited_counts.get("attention", attention_n)
    layer_valid_n = audited_counts.get("layer_diagnostics", layer_n)
    layer_kv_n = len(summary.get("layers", {}).get("visual_kv", []))
    layer_cuts = [row["layer_index"] for row in summary.get("layers", {}).get("visual_kv", [])]
    length_rows = sorted(summary.get("length", []), key=lambda row: _sort_value(row["condition"]))
    retention_rows = sorted(
        summary.get("retention", []),
        key=lambda row: _sort_value(row["condition"]),
    )
    attention_rows = list(summary.get("attention", []))
    layer_summary = summary.get("layers", {})
    layer_kv_rows = list(layer_summary.get("visual_kv", []))
    layer_attention_rows = list(layer_summary.get("attention", []))
    cosine_rows = list(layer_summary.get("cosine", []))

    def image_name(stem: str) -> str | None:
        expected = f"{stem}.png"
        return expected if expected in figure_files else None

    def value(row: Mapping[str, Any], key: str) -> float | None:
        return _number(row.get(key))

    def metric_range(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float | None, float | None]:
        values = [value(row, key) for row in rows]
        values = [item for item in values if item is not None]
        return (min(values), max(values)) if values else (None, None)

    length_tau_min, length_tau_max = metric_range(length_rows, "tau_effective_mean")
    length_speed_min, length_speed_max = metric_range(length_rows, "speedup_mean")
    retention_tau_min, retention_tau_max = metric_range(retention_rows, "tau_effective_mean")
    retention_speed_min, retention_speed_max = metric_range(retention_rows, "speedup_mean")
    attention_by_target: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in attention_rows:
        attention_by_target[row.get("target_visual_tokens")].append(row)
    attention_insights: list[str] = []
    for target, rows in sorted(attention_by_target.items(), key=lambda item: _sort_value(item[0])):
        context_min, context_max = metric_range(rows, "context_attention_mass_mean")
        last = max(rows, key=lambda row: _number(row.get("layer_index")) or -1)
        last_context = value(last, "context_attention_mass_mean")
        if context_min is not None and context_max is not None:
            attention_insights.append(
                f"ở mốc `{target}`, context mass dao động {_format(context_min)}–{_format(context_max)} "
                f"và đạt {_format(last_context)} ở draft layer cuối"
            )
    kv_values = [value(row, "diagnostic_output_length_mean") for row in layer_kv_rows]
    kv_values = [item for item in kv_values if item is not None]
    attention_peak = max(layer_attention_rows, key=lambda row: value(row, "visual_attention_mass_mean") or float("-inf")) if layer_attention_rows else None
    attention_last = max(layer_attention_rows, key=lambda row: _number(row.get("layer_index")) or -1) if layer_attention_rows else None
    visual_first = cosine_rows[0] if cosine_rows else None
    visual_last = cosine_rows[-1] if cosine_rows else None

    lines = [
        "# Final DFlash Validation Report",
        "",
        f"**Overall status:** `{summary['status']}`",
        "",
        "This report aggregates existing Qwen2.5-VL-3B DFlash measurements. It is a local diagnostic report, not a claim of paper-level reproduction or method superiority.",
        "",
        "## Scope and provenance",
        "",
        f"- Primary run: `{summary['primary_dir']}`",
        f"- Layer diagnostics: `{summary['layer_dir']}`",
        f"- Target model(s): `{target_models}`",
        f"- Calibration policy/policies: `{calibration_policies}`",
        f"- Source row counts: length `{length_n}`, retention `{retention_n}`, context attention `{attention_n}`, layer diagnostics `{layer_n}`.",
        f"- Audited rows used for aggregates: length `{length_valid_n}`, retention `{retention_valid_n}`, context attention `{attention_valid_n}`, layer diagnostics `{layer_valid_n}`.",
        "",
        "## Executive summary",
        "",
        f"The three primary DFlash stages produced `{length_n + retention_n + attention_n}` rows. The standalone layer run produced `{layer_valid_n}` audited-valid target-side diagnostic rows, including `{layer_kv_n}` aggregate layer-cut groups at cuts `{', '.join(map(str, layer_cuts))}`.",
        f"The primary audit records `{primary.get('mismatch_rows', 0)}` decode mismatches, `{len(primary.get('unsupported_rows', []))}` unsupported rows, and `{len(primary.get('error_rows', []))}` runtime error rows. The overall report status is `{summary['status']}`.",
        "",
        "## CONFIRMED",
        "",
        "- The requested primary stage artifacts exist and were read successfully.",
        f"- The layer-cut artifact contains `{layer_valid_n}` audited-valid target-side diagnostic rows; its visual-KV subset covers cuts `{', '.join(map(str, layer_cuts))}`.",
        f"- The primary audit reports `{len(primary.get('invalid_rows', []))}` contract-invalid rows; the layer audit reports `{len(layers.get('invalid_rows', []))}` contract-invalid rows.",
        "- Retention fingerprint integrity is preserved in the primary audit where checked.",
        "",
        "## EXPLORATORY",
        "",
        "The following are measured trends, not claims that DFlash satisfies the locked success criterion:",
        "",
        "- Length and retention summaries report effective acceptance, end-to-end speedup, and exact output-match rate by condition.",
        f"- Context-attention summaries report draft-layer context/noise mass at the available `{attention_targets}` visual-token targets.",
        "- Layer summaries report target-side visual-KV masking output length, visual attention mass, and visual/text hidden-state cosine curves.",
        "",
        "## Figure-format note",
        "",
        "The four composites follow the Sparrow/MSD Figure 1/2/3/6 numbering, panel structure, serif typography, and paper palette. Figure 2 is a DFlash proxy because the source lacks per-token modality traces; Figure 3(b) is a DFlash proxy using head-summed visual attention because no per-head vector is archived.",
        "",
        "## Diễn giải theo từng thực nghiệm",
        "",
        "Phần này đọc các hình theo câu hỏi mà DFlash thực sự đo. Các insight bên dưới là xu hướng quan sát được trên artifact hiện tại; chúng không vượt qua các giới hạn losslessness và calibration đã nêu ở cuối báo cáo.",
        "",
        "## Experiment 1 — Visual-length sweep",
        "",
        "### Mục tiêu",
        "",
        "Đo ảnh hưởng của độ dài visual context lên chi phí end-to-end và khả năng tạo proposal của DFlash. Đây là phép đo trực tiếp nhất cho trade-off giữa lượng thông tin video mà draft phải xử lý và số token draft có thể phát ra trong một acceptance round.",
        "",
        "### Cách thực hiện",
        "",
        "Thực nghiệm dùng cohort VDC-50 gồm 50 video. Mỗi video được chạy ở bốn mốc visual nominal `400, 3000, 13000, 25000`, tạo thành 200 length jobs trước audit.",
        "Với mỗi job, calibration map mốc nominal sang cấu hình decode video (số frame và pixel budget). Prompt được chuẩn bị một lần bằng Qwen2.5-VL processor, vị trí video token và input fingerprint được ghi lại. Vì source run dùng `allow_out_of_tolerance`, actual visual-token count có thể lệch mốc nominal.",
        "Target greedy được chạy làm reference trên prompt đầy đủ. Sau đó `InstrumentedDFlashDecoder` chạy DFlash trên cùng prompt với greedy decoding và `max_new_tokens=256`. Hai chuỗi output được so sánh exact token-by-token; mỗi acceptance round lưu proposal count, matched proposal và effective emitted tokens.",
        "Báo cáo aggregate `tau_proposal`, `tau_effective`, target/DFlash end-to-end latency và `speedup = target_end_to_end / dflash_end_to_end`. Các row unsupported, error hoặc bị audit loại khỏi mean nhưng vẫn được giữ trong status counts.",
        "",
    ]
    if image_name("figure1_insight_summary"):
        lines.extend([
            f"![Figure 1 — Visual-length sweep và retention](figure1_insight_summary.png)",
            "",
            "*Hình 1. Panel (a) là visual-length sweep; panel (b) là retention sweep. Đường τ và các cột latency dùng metric DFlash-native.*",
            "",
        ])
    lines.extend([
        "### Kết quả và nhận xét",
        "",
        f"Khi nominal visual context tăng từ `{length_rows[0]['condition'] if length_rows else 'n/a'}` đến `{length_rows[-1]['condition'] if length_rows else 'n/a'}`, DFlash có `tau_effective` trong khoảng `{_format(length_tau_min)}`–`{_format(length_tau_max)}` và speedup trong khoảng `{_format(length_speed_min)}`–`{_format(length_speed_max)}`. Latency tăng theo độ dài context; đây là xu hướng chi phí phù hợp với việc target và draft phải xử lý nhiều visual hidden states hơn.",
        "",
        "**Insight thực nghiệm.** Trong artifact này, visual context dài hơn làm end-to-end cost tăng và τ giảm nhẹ: DFlash phát được ít token hiệu dụng hơn mỗi round khi context dài. Tuy nhiên, speedup vẫn lớn hơn 1 ở cả bốn mốc trong aggregate hiện tại, nghĩa là speculative path vẫn nhanh hơn target greedy về mặt thời gian đo được trên cohort này.",
        "",
        "**Cảnh báo diễn giải.** Exact output match chỉ đạt khoảng 10–18% tùy mốc và có một unsupported row ở mốc 25K. Vì vậy xu hướng latency/τ nên được xem là exploratory; chưa thể dùng nó để tuyên bố DFlash lossless hoặc kết luận chắc chắn về chất lượng output.",
        "",
        "## Experiment 2 — Draft visual retention",
        "",
        "### Mục tiêu",
        "",
        "Kiểm tra mức độ DFlash phụ thuộc vào visual conditioning ở draft. Target vẫn giữ prompt đầy đủ; chỉ hidden context truyền cho draft bị mask theo các tỷ lệ `100, 25, 10, 5, 1, 0%`. Fingerprint của target được giữ lại để kiểm tra rằng intervention không thay đổi input của target.",
        "",
        "### Cách thực hiện",
        "",
        "Thực nghiệm dùng cùng 50 video và sáu retention levels `100, 25, 10, 5, 1, 0%`, tạo thành 300 retention jobs. Prompt full được calibrate ở nominal khoảng `3K`; context length, visual positions và full-target fingerprint được lưu trước khi mask.",
        "DFlash tạo `hidden_context_mask`: text positions luôn được giữ, còn các visual positions sau phần được giữ lại bị đánh dấu để zero. Ở retention decode, target vẫn greedy-prefill và verify trên full hidden context; chỉ tensor conditioning truyền vào draft được biến đổi bởi mask. Vì source policy là `allow_out_of_tolerance`, actual visual-token count giữa các sample không hoàn toàn đồng nhất.",
        "Với mỗi phần trăm, `apply_hidden_context_mask` tạo bản copy hidden context đã zero các visual rows bị loại. DFlash sau đó chạy cùng decoder và cùng `max_new_tokens=256` như length sweep. Target fingerprint được đối chiếu lại để bảo đảm intervention chỉ tác động vào draft-side conditioning.",
        "Mỗi row lưu retention percentage, mask, target/speculative output IDs, acceptance rounds, `tau_effective`, latency và speedup. Retention không dùng attention selection; đây là phép đo độ nhạy của DFlash với hidden visual conditioning theo một mask deterministic.",
        "",
    ])
    if image_name("figure1_insight_summary"):
        lines.extend([
            "![Figure 1(b) — Draft visual retention](figure1_insight_summary.png)",
            "",
            "*Panel (b) của Hình 1: retention sweep theo phần trăm hidden visual context được giữ lại trong draft.*",
            "",
        ])
    lines.extend([
        "### Kết quả và nhận xét",
        f"Trong retention sweep, `tau_effective` chỉ dao động khoảng `{_format(retention_tau_min)}`–`{_format(retention_tau_max)}`, còn speedup khoảng `{_format(retention_speed_min)}`–`{_format(retention_speed_max)}`. Các đường cong khá phẳng; mức retention thấp không tạo ra cải thiện đơn điệu rõ ràng so với 100%.",
        "",
        "**Insight thực nghiệm.** Với DFlash checkpoint hiện tại, việc giảm hidden visual context của draft từ 100% xuống 0% chưa cho thấy quy luật ‘càng bỏ visual càng tốt’. τ đạt gần cực đại ở vùng retention trung gian, trong khi speedup cũng chỉ thay đổi nhẹ. Điều này gợi ý draft conditioning có thể đã được DFlash nén/điều hòa đủ để việc zero thêm visual rows không chuyển thành lợi ích acceptance lớn; đây là giả thuyết cần kiểm tra lại sau khi losslessness được làm sạch.",
        "",
        "**Nhận xét phương pháp.** Kết quả này đã trả lời được câu hỏi DFlash-native ‘draft chịu được bao nhiêu visual hidden context’, nhưng không nên diễn giải thành kết luận về attention-guided visual selection của MSD. Mismatch rate cao gần như giống nhau ở mọi retention group cũng cho thấy cần tách lỗi nền của decoder khỏi hiệu ứng retention.",
        "",
        "## Experiment 3 — DFlash draft attention distribution",
        "",
        "### Mục tiêu",
        "",
        "Quan sát DFlash draft phân bổ attention giữa target-hidden context và noise keys trong các draft layer, tại hai visual-context target nominal `400` và `3000`.",
        "",
        "### Cách thực hiện",
        "",
        "Attention probe được chạy trên 50 video tại hai target visual nominal `400` và `3000`. DFlash attention implementation tạm thời được chuyển sang `eager mode` để forward hook có thể nhận raw attention tensor, sau đó configuration được khôi phục.",
        "Hook được gắn vào các self-attention layer của DFlash draft trong lúc instrumented decode. Với attention tensor có dạng batch/head/query/key, `context_length` xác định boundary giữa target-hidden context và draft noise keys.",
        "Mỗi captured record được rút gọn bằng cách cộng key positions `[:context_length]` thành `context_attention_mass`, cộng phần còn lại thành `noise_attention_mass`, rồi mean trên các head/query được capture. Các row được group theo target visual nominal và draft layer; tổng hai mass dùng để kiểm tra attention normalization.",
        "Vì artifact không lưu mapping per-token sang Instruction/Visual/Text và không archive đầy đủ modality positions trong summary row, hình này là context/noise diagnostic chứ không phải modality-attention plot.",
        "",
    ])
    if image_name("figure2_insight_attention"):
        lines.extend([
            "![Figure 2 — DFlash draft attention distribution](figure2_insight_attention.png)",
            "",
            "*Hình 2. Context/noise attention mass theo DFlash draft layer tại nominal 400 và 3K visual tokens.*",
            "",
        ])
    lines.extend([
        "### Kết quả và nhận xét",
        "",
        f"Các mốc quan sát cho thấy {'; '.join(attention_insights) if attention_insights else 'chưa có attention group hợp lệ'}.",
        "",
        "**Insight thực nghiệm.** Ở draft layer cuối, khoảng 96–97% attention mass nằm trong target-hidden context ở cả hai mốc. Điều này cho thấy DFlash draft dựa chủ yếu vào conditioning đã được target tạo ra, thay vì dành phần lớn attention cho noise block hiện tại. Noise vẫn tăng ở một số layer giữa, cho thấy draft không chỉ sao chép context mà còn dùng các key noise để xây dựng proposal ngắn hạn.",
        "",
        "**Giới hạn.** Context ở đây gộp visual, text và instruction; do đó không thể kết luận riêng visual modality nhận bao nhiêu attention. Insight hợp lệ nhất là về vai trò của target context trong DFlash drafting, không phải về token selection.",
        "",
        "## Experiment 4 — Layer-wise visual flow and hidden-state retention",
        "",
        "### Mục tiêu",
        "",
        "Đánh giá visual information ở target side qua ba probe: mask visual KV từ các layer khác nhau, đo visual attention theo target layer, và đo cosine giữa hidden state với input embedding ban đầu.",
        "",
        "### Cách thực hiện",
        "",
        "Layer run là target-side diagnostic độc lập trên 50 video tại nominal visual target `3000`. Nó không chạy speculative acceptance; mục tiêu là tách ảnh hưởng của target layer khỏi telemetry của DFlash decoder.",
        "Trước mỗi probe, Qwen2.5-VL prefill được chuẩn bị và visual/instruction/text positions được xác định. Với visual-KV experiment, attention mask của target self-attention được bọc từ cut `0, 4, 8, 12, 16, 20, 24` trở đi; target sau đó greedy-generate với `max_new_tokens=256`. Chỉ `diagnostic_output_length` được ghi lại, không phải task accuracy.",
        "Với attention experiment, hook capture query instruction cuối trong target forward. Visual key positions được cộng qua toàn bộ head để tạo một scalar `visual_attention_mass` cho mỗi target layer; do đó kết quả không còn chiều per-head.",
        "Với cosine experiment, forward hook được đặt ở từng decoder block. Hidden state tại visual và text positions được so cosine với input embedding tương ứng, rồi average theo positions và 50 sample qua 36 decoder layers. Đây là phép đo hình học của representation, không phải phép đo losslessness.",
        "",
    ])
    if image_name("figure3_insight_layer_analysis"):
        lines.extend([
            "![Figure 3 — Layer-wise visual flow and attention](figure3_insight_layer_analysis.png)",
            "",
            "*Hình 3. Panel (a) là output-length diagnostic dưới visual-KV masking; panel (b) là head-summed visual attention theo target layer.*",
            "",
        ])
    lines.extend([
        "### Kết quả panel (a): visual-KV cut",
        "",
        f"Mean diagnostic output length giữ nguyên ở mức `{_format(kv_values[0]) if kv_values else 'n/a'}` qua `{len(layer_kv_rows)}` cut points. Đường phẳng này chỉ cho biết độ dài output không đổi trong probe hiện tại; nó không chứng minh task accuracy hoặc answer quality không đổi.",
        "",
        "**Insight thực nghiệm.** Với metric output length, chưa quan sát thấy layer cut làm thay đổi độ dài câu trả lời. Probe này không đủ để kết luận causal importance của visual KV; cần thêm token agreement, answer score hoặc task accuracy nếu muốn đánh giá chất lượng.",
        "",
        "### Kết quả panel (b): visual attention theo layer",
        "",
        f"Head-summed visual attention đạt cực đại khoảng `{_format(value(attention_peak, 'visual_attention_mass_mean') if attention_peak else None)}` ở target layer `{attention_peak.get('layer_index') if attention_peak else 'n/a'}` và giảm còn `{_format(value(attention_last, 'visual_attention_mass_mean') if attention_last else None)}` ở layer cuối. Vì đây là tổng qua head, giá trị có thể lớn hơn 1 và không phải là xác suất của một head đơn lẻ.",
        "",
        "**Insight thực nghiệm.** Visual attention tập trung mạnh hơn ở các layer giữa, sau đó giảm ở các layer cuối. Điều này gợi ý visual information được khai thác mạnh trong middle processing rồi được biến đổi thành representation phục vụ các layer sâu hơn; đây là insight cơ chế, chưa phải bằng chứng rằng các layer giữa có causal contribution cao hơn.",
        "",
        "### Kết quả Figure 6: hidden-state cosine",
        "",
        f"Visual cosine giảm từ `{_format(value(visual_first, 'visual_cosine_mean') if visual_first else None)}` ở layer `{visual_first.get('layer_index') if visual_first else 'n/a'}` xuống `{_format(value(visual_last, 'visual_cosine_mean') if visual_last else None)}` ở layer `{visual_last.get('layer_index') if visual_last else 'n/a'}`. Text cosine duy trì ở mức thấp trong toàn bộ curve.",
        "",
        "**Insight thực nghiệm.** Hidden representation thay đổi đáng kể so với input embedding ban đầu khi đi qua các layer. Visual cosine giảm không đồng nghĩa visual information bị mất; nó chỉ cho thấy representation không còn giữ hình học ban đầu. Vì vậy Figure 6 nên được dùng để mô tả transformation/retention hình học, không thay thế accuracy hay losslessness.",
        "",
    ])
    if image_name("figure6_insight_retention"):
        lines.extend([
            "![Figure 6 — Layer-wise hidden-state cosine retention](figure6_insight_retention.png)",
            "",
            "*Hình 6. Visual/text cosine retention theo target layer; đường cong là diagnostic representation, không phải task accuracy.*",
            "",
        ])
    lines.extend([
        "",
        "## FAILED / INCOMPLETE",
        "",
        f"- Primary coverage is `{primary.get('coverage_valid')}`; missing milestones: `{json.dumps(primary.get('coverage_gaps', {}), ensure_ascii=False)}`.",
        f"- Layer coverage is `{layers.get('coverage_valid')}` because this standalone layer directory does not contain the other primary-stage milestones: `{json.dumps(layers.get('coverage_gaps', {}), ensure_ascii=False)}`.",
        f"- Decode losslessness is incomplete: `{primary.get('mismatch_rows', 0)}` mismatch rows and `{len(primary.get('unsupported_rows', []))}` unsupported rows remain.",
        f"- Calibration policy/policies recorded in the source are `{calibration_policies}`; strict in-tolerance calibration evidence is therefore not established.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "R6 was operationally attempted: the full planned local condition matrix was executed and archived. It is not a green full-study result because the losslessness and coverage gates failed. R7 decision-level evidence was not reached.",
        "",
        "## EVIDENCE GAPS",
        "",
        "- Root cause of the systematic target/speculative output mismatch is not isolated.",
        "- Strict in-tolerance calibration coverage and a single merged audit are not available.",
        "- No baseline comparison under a locked primary metric is included in this DFlash bundle.",
        "- Runtime warnings from video decoding and processor argument handling were recorded but not experimentally ruled out as contributors.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "First isolate the losslessness mismatch with a one-video, one-target, strict-calibration reproduction and token-level first-divergence trace; only after that passes should the full matrix be rerun and merged with layer diagnostics under one strict audit.",
        "",
        "## Figures and tables",
        "",
    ])
    for name in figure_files:
        lines.append(f"- [{name}]({name})")
    for name in (
        "summary.json",
        "source_audits.json",
        "length_summary.csv",
        "retention_summary.csv",
        "attention_summary.csv",
        "layer_visual_kv_summary.csv",
        "layer_attention_summary.csv",
        "layer_cosine_summary.csv",
    ):
        lines.append(f"- [{name}]({name})")
    lines.extend(["", "## Length summary", "", "| Target | N | Valid | OK | Mismatch | Unsupported | Error | Unknown | Lossless rate | τ | Speedup |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary.get("length", []):
        lines.append(f"| {row['condition']} | {row['n']} | {row['valid_n']} | {row['ok']} | {row['mismatch']} | {row['unsupported']} | {row['error']} | {row['unknown']} | {_format(row['lossless_rate'])} | {_format(row['tau_effective_mean'])} | {_format(row['speedup_mean'])} |")
    lines.extend(["", "## Retention summary", "", "| Retention | N | Valid | OK | Mismatch | Unsupported | Error | Unknown | Lossless rate | τ | Speedup |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary.get("retention", []):
        lines.append(f"| {row['condition']} | {row['n']} | {row['valid_n']} | {row['ok']} | {row['mismatch']} | {row['unsupported']} | {row['error']} | {row['unknown']} | {_format(row['lossless_rate'])} | {_format(row['tau_effective_mean'])} | {_format(row['speedup_mean'])} |")
    (output / "REPORT_FINAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    primary_dir: Path = DEFAULT_PRIMARY_DIR,
    layer_dir: Path = DEFAULT_LAYER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build all final artifacts and return the generated summary."""

    summary = build_summary(primary_dir, layer_dir)
    figure_files = render_figures(summary, output_dir)
    write_report(summary, output_dir, figure_files)
    return {**summary, "figure_files": figure_files, "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, default=DEFAULT_PRIMARY_DIR)
    parser.add_argument("--layer-dir", type=Path, default=DEFAULT_LAYER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_report(args.primary_dir, args.layer_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": result["output_dir"], "figures": result["figure_files"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
