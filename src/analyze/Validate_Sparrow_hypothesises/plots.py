"""Paper-style plots generated only from completed, measured rows."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any, Iterable

from .paper_statistics import summarize


def _group_mean(rows: Iterable[dict[str, Any]], key: str, value: str) -> list[tuple[float, float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is None or row.get(value) is None:
            continue
        groups[float(row[key])].append(float(row[value]))
    return [(group, sum(values) / len(values)) for group, values in sorted(groups.items())]


def _paper_group(row: dict[str, Any]) -> float | None:
    value = row.get("calibration_target_visual_tokens")
    if value is None:
        value = row.get("target_visual_tokens")
    if value is None:
        value = row.get("actual_visual_tokens")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _group_by_value(rows: Iterable[dict[str, Any]], value_fn: Any) -> dict[float, list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = value_fn(row)
        if value is not None:
            grouped[float(value)].append(row)
    return grouped


def _errorbar(stat: dict[str, Any]) -> float:
    mean = stat.get("mean")
    low = stat.get("ci95_low")
    high = stat.get("ci95_high")
    if mean is None or low is None or high is None:
        return 0.0
    return max(float(mean) - float(low), float(high) - float(mean))


def _token_axis_label(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def _contiguous_ranges(positions: Iterable[int]) -> list[tuple[int, int]]:
    """Return inclusive contiguous ranges without bridging token gaps."""
    values = sorted({int(position) for position in positions})
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def _cohort_modality_ranges(rows: Iterable[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """Return consensus modality spans for absolute positions across rows.

    Figure 2 averages token weights from samples whose actual prompt lengths
    differ.  A single representative row therefore cannot describe the
    modality boundary of the averaged curve.  Assign each plotted position to
    the modality observed most often at that position, then split the result
    into contiguous spans for shading.
    """
    support: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        modality = str(row.get("modality") or "")
        if modality not in {"instruction", "visual", "text"}:
            continue
        try:
            position = int(row["token_position"])
        except (KeyError, TypeError, ValueError):
            continue
        support[position][modality] += 1
    if not support:
        return []

    # Stable tie-breaking keeps the rendering deterministic at cohort
    # boundaries where two modalities occur equally often.
    tie_order = {"instruction": 0, "visual": 1, "text": 2}
    labels = [
        (position, max(counts, key=lambda modality: (counts[modality], -tie_order[modality])))
        for position, counts in sorted(support.items())
    ]
    ranges: list[tuple[str, int, int]] = []
    start_position, previous_position = labels[0][0], labels[0][0]
    current_modality = labels[0][1]
    for position, modality in labels[1:]:
        if position != previous_position + 1 or modality != current_modality:
            ranges.append((current_modality, start_position, previous_position))
            start_position = position
            current_modality = modality
        previous_position = position
    ranges.append((current_modality, start_position, previous_position))
    return ranges


def _figure2_region_statistics(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Summarize regional mass and per-token weight for Figure 2 labels."""
    unique_rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        weights = row.get("attention_weights")
        if not isinstance(weights, list):
            continue
        row_id = str(row.get("row_id") or f"row-{index}")
        unique_rows[row_id] = row
    result: dict[str, dict[str, float | int]] = {}
    for modality in ("instruction", "visual", "text"):
        masses: list[float] = []
        per_token_means: list[float] = []
        for row in unique_rows.values():
            try:
                positions = [int(value) for value in row.get(f"{modality}_positions", [])]
                values = [float(row["attention_weights"][position]) for position in positions]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            values = [value for value in values if math.isfinite(value)]
            if not values:
                continue
            mass = row.get(f"{modality}_mass")
            try:
                mass_value = float(mass) if mass is not None else sum(values)
            except (TypeError, ValueError):
                mass_value = sum(values)
            if math.isfinite(mass_value):
                masses.append(mass_value)
            per_token_means.append(sum(values) / len(values))
        result[modality] = {
            "mass_mean": sum(masses) / len(masses) if masses else 0.0,
            "mean_weight_per_token": sum(per_token_means) / len(per_token_means) if per_token_means else 0.0,
            "sample_count": len(per_token_means),
        }
    return result


def _empty_panel(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=axis.transAxes, color="#666")
    axis.set_axis_off()


def _save_paper(fig: Any, output: Path, stem: str, plt: Any, formats: tuple[str, ...], watermark: str | None = None) -> list[str]:
    """Save a paper-shaped figure in every requested vector/raster format."""
    if watermark:
        fig.text(0.5, 0.01, watermark, ha="center", va="bottom", color="#b2182b", fontsize=8)
    fig.tight_layout(rect=(0, 0.03 if watermark else 0, 1, 1))
    names: list[str] = []
    for fmt in formats:
        path = output / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        names.append(path.name)
    plt.close(fig)
    return names


def _write_paper_figure1(rows: list[dict[str, Any]], output: Path, plt: Any, formats: tuple[str, ...], watermark: str | None) -> list[str]:
    length_rows = [row for row in rows if row.get("paper_figure") == "Figure 1(a)"]
    retention_rows = [row for row in rows if row.get("paper_figure") == "Figure 1(b)"]
    length_groups = _group_by_value(length_rows, _paper_group)
    series_groups: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in length_rows:
        value = _paper_group(row)
        if value is not None:
            series_groups[str(row.get("series_id") or "msd_keep_visual")][value].append(row)
    retention_groups = _group_by_value(retention_rows, lambda row: row.get("retention_percentage"))
    if not length_groups and not retention_groups:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), squeeze=False)
    axes = axes[0]
    files: list[str] = []
    if length_groups:
        x = list(sorted(length_groups))
        keep_groups = series_groups.get("msd_keep_visual", {})
        accepted = [summarize([row.get("accepted_prefix_tokens") for row in keep_groups.get(value, [])]) for value in x]
        # Figure 1(a) compares the user-visible MSD latency.  Draft-tree
        # prefill is an internal component and is not sufficient to describe
        # the cost of a condition, so use the measured end-to-end duration.
        latency = [summarize([
            row.get("end_to_end_seconds")
            for row in keep_groups.get(value, [])
        ]) for value in x]
        remove_groups = series_groups.get("msd_remove_all", {})
        remove_latency = [summarize([
            row.get("end_to_end_seconds")
            for row in remove_groups.get(value, [])
        ]) for value in x]
        axis = axes[0]
        twin = axis.twinx()
        positions = list(range(len(x)))
        keep_bars = twin.bar(
            [position - 0.16 for position in positions],
            [float(stat["mean"] or 0.0) for stat in latency],
            width=0.30,
            color="#f4a582",
            alpha=0.80,
            label="Keep Visual end-to-end",
        )
        remove_bars = None
        if remove_groups:
            remove_bars = twin.bar(
                [position + 0.16 for position in positions],
                [float(stat["mean"] or 0.0) for stat in remove_latency],
                width=0.30,
                color="#d6604d",
                alpha=0.80,
                label="Remove All end-to-end",
            )
        line = axis.plot(
            positions,
            [float(stat["mean"]) if stat["mean"] is not None else float("nan") for stat in accepted],
            marker="o",
            linewidth=1.7,
            color="#2166ac",
            label="MSD average accepted length",
        )
        remove_groups = series_groups.get("msd_remove_all", {})
        remove_stats = [summarize([row.get("accepted_prefix_tokens") for row in remove_groups.get(value, [])]) for value in x]
        remove_line = axis.plot(
            positions,
            [float(stat["mean"]) if stat["mean"] is not None else float("nan") for stat in remove_stats],
            marker="s", linewidth=1.5, color="#b2182b", linestyle="--", label="MSD remove all visual",
        )
        axis.set_xticks(positions, [_token_axis_label(value) for value in x])
        axis.set_xlabel("Visual token length")
        axis.set_ylabel("Average accepted length")
        twin.set_ylabel("MSD end-to-end latency (s)", color="#b2182b")
        axis.set_title("(a) MSD visual-length sweep (VDC-50 local)")
        axis.grid(axis="y", alpha=0.22)
        handles = [line[0], remove_line[0], keep_bars]
        labels = ["MSD keep visual", "MSD remove all visual", "Keep Visual end-to-end"]
        if remove_bars is not None:
            handles.append(remove_bars)
            labels.append("Remove All end-to-end")
        axis.legend(handles, labels, loc="best", fontsize=8)
    else:
        _empty_panel(axes[0], "No measured Figure 1(a) rows")
    if retention_groups:
        axis = axes[1]
        policies = sorted({str(row.get("selection_policy") or "uniform") for row in retention_rows})
        colors = {"last_instruction": "#2166ac", "top_attention": "#2166ac", "all_text": "#b2182b", "uniform": "#4d4d4d"}
        positions = list(sorted(retention_groups))
        for policy in policies:
            means = []
            errors = []
            for percentage in positions:
                group = [row for row in retention_groups[percentage] if str(row.get("selection_policy") or "uniform") == policy]
                stat = summarize([row.get("accepted_prefix_tokens") for row in group])
                means.append(stat["mean"])
                errors.append(_errorbar(stat))
            label = {
                "last_instruction": "Last Instr.",
                "top_attention": "Last Instr.",
                "all_text": "All Text",
                "uniform": "Uniform",
            }.get(policy, policy)
            axis.plot(
                positions, means, marker="o", linewidth=1.7,
                color=colors.get(policy, None), label=label,
            )
        axis.set_xlabel("Retained visual input (%)")
        axis.set_ylabel("Average accepted length")
        axis.set_title("(b) Draft visual retention (VDC-50 local)")
        axis.invert_xaxis()
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=8)
    else:
        _empty_panel(axes[1], "No measured Figure 1(b) rows")
    fig.suptitle("Figure 1. Impact of visual token length and retention on MSD", y=1.02, fontsize=12)
    return _save_paper(fig, output, "figure1_insight_summary", plt, formats, watermark)


def _write_paper_figure2(rows: list[dict[str, Any]], output: Path, plt: Any, formats: tuple[str, ...], watermark: str | None) -> list[str]:
    # The mechanism claim is about the small MSD draft.  Render exactly the
    # short/long pair instead of mixing target and draft proxy panels.
    return _figure2_for_source(rows, output, plt, "msd_draft", "", formats, watermark)


def _figure2_for_source(
    rows: list[dict[str, Any]],
    output: Path,
    plt: Any,
    source: str,
    suffix: str,
    formats: tuple[str, ...] = ("png",),
    watermark: str | None = None,
) -> list[str]:
    attention_rows = [
        row for row in rows
        if row.get("paper_figure") == "Figure 2"
        and row.get("attention_source", "target") == source
        and row.get("modality") in {"visual", "instruction", "text"}
        and row.get("attention_weight") is not None
    ]
    if not attention_rows:
        # Prefer the compact trace when a run intentionally omits the
        # backwards-compatible per-token expansion.
        for summary in rows:
            if (
                summary.get("paper_figure") == "Figure 2"
                and summary.get("attention_source", "target") == source
                and summary.get("modality") == "summary"
                and summary.get("attention_policy", "last_instruction") == "last_instruction"
                and summary.get("attention_weights") is not None
            ):
                visual = set(int(value) for value in summary.get("visual_positions", []))
                instruction = set(int(value) for value in summary.get("instruction_positions", []))
                text = set(int(value) for value in summary.get("text_positions", []))
                for position, weight in enumerate(summary["attention_weights"]):
                    modality = "visual" if position in visual else "instruction" if position in instruction else "text" if position in text else None
                    if modality is not None:
                        attention_rows.append({**summary, "modality": modality, "token_position": position, "attention_weight": weight})
    groups = _group_by_value(attention_rows, _paper_group)
    if not groups:
        return []
    targets = sorted(groups)
    # The 25K draft-attention run is also used as the selector source for
    # Figure 1(b), but the paper-shaped Figure 2 panel is the short/long
    # diagnostic pair at 0.4K and 3K.  Do not accidentally replace 3K with
    # the auxiliary 25K selector run just because it is the largest target.
    preferred_targets = [400, 3000]
    selected = [target for target in preferred_targets if target in targets]
    if len(selected) < 2:
        selected = targets[:2]
    fig, axes = plt.subplots(1, len(selected), figsize=(5.8 * len(selected), 4.2), squeeze=False)
    axes = axes[0]
    if len(selected) == 1:
        axes = [axes[0]]
    files: list[str] = []
    for axis, target in zip(axes, selected):
        subset = groups[target]
        policies = ["last_instruction"]
        # Figure 2 uses the final instruction query only.
        plot_positions: list[int] = []
        for policy in policies:
            policy_rows = [row for row in subset if str(row.get("attention_policy") or "last_instruction") == policy]
            by_position: dict[int, list[float]] = defaultdict(list)
            for row in policy_rows:
                try:
                    by_position[int(row["token_position"])].append(float(row["attention_weight"]))
                except (KeyError, TypeError, ValueError):
                    continue
            points = sorted((position, sum(values) / len(values)) for position, values in by_position.items())
            if not points:
                continue
            plot_positions = [position for position, _ in points]
            axis.plot(
                [position for position, _ in points], [value for _, value in points],
                linewidth=1.0, color="#2166ac", alpha=0.9, label="Last instruction",
            )
        modality_colors = {
            "instruction": ("Instruction", "#d9eaf7"),
            "visual": ("Visual", "#d9f0d3"),
            "text": ("Text", "#f4d35e"),
        }
        consensus_ranges = _cohort_modality_ranges(policy_rows)
        for modality, left, right in consensus_ranges:
            _, color = modality_colors[modality]
            axis.axvspan(left, right, color=color, alpha=0.55, zorder=-5)
        for modality, (label, _) in modality_colors.items():
            modality_ranges = [
                (left, right)
                for range_modality, left, right in consensus_ranges
                if range_modality == modality
            ]
            if modality_ranges:
                # Put one label in the widest consensus span; a modality can
                # occur in multiple disjoint spans in the original prompt.
                left, right = max(modality_ranges, key=lambda span: span[1] - span[0])
                label_position = (left + right) / 2
                axis.text(label_position, 0.98, label, ha="center", va="top", fontsize=8, transform=axis.get_xaxis_transform())
        if plot_positions:
            axis.set_xlim(min(plot_positions), max(plot_positions))
        region_stats = _figure2_region_statistics(policy_rows)
        if any(int(values["sample_count"]) > 0 for values in region_stats.values()):
            axis.text(
                0.02,
                0.80,
                "Average mass\n"
                f"Instruction {region_stats['instruction']['mass_mean']:.3f} · "
                f"Visual {region_stats['visual']['mass_mean']:.3f} · "
                f"Text {region_stats['text']['mass_mean']:.3f}\n"
                "Mean weight/token\n"
                f"Instruction {region_stats['instruction']['mean_weight_per_token']:.2e} · "
                f"Visual {region_stats['visual']['mean_weight_per_token']:.2e} · "
                f"Text {region_stats['text']['mean_weight_per_token']:.2e}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#aaaaaa"},
            )
        axis.set_xlabel("Token position")
        axis.set_ylabel("Average attention weight")
        axis.set_title(f"{_token_axis_label(target)} visual tokens")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Figure 2. Draft final-instruction attention distribution (VDC-50 local)", y=1.02, fontsize=12)
    return _save_paper(fig, output, "figure2_insight_attention", plt, formats, watermark)


def _write_paper_figure3(rows: list[dict[str, Any]], output: Path, plt: Any, formats: tuple[str, ...], watermark: str | None) -> list[str]:
    ablation = [row for row in rows if row.get("paper_figure") == "Figure 3"]
    attention = [row for row in rows if row.get("paper_figure") == "Figure 3(b)"]
    if not ablation and not attention:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    left, right = axes[0]
    if ablation:
        grouped = _group_by_value(ablation, lambda row: row.get("layer_cut"))
        cuts = sorted(grouped)
        stats = [summarize([row.get("prefix_agreement") for row in grouped[cut]]) for cut in cuts]
        left.plot(cuts, [stat["mean"] for stat in stats], marker="o", color="#1b7837", label="Local prefix agreement")
        quality = left.twinx()
        quality_stats = [summarize([row.get("answer_quality_delta") for row in grouped[cut]]) for cut in cuts]
        quality.plot(
            cuts,
            [stat["mean"] for stat in quality_stats],
            marker="s",
            linestyle="--",
            color="#b2182b",
            label="VDC answer-quality delta",
        )
        quality.axhline(0.0, color="#b2182b", linewidth=0.8, alpha=0.6)
        quality.set_ylabel("VDC answer ROUGE-L delta vs native", color="#b2182b")
        left.axhline(1.0, color="#555", linestyle="--", linewidth=1, label="Native target")
        left.axvline(20, color="#555", linestyle=":", linewidth=1, label="Layer 20")
        left.set_xlabel("Visual KV removal starting layer")
        left.set_ylabel("Prefix agreement vs native output")
        left.set_title("(a) Output agreement and VDC answer quality")
        left.set_ylim(0, 1.05)
        left.grid(alpha=0.2)
        handles, labels = left.get_legend_handles_labels()
        quality_handles, quality_labels = quality.get_legend_handles_labels()
        left.legend(handles + quality_handles, labels + quality_labels, fontsize=8, loc="best")
    else:
        _empty_panel(left, "No measured Figure 3(a) rows")
    if attention:
        head_groups: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in attention:
            layer = int(row.get("layer", 0))
            values = [float(value) for value in row.get("per_head_visual_mass", [])]
            for head, value in enumerate(values):
                head_groups[layer][head].append(value)
        by_layer = {layer: [sum(values) / len(values) for _head, values in sorted(heads.items())] for layer, heads in head_groups.items()}
        layer_values = sorted(by_layer)
        max_heads = max((len(by_layer[layer]) for layer in layer_values), default=0)
        matrix = [by_layer[layer] + [float("nan")] * (max_heads - len(by_layer[layer])) for layer in layer_values]
        # Sort heads by aggregate visual mass, matching the paper's heatmap.
        order = sorted(range(max_heads), key=lambda idx: sum((row[idx] if idx < len(row) and row[idx] == row[idx] else 0.0) for row in matrix), reverse=True)
        matrix = [[row[idx] if idx < len(row) else float("nan") for idx in order] for row in matrix]
        image = right.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest", cmap="Blues")
        right.set_yticks(range(len(layer_values)), layer_values)
        right.set_ylabel("Model layer")
        right.set_xlabel("Heads sorted by visual attention")
        right.set_title("(b) Head/layer visual attention")
        if 20 in layer_values:
            right.axhline(layer_values.index(20), color="#b2182b", linestyle="--", linewidth=1)
        totals = [sum(value for value in by_layer[layer] if value == value) for layer in layer_values]
        if totals:
            inset = right.twinx()
            inset.plot(totals, range(len(layer_values)), color="#b2182b", linestyle="--", linewidth=1.2)
            inset.set_ylim(right.get_ylim())
            inset.set_yticks([])
        fig.colorbar(image, ax=right, fraction=0.046, pad=0.04, label="Visual attention mass")
    else:
        _empty_panel(right, "No measured Figure 3(b) rows")
    fig.suptitle("Figure 3. Layer-wise visual importance (VDC-50 local proxy)", y=1.02, fontsize=12)
    return _save_paper(fig, output, "figure3_insight_layer_analysis", plt, formats, watermark)


def _write_paper_figure6(rows: list[dict[str, Any]], output: Path, plt: Any, formats: tuple[str, ...], watermark: str | None) -> list[str]:
    retention = [row for row in rows if row.get("paper_figure") == "Figure 6 / Appendix D"]
    if not retention:
        return []
    grouped = _group_by_value(retention, lambda row: row.get("layer"))
    layers = sorted(grouped)
    visual = [summarize([row.get("visual_cosine") for row in grouped[layer]]) for layer in layers]
    text = [summarize([row.get("text_cosine") for row in grouped[layer]]) for layer in layers]
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.plot(layers, [stat["mean"] for stat in visual], color="#d73027", marker="o", linewidth=1.5, label="Visual information retention")
    axis.plot(layers, [stat["mean"] for stat in text], color="#2166ac", marker="o", linewidth=1.5, label="Text information retention")
    axis.axvline(20, color="#555", linestyle="--", linewidth=1, label="Layer 20")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Retention rate")
    axis.set_ylim(0, 1.05)
    axis.set_title("Figure 6. Layer-wise information retention (VDC-50 local)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return _save_paper(fig, output, "figure6_insight_retention", plt, formats, watermark)


def write_paper_style_plots(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png",),
    watermark: str | None = None,
) -> list[str]:
    """Write composite figures whose layouts mirror the paper's insight plots."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    rows = list(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Match the paper's compact serif typography while keeping all values
    # measured locally (the style is cosmetic, never a numerical fallback).
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })
    files: list[str] = []
    files.extend(_write_paper_figure1(rows, output, plt, formats, watermark))
    files.extend(_write_paper_figure2(rows, output, plt, formats, watermark))
    files.extend(_write_paper_figure3(rows, output, plt, formats, watermark))
    files.extend(_write_paper_figure6(rows, output, plt, formats, watermark))
    return files


def write_plots(rows: Iterable[dict[str, Any]], output_dir: str | Path) -> list[str]:
    """Write available plots and return their filenames.

    Missing result fields produce no plot; this prevents a planned manifest
    from being mistaken for completed evidence.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    rows = list(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    # The diagnostic length plots describe the keep-visual sweep.  Remove-All
    # rows are still rendered in the paper-shaped Figure 1 panel above, but
    # must not be averaged into the keep-visual latency/speedup curves.
    length_rows = [
        row for row in rows
        if row.get("paper_figure") == "Figure 1(a)"
        and str(row.get("series_id") or "msd_keep_visual") == "msd_keep_visual"
    ]
    length = _group_mean(length_rows, "actual_visual_tokens", "accepted_prefix_tokens")
    if length:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x / 1000 for x, _ in length], [y for _, y in length], marker="o", color="#1769aa")
        axis.set_xlabel("Actual visual tokens (K)")
        axis.set_ylabel("Average accepted prefix length")
        axis.set_title("Figure 1(a) validation: MSD vs visual length")
        axis.grid(alpha=0.25)
        save(fig, "figure1a_acceptance_vs_visual_length.png")

    latency = _group_mean(length_rows, "actual_visual_tokens", "end_to_end_seconds")
    if latency:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x / 1000 for x, _ in latency], [y for _, y in latency], marker="o", color="#b2182b")
        axis.set_xlabel("Actual visual tokens (K)")
        axis.set_ylabel("MSD end-to-end seconds")
        axis.set_title("Figure 1(a) validation: latency vs visual length")
        axis.grid(alpha=0.25)
        save(fig, "figure1a_latency_vs_visual_length.png")

    speedup = _group_mean(length_rows, "actual_visual_tokens", "end_to_end_speedup")
    if speedup:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x / 1000 for x, _ in speedup], [y for _, y in speedup], marker="o", color="#1b7837")
        axis.axhline(1.0, color="#555", linestyle="--", linewidth=1)
        axis.set_xlabel("Actual visual tokens (K)")
        axis.set_ylabel("AR/MSD end-to-end speedup")
        axis.set_title("Figure 1(a) validation: speedup vs visual length")
        axis.grid(alpha=0.25)
        save(fig, "figure1a_speedup_vs_visual_length.png")

    retention_rows = [row for row in rows if row.get("paper_figure") == "Figure 1(b)"]
    retention = _group_mean(retention_rows, "retention_percentage", "accepted_prefix_tokens")
    if retention:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x for x, _ in retention], [y for _, y in retention], marker="o", color="#b2182b")
        axis.set_xlabel("Draft visual input retained (%)")
        axis.set_ylabel("Average accepted prefix length")
        axis.set_title("Figure 1(b) validation: negative visual gain")
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        save(fig, "figure1b_acceptance_vs_retention.png")

    retention_lossless = _group_mean(retention_rows, "retention_percentage", "lossless")
    if retention_lossless:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x for x, _ in retention_lossless], [y for _, y in retention_lossless], marker="o", color="#1b7837")
        axis.set_xlabel("Draft visual input retained (%)")
        axis.set_ylabel("Exact lossless rate")
        axis.set_ylim(-0.05, 1.05)
        axis.invert_xaxis()
        axis.grid(alpha=0.25)
        save(fig, "figure1b_lossless_rate_vs_retention.png")

    attention_sources = ("target", "msd_draft")
    for source in attention_sources:
        attention_rows = [
            row for row in rows
            if row.get("paper_figure") == "Figure 2"
            and row.get("attention_source", "target") == source
            and row.get("modality") == "visual"
            and row.get("attention_weight") is not None
        ]
        if attention_rows:
            grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for row in attention_rows:
                grouped[float(row["token_position"])].append(row)
            points = sorted((position, sum(float(row["attention_weight"]) for row in values) / len(values)) for position, values in grouped.items())
            fig, axis = plt.subplots(figsize=(7.0, 4.2))
            axis.plot([x for x, _ in points], [y for _, y in points], color="#762a83")
            axis.set_xlabel("Token position")
            axis.set_ylabel("Attention weight")
            model_label = "MSD draft" if source == "msd_draft" else "target"
            axis.set_title(f"Figure 2 validation: final-instruction visual attention ({model_label})")
            axis.grid(alpha=0.25)
            name = "figure2_draft_attention_dilution.png" if source == "msd_draft" else "figure2_attention_dilution.png"
            save(fig, name)

    layer_attention_rows = [
        row for row in rows
        if row.get("paper_figure") == "Figure 3(b)" and row.get("visual_mass") is not None
    ]
    layer_attention = _group_mean(layer_attention_rows, "layer", "visual_mass")
    if layer_attention:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x for x, _ in layer_attention], [y for _, y in layer_attention], marker="o", color="#762a83")
        axis.axvline(20, color="#555", linestyle="--", linewidth=1, label="paper Layer 20")
        axis.set_xlabel("Model layer")
        axis.set_ylabel("Visual attention mass")
        axis.set_title("Figure 3(b) validation: visual attention by layer")
        axis.legend()
        axis.grid(alpha=0.25)
        save(fig, "figure3b_visual_attention_by_layer.png")

    layer_rows = [row for row in rows if row.get("paper_figure") == "Figure 3"]
    layer = _group_mean(layer_rows, "layer_cut", "rouge_l")
    if layer:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x for x, _ in layer], [y for _, y in layer], marker="o", color="#1b7837")
        axis.axvline(20, color="#555", linestyle="--", linewidth=1, label="paper Layer 20")
        axis.set_xlabel("First layer with visual KV masked")
        axis.set_ylabel("ROUGE-L vs native target")
        axis.set_title("Figure 3 validation: layer-wise visual-flow ablation")
        axis.legend()
        axis.grid(alpha=0.25)
        save(fig, "figure3_layerwise_visual_ablation.png")

    retention_layer_rows = [row for row in rows if row.get("paper_figure") in {"Figure 6", "Figure 6 / Appendix D"}]
    if retention_layer_rows and all(row.get(key) is not None for row in retention_layer_rows for key in ("layer", "visual_cosine", "text_cosine")):
        visual_points = _group_mean(retention_layer_rows, "layer", "visual_cosine")
        text_points = _group_mean(retention_layer_rows, "layer", "text_cosine")
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([x for x, _ in visual_points], [y for _, y in visual_points], color="#d73027", label="visual")
        axis.plot([x for x, _ in text_points], [y for _, y in text_points], color="#2166ac", label="text")
        axis.axvline(20, color="#555", linestyle="--", linewidth=1, label="paper Layer 20")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Cosine similarity to input embedding")
        axis.set_title("Figure 6 validation: layer-wise information retention")
        axis.legend()
        axis.grid(alpha=0.25)
        save(fig, "figure6_information_retention.png")
    return files
