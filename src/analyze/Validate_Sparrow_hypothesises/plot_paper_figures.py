"""Render measured Sparrow-validation results as paper-style figures.

Reads the JSONL result files written by the GPU runners
(``msd.jsonl``, ``figure2_attention.jsonl``, ``figure2_draft_attention.jsonl``,
``layer_analysis.jsonl``) and produces figures whose layout, labels and colors
mirror the corresponding figures in the Sparrow paper (ACL 2026):

* ``figure1_insight_summary``  — Figure 1: (a) dual-axis plot of MSD average
  accepted length (line) and draft prefill latency (bars) versus visual token
  length; (b) MSD average accepted length versus retained visual input.
* ``figure2_insight_attention`` — Figure 2: (a)/(b) average attention weight
  from the last instruction token to preceding Instruction / Visual / Text
  tokens at 0.4k and 3k.
* ``figure3_insight_layer_analysis`` — Figure 3: (a) local prefix agreement
  after removing visual KV starting at layer x (native baseline); (b) heatmap
  of visual attention mass across model layers and attention heads.
* ``figure6_insight_retention`` — Figure 6: layer-wise visual/text
  information retention (cosine to input embeddings).

Individual panel files (``figure1a_*``, ``figure1b_*``, ...) are also emitted
for reuse.  Only measured rows are used; error rows and missing fields are
skipped.

Run from the repo root::

    python -m src.analyze.Validate_Sparrow_hypothesises.plot_paper_figures \
        --input results/sparrow_validation \
        --output results/sparrow_validation/figures
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ok(row: dict[str, Any]) -> bool:
    return row.get("status") != "error"


def _mean(values: Sequence[float]) -> float:
    values = [v for v in values if v == v]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _stderr(values: Sequence[float]) -> float:
    values = [v for v in values if v == v]
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


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


def _token_axis_label(value: float) -> str:
    return f"{value / 1000:g}K" if value >= 1000 else f"{value:g}"


# ---------------------------------------------------------------------------
# Shared styling: match the paper's compact serif typography
# ---------------------------------------------------------------------------

PAPER_RC = {
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
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": False,
}

# Paper-like color palette (colorblind-friendly).
C_KEEP = "#2166ac"      # MSD keep visual / text
C_REMOVE = "#b2182b"    # MSD remove all visual
C_LATENCY = "#f4a582"   # draft latency bars
C_INSTRUCTION = "#2166ac"
C_VISUAL = "#1b7837"
C_TEXT = "#d6604d"
C_BASELINE = "#666666"
C_LAYER20 = "#b2182b"


def _style_axis(axis: Any, *, grid: bool = True) -> None:
    if grid:
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(fig: Any, output: Path, stem: str, formats: Sequence[str]) -> list[str]:
    fig.tight_layout()
    names: list[str] = []
    for fmt in formats:
        path = output / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        names.append(path.name)
    plt.close(fig)
    return names


# ---------------------------------------------------------------------------
# Figure 1(a): visual-length sweep (dual axis: accepted length + latency)
# ---------------------------------------------------------------------------

def _fig1a_data(rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]] | None:
    length_rows = [r for r in rows if r.get("paper_figure") == "Figure 1(a)" and _ok(r)]
    series_groups: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in length_rows:
        value = _paper_group(row)
        if value is not None:
            series_groups[str(row.get("series_id") or "msd_keep_visual")][value].append(row)
    keep = series_groups.get("msd_keep_visual", {})
    remove = series_groups.get("msd_remove_all", {})
    if not keep:
        return None
    x = sorted(keep)
    latency = [_mean([r.get("prefill_seconds") for r in keep[v] if r.get("prefill_seconds") is not None]) for v in x]
    accepted = [_mean([r.get("accepted_prefix_tokens") for r in keep[v]]) for v in x]
    accepted_rm = [_mean([r.get("accepted_prefix_tokens") for r in remove[v]]) for v in x] if remove else []
    return x, {"latency": latency, "accepted": accepted, "accepted_rm": accepted_rm}


def _draw_fig1a(axis: Any, twin: Any, data: tuple[list[float], dict[str, Any]]) -> None:
    x, values = data
    positions = list(range(len(x)))
    latency, accepted, accepted_rm = values["latency"], values["accepted"], values["accepted_rm"]
    twin.bar(
        positions,
        [v * 1000.0 if v == v else 0.0 for v in latency],
        width=0.55,
        color=C_LATENCY,
        alpha=0.85,
        label="Draft prefill latency",
        zorder=1,
    )
    axis.plot(
        positions,
        accepted,
        marker="o",
        markersize=4.5,
        linewidth=1.7,
        color=C_KEEP,
        label="MSD keep visual",
        zorder=3,
    )
    if accepted_rm:
        axis.plot(
            positions,
            accepted_rm,
            marker="s",
            markersize=4.0,
            linewidth=1.5,
            color=C_REMOVE,
            linestyle="--",
            label="MSD remove all visual",
            zorder=3,
        )
    axis.set_xticks(positions, [_token_axis_label(v) for v in x])
    axis.set_xlabel("Visual token length")
    axis.set_ylabel("Average accepted length")
    twin.set_ylabel("Draft prefill latency (ms)")
    twin.set_ylim(bottom=0)
    _style_axis(axis)
    twin.spines["top"].set_visible(False)


def _figure_1a(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    data = _fig1a_data(rows)
    if data is None:
        return []
    fig, axis = plt.subplots(figsize=(5.0, 3.4))
    twin = axis.twinx()
    _draw_fig1a(axis, twin, data)
    axis.set_title("(a) Visual-length sweep")
    handles, labels = axis.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    axis.legend(handles + h2, labels + l2, loc="upper right")
    return _save(fig, output, "figure1a_visual_length", formats)


# ---------------------------------------------------------------------------
# Figure 1(b): visual retention sweep (negative visual gain)
# ---------------------------------------------------------------------------

def _fig1b_data(rows: list[dict[str, Any]]) -> list[tuple[str, list[float], list[float], list[float]]] | None:
    retention_rows = [r for r in rows if r.get("paper_figure") == "Figure 1(b)" and _ok(r)]
    if not retention_rows:
        return None
    policies = sorted({str(r.get("selection_policy") or "uniform") for r in retention_rows})
    percentages = sorted({float(r.get("retention_percentage")) for r in retention_rows}, reverse=True)
    series: list[tuple[str, list[float], list[float], list[float]]] = []
    for policy in policies:
        means, errors = [], []
        for percentage in percentages:
            group = [r for r in retention_rows if float(r.get("retention_percentage")) == percentage and str(r.get("selection_policy") or "uniform") == policy]
            values = [r.get("accepted_prefix_tokens") for r in group if r.get("accepted_prefix_tokens") is not None]
            means.append(_mean(values))
            errors.append(_stderr(values))
        series.append((policy, percentages, means, errors))
    return series


def _draw_fig1b(axis: Any, series: list[tuple[str, list[float], list[float], list[float]]]) -> None:
    colors = {
        "last_instruction": C_INSTRUCTION,
        "top_attention": C_INSTRUCTION,
        "all_text": C_TEXT,
        "uniform": "#4d4d4d",
    }
    labels = {
        "last_instruction": "Last Instr.",
        "top_attention": "Last Instr.",
        "all_text": "All Text",
        "uniform": "Uniform",
    }
    for policy, percentages, means, errors in series:
        axis.errorbar(
            percentages,
            means,
            yerr=errors,
            marker="o",
            markersize=4.0,
            capsize=2.5,
            linewidth=1.6,
            color=colors.get(policy, "#4d4d4d"),
            label=labels.get(policy, policy),
        )
    axis.set_xlabel("Retained visual input (%)")
    axis.set_ylabel("Average accepted length")
    axis.set_xlim(105, -8)
    axis.invert_xaxis()
    _style_axis(axis)


def _figure_1b(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    series = _fig1b_data(rows)
    if series is None:
        return []
    fig, axis = plt.subplots(figsize=(4.6, 3.4))
    _draw_fig1b(axis, series)
    axis.set_title("(b) Draft visual retention")
    axis.legend(loc="lower left")
    return _save(fig, output, "figure1b_retention", formats)


# ---------------------------------------------------------------------------
# Figure 2(a)/(b): attention weight distribution from last instruction token
# ---------------------------------------------------------------------------

def _fig2_data(rows: list[dict[str, Any]]) -> dict[float, list[dict[str, Any]]] | None:
    attention_rows = [
        r for r in rows
        if r.get("paper_figure") == "Figure 2"
        and r.get("attention_source", "target") == "msd_draft"
        and r.get("modality") in {"visual", "instruction", "text"}
        and r.get("attention_weight") is not None
        and r.get("attention_policy", "last_instruction") == "last_instruction"
        and _ok(r)
    ]
    if not attention_rows:
        for row in rows:
            if (
                row.get("paper_figure") == "Figure 2"
                and row.get("attention_source", "target") == "msd_draft"
                and row.get("modality") == "summary"
                and row.get("attention_policy", "last_instruction") == "last_instruction"
                and row.get("attention_weights") is not None
            ):
                visual = set(int(v) for v in row.get("visual_positions", []))
                instruction = set(int(v) for v in row.get("instruction_positions", []))
                text = set(int(v) for v in row.get("text_positions", []))
                for position, weight in enumerate(row["attention_weights"]):
                    modality = (
                        "visual" if position in visual
                        else "instruction" if position in instruction
                        else "text" if position in text else None
                    )
                    if modality is not None:
                        attention_rows.append({**row, "modality": modality, "token_position": position, "attention_weight": weight})
    if not attention_rows:
        return None
    groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in attention_rows:
        target = _paper_group(row)
        if target is not None:
            groups[target].append(row)
    return groups or None


def _draw_fig2(axis: Any, subset: list[dict[str, Any]], target: float) -> None:
    by_pos_mod: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in subset:
        modality = row.get("modality")
        try:
            pos = int(row["token_position"])
        except (KeyError, TypeError, ValueError):
            continue
        by_pos_mod[modality][pos].append(float(row["attention_weight"]))
    modality_styles = {
        "instruction": (C_INSTRUCTION, "Instruction", "o"),
        "visual": (C_VISUAL, "Visual", "s"),
        "text": (C_TEXT, "Text", "^"),
    }
    for modality, (color, label, marker) in modality_styles.items():
        if not by_pos_mod[modality]:
            continue
        points = sorted((pos, _mean(vals)) for pos, vals in by_pos_mod[modality].items())
        axis.plot(
            [p for p, _ in points],
            [v for _, v in points],
            marker=marker,
            markersize=2.5,
            linewidth=1.0,
            color=color,
            alpha=0.9,
            label=label,
        )
    representative = next((r for r in subset if r.get("modality") == "visual"), None)
    if representative:
        spans = [
            ("Instruction", representative.get("instruction_positions", []), "#d9eaf7"),
            ("Visual", representative.get("visual_positions", []), "#d9f0d3"),
            ("Text", representative.get("text_positions", []), "#fce4c4"),
        ]
        for _label, positions, color in spans:
            if positions:
                left, right = min(positions), max(positions)
                axis.axvspan(left, right, color=color, alpha=0.5, zorder=-5)
    axis.set_xlabel("Token position")
    axis.set_ylabel("Average attention weight")
    axis.set_title(f"{_token_axis_label(target)} visual tokens")
    _style_axis(axis)


def _figure_2(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    groups = _fig2_data(rows)
    if groups is None:
        return []
    targets = sorted(groups)
    selected = [targets[0], targets[-1]] if len(targets) >= 2 else targets[:1]
    fig, axes = plt.subplots(1, len(selected), figsize=(5.4 * len(selected), 3.4), squeeze=False)
    axes = axes[0]
    for axis, target in zip(axes, selected):
        _draw_fig2(axis, groups[target], target)
        axis.legend(loc="upper right", ncol=3, fontsize=7)
    return _save(fig, output, "figure2_attention", formats)


# ---------------------------------------------------------------------------
# Figure 3(a): layer-wise visual KV removal ablation
# ---------------------------------------------------------------------------

def _fig3a_data(rows: list[dict[str, Any]]) -> list[float] | None:
    ablation = [r for r in rows if r.get("paper_figure") == "Figure 3" and _ok(r)]
    if not ablation:
        return None
    grouped: dict[float, list[float]] = defaultdict(list)
    for r in ablation:
        if r.get("layer_cut") is None or r.get("prefix_agreement") is None:
            continue
        grouped[float(r["layer_cut"])].append(float(r["prefix_agreement"]))
    return sorted(grouped) or None


def _draw_fig3a(axis: Any, cuts: Sequence[float], rows: list[dict[str, Any]]) -> None:
    grouped: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("paper_figure") == "Figure 3" and _ok(r) and r.get("layer_cut") is not None and r.get("prefix_agreement") is not None:
            grouped[float(r["layer_cut"])].append(float(r["prefix_agreement"]))
    means = [_mean(grouped[c]) for c in cuts]
    errors = [_stderr(grouped[c]) for c in cuts]
    axis.errorbar(
        cuts,
        means,
        yerr=errors,
        marker="o",
        markersize=4.0,
        capsize=2.5,
        linewidth=1.6,
        color=C_KEEP,
        label="Local prefix agreement",
    )
    axis.axhline(1.0, color=C_BASELINE, linestyle="--", linewidth=1.0, label="Native target")
    if 20 in cuts:
        axis.axvline(20, color=C_LAYER20, linestyle=":", linewidth=1.0, label="Layer 20")
    axis.set_xlabel("Layer where visual KV removal starts")
    axis.set_ylabel("Prefix agreement vs native")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(cuts)
    _style_axis(axis)


def _figure_3a(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    cuts = _fig3a_data(rows)
    if cuts is None:
        return []
    fig, axis = plt.subplots(figsize=(4.6, 3.4))
    _draw_fig3a(axis, cuts, rows)
    axis.set_title("(a) Layer-wise visual ablation")
    axis.legend(loc="lower left")
    return _save(fig, output, "figure3a_layer_ablation", formats)


# ---------------------------------------------------------------------------
# Figure 3(b): head x layer visual attention heatmap
# ---------------------------------------------------------------------------

def _fig3b_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[int]] | None:
    attention = [r for r in rows if r.get("paper_figure") == "Figure 3(b)" and _ok(r)]
    if not attention:
        return None
    head_groups: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in attention:
        try:
            layer = int(row.get("layer", 0))
        except (TypeError, ValueError):
            continue
        values = [float(v) for v in row.get("per_head_visual_mass", [])]
        for head, value in enumerate(values):
            head_groups[layer][head].append(value)
    layer_values = sorted(head_groups)
    max_heads = max((len(head_groups[layer]) for layer in layer_values), default=0)
    if not layer_values or max_heads == 0:
        return None
    matrix = np.full((len(layer_values), max_heads), np.nan)
    for i, layer in enumerate(layer_values):
        for head, values in head_groups[layer].items():
            matrix[i, head] = _mean(values)
    order = np.argsort(-np.nansum(matrix, axis=0), kind="stable")
    matrix = matrix[:, order]
    return matrix, layer_values


def _draw_fig3b(axis: Any, matrix: np.ndarray, layer_values: Sequence[int], *, colorbar: bool = True) -> None:
    image = axis.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="YlGnBu",
    )
    axis.set_yticks(range(len(layer_values)), layer_values)
    axis.set_ylabel("Model layer")
    axis.set_xlabel("Attention heads (sorted)")
    if 20 in layer_values:
        axis.axhline(layer_values.index(20), color="#b2182b", linestyle="--", linewidth=1.0)
    if colorbar:
        fig = axis.get_figure()
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Visual attention mass")


def _figure_3b(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    result = _fig3b_matrix(rows)
    if result is None:
        return []
    matrix, layer_values = result
    fig, axis = plt.subplots(figsize=(4.8, 3.6))
    _draw_fig3b(axis, matrix, layer_values)
    axis.set_title("(b) Visual attention mass")
    return _save(fig, output, "figure3b_head_layer_heatmap", formats)


# ---------------------------------------------------------------------------
# Figure 6: layer-wise information retention (visual/text cosine)
# ---------------------------------------------------------------------------

def _fig6_data(rows: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]] | None:
    retention = [r for r in rows if r.get("paper_figure") == "Figure 6 / Appendix D" and _ok(r)]
    if not retention:
        return None
    visual: dict[float, list[float]] = defaultdict(list)
    text: dict[float, list[float]] = defaultdict(list)
    for r in retention:
        if r.get("layer") is None:
            continue
        if r.get("visual_cosine") is not None:
            visual[float(r["layer"])].append(float(r["visual_cosine"]))
        if r.get("text_cosine") is not None:
            text[float(r["layer"])].append(float(r["text_cosine"]))
    layers = sorted(set(visual) | set(text))
    if not layers:
        return None
    return (
        layers,
        [_mean(visual.get(l, [])) for l in layers],
        [_stderr(visual.get(l, [])) for l in layers],
        [_mean(text.get(l, [])) for l in layers],
        [_stderr(text.get(l, [])) for l in layers],
    )


def _draw_fig6(axis: Any, data: tuple[list[float], list[float], list[float], list[float], list[float], list[float]]) -> None:
    layers, v_mean, v_err, t_mean, t_err = data
    axis.errorbar(layers, v_mean, yerr=v_err, marker="o", markersize=3.5, capsize=2.0, linewidth=1.5, color=C_VISUAL, label="Visual retention")
    axis.errorbar(layers, t_mean, yerr=t_err, marker="s", markersize=3.5, capsize=2.0, linewidth=1.5, color=C_INSTRUCTION, label="Text retention")
    if 20 in layers:
        axis.axvline(20, color=C_LAYER20, linestyle=":", linewidth=1.0, label="Layer 20")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Retention rate (cosine)")
    axis.set_ylim(0, 1.08)
    _style_axis(axis)


def _figure_6(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    data = _fig6_data(rows)
    if data is None:
        return []
    fig, axis = plt.subplots(figsize=(4.6, 3.4))
    _draw_fig6(axis, data)
    axis.set_title("Layer-wise information retention")
    axis.legend(loc="lower left")
    return _save(fig, output, "figure6_information_retention", formats)


# ---------------------------------------------------------------------------
# Combined paper-layout figures (mirror the paper's multi-panel arrangement)
# ---------------------------------------------------------------------------

def _combined_figure1(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    data1a = _fig1a_data(rows)
    series1b = _fig1b_data(rows)
    if data1a is None and series1b is None:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), squeeze=False)
    left, right = axes[0]
    if data1a is not None:
        twin = left.twinx()
        _draw_fig1a(left, twin, data1a)
        left.set_title("(a) Visual-length sweep")
        handles, labels = left.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        left.legend(handles + h2, labels + l2, loc="upper right")
    if series1b is not None:
        _draw_fig1b(right, series1b)
        right.set_title("(b) Draft visual retention")
        right.legend(loc="lower left")
    fig.suptitle("Figure 1. Impact of visual token length and retention on MSD", y=1.02, fontsize=11)
    return _save(fig, output, "figure1_insight_summary", formats)


def _combined_figure2(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    groups = _fig2_data(rows)
    if groups is None:
        return []
    targets = sorted(groups)
    selected = [targets[0], targets[-1]] if len(targets) >= 2 else targets[:1]
    fig, axes = plt.subplots(1, len(selected), figsize=(10.5, 3.6), squeeze=False)
    axes = axes[0]
    for axis, target in zip(axes, selected):
        _draw_fig2(axis, groups[target], target)
        axis.legend(loc="upper right", ncol=3, fontsize=7)
        if target == selected[0]:
            axis.set_title("(a) Short context")
        else:
            axis.set_title("(b) Long context")
    fig.suptitle("Figure 2. Draft final-instruction attention distribution", y=1.02, fontsize=11)
    return _save(fig, output, "figure2_insight_attention", formats)


def _combined_figure3(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    cuts = _fig3a_data(rows)
    result3b = _fig3b_matrix(rows)
    if cuts is None and result3b is None:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), squeeze=False)
    left, right = axes[0]
    if cuts is not None:
        _draw_fig3a(left, cuts, rows)
        left.set_title("(a) Layer-wise visual ablation")
        left.legend(loc="lower left")
    if result3b is not None:
        matrix, layer_values = result3b
        _draw_fig3b(right, matrix, layer_values)
        right.set_title("(b) Visual attention mass")
    fig.suptitle("Figure 3. Layer-wise visual importance", y=1.02, fontsize=11)
    return _save(fig, output, "figure3_insight_layer_analysis", formats)


def _combined_figure6(rows: list[dict[str, Any]], output: Path, formats: Sequence[str]) -> list[str]:
    data = _fig6_data(rows)
    if data is None:
        return []
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    _draw_fig6(axis, data)
    axis.set_title("Layer-wise information retention")
    axis.legend(loc="lower left")
    fig.suptitle("Figure 6. Layer-wise information retention analysis", y=1.02, fontsize=11)
    return _save(fig, output, "figure6_insight_retention", formats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/sparrow_validation"),
                        help="Directory containing the measured JSONL result files.")
    parser.add_argument("--output", type=Path, default=Path("results/sparrow_validation/figures"),
                        help="Directory where figures are written.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        help="Output formats (png/pdf/svg).")
    args = parser.parse_args()

    input_dir = args.input
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(PAPER_RC)

    sources = {
        "msd.jsonl": "Figure 1",
        "figure2_attention.jsonl": "Figure 2",
        "figure2_draft_attention.jsonl": "Figure 2",
        "layer_analysis.jsonl": "Figure 3/6",
    }
    all_rows: list[dict[str, Any]] = []
    for filename, label in sources.items():
        path = input_dir / filename
        if not path.exists():
            print(f"[skip] {filename} not found")
            continue
        rows = read_jsonl(path)
        all_rows.extend(rows)
        print(f"[load] {filename}: {len(rows)} rows ({label})")

    if not all_rows:
        print("No result rows found; nothing to plot.")
        return

    writers = [
        ("Figure 1(a)", _figure_1a, all_rows),
        ("Figure 1(b)", _figure_1b, all_rows),
        ("Figure 2", _figure_2, all_rows),
        ("Figure 3(a)", _figure_3a, all_rows),
        ("Figure 3(b)", _figure_3b, all_rows),
        ("Figure 6", _figure_6, all_rows),
    ]
    for label, writer, rows in writers:
        try:
            names = writer(rows, output, tuple(args.formats))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[error] {label}: {exc}")
            continue
        if names:
            print(f"[ok] {label}: {', '.join(names)}")
        else:
            print(f"[empty] {label}: no measured rows")

    combined = [
        ("Figure 1 (combined)", _combined_figure1, all_rows),
        ("Figure 2 (combined)", _combined_figure2, all_rows),
        ("Figure 3 (combined)", _combined_figure3, all_rows),
        ("Figure 6 (combined)", _combined_figure6, all_rows),
    ]
    for label, writer, rows in combined:
        try:
            names = writer(rows, output, tuple(args.formats))
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {label}: {exc}")
            continue
        if names:
            print(f"[ok] {label}: {', '.join(names)}")
        else:
            print(f"[empty] {label}: no measured rows")

    print(f"\nFigures written to {output.resolve()}")


if __name__ == "__main__":
    main()
