"""Paper-style plots generated only from completed, measured rows."""

from __future__ import annotations

from collections import defaultdict
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


def _empty_panel(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=axis.transAxes, color="#666")
    axis.set_axis_off()


def _write_paper_figure1(rows: list[dict[str, Any]], output: Path, plt: Any) -> list[str]:
    length_rows = [row for row in rows if row.get("paper_figure") == "Figure 1(a)"]
    retention_rows = [row for row in rows if row.get("paper_figure") == "Figure 1(b)"]
    length_groups = _group_by_value(length_rows, _paper_group)
    retention_groups = _group_by_value(retention_rows, lambda row: row.get("retention_percentage"))
    if not length_groups and not retention_groups:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), squeeze=False)
    axes = axes[0]
    files: list[str] = []
    if length_groups:
        x = list(sorted(length_groups))
        accepted = [summarize([row.get("accepted_prefix_tokens") for row in length_groups[value]]) for value in x]
        latency = [summarize([
            row.get("decode_seconds") if row.get("decode_seconds") is not None else row.get("end_to_end_seconds")
            for row in length_groups[value]
        ]) for value in x]
        axis = axes[0]
        twin = axis.twinx()
        positions = list(range(len(x)))
        bars = twin.bar(positions, [float(stat["mean"] or 0.0) * 1000 for stat in latency], width=0.58, color="#f4a582", alpha=0.75, label="MSD draft/decode time")
        line = axis.errorbar(
            positions,
            [float(stat["mean"] or 0.0) for stat in accepted],
            yerr=[_errorbar(stat) for stat in accepted],
            marker="o",
            linewidth=1.7,
            color="#2166ac",
            capsize=3,
            label="MSD average accepted length",
        )
        axis.set_xticks(positions, [_token_axis_label(value) for value in x])
        axis.set_xlabel("Visual token length")
        axis.set_ylabel("Average accepted length")
        twin.set_ylabel("Draft/decode time (ms)", color="#b2182b")
        axis.set_title("(a) Visual length")
        axis.grid(axis="y", alpha=0.22)
        axis.legend([line, bars], ["MSD average accepted length", "MSD draft/decode time"], loc="best", fontsize=8)
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
            axis.errorbar(
                positions, means, yerr=errors, marker="o", linewidth=1.7, capsize=3,
                color=colors.get(policy, None), label=label,
            )
        axis.set_xlabel("Retained visual input (%)")
        axis.set_ylabel("Average accepted length")
        axis.set_title("(b) Visual retention")
        axis.invert_xaxis()
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=8)
    else:
        _empty_panel(axes[1], "No measured Figure 1(b) rows")
    fig.suptitle("Figure 1. Impact of visual token length and retention on MSD", y=1.02, fontsize=12)
    fig.tight_layout()
    path = output / "figure1_insight_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    files.append(path.name)
    return files


def _write_paper_figure2(rows: list[dict[str, Any]], output: Path, plt: Any) -> list[str]:
    attention_rows = [
        row for row in rows
        if row.get("paper_figure") == "Figure 2" and row.get("modality") in {"visual", "instruction", "text"}
        and row.get("attention_weight") is not None
    ]
    groups = _group_by_value(attention_rows, _paper_group)
    if not groups:
        return []
    targets = sorted(groups)
    if len(targets) >= 2:
        selected = [targets[0], targets[-1]]
    else:
        selected = targets[:1]
    fig, axes = plt.subplots(1, len(selected), figsize=(5.8 * len(selected), 4.2), squeeze=False)
    axes = axes[0]
    if len(selected) == 1:
        axes = [axes[0]]
    files: list[str] = []
    for axis, target in zip(axes, selected):
        subset = groups[target]
        policies = ["last_instruction"]
        if any(row.get("attention_policy") == "all_text" for row in subset):
            policies.append("all_text")
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
            label = "Last Instr." if policy == "last_instruction" else "All Text"
            axis.plot(
                [position for position, _ in points], [value for _, value in points],
                linewidth=0.8 if policy == "last_instruction" else 1.0,
                color="#2166ac" if policy == "last_instruction" else "#b2182b",
                alpha=0.85, label=label,
            )
        representative = next((row for row in subset if row.get("modality") == "visual"), None)
        if representative:
            spans = [
                ("Instruction", representative.get("instruction_positions", []), "#d9eaf7"),
                ("Visual", representative.get("visual_positions", []), "#d9f0d3"),
                ("Text", representative.get("text_positions", []), "#fce4c4"),
            ]
            for label, positions, color in spans:
                if positions:
                    left, right = min(positions), max(positions)
                    axis.axvspan(left, right, color=color, alpha=0.55, zorder=-5)
                    axis.text((left + right) / 2, 0.98, label, ha="center", va="top", fontsize=8, transform=axis.get_xaxis_transform())
        axis.set_xlabel("Token position")
        axis.set_ylabel("Average attention weight")
        axis.set_title(f"{_token_axis_label(target)} visual tokens")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Figure 2. Final-instruction attention distribution", y=1.02, fontsize=12)
    fig.tight_layout()
    path = output / "figure2_insight_attention.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    files.append(path.name)
    return files


def _write_paper_figure3(rows: list[dict[str, Any]], output: Path, plt: Any) -> list[str]:
    ablation = [row for row in rows if row.get("paper_figure") == "Figure 3"]
    attention = [row for row in rows if row.get("paper_figure") == "Figure 3(b)"]
    if not ablation and not attention:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    left, right = axes[0]
    if ablation:
        grouped = _group_by_value(ablation, lambda row: row.get("layer_cut"))
        cuts = sorted(grouped)
        stats = [summarize([row.get("rouge_l") for row in grouped[cut]]) for cut in cuts]
        left.errorbar(cuts, [stat["mean"] for stat in stats], yerr=[_errorbar(stat) for stat in stats], marker="o", color="#1b7837", capsize=3, label="Ablated output")
        left.axhline(1.0, color="#555", linestyle="--", linewidth=1, label="Native target")
        left.axvline(20, color="#555", linestyle=":", linewidth=1, label="Layer 20")
        left.set_xlabel("Visual KV removal starting layer")
        left.set_ylabel("ROUGE-L vs native target")
        left.set_title("(a) Layer-wise visual ablation")
        left.set_ylim(0, 1.05)
        left.grid(alpha=0.2)
        left.legend(fontsize=8)
    else:
        _empty_panel(left, "No measured Figure 3(a) rows")
    if attention:
        layer_rows = sorted(attention, key=lambda row: int(row.get("layer", 0)))
        max_heads = max((len(row.get("per_head_visual_mass", [])) for row in layer_rows), default=0)
        matrix = []
        layer_values = []
        for row in layer_rows:
            values = [float(value) for value in row.get("per_head_visual_mass", [])]
            if not values:
                values = [float(row.get("visual_mass", 0.0))]
            max_heads = max(max_heads, len(values))
            matrix.append(values)
            layer_values.append(int(row.get("layer", 0)))
        padded = [values + [float("nan")] * (max_heads - len(values)) for values in matrix]
        image = right.imshow(list(map(list, zip(*padded))), aspect="auto", origin="lower", interpolation="nearest", cmap="Blues")
        right.set_xticks(range(len(layer_values)), layer_values)
        right.set_xlabel("Model layer")
        right.set_ylabel("Attention head")
        right.set_title("(b) Visual attention by head/layer")
        right.axvline(layer_values.index(20) if 20 in layer_values else 0, color="#b2182b", linestyle="--", linewidth=1)
        fig.colorbar(image, ax=right, fraction=0.046, pad=0.04, label="Visual attention mass")
    else:
        _empty_panel(right, "No measured Figure 3(b) rows")
    fig.suptitle("Figure 3. Layer-wise visual importance analysis", y=1.02, fontsize=12)
    fig.tight_layout()
    path = output / "figure3_insight_layer_analysis.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def _write_paper_figure6(rows: list[dict[str, Any]], output: Path, plt: Any) -> list[str]:
    retention = [row for row in rows if row.get("paper_figure") == "Figure 6 / Appendix D"]
    if not retention:
        return []
    grouped = _group_by_value(retention, lambda row: row.get("layer"))
    layers = sorted(grouped)
    visual = [summarize([row.get("visual_cosine") for row in grouped[layer]]) for layer in layers]
    text = [summarize([row.get("text_cosine") for row in grouped[layer]]) for layer in layers]
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    axis.errorbar(layers, [stat["mean"] for stat in visual], yerr=[_errorbar(stat) for stat in visual], color="#d73027", marker="o", linewidth=1.5, capsize=2, label="Visual information retention")
    axis.errorbar(layers, [stat["mean"] for stat in text], yerr=[_errorbar(stat) for stat in text], color="#2166ac", marker="o", linewidth=1.5, capsize=2, label="Text information retention")
    axis.axvline(20, color="#555", linestyle="--", linewidth=1, label="Layer 20")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Retention rate")
    axis.set_ylim(0, 1.05)
    axis.set_title("Figure 6. Layer-wise information retention")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output / "figure6_insight_retention.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [path.name]


def write_paper_style_plots(rows: Iterable[dict[str, Any]], output_dir: str | Path) -> list[str]:
    """Write composite figures whose layouts mirror the paper's insight plots."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []
    rows = list(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    files.extend(_write_paper_figure1(rows, output, plt))
    files.extend(_write_paper_figure2(rows, output, plt))
    files.extend(_write_paper_figure3(rows, output, plt))
    files.extend(_write_paper_figure6(rows, output, plt))
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

    length_rows = [row for row in rows if row.get("paper_figure") == "Figure 1(a)"]
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

    attention_rows = [
        row for row in rows
        if row.get("paper_figure") == "Figure 2" and row.get("modality") == "visual"
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
        axis.set_title("Figure 2 validation: final-instruction visual attention")
        axis.grid(alpha=0.25)
        save(fig, "figure2_attention_dilution.png")

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
