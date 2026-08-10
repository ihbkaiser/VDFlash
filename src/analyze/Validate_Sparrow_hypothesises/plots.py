"""Paper-style plots generated only from completed, measured rows."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _group_mean(rows: Iterable[dict[str, Any]], key: str, value: str) -> list[tuple[float, float]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is None or row.get(value) is None:
            continue
        groups[float(row[key])].append(float(row[value]))
    return [(group, sum(values) / len(values)) for group, values in sorted(groups.items())]


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

    attention_rows = [row for row in rows if row.get("paper_figure") == "Figure 2"]
    if attention_rows and all(row.get(key) is not None for row in attention_rows for key in ("visual_mass", "text_mass")):
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in attention_rows:
            grouped[float(row["token_position"])].append(row)
        points = sorted((position, sum(float(row["visual_mass"]) for row in values) / len(values)) for position, values in grouped.items())
        fig, axis = plt.subplots(figsize=(7.0, 4.2))
        axis.plot([x for x, _ in points], [y for _, y in points], color="#762a83")
        axis.set_xlabel("Token position")
        axis.set_ylabel("Attention mass to visual tokens")
        axis.set_title("Figure 2 validation: query-only attention probe")
        axis.grid(alpha=0.25)
        save(fig, "figure2_attention_dilution.png")

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
        points = sorted(retention_layer_rows, key=lambda row: int(row["layer"]))
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.plot([row["layer"] for row in points], [row["visual_cosine"] for row in points], color="#d73027", label="visual")
        axis.plot([row["layer"] for row in points], [row["text_cosine"] for row in points], color="#2166ac", label="text")
        axis.axvline(20, color="#555", linestyle="--", linewidth=1, label="paper Layer 20")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Cosine similarity to input embedding")
        axis.set_title("Figure 6 validation: layer-wise information retention")
        axis.legend()
        axis.grid(alpha=0.25)
        save(fig, "figure6_information_retention.png")
    return files
