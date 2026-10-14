"""Analyze MSD draft entropy against target-defined visual-heavy positions.

The runner writes one JSON object per (sample, generated position, draft depth)
row.  This module deliberately keeps the analysis independent of lmms-eval so
that the same JSONL can be re-analyzed without loading a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, MutableMapping, Sequence


Row = MutableMapping[str, object]


def _float(row: Row, key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def _bool(row: Row, key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def add_derived_metrics(rows: Iterable[Row]) -> list[Row]:
    """Add the task definitions used by the report to each row."""

    materialized = list(rows)
    for row in materialized:
        row["gamma_greedy"] = int(_bool(row, "argmax_match_full")) - int(
            _bool(row, "argmax_match_cut")
        )
        row["gamma_tv"] = _float(row, "target_tv_cut") - _float(
            row, "target_tv_full"
        )
        # Primary definition: vision is useful when the full-visual drafter
        # matches the target's greedy token and the cut-visual drafter does
        # not.  This is exactly Gamma_greedy > 0.
        row["visual_heavy_target_match"] = row["gamma_greedy"] > 0
        necessity = _float(row, "necessity")
        # Necessity remains a secondary, offline sensitivity diagnostic.
        row["necessity_heavy_01"] = necessity > 0.1
        row["necessity_heavy_03"] = necessity > 0.3
        row["cut_confident_wrong"] = _float(row, "draft_prob_cut") > 0.9 and not _bool(
            row, "argmax_match_cut"
        )
    return materialized


def _assign_deciles(rows: Sequence[Row]) -> None:
    ordered = sorted(
        range(len(rows)), key=lambda index: _float(rows[index], "draft_entropy_cut")
    )
    size = len(ordered)
    for rank, index in enumerate(ordered):
        rows[index]["entropy_decile"] = min(9, (rank * 10) // size) if size else 0


def _safe_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else float("nan")


def _signal_value(row: Row, signal: str) -> float:
    if signal == "cut_entropy":
        return _float(row, "draft_entropy_cut")
    if signal == "one_minus_cut_prob":
        return 1.0 - _float(row, "draft_prob_cut")
    if signal == "cut_top2_prob":
        return _float(row, "draft_prob_cut") - _float(row, "draft_margin_cut")
    if signal == "cut_margin":
        return _float(row, "draft_margin_cut")
    if signal == "inverse_cut_margin":
        return -_float(row, "draft_margin_cut")
    if signal == "full_visual_attention":
        return _float(row, "draft_attention_mass_full")
    raise KeyError(f"Unknown signal: {signal}")


def _pearson(scores: Sequence[float], targets: Sequence[float]) -> float:
    pairs = [
        (float(score), float(target))
        for score, target in zip(scores, targets)
        if math.isfinite(float(score)) and math.isfinite(float(target))
    ]
    if len(pairs) < 2:
        return float("nan")
    score_mean = mean(score for score, _ in pairs)
    target_mean = mean(target for _, target in pairs)
    numerator = sum(
        (score - score_mean) * (target - target_mean) for score, target in pairs
    )
    score_norm = math.sqrt(sum((score - score_mean) ** 2 for score, _ in pairs))
    target_norm = math.sqrt(sum((target - target_mean) ** 2 for _, target in pairs))
    if score_norm == 0.0 or target_norm == 0.0:
        return float("nan")
    return numerator / (score_norm * target_norm)


SIGNALS = (
    "one_minus_cut_prob",
    "cut_top2_prob",
    "cut_entropy",
    "inverse_cut_margin",
    "full_visual_attention",
)


def compute_signal_comparison(rows: Sequence[Row], group_name: str = "all") -> list[dict]:
    """Compare gate signals with both binary and signed visual-help labels."""

    event_labels = [
        float(_bool(row, "visual_heavy_target_match")) for row in rows
    ]
    gamma_labels = [_float(row, "gamma_greedy") for row in rows]
    result = []
    for signal in SIGNALS:
        scores = [_signal_value(row, signal) for row in rows]
        auroc, pr_auc = _auc(scores, [bool(label) for label in event_labels])
        result.append(
            {
                "group": group_name,
                "signal": signal,
                "point_biserial_visual_heavy": _pearson(scores, event_labels),
                "pearson_gamma_greedy": _pearson(scores, gamma_labels),
                "auroc": auroc,
                "pr_auc": pr_auc,
                "positive_count": int(sum(event_labels)),
                "count": len(rows),
            }
        )
    return result


def compute_sequence_heaviness(rows: Sequence[Row]) -> tuple[list[dict], list[dict]]:
    """Aggregate rows into chains and correlate signals with visual heaviness."""

    sequence_rows = []
    for chain in _chain_groups(rows):
        if not chain:
            continue
        visual_events = [
            float(_bool(row, "visual_heavy_target_match")) for row in chain
        ]
        gamma_values = [_float(row, "gamma_greedy") for row in chain]
        item = {
            "sample_id": chain[0].get("sample_id"),
            "task": chain[0].get("task"),
            "t": chain[0].get("t"),
            "positions": len(chain),
            "visual_heaviness": mean(visual_events),
            "mean_gamma_greedy": mean(gamma_values),
        }
        for signal in SIGNALS:
            item[signal] = mean(_signal_value(row, signal) for row in chain)
        sequence_rows.append(item)

    correlations = []
    heaviness = [float(row["visual_heaviness"]) for row in sequence_rows]
    gamma = [float(row["mean_gamma_greedy"]) for row in sequence_rows]
    for signal in SIGNALS:
        values = [float(row[signal]) for row in sequence_rows]
        correlations.append(
            {
                "signal": signal,
                "pearson_sequence_visual_heaviness": _pearson(values, heaviness),
                "pearson_sequence_mean_gamma": _pearson(values, gamma),
                "chains": len(sequence_rows),
            }
        )
    return sequence_rows, correlations


def _signal_thresholds(rows: Sequence[Row], signal: str) -> list[float]:
    values = sorted(
        {
            _signal_value(row, signal)
            for row in rows
            if math.isfinite(_signal_value(row, signal))
        }
    )
    if not values:
        return []
    return [
        values[0] - 1e-6,
        *(
            values[min(len(values) - 1, int(index * (len(values) - 1) / 20))]
            for index in range(1, 21)
        ),
    ]


def compute_gate_thresholds(rows: Sequence[Row]) -> list[dict]:
    """Evaluate deployable threshold gates against target-defined visual help."""

    chains = _chain_groups(rows)
    positive_count = sum(_bool(row, "visual_heavy_target_match") for row in rows)
    negative_count = len(rows) - positive_count
    result = []
    for signal in SIGNALS:
        for threshold in _signal_thresholds(rows, signal):
            selected = [
                _signal_value(row, signal) > threshold for row in rows
            ]
            true_positive = sum(
                use_full and _bool(row, "visual_heavy_target_match")
                for use_full, row in zip(selected, rows)
            )
            false_positive = sum(
                use_full and not _bool(row, "visual_heavy_target_match")
                for use_full, row in zip(selected, rows)
            )
            selected_count = sum(selected)
            precision = true_positive / selected_count if selected_count else float("nan")
            recall = true_positive / positive_count if positive_count else float("nan")
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if math.isfinite(precision) and precision + recall
                else float("nan")
            )
            chain_lengths = []
            for chain in chains:
                running = 1
                length = 0
                for row in chain:
                    use_full = _signal_value(row, signal) > threshold
                    match = int(
                        _bool(row, "argmax_match_full")
                        if use_full
                        else _bool(row, "argmax_match_cut")
                    )
                    running *= match
                    length += running
                chain_lengths.append(length)
            result.append(
                {
                    "signal": signal,
                    "threshold": threshold,
                    "full_fraction": selected_count / len(rows) if rows else float("nan"),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_positive_rate": (
                        false_positive / negative_count if negative_count else float("nan")
                    ),
                    "mean_chain_length": mean(chain_lengths) if chain_lengths else float("nan"),
                    "positive_count": positive_count,
                    "count": len(rows),
                }
            )
    return result


def compute_decile_summary(rows: Sequence[Row], threshold: float = 0.3) -> list[dict]:
    """Summarize cut-entropy bins using target agreement as the main label.

    ``threshold`` selects the secondary necessity diagnostic reported beside
    the target-agreement label; it does not redefine visual-heavy.
    """

    _assign_deciles(rows)
    necessity_key = "necessity_heavy_03" if threshold == 0.3 else "necessity_heavy_01"
    summaries: list[dict] = []
    for decile in range(10):
        group = [row for row in rows if int(row.get("entropy_decile", 0)) == decile]
        summaries.append(
            {
                "decile": decile + 1,
                "count": len(group),
                "mean_cut_entropy": _safe_mean(
                    [_float(row, "draft_entropy_cut") for row in group]
                ),
                "mean_gamma_greedy": _safe_mean(
                    [_float(row, "gamma_greedy") for row in group]
                ),
                "mean_gamma_tv": _safe_mean(
                    [_float(row, "gamma_tv") for row in group]
                ),
                "mean_necessity": _safe_mean(
                    [_float(row, "necessity") for row in group]
                ),
                "visual_heavy_pct": 100.0
                * _safe_mean(
                    [float(_bool(row, "visual_heavy_target_match")) for row in group]
                ),
                "necessity_heavy_pct": 100.0
                * _safe_mean([float(_bool(row, necessity_key)) for row in group]),
                "cut_confident_wrong_pct": 100.0
                * _safe_mean(
                    [float(_bool(row, "cut_confident_wrong")) for row in group]
                ),
            }
        )
    return summaries


def _chain_groups(rows: Sequence[Row]) -> list[list[Row]]:
    groups: dict[tuple[object, object, object], list[Row]] = defaultdict(list)
    for row in rows:
        groups[(row.get("sample_id"), row.get("task"), row.get("t"))].append(row)
    return [sorted(group, key=lambda row: int(row.get("j", 0))) for group in groups.values()]


def _chain_length(group: Sequence[Row], match_key: str) -> int:
    running = 1
    length = 0
    for row in group:
        running *= int(_bool(row, match_key))
        length += running
    return length


def simulate_chain_gates(rows: Sequence[Row], thresholds: Sequence[float]) -> list[dict]:
    """Simulate full/cut/oracle policies on each logged top-1 chain."""

    chains = _chain_groups(rows)
    if not chains:
        return []

    policies: list[tuple[str, object]] = [
        ("always_full", None),
        ("always_cut", None),
    ]
    policies.extend(("entropy_gate", threshold) for threshold in thresholds)
    policies.append(("oracle", None))

    results: list[dict] = []
    for policy, threshold in policies:
        lengths = []
        full_fractions = []
        for chain in chains:
            selected_full = []
            selected_match = []
            for row in chain:
                full_match = int(_bool(row, "argmax_match_full"))
                cut_match = int(_bool(row, "argmax_match_cut"))
                if policy == "always_full":
                    use_full = True
                elif policy == "always_cut":
                    use_full = False
                elif policy == "entropy_gate":
                    use_full = _float(row, "draft_entropy_cut") > float(threshold)
                else:
                    use_full = full_match > cut_match
                selected_full.append(use_full)
                selected_match.append(full_match if use_full else cut_match)

            running = 1
            length = 0
            for match in selected_match:
                running *= match
                length += running
            lengths.append(length)
            full_fractions.append(mean(float(item) for item in selected_full))

        result = {
            "policy": policy,
            "threshold": threshold,
            "length": mean(lengths),
            "full_fraction": mean(full_fractions),
            "chains": len(chains),
        }
        results.append(result)
    return results


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    finite = [
        (float(score), int(label))
        for score, label in zip(scores, labels)
        if math.isfinite(float(score))
    ]
    if not finite or len({label for _, label in finite}) < 2:
        return float("nan"), float("nan")
    values, targets = zip(*finite)
    return float(roc_auc_score(targets, values)), float(
        average_precision_score(targets, values)
    )


def compute_auc_rows(rows: Sequence[Row], group_name: str = "all") -> list[dict]:
    signals = {
        "cut_entropy": "draft_entropy_cut",
        "one_minus_cut_prob": None,
        "cut_top2_prob": None,
        "cut_margin": "draft_margin_cut",
        "inverse_cut_margin": None,
        "full_visual_attention": "draft_attention_mass_full",
    }
    result = []
    for label_name, label_key in (
        ("visual_heavy_target_match", "visual_heavy_target_match"),
        ("necessity_heavy_01", "necessity_heavy_01"),
        ("necessity_heavy_03", "necessity_heavy_03"),
    ):
        labels = [
            _bool(row, label_key)
            if label_key is not None
            else _float(row, "gamma_greedy") > 0
            for row in rows
        ]
        for signal_name, signal_key in signals.items():
            if signal_key is None:
                if signal_name == "one_minus_cut_prob":
                    scores = [1.0 - _float(row, "draft_prob_cut") for row in rows]
                elif signal_name == "cut_top2_prob":
                    scores = [
                        _float(row, "draft_prob_cut")
                        - _float(row, "draft_margin_cut")
                        for row in rows
                    ]
                else:
                    scores = [-_float(row, "draft_margin_cut") for row in rows]
            else:
                scores = [_float(row, signal_key) for row in rows]
            roc_auc, pr_auc = _auc(scores, labels)
            result.append(
                {
                    "group": group_name,
                    "label": label_name,
                    "signal": signal_name,
                    "auroc": roc_auc,
                    "pr_auc": pr_auc,
                    "count": len(rows),
                }
            )
    return result


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: Sequence[dict], fields: Sequence[str], limit: int = 20) -> str:
    if not rows:
        return "_No rows._"
    selected = rows[:limit]
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in selected
    ]
    return "\n".join([header, divider, *body])


def _write_gate_plot(path: Path, gate_rows: Sequence[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.0, 4.2))
    for policy in ("always_full", "always_cut", "entropy_gate", "oracle"):
        points = [row for row in gate_rows if row["policy"] == policy]
        if not points:
            continue
        axis.plot(
            [row["full_fraction"] for row in points],
            [row["length"] for row in points],
            marker="o",
            label=policy,
        )
    axis.set_xlabel("Fraction of positions using full vision")
    axis.set_ylabel("Chain L")
    axis.grid(alpha=0.25)
    if gate_rows:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(input_path: Path, output_dir: Path) -> dict[str, Path]:
    rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    add_derived_metrics(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    decile_rows = []
    for threshold in (0.1, 0.3):
        for item in compute_decile_summary(rows, threshold):
            item.update({"threshold": threshold, "task": "all", "j": "all"})
            decile_rows.append(item)
        for task in sorted({str(row.get("task")) for row in rows}):
            for depth in sorted({int(row.get("j", 0)) for row in rows}):
                subset = [
                    row
                    for row in rows
                    if str(row.get("task")) == task and int(row.get("j", 0)) == depth
                ]
                for item in compute_decile_summary(subset, threshold):
                    item.update({"threshold": threshold, "task": task, "j": depth})
                    decile_rows.append(item)
    _write_csv(output_dir / "deciles.csv", decile_rows)

    auc_rows = compute_auc_rows(rows)
    for task in sorted({str(row.get("task")) for row in rows}):
        auc_rows.extend(compute_auc_rows([row for row in rows if str(row.get("task")) == task], task))
    for depth in sorted({int(row.get("j", 0)) for row in rows}):
        auc_rows.extend(compute_auc_rows([row for row in rows if int(row.get("j", 0)) == depth], f"j={depth}"))
    _write_csv(output_dir / "auc.csv", auc_rows)

    signal_rows = compute_signal_comparison(rows)
    for task in sorted({str(row.get("task")) for row in rows}):
        signal_rows.extend(
            compute_signal_comparison(
                [row for row in rows if str(row.get("task")) == task], task
            )
        )
    for depth in sorted({int(row.get("j", 0)) for row in rows}):
        signal_rows.extend(
            compute_signal_comparison(
                [row for row in rows if int(row.get("j", 0)) == depth],
                f"j={depth}",
            )
        )
    _write_csv(output_dir / "signal_comparison.csv", signal_rows)

    sequence_rows, sequence_correlations = compute_sequence_heaviness(rows)
    _write_csv(output_dir / "sequence_heaviness.csv", sequence_rows)
    _write_csv(output_dir / "sequence_signal_correlation.csv", sequence_correlations)

    gate_threshold_rows = compute_gate_thresholds(rows)
    _write_csv(output_dir / "gate_thresholds.csv", gate_threshold_rows)

    low_entropy_rows = []
    _assign_deciles(rows)
    for threshold, key in (
        ("target_match", "visual_heavy_target_match"),
        ("necessity_0.1", "necessity_heavy_01"),
        ("necessity_0.3", "necessity_heavy_03"),
    ):
        heavy = [row for row in rows if _bool(row, key)]
        low = [row for row in heavy if int(row.get("entropy_decile", 0)) < 3]
        low_entropy_rows.append(
            {
                "threshold": threshold,
                "visual_heavy_count": len(heavy),
                "low_entropy_count": len(low),
                "low_entropy_share_pct": 100.0 * len(low) / len(heavy) if heavy else float("nan"),
            }
        )
    _write_csv(output_dir / "low_entropy_share.csv", low_entropy_rows)

    entropy_values = sorted({_float(row, "draft_entropy_cut") for row in rows if math.isfinite(_float(row, "draft_entropy_cut"))})
    thresholds = []
    if entropy_values:
        thresholds = [entropy_values[0] - 1e-6]
        thresholds.extend(
            entropy_values[min(len(entropy_values) - 1, int(index * (len(entropy_values) - 1) / 20))]
            for index in range(1, 21)
        )
    gate_rows = simulate_chain_gates(rows, thresholds)
    _write_csv(output_dir / "gate.csv", gate_rows)
    _write_gate_plot(output_dir / "gate.png", gate_rows)

    report = output_dir / "REPORT.md"
    all_deciles = [row for row in decile_rows if row["task"] == "all" and row["j"] == "all"]
    if rows:
        overall_signals = [row for row in signal_rows if row["group"] == "all"]
        deployable_signals = [
            row
            for row in overall_signals
            if row["signal"] != "full_visual_attention"
        ]
        best_deployable = max(
            deployable_signals,
            key=lambda row: (float(row["auroc"]), float(row["pr_auc"])),
        )
        offline_attention = next(
            row for row in overall_signals if row["signal"] == "full_visual_attention"
        )
        conclusion = (
            "1. Visual-heavy is defined by target agreement: the full-visual drafter matches the target while the cut-visual drafter does not.\n"
            f"2. Among cut-only deployable signals, `{best_deployable['signal']}` ranks best here (AUROC={float(best_deployable['auroc']):.3f}, PR-AUC={float(best_deployable['pr_auc']):.3f}); this is a weak signal, not a reliable detector.\n"
            f"3. Full-mode attention has higher PR-AUC ({float(offline_attention['pr_auc']):.3f}) but is not available to a cut-only gate and has near-chance AUROC ({float(offline_attention['auroc']):.3f}).\n"
            "4. Sequence visual heaviness is the fraction of positions where full vision helps; threshold sweeps report the precision/recall and acceptance trade-off.\n"
            "5. Necessity thresholds 0.1 and 0.3 remain secondary diagnostics, and their masked-target construction is conservative.\n"
        )
    else:
        conclusion = (
            "1. No diagnostic rows were produced, so entropy cannot be evaluated.\n"
            "2. No AUROC/PR-AUC estimate is available.\n"
            "3. No visual-heavy tokens were observed in this run.\n"
            "4. No chain-gate acceptance-length estimate is available.\n"
            "5. Rerun after the losslessness check passes; necessity is conservative because it masks visual keys for every generated query.\n"
        )
    report.write_text(
        "# MSD entropy gate diagnostic\n\n"
        f"Input: `{input_path}`\n\n"
        f"Rows: **{len(rows)}**. Primary visual-heavy means `argmax_match_full=1` and `argmax_match_cut=0` "
        "(`Gamma_greedy > 0`). Necessity thresholds `> 0.1` and `> 0.3` are secondary diagnostics; "
        "necessity is conservative because the masked target blocks visual keys for every generated-token query.\n\n"
        "## Decile summary\n\n"
        + _markdown_table(
            all_deciles,
            ["threshold", "decile", "count", "mean_gamma_greedy", "mean_gamma_tv", "mean_necessity", "visual_heavy_pct", "necessity_heavy_pct", "cut_confident_wrong_pct"],
            limit=30,
        )
        + "\n\n## AUROC / PR-AUC\n\n"
        + _markdown_table(auc_rows, ["group", "label", "signal", "auroc", "pr_auc", "count"], limit=30)
        + "\n\n## Gate-signal correlation\n\n"
        + _markdown_table(
            signal_rows,
            [
                "group",
                "signal",
                "point_biserial_visual_heavy",
                "pearson_gamma_greedy",
                "auroc",
                "pr_auc",
                "positive_count",
                "count",
            ],
            limit=30,
        )
        + "\n\n## Sequence-level visual heaviness\n\n"
        + "Sequence visual heaviness is the fraction of chain positions where full vision matches the target and cut vision does not.\n\n"
        + _markdown_table(
            sequence_correlations,
            [
                "signal",
                "pearson_sequence_visual_heaviness",
                "pearson_sequence_mean_gamma",
                "chains",
            ],
        )
        + "\n\n## Threshold gate behavior\n\n"
        + "A gate selects full vision when the signal exceeds its threshold. The table shows the first 30 operating points; the complete sweep is in `gate_thresholds.csv`.\n\n"
        + _markdown_table(
            gate_threshold_rows,
            [
                "signal",
                "threshold",
                "full_fraction",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "mean_chain_length",
            ],
            limit=30,
        )
        + "\n\n## Low-entropy share of visual-heavy rows\n\n"
        + _markdown_table(low_entropy_rows, ["threshold", "visual_heavy_count", "low_entropy_count", "low_entropy_share_pct"])
        + "\n\n## Chain gate\n\n"
        + "![chain gate](gate.png)\n\n"
        + _markdown_table(gate_rows[:25], ["policy", "threshold", "length", "full_fraction", "chains"])
        + "\n\n## Five-line conclusion\n\n"
        + conclusion,
        encoding="utf-8",
    )
    return {
        "deciles": output_dir / "deciles.csv",
        "auc": output_dir / "auc.csv",
        "signal_comparison": output_dir / "signal_comparison.csv",
        "sequence_heaviness": output_dir / "sequence_heaviness.csv",
        "sequence_signal_correlation": output_dir / "sequence_signal_correlation.csv",
        "gate_thresholds": output_dir / "gate_thresholds.csv",
        "gate": output_dir / "gate.csv",
        "plot": output_dir / "gate.png",
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = analyze(args.input, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
