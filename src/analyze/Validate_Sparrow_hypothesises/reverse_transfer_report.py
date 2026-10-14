"""Aggregate MSD reverse-transfer JSONL into paired tables, figures, and report."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .reverse_transfer import EXPERIMENTS, paired_bootstrap_delta

METRICS = ("accuracy", "correct_option_probability", "nll", "margin", "prediction_entropy")
CONTRASTS = (("E2", "E1"), ("E4", "E3"), ("E4", "E2"), ("E4", "E6"))


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _bootstrap_mean(values: list[float], repeats: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    draws = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(max(1, repeats))]
    draws.sort()
    return [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]


def summarize_rows(rows: list[Mapping[str, Any]], *, evidence_only: bool = True, repeats: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if str(row.get("permutation_id", "identity")) == "identity"
        and (not evidence_only or row.get("evidence_eligible") is True)
    ]
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row.get("dataset", "unknown")), str(row.get("task_type", "all")), str(row.get("condition")))].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, task_type, condition), values in sorted(grouped.items()):
        first = values[0]
        summary: dict[str, Any] = {
            "dataset": dataset,
            "task_type": task_type,
            "condition": condition,
            "input_order": first.get("input_order"),
            "msd_conditioning": first.get("msd_conditioning"),
            "samples": len(values),
            "evidence_only": evidence_only,
        }
        for offset, metric in enumerate(METRICS):
            metric_values = [float(row[metric]) for row in values if row.get(metric) is not None]
            summary[metric] = _mean(metric_values)
            summary[f"{metric}_ci95"] = _bootstrap_mean(metric_values, repeats, seed + offset)
        summaries.append(summary)
    return summaries


def contrast_rows(rows: list[Mapping[str, Any]], *, repeats: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    usable = [
        row for row in rows
        if str(row.get("permutation_id", "identity")) == "identity"
        and row.get("evidence_eligible") is True
    ]
    for row in usable:
        row["_pair_group"] = f"{row.get('video_id', row.get('sample_id'))}|{row.get('permutation_id', 'identity')}"
    output: list[dict[str, Any]] = []
    datasets = sorted({str(row.get("dataset")) for row in usable})
    for dataset in datasets:
        subset = [row for row in usable if str(row.get("dataset")) == dataset]
        tasks = ["all"] + sorted({str(row.get("task_type", "unknown")) for row in subset})
        for task in tasks:
            task_rows = subset if task == "all" else [row for row in subset if str(row.get("task_type")) == task]
            for condition_a, condition_b in CONTRASTS:
                for metric in METRICS:
                    result = paired_bootstrap_delta(
                        task_rows,
                        value_key=metric,
                        condition_a=condition_a,
                        condition_b=condition_b,
                        group_key="_pair_group",
                        repeats=repeats,
                        seed=seed,
                    )
                    output.append({
                        "dataset": dataset,
                        "task_type": task,
                        "contrast": f"{condition_a}-{condition_b}",
                        "metric": metric,
                        **result,
                    })
    for row in usable:
        row.pop("_pair_group", None)
    return output


def permutation_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("evidence_eligible") is True:
            by_key[(str(row.get("dataset")), str(row.get("sample_id")), str(row.get("condition")))][str(row.get("permutation_id"))] = row
    for (dataset, sample_id, condition), variants in sorted(by_key.items()):
        if "identity" not in variants:
            continue
        for permutation_id, row in variants.items():
            if permutation_id == "identity":
                continue
            identity = variants["identity"]
            identity_options = list(identity.get("options", []))
            permuted_options = list(row.get("options", []))
            identity_prediction = identity.get("predicted_answer")
            expected_permuted_prediction = None
            if (
                isinstance(identity_prediction, str)
                and identity_prediction in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                and len(identity_prediction) == 1
                and ord(identity_prediction) - ord("A") < len(identity_options)
                and permuted_options
            ):
                predicted_content = identity_options[ord(identity_prediction) - ord("A")]
                if predicted_content in permuted_options:
                    expected_permuted_prediction = chr(
                        ord("A") + permuted_options.index(predicted_content)
                    )
            output.append({
                "dataset": dataset,
                "sample_id": sample_id,
                "condition": condition,
                "permutation_id": permutation_id,
                "identity_prediction": identity_prediction,
                "permuted_prediction": row.get("predicted_answer"),
                "expected_permuted_prediction": expected_permuted_prediction,
                "identity_correct_label": identity.get("ground_truth"),
                "permuted_correct_label": row.get("ground_truth"),
                "prediction_follows_permutation": bool(
                    expected_permuted_prediction is not None
                    and row.get("predicted_answer") == expected_permuted_prediction
                ),
                "identity_accuracy": identity.get("accuracy"),
                "permuted_accuracy": row.get("accuracy"),
            })
    return output


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _make_figures(summaries: list[dict[str, Any]], contrasts: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return {"status": "unavailable", "error": str(exc)}
    output.mkdir(parents=True, exist_ok=True)
    evidence = [row for row in summaries if row.get("evidence_only")]
    overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evidence:
        if row.get("accuracy") is not None:
            overall[str(row["condition"])].append(row)
    labels = [condition for condition in EXPERIMENTS if condition in overall]
    if labels:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        datasets = sorted({str(row["dataset"]) for row in evidence})
        for axis, dataset in zip(axes, datasets[:2]):
            values = []
            for label in labels:
                rows = [row for row in overall[label] if row["dataset"] == dataset]
                total = sum(int(row["samples"]) for row in rows)
                values.append(sum(float(row["accuracy"]) * int(row["samples"]) for row in rows) / total if total else 0.0)
            axis.bar(labels, values)
            axis.set_title(dataset)
            axis.set_ylim(0, 1)
            axis.set_ylabel("MSD draft answer accuracy")
        fig.suptitle("MSD reverse text-to-visual transfer: E1-E6")
        fig.tight_layout()
        fig.savefig(output / "e1_e6_accuracy.png", dpi=180)
        plt.close(fig)
    contrast = [row for row in contrasts if row["contrast"] == "E4-E3" and row["metric"] == "accuracy" and row["task_type"] == "all" and row.get("estimate") is not None]
    if contrast:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.bar([row["dataset"] for row in contrast], [row["estimate"] for row in contrast])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("E4 - E3 accuracy")
        axis.set_title("Text-to-Video versus Video-to-Text, visual-only MSD")
        fig.tight_layout()
        fig.savefig(output / "e4_minus_e3_visual_only.png", dpi=180)
        plt.close(fig)
    return {"status": "ok", "files": sorted(path.name for path in output.glob("*.png"))}


def build_report(rows: list[dict[str, Any]], output_dir: str | Path, *, bootstrap_repeats: int = 2000, seed: int = 0) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize_rows(rows, evidence_only=True, repeats=bootstrap_repeats, seed=seed)
    all_summaries = summarize_rows(rows, evidence_only=False, repeats=bootstrap_repeats, seed=seed)
    contrasts = contrast_rows(rows, repeats=bootstrap_repeats, seed=seed)
    permutations = permutation_rows(rows)
    figures = _make_figures(summaries, contrasts, output / "figures")
    primary_rows = [row for row in rows if str(row.get("permutation_id", "identity")) == "identity"]
    payload = {
        "raw_rows": len(rows),
        "evidence_rows": sum(row.get("evidence_eligible") is True for row in rows),
        "primary_identity_rows": len(primary_rows),
        "status_counts": {status: sum(row.get("status") == status for row in rows) for status in sorted({str(row.get("status")) for row in rows})},
        "summaries": summaries,
        "all_summaries": all_summaries,
        "contrasts": contrasts,
        "permutation_rows": permutations,
        "figures": figures,
        "prediction_source_guard": "Primary rows are eligible only when MSD ea_layer output is projected by base_model.lm_head and target logits are not used.",
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output / "condition_summary.csv", summaries)
    _write_csv(output / "condition_summary_all_rows.csv", all_summaries)
    _write_csv(output / "paired_contrasts.csv", contrasts)
    _write_csv(output / "label_permutation.csv", permutations)
    lines = [
        "# MSD Reverse Text-to-Visual Transfer Report",
        "",
        f"Raw rows: {len(rows)}; primary identity rows: {payload['primary_identity_rows']}; evidence-eligible rows: {payload['evidence_rows']}.",
        "",
        "Primary evidence uses only trained MSD draft hidden states from `model.ea_layer`, projected by the frozen target LM head. Target vocabulary logits are not used for primary scores.",
        "",
        "## E1-E6 condition summaries",
        "",
        "| Dataset | Task | Condition | Input order | MSD conditioning | N | Accuracy | Correct-option probability | NLL | 95% CI (accuracy) |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['task_type']} | {row['condition']} | {row['input_order']} | {row['msd_conditioning']} | {row['samples']} | {row['accuracy']} | {row['correct_option_probability']} | {row['nll']} | {row['accuracy_ci95']} |")
    lines.extend(["", "## Paired contrasts", "", "| Dataset | Task | Contrast | Metric | N groups | Estimate | 95% CI |", "|---|---|---|---|---:|---:|---|"])
    for row in contrasts:
        if row["task_type"] == "all" and row["metric"] == "accuracy":
            lines.append(f"| {row['dataset']} | all | {row['contrast']} | accuracy | {row['n_groups']} | {row['estimate']} | {row['ci95']} |")
    lines.extend(["", "## Label permutation", "", f"Permutation rows: {len(permutations)}.", "", "See `label_permutation.csv` for sample-level consistency.", "", "## Caveat", "", "For EAGLE-style MSD, the normal `msdgenerate` loop seeds its first emitted token from the target. This experiment instead uses the trained draft forward on the complete prompt context at the final prompt position; this is a valid MSD draft next-token state, but it is not a claim that normal speculative decoding independently proposes its first token.", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    payload = build_report(read_rows(args.input), args.output_dir, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed)
    print(json.dumps({"raw_rows": payload["raw_rows"], "evidence_rows": payload["evidence_rows"], "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
