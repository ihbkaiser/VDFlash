"""Aggregate paired distribution-replay records with sample-level bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from statistics import fmean
from typing import Any


CONDITION_METRICS = (
    "target_hit_top_1",
    "target_hit_top_2",
    "target_hit_top_5",
    "target_hit_top_10",
    "target_rank",
    "target_log_probability_nats",
    "target_probability",
    "top1_probability",
    "top2_probability",
    "top1_logit_margin",
    "entropy_bits",
    "union_top100_mass",
    "tail_mass_outside_union_top100",
)
PAIR_METRICS = (
    "jsd_bits_full_vocab",
    "jsd_bits_head_union_top100",
    "top1_changed",
)
OVERLAP_K = (1, 2, 5, 10, 50, 100)


def _read_pair(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    metadata = next((row for row in rows if row.get("record_type") == "pair_meta"), None)
    status = next((row for row in rows if row.get("record_type") == "pair_status"), None)
    positions = [row for row in rows if row.get("record_type") == "position"]
    if metadata is None or status is None:
        raise ValueError(f"pair artifact is incomplete: {path}")
    if status.get("status") != "ok":
        raise ValueError(f"pair status is not ok in {path}: {status.get('status')}")
    if int(status.get("position_count", -1)) != len(positions):
        raise ValueError(f"position count mismatch in {path}")
    if any(row.get("paired_state") != "ok" for row in positions):
        raise ValueError(f"unpaired position found in {path}")
    return metadata, positions, status


def _mean_position_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for name in CONDITION_METRICS:
            for condition in ("full", "cut"):
                values.setdefault(f"{condition}.{name}", []).append(
                    float(row[f"{condition}_metrics"][name])
                )
        for name in PAIR_METRICS:
            values.setdefault(f"pair.{name}", []).append(
                float(row["pair_metrics"][name])
            )
        for k in OVERLAP_K:
            overlap = row["pair_metrics"]["topk_overlap"][str(k)]
            for measure in ("fraction_of_k", "jaccard", "intersection_count"):
                values.setdefault(f"pair.top{k}_{measure}", []).append(
                    float(overlap[measure])
                )
    return {name: fmean(sample_values) for name, sample_values in values.items()}


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile from an empty list")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_interval(
    values: list[float],
    draws: list[list[int]],
) -> tuple[float, float]:
    estimates = sorted(fmean(values[index] for index in draw) for draw in draws)
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _aggregate(
    run_dir: Path,
    *,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    selected_ids = [str(value) for value in run_config["selected_sample_ids"]]
    budgets = [int(value) for value in run_config["budgets"]]
    expected = {(sample_id, budget) for sample_id in selected_ids for budget in budgets}
    sample_means: dict[int, dict[str, dict[str, float]]] = {budget: {} for budget in budgets}
    pair_statuses: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "pairs").glob("*.jsonl.gz")):
        metadata, positions, status = _read_pair(path)
        sample_id = str(metadata["sample_id"])
        budget = int(metadata["target_visual_tokens"])
        key = (sample_id, budget)
        if key not in expected:
            raise ValueError(f"unexpected pair in run artifact: {key}")
        if sample_id in sample_means[budget]:
            raise ValueError(f"duplicate pair artifact: {key}")
        if not positions:
            raise ValueError(f"pair has no draft proposal positions: {path}")
        sample_means[budget][sample_id] = _mean_position_values(positions)
        pair_statuses[f"{sample_id}:{budget}"] = {
            "status": status["status"],
            "position_count": len(positions),
            "target_greedy_ids_sha256": metadata["target_greedy_ids_sha256"],
            "target_input_fingerprint": metadata["target_input_fingerprint"],
        }
    observed = {
        (sample_id, budget)
        for budget, samples in sample_means.items()
        for sample_id in samples
    }
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"run is incomplete; missing {len(missing)} pairs, first: {missing[:5]}")
    if any(set(sample_means[budget]) != set(selected_ids) for budget in budgets):
        raise ValueError("sample cohort differs between visual budgets")

    sample_count = len(selected_ids)
    rng = random.Random(seed)
    draws = [
        [rng.randrange(sample_count) for _ in range(sample_count)]
        for _ in range(bootstrap_reps)
    ]
    all_value_names = sorted(
        next(iter(sample_means[budgets[0]].values())).keys()
    )
    summary_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for budget in budgets:
        for sample_id in selected_ids:
            sample_values = sample_means[budget][sample_id]
            for name, value in sample_values.items():
                per_sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "target_visual_tokens": budget,
                        "metric": name,
                        "sample_mean": value,
                    }
                )
        for name in all_value_names:
            values = [sample_means[budget][sample_id][name] for sample_id in selected_ids]
            if name.startswith("pair."):
                low, high = _bootstrap_interval(values, draws)
                summary_rows.append(
                    {
                        "target_visual_tokens": budget,
                        "metric": name.removeprefix("pair."),
                        "full_mean": None,
                        "cut_mean": None,
                        "delta_cut_minus_full_mean": None,
                        "mean": fmean(values),
                        "ci95_low": low,
                        "ci95_high": high,
                        "full_ci95_low": None,
                        "full_ci95_high": None,
                        "cut_ci95_low": None,
                        "cut_ci95_high": None,
                        "sample_count": sample_count,
                    }
                )
                continue
            condition, metric = name.split(".", 1)
            condition_values = [
                sample_means[budget][sample_id][name] for sample_id in selected_ids
            ]
            delta_values = [
                sample_means[budget][sample_id][f"cut.{metric}"]
                - sample_means[budget][sample_id][f"full.{metric}"]
                for sample_id in selected_ids
            ]
            low, high = _bootstrap_interval(delta_values, draws)
            if condition == "full":
                cut_values = [
                    sample_means[budget][sample_id][f"cut.{metric}"]
                    for sample_id in selected_ids
                ]
                full_low, full_high = _bootstrap_interval(condition_values, draws)
                cut_low, cut_high = _bootstrap_interval(cut_values, draws)
                summary_rows.append(
                    {
                        "target_visual_tokens": budget,
                        "metric": metric,
                        "full_mean": fmean(condition_values),
                        "cut_mean": fmean(cut_values),
                        "delta_cut_minus_full_mean": fmean(delta_values),
                        "mean": None,
                        "ci95_low": low,
                        "ci95_high": high,
                        "full_ci95_low": full_low,
                        "full_ci95_high": full_high,
                        "cut_ci95_low": cut_low,
                        "cut_ci95_high": cut_high,
                        "sample_count": sample_count,
                    }
                )

    return {
        "schema_version": 1,
        "run_id": run_config["run_id"],
        "run_config": str(run_config_path),
        "bootstrap": {
            "unit": "sample_id",
            "replicates": bootstrap_reps,
            "confidence_level": 0.95,
            "interval": "percentile",
            "seed": seed,
            "sample_count": sample_count,
            "same_sample_resamples_reused_across_budgets": True,
            "position_aggregation": "mean_within_sample_before_bootstrap",
        },
        "budgets": budgets,
        "selected_sample_ids": selected_ids,
        "pair_count": len(observed),
        "expected_pair_count": len(expected),
        "pair_statuses": pair_statuses,
        "summary_rows": summary_rows,
        "sample_mean_rows": per_sample_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "target_visual_tokens",
        "metric",
        "full_mean",
        "cut_mean",
        "delta_cut_minus_full_mean",
        "mean",
        "ci95_low",
        "ci95_high",
        "full_ci95_low",
        "full_ci95_high",
        "cut_ci95_low",
        "cut_ci95_high",
        "sample_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_reps < 1:
        raise SystemExit("bootstrap-reps must be positive")
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    result = _aggregate(
        run_dir,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "summary.csv", result["summary_rows"])
    with (output_dir / "sample_means.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("sample_id", "target_visual_tokens", "metric", "sample_mean")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["sample_mean_rows"])
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "pair_count": result["pair_count"],
                "summary_json": str(output_dir / "summary.json"),
                "summary_csv": str(output_dir / "summary.csv"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
