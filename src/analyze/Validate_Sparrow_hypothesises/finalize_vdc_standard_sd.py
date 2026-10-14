"""Merge the VDC standard-SD shards and compare them with the existing MSD curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_ci(values: list[float], seed: int = 42, draws: int = 5000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def _ols_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else None


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = math.sqrt(sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys))
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom if denom else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def finalize(first: Path, second: Path, msd_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates = []
    for path in (first, second):
        for row in _read_jsonl(path):
            row_id = str(row.get("row_id"))
            if row_id in by_id:
                duplicates.append(row_id)
                if by_id[row_id].get("target_output_hash") != row.get("target_output_hash"):
                    raise RuntimeError(f"conflicting duplicate row: {row_id}")
                continue
            by_id[row_id] = row

    rows = list(by_id.values())
    rows.sort(key=lambda row: (
        str(row.get("sample_id")),
        str(row.get("selection_policy")),
        -float(row.get("retention_percentage", 0)),
    ))
    merged_path = output_dir / "raw_results.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    policies = sorted({str(row["selection_policy"]) for row in rows if row.get("status") == "ok"})
    retentions = sorted({float(row["retention_percentage"]) for row in rows})
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(str(row["selection_policy"]), float(row["retention_percentage"]))].append(row)

    standard_table: list[dict[str, Any]] = []
    for policy in policies:
        for retention in sorted(retentions, reverse=True):
            values = groups[(policy, retention)]
            tau = [float(row["tau_proposal"]) for row in values]
            effective = [float(row["tau_effective"]) for row in values]
            ci_low, ci_high = _bootstrap_ci(tau, seed=42 + int(retention))
            standard_table.append({
                "method": "standard_sd",
                "policy": policy,
                "retention_percentage": retention,
                "samples": len(values),
                "lossless_rate": _mean([float(bool(row["lossless"])) for row in values]),
                "mean_tau_proposal": _mean(tau),
                "tau_proposal_ci95_low": ci_low,
                "tau_proposal_ci95_high": ci_high,
                "mean_tau_effective": _mean(effective),
                "mean_draft_visual_tokens": _mean([float(row["draft_visual_tokens"]) for row in values]),
                "mean_output_tokens": _mean([float(len(row["target_output_ids"])) for row in values]),
                "mean_decode_speedup": _mean([float(row["decode_speedup"]) for row in values]),
                "mean_end_to_end_speedup": _mean([float(row["end_to_end_speedup"]) for row in values]),
            })

    msd_table: list[dict[str, Any]] = []
    with msd_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            policy = row.get("selection_policy", "")
            if policy not in policies:
                continue
            msd_table.append({
                "method": "msd_existing",
                "policy": policy,
                "retention_percentage": float(row["retention_percentage"]),
                "samples": int(float(row.get("sample_count", row.get("rows", 0)))),
                "lossless_rate": float(row["lossless.mean"]),
                "mean_tau_proposal": float(row["accepted_prefix_tokens.mean"]),
                "tau_proposal_ci95_low": float(row["accepted_prefix_tokens.ci95_low"]),
                "tau_proposal_ci95_high": float(row["accepted_prefix_tokens.ci95_high"]),
            })

    comparison = standard_table + msd_table
    with (output_dir / "curve_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in comparison for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison)

    trend_rows = []
    for policy in policies:
        values = [row for row in standard_table if row["policy"] == policy]
        values.sort(key=lambda row: row["retention_percentage"])
        xs = [float(row["retention_percentage"]) for row in values]
        ys = [float(row["mean_tau_proposal"]) for row in values]
        trend_rows.append({
            "method": "standard_sd",
            "policy": policy,
            "slope_tau_per_retention_pct": _ols_slope(xs, ys),
            "pearson_retention_tau": _corr(xs, ys),
            "spearman_retention_tau": _corr(_rank(xs), _rank(ys)),
            "mean_tau_at_0": ys[0] if ys else None,
            "mean_tau_at_100": ys[-1] if ys else None,
        })
    for policy in policies:
        values = [row for row in msd_table if row["policy"] == policy]
        values.sort(key=lambda row: row["retention_percentage"])
        xs = [float(row["retention_percentage"]) for row in values]
        ys = [float(row["mean_tau_proposal"]) for row in values]
        trend_rows.append({
            "method": "msd_existing",
            "policy": policy,
            "slope_tau_per_retention_pct": _ols_slope(xs, ys),
            "pearson_retention_tau": _corr(xs, ys),
            "spearman_retention_tau": _corr(_rank(xs), _rank(ys)),
            "mean_tau_at_0": next((y for x, y in zip(xs, ys) if x == 0), None),
            "mean_tau_at_100": next((y for x, y in zip(xs, ys) if x == 100), None),
        })

    summary = {
        "rows": len(rows),
        "expected_rows": len(policies) * len(retentions) * len({row["sample_id"] for row in rows}),
        "samples": sorted({row["sample_id"] for row in rows}),
        "policies": policies,
        "retentions": retentions,
        "status_counts": {status: sum(row.get("status") == status for row in rows) for status in sorted({row.get("status") for row in rows})},
        "lossless_count": sum(bool(row.get("lossless")) for row in rows),
        "duplicate_rows_removed": len(duplicates),
        "standard_curve": standard_table,
        "existing_msd_curve": msd_table,
        "trend": trend_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VDC subset: standard speculative decoding visual-retention study",
        "",
        "## Protocol",
        "",
        "This is an independent speculative-decoding control: `Qwen/Qwen2.5-VL-3B-Instruct` is the draft and `Qwen/Qwen2.5-VL-7B-Instruct` is the target. The target always receives the full video context; only the draft visual context is reduced. Sequential target verification is used to enforce exact greedy losslessness.",
        "",
        f"- Samples: {len(summary['samples'])} VDC videos",
        f"- Conditions: {len(rows)} unique rows ({len(policies)} policies × {len(retentions)} retentions × {len(summary['samples'])} samples)",
        f"- Lossless rows: {summary['lossless_count']}/{len(rows)}",
        "- Retentions: 0, 1, 5, 10, 25, 50, 75, 100%",
        "- Policies: last-instruction query and all-text queries",
        "",
        "## Standard-SD curve",
        "",
        "`tau_proposal` is the mean number of draft proposals accepted per verification round. The bootstrap interval is over videos, not decoding rounds.",
        "",
        "| Policy | Retention | n | Lossless | τ proposal (95% bootstrap CI) | τ effective | Draft visual tokens | E2E speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in standard_table:
        lines.append(
            f"| {row['policy']} | {row['retention_percentage']:g}% | {row['samples']} | "
            f"{row['lossless_rate']:.0%} | {_fmt(row['mean_tau_proposal'])} "
            f"({_fmt(row['tau_proposal_ci95_low'])}, {_fmt(row['tau_proposal_ci95_high'])}) | "
            f"{_fmt(row['mean_tau_effective'])} | {_fmt(row['mean_draft_visual_tokens'], 1)} | "
            f"{_fmt(row['mean_end_to_end_speedup'])} |"
        )
    lines.extend(["", "## Retention trend", "", "A Sparrow-like generic SD effect would have τ increasing as retention decreases, i.e. a negative slope of τ against retention percentage.", "", "| Method | Policy | Slope τ/% | Pearson | Spearman | τ at 0% | τ at 100% |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in trend_rows:
        lines.append(
            f"| {row['method']} | {row['policy']} | {_fmt(row['slope_tau_per_retention_pct'], 5)} | "
            f"{_fmt(row['pearson_retention_tau'])} | {_fmt(row['spearman_retention_tau'])} | "
            f"{_fmt(row['mean_tau_at_0'])} | {_fmt(row['mean_tau_at_100'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The standard-SD control is considered consistent with Sparrow only if the negative retention–τ trend is reproducible while all rows remain lossless. The existing MSD CSV is included as a directional reference; it uses a different model family/checkpoint and should not be pooled with this VDC control.",
        "",
        "The primary test is therefore the sign and uncertainty of the standard-SD trend, not an absolute comparison of τ between MSD and standard SD. A flat or positive slope supports the hypothesis that the retention penalty is not generic to speculative decoding and may be specific to MSD or its visual-state interface.",
        "",
        "## Artifacts",
        "",
        "- `raw_results.jsonl`: 160 deduplicated row-level results",
        "- `curve_comparison.csv`: standard-SD and existing-MSD curve values",
        "- `summary.json`: counts, bootstrap intervals, and trend statistics",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--msd-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.first, args.second, args.msd_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
