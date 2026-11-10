"""Token-ID controlled Cut-vs-Full replay analysis on a selected VDC subset."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _read_pair(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = status = None
    positions: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") == "pair_meta":
                metadata = row
            elif row.get("record_type") == "pair_status":
                status = row
            elif row.get("record_type") == "position":
                positions.append(row)
    if metadata is None or status is None:
        raise ValueError(f"missing pair metadata/status: {path}")
    if status.get("status") != "ok" or not status.get("full_matches_target_greedy"):
        raise ValueError(f"invalid or unaligned replay pair: {path}")
    if int(status["position_count"]) != len(positions):
        raise ValueError(f"position count differs from status: {path}")
    return metadata, positions


def _matched_estimate(
    by_sample: dict[str, dict[tuple[int, int], dict[int, list[float]]]],
    sampled_ids: list[str],
    eligible_strata: list[tuple[int, int]],
) -> tuple[float | None, int]:
    multiplicity: dict[str, int] = defaultdict(int)
    for sample_id in sampled_ids:
        multiplicity[sample_id] += 1
    effects = []
    used = 0
    for stratum in eligible_strata:
        sums = {0: 0.0, 1: 0.0}
        counts = {0: 0, 1: 0}
        for sample_id, repeats in multiplicity.items():
            labels = by_sample.get(sample_id, {}).get(stratum, {})
            for label in (0, 1):
                values = labels.get(label, [])
                sums[label] += repeats * sum(values)
                counts[label] += repeats * len(values)
        if counts[0] and counts[1]:
            effects.append(sums[1] / counts[1] - sums[0] / counts[0])
            used += 1
    if not effects:
        return None, 0
    return fmean(effects), used


def _weighted_matched_estimate(
    by_sample: dict[str, dict[tuple[int, int], dict[int, list[float]]]],
    sample_ids: list[str],
    eligible_strata: list[tuple[int, int]],
    weights: dict[str, float],
) -> tuple[float | None, int]:
    effects = []
    for stratum in eligible_strata:
        sums = {0: 0.0, 1: 0.0}
        counts = {0: 0.0, 1: 0.0}
        for sample_id in sample_ids:
            labels = by_sample.get(sample_id, {}).get(stratum, {})
            weight = weights[sample_id]
            for label in (0, 1):
                values = labels.get(label, [])
                sums[label] += weight * sum(values)
                counts[label] += weight * len(values)
        if not counts[0] or not counts[1]:
            return None, len(effects)
        effects.append(sums[1] / counts[1] - sums[0] / counts[0])
    if not effects:
        return None, 0
    return fmean(effects), len(effects)


def _matched_group_means(
    by_sample: dict[str, dict[tuple[int, int], dict[int, list[float]]]],
    eligible_strata: list[tuple[int, int]],
) -> dict[str, float]:
    result = {}
    for label in (0, 1):
        stratum_means = []
        for stratum in eligible_strata:
            values = [
                value
                for sample_id in by_sample
                for value in by_sample[sample_id].get(stratum, {}).get(label, [])
            ]
            stratum_means.append(fmean(values))
        result[str(label)] = fmean(stratum_means)
    return result


def analyze(
    run_dir: Path,
    proxy_positions: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    budget: int = 3000,
    max_slot: int = 4,
    min_occurrences_per_label: int = 3,
    min_videos_per_label: int = 2,
    bootstrap_reps: int = 2000,
    seed: int = 20261105,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in selection["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample selection contains duplicate IDs")
    duration_groups = [
        [str(item["sample_id"]) for item in group["selected"]]
        for group in selection.get("groups", [])
    ]
    if not duration_groups:
        duration_groups = [sample_ids]
    grouped_ids = [sample_id for group in duration_groups for sample_id in group]
    if sorted(grouped_ids) != sorted(sample_ids):
        raise ValueError("duration groups must partition the selected sample IDs")

    proxy_label: dict[tuple[str, int, int, int], int] = {}
    with proxy_positions.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["budget"]) != budget or row["condition"] != "full":
                continue
            key = (
                str(row["sample_id"]),
                int(row["block_index"]),
                int(row["slot"]),
                int(row["target_position"]),
            )
            proxy_label[key] = int(row["reference_overlap_proxy"])

    by_sample: dict[str, dict[tuple[int, int], dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    raw_rows: list[dict[str, Any]] = []
    complete_blocks_by_sample: dict[str, int] = defaultdict(int)
    for path in sorted((run_dir / "pairs").glob("*.jsonl.gz")):
        metadata, positions = _read_pair(path)
        sample_id = str(metadata["sample_id"])
        if sample_id not in sample_ids or int(metadata["target_visual_tokens"]) != budget:
            continue
        by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in positions:
            by_block[int(row["block_index"])].append(row)
        nonterminal = sorted(by_block)[:-1]
        for block_id in nonterminal:
            rows = sorted(by_block[block_id], key=lambda row: int(row["draft_slot_in_block"]))
            if len(rows) != 15:
                continue
            complete_blocks_by_sample[sample_id] += 1
            for row in rows:
                slot = int(row["draft_slot_in_block"])
                if slot > max_slot:
                    continue
                position = int(row["target_position"])
                label_key = (sample_id, block_id, slot, position)
                if label_key not in proxy_label:
                    raise ValueError(f"missing lexical-proxy label for {label_key}")
                label = proxy_label[label_key]
                token_id = int(row["target_greedy_token_id"])
                full_miss = int(not bool(row["full_metrics"]["target_hit_top_1"]))
                cut_miss = int(not bool(row["cut_metrics"]["target_hit_top_1"]))
                delta = float(cut_miss - full_miss)
                stratum = (slot, token_id)
                by_sample[sample_id][stratum][label].append(delta)
                raw_rows.append(
                    {
                        "sample_id": sample_id,
                        "budget": budget,
                        "block_index": block_id,
                        "slot": slot,
                        "target_position": position,
                        "token_id": token_id,
                        "reference_overlap_proxy": label,
                        "full_miss": full_miss,
                        "cut_miss": cut_miss,
                        "cut_minus_full_miss": delta,
                    }
                )

    if set(complete_blocks_by_sample) != set(sample_ids):
        raise ValueError("selected sample set does not have complete pairs at the requested budget")
    totals_by_stratum: dict[tuple[int, int], dict[int, list[Any]]] = defaultdict(
        lambda: {0: [0, set()], 1: [0, set()]}
    )
    for sample_id, strata in by_sample.items():
        for stratum, labels in strata.items():
            for label, values in labels.items():
                totals_by_stratum[stratum][label][0] += len(values)
                totals_by_stratum[stratum][label][1].add(sample_id)
    eligible_strata = sorted(
        stratum
        for stratum, labels in totals_by_stratum.items()
        if all(
            labels[label][0] >= min_occurrences_per_label
            and len(labels[label][1]) >= min_videos_per_label
            for label in (0, 1)
        )
    )
    if not eligible_strata:
        raise ValueError("no token-ID x slot strata have the predeclared common support")

    point, strata_used = _matched_estimate(by_sample, sample_ids, eligible_strata)
    if point is None or strata_used != len(eligible_strata):
        raise ValueError("full-sample estimate does not cover all predeclared strata")
    rng = random.Random(seed)
    bootstrap = []
    used_strata_per_draw = []
    for _ in range(bootstrap_reps):
        weights = {}
        for group in duration_groups:
            raw_weights = [rng.expovariate(1.0) for _ in group]
            scale = len(group) / sum(raw_weights)
            weights.update(
                (sample_id, weight * scale)
                for sample_id, weight in zip(group, raw_weights)
            )
        estimate, used = _weighted_matched_estimate(
            by_sample, sample_ids, eligible_strata, weights
        )
        if estimate is None or used != len(eligible_strata):
            raise ValueError("positive-weight cluster bootstrap lost fixed common support")
        bootstrap.append(estimate)
        used_strata_per_draw.append(used)
    bootstrap.sort()
    lo = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    hi = bootstrap[int(0.975 * (len(bootstrap) - 1))]

    overlap_rows = [row for row in raw_rows if row["reference_overlap_proxy"] == 1]
    other_rows = [row for row in raw_rows if row["reference_overlap_proxy"] == 0]
    matched_keys = set(eligible_strata)
    matched_rows = [
        row for row in raw_rows if (row["slot"], row["token_id"]) in matched_keys
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "token_control_positions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)
    summary = {
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "budget": budget,
        "slots": list(range(1, max_slot + 1)),
        "complete_blocks": sum(complete_blocks_by_sample.values()),
        "position_count": len(raw_rows),
        "overlap_position_count": len(overlap_rows),
        "other_position_count": len(other_rows),
        "matched_position_count": len(matched_rows),
        "matched_position_count_by_label": {
            str(label): sum(
                row["reference_overlap_proxy"] == label for row in matched_rows
            )
            for label in (0, 1)
        },
        "matched_strata_count": len(eligible_strata),
        "macro_mean_cut_minus_full_mismatch_by_label": _matched_group_means(
            by_sample, eligible_strata
        ),
        "common_support_rule": {
            "stratum": "draft slot x target token ID",
            "minimum_positions_per_label": min_occurrences_per_label,
            "minimum_videos_per_label": min_videos_per_label,
        },
        "primary_metric": "macro mean across common-support strata of (Cut-Full mismatch)_overlap - (Cut-Full mismatch)_other",
        "interaction": point,
        "bootstrap_interval95": [lo, hi],
        "bootstrap_reps_requested": bootstrap_reps,
        "bootstrap_reps_valid": len(bootstrap),
        "bootstrap_method": "stratified video-cluster Bayesian bootstrap with positive Exp(1) weights normalized within duration quartile",
        "bootstrap_min_strata_used": min(used_strata_per_draw),
        "bootstrap_median_strata_used": sorted(used_strata_per_draw)[len(used_strata_per_draw) // 2],
        "bootstrap_seed": seed,
        "limitations": [
            f"exploratory replay on a deterministic {len(sample_ids)}-video duration-stratified subset",
            "reference-answer lexical overlap is an imperfect semantic label",
            "token-ID matching controls token identity and slot, but not all word or context difficulty",
            "fixed-trajectory replay does not establish free-running acceptance or latency",
        ],
    }
    (output_dir / "token_control_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--proxy-positions", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--max-slot", type=int, default=4)
    parser.add_argument("--min-occurrences-per-label", type=int, default=3)
    parser.add_argument("--min-videos-per-label", type=int, default=2)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    result = analyze(
        args.run_dir,
        args.proxy_positions,
        args.selection,
        args.output_dir,
        budget=args.budget,
        max_slot=args.max_slot,
        min_occurrences_per_label=args.min_occurrences_per_label,
        min_videos_per_label=args.min_videos_per_label,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
