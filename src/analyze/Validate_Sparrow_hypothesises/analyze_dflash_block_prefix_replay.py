"""Analyze contiguous DFlash proposal prefixes in a paired distribution replay."""

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


def _read_pair(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metadata = status = None
    positions: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "pair_meta":
                metadata = row
            elif row.get("record_type") == "pair_status":
                status = row
            elif row.get("record_type") == "position":
                positions.append(row)
    if metadata is None or status is None:
        raise ValueError(f"missing pair metadata/status: {path}")
    if status.get("status") != "ok" or int(status["position_count"]) != len(positions):
        raise ValueError(f"invalid pair status or position count: {path}")
    if not status.get("full_matches_target_greedy"):
        raise ValueError(f"Full output does not match target trajectory: {path}")
    if any(row.get("paired_state") != "ok" for row in positions):
        raise ValueError(f"unpaired draft position: {path}")
    return metadata, positions, status


def _prefix_length(rows: list[dict[str, Any]], condition: str) -> int:
    prefix = 0
    for row in rows:
        if not bool(row[f"{condition}_metrics"]["target_hit_top_1"]):
            break
        prefix += 1
    return prefix


def _bootstrap_ci(
    values: list[float], draws: list[list[int]], confidence: float = 0.95
) -> list[float]:
    estimates = sorted(fmean(values[index] for index in draw) for draw in draws)

    def percentile(q: float) -> float:
        position = (len(estimates) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        fraction = position - lower
        return estimates[lower] * (1.0 - fraction) + estimates[upper] * fraction

    tail = (1.0 - confidence) / 2.0
    return [percentile(tail), percentile(1.0 - tail)]


def analyze(
    run_dir: Path,
    output_dir: Path,
    *,
    block_size: int = 16,
    bootstrap_reps: int = 2000,
    seed: int = 20261105,
) -> dict[str, Any]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in config["selected_sample_ids"]]
    budgets = [int(value) for value in config["budgets"]]
    expected_pairs = {(sample_id, budget) for sample_id in sample_ids for budget in budgets}
    pair_data: dict[tuple[str, int], dict[str, Any]] = {}
    position_counts: dict[int, int] = defaultdict(int)
    anchor_checks = 0
    pair_checks = 0
    max_tau_error = 0.0

    for path in sorted((run_dir / "pairs").glob("*.jsonl.gz")):
        metadata, positions, status = _read_pair(path)
        sample_id = str(metadata["sample_id"])
        budget = int(metadata["target_visual_tokens"])
        key = (sample_id, budget)
        if key not in expected_pairs or key in pair_data:
            raise ValueError(f"unexpected or duplicate pair: {key}")
        position_counts[budget] += len(positions)
        by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in positions:
            by_block[int(row["block_index"])].append(row)
        block_ids = sorted(by_block)
        block_rows = {
            block_id: sorted(
                by_block[block_id], key=lambda row: int(row["draft_slot_in_block"])
            )
            for block_id in block_ids
        }
        nonterminal_ids = block_ids[:-1]
        if not nonterminal_ids:
            raise ValueError(f"no nonterminal blocks for {key}")

        # The replay artifact is truncated at the target trajectory boundary.
        # Recover Full's exact acceptance from the next anchor, and compare
        # Full/Cut only when all draft slots have paired target positions.
        full_prefix_all_rounds: list[int] = []
        for block_id in nonterminal_ids:
            rows = block_rows[block_id]
            next_block_id = block_id + 1
            if next_block_id not in block_rows:
                raise ValueError(f"missing sequential next block for {key}, block {block_id}")
            anchor = int(rows[0]["block_anchor_position"])
            next_anchor = int(block_rows[next_block_id][0]["block_anchor_position"])
            actual_accepts = next_anchor - anchor - 1
            if not 0 <= actual_accepts <= block_size - 1:
                raise ValueError(f"invalid Full anchor advance for {key}, block {block_id}")
            observed_prefix = _prefix_length(rows, "full")
            if observed_prefix != min(actual_accepts, len(rows)):
                raise ValueError(
                    f"Full target-hit prefix disagrees with anchor advance for {key}, "
                    f"block {block_id}: observed={observed_prefix}, accepted={actual_accepts}"
                )
            full_prefix_all_rounds.append(actual_accepts)
            anchor_checks += 1

        recorded_tau = float(status["full_metrics"]["tau_proposal"])
        reconstructed_tau = fmean(full_prefix_all_rounds)
        error = abs(recorded_tau - reconstructed_tau)
        max_tau_error = max(max_tau_error, error)
        pair_checks += 1
        if error > 1e-6:
            raise ValueError(
                f"reconstructed Full tau_proposal differs for {key}: "
                f"{reconstructed_tau} vs {recorded_tau}"
            )

        complete_ids = [
            block_id
            for block_id in nonterminal_ids
            if len(block_rows[block_id]) == block_size - 1
        ]
        if not complete_ids:
            raise ValueError(f"no fully observed paired replay blocks for {key}")

        per_condition: dict[str, list[int]] = {"full": [], "cut": []}
        survival: dict[str, list[list[int]]] = {"full": [], "cut": []}
        first_reject: dict[str, list[int]] = {"full": [], "cut": []}
        for block_id in complete_ids:
            rows = block_rows[block_id]
            if [int(row["draft_slot_in_block"]) for row in rows] != list(
                range(1, block_size)
            ):
                raise ValueError(f"invalid slot order for {key}, block {block_id}")
            next_anchor = int(block_rows[block_id + 1][0]["block_anchor_position"])
            full_prefix = next_anchor - int(rows[0]["block_anchor_position"]) - 1
            prefixes = {"full": full_prefix, "cut": _prefix_length(rows, "cut")}
            for condition in ("full", "cut"):
                prefix = prefixes[condition]
                matches = [
                    bool(row[f"{condition}_metrics"]["target_hit_top_1"])
                    for row in rows
                ]
                prefixes[condition] = prefix
                per_condition[condition].append(prefix)
                survival[condition].append(
                    [int(all(matches[:slot])) for slot in range(1, block_size)]
                )
                first_reject[condition].append(
                    prefix + 1 if prefix < block_size - 1 else block_size
                )
        pair_data[key] = {
            "complete_blocks": len(complete_ids),
            "nonterminal_rounds": len(nonterminal_ids),
            "prefix_mean": {name: fmean(values) for name, values in per_condition.items()},
            "full_true_bonus_count": sum(
                prefix == block_size - 1 for prefix in full_prefix_all_rounds
            ),
            "survival": {
                name: [fmean(block[slot] for block in blocks) for slot in range(block_size - 1)]
                for name, blocks in survival.items()
            },
            "first_reject": {
                name: [
                    sum(slot == event_slot for slot in slots) / len(slots)
                    for event_slot in range(1, block_size)
                ]
                for name, slots in first_reject.items()
            },
            "recorded_tau_proposal": recorded_tau,
        }

    if set(pair_data) != expected_pairs:
        missing = sorted(expected_pairs - set(pair_data))
        extra = sorted(set(pair_data) - expected_pairs)
        raise ValueError(f"incomplete pair grid: missing={missing[:5]}, extra={extra[:5]}")

    rng = random.Random(seed)
    draws = [
        [rng.randrange(len(sample_ids)) for _ in sample_ids]
        for _ in range(bootstrap_reps)
    ]
    summary = []
    for budget in budgets:
        sample_values = {
            condition: [pair_data[(sample_id, budget)]["prefix_mean"][condition] for sample_id in sample_ids]
            for condition in ("full", "cut")
        }
        deltas = [cut - full for full, cut in zip(sample_values["full"], sample_values["cut"], strict=True)]
        bonus_fractions = [
            pair_data[(sample_id, budget)]["full_true_bonus_count"]
            / pair_data[(sample_id, budget)]["nonterminal_rounds"]
            for sample_id in sample_ids
        ]
        slot_metrics = []
        for slot in range(block_size - 1):
            slot_metrics.append(
                {
                    "slot": slot + 1,
                    **{
                        f"{condition}_survival_mean_ci95": [
                            fmean(pair_data[(sample_id, budget)]["survival"][condition][slot] for sample_id in sample_ids),
                            _bootstrap_ci(
                                [pair_data[(sample_id, budget)]["survival"][condition][slot] for sample_id in sample_ids],
                                draws,
                            ),
                        ]
                        for condition in ("full", "cut")
                    },
                    **{
                        f"{condition}_first_reject_fraction_ci95": [
                            fmean(pair_data[(sample_id, budget)]["first_reject"][condition][slot] for sample_id in sample_ids),
                            _bootstrap_ci(
                                [pair_data[(sample_id, budget)]["first_reject"][condition][slot] for sample_id in sample_ids],
                                draws,
                            ),
                        ]
                        for condition in ("full", "cut")
                    },
                }
            )
        summary.append(
            {
                "budget": budget,
                "sample_count": len(sample_ids),
                "position_count": position_counts[budget],
                "complete_block_count": sum(
                    pair_data[(sample_id, budget)]["complete_blocks"] for sample_id in sample_ids
                ),
                "nonterminal_round_count": sum(
                    pair_data[(sample_id, budget)]["nonterminal_rounds"] for sample_id in sample_ids
                ),
                "mean_complete_blocks_per_video": fmean(
                    pair_data[(sample_id, budget)]["complete_blocks"] for sample_id in sample_ids
                ),
                "mean_prefix_full": fmean(sample_values["full"]),
                "mean_prefix_full_ci95": _bootstrap_ci(sample_values["full"], draws),
                "mean_prefix_cut_forced_replay": fmean(sample_values["cut"]),
                "mean_prefix_cut_ci95": _bootstrap_ci(sample_values["cut"], draws),
                "delta_cut_minus_full": fmean(deltas),
                "delta_ci95": _bootstrap_ci(deltas, draws),
                "full_true_bonus_count": sum(
                    pair_data[(sample_id, budget)]["full_true_bonus_count"]
                    for sample_id in sample_ids
                ),
                "full_correction_count": sum(
                    pair_data[(sample_id, budget)]["nonterminal_rounds"]
                    - pair_data[(sample_id, budget)]["full_true_bonus_count"]
                    for sample_id in sample_ids
                ),
                "full_true_bonus_fraction": fmean(bonus_fractions),
                "full_true_bonus_fraction_ci95": _bootstrap_ci(bonus_fractions, draws),
                "full_correction_fraction": 1.0 - fmean(bonus_fractions),
                "logged_tau_proposal_full_mean": fmean(
                    pair_data[(sample_id, budget)]["recorded_tau_proposal"]
                    for sample_id in sample_ids
                ),
                "slot_metrics": slot_metrics,
            }
        )

    result = {
        "run_id": config["run_id"],
        "analysis": "contiguous top-1 proposal prefix on fully observed, nonterminal replay blocks",
        "primary_metric": "mean accepted-proposal prefix per fully observed block, averaged within video",
        "cut_interpretation": "forced replay on Full anchors; not free-running Cut acceptance",
        "semantic_token_labels": "not available in replay artifact",
        "bootstrap": {
            "unit": "video",
            "replicates": bootstrap_reps,
            "seed": seed,
            "ci": "percentile 95%",
            "shared_resamples_across_budgets": True,
        },
        "audit": {
            "expected_pairs": len(expected_pairs),
            "observed_pairs": len(pair_data),
            "anchor_advance_checks": anchor_checks,
            "full_tau_reconstruction_checks": pair_checks,
            "max_full_tau_abs_error": max_tau_error,
        },
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "budget",
                "videos",
                "positions",
                "complete_blocks",
                "full_prefix_mean",
                "cut_forced_prefix_mean",
                "cut_minus_full",
                "delta_ci95_low",
                "delta_ci95_high",
                "full_true_bonus_count",
                "full_correction_count",
                "full_true_bonus_fraction",
                "full_correction_fraction",
                "logged_full_tau_proposal_mean",
            ]
        )
        for row in summary:
            writer.writerow(
                [
                    row["budget"],
                    row["sample_count"],
                    row["position_count"],
                    row["complete_block_count"],
                    row["mean_prefix_full"],
                    row["mean_prefix_cut_forced_replay"],
                    row["delta_cut_minus_full"],
                    *row["delta_ci95"],
                    row["full_true_bonus_count"],
                    row["full_correction_count"],
                    row["full_true_bonus_fraction"],
                    row["full_correction_fraction"],
                    row["logged_tau_proposal_full_mean"],
                ]
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    result = analyze(
        args.run_dir,
        args.output_dir,
        block_size=args.block_size,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "summary"}, indent=2))
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "slot_metrics"}
                for row in result["summary"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
