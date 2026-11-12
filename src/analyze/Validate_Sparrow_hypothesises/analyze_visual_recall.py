"""Analyze Visual Attention Pareto Coverage and Draft Proposal Recall@K.

This module computes:
1. Attention Mass Pareto Coverage curve across percentiles of visual tokens.
2. Draft Proposal Recall@K (K=1, 2, 3, 5, 10) across slots and conditions from
   replay distribution records.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np


def compute_attention_pareto_coverage(
    scores: Union[Sequence[float], np.ndarray],
    percentiles: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0),
) -> Dict[float, float]:
    """Compute cumulative attention mass covered by top-P fraction of keys.

    Args:
        scores: 1D array of non-negative attention scores.
        percentiles: list of fractions in (0, 1.0] representing top token fractions.

    Returns:
        Dictionary mapping percentile to cumulative fraction of total attention mass.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {p: 0.0 for p in percentiles}
    total_mass = np.sum(arr)
    if total_mass <= 0:
        return {p: 0.0 for p in percentiles}

    # Sort descending
    sorted_scores = np.sort(arr)[::-1]
    cumsum = np.cumsum(sorted_scores)

    results: Dict[float, float] = {}
    n = len(sorted_scores)
    for p in percentiles:
        if p <= 0:
            results[p] = 0.0
            continue
        k = max(1, int(np.ceil(p * n)))
        k = min(k, n)
        covered = float(cumsum[k - 1] / total_mass)
        results[p] = covered
    return results


def compute_proposal_recall_at_k(
    target_ranks: Sequence[int],
    k_values: Sequence[int] = (1, 2, 3, 5, 10),
    one_indexed: bool = True,
) -> Dict[int, float]:
    """Compute Recall@K given ranks of target tokens.

    Args:
        target_ranks: Sequence of ranks (1-indexed if one_indexed=True, where 1 is top-1).
        k_values: Cutoff values K.
        one_indexed: If True, target is in Top-K when 1 <= rank <= K.
                     If False, target is in Top-K when 0 <= rank < K.

    Returns:
        Dictionary mapping K to proportion of positions where target was in top-K.
    """
    ranks = np.asarray(target_ranks)
    total = len(ranks)
    if total == 0:
        return {k: 0.0 for k in k_values}

    recalls: Dict[int, float] = {}
    for k in k_values:
        if one_indexed:
            hits = np.sum((ranks >= 1) & (ranks <= k))
        else:
            hits = np.sum((ranks >= 0) & (ranks < k))
        recalls[k] = float(hits / total)
    return recalls


def summarize_position_recalls(
    records: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (1, 2, 3, 5, 10),
    one_indexed: bool = True,
) -> Dict[str, Any]:
    """Group position records by draft slot and condition, computing recall metrics.

    Args:
        records: List of dicts with keys 'draft_slot_in_block', 'target_rank',
                 and optionally 'condition', 'budget'.
        k_values: Cutoff values.
        one_indexed: Whether target_rank is 1-indexed.

    Returns:
        Structured dictionary summarizing metrics overall and by slot.
    """
    by_slot: Dict[int, List[int]] = {}
    overall_ranks: List[int] = []

    for r in records:
        slot = int(r.get("draft_slot_in_block", 0))
        rank = r.get("target_rank")
        if rank is None:
            # Check boolean flags if rank is not explicit
            if r.get("target_hit_top_1"):
                rank = 1 if one_indexed else 0
            elif r.get("target_hit_top_2"):
                rank = 2 if one_indexed else 1
            elif r.get("target_hit_top_5"):
                rank = 5 if one_indexed else 4
            else:
                rank = 999999
        rank = int(rank)
        overall_ranks.append(rank)
        by_slot.setdefault(slot, []).append(rank)

    summary_by_slot: Dict[int, Dict[str, Any]] = {}
    for slot, ranks in sorted(by_slot.items()):
        recalls = compute_proposal_recall_at_k(ranks, k_values=k_values, one_indexed=one_indexed)
        summary_by_slot[slot] = {
            "count": len(ranks),
            **{f"recall_at_{k}": recalls[k] for k in k_values},
            "median_rank": float(np.median(ranks)) if ranks else 0.0,
            "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        }

    overall_recalls = compute_proposal_recall_at_k(overall_ranks, k_values=k_values, one_indexed=one_indexed)
    return {
        "total_positions": len(overall_ranks),
        "overall_recalls": {f"recall_at_{k}": overall_recalls[k] for k in k_values},
        "by_slot": summary_by_slot,
    }


def analyze_replay_pairs_directory(
    pairs_dir: Path,
    k_values: Sequence[int] = (1, 2, 3, 5, 10),
) -> Dict[str, Any]:
    """Load and analyze all replay pair JSONL.GZ files in a directory."""
    pair_files = list(pairs_dir.glob("*.jsonl.gz"))
    if not pair_files:
        raise FileNotFoundError(f"No .jsonl.gz files found in {pairs_dir}")

    full_records: List[Dict[str, Any]] = []
    cut_records: List[Dict[str, Any]] = []
    budget_records: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}

    for p in pair_files:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            meta_line = f.readline()
            if not meta_line:
                continue
            meta = json.loads(meta_line)
            budget = meta.get("target_visual_tokens", 0)
            if budget not in budget_records:
                budget_records[budget] = {"full": [], "cut": []}

            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") != "position":
                    continue
                slot = row.get("draft_slot_in_block", 0)

                # Process Full metrics
                full_m = row.get("full_metrics")
                if full_m:
                    rec_full = {
                        "draft_slot_in_block": slot,
                        "target_rank": full_m.get("target_rank"),
                        "target_hit_top_1": full_m.get("target_hit_top_1"),
                        "target_hit_top_2": full_m.get("target_hit_top_2"),
                        "target_hit_top_5": full_m.get("target_hit_top_5"),
                    }
                    full_records.append(rec_full)
                    budget_records[budget]["full"].append(rec_full)

                # Process Cut metrics
                cut_m = row.get("cut_metrics")
                if cut_m:
                    rec_cut = {
                        "draft_slot_in_block": slot,
                        "target_rank": cut_m.get("target_rank"),
                        "target_hit_top_1": cut_m.get("target_hit_top_1"),
                        "target_hit_top_2": cut_m.get("target_hit_top_2"),
                        "target_hit_top_5": cut_m.get("target_hit_top_5"),
                    }
                    cut_records.append(rec_cut)
                    budget_records[budget]["cut"].append(rec_cut)

    results: Dict[str, Any] = {
        "files_analyzed": len(pair_files),
        "full_overall": summarize_position_recalls(full_records, k_values=k_values),
        "cut_overall": summarize_position_recalls(cut_records, k_values=k_values),
        "by_budget": {},
    }

    for b, cond_map in sorted(budget_records.items()):
        results["by_budget"][b] = {
            "full": summarize_position_recalls(cond_map["full"], k_values=k_values),
            "cut": summarize_position_recalls(cond_map["cut"], k_values=k_values),
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-l18",
        type=Path,
        default=Path("markdown/Visualcut_retention_report/raw_reports/03_Target_Visual_Attention_L18_Attribution_Data.json"),
        help="Path to Target L18 Attention Attribution JSON",
    )
    parser.add_argument(
        "--replay-pairs-dir",
        type=Path,
        default=Path("results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/full_run/pairs"),
        help="Path to directory containing replay .jsonl.gz pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/visual_recall_analysis"),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analyze Attention Pareto Coverage
    attention_summary = {}
    if args.attention_l18.exists():
        with open(args.attention_l18, "r", encoding="utf-8") as f:
            l18_data = json.load(f)
        all_vis = l18_data.get("all_vis_scores", [])
        if all_vis:
            percentiles = [0.01, 0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 0.75, 1.0]
            coverage = compute_attention_pareto_coverage(all_vis, percentiles=percentiles)
            attention_summary = {
                "sample_count": len(all_vis),
                "pareto_coverage": coverage,
            }
            print("=== Attention Mass Pareto Coverage (L18 Visual Tokens) ===")
            for p, cov in coverage.items():
                print(f"  Top {p*100:5.1f}% Visual Tokens -> {cov*100:6.2f}% Target Attention Mass")

    # 2. Analyze Proposal Recall@K from Replay Pairs
    replay_summary = {}
    if args.replay_pairs_dir.exists():
        print(f"\nAnalyzing Replay Pairs in {args.replay_pairs_dir}...")
        replay_summary = analyze_replay_pairs_directory(args.replay_pairs_dir)
        full_ov = replay_summary["full_overall"]["overall_recalls"]
        cut_ov = replay_summary["cut_overall"]["overall_recalls"]
        print("=== Overall Proposal Recall ===")
        print(f"  Full: R@1={full_ov['recall_at_1']*100:.2f}%, R@2={full_ov['recall_at_2']*100:.2f}%, R@5={full_ov['recall_at_5']*100:.2f}%")
        print(f"  Cut:  R@1={cut_ov['recall_at_1']*100:.2f}%, R@2={cut_ov['recall_at_2']*100:.2f}%, R@5={cut_ov['recall_at_5']*100:.2f}%")

    out_file = args.output_dir / "visual_recall_summary.json"
    combined = {
        "attention_analysis": attention_summary,
        "proposal_recall_analysis": replay_summary,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved summary to {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
