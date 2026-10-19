"""Analyze free-running coherence and verified acceptance journals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.analyze.Validate_Sparrow_hypothesises.free_running_experiment import (
    CONDITION_NAMES,
    RETENTION_PERCENTAGES,
    build_blind_pairs,
    build_matched_prefix_pairs,
    read_jsonl,
    protocol_fingerprint,
    summarize_judge_pairs,
    summarize_rollouts,
    validate_paired_condition_rows,
    write_blinded_review_files,
    write_grounding_review_files,
)
from src.workspace import resolve_workspace_path


def _latest_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(key) or "")
        if not identifier:
            raise ValueError(f"journal row is missing {key}")
        latest[identifier] = dict(row)
    return latest


def _rows_for_protocol(
    rows: Sequence[Mapping[str, Any]], *, protocol_sha256: str, key: str
) -> tuple[list[dict[str, Any]], list[str]]:
    current = []
    mismatched = []
    for row in rows:
        identifier = str(row.get(key) or "")
        if row.get("protocol_sha256") == protocol_sha256:
            current.append(dict(row))
        else:
            mismatched.append(identifier)
    return current, sorted(mismatched)


def _mean_ci(values: Sequence[float], *, seed: int, replicates: int) -> tuple[float, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(replicates, len(data)))
    low, high = np.quantile(data[indices].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def _human_calibration(
    csv_path: Path,
    judge_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not csv_path.is_file():
        return None
    human_winners: list[str] = []
    judge_winners: list[str] = []
    dimension_differences: dict[str, tuple[list[float], list[float]]] = {
        name: ([], []) for name in ("grammar", "reference", "discourse", "repetition", "overall")
    }
    reviewed = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = str(row.get("review_id", ""))
            if not pair_id or not row.get("human_winner", "").strip():
                continue
            winner = row["human_winner"].strip().lower()
            if winner not in {"a", "b", "tie"}:
                raise ValueError(f"invalid human_winner for review {pair_id}: {winner}")
            model_row = judge_rows.get(pair_id)
            if model_row is None or model_row.get("status") != "ok":
                continue
            human_winners.append(winner)
            judge_winners.append(str(model_row["judge"]["winner"]).lower())
            reviewed += 1
            for dimension, (human_values, model_values) in dimension_differences.items():
                try:
                    human_a = float(row[f"human_{dimension}_a"])
                    human_b = float(row[f"human_{dimension}_b"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (1 <= human_a <= 5 and 1 <= human_b <= 5):
                    raise ValueError(f"human {dimension} scores must be in [1, 5]: {pair_id}")
                model = model_row["judge"]["scores"]
                human_values.append(human_a - human_b)
                model_values.append(
                    float(model["A"][dimension] - model["B"][dimension])
                )

    if not human_winners:
        return {"status": "awaiting_human_labels", "reviewed_pairs": 0}
    agreement = sum(left == right for left, right in zip(human_winners, judge_winners)) / len(human_winners)
    categories = ("a", "b", "tie")
    expected = sum(
        (human_winners.count(category) / len(human_winners))
        * (judge_winners.count(category) / len(judge_winners))
        for category in categories
    )
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else None
    correlations = {}
    for dimension, (human, model) in dimension_differences.items():
        correlation = None
        if len(human) >= 3 and np.std(human) > 0 and np.std(model) > 0:
            correlation = float(np.corrcoef(human, model)[0, 1])
        correlations[dimension] = {"paired_scores": len(human), "pearson_r": correlation}
    return {
        "status": "scored",
        "reviewed_pairs": reviewed,
        "winner_exact_agreement": agreement,
        "winner_cohen_kappa": kappa,
        "score_difference_correlations": correlations,
        "interpretation": "Small blinded calibration sample; report descriptively and do not treat it as validation on its own.",
    }


def _grounding_review_summary(
    csv_path: Path,
    blind_key: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any] | None:
    if not csv_path.is_file():
        return None
    retentions = {"retain_75": 75, "retain_50": 50, "retain_25": 25, "cut": 0}
    effects: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"groundedness": [], "relevance": []})
    )
    completed = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_id = str(row.get("review_id", ""))
            key = blind_key.get(pair_id)
            if key is None:
                continue
            values: dict[str, dict[str, float]] = {}
            for dimension in ("groundedness", "relevance"):
                try:
                    score_a = float(row[f"{dimension}_a"])
                    score_b = float(row[f"{dimension}_b"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (1 <= score_a <= 5 and 1 <= score_b <= 5):
                    raise ValueError(f"{dimension} ratings must be in [1, 5]: {pair_id}")
                values[dimension] = {"a": score_a, "b": score_b}
            if not values:
                continue
            completed += 1
            a_condition = str(key["a_condition"])
            b_condition = str(key["b_condition"])
            if a_condition == "full":
                reduced_condition, reduced_side, full_side = b_condition, "b", "a"
            elif b_condition == "full":
                reduced_condition, reduced_side, full_side = a_condition, "a", "b"
            else:
                continue
            retention = retentions.get(reduced_condition)
            if retention is None:
                continue
            sample_id = str(key["sample_id"])
            for dimension, scores in values.items():
                effects[retention][sample_id][dimension].append(
                    scores[reduced_side] - scores[full_side]
                )

    conditions = []
    for retention in (75, 50, 25, 0):
        sample_values = effects.get(retention, {})
        result = {
            "retention_percentage": retention,
            "condition": CONDITION_NAMES[retention],
            "paired_samples": 0,
        }
        for dimension in ("groundedness", "relevance"):
            per_sample = [
                float(np.mean(scores[dimension]))
                for scores in sample_values.values() if scores[dimension]
            ]
            ci = _mean_ci(
                per_sample,
                seed=seed + 3000 + retention + int(dimension == "relevance"),
                replicates=bootstrap_replicates,
            )
            result[f"mean_{dimension}_delta_vs_full"] = float(np.mean(per_sample)) if per_sample else None
            result[f"{dimension}_delta_ci95_low"] = ci[0] if ci else None
            result[f"{dimension}_delta_ci95_high"] = ci[1] if ci else None
            result["paired_samples"] = max(result["paired_samples"], len(per_sample))
        conditions.append(result)
    return {
        "status": "awaiting_human_labels" if completed == 0 else "scored",
        "completed_review_pairs": completed,
        "conditions": conditions,
        "note": "Human video-grounding and relevance ratings are separate from text-only coherence scores.",
    }


def summarize_acceptance(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
    coherence_rows: Sequence[Mapping[str, Any]] = (),
    expected_sample_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize only lossless verified runs; preserve parity failures as diagnostics."""
    by_retention: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        retention = int(row["retention_percentage"])
        if retention not in RETENTION_PERCENTAGES:
            raise ValueError(f"unknown acceptance retention {retention}")
        sample_id = str(row["sample_id"])
        if expected_sample_ids is not None and sample_id not in set(str(value) for value in expected_sample_ids):
            raise ValueError(f"acceptance journal contains sample outside protocol cohort: {sample_id}")
        if sample_id in by_retention[retention]:
            raise ValueError(f"duplicate acceptance sample/retention: {sample_id}/{retention}")
        by_retention[retention][sample_id] = dict(row)

    for sample_id in sorted(set().union(*(set(values) for values in by_retention.values()))):
        full = by_retention.get(100, {}).get(sample_id)
        if full is None:
            continue
        if full.get("status") not in {"ok", "empty_output", "mismatch"}:
            continue
        for retention in (75, 50, 25, 0):
            reduced = by_retention.get(retention, {}).get(sample_id)
            if reduced is None or reduced.get("status") not in {"ok", "empty_output", "mismatch"}:
                continue
            validate_paired_condition_rows(
                full, reduced, context=f"acceptance {sample_id}/{retention}%"
            )

    condition_rows = []
    paired_rows = []
    usable: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for retention in RETENTION_PERCENTAGES:
        values = by_retention.get(retention, {})
        parity_rows = [
            row for row in values.values()
            if row.get("status") in {"ok", "empty_output", "mismatch"}
        ]
        valid = {
            sample_id: row for sample_id, row in values.items()
            if row.get("status") == "ok"
            and row.get("outputs_match_target_greedy") is True
        }
        effective = [float(row["tau_effective"]) for row in valid.values()
                     if row.get("tau_effective") is not None]
        proposal = [float(row["tau_proposal"]) for row in valid.values()
                    if row.get("tau_proposal") is not None]
        effective_ci = _mean_ci(
            effective, seed=seed + retention, replicates=bootstrap_replicates
        )
        proposal_ci = _mean_ci(
            proposal, seed=seed + 1000 + retention, replicates=bootstrap_replicates
        )
        condition_rows.append({
            "retention_percentage": retention,
            "condition": CONDITION_NAMES[retention],
            "expected_samples": len(expected_sample_ids) if expected_sample_ids is not None else None,
            "missing_sample_ids": (
                sorted(set(str(value) for value in expected_sample_ids) - set(values))
                if expected_sample_ids is not None else []
            ),
            "rows": len(values),
            "lossless_rows": len(valid),
            "target_parity_rate": (
                sum(row.get("outputs_match_target_greedy") is True for row in parity_rows)
                / len(parity_rows) if parity_rows else None
            ),
            "mean_tau_effective": float(np.mean(effective)) if effective else None,
            "tau_effective_ci95_low": effective_ci[0] if effective_ci else None,
            "tau_effective_ci95_high": effective_ci[1] if effective_ci else None,
            "mean_tau_proposal": float(np.mean(proposal)) if proposal else None,
            "tau_proposal_ci95_low": proposal_ci[0] if proposal_ci else None,
            "tau_proposal_ci95_high": proposal_ci[1] if proposal_ci else None,
            "tau_valid_samples": len(effective),
            "mismatches": sum(row.get("status") == "mismatch" for row in values.values()),
            "errors": sum(row.get("status") == "error" for row in values.values()),
        })
        for sample_id, row in valid.items():
            if row.get("tau_effective") is None or row.get("tau_proposal") is None:
                continue
            usable[retention][sample_id] = {
                "tau_effective": float(row["tau_effective"]),
                "tau_proposal": float(row["tau_proposal"]),
            }
            paired_rows.append({
                "sample_id": sample_id,
                "retention_percentage": retention,
                "condition": CONDITION_NAMES[retention],
                "tau_effective": float(row["tau_effective"]),
                "tau_proposal": float(row["tau_proposal"]),
            })

    contrasts = []
    for retention in (75, 50, 25, 0):
        full = usable.get(100, {})
        reduced = usable.get(retention, {})
        sample_ids = sorted(set(full) & set(reduced))
        tau_deltas = [
            reduced[sample_id]["tau_effective"] - full[sample_id]["tau_effective"]
            for sample_id in sample_ids
        ]
        proposal_deltas = [
            reduced[sample_id]["tau_proposal"] - full[sample_id]["tau_proposal"]
            for sample_id in sample_ids
        ]
        ci = _mean_ci(tau_deltas, seed=seed + 10_000 + retention,
                      replicates=bootstrap_replicates)
        contrasts.append({
            "retention_percentage": retention,
            "condition": CONDITION_NAMES[retention],
            "paired_samples": len(sample_ids),
            "mean_tau_effective_delta": float(np.mean(tau_deltas)) if tau_deltas else None,
            "tau_effective_delta_ci95_low": ci[0] if ci else None,
            "tau_effective_delta_ci95_high": ci[1] if ci else None,
            "mean_tau_proposal_delta": float(np.mean(proposal_deltas)) if proposal_deltas else None,
            "wins": sum(value > 0 for value in tau_deltas),
            "ties": sum(value == 0 for value in tau_deltas),
            "losses": sum(value < 0 for value in tau_deltas),
        })

    coherence_delta_by_retention: dict[int, dict[str, float]] = defaultdict(dict)
    for row in coherence_rows:
        dimension = row.get("overall")
        if dimension is not None:
            coherence_delta_by_retention[int(row["retention_percentage"])][str(row["sample_id"])] = float(dimension)
    associations = []
    full_acceptance = usable.get(100, {})
    for retention in (75, 50, 25, 0):
        sample_ids = sorted(
            set(coherence_delta_by_retention.get(retention, {}))
            & set(full_acceptance) & set(usable.get(retention, {}))
        )
        coherence_delta = [
            coherence_delta_by_retention[retention][sample_id]
            for sample_id in sample_ids
        ]
        acceptance_delta = [
            usable[retention][sample_id]["tau_effective"]
            - full_acceptance[sample_id]["tau_effective"]
            for sample_id in sample_ids
        ]
        correlation = None
        if len(sample_ids) >= 3 and np.std(coherence_delta) > 0 and np.std(acceptance_delta) > 0:
            correlation = float(np.corrcoef(coherence_delta, acceptance_delta)[0, 1])
        associations.append({
            "retention_percentage": retention,
            "condition": CONDITION_NAMES[retention],
            "paired_samples": len(sample_ids),
            "pearson_r": correlation,
            "interpretation": "exploratory association; not mediation evidence",
        })

    summary = {
        "analysis": "separate_verified_speculative_acceptance",
        "primary_contrast": "tau_effective(Cut) - tau_effective(Full)",
        "primary": next(row for row in contrasts if row["retention_percentage"] == 0),
        "conditions": condition_rows,
        "expected_cohort_size": len(expected_sample_ids) if expected_sample_ids is not None else None,
        "contrasts": contrasts,
        "coherence_acceptance_associations": associations,
        "metric_definition": {
            "tau_effective": "mean emitted tokens per valid nonterminal verification round, including target bonus token",
            "tau_proposal": "mean matched draft proposals per valid nonterminal verification round",
        },
        "parity_requirement": "acceptance metrics include only status=ok rows with exact target-greedy token parity",
        "note": "This verified study is separate from raw unverified coherence rollouts.",
    }
    return summary, paired_rows


def summarize_visual_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
    expected_sample_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Align proposal changes by token position on identical target prefixes."""
    by_retention: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        retention = int(row["retention_percentage"])
        if retention not in RETENTION_PERCENTAGES:
            raise ValueError(f"unknown sensitivity retention {retention}")
        sample_id = str(row["sample_id"])
        if expected_sample_ids is not None and sample_id not in set(str(value) for value in expected_sample_ids):
            raise ValueError(f"sensitivity journal contains sample outside protocol cohort: {sample_id}")
        if sample_id in by_retention[retention]:
            raise ValueError(f"duplicate sensitivity sample/retention: {sample_id}/{retention}")
        by_retention[retention][sample_id] = dict(row)

    valid_by_retention = {
        retention: {
            sample_id: row for sample_id, row in by_retention.get(retention, {}).items()
            if row.get("status") == "ok"
        }
        for retention in RETENTION_PERCENTAGES
    }
    paired_rows = []
    token_rows = []
    condition_summaries = []
    sample_agreement: dict[int, dict[str, float]] = defaultdict(dict)
    sample_distribution_js: dict[int, dict[str, float]] = defaultdict(dict)
    full_rows = valid_by_retention.get(100, {})

    for retention in RETENTION_PERCENTAGES:
        current = valid_by_retention.get(retention, {})
        agreements = []
        target_match_rates = []
        aligned_tokens = 0
        for sample_id in sorted(set(full_rows) & set(current)):
            full_row = full_rows[sample_id]
            row = current[sample_id]
            if not full_row.get("input_fingerprint") or (
                full_row.get("input_fingerprint") != row.get("input_fingerprint")
            ):
                raise ValueError(f"sensitivity input_fingerprint differs across retention for {sample_id}")
            if full_row.get("teacher_forced_answer_sha256") != row.get("teacher_forced_answer_sha256"):
                raise ValueError(f"teacher-forced target prefix differs across retention for {sample_id}")
            full_tokens = {
                int(token["answer_position"]): token
                for token in full_row["probe_tokens"]
            }
            retained_tokens = {
                int(token["answer_position"]): token
                for token in row["probe_tokens"]
            }
            if len(full_tokens) != len(full_row["probe_tokens"]):
                raise ValueError(f"duplicate sensitivity positions for {sample_id}/Full")
            if len(retained_tokens) != len(row["probe_tokens"]):
                raise ValueError(f"duplicate sensitivity positions for {sample_id}/{retention}%")
            if set(full_tokens) != set(retained_tokens):
                raise ValueError(f"sensitivity positions differ across retention for {sample_id}")
            positions = sorted(full_tokens)
            if not positions:
                continue
            for position in positions:
                if int(full_tokens[position]["target_token_id"]) != int(
                    retained_tokens[position]["target_token_id"]
                ):
                    raise ValueError(f"teacher-forced target token differs at {sample_id}:{position}")
            matches_full = [
                int(full_tokens[position]["draft_proposal_token_id"])
                == int(retained_tokens[position]["draft_proposal_token_id"])
                for position in positions
            ]
            matches_target = [
                int(retained_tokens[position]["draft_proposal_token_id"])
                == int(retained_tokens[position]["target_token_id"])
                for position in positions
            ]
            agreement = float(np.mean(matches_full))
            target_match = float(np.mean(matches_target))
            distribution_js = [
                _coarse_distribution_js(
                    full_tokens[position], retained_tokens[position]
                )
                for position in positions
            ]
            available_js = [value for value in distribution_js if value is not None]
            agreements.append(agreement)
            target_match_rates.append(target_match)
            aligned_tokens += len(positions)
            sample_agreement[retention][sample_id] = agreement
            if available_js:
                sample_distribution_js[retention][sample_id] = float(np.mean(available_js))
            paired_rows.append({
                "sample_id": sample_id,
                "retention_percentage": retention,
                "condition": CONDITION_NAMES[retention],
                "aligned_tokens": len(positions),
                "proposal_agreement_with_full": agreement,
                "proposal_change_fraction_vs_full": 1.0 - agreement,
                "proposal_match_to_teacher_forced_target": target_match,
                "mean_coarse_distribution_js_vs_full": (
                    float(np.mean(available_js)) if available_js else None
                ),
            })
            for position in positions:
                full_token = full_tokens[position]
                retained_token = retained_tokens[position]
                token_rows.append({
                    "sample_id": sample_id,
                    "retention_percentage": retention,
                    "condition": CONDITION_NAMES[retention],
                    "answer_position": position,
                    "teacher_forced_target_token_id": int(retained_token["target_token_id"]),
                    "teacher_forced_target_token": retained_token.get("target_token", ""),
                    "teacher_forced_target_surface_class": retained_token.get("target_surface_class"),
                    "full_proposal_token_id": int(full_token["draft_proposal_token_id"]),
                    "full_proposal_token": full_token.get("draft_proposal_token", ""),
                    "full_proposal_surface_class": full_token.get("proposal_surface_class"),
                    "retained_proposal_token_id": int(retained_token["draft_proposal_token_id"]),
                    "retained_proposal_token": retained_token.get("draft_proposal_token", ""),
                    "retained_proposal_surface_class": retained_token.get("proposal_surface_class"),
                    "changed_vs_full": not matches_full[positions.index(position)],
                    "matches_teacher_forced_target": matches_target[positions.index(position)],
                    "coarse_distribution_js_vs_full": distribution_js[positions.index(position)],
                    "full_top1_probability": full_token.get("draft_top1_probability"),
                    "retained_top1_probability": retained_token.get("draft_top1_probability"),
                    "full_entropy_nats": full_token.get("draft_entropy_nats"),
                    "retained_entropy_nats": retained_token.get("draft_entropy_nats"),
                    "full_target_prefix_probability": full_token.get("draft_target_prefix_probability"),
                    "retained_target_prefix_probability": retained_token.get("draft_target_prefix_probability"),
                    "teacher_forced_target_surface_class": retained_token.get("target_surface_class"),
                    "full_proposal_surface_class": full_token.get("proposal_surface_class"),
                    "retained_proposal_surface_class": retained_token.get("proposal_surface_class"),
                })
        condition_summaries.append({
            "retention_percentage": retention,
            "condition": CONDITION_NAMES[retention],
            "paired_samples": len(agreements),
            "aligned_token_positions": aligned_tokens,
            "mean_proposal_agreement_with_full": float(np.mean(agreements)) if agreements else None,
            "mean_proposal_change_fraction_vs_full": 1.0 - float(np.mean(agreements)) if agreements else None,
            "mean_proposal_match_to_teacher_forced_target": float(np.mean(target_match_rates)) if target_match_rates else None,
            "mean_coarse_distribution_js_vs_full": (
                float(np.mean(list(sample_distribution_js[retention].values())))
                if sample_distribution_js[retention] else None
            ),
            "runtime_errors": sum(row.get("status") == "error" for row in by_retention.get(retention, {}).values()),
            "missing_sample_ids": (
                sorted(set(str(value) for value in expected_sample_ids)
                       - set(by_retention.get(retention, {})))
                if expected_sample_ids is not None else []
            ),
            "expected_samples": len(expected_sample_ids) if expected_sample_ids is not None else None,
        })

    contrasts = []
    for retention in (75, 50, 25, 0):
        full = sample_agreement.get(100, {})
        current = sample_agreement.get(retention, {})
        sample_ids = sorted(set(full) & set(current))
        deltas = [current[sample_id] - full[sample_id] for sample_id in sample_ids]
        # Full has agreement 1.0 with itself; the paired change fraction is
        # therefore the direct per-video 1 - agreement against Full.
        changed = [1.0 - current[sample_id] for sample_id in sample_ids]
        distribution_sample_ids = sorted(
            set(sample_distribution_js.get(100, {}))
            & set(sample_distribution_js.get(retention, {}))
        )
        distribution_deltas = [
            sample_distribution_js[retention][sample_id]
            for sample_id in distribution_sample_ids
        ]
        distribution_ci = _mean_ci(
            distribution_deltas, seed=seed + 3000 + retention,
            replicates=bootstrap_replicates,
        )
        ci = _mean_ci(changed, seed=seed + 2000 + retention,
                      replicates=bootstrap_replicates)
        contrasts.append({
            "retention_percentage": retention,
            "condition": CONDITION_NAMES[retention],
            "paired_samples": len(sample_ids),
            "mean_changed_proposal_fraction_vs_full": float(np.mean(changed)) if changed else None,
            "changed_fraction_ci95_low": ci[0] if ci else None,
            "changed_fraction_ci95_high": ci[1] if ci else None,
            "mean_agreement_delta_vs_full": float(np.mean(deltas)) if deltas else None,
            "mean_coarse_distribution_js_vs_full": (
                float(np.mean(distribution_deltas)) if distribution_deltas else None
            ),
            "coarse_distribution_js_ci95_low": distribution_ci[0] if distribution_ci else None,
            "coarse_distribution_js_ci95_high": distribution_ci[1] if distribution_ci else None,
            "distribution_paired_samples": len(distribution_deltas),
        })

    summary = {
        "analysis": "teacher_forced_draft_visual_retention_sensitivity",
        "prefix_source": "one target-greedy answer per sample generated with full visual input",
        "primary_contrast": "fraction of draft greedy proposals changed by Cut vs Full on identical prefixes",
        "primary": next(row for row in contrasts if row["retention_percentage"] == 0),
        "conditions": condition_summaries,
        "contrasts": contrasts,
        "interpretation": "Proposal changes indicate sensitivity to retained visual hidden rows; they do not prove that a token semantically requires visual evidence.",
        "distribution_metric": "Lower-bound Jensen-Shannon divergence in nats on shared top-32 token IDs plus one pooled remainder bucket; changes confined to non-shared tokens are intentionally hidden by this coarse partition.",
    }
    return summary, paired_rows, token_rows


def _coarse_distribution_js(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float | None:
    """Compute a lower-bound JS divergence on shared top-k IDs and a pooled tail."""
    required = (
        "draft_top_token_ids", "draft_top_token_probabilities",
        "draft_tail_probability_mass",
    )
    if any(name not in left or name not in right for name in required):
        return None

    def distribution(row: Mapping[str, Any]) -> dict[int, float]:
        ids = row["draft_top_token_ids"]
        probabilities = row["draft_top_token_probabilities"]
        if len(ids) != len(probabilities) or len(set(int(value) for value in ids)) != len(ids):
            raise ValueError("top-token distribution IDs and probabilities are malformed")
        values = {int(token_id): float(probability)
                  for token_id, probability in zip(ids, probabilities)}
        tail = float(row["draft_tail_probability_mass"])
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("top-token probabilities must be finite and non-negative")
        if not math.isfinite(tail) or tail < 0:
            raise ValueError("top-token tail mass must be finite and non-negative")
        total = sum(values.values()) + tail
        if total <= 0 or abs(total - 1.0) > 1e-3:
            raise ValueError("top-token distribution probability mass must sum to one")
        return {token_id: probability / total for token_id, probability in values.items()}

    left_values, right_values = distribution(left), distribution(right)
    shared_token_ids = set(left_values) & set(right_values)
    p = {token_id: left_values[token_id] for token_id in shared_token_ids}
    q = {token_id: right_values[token_id] for token_id in shared_token_ids}
    p[("pooled_remainder",)] = max(0.0, 1.0 - sum(p.values()))
    q[("pooled_remainder",)] = max(0.0, 1.0 - sum(q.values()))
    midpoint = {key: 0.5 * (p[key] + q[key]) for key in p}

    def kl(source: Mapping[Any, float]) -> float:
        return sum(
            probability * math.log(probability / midpoint[key])
            for key, probability in source.items()
            if probability > 0
        )

    return 0.5 * (kl(p) + kl(q))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def analyze_free_running(
    *,
    output_dir: str | Path,
    seed: int | None = None,
    bootstrap_replicates: int = 10_000,
    review_samples: int | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    protocol_path = root / "protocol.json"
    if protocol is None:
        if not protocol_path.is_file():
            raise FileNotFoundError(f"missing run protocol: {protocol_path}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_seed = int(protocol["random_seed"])
    if seed is not None and int(seed) != protocol_seed:
        raise ValueError(f"analysis seed {seed} differs from protocol seed {protocol_seed}")
    seed = protocol_seed
    protocol_review_samples = int(protocol.get("review_sample_count", 10))
    if review_samples is not None and int(review_samples) != protocol_review_samples:
        raise ValueError(
            f"review sample count {review_samples} differs from protocol count {protocol_review_samples}"
        )
    review_samples = protocol_review_samples
    cohort_sample_ids = [str(value) for value in protocol.get("cohort_sample_ids", [])]
    if not cohort_sample_ids:
        raise ValueError("run protocol is missing cohort_sample_ids")
    actual_cohort_hash = hashlib.sha256(
        "\n".join(sorted(cohort_sample_ids)).encode("utf-8")
    ).hexdigest()
    if actual_cohort_hash != protocol.get("cohort_sha256"):
        raise ValueError("protocol cohort_sample_ids do not match cohort_sha256")
    review_sample_ids = [str(value) for value in protocol.get("review_sample_ids", [])]
    if len(review_sample_ids) > review_samples:
        raise ValueError("protocol review_sample_ids exceed review_sample_count")
    expected_review_ids = sorted(
        cohort_sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"{seed}:{sample_id}".encode("utf-8")
        ).digest(),
    )[:review_samples]
    if review_sample_ids != expected_review_ids:
        raise ValueError("protocol review_sample_ids differ from the prespecified cohort selection")
    if int(protocol.get("cohort_size", len(cohort_sample_ids))) != len(cohort_sample_ids):
        raise ValueError("protocol cohort_size does not match cohort_sample_ids")
    current_protocol_sha256 = protocol_fingerprint(protocol)
    rollout_path = root / "raw_rollouts.jsonl"
    if not rollout_path.is_file():
        raise FileNotFoundError(f"missing free-running journal: {rollout_path}")
    raw_latest = _latest_rows(read_jsonl(rollout_path), "row_id")
    raw_rows, protocol_mismatch_rollout_ids = _rows_for_protocol(
        list(raw_latest.values()),
        protocol_sha256=current_protocol_sha256,
        key="row_id",
    )
    unexpected_samples = {str(row.get("sample_id")) for row in raw_rows} - set(cohort_sample_ids)
    if unexpected_samples:
        raise ValueError(f"rollout journal contains samples outside protocol cohort: {sorted(unexpected_samples)}")
    retentions = [int(value) for value in protocol["retention_percentages"]]
    coverage = summarize_rollouts(
        raw_rows,
        expected_sample_ids=cohort_sample_ids,
        expected_retentions=retentions,
    )
    coverage["protocol_sha256"] = current_protocol_sha256
    coverage["protocol_mismatch_row_ids"] = protocol_mismatch_rollout_ids
    coverage["cohort_complete"] = all(
        not row["missing_sample_ids"] and not row["unexpected_sample_ids"]
        and row["runtime_errors"] == 0
        and row["status_ok"] == row["expected_samples"]
        for row in coverage["conditions"] if row["scheduled"]
    ) and not protocol_mismatch_rollout_ids
    _write_json(root / "rollout_coverage.json", coverage)

    pairs, blind_key = build_blind_pairs(raw_rows, seed=seed)
    key_path = root / "blind_key.json"
    _write_json(key_path, blind_key)
    review = write_blinded_review_files(
        pairs, blind_key, root, seed=seed, sample_limit=review_samples,
        preselected_sample_ids=review_sample_ids,
    )
    questions = {
        str(row["sample_id"]): str(row.get("question", ""))
        for row in raw_rows if row.get("status") == "ok"
    }
    grounding_review = write_grounding_review_files(
        pairs, blind_key, questions, root, seed=seed, sample_limit=review_samples,
        preselected_sample_ids=review_sample_ids,
    )
    (root / "human_review_instructions.md").write_text(
        "# Blind coherence review\n\n"
        "Review each A/B pair for language coherence only. Ignore visual grounding, factual accuracy, relevance, helpfulness, and answer length. Do not infer the condition. The capped_a/capped_b columns identify which generation reached its token limit; do not penalize an unfinished final clause on that side.\n\n"
        "Score each dimension from 1 to 5: 1 = severe breaks that make the text hard to follow; 2 = repeated major errors; 3 = understandable with noticeable flaws; 4 = coherent with minor flaws; 5 = clear and consistent. Grammar covers sentence structure and continuation; reference covers subjects/entities/pronouns; discourse covers links and internal consistency; repetition covers unnecessary loops or repeated phrases; overall covers language coherence as a whole. Choose A, B, or tie for overall coherence and use notes to cite a short span supporting the decision. Complete `human_review_blinded.csv` before opening the key under `human_review_private/`.\n",
        encoding="utf-8",
    )
    (root / "grounding_review_instructions.md").write_text(
        "# Blind video-grounding review\n\n"
        "Use `sample_id` to open the corresponding local VDC video and consider the question shown. Rate whether each answer is supported by the visible video (groundedness) and whether it addresses the question (relevance), independently from writing quality. Use 1 = unsupported/off-topic, 3 = partly supported or partly relevant, and 5 = strongly supported/relevant. Choose winners separately; do not infer which condition produced A or B. Complete `grounding_review_blinded.csv` before opening the key under `grounding_review_private/`.\n",
        encoding="utf-8",
    )

    coherence_rows: list[dict[str, Any]] = []
    judge_path = root / "judge_pairs.jsonl"
    judge_latest = _latest_rows(read_jsonl(judge_path), "pair_id") if judge_path.is_file() else {}
    protocol_judge_rows, protocol_mismatch_judge_ids = _rows_for_protocol(
        list(judge_latest.values()),
        protocol_sha256=current_protocol_sha256,
        key="pair_id",
    )
    current_judge_rows = [
        row for row in protocol_judge_rows
        if str(row.get("pair_id")) in blind_key
    ]
    coherence_summary, coherence_rows = summarize_judge_pairs(
        current_judge_rows,
        blind_key, seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        expected_pair_ids=list(blind_key),
    )
    coherence_summary["protocol_sha256"] = current_protocol_sha256
    coherence_summary["protocol_mismatch_pair_ids"] = protocol_mismatch_judge_ids
    if judge_path.is_file():
        coherence_summary["human_calibration"] = _human_calibration(
            root / "human_review_blinded.csv", {
                str(row["pair_id"]): row for row in current_judge_rows
            }
        )
        _write_json(root / "coherence_summary.json", coherence_summary)
        with (root / "coherence_paired.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "sample_id", "retention_percentage", "condition", "grammar",
                "reference", "discourse", "repetition", "overall", "judge_winner",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(coherence_rows)

    matched_coherence_summary = None
    matched_judge_path = root / "judge_matched_prefix_pairs.jsonl"
    if matched_judge_path.is_file():
        matched_key_path = root / "matched_prefix_key.json"
        prefix_tokens = int(protocol["matched_prefix_tokens"])
        _, matched_key, matched_coverage = build_matched_prefix_pairs(
            raw_rows,
            seed=seed,
            prefix_tokens=prefix_tokens,
            decode_token_ids=lambda _token_ids: "",
        )
        _write_json(matched_key_path, matched_key)
        _write_json(root / "matched_prefix_coverage.json", matched_coverage)
        matched_latest = _latest_rows(read_jsonl(matched_judge_path), "pair_id")
        protocol_matched_rows, protocol_mismatch_matched_ids = _rows_for_protocol(
            list(matched_latest.values()),
            protocol_sha256=current_protocol_sha256,
            key="pair_id",
        )
        matched_coherence_summary, _matched_coherence_rows = summarize_judge_pairs(
            [row for row in protocol_matched_rows if str(row.get("pair_id")) in matched_key],
            matched_key, seed=seed + 1,
            bootstrap_replicates=bootstrap_replicates,
            expected_pair_ids=list(matched_key),
        )
        matched_coherence_summary["protocol_mismatch_pair_ids"] = protocol_mismatch_matched_ids
        matched_coherence_summary["length_matched_coverage"] = matched_coverage
        matched_coherence_summary["analysis"] = "paired_coherence_at_equal_token_prefix_length"
        _write_json(root / "coherence_matched_prefix_summary.json", matched_coherence_summary)

    acceptance_summary = None
    acceptance_path = root / "raw_acceptance.jsonl"
    if acceptance_path.is_file():
        acceptance_latest = _latest_rows(read_jsonl(acceptance_path), "row_id")
        current_acceptance_rows, protocol_mismatch_acceptance_ids = _rows_for_protocol(
            list(acceptance_latest.values()),
            protocol_sha256=current_protocol_sha256,
            key="row_id",
        )
        acceptance_summary, acceptance_rows = summarize_acceptance(
            current_acceptance_rows, seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            coherence_rows=coherence_rows,
            expected_sample_ids=cohort_sample_ids,
        )
        acceptance_summary["protocol_mismatch_row_ids"] = protocol_mismatch_acceptance_ids
        _write_json(root / "acceptance_summary.json", acceptance_summary)
        with (root / "acceptance_paired.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ("sample_id", "retention_percentage", "condition", "tau_effective", "tau_proposal")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(acceptance_rows)

    sensitivity_summary = None
    sensitivity_path = root / "raw_sensitivity.jsonl"
    if sensitivity_path.is_file():
        sensitivity_latest = _latest_rows(read_jsonl(sensitivity_path), "row_id")
        current_sensitivity_rows, protocol_mismatch_sensitivity_ids = _rows_for_protocol(
            list(sensitivity_latest.values()),
            protocol_sha256=current_protocol_sha256,
            key="row_id",
        )
        sensitivity_summary, sensitivity_paired, sensitivity_tokens = summarize_visual_sensitivity(
            current_sensitivity_rows, seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            expected_sample_ids=protocol.get("sensitivity_sample_ids", []),
        )
        sensitivity_summary["protocol_mismatch_row_ids"] = protocol_mismatch_sensitivity_ids
        _write_json(root / "visual_sensitivity_summary.json", sensitivity_summary)
        with (root / "visual_sensitivity_paired.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "sample_id", "retention_percentage", "condition", "aligned_tokens",
                "proposal_agreement_with_full", "proposal_change_fraction_vs_full",
                "proposal_match_to_teacher_forced_target", "mean_coarse_distribution_js_vs_full",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sensitivity_paired)
        with (root / "visual_sensitivity_by_token.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "sample_id", "retention_percentage", "condition", "answer_position",
                "teacher_forced_target_token_id", "teacher_forced_target_token",
                "teacher_forced_target_surface_class",
                "full_proposal_token_id", "full_proposal_token", "full_proposal_surface_class",
                "retained_proposal_token_id", "retained_proposal_token", "retained_proposal_surface_class",
                "changed_vs_full", "matches_teacher_forced_target",
                "coarse_distribution_js_vs_full", "full_top1_probability",
                "retained_top1_probability", "full_entropy_nats", "retained_entropy_nats",
                "full_target_prefix_probability", "retained_target_prefix_probability",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sensitivity_tokens)

    grounding_summary = _grounding_review_summary(
        root / "grounding_review_blinded.csv", blind_key,
        seed=seed, bootstrap_replicates=bootstrap_replicates,
    )
    if grounding_summary is not None:
        _write_json(root / "grounding_review_summary.json", grounding_summary)

    result = {
        "output_dir": str(root),
        "rollout_coverage": coverage,
        "coherence": coherence_summary,
        "coherence_matched_prefix": matched_coherence_summary,
        "acceptance": acceptance_summary,
        "visual_sensitivity": sensitivity_summary,
        "grounding_review": grounding_summary,
        "human_review": review,
        "grounding_review_files": grounding_review,
        "rollout_generation_complete": coverage["cohort_complete"],
        "rollout_study_ready": bool(
            coverage["cohort_complete"]
            and not coherence_summary["missing_pair_ids"]
            and not coherence_summary["score_errors"]
            and not coherence_summary["protocol_mismatch_pair_ids"]
            and bool(blind_key)
        ),
    }
    _write_json(root / "summary.json", result)
    lines = [
        "# DFlash free-running study summary",
        "",
        f"- Raw rollouts: {coverage['total_rows']} journal rows; {coverage['unique_samples']} unique videos.",
        "- Unverified rollout coherence and verified speculative acceptance are analyzed as separate outcomes.",
    ]
    if coherence_summary:
        primary = coherence_summary["primary"]
        lines.append(
            f"- LLM-judge primary contrast Cut − Full: {primary['mean_coherence_delta']} "
            f"(video-bootstrap 95% CI {primary['ci95_low']} to {primary['ci95_high']}; "
            f"n={primary['paired_samples']}). Human calibration is still required."
        )
    if matched_coherence_summary:
        primary = matched_coherence_summary["primary"]
        matched_len = matched_coherence_summary.get("length_matched_coverage", {}).get("prefix_tokens")
        lines.append(
            f"- Equal-length ({matched_len}-token) Cut − Full coherence: "
            f"{primary['mean_coherence_delta']} "
            f"(video-bootstrap 95% CI {primary['ci95_low']} to {primary['ci95_high']}; "
            f"n={primary['paired_samples']})."
        )
    if acceptance_summary:
        primary = acceptance_summary["primary"]
        lines.append(
            f"- Verified acceptance primary contrast Cut − Full, tau_effective: "
            f"{primary['mean_tau_effective_delta']} "
            f"(video-bootstrap 95% CI {primary['tau_effective_delta_ci95_low']} to "
            f"{primary['tau_effective_delta_ci95_high']}; n={primary['paired_samples']})."
        )
    if sensitivity_summary:
        primary = sensitivity_summary["primary"]
        lines.append(
            f"- Teacher-forced Cut vs Full proposal change fraction: "
            f"{primary['mean_changed_proposal_fraction_vs_full']} "
            f"(video-bootstrap 95% CI {primary['changed_fraction_ci95_low']} to "
            f"{primary['changed_fraction_ci95_high']}; n={primary['paired_samples']})."
        )
    lines.extend([
        "",
        "See `rollout_coverage.json`, `coherence_summary.json`, `coherence_matched_prefix_summary.json`, `acceptance_summary.json`, `visual_sensitivity_summary.json`, and `grounding_review_summary.json` for coverage and condition-level results.",
        "Raw free-running outputs have no acceptance length. LLM-judge scores remain exploratory until checked against blinded human labels; video grounding is reviewed separately.",
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--review-samples", type=int, default=None)
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if args.review_samples < 0:
        parser.error("--review-samples must be non-negative")
    result = analyze_free_running(
        output_dir=resolve_workspace_path(args.output_dir),
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        review_samples=args.review_samples,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
