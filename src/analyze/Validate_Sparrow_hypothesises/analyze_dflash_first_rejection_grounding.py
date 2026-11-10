"""Compare first-rejection positions with reference-answer content words.

This is an exploratory lexical-overlap proxy for semantic grounding labels,
not a replacement for blinded human annotation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import gzip
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
_STOPWORDS = set(
    "a an and are as at be been being by can could did do does for from had has "
    "have he her hers him his how i if in into is it its may might more most much "
    "must my near no not of off on once one or our out over she should so some "
    "such than that the their them then there these they this those through to "
    "under until up us was we were what when where which while who why will with "
    "would you your also video videos scene scenes camera footage clip clips "
    "appears seem seems likely possibly perhaps overall starts begins continues "
    "ends finally then next first second another".split()
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _content_forms(word: str) -> set[str]:
    """Return a small set of morphology variants for a lexical-match proxy."""
    value = word.casefold().replace("’", "'").strip("'")
    if not value or value in _STOPWORDS:
        return set()
    forms = {value}
    if len(value) > 5 and value.endswith("ies"):
        forms.add(value[:-3] + "y")
    if len(value) > 5 and value.endswith("ing"):
        stem = value[:-3]
        forms.add(stem)
        forms.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if len(value) > 4 and value.endswith("ed"):
        stem = value[:-2]
        forms.add(stem)
        forms.add(stem + "e")
        if len(stem) > 2 and stem[-1] == stem[-2]:
            forms.add(stem[:-1])
    if len(value) > 4 and value.endswith("es"):
        forms.add(value[:-2])
    if len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        forms.add(value[:-1])
    return forms


def _reference_patterns(reference: str, tokenizer: Any) -> dict[int, set[tuple[int, ...]]]:
    ref_words = {match.group(0) for match in _WORD_RE.finditer(reference)}
    forms = {
        prefix + variant
        for word in ref_words
        for form in _content_forms(word)
        for variant in {form, form.capitalize(), form.upper()}
        for prefix in ("", " ")
    }
    if not forms:
        return {}
    encoded = tokenizer(sorted(forms), add_special_tokens=False)["input_ids"]
    patterns_by_first: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for sequence in encoded:
        pattern = tuple(int(value) for value in sequence)
        if pattern:
            patterns_by_first[pattern[0]].add(pattern)
    return patterns_by_first


def _token_labels(
    token_ids: list[int],
    tokenizer: Any,
    patterns_by_first: dict[int, set[tuple[int, ...]]],
    sample_id: str,
) -> tuple[list[int], str]:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    labels = [0] * len(token_ids)
    for start, token_id in enumerate(token_ids):
        for pattern in patterns_by_first.get(token_id, ()):
            end = start + len(pattern)
            if end > len(token_ids) or tuple(token_ids[start:end]) != pattern:
                continue
            if start > 0 and not (
                token_strings[start].startswith("Ġ")
                or token_strings[start - 1].startswith("<|")
                or not any(char.isalnum() for char in token_strings[start - 1].replace("Ġ", ""))
            ):
                continue
            if end < len(token_ids) and not (
                token_strings[end].startswith(("Ġ", "<|"))
                or not any(char.isalnum() for char in token_strings[end].replace("Ġ", ""))
            ):
                continue
            labels[start:end] = [1] * len(pattern)
    if len(labels) != len(token_ids):
        raise ValueError(f"token label count differs from output tokens for {sample_id}")
    return labels, text


def _read_pair(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = status = None
    positions: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("record_type")
            if kind == "pair_meta":
                metadata = row
            elif kind == "pair_status":
                status = row
            elif kind == "position":
                positions.append(row)
    if metadata is None or status is None:
        raise ValueError(f"missing pair metadata/status: {path}")
    if status.get("status") != "ok" or int(status["position_count"]) != len(positions):
        raise ValueError(f"invalid pair status or position count: {path}")
    if not status.get("full_matches_target_greedy"):
        raise ValueError(f"Full replay differs from target trajectory: {path}")
    return metadata, positions


def _macro_rates(
    per_sample: dict[str, dict[tuple[int, str, int], list[int]]],
    sample_counts: dict[str, int],
    budget: int,
    condition: str,
    block_size: int,
) -> tuple[float, float, float, int]:
    """Macro-average first-rejection rates over slots, matched by slot."""
    critical_rates: list[float] = []
    other_rates: list[float] = []
    for slot in range(1, block_size):
        critical_n = critical_events = other_n = other_events = 0
        for sample_id, multiplicity in sample_counts.items():
            if multiplicity <= 0:
                continue
            values = per_sample.get(sample_id, {}).get(
                (budget, condition, slot), [0, 0, 0, 0]
            )
            critical_n += multiplicity * values[0]
            critical_events += multiplicity * values[1]
            other_n += multiplicity * values[2]
            other_events += multiplicity * values[3]
        if critical_n and other_n:
            critical_rates.append(critical_events / critical_n)
            other_rates.append(other_events / other_n)
    if not critical_rates:
        raise ValueError(f"no slots contain both labels at budget {budget}, {condition}")
    critical_rate = fmean(critical_rates)
    other_rate = fmean(other_rates)
    return critical_rate, other_rate, critical_rate - other_rate, len(critical_rates)


def _risk_set_metrics(
    counts_by_sample: dict[str, dict[tuple[int, str, int], list[int]]],
    sample_multiplicities: dict[str, int],
    budget: int,
    block_size: int,
    max_slot: int = 4,
) -> dict[str, float]:
    rates: dict[tuple[str, str], list[float]] = {
        (condition, label): []
        for condition in ("full", "cut")
        for label in ("overlap", "other")
    }
    for slot in range(1, max_slot + 1):
        totals = {
            (condition, label): [0, 0]
            for condition in ("full", "cut")
            for label in ("overlap", "other")
        }
        for sample_id, multiplicity in sample_multiplicities.items():
            if multiplicity <= 0:
                continue
            for condition in ("full", "cut"):
                values = counts_by_sample.get(sample_id, {}).get(
                    (budget, condition, slot), [0, 0, 0, 0]
                )
                totals[(condition, "overlap")][0] += multiplicity * values[0]
                totals[(condition, "overlap")][1] += multiplicity * values[1]
                totals[(condition, "other")][0] += multiplicity * values[2]
                totals[(condition, "other")][1] += multiplicity * values[3]
        if all(value[0] > 0 for value in totals.values()):
            for key, (at_risk, rejects) in totals.items():
                rates[key].append(rejects / at_risk)
    if not rates or any(len(values) != max_slot for values in rates.values()):
        raise ValueError(f"risk-set strata incomplete at budget {budget}")
    means = {key: fmean(values) for key, values in rates.items()}
    overlap_delta = means[("cut", "overlap")] - means[("full", "overlap")]
    other_delta = means[("cut", "other")] - means[("full", "other")]
    return {
        "full_overlap_rate": means[("full", "overlap")],
        "full_other_rate": means[("full", "other")],
        "cut_overlap_rate": means[("cut", "overlap")],
        "cut_other_rate": means[("cut", "other")],
        "cut_minus_full_overlap": overlap_delta,
        "cut_minus_full_other": other_delta,
        "interaction_difference_in_differences": overlap_delta - other_delta,
    }


def analyze(
    run_dir: Path,
    manifest: Path,
    tokenizer_path: str,
    output_dir: Path,
    *,
    block_size: int = 16,
    bootstrap_reps: int = 2000,
    seed: int = 20261105,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in config["selected_sample_ids"]]
    budgets = [int(value) for value in config["budgets"]]
    expected_pairs = {(sample_id, budget) for sample_id in sample_ids for budget in budgets}
    manifest_rows = _read_jsonl(manifest)
    references = {str(row["video_name"]): str(row["answer"]) for row in manifest_rows}
    if not set(sample_ids).issubset(references):
        raise ValueError("manifest does not contain the full replay cohort")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    labels_by_pair: dict[tuple[str, int], list[int]] = {}
    texts_by_pair: dict[tuple[str, int], str] = {}
    pair_data: dict[tuple[str, int], list[dict[str, Any]]] = {}
    ids_by_pair: dict[tuple[str, int], list[int]] = {}
    patterns_by_sample = {
        sample_id: _reference_patterns(references[sample_id], tokenizer)
        for sample_id in sample_ids
    }
    for path in sorted((run_dir / "pairs").glob("*.jsonl.gz")):
        metadata, positions = _read_pair(path)
        sample_id = str(metadata["sample_id"])
        budget = int(metadata["target_visual_tokens"])
        key = (sample_id, budget)
        if key not in expected_pairs or key in pair_data:
            raise ValueError(f"unexpected or duplicate pair: {key}")
        token_ids = [int(value) for value in metadata["target_greedy_ids"]]
        ids_by_pair[key] = token_ids
        labels_by_pair[key], texts_by_pair[key] = _token_labels(
            token_ids,
            tokenizer,
            patterns_by_sample[sample_id],
            f"{sample_id}/{budget}",
        )
        pair_data[key] = positions
    if set(pair_data) != expected_pairs:
        raise ValueError(f"incomplete pair grid: found {len(pair_data)}/{len(expected_pairs)}")

    # The labels are derived from reference-answer text only. They are created
    # before using either Full or Cut rejection fields below.
    output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = output_dir / "blind_reference_overlap_labels.jsonl"
    with blind_path.open("w", encoding="utf-8") as handle:
        for sample_id, budget in sorted(expected_pairs):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "budget": budget,
                        "target_text": texts_by_pair[(sample_id, budget)],
                        "reference_overlap_by_token": labels_by_pair[(sample_id, budget)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Per-video x budget x condition x slot totals:
    # [reference-overlap opportunities, first rejects, other opportunities, first rejects].
    per_sample: dict[str, dict[tuple[int, str, int], list[int]]] = defaultdict(dict)
    risk_counts: dict[str, dict[tuple[int, str, int], list[int]]] = defaultdict(dict)
    opportunity_rows: list[dict[str, Any]] = []
    complete_blocks = defaultdict(int)
    for (sample_id, budget), positions in pair_data.items():
        by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in positions:
            by_block[int(row["block_index"])].append(row)
        nonterminal_ids = sorted(by_block)[:-1]
        for block_id in nonterminal_ids:
            rows = sorted(by_block[block_id], key=lambda row: int(row["draft_slot_in_block"]))
            if len(rows) != block_size - 1:
                continue
            if [int(row["draft_slot_in_block"]) for row in rows] != list(range(1, block_size)):
                raise ValueError(f"invalid slot sequence for {sample_id}/{budget}/{block_id}")
            complete_blocks[(sample_id, budget)] += 1
            for condition in ("full", "cut"):
                first_reject_slot = next(
                    (
                        int(row["draft_slot_in_block"])
                        for row in rows
                        if not bool(row[f"{condition}_metrics"]["target_hit_top_1"])
                    ),
                    None,
                )
                reached = True
                for row in rows:
                    position = int(row["target_position"])
                    if not 0 <= position < len(labels_by_pair[(sample_id, budget)]):
                        raise ValueError(f"target position out of range: {sample_id}/{position}")
                    if int(row["target_greedy_token_id"]) != ids_by_pair[(sample_id, budget)][position]:
                        raise ValueError(f"target token alignment mismatch: {sample_id}/{position}")
                    slot = int(row["draft_slot_in_block"])
                    is_critical = labels_by_pair[(sample_id, budget)][position]
                    event = int(slot == first_reject_slot)
                    key = (budget, condition, slot)
                    counts = per_sample[sample_id].setdefault(key, [0, 0, 0, 0])
                    if is_critical:
                        counts[0] += 1
                        counts[1] += event
                    else:
                        counts[2] += 1
                        counts[3] += event
                    at_risk = reached
                    if at_risk:
                        risk_values = risk_counts[sample_id].setdefault(key, [0, 0, 0, 0])
                        if is_critical:
                            risk_values[0] += 1
                            risk_values[1] += event
                        else:
                            risk_values[2] += 1
                            risk_values[3] += event
                    opportunity_rows.append(
                        {
                            "sample_id": sample_id,
                            "budget": budget,
                            "condition": condition,
                            "block_index": block_id,
                            "slot": slot,
                            "target_position": position,
                            "token_id": ids_by_pair[(sample_id, budget)][position],
                            "reference_overlap_proxy": is_critical,
                            "is_first_rejection": event,
                            "at_risk": int(at_risk),
                        }
                    )
                    if event:
                        reached = False

    rng = random.Random(seed)
    draws = [
        [sample_ids[rng.randrange(len(sample_ids))] for _ in sample_ids]
        for _ in range(bootstrap_reps)
    ]
    summary_rows: list[dict[str, Any]] = []
    risk_summary_rows: list[dict[str, Any]] = []
    for budget in budgets:
        point = _risk_set_metrics(
            risk_counts,
            {sample_id: 1 for sample_id in sample_ids},
            budget,
            block_size,
            max_slot=4,
        )
        boot = {key: [] for key in point}
        for draw in draws:
            multiplicities: dict[str, int] = defaultdict(int)
            for sample_id in draw:
                multiplicities[sample_id] += 1
            estimate = _risk_set_metrics(
                risk_counts, multiplicities, budget, block_size, max_slot=4
            )
            for key, value in estimate.items():
                boot[key].append(value)
        row: dict[str, Any] = {"budget": budget, **point, "sample_count": len(sample_ids)}
        for key, values in boot.items():
            values.sort()
            lo = values[int(0.025 * (len(values) - 1))]
            hi = values[int(0.975 * (len(values) - 1))]
            row[f"{key}_ci95_low"] = lo
            row[f"{key}_ci95_high"] = hi
        row["complete_blocks"] = sum(
            complete_blocks[(sample_id, budget)] for sample_id in sample_ids
        )
        risk_summary_rows.append(row)

    for condition in ("full", "cut"):
        for budget in budgets:
            critical_rate, other_rate, difference, used_slots = _macro_rates(
                per_sample,
                {sample_id: 1 for sample_id in sample_ids},
                budget,
                condition,
                block_size,
            )
            bootstrap_differences = []
            for draw in draws:
                counts: dict[str, int] = defaultdict(int)
                for sample_id in draw:
                    counts[sample_id] += 1
                _, _, estimate, _ = _macro_rates(
                    per_sample, counts, budget, condition, block_size
                )
                bootstrap_differences.append(estimate)
            bootstrap_differences.sort()
            lo_i = int(0.025 * (len(bootstrap_differences) - 1))
            hi_i = int(0.975 * (len(bootstrap_differences) - 1))
            critical_n = critical_events = other_n = other_events = 0
            for slot in range(1, block_size):
                for sample_id in sample_ids:
                    values = per_sample.get(sample_id, {}).get(
                        (budget, condition, slot), [0, 0, 0, 0]
                    )
                    critical_n += values[0]
                    critical_events += values[1]
                    other_n += values[2]
                    other_events += values[3]
            summary_rows.append(
                {
                    "condition": condition,
                    "budget": budget,
                    "critical_first_rejections": critical_events,
                    "critical_positions": critical_n,
                    "other_first_rejections": other_events,
                    "other_positions": other_n,
                    "critical_first_rejection_rate_macro_by_slot": critical_rate,
                    "other_first_rejection_rate_macro_by_slot": other_rate,
                    "risk_difference_macro_by_slot": difference,
                    "risk_difference_ci95_low": bootstrap_differences[lo_i],
                    "risk_difference_ci95_high": bootstrap_differences[hi_i],
                    "slots_in_macro": used_slots,
                    "complete_blocks": sum(
                        complete_blocks[(sample_id, budget)] for sample_id in sample_ids
                    ),
                    "sample_count": len(sample_ids),
                }
            )

    fields = list(summary_rows[0])
    with (output_dir / "grounding_proxy_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "grounding_proxy_positions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(opportunity_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(opportunity_rows)

    with (output_dir / "grounding_proxy_riskset.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risk_summary_rows[0]))
        writer.writeheader()
        writer.writerows(risk_summary_rows)

    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = {
        "run_dir": str(run_dir),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "tokenizer_path": tokenizer_path,
        "sample_count": len(sample_ids),
        "budget_values": budgets,
        "pair_count": len(pair_data),
        "complete_block_count": sum(complete_blocks.values()),
        "annotated_target_tokens": sum(len(value) for value in labels_by_pair.values()),
        "reference_overlap_token_count": sum(sum(value) for value in labels_by_pair.values()),
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": seed,
        "label_definition": (
            "weak proxy: a generated token overlaps a non-stopword content term in the "
            "dataset reference answer, with simple suffix variants"
        ),
        "limitations": [
            "not blinded human semantic annotation; lexical overlap misses synonyms and uncaptured details",
            "first rejection is from fixed-trajectory replay, not free-running Full/Cut",
            "Cut remains a forced-replay proxy",
        ],
        "results": summary_rows,
        "risk_set_definition": "a proposal position is at risk only if all earlier slots in the same block and condition were top-1 accepted",
        "risk_set_slots": [1, 2, 3, 4],
        "risk_set_results": risk_summary_rows,
    }
    (output_dir / "grounding_proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    summary = analyze(
        args.run_dir,
        args.manifest,
        args.tokenizer,
        args.output_dir,
        block_size=args.block_size,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
