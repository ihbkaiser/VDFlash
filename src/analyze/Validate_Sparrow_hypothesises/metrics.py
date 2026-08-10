"""Metrics used by the paper-aligned validation report."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def exact_token_match(reference: Sequence[int], candidate: Sequence[int]) -> bool:
    return list(reference) == list(candidate)


def common_prefix_length(reference: Sequence[int], candidate: Sequence[int]) -> int:
    length = 0
    for left, right in zip(reference, candidate):
        if left != right:
            break
        length += 1
    return length


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_item == right_item else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, candidate: str) -> float:
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    if not reference_tokens or not candidate_tokens:
        return 1.0 if reference_tokens == candidate_tokens else 0.0
    score = lcs_length(reference_tokens, candidate_tokens)
    precision = score / len(candidate_tokens)
    recall = score / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def normalized_entropy(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if value > 0]
    if len(clean) <= 1:
        return 0.0
    total = sum(clean)
    entropy = -sum((value / total) * math.log(value / total) for value in clean)
    return entropy / math.log(len(clean))


def bootstrap_mean(values: Sequence[float], replicates: int = 2000, seed: int = 42) -> dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, replicates)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "ci_low": means[low_index],
        "ci_high": means[high_index],
    }


def acceptance_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accepted = [float(row["accepted_prefix_tokens"]) for row in rows if row.get("accepted_prefix_tokens") is not None]
    prefill = [float(row["prefill_seconds"]) for row in rows if row.get("prefill_seconds") is not None]
    decode = [float(row["decode_seconds"]) for row in rows if row.get("decode_seconds") is not None]
    end_to_end = [float(row["end_to_end_seconds"]) for row in rows if row.get("end_to_end_seconds") is not None]
    lossless = [bool(row["lossless"]) for row in rows if row.get("lossless") is not None]
    result: dict[str, Any] = {
        "n": len(rows),
        "accepted_length": bootstrap_mean(accepted),
        "prefill_seconds": bootstrap_mean(prefill),
        "decode_seconds": bootstrap_mean(decode),
        "end_to_end_seconds": bootstrap_mean(end_to_end),
        "lossless_rate": (sum(lossless) / len(lossless)) if lossless else None,
    }
    ar_decode = [float(row["ar_decode_seconds"]) for row in rows if row.get("ar_decode_seconds") is not None]
    ar_e2e = [float(row["ar_end_to_end_seconds"]) for row in rows if row.get("ar_end_to_end_seconds") is not None]
    if decode and len(decode) == len(ar_decode):
        result["decode_speedup"] = sum(ar_decode) / sum(decode) if sum(decode) else None
    if end_to_end and len(end_to_end) == len(ar_e2e):
        result["end_to_end_speedup"] = sum(ar_e2e) / sum(end_to_end) if sum(end_to_end) else None
    return result
