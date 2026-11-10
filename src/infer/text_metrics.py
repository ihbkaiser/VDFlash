"""Dependency-light text metrics shared by inference and analysis reports."""

from __future__ import annotations

import math
import re
from collections import Counter


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_STOPWORDS = set(
    """a an the and or but if then than so of in on at to for with without from by as is are was were be been being am
    this that these those it its it's there here i you he she we they them his her their our your my me him us
    video videos scene scenes man men show shows people person one two three some any all most each other another
    visible seen shows begins starts continues looks appears around about into onto over under during while
    when after before near next first second again also very really quite more much many""".split()
)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(
                previous[index - 1]
                if left_item == right_item
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _rouge_l(reference: str, candidate: str) -> float:
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    if not reference_tokens or not candidate_tokens:
        return 1.0 if reference_tokens == candidate_tokens else 0.0
    score = _lcs_length(reference_tokens, candidate_tokens)
    precision = score / len(candidate_tokens)
    recall = score / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    """Compute the dependency-light text metrics used by DFlash reports."""

    def tokenize(text: str) -> list[str]:
        return _WORD_RE.findall((text or "").lower())

    def ngrams(tokens: list[str], n: int) -> Counter:
        return Counter(zip(*[tokens[index:] for index in range(n)])) if len(tokens) >= n else Counter()

    def smoothed_bleu(candidate: list[str], reference_tokens: list[str]) -> dict[int | str, float]:
        scores: dict[int | str, float] = {}
        for n in range(1, 5):
            candidate_ngrams = ngrams(candidate, n)
            reference_ngrams = ngrams(reference_tokens, n)
            if not candidate_ngrams:
                scores[n] = 0.0
                continue
            matches = sum(
                min(count, reference_ngrams.get(gram, 0))
                for gram, count in candidate_ngrams.items()
            )
            scores[n] = (matches + 1.0) / (sum(candidate_ngrams.values()) + 1.0)
        brevity = (
            min(1.0, len(reference_tokens) / len(candidate))
            if candidate and reference_tokens
            else 0.0
        )
        log_average = sum(
            math.log(scores[n]) if scores[n] > 0 else 0.0 for n in range(1, 5)
        ) / 4
        scores["bleu"] = brevity * math.exp(log_average)
        return scores

    candidate = tokenize(prediction)
    reference_tokens = tokenize(reference)
    bleu = smoothed_bleu(candidate, reference_tokens)
    overlap = sum((Counter(candidate) & Counter(reference_tokens)).values())
    precision = overlap / len(candidate) if candidate else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reference_content = [word for word in reference_tokens if word not in _STOPWORDS and len(word) > 1]
    candidate_content = set(word for word in candidate if word not in _STOPWORDS and len(word) > 1)
    coverage = (
        sum(word in candidate_content for word in reference_content) / len(reference_content)
        if reference_content
        else 0.0
    )
    normalized_candidate = " ".join(candidate)
    normalized_reference = " ".join(reference_tokens)
    return {
        "exact_match": 1.0 if normalized_candidate == normalized_reference else 0.0,
        "bleu1": float(bleu[1]),
        "bleu2": float(bleu[2]),
        "bleu3": float(bleu[3]),
        "bleu4": float(bleu[4]),
        "bleu": float(bleu["bleu"]),
        "rouge_l": _rouge_l(reference, prediction),
        "coverage": float(coverage),
        "unigram_precision": float(precision),
        "unigram_recall": float(recall),
        "unigram_f1": float(f1),
    }
