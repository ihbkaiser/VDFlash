"""Compare DFlash speculative outputs against the dataset reference captions.

Loads one or more DFlash result JSONL files (the format produced by
``qwen35_dflash_benchmark``) and scores each prediction against the ``reference``
field with text-overlap metrics that need no extra dependencies:

- exact match (normalized)
- BLEU-1/2/3/4 (sentence level, with the standard geometric-mean + brevity
  penalty and a ``+1`` smoothing for zero counts)
- ROUGE-L F1 (LCS-based)
- unigram precision / recall / F1

The script prints a per-file summary, writes per-sample rows to
``<file>.scores.csv``, and optionally a combined markdown report.

Usage::

    python -m src.analyze.Whether_they_are_appliable_for_dDrafter.compare_dflash_reference \
        --inputs results/qwen35_dflash/vdc_t4_smoke_fixed.jsonl \
        [--inputs more.jsonl ...] \
        [--report results/qwen35_dflash/reference_comparison.md]

Without ``--inputs``, every ``results/qwen35_dflash/*.jsonl`` file is scored.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

_STOPWORDS = set(
    """a an the and or but if then than so of in on at to for with without from by as is are was were be been being am
    this that these those it its it's there here i you he she we they them his her their our your my me him us
    video videos scene scenes man men show shows people person one two three some any all most each other another
    visible seen shows begins starts starts continues looks appears around about into onto over under during while
    when after before near next first second again also very really quite more most much many""".split()
)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def content_words(text: str) -> list[str]:
    return [w for w in tokenize(text) if w not in _STOPWORDS and len(w) > 1]


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else Counter()


def _bleu(candidate: list[str], reference: list[str], max_n: int = 4) -> dict[int, float]:
    """Sentence-level BLEU with +1 smoothing for missing n-gram matches."""
    ref_len = len(reference)
    cand_len = len(candidate)
    scores: dict[int, float] = {}
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(candidate, n)
        ref_ng = _ngrams(reference, n)
        if not cand_ng:
            scores[n] = 0.0
            continue
        matches = sum(min(count, ref_ng.get(gram, 0)) for gram, count in cand_ng.items())
        precision = (matches + 1.0) / (sum(cand_ng.values()) + 1.0)  # +1 smoothing
        scores[n] = precision
    if cand_len == 0:
        brevity = 0.0
    else:
        brevity = min(1.0, ref_len / cand_len) if ref_len > 0 else 0.0
    log_avg = sum(score and __import__("math").log(score) for score in scores.values()) / max_n
    scores["bleu"] = brevity * __import__("math").exp(log_avg)
    return scores


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            current[j] = (
                previous[j - 1] + 1
                if token_a == token_b
                else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: list[str], reference: list[str]) -> float:
    lcs = _lcs(candidate, reference)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def score_pair(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    bleu = _bleu(pred_tokens, ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    ref_content = content_words(reference)
    pred_content_set = set(content_words(prediction))
    covered = sum(1 for w in ref_content if w in pred_content_set)
    coverage = covered / len(ref_content) if ref_content else 0.0

    return {
        "exact_match": 1.0 if normalize(prediction) == normalize(reference) else 0.0,
        "bleu1": bleu[1],
        "bleu2": bleu[2],
        "bleu3": bleu[3],
        "bleu4": bleu[4],
        "bleu": bleu["bleu"],
        "rouge_l": rouge_l_f1(pred_tokens, ref_tokens),
        "coverage": coverage,
        "unigram_precision": precision,
        "unigram_recall": recall,
        "unigram_f1": f1,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_file(path: Path) -> list[dict]:
    rows = load_rows(path)
    scored: list[dict] = []
    for index, row in enumerate(rows):
        prediction = row.get("prediction") or ""
        reference = row.get("reference") or ""
        metrics = score_pair(prediction, reference)
        scored.append(
            {
                "file": path.name,
                "row": index,
                "sample_id": row.get("sample_id"),
                "visual_percentage": row.get("visual_percentage"),
                "status": row.get("status"),
                "outputs_match": row.get("outputs_match"),
                "num_output_tokens": row.get("num_output_tokens"),
                "prediction_len": len(prediction),
                "reference_len": len(reference),
                **metrics,
            }
        )
    return scored


def print_summary(path: Path, scored: list[dict]) -> None:
    ok = [s for s in scored if s["status"] == "ok" and s["prediction_len"] > 0]
    print(f"\n### {path.name}  (rows={len(scored)}, scored_ok={len(ok)})")
    if not ok:
        print("  no scored (ok, non-empty) rows")
        return
    keys = ["exact_match", "bleu1", "bleu2", "bleu4", "bleu", "rouge_l", "unigram_f1"]
    print("  metric   mean    min     max")
    for key in keys:
        values = [s[key] for s in ok]
        print(f"  {key:<12} {_mean(values):.3f}  {min(values):.3f}  {max(values):.3f}")
    print(f"  outputs_match=True: {sum(1 for s in ok if s['outputs_match'] is True)}/{len(ok)}")


def build_report(path: Path, scored: list[dict]) -> list[str]:
    lines: list[str] = [f"## `{path.name}` ({len(scored)} rows)"]
    ok = [s for s in scored if s["status"] == "ok" and s["prediction_len"] > 0]
    if not ok:
        lines.append(f"No scored rows; statuses = {sorted({s['status'] for s in scored})}")
        lines.append("")
        return lines
    lines.append("| metric | mean | min | max |")
    lines.append("|---|---|---|---|")
    for key in ["exact_match", "bleu1", "bleu2", "bleu3", "bleu4", "bleu", "rouge_l", "coverage", "unigram_precision", "unigram_recall", "unigram_f1"]:
        values = [s[key] for s in ok]
        lines.append(f"| {key} | {_mean(values):.4f} | {min(values):.4f} | {max(values):.4f} |")
    lines.append("")
    lines.append("### Per-sample detail")
    lines.append("| sample_id | vp% | tok | match | exact | bleu | rouge_l | cov% |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in ok:
        lines.append(
            f"| {s['sample_id']} | {s['visual_percentage']} | {s['num_output_tokens']} "
            f"| {s['outputs_match']} | {s['exact_match']:.0f} | {s['bleu']:.3f} "
            f"| {s['rouge_l']:.3f} | {s['coverage'] * 100:.0f} |"
        )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", action="append", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        paths = sorted(Path("results/qwen35_dflash").glob("*.jsonl"))

    report_lines: list[str] = ["# DFlash output vs reference comparison", ""]
    for path in paths:
        scored = score_file(path)
        print_summary(path, scored)
        csv_path = path.with_suffix(path.suffix + ".scores.csv")
        import csv

        if scored:
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(scored[0].keys()))
                writer.writeheader()
                writer.writerows(scored)
        report_lines.extend(build_report(path, scored))

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"\nWrote report -> {out}")


if __name__ == "__main__":
    main()
