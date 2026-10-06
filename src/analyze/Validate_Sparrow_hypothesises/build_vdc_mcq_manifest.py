"""Build a deterministic, auditable multiple-choice VDC manifest.

Phase 1 only constructs the raw task manifest. It does not score a model or
compute any hypothesis-level comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


QUESTION = "Which description best matches the video?"
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD_RE.findall(str(text))}


def _stable_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _distractor_score(answer: str, candidate: str) -> tuple[float, float, str]:
    answer_words = _words(answer)
    candidate_words = _words(candidate)
    union = answer_words | candidate_words
    overlap = len(answer_words & candidate_words) / max(1, len(union))
    length_ratio = min(len(answer), len(candidate)) / max(1, max(len(answer), len(candidate)))
    return (overlap, length_ratio, candidate)


def _select_distractors(
    rows: list[dict[str, Any]],
    index: int,
    *,
    option_count: int,
) -> list[dict[str, Any]]:
    source = rows[index]
    answer = str(source["answer"])
    candidates: list[dict[str, Any]] = []
    for other_index, other in enumerate(rows):
        if other_index == index:
            continue
        candidate = str(other["answer"])
        score = _distractor_score(answer, candidate)
        candidates.append(
            {
                "source_index": other_index,
                "sample_id": str(other.get("video_name") or other.get("sample_id")),
                "text": candidate,
                "lexical_overlap": score[0],
                "length_similarity": score[1],
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["lexical_overlap"]),
            -float(item["length_similarity"]),
            str(item["sample_id"]),
        )
    )
    if len(candidates) < option_count - 1:
        raise ValueError("VDC-MCQ needs at least option_count distinct source rows")
    return candidates[: option_count - 1]


def build_vdc_mcq_rows(
    source_rows: Iterable[dict[str, Any]],
    *,
    seed: int = 42,
    option_count: int = 4,
) -> list[dict[str, Any]]:
    """Return deterministic VDC-MCQ rows with source provenance."""

    rows = [dict(row) for row in source_rows]
    if option_count != 4:
        raise ValueError("The Phase-1 VDC-MCQ contract requires exactly four options")
    if len(rows) < option_count:
        raise ValueError("VDC-MCQ requires at least four source rows")

    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        sample_id = str(source.get("video_name") or source.get("sample_id"))
        answer = str(source["answer"])
        distractors = _select_distractors(rows, index, option_count=option_count)
        option_records = [{
            "source_index": index,
            "sample_id": sample_id,
            "text": answer,
            "is_correct": True,
            "lexical_overlap": 1.0,
            "length_similarity": 1.0,
        }, *[
            {
                **item,
                "is_correct": False,
            }
            for item in distractors
        ]]
        rng = random.Random(_stable_seed(seed, sample_id))
        rng.shuffle(option_records)
        correct_index = next(
            position for position, item in enumerate(option_records) if item["is_correct"]
        )
        correct_option = chr(ord("A") + correct_index)
        candidates = [str(item["text"]) for item in option_records]
        option_provenance = [
            {
                "option": chr(ord("A") + position),
                "sample_id": str(item["sample_id"]),
                "source_index": int(item["source_index"]),
                "is_correct": bool(item["is_correct"]),
                "lexical_overlap": float(item["lexical_overlap"]),
                "length_similarity": float(item["length_similarity"]),
            }
            for position, item in enumerate(option_records)
        ]
        distractor_provenance = [
            item for item in option_provenance if not item["is_correct"]
        ]
        output.append(
            {
                **source,
                "question": QUESTION,
                "natural_question": str(source.get("question", "")),
                "answer": answer,
                "reference_answer": answer,
                "candidates": candidates,
                "correct_option": correct_option,
                "dataset_format": "vdc_mcq",
                "option_provenance": option_provenance,
                "distractor_provenance": distractor_provenance,
                "automatic_audit": {
                    "candidate_count": len(candidates),
                    "exactly_one_answer": sum(item["is_correct"] for item in option_records) == 1,
                    "unique_candidate_text": len(set(candidates)) == len(candidates),
                    "source_row_index": index,
                },
                "audit_status": "automatic_pass_pending_manual_review",
                "mcq_seed": int(seed),
            }
        )
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument(
        "--output",
        default="dataset/VideoDetailCaption/vdc_mcq_manifest.jsonl",
    )
    parser.add_argument(
        "--audit-output",
        default="dataset/VideoDetailCaption/vdc_mcq_audit.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    rows = build_vdc_mcq_rows(_read_jsonl(Path(args.input)), seed=args.seed)
    _write_jsonl(Path(args.output), rows)
    audit = {
        "dataset_format": "vdc_mcq",
        "source_manifest": str(args.input),
        "output_manifest": str(args.output),
        "seed": int(args.seed),
        "row_count": len(rows),
        "all_automatic_audits_pass": all(
            row["automatic_audit"]["exactly_one_answer"]
            and row["automatic_audit"]["unique_candidate_text"]
            for row in rows
        ),
        "manual_review_required": True,
    }
    Path(args.audit_output).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
