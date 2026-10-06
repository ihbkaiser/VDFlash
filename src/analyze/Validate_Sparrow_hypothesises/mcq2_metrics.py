"""Response-format and option metrics for VDC-MCQ prompt comparisons."""

from __future__ import annotations

import re
from typing import Any


_WORD_PATTERN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", flags=re.UNICODE)
_SENTENCE_END_PATTERN = re.compile(r"[.!?](?:[\"')\]]*)?(?=\s|$)")
_LEADING_OPTION_PATTERN = re.compile(r"^\s*\(([A-D])\)(?:\s+|$)(.*)$", re.IGNORECASE | re.DOTALL)


def _sentence_count(text: str) -> int:
    return len(_SENTENCE_END_PATTERN.findall(text.strip()))


def _normalized_option(value: Any) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*\(?([A-D])\)?\s*", str(value), re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_vdc_mcq2_response(response: str | None, correct_option: Any) -> dict[str, Any]:
    """Parse leading '(A)' and score the answer body separately from the choice.

    Format compliance is a transparent proxy: one leading option label, at
    least five answer words, and one or two punctuation-terminated sentences.
    It is not a semantic-quality judgment.
    """

    text = str(response or "").strip()
    match = _LEADING_OPTION_PATTERN.match(text)
    selected = match.group(1).upper() if match else None
    answer_text = match.group(2).strip() if match else ""
    words = _WORD_PATTERN.findall(answer_text)
    sentences = _sentence_count(answer_text)
    expected = _normalized_option(correct_option)
    return {
        "selected_option": selected,
        "answer_text": answer_text,
        "answer_word_count": len(words),
        "answer_sentence_count": sentences,
        "format_compliant": bool(
            selected is not None and len(words) >= 5 and 1 <= sentences <= 2
        ),
        "choice_correct": (
            selected == expected if selected is not None and expected is not None else None
        ),
    }


def measure_prompt_response(
    response: str | None, correct_option: Any, prompt_variant: str
) -> dict[str, Any]:
    """Measure answer length, option correctness, and instructed format."""

    text = str(response or "").strip()
    if prompt_variant not in {"mcq", "mcq2"}:
        answer_text = text
        selected = None
        choice_correct = None
        format_compliant = None
    else:
        parsed = parse_vdc_mcq2_response(text, correct_option)
        answer_text = parsed["answer_text"]
        selected = parsed["selected_option"]
        choice_correct = parsed["choice_correct"]
        format_compliant = (
            parsed["format_compliant"]
            if prompt_variant == "mcq2"
            else bool(selected is not None and not answer_text)
        )
    return {
        "selected_option": selected,
        "answer_text": answer_text,
        "answer_word_count": len(_WORD_PATTERN.findall(answer_text)),
        "answer_sentence_count": _sentence_count(answer_text),
        "choice_correct": choice_correct,
        "expected_format_compliant": format_compliant,
    }

