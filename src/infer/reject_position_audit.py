"""Audit DFlash speculative-decoding reject positions.

The rule-based extractor is the source of truth for token positions.  Optional
LM reviewers only audit that result in blind and informed passes; they never
overwrite the numeric rule result.

Example::

    python -m src.infer.reject_position_audit \
        --input-dir results/infer/qwen25vl_3b_dflash_vdc50_8frames_isolated_20260820 \
        --output-dir results/infer/qwen25vl_3b_dflash_vdc50_8frames_isolated_20260820/reject_position_audit

For a self-hosted OpenAI-compatible endpoint, add ``--lm-backend http`` plus
``--lm-endpoint`` and ``--lm-model``.  For a local Transformers checkpoint,
use ``--lm-backend transformers`` and ``--lm-model``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib import request as urllib_request


REGIONS = ("early", "middle", "late")
LM_MODES = ("blind", "informed", "both")


class LMReviewer(Protocol):
    """Minimal interface implemented by optional LM backends."""

    def review(self, payload: Mapping[str, Any], *, informed: bool) -> Mapping[str, Any]:
        ...


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _region(relative_position: float | None) -> str | None:
    if relative_position is None:
        return None
    if relative_position < 1 / 3:
        return "early"
    if relative_position < 2 / 3:
        return "middle"
    return "late"


def _output_token_count(sample: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> int:
    value = checkpoint.get("num_output_tokens")
    if value is not None:
        return _as_int(value, "num_output_tokens")
    output_tokens = checkpoint.get("output_tokens")
    if isinstance(output_tokens, Sequence) and not isinstance(output_tokens, (str, bytes)):
        return len(output_tokens)
    baseline = sample.get("target_baseline", {})
    baseline_tokens = baseline.get("output_tokens") if isinstance(baseline, Mapping) else None
    if isinstance(baseline_tokens, Sequence) and not isinstance(baseline_tokens, (str, bytes)):
        return len(baseline_tokens)
    raise ValueError("checkpoint has no num_output_tokens/output_tokens fallback")


def validate_round_fields(round_data: Mapping[str, Any]) -> list[str]:
    """Validate the acceptance fields needed for position extraction."""

    errors: list[str] = []
    values: dict[str, int] = {}
    for field in ("proposal_count", "matched_proposals", "effective_emitted_tokens"):
        if field not in round_data:
            errors.append(f"missing {field}")
            continue
        try:
            values[field] = _as_int(round_data[field], field)
        except ValueError as exc:
            errors.append(str(exc))

    if values.get("proposal_count", 0) < 0:
        errors.append("proposal_count must be non-negative")
    if values.get("matched_proposals", 0) < 0:
        errors.append("matched_proposals must be non-negative")
    if values.get("effective_emitted_tokens", 0) < 0:
        errors.append("effective_emitted_tokens must be non-negative")
    if values.get("matched_proposals", 0) > values.get("proposal_count", 0):
        errors.append("matched_proposals cannot exceed proposal_count")
    if (
        "matched_proposals" in values
        and "effective_emitted_tokens" in values
        and values["effective_emitted_tokens"] != values["matched_proposals"] + 1
    ):
        errors.append("effective_emitted_tokens must equal matched_proposals + 1")

    proposal_ids = round_data.get("draft_proposal_token_ids")
    if proposal_ids is not None and isinstance(proposal_ids, Sequence):
        if "proposal_count" in values and len(proposal_ids) != values["proposal_count"]:
            errors.append("draft_proposal_token_ids length differs from proposal_count")
    return errors


def extract_rule_event(
    sample: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    round_index: int,
    previous_emitted: int,
) -> dict[str, Any]:
    """Extract one round and its exact first-reject position.

    ``previous_emitted`` is the sum of ``effective_emitted_tokens`` from
    earlier rounds.  The decoder emits one initial target anchor before the
    first acceptance round, hence the ``1`` in the absolute position formula.
    Generated positions are zero-based.
    """

    rounds = checkpoint.get("acceptance", {}).get("acceptance_rounds", [])
    if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
        raise ValueError("checkpoint acceptance_rounds must be a sequence")
    if round_index < 0 or round_index >= len(rounds):
        raise IndexError(f"round_index {round_index} outside {len(rounds)} rounds")
    round_data = rounds[round_index]
    if not isinstance(round_data, Mapping):
        raise ValueError("acceptance round must be an object")

    issues = validate_round_fields(round_data)
    proposal_count = _as_int(round_data.get("proposal_count", 0), "proposal_count")
    matched = _as_int(round_data.get("matched_proposals", 0), "matched_proposals")
    emitted = _as_int(
        round_data.get("effective_emitted_tokens", matched + 1),
        "effective_emitted_tokens",
    )
    output_count = _output_token_count(sample, checkpoint)
    if output_count <= 0:
        issues.append("num_output_tokens must be positive")
    if previous_emitted < 0:
        issues.append("previous_emitted must be non-negative")

    reject = matched < proposal_count
    reject_index = matched if reject else None
    generated_position = None
    relative_position = None
    region = None
    if reject and not issues:
        generated_position = 1 + previous_emitted + matched
        relative_position = generated_position / max(output_count - 1, 1)
        relative_position = min(max(relative_position, 0.0), 1.0)
        region = _region(relative_position)

    sample_id = sample.get("sample_id")
    checkpoint_label = checkpoint.get("label") or checkpoint.get("checkpoint")
    return {
        "sample_id": str(sample_id) if sample_id is not None else "unknown",
        "checkpoint": str(checkpoint_label) if checkpoint_label is not None else "unknown",
        "question": sample.get("question"),
        "reference": sample.get("reference"),
        "round_index": int(round_index),
        "proposal_count": proposal_count,
        "matched_proposals": matched,
        "effective_emitted_tokens": emitted,
        "previous_effective_emitted_tokens": int(previous_emitted),
        "num_output_tokens": output_count,
        "rule_reject": bool(reject),
        "rule_reject_index_in_block": reject_index,
        "rule_generated_position_0based": generated_position,
        "rule_generated_position_1based": generated_position + 1 if generated_position is not None else None,
        "rule_relative_position": relative_position,
        "rule_region": region,
        "is_partial_block": bool(round_data.get("is_partial_block", False)),
        "is_terminal": bool(round_data.get("is_terminal", False)),
        "draft_proposal_token_ids": list(round_data.get("draft_proposal_token_ids", [])),
        "draft_proposal_text": round_data.get("draft_proposal_text", ""),
        "block_token_ids": list(round_data.get("block_token_ids", [])),
        "block_text": round_data.get("block_text", ""),
        "next_anchor_text": round_data.get("next_anchor_text", ""),
        "rule_issues": issues,
    }


def _sample_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    paths = sorted(root.glob("sample_*.json"))
    if not paths:
        raise FileNotFoundError(f"no sample_*.json files found under {root}")
    return paths


def extract_rule_rounds(
    input_dir: str | Path,
    checkpoint: str | None = None,
) -> list[dict[str, Any]]:
    """Extract all checkpoint acceptance rounds from per-sample reports."""

    records: list[dict[str, Any]] = []
    for path in _sample_paths(input_dir):
        sample = json.loads(path.read_text(encoding="utf-8"))
        checkpoints = sample.get("checkpoints", [])
        if not isinstance(checkpoints, Sequence):
            raise ValueError(f"{path}: checkpoints must be a sequence")
        for checkpoint_data in checkpoints:
            if not isinstance(checkpoint_data, Mapping):
                raise ValueError(f"{path}: checkpoint must be an object")
            label = checkpoint_data.get("label") or checkpoint_data.get("checkpoint")
            if checkpoint is not None and label != checkpoint:
                continue
            acceptance = checkpoint_data.get("acceptance", {})
            rounds = acceptance.get("acceptance_rounds", []) if isinstance(acceptance, Mapping) else []
            if not isinstance(rounds, Sequence):
                raise ValueError(f"{path}: acceptance_rounds must be a sequence")
            previous_emitted = 0
            for round_index in range(len(rounds)):
                event = extract_rule_event(sample, checkpoint_data, round_index, previous_emitted)
                event["source_file"] = str(path)
                records.append(event)
                previous_emitted += event["effective_emitted_tokens"]
    if not records:
        raise ValueError("no matching checkpoint acceptance rounds found")
    return records


def _numeric_stats(values: Iterable[float | int]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def _summary_for_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if not row.get("rule_issues")]
    rejects = [row for row in valid if row.get("rule_reject")]
    no_reject = [row for row in valid if not row.get("rule_reject")]
    regions = {region: sum(row.get("rule_region") == region for row in rejects) for region in REGIONS}
    return {
        "total_rounds": len(records),
        "valid_rounds": len(valid),
        "invalid_rounds": len(records) - len(valid),
        "reject_rounds": len(rejects),
        "no_reject_rounds": len(no_reject),
        "negative_control_rounds_available": len(no_reject),
        "reject_rate": len(rejects) / len(valid) if valid else None,
        "partial_rounds": sum(bool(row.get("is_partial_block")) for row in records),
        "terminal_rounds": sum(bool(row.get("is_terminal")) for row in records),
        "region_counts": regions,
        "reject_relative_position": _numeric_stats(
            row["rule_relative_position"] for row in rejects if row.get("rule_relative_position") is not None
        ),
        "reject_index_in_block": _numeric_stats(
            row["rule_reject_index_in_block"] for row in rejects if row.get("rule_reject_index_in_block") is not None
        ),
    }


def _summarize_lm(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cross = [row.get("cross_validation") for row in records if row.get("cross_validation")]
    if not cross:
        return {"reviewed_rounds": 0}
    statuses = Counter(str(row.get("final_status")) for row in cross)
    blind_agreement = [row.get("blind_reject_agreement") for row in cross if row.get("blind_reject_agreement") is not None]
    position_agreement = [row.get("blind_position_agreement") for row in cross if row.get("blind_position_agreement") is not None]
    return {
        "reviewed_rounds": len(cross),
        "final_status_counts": dict(sorted(statuses.items())),
        "blind_reject_agreement_rate": sum(blind_agreement) / len(blind_agreement) if blind_agreement else None,
        "blind_position_agreement_rate": sum(position_agreement) / len(position_agreement) if position_agreement else None,
    }


def summarize_rule_events(rounds: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize rule and optional cross-validation results."""

    records = [dict(row) for row in rounds]
    by_checkpoint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_checkpoint[str(row.get("checkpoint", "unknown"))].append(row)
        by_sample[str(row.get("sample_id", "unknown"))].append(row)
    sample_summaries: dict[str, Any] = {}
    for sample_id, rows in sorted(by_sample.items()):
        rejects = [row for row in rows if row.get("rule_reject") and not row.get("rule_issues")]
        positions = [row["rule_relative_position"] for row in rejects if row.get("rule_relative_position") is not None]
        sample_summaries[sample_id] = {
            "checkpoint_count": len({row.get("checkpoint") for row in rows}),
            "rounds": len(rows),
            "reject_rounds": len(rejects),
            "first_reject_relative_position": min(positions) if positions else None,
            "last_reject_relative_position": max(positions) if positions else None,
        }
    summary = _summary_for_group(records)
    summary.update(
        {
            "checkpoint_summaries": {
                label: _summary_for_group(rows) for label, rows in sorted(by_checkpoint.items())
            },
            "sample_summaries": sample_summaries,
            "lm_cross_validation": _summarize_lm(records),
        }
    )
    return summary


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _markdown_report(summary: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    lines = [
        "# DFlash Reject-Position Audit",
        "",
        "Exact numeric positions come from acceptance token metadata. LM results are audit signals and never overwrite rule-based positions.",
        "",
        f"- Input: `{metadata.get('input_dir', 'unknown')}`",
        f"- Samples: **{metadata.get('sample_count', 'unknown')}**",
        f"- Checkpoints: **{metadata.get('checkpoint_count', 'unknown')}**",
        f"- Total rounds: **{summary.get('total_rounds', 0)}**",
        f"- Reject rounds: **{summary.get('reject_rounds', 0)}**",
        f"- Reject rate: **{summary.get('reject_rate', 0):.4f}**" if summary.get("reject_rate") is not None else "- Reject rate: **n/a**",
        "",
        "## Overall position regions",
        "",
        "| Region | Reject events |",
        "|---|---:|",
    ]
    for region in REGIONS:
        lines.append(f"| {region} | {summary.get('region_counts', {}).get(region, 0)} |")
    lines.extend(["", "## By checkpoint", "", "| Checkpoint | Rounds | Rejects | Reject rate | Early | Middle | Late |", "|---|---:|---:|---:|---:|---:|---:|"])
    for label, item in sorted(summary.get("checkpoint_summaries", {}).items()):
        rate = item.get("reject_rate")
        rate_text = f"{rate:.4f}" if isinstance(rate, (int, float)) else "n/a"
        counts = item.get("region_counts", {})
        lines.append(
            f"| `{label}` | {item.get('total_rounds', 0)} | {item.get('reject_rounds', 0)} | {rate_text} | "
            f"{counts.get('early', 0)} | {counts.get('middle', 0)} | {counts.get('late', 0)} |"
        )
    lm_summary = summary.get("lm_cross_validation", {})
    lines.extend(["", "## LM cross-validation", ""])
    if not lm_summary.get("reviewed_rounds"):
        lines.append("No LM reviews were run. Use `--lm-backend http` or `--lm-backend transformers` for live review, or `--lm-backend jsonl` for offline replay.")
        if summary.get("negative_control_rounds_available", 0) == 0:
            lines.append("No full-acceptance/no-reject rounds were available as negative controls in this source run.")
    else:
        lines.extend([
            f"- Reviewed rounds: **{lm_summary.get('reviewed_rounds')}**",
            f"- Blind reject agreement: **{lm_summary.get('blind_reject_agreement_rate')}**",
            f"- Blind position agreement: **{lm_summary.get('blind_position_agreement_rate')}**",
            f"- Final statuses: `{json.dumps(lm_summary.get('final_status_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        ])
    lines.extend(
        [
            "",
            "## Position definition",
            "",
            "A reject exists when `matched_proposals < proposal_count`; the first rejected proposal index is `matched_proposals` (zero-based).",
            "The generated output index is `1 + sum(previous effective_emitted_tokens) + matched_proposals`; the initial `1` is the pre-round target anchor.",
            "Relative position is the generated index divided by `max(num_output_tokens - 1, 1)`. Early/middle/late are presentation bins at 1/3 and 2/3.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_audit_artifacts(
    rounds: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write rule rounds, reject events, summary JSON, and Markdown report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in rounds]
    metadata_dict = dict(metadata or {})
    metadata_dict.setdefault("sample_count", len({row.get("sample_id") for row in records}))
    metadata_dict.setdefault("checkpoint_count", len({row.get("checkpoint") for row in records}))
    summary = summarize_rule_events(records)
    rounds_path = output / "rounds.jsonl"
    rejects_path = output / "reject_events.jsonl"
    summary_path = output / "rule_summary.json"
    report_path = output / "reject_position_report.md"
    _jsonl_write(rounds_path, records)
    _jsonl_write(rejects_path, (row for row in records if row.get("rule_reject")))
    summary_path.write_text(
        json.dumps({"metadata": metadata_dict, **summary}, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(summary, metadata_dict), encoding="utf-8")
    return {"rounds": rounds_path, "reject_events": rejects_path, "summary": summary_path, "report": report_path}


def _prompt_payload(record: Mapping[str, Any], *, include_rule: bool) -> dict[str, Any]:
    fields = {
        "sample_id": record.get("sample_id"),
        "checkpoint": record.get("checkpoint"),
        "round_index": record.get("round_index"),
        "question": record.get("question"),
        "reference": record.get("reference"),
        "proposal_count": record.get("proposal_count"),
        "draft_proposal_token_ids": record.get("draft_proposal_token_ids", []),
        "draft_proposal_text": record.get("draft_proposal_text", ""),
        "block_token_ids": record.get("block_token_ids", []),
        "block_text": record.get("block_text", ""),
        "next_anchor_text": record.get("next_anchor_text", ""),
        "is_partial_block": record.get("is_partial_block", False),
        "is_terminal": record.get("is_terminal", False),
    }
    if include_rule:
        fields.update(
            {
                "rule_reject": record.get("rule_reject"),
                "rule_reject_index_in_block": record.get("rule_reject_index_in_block"),
                "rule_generated_position_0based": record.get("rule_generated_position_0based"),
                "rule_relative_position": record.get("rule_relative_position"),
                "rule_region": record.get("rule_region"),
                "rule_issues": record.get("rule_issues", []),
            }
        )
    return fields


def _prompt_from_payload(payload: Mapping[str, Any], *, informed: bool) -> str:
    policy = (
        "You are reviewing a speculative decoding acceptance round. Identify whether the draft proposal sequence first diverges from the target accepted prefix. "
        "Use zero-based proposal indexes. Do not treat later proposals after the first divergence as separate rejects."
    )
    if informed:
        policy += " The rule-based fields are a proposed deterministic result; audit whether they are internally consistent, but do not silently change them."
    else:
        policy += " Make an independent judgment from the raw round evidence; no rule-based conclusion is provided."
    schema = (
        '{"reject_detected": true|false, "reject_position_in_block": integer|null, '
        '"position_region": "early"|"middle"|"late"|null, "confidence": "high"|"medium"|"low", '
        '"rule_result_valid": true|false|null, "agreement": true|false|null, '
        '"position_consistent": true|false|null, "reason": "at most 5 words"}'
    )
    return f"{policy}\nReturn compact JSON only with this schema; keep reason to at most 5 words:\n{schema}\nEvidence:\n{json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)}"


def build_blind_prompt(record: Mapping[str, Any]) -> str:
    """Build an LM prompt without rule-derived conclusions."""

    return _prompt_from_payload(_prompt_payload(record, include_rule=False), informed=False)


def build_informed_prompt(record: Mapping[str, Any]) -> str:
    """Build an LM prompt containing the rule definition and derived result."""

    return _prompt_from_payload(_prompt_payload(record, include_rule=True), informed=True)


def parse_lm_json(value: str | Mapping[str, Any]) -> dict[str, Any]:
    """Parse JSON returned by a local LM, tolerating a single fenced block."""

    if isinstance(value, Mapping):
        return dict(value)
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("LM response is not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("LM JSON response must be an object")
    return dict(parsed)


def review_lm_response(response: Mapping[str, Any], rule_record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one LM response against rule field bounds."""

    result = dict(response)
    issues: list[str] = []
    detected = response.get("reject_detected")
    if not isinstance(detected, bool):
        issues.append("reject_detected must be boolean")
        detected = None
    position = response.get("reject_position_in_block")
    if position is not None:
        try:
            position = _as_int(position, "reject_position_in_block")
        except ValueError as exc:
            issues.append(str(exc))
            position = None
    proposal_count = int(rule_record.get("proposal_count", 0))
    if position is not None and not 0 <= position < proposal_count:
        issues.append("reject_position_in_block is outside proposal_count")
    if detected is False and position is not None:
        issues.append("non-reject LM response cannot include a reject position")
    region = response.get("position_region")
    if region not in (*REGIONS, None):
        issues.append("position_region must be early, middle, late, or null")
    if detected is False and region not in (None, ""):
        issues.append("non-reject LM response cannot include a position region")
    confidence = response.get("confidence")
    if confidence is not None and confidence not in {"high", "medium", "low"}:
        issues.append("confidence must be high, medium, or low")
    result.update(
        {
            "reject_detected": detected,
            "reject_position_in_block": position,
            "position_region": None if region == "" else region,
            "valid": not issues,
            "issues": issues,
        }
    )
    return result


def cross_validate_round(
    rule_record: Mapping[str, Any],
    blind: Mapping[str, Any] | None,
    informed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare blind/informed LM results while preserving rule authority."""

    output: dict[str, Any] = {}
    blind_checked = review_lm_response(blind, rule_record) if blind is not None else None
    if blind_checked is not None:
        output["blind_response"] = blind_checked
        rule_reject = bool(rule_record.get("rule_reject"))
        output["blind_reject_agreement"] = (
            blind_checked["valid"] and blind_checked.get("reject_detected") == rule_reject
        )
        if rule_reject and blind_checked.get("reject_detected"):
            output["blind_position_agreement"] = (
                blind_checked.get("reject_position_in_block") == rule_record.get("rule_reject_index_in_block")
            )
        else:
            output["blind_position_agreement"] = (
                blind_checked["valid"] and not rule_reject and not blind_checked.get("reject_detected")
            )
    else:
        output["blind_reject_agreement"] = None
        output["blind_position_agreement"] = None

    if informed is not None:
        informed_checked = review_lm_response(informed, rule_record)
        output["informed_response"] = informed_checked
        informed_agreement = informed_checked.get("agreement")
        informed_valid = informed_checked.get("rule_result_valid")
        output["informed_rule_agreement"] = (
            informed_checked["valid"]
            and informed_agreement is True
            and informed_valid is True
            and informed_checked.get("position_consistent") is True
        )
        output["informed_lm_valid"] = informed_checked["valid"]
    else:
        output["informed_rule_agreement"] = None
        output["informed_lm_valid"] = None

    has_blind = blind_checked is not None
    blind_ok = (
        not has_blind
        or (
            blind_checked["valid"]
            and output["blind_reject_agreement"] is True
            and output["blind_position_agreement"] is True
        )
    )
    informed_ok = informed is None or output["informed_rule_agreement"] is True
    invalid_lm = (
        (blind_checked is not None and not blind_checked["valid"])
        or (informed is not None and output["informed_lm_valid"] is False)
    )
    if invalid_lm:
        status = "lm_invalid"
    elif not has_blind and informed is None:
        status = "rule_only"
    elif blind_ok and informed_ok:
        status = "validated"
    else:
        status = "disagreement"
    output["final_status"] = status
    return output


class NullLMReviewer:
    """No-op backend used by default and in CPU-only tests."""

    def review(self, payload: Mapping[str, Any], *, informed: bool) -> Mapping[str, Any]:
        del payload, informed
        return {"skipped": True}


class JsonlLMReviewer:
    """Replay deterministic LM responses from JSONL for offline validation."""

    def __init__(self, path: str | Path):
        self.responses: dict[tuple[str, str, int, str], dict[str, Any]] = {}
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                key = (str(row["sample_id"]), str(row["checkpoint"]), int(row["round_index"]), str(row["phase"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid LM response key at line {line_number}") from exc
            self.responses[key] = dict(row.get("response", row))

    def review(self, payload: Mapping[str, Any], *, informed: bool) -> Mapping[str, Any]:
        phase = "informed" if informed else "blind"
        key = (str(payload.get("sample_id")), str(payload.get("checkpoint")), int(payload.get("round_index", -1)), phase)
        return self.responses.get(key, {"skipped": True, "missing_replay": True})


class OpenAICompatibleLMReviewer:
    """Call a self-hosted OpenAI-compatible chat-completions endpoint."""

    def __init__(self, endpoint: str, model: str, timeout: float = 120.0):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    def review(self, payload: Mapping[str, Any], *, informed: bool) -> Mapping[str, Any]:
        prompt = _prompt_from_payload(payload, informed=informed)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib_request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return parse_lm_json(content)


class TransformersLMReviewer:
    """Lazy local Transformers reviewer for a self-hosted checkpoint."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        batch_size: int = 4,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("--lm-backend transformers requires torch and transformers") from exc
        self._torch = torch
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype_value = getattr(torch, dtype, None)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device_map:
            kwargs["device_map"] = device_map
        if dtype_value is not None:
            kwargs["torch_dtype"] = dtype_value
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        except (ImportError, ValueError, RuntimeError):  # pragma: no cover - model dependent
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **kwargs)
        self.model.eval()

    def _format_prompt(self, payload: Mapping[str, Any], *, informed: bool) -> str:
        prompt = _prompt_from_payload(payload, informed=informed)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _parse_generated(self, text: str) -> Mapping[str, Any]:
        try:
            return parse_lm_json(text)
        except ValueError as exc:
            return {"raw_text": text, "parse_error": str(exc)}

    def review_many(self, payloads: Sequence[Mapping[str, Any]], *, informed: bool) -> list[Mapping[str, Any]]:
        if not payloads:
            return []
        prompts = [self._format_prompt(payload, informed=informed) for payload in payloads]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        try:
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
        except StopIteration:  # pragma: no cover - unusual empty model
            pass
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output[:, prompt_length:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [self._parse_generated(text) for text in texts]

    def review(self, payload: Mapping[str, Any], *, informed: bool) -> Mapping[str, Any]:
        return self.review_many([payload], informed=informed)[0]


def select_lm_records(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 0,
    negative_controls_per_checkpoint: int = 8,
) -> list[dict[str, Any]]:
    """Select rejects plus balanced no-reject controls for LM review."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("checkpoint", "unknown"))].append(row)
    selected: list[dict[str, Any]] = []
    for checkpoint, rows in sorted(grouped.items()):
        rejects = [row for row in rows if row.get("rule_reject")]
        controls = [row for row in rows if not row.get("rule_reject")][:negative_controls_per_checkpoint]
        selected.extend(dict(row) for row in rejects + controls)
    selected.sort(key=lambda row: (str(row.get("checkpoint")), str(row.get("sample_id")), int(row.get("round_index", 0))))
    return selected[:limit] if limit > 0 else selected


def _make_reviewer(args: argparse.Namespace) -> LMReviewer:
    if args.lm_backend == "none":
        return NullLMReviewer()
    if args.lm_backend == "jsonl":
        if not args.lm_responses:
            raise ValueError("--lm-responses is required for --lm-backend jsonl")
        return JsonlLMReviewer(args.lm_responses)
    if args.lm_backend == "http":
        if not args.lm_endpoint or not args.lm_model:
            raise ValueError("--lm-endpoint and --lm-model are required for --lm-backend http")
        return OpenAICompatibleLMReviewer(args.lm_endpoint, args.lm_model, args.lm_timeout)
    if args.lm_backend == "transformers":
        if not args.lm_model:
            raise ValueError("--lm-model is required for --lm-backend transformers")
        return TransformersLMReviewer(
            args.lm_model,
            device_map=args.lm_device_map,
            dtype=args.lm_dtype,
            max_new_tokens=args.lm_max_new_tokens,
            batch_size=args.lm_batch_size,
        )
    raise ValueError(f"unsupported LM backend: {args.lm_backend}")


def _apply_lm_reviews(records: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    reviewer = _make_reviewer(args)
    selected = select_lm_records(
        records,
        limit=args.lm_limit,
        negative_controls_per_checkpoint=args.negative_controls_per_checkpoint,
    )
    selected_keys = {
        (row.get("sample_id"), row.get("checkpoint"), row.get("round_index")) for row in selected
    }
    responses: dict[tuple[Any, Any, Any, str], Mapping[str, Any]] = {}

    def review_payloads(payloads: Sequence[Mapping[str, Any]], *, informed: bool) -> list[Mapping[str, Any]]:
        if hasattr(reviewer, "review_many"):
            batch_size = max(1, int(args.lm_batch_size))
            output: list[Mapping[str, Any]] = []
            for start in range(0, len(payloads), batch_size):
                output.extend(reviewer.review_many(payloads[start : start + batch_size], informed=informed))
            return output
        return [reviewer.review(payload, informed=informed) for payload in payloads]

    for informed in (False, True):
        if (informed and args.lm_mode not in {"informed", "both"}) or (
            not informed and args.lm_mode not in {"blind", "both"}
        ):
            continue
        payloads = [_prompt_payload(row, include_rule=informed) for row in selected]
        phase = "informed" if informed else "blind"
        for row, response in zip(selected, review_payloads(payloads, informed=informed), strict=True):
            key = (row.get("sample_id"), row.get("checkpoint"), row.get("round_index"), phase)
            responses[key] = response

    updated: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        key = (row.get("sample_id"), row.get("checkpoint"), row.get("round_index"))
        if key in selected_keys:
            blind = responses.get((*key, "blind"))
            informed = responses.get((*key, "informed"))
            row["cross_validation"] = cross_validate_round(row, blind, informed)
        updated.append(row)
    return updated


def run(args: argparse.Namespace) -> dict[str, Path]:
    records = extract_rule_rounds(args.input_dir, checkpoint=args.checkpoint)
    if args.lm_backend != "none":
        records = _apply_lm_reviews(records, args)
    metadata = {
        "input_dir": str(Path(args.input_dir)),
        "lm_backend": args.lm_backend,
        "lm_mode": args.lm_mode,
    }
    paths = write_audit_artifacts(records, args.output_dir, metadata=metadata)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="directory containing sample_*.json reports")
    parser.add_argument("--output-dir", required=True, help="directory for audit artifacts")
    parser.add_argument("--checkpoint", default=None, help="analyze one checkpoint label; default: all")
    parser.add_argument("--lm-backend", choices=("none", "jsonl", "http", "transformers"), default="none")
    parser.add_argument("--lm-mode", choices=LM_MODES, default="both")
    parser.add_argument("--lm-responses", default=None, help="JSONL responses for the jsonl replay backend")
    parser.add_argument("--lm-endpoint", default=None, help="self-hosted OpenAI-compatible chat-completions URL")
    parser.add_argument("--lm-model", default=None, help="LM model name/path")
    parser.add_argument("--lm-timeout", type=float, default=120.0)
    parser.add_argument("--lm-device-map", default="auto")
    parser.add_argument("--lm-dtype", default="bfloat16")
    parser.add_argument(
        "--lm-max-new-tokens",
        type=int,
        default=64,
        help="maximum tokens generated per Transformers LM review",
    )
    parser.add_argument(
        "--lm-batch-size",
        type=int,
        default=4,
        help="batch size for Transformers LM reviews",
    )
    parser.add_argument("--lm-limit", type=int, default=0, help="maximum rounds sent to LM; 0 means all selected rounds")
    parser.add_argument("--negative-controls-per-checkpoint", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JsonlLMReviewer",
    "LMReviewer",
    "NullLMReviewer",
    "OpenAICompatibleLMReviewer",
    "TransformersLMReviewer",
    "build_blind_prompt",
    "build_informed_prompt",
    "build_parser",
    "cross_validate_round",
    "extract_rule_event",
    "extract_rule_rounds",
    "main",
    "parse_lm_json",
    "review_lm_response",
    "select_lm_records",
    "summarize_rule_events",
    "validate_round_fields",
    "write_audit_artifacts",
]
