"""Collect and plot metrics already recorded in DFlash experiment journals.

This module is intentionally descriptive.  It does not calculate hypothesis
contrasts such as ``I_format``, early/late deltas, or depth interactions.  It
only flattens fields emitted by the runtime, summarizes them by raw condition,
and plots those condition summaries for the H2.1, H3.1, and H3.2 journals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from src.analyze.Validate_Sparrow_hypothesises.mcq2_metrics import measure_prompt_response


ACCEPTANCE_FIELDS = (
    "accepted_prefix_tokens",
    "accepted_effective_tokens",
    "tau_proposal",
    "tau_effective",
    "acceptance_rate",
)
TIMING_FIELDS = (
    "speculative_prefill_s",
    "speculative_draft_s",
    "speculative_verify_s",
    "speculative_decode_s",
    "speculative_end_to_end_s",
    "speculative_tokens_per_second",
    "target_prefill_s",
    "target_decode_s",
    "target_end_to_end_s",
    "target_tokens_per_second",
    "speedup_end_to_end",
)
OUTPUT_FIELDS = (
    "speculative_output_tokens",
    "target_output_tokens",
    "lossless",
    "lossless_prefix_length",
    "output_hash_match",
)
VISUAL_FIELDS = (
    "actual_visual_tokens",
    "draft_context_visual_tokens",
    "target_visual_tokens",
)
ATTENTION_FIELDS = (
    "visual_mass",
    "question_text_mass",
    "answer_hint_mass",
    "visual_density_ratio_vs_context_uniform",
    "question_text_density_ratio_vs_context_uniform",
    "answer_hint_density_ratio_vs_context_uniform",
)
RESPONSE_FIELDS = (
    "target_answer_word_count",
    "speculative_answer_word_count",
    "target_answer_sentence_count",
    "speculative_answer_sentence_count",
    "target_choice_correct",
    "speculative_choice_correct",
    "target_expected_format_compliant",
    "speculative_expected_format_compliant",
    "max_new_tokens",
    "target_reached_output_cap",
    "speculative_reached_output_cap",
)

RESPONSE_PLOT_FIELDS = (
    "target_answer_word_count",
    "speculative_answer_word_count",
    "target_choice_correct",
    "speculative_choice_correct",
    "target_expected_format_compliant",
    "speculative_expected_format_compliant",
    "target_reached_output_cap",
    "speculative_reached_output_cap",
)

ALL_METRIC_FIELDS = ACCEPTANCE_FIELDS + TIMING_FIELDS + OUTPUT_FIELDS + VISUAL_FIELDS + ATTENTION_FIELDS + RESPONSE_FIELDS

GROUP_FIELDS = {
    "h21": ("prompt_variant", "visual_condition"),
    "h31": ("retention_percentage",),
    "h32": ("dataset", "training_corpus", "draft_depth", "visual_condition"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [int(item) for item in value if isinstance(item, (int, float))]


def _decode(tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str | None:
    if tokenizer is None:
        return None
    kwargs = {
        "skip_special_tokens": skip_special_tokens,
        "clean_up_tokenization_spaces": False,
    }
    try:
        return str(tokenizer.decode(list(token_ids), **kwargs))
    except TypeError:
        kwargs.pop("clean_up_tokenization_spaces")
        return str(tokenizer.decode(list(token_ids), **kwargs))


def load_tokenizer(tokenizer_path: Path | None, rows: Sequence[Mapping[str, Any]]) -> tuple[Any, str | None]:
    """Load a local tokenizer for output text; never download implicitly."""

    candidates = [tokenizer_path] if tokenizer_path is not None else []
    if not candidates:
        for row in rows:
            model = row.get("target_model")
            if model and Path(str(model)).exists():
                candidates.append(Path(str(model)))
                break
    if not candidates:
        return None, None
    from transformers import AutoTokenizer

    path = candidates[0]
    tokenizer = AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )
    return tokenizer, str(path)


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _position_acceptance_rate(row: Mapping[str, Any]) -> float | None:
    """Compute acceptance only over proposal positions retained in the output.

    This is useful for position-wise diagnostics, but it is biased for short
    outputs because proposals beyond the terminal output position are absent.
    """

    trace = row.get("acceptance_by_position") or []
    proposed = 0
    accepted = 0
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        proposed_value = item.get("proposed")
        accepted_value = item.get("accepted")
        if isinstance(proposed_value, (int, float)) and isinstance(accepted_value, (int, float)):
            if proposed_value > 0:
                proposed += int(proposed_value)
                accepted += int(accepted_value)
    if proposed == 0:
        return None
    return accepted / proposed


def _acceptance_rate(row: Mapping[str, Any]) -> float | None:
    """Compute global proposal acceptance from complete speculative rounds.

    DFlash round records retain the full ``proposal_count`` even when the
    target output terminates early. Prefer these records so short MCQ outputs
    do not discard proposals from the denominator. MSD's current flat trace
    does not expose ``proposal_count``; for that backend, retain the legacy
    position-based fallback until its runner emits the round denominator.
    """

    rounds = row.get("acceptance_rounds") or []
    round_proposed = 0
    round_accepted = 0
    has_round_counts = False
    for item in rounds:
        if not isinstance(item, Mapping):
            continue
        proposal_count = item.get("proposal_count")
        matched_proposals = item.get("matched_proposals")
        if not isinstance(proposal_count, (int, float)):
            continue
        if not isinstance(matched_proposals, (int, float)):
            continue
        if proposal_count <= 0:
            continue
        has_round_counts = True
        round_proposed += int(proposal_count)
        round_accepted += int(matched_proposals)
    if has_round_counts and round_proposed > 0:
        return round_accepted / round_proposed
    return _position_acceptance_rate(row)


def flatten_row(row: Mapping[str, Any], *, source: Path, row_index: int) -> dict[str, Any]:
    """Flatten runtime-emitted acceptance, timing, output, and visual fields."""

    retention_percentage = row.get("retention_percentage")
    visual_condition = row.get("visual_condition")
    if visual_condition is None and row.get("condition") == "retention" and retention_percentage is not None:
        visual_condition = f"retention-{float(retention_percentage):g}"
    flat: dict[str, Any] = {
        "source": str(source),
        "row_index": row_index,
        "status": row.get("status"),
        "sample_id": row.get("sample_id"),
        "row_id": row.get("row_id"),
        "prompt_variant": row.get("prompt_variant", "natural"),
        "correct_option": row.get("correct_option"),
        "target_output_text": row.get("target_output_text", ""),
        "speculative_output_text": row.get("speculative_output_text", ""),
        "max_new_tokens": row.get("max_new_tokens"),
        "visual_condition": visual_condition,
        "retention_percentage": retention_percentage,
        "dataset": row.get("dataset"),
        "training_corpus": row.get("training_corpus"),
        "draft_depth": row.get("draft_depth"),
        "target_model": row.get("target_model"),
        "acceptance_by_position": row.get("acceptance_by_position") or [],
        "acceptance_rounds": row.get("acceptance_rounds") or [],
        "target_output_ids": _int_list(row.get("target_output_ids")),
        "speculative_output_ids": _int_list(row.get("speculative_output_ids")),
    }
    for field in (
        "accepted_prefix_tokens",
        "accepted_effective_tokens",
        "actual_visual_tokens",
        "draft_context_visual_tokens",
        "target_visual_tokens",
        "lossless_prefix_length",
    ):
        flat[field] = row.get(field)
    metrics = row.get("metrics") or {}
    for field in ("tau_proposal", "tau_effective", "speculative_output_tokens", "target_output_tokens"):
        flat[field] = metrics.get(field)
    if flat["speculative_output_tokens"] is None:
        flat["speculative_output_tokens"] = len(flat["speculative_output_ids"])
    if flat["target_output_tokens"] is None:
        flat["target_output_tokens"] = len(flat["target_output_ids"])
    speculative_timing = (row.get("timing") or {}).get("speculative") or {}
    target_timing = (row.get("timing") or {}).get("target") or {}
    for field in ("prefill_s", "draft_s", "verify_s", "decode_s", "end_to_end_s", "tokens_per_second"):
        flat[f"speculative_{field}"] = speculative_timing.get(field)
    for field in ("prefill_s", "decode_s", "end_to_end_s", "tokens_per_second"):
        flat[f"target_{field}"] = target_timing.get(field)
    flat["speedup_end_to_end"] = _nested(row, "speedup", "end_to_end")
    if not speculative_timing and row.get("condition") == "retention":
        flat["speculative_prefill_s"] = row.get("prefill_seconds")
        flat["speculative_draft_s"] = row.get("draft_tree_prefill_seconds")
        flat["speculative_verify_s"] = row.get("verification_seconds")
        flat["speculative_decode_s"] = row.get("decode_seconds")
        flat["speculative_end_to_end_s"] = row.get("end_to_end_seconds")
        decode_seconds = _number(row.get("decode_seconds"))
        output_tokens = _number(flat["speculative_output_tokens"])
        flat["speculative_tokens_per_second"] = (
            output_tokens / decode_seconds
            if output_tokens is not None and decode_seconds and decode_seconds > 0
            else None
        )
        flat["target_prefill_s"] = row.get("ar_prefill_seconds")
        flat["target_decode_s"] = row.get("ar_decode_seconds")
        flat["target_end_to_end_s"] = row.get("ar_end_to_end_seconds")
        ar_decode_seconds = _number(row.get("ar_decode_seconds"))
        flat["target_tokens_per_second"] = (
            output_tokens / ar_decode_seconds
            if output_tokens is not None and ar_decode_seconds and ar_decode_seconds > 0
            else None
        )
        flat["speedup_end_to_end"] = row.get("end_to_end_speedup")
    flat["lossless"] = float(bool(row.get("lossless"))) if row.get("lossless") is not None else None
    target_hash = row.get("target_output_hash")
    speculative_hash = row.get("speculative_output_hash")
    flat["target_output_hash"] = target_hash
    flat["speculative_output_hash"] = speculative_hash
    flat["output_hash_match"] = (
        float(target_hash == speculative_hash)
        if target_hash is not None and speculative_hash is not None
        else None
    )
    flat["acceptance_rate"] = _acceptance_rate(row)
    for output_kind in ("target", "speculative"):
        response_metrics = measure_prompt_response(
            row.get(f"{output_kind}_output_text"),
            row.get("correct_option"),
            str(flat["prompt_variant"]),
        )
        flat[f"{output_kind}_selected_option"] = response_metrics["selected_option"]
        flat[f"{output_kind}_answer_text"] = response_metrics["answer_text"]
        flat[f"{output_kind}_answer_word_count"] = response_metrics["answer_word_count"]
        flat[f"{output_kind}_answer_sentence_count"] = response_metrics["answer_sentence_count"]
        flat[f"{output_kind}_choice_correct"] = (
            float(response_metrics["choice_correct"])
            if response_metrics["choice_correct"] is not None
            else None
        )
        flat[f"{output_kind}_expected_format_compliant"] = (
            float(response_metrics["expected_format_compliant"])
            if response_metrics["expected_format_compliant"] is not None
            else None
        )
    cap = _number(flat["max_new_tokens"])
    for output_kind in ("target", "speculative"):
        output_length = _number(flat[f"{output_kind}_output_tokens"])
        flat[f"{output_kind}_reached_output_cap"] = (
            float(output_length >= cap)
            if output_length is not None and cap is not None
            else None
        )
    attention = row.get("attention_summary") or {}
    for field in ATTENTION_FIELDS:
        flat[field] = attention.get(field)
    return flat


def _group_key(row: Mapping[str, Any], group_fields: Sequence[str]) -> tuple[str, ...]:
    return tuple("unknown" if row.get(field) is None else str(row.get(field)) for field in group_fields)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "median": statistics.median(clean),
        "min": min(clean),
        "max": max(clean),
    }


def group_metrics(rows: Sequence[Mapping[str, Any]], *, group_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Summarize only runtime metrics by raw experimental condition."""

    output: list[dict[str, Any]] = []
    groups = sorted({_group_key(row, group_fields) for row in rows})
    for group in groups:
        selected = [row for row in rows if _group_key(row, group_fields) == group]
        for metric in ALL_METRIC_FIELDS:
            values = [_number(row.get(metric)) for row in selected]
            clean = [value for value in values if value is not None]
            output.append({**dict(zip(group_fields, group)), "metric": metric, **_summary(clean)})
    return output


def metric_coverage(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report whether each recorded metric is present; never impute missing values."""

    total = len(rows)
    output: list[dict[str, Any]] = []
    for metric in ALL_METRIC_FIELDS:
        present = sum(_number(row.get(metric)) is not None for row in rows)
        output.append(
            {
                "metric": metric,
                "total": total,
                "present": present,
                "missing": total - present,
                "coverage": present / total if total else None,
            }
        )
    return output


def position_metrics(rows: Sequence[Mapping[str, Any]], *, group_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Preserve the runtime's position-level proposal/acceptance trace."""

    output: list[dict[str, Any]] = []
    for row in rows:
        trace = row.get("acceptance_by_position") or []
        for item in trace:
            if not isinstance(item, Mapping) or item.get("rate") is None:
                continue
            output.append(
                {
                    **{field: "unknown" if row.get(field) is None else str(row.get(field)) for field in group_fields},
                    "sample_id": row.get("sample_id"),
                    "position": int(item["position"]),
                    "proposed": int(item["proposed"]),
                    "accepted": int(item["accepted"]),
                    "rate": float(item["rate"]),
                }
            )
    return sorted(output, key=lambda row: tuple(str(row.get(field)) for field in group_fields) + (int(row["position"]), str(row["sample_id"])))


def _detail_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": row.get("source"),
        "row_index": row.get("row_index"),
        "sample_id": row.get("sample_id"),
        "row_id": row.get("row_id"),
        "prompt_variant": row.get("prompt_variant"),
        "visual_condition": row.get("visual_condition"),
        "retention_percentage": row.get("retention_percentage"),
        "dataset": row.get("dataset"),
        "training_corpus": row.get("training_corpus"),
        "draft_depth": row.get("draft_depth"),
    }


def output_detail_records(rows: Sequence[Mapping[str, Any]], *, tokenizer: Any = None) -> list[dict[str, Any]]:
    """Persist complete runtime output IDs and decoded output text per sample."""

    output: list[dict[str, Any]] = []
    for row in rows:
        target_ids = _int_list(row.get("target_output_ids"))
        speculative_ids = _int_list(row.get("speculative_output_ids"))
        output.append(
            {
                **_detail_identity(row),
                "target_output_ids": target_ids,
                "speculative_output_ids": speculative_ids,
                "target_output_length": len(target_ids),
                "speculative_output_length": len(speculative_ids),
                "target_output_text": _decode(tokenizer, target_ids, skip_special_tokens=True),
                "speculative_output_text": _decode(tokenizer, speculative_ids, skip_special_tokens=True),
                "target_output_text_with_special_tokens": _decode(
                    tokenizer, target_ids, skip_special_tokens=False
                ),
                "speculative_output_text_with_special_tokens": _decode(
                    tokenizer, speculative_ids, skip_special_tokens=False
                ),
                "target_output_hash": row.get("target_output_hash"),
                "speculative_output_hash": row.get("speculative_output_hash"),
                "lossless": row.get("lossless"),
                "lossless_prefix_length": row.get("lossless_prefix_length"),
            }
        )
    return output


def output_position_records(rows: Sequence[Mapping[str, Any]], *, tokenizer: Any = None) -> list[dict[str, Any]]:
    """Persist acceptance and the target/speculative token text at each output position."""

    output: list[dict[str, Any]] = []
    for row in rows:
        target_ids = _int_list(row.get("target_output_ids"))
        speculative_ids = _int_list(row.get("speculative_output_ids"))
        trace_by_position = {
            int(item["position"]): item
            for item in (row.get("acceptance_by_position") or [])
            if isinstance(item, Mapping) and item.get("position") is not None
        }
        length = max(len(target_ids), len(speculative_ids), (max(trace_by_position) + 1 if trace_by_position else 0))
        for position in range(length):
            trace = trace_by_position.get(position, {})
            target_id = target_ids[position] if position < len(target_ids) else None
            speculative_id = speculative_ids[position] if position < len(speculative_ids) else None
            output.append(
                {
                    **_detail_identity(row),
                    "position": position,
                    "accepted_tokens_at_position": trace.get("accepted"),
                    "proposed_tokens_at_position": trace.get("proposed"),
                    "acceptance_rate_at_position": trace.get("rate"),
                    "target_token_id": target_id,
                    "target_token_text": _decode(
                        tokenizer, [target_id] if target_id is not None else [], skip_special_tokens=False
                    ),
                    "speculative_token_id": speculative_id,
                    "speculative_token_text": _decode(
                        tokenizer,
                        [speculative_id] if speculative_id is not None else [],
                        skip_special_tokens=False,
                    ),
                }
            )
    return output


def acceptance_round_records(rows: Sequence[Mapping[str, Any]], *, tokenizer: Any = None) -> list[dict[str, Any]]:
    """Persist each speculative round's raw acceptance length and corresponding text."""

    output: list[dict[str, Any]] = []
    for row in rows:
        target_ids = _int_list(row.get("target_output_ids"))
        for round_index, item in enumerate(row.get("acceptance_rounds") or []):
            if not isinstance(item, Mapping):
                continue
            block_ids = _int_list(item.get("block_token_ids"))
            draft_ids = _int_list(item.get("draft_proposal_token_ids"))
            accepted_positions = [int(value) for value in item.get("accepted_proposal_positions", [])]
            accepted_ids = [target_ids[position] for position in accepted_positions if 0 <= position < len(target_ids)]
            output.append(
                {
                    **_detail_identity(row),
                    "round_index": round_index,
                    "answer_position_start": item.get("answer_position_start"),
                    "proposal_positions": item.get("proposal_positions", []),
                    "accepted_proposal_positions": accepted_positions,
                    "proposal_count": item.get("proposal_count"),
                    "matched_proposals": item.get("matched_proposals"),
                    "effective_emitted_tokens": item.get("effective_emitted_tokens"),
                    "is_partial_block": item.get("is_partial_block"),
                    "is_terminal": item.get("is_terminal"),
                    "block_token_ids": block_ids,
                    "block_text": _decode(tokenizer, block_ids, skip_special_tokens=False)
                    or item.get("block_text", ""),
                    "draft_proposal_token_ids": draft_ids,
                    "draft_proposal_text": _decode(tokenizer, draft_ids, skip_special_tokens=False)
                    or item.get("draft_proposal_text", ""),
                    "accepted_token_ids": accepted_ids,
                    "accepted_text": _decode(tokenizer, accepted_ids, skip_special_tokens=False),
                    "recorded_block_text": item.get("block_text"),
                    "recorded_draft_proposal_text": item.get("draft_proposal_text"),
                }
            )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_label(row: Mapping[str, Any], group_fields: Sequence[str]) -> str:
    return " · ".join(str(row.get(field, "unknown")) for field in group_fields)


def _plot_metric_panels(summary: Sequence[Mapping[str, Any]], group_fields: Sequence[str], metrics: Sequence[str], output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({_group_label(row, group_fields) for row in summary})
    figure, axes = plt.subplots(len(metrics), 1, figsize=(max(9, len(groups) * 0.8), 3.2 * len(metrics)), squeeze=False)
    for axis, metric in zip(axes[:, 0], metrics):
        rows = [row for row in summary if row["metric"] == metric]
        by_label = {_group_label(row, group_fields): row for row in rows}
        means = [
            float(by_label[group]["mean"])
            if by_label.get(group, {}).get("mean") is not None
            else float("nan")
            for group in groups
        ]
        errors = [
            float(by_label[group]["std"] or 0.0)
            if by_label.get(group, {}).get("std") is not None
            else 0.0
            for group in groups
        ]
        axis.errorbar(range(len(groups)), means, yerr=errors, fmt="o", capsize=3)
        axis.set_ylabel(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    axes[-1, 0].set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_position(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], output: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.5))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_group_label(row, group_fields), []).append(row)
    for label, values in sorted(grouped.items()):
        by_position: dict[int, list[float]] = {}
        for row in values:
            by_position.setdefault(int(row["position"]), []).append(float(row["rate"]))
        positions = sorted(by_position)
        axis.plot(positions, [statistics.fmean(by_position[p]) for p in positions], marker=".", label=label)
    axis.set_xlabel("answer position")
    axis.set_ylabel("runtime acceptance rate")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    if grouped:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_attention(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], output: Path, title: str) -> None:
    metrics = [field for field in ATTENTION_FIELDS if any(_number(row.get(field)) is not None for row in rows)]
    if not metrics:
        return
    summary = group_metrics(rows, group_fields=group_fields)
    _plot_metric_panels(summary, group_fields, metrics, output, title)


def analyze(
    experiment: str,
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    group_fields = GROUP_FIELDS[experiment]
    raw_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in inputs:
        source_rows = read_jsonl(path)
        source_counts[str(path)] = len(source_rows)
        for index, row in enumerate(source_rows):
            is_msd_success = (
                row.get("status") is None
                and row.get("condition") == "retention"
                and row.get("retention_percentage") is not None
            )
            if row.get("status") == "ok" or is_msd_success:
                raw_rows.append(flatten_row(row, source=path, row_index=index))
    summary = group_metrics(raw_rows, group_fields=group_fields)
    positions = position_metrics(raw_rows, group_fields=group_fields)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "metric_records.csv", raw_rows)
    _write_csv(output_dir / "metric_summary.csv", summary)
    coverage = metric_coverage(raw_rows)
    _write_csv(output_dir / "metric_coverage.csv", coverage)
    _write_csv(output_dir / "position_metrics.csv", positions)
    tokenizer, tokenizer_source = load_tokenizer(tokenizer_path, raw_rows)
    _write_jsonl(
        output_dir / "output_detail_records.jsonl",
        output_detail_records(raw_rows, tokenizer=tokenizer),
    )
    _write_jsonl(
        output_dir / "output_position_records.jsonl",
        output_position_records(raw_rows, tokenizer=tokenizer),
    )
    _write_jsonl(
        output_dir / "acceptance_round_records.jsonl",
        acceptance_round_records(raw_rows, tokenizer=tokenizer),
    )
    _plot_metric_panels(
        summary,
        group_fields,
        ACCEPTANCE_FIELDS,
        output_dir / "acceptance_metrics.png",
        f"{experiment}: recorded acceptance metrics by condition",
    )
    _plot_metric_panels(
        summary,
        group_fields,
        TIMING_FIELDS,
        output_dir / "timing_metrics.png",
        f"{experiment}: recorded timing and speed metrics by condition",
    )
    _plot_metric_panels(
        summary,
        group_fields,
        OUTPUT_FIELDS,
        output_dir / "output_metrics.png",
        f"{experiment}: recorded output and losslessness metrics by condition",
    )
    if experiment == "h21":
        _plot_metric_panels(
            summary,
            group_fields,
            RESPONSE_PLOT_FIELDS,
            output_dir / "response_metrics.png",
            "h21: response length, choice correctness, and instruction-format compliance",
        )
    _plot_metric_panels(
        summary,
        group_fields,
        VISUAL_FIELDS,
        output_dir / "visual_token_metrics.png",
        f"{experiment}: recorded visual-token counts by condition",
    )
    _plot_position(positions, group_fields, output_dir / "position_acceptance.png", f"{experiment}: recorded position-wise acceptance")
    if experiment == "h21":
        _plot_attention(raw_rows, group_fields, output_dir / "attention_metrics.png", "h21: recorded attention summary metrics")
    manifest = {
        "experiment": experiment,
        "input_files": [str(path) for path in inputs],
        "source_row_counts": source_counts,
        "rows_ok": len(raw_rows),
        "group_fields": list(group_fields),
        "plots": sorted(path.name for path in output_dir.glob("*.png")),
        "metric_fields": list(ALL_METRIC_FIELDS),
        "metric_coverage": coverage,
        "detail_outputs": [
            "output_detail_records.jsonl",
            "output_position_records.jsonl",
            "acceptance_round_records.jsonl",
        ],
        "tokenizer_source": tokenizer_source,
        "text_decoded": tokenizer is not None,
        "descriptive_only": True,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(GROUP_FIELDS))
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer path for decoded output text; no tokenizer download is attempted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = analyze(
        args.experiment,
        args.input,
        args.output_dir,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps({"experiment": manifest["experiment"], "rows_ok": manifest["rows_ok"], "plots": manifest["plots"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
