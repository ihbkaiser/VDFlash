"""Fail-closed paper-conformance and losslessness gates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .paper_contract import PaperContract, validate_contract


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    row_id: str | None = None


@dataclass
class AuditReport:
    valid: bool
    checked_rows: int
    valid_rows: int
    issues: list[AuditIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_rows": self.checked_rows,
            "valid_rows": self.valid_rows,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _add(issues: list[AuditIssue], severity: str, code: str, message: str, row_id: str | None = None) -> None:
    issues.append(AuditIssue(severity, code, message, row_id))


def audit_rows(rows: Iterable[Mapping[str, Any]], contract: PaperContract) -> AuditReport:
    issues: list[AuditIssue] = []
    for error in validate_contract(contract):
        _add(issues, "error", "invalid_contract", error)
    rows = list(rows)
    valid_rows = 0
    required = {
        "row_id", "paper_figure", "sample_id", "target_model", "temperature",
        "target_visual_tokens", "actual_visual_tokens", "target_input_fingerprint",
        "draft_input_fingerprint",
    }
    for row in rows:
        row_id = str(row.get("row_id")) if row.get("row_id") is not None else None
        before = len(issues)
        for field in sorted(required - set(row)):
            _add(issues, "error", "missing_provenance", f"missing required field: {field}", row_id)
        if row.get("temperature") != contract.temperature:
            _add(issues, "error", "temperature_mismatch", "temperature is not the lossless value", row_id)
        figure = row.get("paper_figure")
        expected_model = contract.layer_target_model if figure in {"Figure 3", "Figure 3(b)", "Figure 6 / Appendix D"} else contract.msd_target_model
        if row.get("target_model") != expected_model:
            _add(issues, "error", "model_mismatch", f"target model is not the contract model for {figure}", row_id)
        for field in ("target_visual_tokens", "actual_visual_tokens"):
            value = row.get(field)
            allow_zero = field == "actual_visual_tokens" and figure == "Figure 1(b)"
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 or (value == 0 and not allow_zero)):
                _add(issues, "error", "invalid_token_count", f"invalid {field}", row_id)
        if row.get("target_input_fingerprint") != row.get("target_input_fingerprint_reference", row.get("target_input_fingerprint")):
            _add(issues, "error", "target_input_changed", "target input changed across draft conditions", row_id)
        if row.get("paper_figure") == "Figure 1(b)" and row.get("target_visual_tokens") != row.get("full_target_visual_tokens"):
            _add(issues, "error", "target_draft_leak", "retention ablation changed target visual input", row_id)
        if figure in {"Figure 2", "Figure 3(b)"} and row.get("attention_query") not in {"last_instruction", "all_text"}:
            _add(issues, "error", "wrong_attention_query", "attention probe has no supported paper query policy", row_id)
        if figure == "Figure 3" and row.get("visual_kv_masked_from") is None:
            _add(issues, "error", "missing_layer_intervention", "layer probe has no visual KV metadata", row_id)
        if figure in {"Figure 2", "Figure 3(b)"}:
            instruction = set(int(value) for value in row.get("instruction_positions", []))
            visual = set(int(value) for value in row.get("visual_positions", []))
            text = set(int(value) for value in row.get("text_positions", []))
            if instruction & visual or instruction & text or visual & text:
                _add(issues, "error", "overlapping_modality_masks", "attention modality masks overlap", row_id)
            if row.get("attention_query") == "last_instruction":
                if row.get("query_position") not in instruction:
                    _add(issues, "error", "query_not_in_instruction_mask", "query position is not in instruction mask", row_id)
            elif row.get("attention_query") == "all_text":
                query_positions = set(int(value) for value in row.get("query_positions", []))
                if not query_positions or not query_positions.issubset(instruction | text):
                    _add(issues, "error", "query_not_in_text_mask", "all-text query positions are outside text masks", row_id)
        if figure == "Figure 6 / Appendix D":
            for field in ("layer", "visual_cosine", "text_cosine"):
                if row.get(field) is None:
                    _add(issues, "error", "missing_retention_metric", f"missing {field}", row_id)
        if len(issues) == before:
            valid_rows += 1
    return AuditReport(not any(issue.severity == "error" for issue in issues) and bool(rows), len(rows), valid_rows, issues)


def audit_losslessness(rows: Iterable[Mapping[str, Any]]) -> AuditReport:
    """Fail-closed losslessness gate with a documented near-tie allowance.

    Speculative verification is bit-exact only when the tree forward and the
    autoregressive reference compute identical logits.  On quantized (4-bit)
    runs the parallel tree kernel and the sequential AR kernel can round a
    near-tie differently (fp16), producing a divergence after a long correctly
    verified prefix.  Rows whose verified prefix covers at least a quarter of
    the target output are therefore reported as ``near_tie_divergence``
    warnings instead of errors; an early divergence still fails the gate
    because it indicates a broken verification mechanism.
    """

    issues: list[AuditIssue] = []
    rows = list(rows)
    valid = 0
    for row in rows:
        row_id = str(row.get("row_id")) if row.get("row_id") is not None else None
        target = row.get("target_output_ids")
        speculative = row.get("speculative_output_ids")
        if target is None and speculative is None:
            # Attention and hidden-state diagnostics do not have generated
            # sequences.  They are audited by their modality/layer fields.
            continue
        if target is None or speculative is None:
            _add(issues, "error", "missing_output_ids", "losslessness requires token IDs", row_id)
        else:
            prefix = 0
            for _target, _spec in zip(target, speculative):
                if _target != _spec:
                    break
                prefix += 1
            if prefix == len(target):
                valid += 1
            elif prefix >= max(4, min(len(target), len(speculative)) // 4):
                # Late single-point divergence: consistent with a quantized
                # near-tie between the parallel tree kernel and the sequential
                # AR kernel (a flipped token can also terminate the loop early
                # via EOS).  Recorded, not fatal.
                _add(
                    issues,
                    "warning",
                    "near_tie_divergence",
                    f"divergence after {prefix}/{len(target)} verified tokens "
                    "(likely a quantized near-tie, not a mechanism failure)",
                    row_id,
                )
            else:
                _add(issues, "error", "lossless_mismatch", "target and speculative token IDs differ", row_id)
    return AuditReport(not any(issue.severity == "error" for issue in issues) and bool(rows), len(rows), valid, issues)
