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
        if row.get("target_model") not in {contract.msd_target_model, contract.layer_target_model}:
            _add(issues, "error", "model_mismatch", "target model is not in the paper contract", row_id)
        for field in ("target_visual_tokens", "actual_visual_tokens"):
            value = row.get(field)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0):
                _add(issues, "error", "invalid_token_count", f"invalid {field}", row_id)
        if row.get("target_input_fingerprint") != row.get("target_input_fingerprint_reference", row.get("target_input_fingerprint")):
            _add(issues, "error", "target_input_changed", "target input changed across draft conditions", row_id)
        if row.get("paper_figure") == "Figure 1(b)" and row.get("target_visual_tokens") != row.get("full_target_visual_tokens"):
            _add(issues, "error", "target_draft_leak", "retention ablation changed target visual input", row_id)
        if row.get("paper_figure") == "Figure 2" and row.get("attention_query") != "last_instruction":
            _add(issues, "error", "wrong_attention_query", "attention probe is not the final instruction query", row_id)
        if row.get("paper_figure") == "Figure 3" and row.get("visual_kv_masked_from") is None:
            _add(issues, "error", "missing_layer_intervention", "layer probe has no visual KV metadata", row_id)
        if len(issues) == before:
            valid_rows += 1
    return AuditReport(not any(issue.severity == "error" for issue in issues) and bool(rows), len(rows), valid_rows, issues)


def audit_losslessness(rows: Iterable[Mapping[str, Any]]) -> AuditReport:
    issues: list[AuditIssue] = []
    rows = list(rows)
    valid = 0
    for row in rows:
        row_id = str(row.get("row_id")) if row.get("row_id") is not None else None
        target = row.get("target_output_ids")
        speculative = row.get("speculative_output_ids")
        if target is None or speculative is None:
            _add(issues, "error", "missing_output_ids", "losslessness requires token IDs", row_id)
        elif list(target) != list(speculative):
            _add(issues, "error", "lossless_mismatch", "target and speculative token IDs differ", row_id)
        else:
            valid += 1
    return AuditReport(not issues and bool(rows), len(rows), valid, issues)
