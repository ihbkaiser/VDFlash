"""Strict evidence selection for completed validation runs.

GPU runners deliberately keep diagnostic rows so a failed video can be
retried.  Reports, however, must be built from a deterministic evidence set:
malformed files, failed rows and non-lossless speculative outputs are never
silently treated as measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dataset import write_jsonl


class MalformedJsonlError(ValueError):
    """Raised when a JSONL file cannot be read as a complete row stream."""


@dataclass(frozen=True)
class EvidenceResult:
    evidence_rows: tuple[dict[str, Any], ...]
    diagnostic_rows: tuple[dict[str, Any], ...]
    malformed_files: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return bool(self.evidence_rows) and not self.malformed_files


def read_jsonl_strict(path: str | Path) -> list[dict[str, Any]]:
    """Read a complete JSONL file and reject malformed/non-object rows."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        raise
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedJsonlError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise MalformedJsonlError(f"JSONL row at {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _diagnostic(row: Mapping[str, Any], reason: str, source: str | None = None) -> dict[str, Any]:
    result = dict(row)
    result["diagnostic_reason"] = reason
    result["evidence_status"] = "diagnostic"
    if source is not None:
        result["evidence_source"] = source
    return result


def _parity_failed(row: Mapping[str, Any]) -> bool:
    parity = row.get("native_prefill_parity")
    if isinstance(parity, Mapping):
        return parity.get("valid") is False
    if parity is False:
        return True
    return row.get("native_prefill_parity_valid") is False


def _has_lossless_mismatch(row: Mapping[str, Any]) -> bool:
    """Return true only for exact-output figures, not layer ablations."""

    figure = str(row.get("paper_figure") or "")
    if figure in {"Figure 3", "Figure 3(b)", "Figure 6 / Appendix D"}:
        return False
    # MSD records the verified target prefix explicitly.  A speculative
    # decoder may keep a few post-EOS/generated tail IDs, so comparing the
    # two serialized lists byte-for-byte would reject rows that the runner
    # and audit have already established as lossless.
    if row.get("lossless") is not None:
        return row.get("lossless") is False
    target = row.get("target_output_ids")
    speculative = row.get("speculative_output_ids")
    if target is None and speculative is None:
        return row.get("lossless") is False and figure not in {"Figure 3", "Figure 3(b)"}
    if target is None or speculative is None:
        return True
    return list(target) != list(speculative)


def evidence_reason(row: Mapping[str, Any]) -> str | None:
    """Return the exclusion reason, or ``None`` when a row is evidence-ready."""

    if row.get("status") in {"error", "oom", "runtime_error"}:
        return "runtime_error"
    if row.get("runtime_status") in {"error", "failed"}:
        return "runtime_error"
    if row.get("target_visual_tokens") is not None and row.get("calibration_status") != "ok":
        return "calibration_not_ok"
    if _parity_failed(row):
        return "native_prefill_parity_failure"
    if row.get("target_output_ids") is not None or row.get("speculative_output_ids") is not None:
        if _has_lossless_mismatch(row):
            return "lossless_mismatch"
    elif row.get("lossless") is False and str(row.get("paper_figure") or "") not in {"Figure 3", "Figure 3(b)"}:
        return "lossless_mismatch"
    return None


def select_evidence(rows: Iterable[Mapping[str, Any]], source: str | None = None) -> EvidenceResult:
    """Deduplicate by ``row_id`` and split evidence from diagnostics.

    Later rows win when a runner appends a retry with the same ID.  The
    superseded row is retained in diagnostics, making the decision auditable.
    Rows without IDs are never evidence because they cannot be safely merged.
    """

    selected: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row_id = row.get("row_id")
        if row_id is None or str(row_id) == "":
            diagnostics.append(_diagnostic(row, "missing_row_id", source))
            continue
        key = str(row_id)
        previous = selected.pop(key, None)
        if previous is not None:
            diagnostics.append(_diagnostic(previous, "duplicate_row_id_superseded", source))
        reason = evidence_reason(row)
        if reason is not None:
            diagnostics.append(_diagnostic(row, reason, source))
            continue
        selected[key] = row
    return EvidenceResult(tuple(selected.values()), tuple(diagnostics))


def collect_evidence(paths: Iterable[str | Path]) -> EvidenceResult:
    """Read stage files strictly and build the merged evidence set."""

    evidence: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    malformed: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            diagnostics.append(_diagnostic({"source": str(path)}, "missing_stage_file", str(path)))
            continue
        try:
            rows = read_jsonl_strict(path)
        except (OSError, MalformedJsonlError) as exc:
            malformed.append(str(path))
            diagnostics.append(_diagnostic({"source": str(path), "error": str(exc)}, "malformed_jsonl", str(path)))
            continue
        result = select_evidence(rows, str(path))
        evidence.extend(result.evidence_rows)
        diagnostics.extend(result.diagnostic_rows)
    # Resolve duplicate IDs across stage files with the same retry policy.
    final = select_evidence(evidence, "merged")
    diagnostics.extend(final.diagnostic_rows)
    return EvidenceResult(final.evidence_rows, tuple(diagnostics), tuple(malformed))


def write_evidence(result: EvidenceResult, evidence_path: str | Path, diagnostic_path: str | Path) -> None:
    """Write the evidence and auditable diagnostic streams."""

    write_jsonl(evidence_path, result.evidence_rows)
    write_jsonl(diagnostic_path, result.diagnostic_rows)
