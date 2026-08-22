"""Report writer for DFlash validation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .dflash_audit import audit_dflash_rows
from .dflash_contract import DFLASH_LAYER_CUTS, DFLASH_LENGTH_TARGETS, DFLASH_RETENTION_PERCENTAGES


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_dflash_report(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write DFlash audit JSON and a semantic-status-separated Markdown report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    audit = audit_dflash_rows(
        materialized,
        expected_grid={
            "length_targets": DFLASH_LENGTH_TARGETS,
            "retention_percentages": DFLASH_RETENTION_PERCENTAGES,
            "attention_targets": (400, 3000),
            "layer_cuts": DFLASH_LAYER_CUTS,
        },
    )
    (output / "dflash_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    model = next((row.get("target_model") for row in materialized if row.get("target_model")), "unknown")
    checkpoint = next(
        (row.get("draft_checkpoint") for row in materialized if row.get("draft_checkpoint")),
        "unknown",
    )
    lines = [
        "# DFlash Qwen2.5-VL Sparrow/MSD Validation",
        "",
        "This report is produced by the isolated DFlash backend; it is not an MSD report.",
        "",
        f"- Target model: `{model}`",
        f"- Draft checkpoint: `{checkpoint}`",
        f"- Valid rows: **{audit['valid_rows']}**",
        f"- Error rows: **{len(audit['error_rows'])}**",
        f"- Contract-invalid rows: **{len(audit['invalid_rows'])}**",
        f"- Lossless decode rows: **{audit['lossless_rows']}**",
        f"- Coverage valid: **{audit['coverage_valid']}**",
        "",
        "## Semantic coverage",
        "",
        "| Semantic status | Rows |",
        "|---|---:|",
    ]
    for status, count in sorted(audit["semantic_status_counts"].items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## Experiment coverage", "", "| Experiment | Rows |", "|---|---:|"])
    for experiment, count in sorted(audit["experiment_counts"].items()):
        lines.append(f"| `{experiment}` | {count} |")
    if audit["coverage_gaps"]:
        lines.extend(["", "## Missing requested milestones", ""])
        for key, values in sorted(audit["coverage_gaps"].items()):
            lines.append(f"- `{key}`: {values}")
    if audit["retention_fingerprint_errors"]:
        lines.extend([
            "",
            "## Retention fingerprint errors",
            "",
            *[f"- `{sample_id}`" for sample_id in audit["retention_fingerprint_errors"]],
        ])
    if metadata:
        lines.extend(["", "## Run metadata", "", "```json", json.dumps(dict(metadata), indent=2, default=_json_default), "```"])
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


__all__ = ["write_dflash_report"]
