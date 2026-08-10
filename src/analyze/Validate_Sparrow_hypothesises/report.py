"""Small, dependency-light report writer for completed JSONL runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .audit import AuditReport, audit_losslessness, audit_rows
from .metrics import acceptance_summary
from .paper_contract import PaperContract, paper_contract_rows
from .plots import write_plots


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_report(rows: Iterable[dict[str, Any]], contract: PaperContract) -> dict[str, Any]:
    rows = list(rows)
    conformance = audit_rows(rows, contract)
    lossless = audit_losslessness(rows) if rows and "target_output_ids" in rows[0] else None
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = " | ".join(
            str(row.get(field, "unknown"))
            for field in ("paper_figure", "actual_visual_tokens", "retention_percentage", "layer_cut")
        )
        groups[key].append(row)
    summaries = {key: acceptance_summary(group) for key, group in sorted(groups.items())}
    return {
        "valid": conformance.valid and (lossless is None or lossless.valid),
        "contract": contract.to_dict(),
        "traceability": paper_contract_rows(contract),
        "conformance": conformance.to_dict(),
        "losslessness": lossless.to_dict() if lossless else None,
        "summaries": summaries,
        "_rows": rows,
    }


def write_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_files = write_plots(report.pop("_rows", []), output)
    report["plots"] = plot_files
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Sparrow insight validation report",
        "",
        f"**Validity gate:** `{str(report['valid']).upper()}`",
        "",
        "This report distinguishes paper-conformance from numerical reproduction. "
        "The local run uses the VDC-50 subset and may use T4/4-bit inference.",
        "",
        "## Paper traceability",
        "",
        "| Figure | Claim | Model | Metric |",
        "|---|---|---|---|",
    ]
    for item in report["traceability"]:
        lines.append(f"| {item['figure']} | {item['claim']} | {item['model']} | {item['metric']} |")
    lines.extend(["", "## Aggregate summaries", "", "| Condition | N | Accepted length | Lossless rate |", "|---|---:|---:|---:|"])
    for key, summary in report["summaries"].items():
        accepted = summary["accepted_length"]["mean"]
        lossless = summary["lossless_rate"]
        accepted_text = "n/a" if accepted != accepted else f"{accepted:.3f}"
        lossless_text = "n/a" if lossless is None else f"{lossless:.1%}"
        lines.append(f"| {key} | {summary['n']} | {accepted_text} | {lossless_text} |")
    lines.extend(["", "## Audit issues", ""])
    issues = report["conformance"]["issues"]
    if not issues:
        lines.append("No paper-conformance issues were found.")
    else:
        for issue in issues:
            lines.append(f"- `{issue['severity']}` `{issue['code']}`: {issue['message']} ({issue.get('row_id')})")
    if report.get("losslessness"):
        lines.extend(["", "## Losslessness", ""])
        lines.append(json.dumps(report["losslessness"], ensure_ascii=False, indent=2))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
