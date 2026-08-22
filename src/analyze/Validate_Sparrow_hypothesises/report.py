"""Small, dependency-light report writer for completed JSONL runs."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .audit import AuditReport, audit_losslessness, audit_rows
from .coverage import build_coverage
from .metrics import acceptance_summary
from .paper_contract import PaperContract, paper_contract_rows
from .paper_statistics import build_paper_statistics, write_statistics
from .plots import write_paper_style_plots, write_plots
from .figure3_pipeline import validate_figure3_summaries


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_report(
    rows: Iterable[dict[str, Any]],
    contract: PaperContract,
    *,
    figure2_rows: Iterable[dict[str, Any]] | None = None,
    figure2_selection: dict[str, Any] | None = None,
    figure2_audit: dict[str, Any] | None = None,
    current_figure3: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(rows)
    external_figure3_names = {"Figure 3", "Figure 3(b)"} if current_figure3 is not None else set()
    audit_rows_input = [
        row for row in rows if row.get("paper_figure") not in external_figure3_names
    ]
    report_rows = list(audit_rows_input)
    if figure2_rows is not None:
        report_rows = [row for row in report_rows if row.get("paper_figure") != "Figure 2"]
        report_rows.extend(dict(row) for row in figure2_rows)
    conformance = audit_rows(audit_rows_input, contract)
    coverage = build_coverage(
        audit_rows_input,
        contract,
        excluded_figures=external_figure3_names,
    )
    lossless = audit_losslessness(audit_rows_input) if any(
        "target_output_ids" in row or "speculative_output_ids" in row for row in audit_rows_input
    ) else None
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary_rows = [
        row for row in report_rows
        if row.get("accepted_prefix_tokens") is not None or row.get("rouge_l") is not None
    ]
    for row in summary_rows:
        key = " | ".join(
            str(row.get(field, "unknown"))
            for field in ("paper_figure", "actual_visual_tokens", "retention_percentage", "layer_cut")
        )
        groups[key].append(row)
    summaries = {key: acceptance_summary(group) for key, group in sorted(groups.items())}
    figure2_valid = figure2_audit is None or bool(figure2_audit.get("valid", False))
    current_figure3_payload = None
    current_figure3_valid = True
    if current_figure3 is not None:
        metadata = dict(current_figure3.get("metadata") or {})
        stored_audit = dict(current_figure3.get("audit") or {})
        expected_model = str(metadata.get("model") or current_figure3["a_summary"].get("model"))
        summary_validation = validate_figure3_summaries(
            current_figure3["a_summary"],
            current_figure3["b_summary"],
            expected_model=expected_model,
        )
        current_figure3_valid = bool(stored_audit.get("valid", False)) and summary_validation["valid"]
        current_figure3_payload = {
            "bundle_dir": current_figure3.get("output_dir"),
            "metadata": metadata,
            "audit": {
                **stored_audit,
                "summary_validation": summary_validation,
                "valid": current_figure3_valid,
            },
            "statistics": metadata.get("statistics", {}),
        }
        diagnostic_counts = {
            figure: sum(1 for row in rows if row.get("paper_figure") == figure)
            for figure in sorted({str(row.get("paper_figure")) for row in rows})
        }
        diagnostic_counts["Figure 3(a) current"] = int(
            current_figure3["a_summary"].get("num_scored_rows", 0) or 0
        )
        diagnostic_counts["Figure 3(b) current"] = int(
            current_figure3["b_summary"].get("num_success_rows", 0) or 0
        )
    else:
        diagnostic_counts = {
            figure: sum(1 for row in rows if row.get("paper_figure") == figure)
            for figure in sorted({str(row.get("paper_figure")) for row in rows})
        }
    return {
        "valid": conformance.valid and coverage.valid and (lossless is None or lossless.valid) and figure2_valid and current_figure3_valid,
        "contract": contract.to_dict(),
        "traceability": paper_contract_rows(contract),
        "conformance": conformance.to_dict(),
        "coverage": coverage.to_dict(),
        "losslessness": lossless.to_dict() if lossless else None,
        "summaries": summaries,
        "diagnostic_counts": diagnostic_counts,
        "figure2_homogeneous_cohort": figure2_selection,
        "figure2_homogeneous_audit": figure2_audit,
        "current_figure3": current_figure3_payload,
        "_rows": report_rows,
    }


def write_report(output_dir: str | Path, report: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = report.pop("_rows", [])
    # A previous incomplete run may have left paper-shaped plots under the
    # diagnostic directory.  They carry the incomplete watermark and must
    # not remain discoverable beside a later valid report.  Keep the ordinary
    # exploratory PNG diagnostics; only remove stale composite paper plots
    # before regenerating the current diagnostic set.
    diagnostic_root = output / "diagnostic"
    if diagnostic_root.exists():
        for stale in diagnostic_root.glob("figure*_insight_*.*"):
            try:
                stale.unlink()
            except OSError:
                pass
    # Legacy exploratory plots are always useful for debugging, but the
    # paper-shaped figures are emitted only for a complete enforced cohort.
    plot_files = write_plots(rows, diagnostic_root)
    paper_root = output if report.get("valid", False) else output / "diagnostic"
    watermark = None if report.get("valid", False) else "INCOMPLETE DIAGNOSTIC — NOT PAPER EVIDENCE"
    paper_plot_files = write_paper_style_plots(
        rows,
        paper_root,
        formats=tuple(report.get("contract", {}).get("paper_plot_formats", ["png"])),
        watermark=watermark,
    )
    if paper_root != output:
        paper_plot_files = [str(Path("diagnostic") / name) for name in paper_plot_files]
    plot_files = [str(Path("diagnostic") / name) for name in plot_files]
    statistics = build_paper_statistics(rows)
    current_figure3 = report.get("current_figure3")
    statistics_for_files = (
        {key: value for key, value in statistics.items() if key not in {"figure3a", "figure3b"}}
        if current_figure3
        else statistics
    )
    statistics_files = write_statistics(statistics_for_files, output)
    current_files: list[str] = []
    if current_figure3:
        bundle_dir = Path(str(current_figure3.get("bundle_dir", "")))
        try:
            relative_bundle = bundle_dir.relative_to(output)
            bundle_prefix = str(relative_bundle) if str(relative_bundle) != "." else ""
        except ValueError:
            bundle_prefix = os.path.relpath(bundle_dir, output) if bundle_dir else "figure3"
        if bundle_prefix == ".":
            bundle_prefix = ""
        prefix = f"{bundle_prefix}/" if bundle_prefix else ""
        current_files = [
            f"{prefix}figure3_insight_layer_analysis.{file_format}"
            for file_format in ("png", "pdf", "svg")
            if (bundle_dir / f"figure3_insight_layer_analysis.{file_format}").is_file()
        ]
        current_files.extend(
            f"{prefix}{filename}"
            for filename in (
                "figure3a_statistics.csv",
                "figure3b_statistics.csv",
                "figure3_metadata.json",
                "figure3_audit.json",
            )
            if (bundle_dir / filename).is_file()
        )
        if current_figure3.get("statistics"):
            statistics.update(current_figure3["statistics"])
        (output / "paper_statistics.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    report["plots"] = plot_files + paper_plot_files
    report["plots"].extend(
        name for name in current_files if name.endswith((".png", ".pdf", ".svg"))
    )
    report["paper_statistics_files"] = statistics_files
    report["paper_statistics_files"].extend(
        name for name in current_files if name.endswith(".csv")
    )
    report["paper_statistics"] = statistics
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Sparrow insight validation report",
        "",
        f"**Validity gate:** `{str(report['valid']).upper()}`",
        "",
        "This report distinguishes paper-conformance from numerical reproduction. "
        "The local run uses the VDC-50 subset and may use 4-bit inference on a "
        "3090+A4000 model-parallel setup.",
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
        display_key = key.replace(" | ", " / ")
        lines.append(f"| {display_key} | {summary['n']} | {accepted_text} | {lossless_text} |")
    lines.extend(["", "## Diagnostic row counts", "", "| Figure | Rows |", "|---|---:|"])
    for figure, count in report.get("diagnostic_counts", {}).items():
        lines.append(f"| {figure} | {count} |")
    lines.extend(["", "## Paper-shaped statistics", "", "The following aggregates are computed only from measured rows. Each metric includes N, mean, spread, and a deterministic bootstrap 95% interval.", ""])
    lines.append("[paper_statistics.json](paper_statistics.json) · " + " · ".join(
        f"[{name}]({name})" for name in report.get("paper_statistics_files", []) if name != "paper_statistics.json"
    ))
    lines.extend(["", "## Paper-style figures", ""])
    if report.get("plots"):
        embedded_pngs = {
            name for name in report["plots"]
            if name in {
                "figure1_insight_summary.png",
                "figure2_insight_attention.png",
                "figure3_insight_layer_analysis.png",
                "figure6_insight_retention.png",
                "figure3/figure3_insight_layer_analysis.png",
            }
        }
        for name in sorted(embedded_pngs):
            title = Path(name).stem.replace("_", " ").title()
            lines.extend([f"### {title}", "", f"![{title}](./{name})", ""])
        lines.append("Download links for all generated formats:")
        for name in report["plots"]:
            lines.append(f"- [{name}]({name})")
    else:
        lines.append("No plot was generated because no completed metric rows were available.")
    if current_figure3:
        metadata = current_figure3.get("metadata", {})
        lines.extend(["", "## Current Qwen2.5-VL-3B Figure 3", ""])
        lines.append(
            "This section is generated from the standalone Qwen2.5-VL-3B MVBench "
            "all-layer bundle; legacy Figure 3 rows remain diagnostics only."
        )
        lines.append("")
        lines.append(
            f"Model: `{metadata.get('model')}`; manifest SHA256: "
            f"`{metadata.get('manifest_sha256')}`; layers: `{metadata.get('layer_count')}`."
        )
        lines.append("")
        for name in current_files:
            lines.append(f"- [{name}]({name})")
    lines.extend(["", "## Audit issues", ""])
    issues = report["conformance"]["issues"]
    if not issues:
        lines.append("No paper-conformance issues were found.")
    else:
        for issue in issues:
            lines.append(f"- `{issue['severity']}` `{issue['code']}`: {issue['message']} ({issue.get('row_id')})")
    if current_figure3:
        lines.extend(["", "## Current Figure 3 audit", ""])
        current_audit = current_figure3.get("audit", {})
        current_issues = list(current_audit.get("issues", []))
        current_issues.extend(current_audit.get("summary_validation", {}).get("issues", []))
        if not current_issues:
            lines.append("Current Figure 3 bundle passed the cross-panel provenance and coverage audit.")
        else:
            for issue in current_issues:
                lines.append(f"- `{issue.get('code', 'current_figure3')}`: {issue.get('message', issue)}")
    lines.extend(["", "## Coverage gate", ""])
    coverage = report.get("coverage", {})
    lines.append(f"Coverage valid: `{str(coverage.get('valid', False)).upper()}`; paired samples: `{coverage.get('paired_samples', 0)}`.")
    for issue in coverage.get("issues", []):
        lines.append(f"- `{issue['code']}`: {issue['message']} ({issue.get('figure') or 'run'})")
    if report.get("losslessness"):
        lines.extend(["", "## Losslessness", ""])
        lines.append(json.dumps(report["losslessness"], ensure_ascii=False, indent=2))
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
