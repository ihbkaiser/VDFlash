"""Compute metrics for the four-config VisualCut report data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .visualcut_report_utils import extract_case_metrics, is_usable_case, summarize_cases


CONDITIONS = ("full", "retained_75", "retained_50", "retained_25", "zero", "cut")
BUDGETS = (400, 3000, 13000, 25000)
CONFIGS = (
    "dflash_3layer_26_28_30_32_34",
    "dflash_3layer_17_19_25_33_34",
    "dflash_3layer_17_28_30_32_34",
    "dflash_3layer_1_28_30_32_34",
)


def _latest_case_rows(paths: list[Path]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    latest: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["checkpoint_name"]),
                int(row["target_visual_tokens"]),
                str(row["visual_condition"]),
                str(row["sample_id"]),
            )
            latest[key] = row
    return latest


def compute_report_data(
    task_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Read latest valid case rows and compute condition/contrast tables."""

    latest = _latest_case_rows(sorted(task_root.glob("*/cases.jsonl")))
    all_data = {
        key: extract_case_metrics(row)
        for key, row in latest.items()
        if is_usable_case(row)
    }
    condition_stats, paired_contrasts = summarize_cases(
        all_data,
        CONFIGS,
        BUDGETS,
        CONDITIONS,
    )
    return condition_stats, paired_contrasts, len(all_data)


def main() -> None:
    run_dir = Path("results/visualcut_attention/vdc50_visualcut_attention_20261026T170825Z")
    output_dir = run_dir / "summary_reports_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_root = run_dir / "tasks"
    if not list(task_root.glob("*/cases.jsonl")):
        raise FileNotFoundError(f"no case journals under {task_root}")
    condition_stats, paired_contrasts, row_count = compute_report_data(task_root)
    (output_dir / "condition_stats.json").write_text(
        json.dumps(condition_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_contrasts.json").write_text(
        json.dumps(paired_contrasts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Computed report data from {row_count} valid case rows in {output_dir}")


if __name__ == "__main__":
    main()
