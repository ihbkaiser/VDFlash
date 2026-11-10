"""Validate all six conditions and raw attention tensors from a GPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .attention_artifacts import load_attention_shard
from .attention_summary_storage import SUMMARY_RECORD_SCHEMA


CONDITIONS = ("full", "zero", "cut", "retained_75", "retained_50", "retained_25")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["row_id"])] = row
    return rows


def validate(run_dir: Path, task_id: str) -> dict[str, Any]:
    journal = run_dir / "tasks" / task_id / "cases.jsonl"
    rows = _latest_rows(journal) if journal.is_file() else {}
    selected = [row for row in rows.values() if row.get("visual_condition") in CONDITIONS]
    sample_ids = sorted({str(row.get("sample_id")) for row in selected})
    failures: list[str] = []
    lossless_failures: list[str] = []
    if len(sample_ids) != 1:
        failures.append(f"expected exactly one pilot sample, found {len(sample_ids)}")
    by_condition = {str(row["visual_condition"]): row for row in selected}
    if set(by_condition) != set(CONDITIONS):
        failures.append(
            f"condition set mismatch: expected={sorted(CONDITIONS)}, actual={sorted(by_condition)}"
        )
    target_hashes = {row.get("target_output_hash") for row in selected}
    if len(target_hashes) != 1:
        failures.append(f"target output differs across paired conditions: {sorted(map(str, target_hashes))}")

    condition_reports = []
    total_records = 0
    total_compact_records = 0
    for condition in CONDITIONS:
        row = by_condition.get(condition)
        if row is None:
            continue
        if row.get("status") not in {"ok", "mismatch"}:
            failures.append(f"{condition}: worker did not complete: {row.get('status')!r}")
        if row.get("lossless") is not True:
            lossless_failures.append(f"{condition}: speculative output is not lossless")
        if row.get("capture_and_timing_outputs_match") is not True:
            failures.append(f"{condition}: captured and timing repeats differ")
        capture = row.get("attention_capture") or {}
        expected_records = int(capture.get("record_count", 0))
        actual_records = 0
        case_shards = capture.get("shards", [])
        if expected_records <= 0 or not case_shards:
            failures.append(f"{condition}: no raw DFlash attention records or shards")
        for shard in case_shards:
            shard_path = Path(shard["path"])
            if not shard_path.is_file():
                failures.append(f"{condition}: missing shard {shard_path}")
                continue
            observed_hash = _sha256(shard_path)
            if observed_hash != shard.get("sha256"):
                failures.append(f"{condition}: SHA256 mismatch for {shard_path}")
                continue
            try:
                records = load_attention_shard(shard_path)
            except Exception as exc:
                failures.append(f"{condition}: cannot decompress {shard_path}: {type(exc).__name__}: {exc}")
                continue
            if len(records) != int(shard.get("record_count", -1)):
                failures.append(f"{condition}: shard record count mismatch for {shard_path}")
            for record in records:
                if record.get("schema") == SUMMARY_RECORD_SCHEMA:
                    summary = record.get("summary") or {}
                    aggregate = record.get("aggregate") or {}
                    if (
                        str(record.get("case_id")) != str(row["row_id"])
                        or str(record.get("row_id")) != str(row["row_id"])
                        or str(summary.get("row_id")) != str(row["row_id"])
                        or int(aggregate.get("head_query_count", 0)) <= 0
                        or int(summary.get("key_count", 0)) <= 0
                    ):
                        failures.append(f"{condition}: malformed compact attention summary")
                    total_compact_records += 1
                    continue
                weights = record.get("weights")
                key_map = record.get("key_map") or {}
                if not torch.is_tensor(weights) or weights.ndim != 4 or weights.shape[0] != 1:
                    failures.append(f"{condition}: unexpected raw weight shape")
                    continue
                if str(record.get("case_id")) != str(row["row_id"]):
                    failures.append(f"{condition}: shard contains a foreign case id")
                for field in ("source_ids", "positions", "token_ids", "is_cached"):
                    if len(key_map.get(field, ())) != weights.shape[-1]:
                        failures.append(f"{condition}: key map {field} is misaligned")
                        break
                if weights.shape[-2] != int(record.get("query_length", -1)):
                    failures.append(f"{condition}: query length metadata is misaligned")
                if weights.shape[-1] != int(record.get("key_length", -1)):
                    failures.append(f"{condition}: key length metadata is misaligned")
            actual_records += len(records)
        if actual_records != expected_records:
            failures.append(
                f"{condition}: expected {expected_records} raw records, found {actual_records}"
            )
        condition_reports.append(
            {
                "visual_condition": condition,
                "row_id": row.get("row_id"),
                "status": row.get("status"),
                "lossless": row.get("lossless"),
                "capture_and_timing_outputs_match": row.get("capture_and_timing_outputs_match"),
                "attention_records": actual_records,
                "attention_shards": len(case_shards),
                "target_output_hash": row.get("target_output_hash"),
            }
        )
        total_records += actual_records

    report = {
        "run_dir": str(run_dir),
        "task_id": task_id,
        "sample_ids": sample_ids,
        "condition_count": len(condition_reports),
        "attention_record_count": total_records,
        "attention_compact_record_count": total_compact_records,
        "passed": not failures,
        "artifact_integrity_passed": not failures,
        "lossless_passed": not lossless_failures,
        "failures": failures,
        "lossless_failures": lossless_failures,
        "conditions": condition_reports,
    }
    output = run_dir / "analysis" / "pilot_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", default="dflash_3layer_26_28_30_32_34_v400")
    args = parser.parse_args(argv)
    report = validate(args.run_dir.resolve(), args.task_id)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
