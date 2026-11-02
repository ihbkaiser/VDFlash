"""Audit and summarize a completed or partial VisualCut/attention campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import torch

from .metrics import score_pair
from .attention_artifacts import load_attention_shard


CONDITIONS = ("full", "zero", "cut", "retained_75", "retained_50", "retained_25")
REGIONS = (
    "visual",
    "question_text",
    "answer_hint",
    "other_text",
    "other_prompt_context",
    "generated_target_hidden",
    "cached_draft",
    "current_draft_query",
)
SOURCE_PROMPT = 0
SOURCE_TARGET_HIDDEN = 1
SOURCE_CACHED_DRAFT = 2
SOURCE_CURRENT_QUERY = 3
QUALITY_METRICS = ("bleu", "rouge_l", "coverage")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
            writer.writerows(rows)


def _bootstrap(values: list[float], *, seed: int, replicates: int = 2000) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = random.Random(seed)
    means = sorted(
        sum(finite[rng.randrange(len(finite))] for _ in finite) / len(finite)
        for _ in range(replicates)
    )
    return {
        "n": len(finite),
        "mean": mean,
        "ci_low": means[int(0.025 * (replicates - 1))],
        "ci_high": means[int(0.975 * (replicates - 1))],
    }


def _valid_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("status") in {"ok", "mismatch"}
        and row.get("capture_and_timing_outputs_match") is True
        and (row.get("attention_capture") or {}).get("record_count", 0) > 0
        and (row.get("attention_capture") or {}).get("shards")
    )


def _quality_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten reference-based output quality for CSV summaries."""

    quality = row.get("quality_metrics") or {}
    target = quality.get("target") or {}
    speculative = quality.get("speculative") or {}
    result: dict[str, Any] = {
        "quality_reference_available": bool(row.get("reference_answer")),
        "quality_reference_source": quality.get("reference_source"),
    }
    for metric in QUALITY_METRICS:
        target_value = target.get(metric)
        speculative_value = speculative.get(metric)
        result[f"target_{metric}"] = target_value
        result[f"speculative_{metric}"] = speculative_value
        result[f"{metric}_delta_vs_target"] = (
            float(speculative_value) - float(target_value)
            if target_value is not None and speculative_value is not None
            else None
        )
    return result


def _manifest_references(run_dir: Path) -> dict[str, str]:
    """Read caption references from the manifest recorded with this run."""

    config_path = next(
        (
            candidate / "run_config.json"
            for candidate in (run_dir, *list(run_dir.parents)[:3])
            if (candidate / "run_config.json").is_file()
        ),
        None,
    )
    if config_path is None:
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_value = config.get("dataset_manifest") or config.get("manifest") or config.get("dataset")
    if not manifest_value:
        return {}
    manifest = Path(str(manifest_value))
    if not manifest.is_file():
        return {}
    references: dict[str, str] = {}
    for record in _read_jsonl(manifest):
        sample_id = record.get("sample_id") or record.get("video_name") or record.get("id")
        reference = record.get("reference_answer") or record.get("answer")
        if sample_id is not None and reference is not None and str(reference).strip():
            references[str(sample_id)] = str(reference).strip()
    return references


def _attach_reference_quality(rows: list[dict[str, Any]], references: dict[str, str]) -> None:
    """Backfill reference scores for older journals that predate quality fields."""

    for row in rows:
        reference = row.get("reference_answer") or references.get(str(row.get("sample_id")))
        if not reference:
            continue
        row["reference_answer"] = str(reference)
        if row.get("quality_metrics"):
            continue
        target_text = row.get("target_output_text")
        speculative_text = row.get("speculative_output_text")
        if target_text is None or speculative_text is None:
            continue
        row["quality_metrics"] = {
            "reference_source": "run_manifest",
            "target": score_pair(str(target_text), str(reference)),
            "speculative": score_pair(str(speculative_text), str(reference)),
        }


def _region_indices(row: dict[str, Any], record: dict[str, Any]) -> dict[str, torch.Tensor]:
    key_map = record["key_map"]
    source_ids = key_map["source_ids"].to(dtype=torch.int64)
    positions = key_map["positions"].to(dtype=torch.int64)
    regions = row.get("prompt_regions_original") or {}
    prompt_sets = {
        name: {int(value) for value in regions.get(name, [])}
        for name in ("visual", "question_text", "answer_hint", "other_text", "other_context")
    }
    selected: dict[str, torch.Tensor] = {}
    for name in REGIONS:
        selected[name] = torch.zeros_like(source_ids, dtype=torch.bool)
    prompt_mask = source_ids == SOURCE_PROMPT
    for name in ("visual", "question_text", "answer_hint", "other_text"):
        if prompt_sets[name]:
            indices = torch.tensor(sorted(prompt_sets[name]), dtype=torch.int64)
            selected[name] = prompt_mask & torch.isin(positions, indices)
    if prompt_sets["other_context"]:
        indices = torch.tensor(sorted(prompt_sets["other_context"]), dtype=torch.int64)
        selected["other_prompt_context"] = prompt_mask & torch.isin(positions, indices)
    covered_prompt = torch.zeros_like(prompt_mask)
    for name in ("visual", "question_text", "answer_hint", "other_text", "other_prompt_context"):
        covered_prompt |= selected[name]
    selected["other_prompt_context"] |= prompt_mask & ~covered_prompt
    selected["generated_target_hidden"] = source_ids == SOURCE_TARGET_HIDDEN
    selected["cached_draft"] = source_ids == SOURCE_CACHED_DRAFT
    selected["current_draft_query"] = source_ids == SOURCE_CURRENT_QUERY
    return selected


def _summarize_attention_record(
    row: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    weights = record["weights"].detach().to(device="cpu", dtype=torch.float32)
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise ValueError(f"unexpected DFlash attention tensor shape: {tuple(weights.shape)}")
    weights = weights[0]
    heads, queries, keys = weights.shape
    selected = _region_indices(row, record)
    region_masks = torch.stack([selected[name] for name in REGIONS], dim=0)
    if region_masks.shape[-1] != keys:
        raise ValueError("attention region map does not align with captured key dimension")
    region_counts = region_masks.sum(dim=1).to(torch.float32)
    region_mass = torch.stack(
        [weights[..., mask].sum(dim=-1) for mask in region_masks], dim=-1
    )
    row_sum = weights.sum(dim=-1)
    valid_rows = row_sum > 0
    valid_counts = valid_rows.sum(dim=1).clamp(min=1).to(torch.float32)
    mass_by_head = (region_mass * valid_rows.unsqueeze(-1)).sum(dim=1) / valid_counts.unsqueeze(-1)
    density_by_head = mass_by_head / region_counts.clamp(min=1).unsqueeze(0)
    probabilities = weights / row_sum.clamp(min=1e-30).unsqueeze(-1)
    entropy = -(probabilities.clamp_min(1e-30).log() * probabilities).sum(dim=-1)
    entropy_by_head = (entropy * valid_rows).sum(dim=1) / valid_counts
    normalized_entropy = entropy_by_head / math.log(max(keys, 2))
    per_query = region_mass.permute(1, 0, 2)
    summary = {
        "row_id": row["row_id"],
        "checkpoint_name": row["checkpoint_name"],
        "sample_id": row["sample_id"],
        "target_visual_tokens": int(row["target_visual_tokens"]),
        "visual_condition": row["visual_condition"],
        "layer_index": int(record["layer_index"]),
        "round_index": int(record["round_index"]),
        "answer_position_start": int(record["answer_position_start"]),
        "head_count": int(heads),
        "query_count": int(queries),
        "key_count": int(keys),
        "valid_query_head_rows": int(valid_rows.sum().item()),
        "region_key_counts": {
            name: int(region_counts[index].item()) for index, name in enumerate(REGIONS)
        },
        "region_mass_by_head": mass_by_head.tolist(),
        "region_density_by_head": density_by_head.tolist(),
        "region_mass_by_query_head": per_query.tolist(),
        "entropy_by_head": entropy_by_head.tolist(),
        "normalized_entropy_by_head": normalized_entropy.tolist(),
    }
    aggregate = {
        "valid_query_head_rows": int(valid_rows.sum().item()),
        "entropy_sum": float((entropy * valid_rows).sum().item()),
        "region_mass_sum": {
            name: float((region_mass[..., index] * valid_rows).sum().item())
            for index, name in enumerate(REGIONS)
        },
        "region_key_count_sum": {
            name: int(region_counts[index].item()) * int(valid_rows.sum().item())
            for index, name in enumerate(REGIONS)
        },
        "normalized_entropy_sum": float((normalized_entropy.unsqueeze(-1) * valid_rows).sum().item()),
        "head_query_count": int(valid_rows.sum().item()),
    }
    return summary, aggregate


def _collect_case_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "tasks").glob("*/cases.jsonl")):
        for row in _read_jsonl(path):
            if row.get("row_id"):
                latest[str(row["row_id"])] = row
    rows = list(latest.values())
    valid = [row for row in rows if _valid_row(row)]
    return rows, valid


def _read_expected_case_ids(run_dir: Path) -> set[str] | None:
    """Return the declared case matrix when the run has a campaign manifest."""

    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    values = config.get("expected_case_ids")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{config_path} must contain a string expected_case_ids list")
    expected = set(values)
    if len(expected) != len(values):
        raise ValueError(f"{config_path} contains duplicate expected case IDs")
    declared_count = config.get("expected_row_count")
    if declared_count is not None and int(declared_count) != len(expected):
        raise ValueError(f"{config_path} expected_row_count does not match expected_case_ids")
    return expected


def _fallback_expected_row_count(run_dir: Path, cohort_size: int | None) -> int | None:
    """Infer the old campaign matrix from worker metadata when possible."""

    if cohort_size is None:
        return None
    pairs = set()
    for path in sorted((run_dir / "tasks").glob("*/worker.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_name = metadata.get("checkpoint_name")
        target = metadata.get("visual_token_target", metadata.get("visual_tokens"))
        if checkpoint_name is not None and target is not None:
            pairs.add((str(checkpoint_name), int(target)))
    if not pairs:
        return None
    return len(pairs) * len(CONDITIONS) * cohort_size


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_rows, valid_rows = _collect_case_rows(run_dir)
    _attach_reference_quality(all_rows, _manifest_references(run_dir))
    cases_by_id = {str(row["row_id"]): row for row in valid_rows}
    attention_file = analysis_dir / "attention_forward_summaries.jsonl"
    attention_file.unlink(missing_ok=True)
    attention_handle = attention_file.open("w", encoding="utf-8")
    case_accumulators: dict[str, dict[str, Any]] = {}
    unique_shards: dict[str, dict[str, Any]] = {}
    malformed_records: list[dict[str, Any]] = []

    for row in valid_rows:
        for shard in (row.get("attention_capture") or {}).get("shards", []):
            shard_path = Path(shard["path"])
            unique_shards[str(shard_path)] = shard
    verified_shards = 0
    for shard_path_string, shard in unique_shards.items():
        shard_path = Path(shard_path_string)
        if not shard_path.is_file():
            malformed_records.append({"shard": shard_path_string, "error": "missing file"})
            continue
        if args.verify_sha256 and _sha256(shard_path) != shard.get("sha256"):
            malformed_records.append({"shard": shard_path_string, "error": "sha256 mismatch"})
            continue
        verified_shards += 1
        for record in load_attention_shard(shard_path):
            case_id = str(record.get("case_id", ""))
            row = cases_by_id.get(case_id)
            if row is None:
                malformed_records.append({"shard": shard_path_string, "error": f"unknown case {case_id}"})
                continue
            try:
                summary, aggregate = _summarize_attention_record(row, record)
            except Exception as exc:
                malformed_records.append(
                    {"shard": shard_path_string, "case_id": case_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            attention_handle.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
            accumulator = case_accumulators.setdefault(
                case_id,
                {
                    "valid_query_head_rows": 0,
                    "entropy_sum": 0.0,
                    "normalized_entropy_sum": 0.0,
                    "region_mass_sum": {name: 0.0 for name in REGIONS},
                    "region_key_count_sum": {name: 0 for name in REGIONS},
                    "record_count": 0,
                },
            )
            accumulator["record_count"] += 1
            accumulator["valid_query_head_rows"] += aggregate["valid_query_head_rows"]
            accumulator["entropy_sum"] += aggregate["entropy_sum"]
            accumulator["normalized_entropy_sum"] += aggregate["normalized_entropy_sum"]
            for name in REGIONS:
                accumulator["region_mass_sum"][name] += aggregate["region_mass_sum"][name]
                accumulator["region_key_count_sum"][name] += aggregate["region_key_count_sum"][name]

    attention_handle.close()

    for case_id, row in cases_by_id.items():
        actual_records = case_accumulators.get(case_id, {}).get("record_count", 0)
        expected_records = int((row.get("attention_capture") or {}).get("record_count", 0))
        if actual_records != expected_records:
            malformed_records.append(
                {
                    "case_id": case_id,
                    "error": f"attention record count mismatch: expected={expected_records}, actual={actual_records}",
                }
            )

    case_summaries: list[dict[str, Any]] = []
    for row in valid_rows:
        accumulator = case_accumulators.get(str(row["row_id"]))
        if accumulator is None or accumulator["valid_query_head_rows"] <= 0:
            continue
        valid_count = int(accumulator["valid_query_head_rows"])
        value = {
            "row_id": row["row_id"],
            "checkpoint_name": row["checkpoint_name"],
            "sample_id": row["sample_id"],
            "target_visual_tokens": int(row["target_visual_tokens"]),
            "visual_condition": row["visual_condition"],
            "status": row.get("status"),
            "tau_proposal": (row.get("metrics") or {}).get("tau_proposal"),
            "tau_effective": (row.get("metrics") or {}).get("tau_effective"),
            "lossless": bool(row.get("lossless")),
            "lossless_prefix_length": row.get("lossless_prefix_length"),
            "target_output_hash": row.get("target_output_hash"),
            "speculative_output_hash": row.get("speculative_output_hash"),
            "tokens_per_second": ((row.get("timing") or {}).get("speculative") or {}).get("tokens_per_second"),
            "attention_valid_query_head_rows": valid_count,
            "attention_entropy": accumulator["entropy_sum"] / valid_count,
            "attention_normalized_entropy": accumulator["normalized_entropy_sum"] / valid_count,
        }
        value.update(_quality_fields(row))
        for name in REGIONS:
            value[f"{name}_mass"] = accumulator["region_mass_sum"][name] / valid_count
            keys = accumulator["region_key_count_sum"][name]
            value[f"{name}_density"] = (
                accumulator["region_mass_sum"][name] / keys if keys > 0 else None
            )
        case_summaries.append(value)
    _write_csv(analysis_dir / "case_attention_summary.csv", case_summaries)

    metrics = [
        "tau_proposal",
        "tau_effective",
        "tokens_per_second",
        "attention_entropy",
        "attention_normalized_entropy",
        "visual_mass",
        "visual_density",
        "question_text_mass",
        "question_text_density",
        "answer_hint_mass",
        "other_text_mass",
        "generated_target_hidden_mass",
        "cached_draft_mass",
        "current_draft_query_mass",
        "target_bleu",
        "speculative_bleu",
        "bleu_delta_vs_target",
        "target_rouge_l",
        "speculative_rouge_l",
        "rouge_l_delta_vs_target",
        "target_coverage",
        "speculative_coverage",
        "coverage_delta_vs_target",
    ]
    by_group: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in case_summaries:
        key = (str(row["checkpoint_name"]), int(row["target_visual_tokens"]), str(row["visual_condition"]))
        by_group.setdefault(key, []).append(row)
    statistics = []
    for (checkpoint, budget, condition), rows in sorted(by_group.items()):
        summary = {
            "checkpoint_name": checkpoint,
            "target_visual_tokens": budget,
            "visual_condition": condition,
            "sample_count": len({row["sample_id"] for row in rows}),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            summary.update({f"{metric}_{name}": value for name, value in _bootstrap(values, seed=42 + len(statistics)).items()})
        statistics.append(summary)
    _write_csv(analysis_dir / "condition_statistics.csv", statistics)

    contrasts = []
    for checkpoint in sorted({str(row["checkpoint_name"]) for row in case_summaries}):
        for budget in sorted({int(row["target_visual_tokens"]) for row in case_summaries}):
            rows = [row for row in case_summaries if row["checkpoint_name"] == checkpoint and int(row["target_visual_tokens"]) == budget]
            per_condition: dict[str, dict[str, dict[str, float]]] = {}
            for row in rows:
                condition = str(row["visual_condition"])
                per_condition.setdefault(condition, {})[str(row["sample_id"])] = {
                    metric: float(row[metric])
                    for metric in metrics
                    if row.get(metric) is not None
                }
            full = per_condition.get("full", {})
            for condition in CONDITIONS[1:]:
                candidate = per_condition.get(condition, {})
                for metric_index, metric in enumerate(metrics):
                    differences = [
                        candidate[sample][metric] - full[sample][metric]
                        for sample in sorted(set(full) & set(candidate))
                        if metric in candidate[sample] and metric in full[sample]
                    ]
                    contrasts.append(
                        {
                            "checkpoint_name": checkpoint,
                            "target_visual_tokens": budget,
                            "contrast": f"{condition}-full",
                            "metric": metric,
                            **_bootstrap(differences, seed=1000 + metric_index + len(contrasts)),
                        }
                    )
    _write_csv(analysis_dir / "paired_contrasts.csv", contrasts)

    run_config_path = run_dir / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file()
        else {}
    )
    calibration_dir = (
        Path(args.calibration_dir).resolve()
        if args.calibration_dir is not None
        else Path(run_config["calibration"]).resolve().parent
        if run_config.get("calibration")
        else run_dir / "calibration"
    )
    cohort_path = (
        Path(run_config["cohort_selection"]).resolve()
        if run_config.get("cohort_selection")
        else calibration_dir / "cohort_selection.json"
    )
    cohort_size = None
    if cohort_path.is_file():
        cohort_size = int(json.loads(cohort_path.read_text(encoding="utf-8")).get("sample_count", 0))
    expected_case_ids = _read_expected_case_ids(run_dir)
    expected_rows = (
        len(expected_case_ids)
        if expected_case_ids is not None
        else _fallback_expected_row_count(run_dir, cohort_size)
    )
    actual_case_ids = {str(row["row_id"]) for row in all_rows if row.get("row_id")}
    if expected_case_ids is not None:
        rows_missing = len(expected_case_ids - actual_case_ids)
        unexpected_rows = len(actual_case_ids - expected_case_ids)
    else:
        rows_missing = (
            max(0, expected_rows - len(all_rows)) if expected_rows is not None else None
        )
        unexpected_rows = 0
    report = {
        "run_dir": str(run_dir),
        "cohort_size": cohort_size,
        "expected_rows": expected_rows,
        "expected_case_ids_source": (
            str(run_config_path)
            if expected_case_ids is not None
            else "worker_metadata"
            if expected_rows is not None
            else None
        ),
        "rows_total": len(all_rows),
        "rows_valid": len(valid_rows),
        "rows_invalid": len(all_rows) - len(valid_rows),
        "rows_lossless": sum(row.get("lossless") is True for row in all_rows),
        "rows_mismatched": sum(row.get("status") == "mismatch" for row in all_rows),
        "rows_with_reference_quality": sum(
            bool((row.get("quality_metrics") or {}).get("speculative")) for row in all_rows
        ),
        "rows_missing_reference_quality": sum(
            not bool((row.get("quality_metrics") or {}).get("speculative")) for row in all_rows
        ),
        "reference_quality_metric_names": list(QUALITY_METRICS),
        "lossless_rate": (
            sum(row.get("lossless") is True for row in all_rows) / len(all_rows)
            if all_rows
            else None
        ),
        "analysis_scope": "descriptive_attention_and_visualcut; nonlossless rows are retained and flagged",
        "quality_metric_basis": "reference caption; BLEU, ROUGE-L, and content coverage for target and speculative outputs",
        "rows_missing": rows_missing,
        "unexpected_rows": unexpected_rows,
        "attention_shards_referenced": len(unique_shards),
        "attention_shards_verified": verified_shards,
        "attention_summary_records": sum(1 for _ in attention_file.open(encoding="utf-8")) if attention_file.is_file() else 0,
        "case_attention_summaries": len(case_summaries),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 42,
        "malformed_records": malformed_records,
        "outputs": {
            "attention_forward_summaries": str(attention_file),
            "case_attention_summary": str(analysis_dir / "case_attention_summary.csv"),
            "condition_statistics": str(analysis_dir / "condition_statistics.csv"),
            "paired_contrasts": str(analysis_dir / "paired_contrasts.csv"),
        },
    }
    _write_json(analysis_dir / "audit_and_summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--skip-sha256", dest="verify_sha256", action="store_false")
    parser.set_defaults(verify_sha256=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    if args.require_complete and (
        report["rows_invalid"] or report["rows_missing"] or report["unexpected_rows"]
    ):
        return 1
    if report["malformed_records"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
