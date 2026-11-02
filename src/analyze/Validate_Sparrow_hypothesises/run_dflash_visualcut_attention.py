"""Run paired VisualCut and per-forward DFlash attention on calibrated VDC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import torch

from src.infer.qwen25vl_dflash_compare import _load_draft, _load_target, _resolve_dtype
from src.workspace import resolve_workspace_path

from .attention_artifacts import AttentionShardWriter
from .run_dflash_h21_h31 import (
    _load_records,
    _prepare_prompt,
    _resolve_reference_answer,
    _run_one,
    build_prompt_record,
)
from .video_compat import configure_video_reader


CHECKPOINTS = (
    "dflash_3layer_26_28_30_32_34",
    "dflash_3layer_17_19_25_33_34",
    "dflash_3layer_17_28_30_32_34",
    "dflash_3layer_1_28_30_32_34",
    "dflash_3layer_1_9_17_25_33",
)
TARGETS = (400, 3000, 13000, 25000)
CONFIGS = {
    "dflash_3layer_26_28_30_32_34": "high_only",
    "dflash_3layer_17_19_25_33_34": "middle_high",
    "dflash_3layer_17_28_30_32_34": "middle_high_2",
    "dflash_3layer_1_28_30_32_34": "shallow_high",
    "dflash_3layer_1_9_17_25_33": "balanced",
}
CONDITIONS = (
    {"visual_condition": "full", "retention_percentage": 100.0},
    {"visual_condition": "zero", "retention_percentage": 0.0},
    {"visual_condition": "cut", "retention_percentage": 0.0},
    {"visual_condition": "retained_75", "retention_percentage": 75.0},
    {"visual_condition": "retained_50", "retention_percentage": 50.0},
    {"visual_condition": "retained_25", "retention_percentage": 25.0},
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        last_complete_line = payload.rfind(b"\n") + 1
        with path.open("r+b") as handle:
            handle.truncate(last_complete_line)
            handle.flush()
            os.fsync(handle.fileno())
        payload = payload[:last_complete_line]
    return [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed(row: dict[str, Any]) -> bool:
    if row.get("status") not in {"ok", "mismatch"}:
        return False
    if row.get("capture_and_timing_outputs_match") is not True:
        return False
    capture = row.get("attention_capture") or {}
    if capture.get("record_count", 0) <= 0 or not capture.get("shards"):
        return False
    for shard in capture["shards"]:
        path = Path(shard["path"])
        if not path.is_file() or _sha256(path) != shard.get("sha256"):
            return False
    return True


def _checkpoint_path(root: Path, checkpoint_name: str) -> Path:
    model_tag = f"qwen25vl-3b-dflash-llava68k-{checkpoint_name}"
    return (
        root
        / checkpoint_name
        / "phase2"
        / model_tag
        / f"{model_tag}-latest"
    )


def _build_prompt_record(sample_record: dict[str, Any]) -> dict[str, Any]:
    sample_proxy = type("SampleProxy", (), sample_record)()
    record = build_prompt_record(sample_proxy, "natural")
    reference_answer, _ = _resolve_reference_answer(sample_record)
    record.update(
        {
            "answer": sample_record.get("answer"),
            "local_video_path": sample_record.get("local_video_path"),
            "video_path": sample_record.get("local_video_path"),
            "dataset_format": "vdc",
            "reference_answer": reference_answer,
        }
    )
    record["calibration_by_target"] = sample_record.get("calibration_by_target", {})
    return record


def _cohort_ids(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("valid"):
        raise ValueError(f"calibrated cohort is invalid: {path}")
    return {str(sample_id) for sample_id in value.get("sample_ids", [])}


def _campaign_expected_case_ids(
    records: list[dict[str, Any]],
    cohort_ids: set[str],
    checkpoint_names: tuple[str, ...] | list[str],
    *,
    num_chunks: int,
    sample_limit: int | None,
) -> list[str]:
    """Build the exact case matrix produced by the configured worker array."""

    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    eligible_records = [
        record for record in records if str(record["sample_id"]) in cohort_ids
    ]
    expected: list[str] = []
    for checkpoint_name in checkpoint_names:
        for target in TARGETS:
            target_records = [
                record
                for record in eligible_records
                if record.get("calibration_by_target", {}).get(target, {}).get("status")
                == "ok"
            ]
            chunk_size = math.ceil(len(target_records) / num_chunks)
            for chunk_index in range(num_chunks):
                start_idx = chunk_index * chunk_size
                chunk_records = target_records[start_idx : start_idx + chunk_size]
                if sample_limit is not None:
                    chunk_records = chunk_records[:sample_limit]
                expected.extend(
                    f"{checkpoint_name}:{record['sample_id']}:{target}:{condition['visual_condition']}"
                    for record in chunk_records
                    for condition in CONDITIONS
                )
    return list(dict.fromkeys(expected))


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    """Write a stable run contract and reject workers with conflicting scope."""

    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"run configuration mismatch at {path}; use a new output directory "
                "for a different campaign or sample limit"
            )
        return
    _atomic_json(path, config)


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repo, check=True, text=True, capture_output=True
        ).stdout.splitlines()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=repo, check=True, capture_output=True
        ).stdout
        return {"revision": revision, "dirty_paths": status, "diff_sha256": hashlib.sha256(diff).hexdigest()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _worker_metadata(
    args,
    task_id: str,
    checkpoint: Path,
    config: Path,
    selected_sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    repo = Path(args.repo_root)
    device = torch.device(args.device)
    return {
        "task_id": task_id,
        "worker_index": getattr(args, "worker_index", None),
        "chunk_index": getattr(args, "chunk_index", 0),
        "num_chunks": getattr(args, "num_chunks", 1),
        "sample_limit_per_chunk": getattr(args, "limit", None),
        "selected_sample_ids": list(selected_sample_ids or []),
        "expected_case_count": len(selected_sample_ids or []) * len(CONDITIONS),
        "run_config": str(Path(args.output_dir) / "run_config.json"),
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_path": str(checkpoint),
        "draft_config": str(config),
        "target_model": str(args.target_model),
        "dataset_manifest": str(args.manifest),
        "calibration": str(args.calibration),
        "cohort_selection": str(args.cohort_selection),
        "visual_token_target": args.visual_tokens,
        "conditions": list(CONDITIONS),
        "max_new_tokens": int(args.requested_max_new_tokens),
        "dtype": str(_resolve_dtype(args.dtype, device)),
        "target_attention": args.target_attention,
        "video_grid_metadata_device_during_target_prefill": "cpu",
        "video_grid_metadata_reason": "Avoid CUDA int64 reduction failure in Qwen2.5-VL video feature splitting; pixel tensors stay on accelerator.",
        "draft_attention": "eager",
        "capture_mode": "all_forwards_all_heads_all_queries",
        "attention_key_source_ids": {
            "0": "original_prompt_target_hidden",
            "1": "generated_target_hidden",
            "2": "cached_generated_key",
            "3": "current_draft_query_noise",
        },
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "pbs_queue": os.environ.get("PBS_QUEUE"),
        "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
        "host": os.uname().nodename,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_properties": str(torch.cuda.get_device_properties(device)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "git": _git_state(repo),
        "python": sys.version,
    }


def _run_task(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this task on an A100 or Hopper node")

    num_chunks = int(getattr(args, "num_chunks", 1))
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    requested_checkpoint = getattr(args, "checkpoint_name", None)
    if requested_checkpoint:
        if requested_checkpoint not in CONFIGS:
            raise ValueError(
                f"unknown checkpoint {requested_checkpoint!r}; choose from {sorted(CONFIGS)}"
            )
        campaign_checkpoint_names = [requested_checkpoint]
        expected_worker_count = len(TARGETS) * num_chunks
        if args.worker_index < 0 or args.worker_index >= expected_worker_count:
            raise ValueError(
                f"worker index must be in [0, {expected_worker_count - 1}] for "
                f"{requested_checkpoint} with {num_chunks} chunks"
            )
        budget_index = args.worker_index // num_chunks
        args.chunk_index = args.worker_index % num_chunks
        args.visual_tokens = TARGETS[budget_index]
        args.target_visual_tokens = args.visual_tokens
        if num_chunks > 1:
            task_id = f"{args.checkpoint_name}_v{args.visual_tokens}_chunk{args.chunk_index}"
        else:
            task_id = f"{args.checkpoint_name}_v{args.visual_tokens}"
    else:
        campaign_checkpoint_names = list(CHECKPOINTS)
        expected_worker_count = len(CHECKPOINTS) * len(TARGETS) * num_chunks
        if args.worker_index < 0 or args.worker_index >= expected_worker_count:
            raise ValueError(f"worker index must be in [0, {expected_worker_count - 1}]")
        checkpoint_index = args.worker_index // (len(TARGETS) * num_chunks)
        budget_index = (args.worker_index // num_chunks) % len(TARGETS)
        args.chunk_index = args.worker_index % num_chunks
        args.checkpoint_name = CHECKPOINTS[checkpoint_index]
        args.visual_tokens = TARGETS[budget_index]
        args.target_visual_tokens = args.visual_tokens
        task_id = (
            f"{args.checkpoint_name}_v{args.visual_tokens}_chunk{args.chunk_index}"
            if num_chunks > 1
            else f"{args.checkpoint_name}_v{args.visual_tokens}"
        )
    args.num_chunks = num_chunks
    args.campaign_checkpoint_names = campaign_checkpoint_names
    args.target_model = str(resolve_workspace_path(args.target_model))
    args.manifest = resolve_workspace_path(args.manifest)
    args.video_root = resolve_workspace_path(args.video_root)
    args.calibration = resolve_workspace_path(args.calibration)
    args.cohort_selection = resolve_workspace_path(args.cohort_selection)
    args.checkpoint_root = resolve_workspace_path(args.checkpoint_root)
    args.config_root = resolve_workspace_path(args.config_root)
    args.output_dir = resolve_workspace_path(args.output_dir)
    configure_video_reader(args.video_reader)
    checkpoint = _checkpoint_path(args.checkpoint_root, args.checkpoint_name)
    config = args.config_root / f"{CONFIGS[args.checkpoint_name]}.json"
    if not checkpoint.exists() or not config.is_file():
        raise FileNotFoundError(f"checkpoint/config pair is unavailable: {checkpoint} / {config}")

    task_dir = args.output_dir / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = task_dir / "worker.json"
    journal_path = task_dir / "cases.jsonl"
    previous: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(journal_path):
        previous[str(row.get("row_id"))] = row

    all_records = _load_records(args.manifest, args.video_root, args.calibration, "vdc")
    cohort_ids = _cohort_ids(args.cohort_selection)
    expected_case_ids = _campaign_expected_case_ids(
        all_records,
        cohort_ids,
        campaign_checkpoint_names,
        num_chunks=num_chunks,
        sample_limit=args.limit,
    )
    run_config = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset_manifest": str(args.manifest),
        "video_root": str(args.video_root),
        "calibration": str(args.calibration),
        "cohort_selection": str(args.cohort_selection),
        "target_model": args.target_model,
        "checkpoint_names": campaign_checkpoint_names,
        "target_visual_tokens": list(TARGETS),
        "conditions": [condition["visual_condition"] for condition in CONDITIONS],
        "num_chunks": num_chunks,
        "sample_limit_per_chunk": args.limit,
        "expected_case_ids": expected_case_ids,
        "expected_row_count": len(expected_case_ids),
        "requested_max_new_tokens": int(args.requested_max_new_tokens),
    }
    _write_run_config(args.output_dir / "run_config.json", run_config)

    records = [record for record in all_records if str(record["sample_id"]) in cohort_ids]
    records = [
        record
        for record in records
        if (record.get("calibration_by_target", {}).get(args.visual_tokens, {}).get("status") == "ok")
    ]
    if num_chunks > 1:
        chunk_size = math.ceil(len(records) / num_chunks)
        start_idx = getattr(args, "chunk_index", 0) * chunk_size
        end_idx = min(start_idx + chunk_size, len(records))
        records = records[start_idx:end_idx]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"no calibrated cohort records at {args.visual_tokens} visual tokens")

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    processor, target, target_load_s = _load_target(
        args.target_model,
        device=device,
        dtype=dtype,
        attention=args.target_attention,
        device_map="cuda",
    )
    draft, draft_load_s = _load_draft(
        str(checkpoint), str(config), device=device, dtype=dtype
    )
    # Eager is required to expose the actual DFlash attention probabilities.
    # Both the trace pass and its timing repeat use this same implementation.
    draft.config._attn_implementation = "eager"

    metadata = _worker_metadata(
        args,
        task_id,
        checkpoint,
        config,
        [str(record["sample_id"]) for record in records],
    )
    metadata["target_load_s"] = target_load_s
    metadata["draft_load_s"] = draft_load_s
    _atomic_json(metadata_path, metadata)

    writer = AttentionShardWriter(
        task_dir / "attention",
        max_buffer_bytes=args.attention_max_mib * 1024 * 1024,
        compression_level=args.zstd_level,
        compression_threads=args.compression_threads,
    )
    args.experiment = "h3_1"
    args.dataset = "vdc"
    args.training_corpus = "llava68k"
    args.draft_depth = 3
    args.checkpoint = str(checkpoint)
    args.draft_config = str(config)
    args.device = str(device)
    args.draft_device = str(device)
    args.allow_out_of_tolerance = False
    args.mvbench_num_frames = 8
    args.mvbench_min_pixels = 256 * 28 * 28
    args.mvbench_max_pixels = 256 * 28 * 28
    args.capture_attention = False
    args.capture_all_attention = False
    args.attention_record_sink = None
    args.attention_metadata = None
    args.max_new_tokens = int(args.requested_max_new_tokens)

    target_cache: dict[str, dict[str, Any]] = {}
    total_rows = 0
    skipped_rows = 0
    for sample_index, sample_record in enumerate(records, start=1):
        expected_case_ids = [
            f"{args.checkpoint_name}:{sample_record['sample_id']}:{args.visual_tokens}:{condition['visual_condition']}"
            for condition in CONDITIONS
        ]
        if all(expected in previous and _completed(previous[expected]) for expected in expected_case_ids):
            skipped_rows += len(expected_case_ids)
            print(f"[resume] {sample_index}/{len(records)} {sample_record['sample_id']} complete", flush=True)
            continue

        prompt_record = _build_prompt_record(sample_record)
        prompt, visual_positions, fingerprint, settings = _prepare_prompt(
            processor=processor,
            target=target,
            sample=prompt_record,
            video_root=str(args.video_root),
            device=device,
            target_visual_tokens=args.visual_tokens,
            allow_out_of_tolerance=False,
            video_reader=args.video_reader,
        )
        prepared_context = (prompt, visual_positions, fingerprint, settings)
        prompt_id = hashlib.sha256(
            f"{sample_record['sample_id']}:{args.visual_tokens}".encode()
        ).hexdigest()[:20]
        prompt_path = task_dir / "prompts" / f"{prompt_id}.json"
        if not prompt_path.exists():
            input_ids = [int(value) for value in prompt.inputs["input_ids"][0].detach().cpu().tolist()]
            _atomic_json(
                prompt_path,
                {
                    "sample_id": sample_record["sample_id"],
                    "target_visual_tokens": args.visual_tokens,
                    "actual_visual_tokens": len(visual_positions),
                    "visual_positions": visual_positions,
                    "input_fingerprint": fingerprint,
                    "input_ids": input_ids,
                    "input_text": processor.tokenizer.decode(input_ids, skip_special_tokens=True),
                    "candidate_settings": settings,
                    "sample_fingerprint": sample_record.get("input_fingerprint"),
                    "video_path": sample_record.get("local_video_path"),
                    "question": sample_record.get("question"),
                    "reference": sample_record.get("answer"),
                },
            )

        if sample_index == 1 and not args.no_warmup:
            warmup_cache: dict[str, dict[str, Any]] = {}
            original_limit = args.max_new_tokens
            args.max_new_tokens = min(16, original_limit)
            try:
                _run_one(
                    args=args,
                    sample_record=sample_record,
                    prompt_variant="natural",
                    condition=CONDITIONS[0],
                    target=target,
                    draft=draft,
                    processor=processor,
                    device=device,
                    draft_device=device,
                    target_cache=warmup_cache,
                    prepared_context=prepared_context,
                )
            finally:
                args.max_new_tokens = original_limit
                warmup_cache.clear()
                torch.cuda.empty_cache()

        for condition in CONDITIONS:
            row_id = (
                f"{args.checkpoint_name}:{sample_record['sample_id']}:"
                f"{args.visual_tokens}:{condition['visual_condition']}"
            )
            old = previous.get(row_id)
            if old is not None and _completed(old):
                skipped_rows += 1
                continue
            print(
                f"[visualcut] {sample_index}/{len(records)} {task_id} "
                f"{sample_record['sample_id']} {condition['visual_condition']}",
                flush=True,
            )
            writer.begin_case(row_id)
            args.max_new_tokens = int(args.requested_max_new_tokens)
            args.capture_attention = True
            args.capture_all_attention = True
            args.attention_record_sink = writer.append
            args.attention_metadata = {
                "run_id": args.run_id,
                "task_id": task_id,
                "row_id": row_id,
                "checkpoint_name": args.checkpoint_name,
                "sample_id": sample_record["sample_id"],
                "visual_token_target": args.visual_tokens,
                "visual_condition": condition["visual_condition"],
                "retention_percentage": condition["retention_percentage"],
                "target_input_fingerprint": fingerprint,
            }
            attention_capture = None
            try:
                captured = _run_one(
                    args=args,
                    sample_record=sample_record,
                    prompt_variant="natural",
                    condition=condition,
                    target=target,
                    draft=draft,
                    processor=processor,
                    device=device,
                    draft_device=device,
                    target_cache=target_cache,
                    prepared_context=prepared_context,
                )
                attention_capture = writer.finish_case()
                args.capture_attention = False
                args.capture_all_attention = False
                args.attention_record_sink = None
                args.attention_metadata = None
                timed = _run_one(
                    args=args,
                    sample_record=sample_record,
                    prompt_variant="natural",
                    condition=condition,
                    target=target,
                    draft=draft,
                    processor=processor,
                    device=device,
                    draft_device=device,
                    target_cache=target_cache,
                    prepared_context=prepared_context,
                )
                matched = (
                    captured["target_output_hash"] == timed["target_output_hash"]
                    and captured["speculative_output_hash"] == timed["speculative_output_hash"]
                    and captured["acceptance_rounds"] == timed["acceptance_rounds"]
                )
                row = dict(captured)
                prompt_input_ids = row.pop("prompt_input_ids", None)
                row.pop("prompt_input_text", None)
                row.pop("attention_summary", None)
                row["prompt_record"] = str(prompt_path)
                row["prompt_input_ids_sha256"] = hashlib.sha256(
                    json.dumps(prompt_input_ids, separators=(",", ":")).encode()
                ).hexdigest()
                row["timing"] = timed.get("timing", {})
                row["timing_measurement_backend"] = "dflash_eager_attention"
                row["capture_pass_timing"] = captured.get("timing", {})
                row["capture_and_timing_outputs_match"] = bool(matched)
                row["run_id"] = args.run_id
                row["checkpoint_name"] = args.checkpoint_name
                row["attention_capture"] = attention_capture
                row["row_id"] = row_id
                row["status"] = (
                    "ok" if matched and row.get("lossless") and attention_capture["record_count"] > 0
                    else "mismatch"
                )
                if not matched:
                    row["pairing_error"] = "capture and timing repeat produced different outputs or acceptance rounds"
                elif not row.get("lossless"):
                    row["pairing_error"] = "speculative output did not match greedy target output"
                elif attention_capture["record_count"] <= 0:
                    row["pairing_error"] = "no DFlash attention weights were captured"
                _append_jsonl(journal_path, row)
                previous[row_id] = row
                total_rows += 1
                print(
                    f"[visualcut-done] {row_id} status={row['status']} "
                    f"lossless={row.get('lossless')} attention_records={attention_capture['record_count']}",
                    flush=True,
                )
            except Exception as exc:
                if writer.active_case_id is not None:
                    try:
                        attention_capture = writer.finish_case()
                    except Exception:
                        attention_capture = writer.abort_case()
                if attention_capture is None:
                    attention_capture = {"case_id": row_id, "record_count": 0, "shards": []}
                args.capture_attention = False
                args.capture_all_attention = False
                args.attention_record_sink = None
                args.attention_metadata = None
                error_row = {
                    "row_id": row_id,
                    "run_id": args.run_id,
                    "checkpoint_name": args.checkpoint_name,
                    "sample_id": sample_record["sample_id"],
                    "visual_token_target": args.visual_tokens,
                    "visual_condition": condition["visual_condition"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "attention_capture": attention_capture,
                }
                _append_jsonl(journal_path, error_row)
                previous[row_id] = error_row
                total_rows += 1
                print(f"[visualcut-error] {row_id}: {error_row['error']}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            finally:
                args.capture_attention = False
                args.capture_all_attention = False
                args.attention_record_sink = None
                args.attention_metadata = None

        target_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    del draft, target, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "task_id": task_id,
        "rows_written": total_rows,
        "rows_skipped": skipped_rows,
        "cohort_size": len(records),
        "journal": str(journal_path),
        "worker_metadata": str(metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", "0")))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--cohort-selection", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--video-reader", choices=("torchvision", "decord"), default="decord")
    parser.add_argument("--max-new-tokens", dest="requested_max_new_tokens", type=int, default=256)
    parser.add_argument("--attention-max-mib", type=int, default=128)
    parser.add_argument("--zstd-level", type=int, default=1)
    parser.add_argument("--compression-threads", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-warmup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.requested_max_new_tokens <= 0 or args.attention_max_mib <= 0:
        raise SystemExit("max-new-tokens and attention-max-mib must be positive")
    if args.num_chunks <= 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("num-chunks and limit must be positive")
    result = _run_task(args)
    print(f"[task-done] {json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
