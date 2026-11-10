"""Run paired full/cut DFlash distribution replay over calibrated VDC samples."""

from __future__ import annotations

import argparse
import gzip
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

from src.analyze.Validate_Sparrow_hypothesises.dflash_distribution_metrics import (
    paired_block_metrics,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import (
    _load_records,
    build_visual_keep_mask,
    build_prompt_record,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import (
    _decode_prompt,
    _prepare_prompt,
)
from src.analyze.Validate_Sparrow_hypothesises.video_compat import configure_video_reader
from src.infer.qwen25vl_dflash_compare import (
    _load_draft,
    _load_target,
    _resolve_dtype,
    _sha256_tokens,
    _target_greedy,
    _eos_token_ids,
)


TARGETS = (400, 3000, 13000, 25000)
BALANCED_CHECKPOINT = "dflash_3layer_1_9_17_25_33"
BALANCED_CONFIG = "balanced.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_inventory_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint model files by path, size, and mtime without rereading weights."""
    resolved = path.resolve()
    if resolved.is_file():
        files = [resolved]
    elif resolved.is_dir():
        files = sorted(item for item in resolved.rglob("*") if item.is_file())
    else:
        return {"path": str(path), "exists": False}
    digest = hashlib.sha256()
    entries = []
    for item in files:
        stat = item.stat()
        relative = item.name if resolved.is_file() else str(item.relative_to(resolved))
        entry = {"path": relative, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        entries.append(entry)
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "fingerprint_method": "relative_path_size_mtime_inventory",
        "file_count": len(entries),
        "inventory_sha256": digest.hexdigest(),
    }


def _source_fingerprints(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "src/analyze/Validate_Sparrow_hypothesises/dflash_distribution_metrics.py",
        "src/analyze/Validate_Sparrow_hypothesises/run_dflash_distribution_replay.py",
        "src/analyze/Validate_Sparrow_hypothesises/run_dflash_experiments.py",
        "src/infer/qwen25vl_dflash_compare.py",
        "src/train_Dflash_SpecForge/specforge/modeling/draft/dflash.py",
    )
    return {name: _sha256(repo_root / name) for name in relative_paths}


def _read_cohort(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("valid"):
        raise ValueError(f"calibrated cohort is invalid: {path}")
    sample_ids = [str(value) for value in payload.get("sample_ids", [])]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"cohort selection contains duplicate sample IDs: {path}")
    if not sample_ids:
        raise ValueError(f"cohort selection is empty: {path}")
    return sample_ids


def _checkpoint_path(root: Path, checkpoint_name: str) -> Path:
    model_tag = f"qwen25vl-3b-dflash-llava68k-{checkpoint_name}"
    return root / checkpoint_name / "phase2" / model_tag / f"{model_tag}-latest"


def _build_prompt_record(sample_record: dict[str, Any]) -> dict[str, Any]:
    sample_proxy = type("SampleProxy", (), sample_record)()
    prompt_record = build_prompt_record(sample_proxy, "natural")
    prompt_record.update(
        {
            "answer": sample_record.get("answer"),
            "local_video_path": sample_record.get("local_video_path"),
            "video_path": sample_record.get("local_video_path"),
            "dataset_format": "vdc",
        }
    )
    prompt_record["calibration_by_target"] = sample_record.get("calibration_by_target", {})
    return prompt_record


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_run_config(path: Path, config: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"run configuration mismatch at {path}; use a new output directory"
            )
        return
    _atomic_json(path, config)


def _read_pair_status(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        if not lines:
            return None
        last = json.loads(lines[-1])
        return last if last.get("record_type") == "pair_status" else None
    except (OSError, EOFError, json.JSONDecodeError):
        return None


def _append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def _pair_file(output_dir: Path, sample_id: str, target: int) -> Path:
    sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    return output_dir / "pairs" / f"{sample_key}.v{target}.jsonl.gz"


def _pair_distribution(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    target_visual_tokens: int,
    target: Any,
    draft: Any,
    processor: Any,
    device: torch.device,
    output_dir: Path,
    checkpoint_fingerprint: dict[str, Any],
    target_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    calibration = record.get("calibration_by_target", {}).get(target_visual_tokens)
    if not isinstance(calibration, dict) or calibration.get("status") != "ok":
        raise ValueError(
            f"missing usable calibration for {record['sample_id']} at {target_visual_tokens}"
        )
    prompt_record = _build_prompt_record(record)
    prompt, visual_positions, fingerprint, settings = _prepare_prompt(
        processor=processor,
        target=target,
        sample=prompt_record,
        video_root=str(args.video_root),
        device=device,
        target_visual_tokens=target_visual_tokens,
        allow_out_of_tolerance=False,
        video_reader=args.video_reader,
    )
    expected_visual_count = calibration.get("actual_visual_tokens")
    if expected_visual_count is not None and int(expected_visual_count) != len(visual_positions):
        raise ValueError(
            f"visual token count changed for {record['sample_id']} at {target_visual_tokens}: "
            f"calibration={expected_visual_count}, prompt={len(visual_positions)}"
        )

    pair_path = _pair_file(output_dir, str(record["sample_id"]), target_visual_tokens)
    if pair_path.exists():
        status = _read_pair_status(pair_path)
        if status is None or status.get("status") != "ok":
            raise ValueError(f"existing pair artifact is incomplete or invalid: {pair_path}")
        return {"pair_id": status["pair_id"], "status": "skipped", "positions": status["position_count"]}

    temporary_path = pair_path.with_suffix(pair_path.suffix + ".partial")
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    input_ids = prompt.inputs["input_ids"]
    keep_mask = build_visual_keep_mask(
        int(input_ids.shape[1]),
        visual_positions,
        retention_percentage=0.0,
    )
    if not visual_positions or bool(keep_mask[visual_positions].any()):
        raise ValueError("cut mask did not remove the calibrated visual positions")
    target_ids_tensor, target_timing = _target_greedy(
        target,
        prompt,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=_eos_token_ids(processor, target),
        device=device,
    )
    prompt_length = int(input_ids.shape[1])
    target_ids = [
        int(value)
        for value in target_ids_tensor[0, prompt_length:].detach().cpu().tolist()
    ]
    pair_id = f"{record['sample_id']}:{target_visual_tokens}"
    prompt_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    pair_meta = {
        "record_type": "pair_meta",
        "pair_id": pair_id,
        "sample_id": str(record["sample_id"]),
        "target_visual_tokens": int(target_visual_tokens),
        "actual_visual_tokens": len(visual_positions),
        "visual_positions": [int(value) for value in visual_positions],
        "prompt_length": prompt_length,
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "target_input_fingerprint": fingerprint,
        "sample_fingerprint": record.get("input_fingerprint"),
        "calibration_status": calibration.get("status"),
        "calibration_actual_visual_tokens": expected_visual_count,
        "calibration_settings": settings,
        "target_greedy_ids": target_ids,
        "target_greedy_ids_sha256": _sha256_tokens(target_ids),
        "max_new_tokens": int(args.max_new_tokens),
        "checkpoint": str(args.checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "target_model": str(args.target_model),
        "target_model_fingerprint": target_fingerprint,
        "draft_config": str(args.draft_config),
        "draft_config_sha256": _sha256(args.draft_config),
        "conditions": ["full", "cut"],
        "cut_keep_prompt_positions": int(keep_mask.sum().item()),
        "target_timing": target_timing,
    }

    position_count = 0
    with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=1) as handle:
        _append_jsonl(handle, pair_meta)

        def collect_pair(
            full_logits: torch.Tensor,
            cut_logits: torch.Tensor,
            block_target_ids: list[int],
            block_metadata: dict[str, Any],
        ) -> None:
            nonlocal position_count
            metrics_by_position = paired_block_metrics(
                full_logits,
                cut_logits,
                block_target_ids,
            )
            for slot, (metrics, target_id) in enumerate(
                zip(metrics_by_position, block_target_ids, strict=True)
            ):
                output_position = int(block_metadata["proposal_position_start"]) + slot
                _append_jsonl(
                    handle,
                    {
                        "record_type": "position",
                        "pair_id": pair_id,
                        "sample_id": str(record["sample_id"]),
                        "target_visual_tokens": int(target_visual_tokens),
                        "target_position": output_position,
                        "target_greedy_token_id": int(target_id),
                        "block_index": int(block_metadata["round_index"]),
                        "block_anchor_position": int(block_metadata["anchor_position"]),
                        "block_anchor_token_id": int(block_metadata["anchor_token_id"]),
                        "draft_slot_in_block": slot + 1,
                        "mask_token_id": int(block_metadata["mask_token_id"]),
                        "paired_state": "ok",
                        **metrics,
                    },
                )
                position_count += 1

        result = _decode_prompt(
            target=target,
            draft=draft,
            processor=processor,
            prompt=prompt,
            device=device,
            draft_device=device,
            max_new_tokens=args.max_new_tokens,
            target_output_ids=target_ids,
            target_timing=target_timing,
            distribution_cut_keep_mask=keep_mask,
            distribution_sink=collect_pair,
        )
        full_matches_target = result["speculative_output_ids"] == target_ids
        status = "ok" if full_matches_target and position_count > 0 else "mismatch"
        _append_jsonl(
            handle,
            {
                "record_type": "pair_status",
                "pair_id": pair_id,
                "sample_id": str(record["sample_id"]),
                "target_visual_tokens": int(target_visual_tokens),
                "status": status,
                "position_count": position_count,
                "full_matches_target_greedy": bool(full_matches_target),
                "target_output_hash": _sha256_tokens(target_ids),
                "full_output_hash": _sha256_tokens(result["speculative_output_ids"]),
                "target_timing": target_timing,
                "full_metrics": result["metrics"],
            },
        )
        handle.flush()
    if status != "ok":
        raise RuntimeError(
            f"paired distribution replay failed for {pair_id}: "
            f"full_matches_target={full_matches_target}, positions={position_count}; "
            f"partial artifact: {temporary_path}"
        )
    os.replace(temporary_path, pair_path)
    del target_ids_tensor, prompt, target_timing
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"pair_id": pair_id, "status": "ok", "positions": position_count}


def _load_config(path: Path) -> argparse.Namespace:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(payload.get("repo_root", Path.cwd())).resolve()

    def resolve_config_path(key: str) -> Path:
        value = Path(payload[key])
        return value.resolve() if value.is_absolute() else (root / value).resolve()
    required = (
        "run_id",
        "manifest",
        "video_root",
        "calibration",
        "cohort_selection",
        "target_model",
        "output_dir",
    )
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(f"run config is missing fields: {missing}")
    if not payload.get("checkpoint") and not payload.get("checkpoint_root"):
        raise ValueError("run config requires checkpoint or checkpoint_root")
    checkpoint_name = payload.get("checkpoint_name", BALANCED_CHECKPOINT)
    budgets = [int(value) for value in payload.get("budgets", TARGETS)]
    if not budgets or any(value not in TARGETS for value in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError(f"budgets must be unique members of {TARGETS}")
    cohort_size = int(payload.get("cohort_size", 50))
    sample_limit = payload.get("sample_limit")
    if sample_limit is not None:
        sample_limit = int(sample_limit)
        if sample_limit <= 0 or sample_limit > cohort_size:
            raise ValueError("sample_limit must be between 1 and cohort_size")
    config = payload.get("draft_config")
    if config:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = root / config_path
    else:
        config_path = root / "configs/dflash_configs" / BALANCED_CONFIG
    checkpoint = payload.get("checkpoint")
    checkpoint_root = (
        resolve_config_path("checkpoint_root")
        if payload.get("checkpoint_root")
        else None
    )
    checkpoint_path = (
        Path(checkpoint)
        if checkpoint
        else _checkpoint_path(checkpoint_root, checkpoint_name)
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    return argparse.Namespace(
        repo_root=root,
        run_id=str(payload["run_id"]),
        manifest=resolve_config_path("manifest"),
        video_root=resolve_config_path("video_root"),
        calibration=resolve_config_path("calibration"),
        cohort_selection=resolve_config_path("cohort_selection"),
        target_model=str(payload["target_model"]),
        target_model_path=resolve_config_path("target_model"),
        target_model_revision=str(payload.get("target_model_revision", "")),
        checkpoint_root=checkpoint_root or checkpoint_path.parent,
        checkpoint_name=checkpoint_name,
        checkpoint=checkpoint_path,
        draft_config=config_path.resolve(),
        output_dir=resolve_config_path("output_dir"),
        budgets=budgets,
        cohort_size=cohort_size,
        sample_limit=sample_limit,
        max_new_tokens=int(payload.get("max_new_tokens", 256)),
        device=str(payload.get("device", "cuda:0")),
        dtype=str(payload.get("dtype", "auto")),
        target_attention=str(payload.get("target_attention", "eager")),
        draft_attention=str(payload.get("draft_attention", "eager")),
        video_reader=str(payload.get("video_reader", "decord")),
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the distribution replay requires a GPU")
    for path in (args.manifest, args.calibration, args.cohort_selection, args.draft_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.video_root, args.checkpoint, args.target_model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    configure_video_reader(args.video_reader)
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    selection_ids = _read_cohort(args.cohort_selection)
    if len(selection_ids) != args.cohort_size:
        raise ValueError(
            f"expected {args.cohort_size} cohort samples, found {len(selection_ids)}"
        )
    all_records = _load_records(args.manifest, args.video_root, args.calibration, "vdc")
    by_id = {str(record["sample_id"]): record for record in all_records}
    absent = [sample_id for sample_id in selection_ids if sample_id not in by_id]
    if absent:
        raise ValueError(f"cohort IDs missing from VDC manifest: {absent[:5]}")
    records = [by_id[sample_id] for sample_id in selection_ids]
    for budget in args.budgets:
        missing = [
            str(record["sample_id"])
            for record in records
            if record.get("calibration_by_target", {}).get(budget, {}).get("status") != "ok"
        ]
        if missing:
            raise ValueError(
                f"budget {budget} is not calibrated for all selected cohort samples; "
                f"first missing IDs: {missing[:5]}"
            )
    selected_records = records[: args.sample_limit] if args.sample_limit else records

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "run_id": args.run_id,
        "checkpoint_name": args.checkpoint_name,
        "checkpoint": str(args.checkpoint),
        "draft_config": str(args.draft_config),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "video_root": str(args.video_root),
        "calibration": str(args.calibration),
        "calibration_sha256": _sha256(args.calibration),
        "cohort_selection": str(args.cohort_selection),
        "cohort_selection_sha256": _sha256(args.cohort_selection),
        "cohort_ids": selection_ids,
        "selected_sample_ids": [str(record["sample_id"]) for record in selected_records],
        "cohort_size": args.cohort_size,
        "sample_limit": args.sample_limit,
        "budgets": args.budgets,
        "conditions": ["full", "cut"],
        "target_model": args.target_model,
        "target_model_revision": args.target_model_revision or None,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "max_new_tokens": args.max_new_tokens,
        "target_generation": "greedy",
        "draft_proposal": "argmax_raw_logits",
        "distribution_probability_temperature": 1.0,
        "distribution_log_base": 2,
        "device": str(device),
        "dtype": str(dtype),
        "target_attention": args.target_attention,
        "draft_attention": args.draft_attention,
        "video_reader": args.video_reader,
        "target_visual_context_in_cut": "unchanged_full_target_context",
        "cut_condition": "remove_visual_positions_from_dflash_context_only",
        "position_policy": "same_full_decode_rounds_anchors_and_masks_for_inline_cut_replay",
        "target_checkpoint_inventory": _path_inventory_fingerprint(args.target_model_path),
        "draft_checkpoint_inventory": _path_inventory_fingerprint(args.checkpoint),
        "draft_config_sha256": _sha256(args.draft_config),
        "source_fingerprints": _source_fingerprints(args.repo_root),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python": sys.version,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_properties": str(torch.cuda.get_device_properties(device)),
        "expected_pair_count": len(selected_records) * len(args.budgets),
    }
    _write_run_config(args.output_dir / "run_config.json", config)

    processor, target, target_load_s = _load_target(
        args.target_model,
        device=device,
        dtype=dtype,
        attention=args.target_attention,
        device_map="cuda",
    )
    draft, draft_load_s = _load_draft(
        str(args.checkpoint),
        str(args.draft_config),
        device=device,
        dtype=dtype,
    )
    expected_target_layers = (1, 9, 17, 25, 33)
    actual_target_layers = tuple(
        int(value) for value in getattr(draft, "target_layer_ids", ())
    )
    if actual_target_layers != expected_target_layers:
        raise ValueError(
            "this distribution replay uses the balanced target-layer selection "
            f"{expected_target_layers}, got {actual_target_layers} from "
            f"{args.draft_config}"
        )
    draft.config._attn_implementation = args.draft_attention
    checkpoint_fingerprint = _path_inventory_fingerprint(args.checkpoint)
    target_fingerprint = _path_inventory_fingerprint(args.target_model_path)
    _atomic_json(
        args.output_dir / "model_load.json",
        {
            "target_load_s": target_load_s,
            "draft_load_s": draft_load_s,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "target_model_fingerprint": target_fingerprint,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
    )

    completed = 0
    skipped = 0
    for sample_index, record in enumerate(selected_records, start=1):
        for budget in args.budgets:
            pair_path = _pair_file(args.output_dir, str(record["sample_id"]), budget)
            if pair_path.exists():
                status = _read_pair_status(pair_path)
                if status is None or status.get("status") != "ok":
                    raise ValueError(f"existing pair artifact is incomplete or invalid: {pair_path}")
                skipped += 1
                continue
            print(
                f"[distribution-pair] {sample_index}/{len(selected_records)} "
                f"{record['sample_id']} visual={budget}",
                flush=True,
            )
            result = _pair_distribution(
                args=args,
                record=record,
                target_visual_tokens=budget,
                target=target,
                draft=draft,
                processor=processor,
                device=device,
                output_dir=args.output_dir,
                checkpoint_fingerprint=checkpoint_fingerprint,
                target_fingerprint=target_fingerprint,
            )
            completed += 1
            print(f"[distribution-pair-done] {json.dumps(result, sort_keys=True)}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    del draft, target, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "run_id": args.run_id,
        "new_pairs": completed,
        "skipped_pairs": skipped,
        "selected_samples": len(selected_records),
        "budgets": args.budgets,
        "run_config": str(args.output_dir / "run_config.json"),
        "pairs_dir": str(args.output_dir / "pairs"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    args = _load_config(parsed.config.resolve())
    result = _run(args)
    print(f"[distribution-run-done] {json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
