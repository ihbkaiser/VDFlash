from __future__ import annotations

"""Cache exact Qwen2.5-VL DFlash inputs for Phase-3 self-distillation.

This command consumes ``responses/{split}.jsonl`` from
``src.train_VLM.prepare_posttraining targets``. It caches only target hidden
features, token IDs, loss masks, and M-RoPE positions. Full-vocabulary logits
from the DFlash teacher are computed on-the-fly by the training model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .config import DFlashTrainConfig
from .target import Qwen25VLTargetAdapter, load_jsonl


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draft_target_layers(path: str) -> list[int]:
    config_path = Path(path).expanduser()
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "--draft-config must be a local DFlash config JSON file or a "
            f"directory containing config.json; got {path!r}"
        )
    with config_path.open(encoding="utf-8") as reader:
        payload = json.load(reader)
    method = payload.get("dflash_config") or {}
    layer_ids = method.get("target_layer_ids")
    if not isinstance(layer_ids, list) or not layer_ids:
        raise ValueError(f"{config_path} does not define dflash_config.target_layer_ids")
    parsed = [int(layer_id) for layer_id in layer_ids]
    if any(layer_id < 0 for layer_id in parsed) or len(set(parsed)) != len(parsed):
        raise ValueError("DFlash target_layer_ids must be unique non-negative integers")
    return parsed


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as writer:
            json.dump(payload, writer, indent=2, sort_keys=True, ensure_ascii=False)
            writer.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _distributed_context() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _all_gather_objects(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return gathered


def _feature_path(split_root: Path, sample_index: int, group_size: int) -> Path:
    group_start = sample_index // group_size * group_size
    group_end = group_start + group_size - 1
    return (
        split_root
        / f"rows_{group_start:06d}-{group_end:06d}"
        / f"sample_{sample_index:06d}.ckpt"
    )


def _cache_one(
    adapter: Qwen25VLTargetAdapter,
    record: dict[str, Any],
    *,
    max_seq_length: int,
    layer_ids: list[int],
    output_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], int, int]:
    example = adapter.prepare_record(record, max_seq_length=max_seq_length)
    loss_mask = torch.zeros_like(example.input_ids, dtype=torch.float32)
    loss_mask[:, example.response_start : example.response_end] = 1.0
    hidden = adapter.forward_selected_hidden(example.inputs, layer_ids)
    if hidden.ndim != 3 or hidden.shape[:2] != example.input_ids.shape:
        raise ValueError(
            f"target hidden shape {tuple(hidden.shape)} does not align with token IDs "
            f"{tuple(example.input_ids.shape)} for sample {example.sample_id}"
        )
    if not torch.any(loss_mask[:, 1:] * loss_mask[:, :-1]):
        raise ValueError(
            f"sample {example.sample_id} has no adjacent supervised target tokens"
        )
    visual_ids = set().union(*adapter.visual_token_ids)
    if not visual_ids:
        raise ValueError("target config does not expose Qwen image/video token IDs")
    visual_count = sum(
        int((example.input_ids == token_id).sum().item()) for token_id in visual_ids
    )
    if visual_count == 0:
        raise ValueError(f"sample {example.sample_id} has no visual placeholder tokens")
    features = {
        "input_ids": example.input_ids.squeeze(0).detach().to("cpu", dtype=torch.int32),
        "loss_mask": loss_mask.squeeze(0).detach().to("cpu", dtype=torch.float32),
        "hidden_states": hidden.squeeze(0).detach().to("cpu", dtype=output_dtype),
        "position_ids": example.position_ids.squeeze(1).detach().to("cpu", dtype=torch.int32),
    }
    return features, int(example.input_ids.shape[-1]), visual_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline target-feature caches for visual-10 DFlash self-distillation."
    )
    parser.add_argument("--data-dir", required=True, help="Prepared post-training dataset directory")
    parser.add_argument("--target-config", required=True, help="JSON used by prepare_posttraining targets")
    parser.add_argument("--draft-config", required=True, help="Phase-2 DFlash architecture config JSON")
    parser.add_argument("--output-dir", required=True, help="Output root for train/validation/test features")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--feature-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--group-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional deterministic pilot limit per split")
    parser.add_argument("--overwrite", action="store_true", help="Replace the selected output split directories")
    return parser.parse_args()


def _load_target_config(path: str) -> DFlashTrainConfig:
    with Path(path).expanduser().open(encoding="utf-8") as reader:
        payload = json.load(reader)
    if not isinstance(payload, dict):
        raise ValueError("target config must contain a JSON object")
    return DFlashTrainConfig(**payload)


def _load_contract(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as reader:
        value = json.load(reader)
    if not isinstance(value, dict):
        raise ValueError(f"cache contract at {path} must be a JSON object")
    return value


def _prepare_split(
    *,
    split: str,
    data_dir: Path,
    output_root: Path,
    adapter: Qwen25VLTargetAdapter,
    layer_ids: list[int],
    max_seq_length: int,
    output_dtype: torch.dtype,
    output_dtype_name: str,
    rank: int,
    world_size: int,
    group_size: int,
    max_samples: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    response_path = data_dir / "responses" / f"{split}.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(
            f"missing {response_path}; run `prepare_posttraining targets` first"
        )
    records = load_jsonl(str(response_path))
    if max_samples is not None:
        records = records[:max_samples]
    split_root = output_root / split
    contract_path = split_root / "cache_contract.json"
    contract = {
        "schema": "dflash_visual10_target_features_v1",
        "split": split,
        "source_response_file": str(response_path.resolve()),
        "source_response_sha256": _sha256_file(response_path),
        "target_provenance": adapter.target_provenance(),
        "target_layer_ids": layer_ids,
        "layer_indexing": "qwen25vl_hf_decoder_zero_based",
        "phase": "phase2",
        "max_seq_length": max_seq_length,
        "feature_dtype": output_dtype_name,
        "num_records": len(records),
        "max_samples": max_samples,
        "group_size": group_size,
    }
    initialization_error = None
    if rank == 0:
        try:
            if overwrite and split_root.exists():
                shutil.rmtree(split_root)
            saved_contract = _load_contract(contract_path)
            if saved_contract is not None and saved_contract != contract:
                raise ValueError(
                    f"feature cache contract differs at {split_root}; use a new output "
                    "directory or pass --overwrite"
                )
            if saved_contract is None and split_root.exists() and any(split_root.iterdir()):
                raise FileExistsError(
                    f"{split_root} is non-empty but has no cache_contract.json; "
                    "choose a new output directory or pass --overwrite"
                )
            split_root.mkdir(parents=True, exist_ok=True)
            if saved_contract is None:
                _atomic_json(contract_path, contract)
            from specforge.hidden_state import (
                build_hidden_state_metadata,
                load_hidden_state_metadata,
                write_hidden_state_metadata,
            )

            text_config = adapter.text_config
            expected_metadata = build_hidden_state_metadata(
                target_layer_ids=layer_ids,
                hidden_size=int(text_config.hidden_size),
                phase="phase2",
                target_model=adapter.requested_model,
                target_model_revision=(
                    adapter.target_provenance().get("target_commit")
                    or adapter.requested_revision
                ),
                capture_method="dflash_visual10_target_cache",
                dtype=output_dtype_name,
                layer_indexing="qwen25vl_hf_decoder_zero_based",
                expected_count=None,
            )
            metadata_path = split_root / "hidden_state_metadata.json"
            if metadata_path.is_file():
                if load_hidden_state_metadata(split_root) != expected_metadata:
                    raise ValueError(
                        f"hidden-state metadata differs from the requested cache "
                        f"contract at {split_root}"
                    )
            else:
                # Repair an interrupted initialization where cache_contract.json
                # was committed but the metadata file was not written yet.
                write_hidden_state_metadata(split_root, expected_metadata)
        except Exception as exc:
            initialization_error = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        status = [initialization_error]
        dist.broadcast_object_list(status, src=0)
        initialization_error = status[0]
    if initialization_error is not None:
        raise RuntimeError(initialization_error)

    rank_index = split_root / f"index_rank{rank:05d}.jsonl"
    rank_records: list[tuple[int, dict[str, Any]]] = []
    mode = "w"
    sample_lengths: list[int] = []
    visual_counts: list[int] = []
    approximate_bytes = 0
    local_error = None
    try:
        completed: set[str] = set()
        if rank_index.exists():
            for row in load_jsonl(str(rank_index)):
                feature_path = split_root / str(row["file"])
                if feature_path.is_file():
                    completed.add(str(row["id"]))
        rank_records = [
            (index, record)
            for index, record in enumerate(records)
            if index % world_size == rank and str(record.get("id")) not in completed
        ]
        mode = "a" if rank_index.exists() else "w"
        with rank_index.open(mode, encoding="utf-8") as manifest:
            progress = tqdm(
                rank_records,
                desc=f"{split} features rank {rank}/{world_size}",
                unit="samples",
                disable=(world_size > 1 and rank != 0),
            )
            for sample_index, record in progress:
                sample_id = str(record.get("id", ""))
                if not sample_id:
                    raise ValueError(f"{response_path} has a record without an ID")
                features, sequence_length, visual_count = _cache_one(
                    adapter,
                    record,
                    max_seq_length=max_seq_length,
                    layer_ids=layer_ids,
                    output_dtype=output_dtype,
                )
                feature_path = _feature_path(split_root, sample_index, group_size)
                _atomic_torch_save(features, feature_path)
                relative_path = feature_path.relative_to(split_root).as_posix()
                row = {
                    "index": sample_index,
                    "id": sample_id,
                    "file": relative_path,
                    "sequence_length": sequence_length,
                    "visual_token_count": visual_count,
                }
                manifest.write(json.dumps(row, sort_keys=True) + "\n")
                manifest.flush()
                sample_lengths.append(sequence_length)
                visual_counts.append(visual_count)
                approximate_bytes += sum(
                    int(tensor.numel() * tensor.element_size())
                    for tensor in features.values()
                )
                progress.set_postfix(tokens=sequence_length, visual=visual_count)
    except Exception as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"

    rank_errors = _all_gather_objects(local_error, world_size)
    failures = [error for error in rank_errors if error is not None]
    if failures:
        raise RuntimeError("distributed feature caching failed: " + " | ".join(failures))

    local_rows: list[dict[str, Any]] = []
    local_error = None
    try:
        local_rows = load_jsonl(str(rank_index)) if rank_index.exists() else []
    except Exception as exc:
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
    rank_errors = _all_gather_objects(local_error, world_size)
    failures = [error for error in rank_errors if error is not None]
    if failures:
        raise RuntimeError("could not read rank manifests: " + " | ".join(failures))

    all_rank_rows = _all_gather_objects(local_rows, world_size)
    merge_error = None
    if rank == 0:
        try:
            merged: dict[int, dict[str, Any]] = {}
            for rows in all_rank_rows:
                for row in rows:
                    merged[int(row["index"])] = row
            missing = sorted(set(range(len(records))).difference(merged))
            if missing:
                raise RuntimeError(
                    f"{split} feature cache incomplete: {len(missing)} samples missing; "
                    f"first indices={missing[:8]}"
                )
            ordered = [merged[index] for index in range(len(records))]
            if not ordered:
                raise ValueError(f"{split} split has no response records to cache")
            with (split_root / "index.jsonl.tmp").open("w", encoding="utf-8") as writer:
                for row in ordered:
                    writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(split_root / "index.jsonl.tmp", split_root / "index.jsonl")
            summary = {
                **contract,
                "status": "complete",
                "num_records_cached": len(ordered),
                "sequence_length_min": min(row["sequence_length"] for row in ordered),
                "sequence_length_max": max(row["sequence_length"] for row in ordered),
                "visual_token_count_min": min(row["visual_token_count"] for row in ordered),
                "visual_token_count_max": max(row["visual_token_count"] for row in ordered),
                "estimated_tensor_bytes": sum(
                    _feature_path(split_root, row["index"], group_size).stat().st_size
                    for row in ordered
                ),
            }
            _atomic_json(split_root / "cache_summary.json", summary)
        except Exception as exc:
            merge_error = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        status = [merge_error]
        dist.broadcast_object_list(status, src=0)
        merge_error = status[0]
    if merge_error is not None:
        raise RuntimeError(merge_error)
    return {
        "split": split,
        "records": len(records),
        "rank_records_processed": len(rank_records),
        "local_feature_bytes": approximate_bytes,
        "sequence_lengths": sample_lengths,
        "visual_counts": visual_counts,
    }


def main() -> None:
    args = _parse_args()
    if args.group_size < 1:
        raise ValueError("--group-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.max_seq_length is not None and args.max_seq_length < 2:
        raise ValueError("--max-seq-length must be at least 2")
    if set(args.splits).difference({"train", "validation", "test"}):
        raise ValueError("--splits may contain only train, validation, and test")

    rank, world_size, local_rank, initialized_here = _distributed_context()
    try:
        initialization_error = None
        adapter = None
        try:
            data_dir = Path(args.data_dir).expanduser().resolve()
            output_root = Path(args.output_dir).expanduser().resolve()
            target_config = _load_target_config(args.target_config)
            layer_ids = _draft_target_layers(args.draft_config)
            max_seq_length = int(args.max_seq_length or target_config.max_seq_length)
            output_dtype = torch.bfloat16 if args.feature_dtype == "bf16" else torch.float16
            device = torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )
            adapter = Qwen25VLTargetAdapter.from_pretrained(
                target_config,
                device=device,
                dtype=(
                    torch.bfloat16
                    if target_config.mixed_precision == "bf16"
                    else None
                ),
            )
            adapter.freeze()
            # Validate provenance before creating feature files.
            for split in args.splits:
                response_path = data_dir / "responses" / f"{split}.jsonl"
                if not response_path.is_file():
                    raise FileNotFoundError(
                        f"missing response manifest: {response_path}"
                    )
                for record in load_jsonl(str(response_path))[: args.max_samples]:
                    adapter.validate_record_provenance(record)
            output_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            initialization_error = f"rank {rank}: {type(exc).__name__}: {exc}"

        if world_size > 1:
            errors = _all_gather_objects(initialization_error, world_size)
            failures = [error for error in errors if error is not None]
            initialization_error = " | ".join(failures) if failures else None
        if initialization_error is not None:
            raise RuntimeError(
                "feature-cache setup failed: " + initialization_error
            )
        assert adapter is not None

        results = []
        for split in args.splits:
            result = _prepare_split(
                split=split,
                data_dir=data_dir,
                output_root=output_root,
                adapter=adapter,
                layer_ids=layer_ids,
                max_seq_length=max_seq_length,
                output_dtype=output_dtype,
                output_dtype_name=args.feature_dtype,
                rank=rank,
                world_size=world_size,
                group_size=args.group_size,
                max_samples=args.max_samples,
                overwrite=args.overwrite,
            )
            results.append(result)
        if rank == 0:
            summary = {
                "status": "complete",
                "target_provenance": adapter.target_provenance(),
                "target_layer_ids": layer_ids,
                "max_seq_length": max_seq_length,
                "feature_dtype": args.feature_dtype,
                "splits": {
                    result["split"]: {
                        "records": result["records"],
                        "path": result["split"],
                    }
                    for result in results
                },
            }
            _atomic_json(output_root / "cache_summary.json", summary)
    finally:
        if initialized_here:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
