#!/usr/bin/env python3
"""Capture standalone EAGLE3 Qwen2.5-VL image-captioning features."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import gzip
import json
import os
import pickle
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
from tqdm.auto import tqdm


_REQUIRED_FEATURE_KEYS = frozenset(
    {
        "input_ids",
        "loss_mask",
        "aux_hidden_state",
        "hidden_state",
        "position_ids",
    }
)


def _load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"empty manifest line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(record)
    return records


def _feature_record_is_complete(path: Path) -> bool:
    """Validate one atomically written Qwen2.5-VL feature record."""

    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                payload = torch.load(handle, map_location="cpu", weights_only=True)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ):
        return False
    if not isinstance(payload, dict) or not _REQUIRED_FEATURE_KEYS.issubset(payload):
        return False
    tensors = {key: payload[key] for key in _REQUIRED_FEATURE_KEYS}
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        return False
    if any(
        not torch.isfinite(tensors[key]).all()
        for key in ("loss_mask", "aux_hidden_state", "hidden_state")
    ):
        return False
    length = tensors["input_ids"].numel()
    if tensors["input_ids"].ndim != 1 or tensors["loss_mask"].ndim != 1:
        return False
    if tensors["loss_mask"].numel() != length:
        return False
    if any(
        tensors[key].ndim != 2 or tensors[key].shape[0] != length
        for key in ("aux_hidden_state", "hidden_state")
    ):
        return False
    position_ids = tensors["position_ids"]
    return (
        position_ids.ndim == 2
        and position_ids.shape[0] == 3
        and position_ids.shape[1] == length
    )


def _validate_capture_shapes(
    *,
    aux_hidden_states: torch.Tensor,
    last_hidden_states: torch.Tensor,
    target_hidden_size: int,
    capture_layers: list[int],
) -> None:
    """Enforce the EAGLE3 three-auxiliary-state storage contract."""

    if len(capture_layers) != 3 or len(set(capture_layers)) != 3:
        raise ValueError(
            "EAGLE3 capture must select exactly three auxiliary layers, got "
            f"{capture_layers!r}"
        )
    if aux_hidden_states.ndim != 2 or last_hidden_states.ndim != 2:
        raise ValueError(
            "EAGLE3 capture states must have shape [tokens, hidden_size]"
        )
    if aux_hidden_states.shape[0] != last_hidden_states.shape[0]:
        raise ValueError("auxiliary and final captures have different token counts")
    if aux_hidden_states.shape[-1] != 3 * target_hidden_size:
        raise ValueError(
            "EAGLE3 capture must concatenate three auxiliary hidden states: "
            f"expected width {3 * target_hidden_size}, "
            f"got {aux_hidden_states.shape[-1]}"
        )
    if last_hidden_states.shape[-1] != target_hidden_size:
        raise ValueError(
            "EAGLE3 final hidden state has unexpected width: "
            f"expected {target_hidden_size}, got {last_hidden_states.shape[-1]}"
        )
    if not torch.isfinite(aux_hidden_states).all() or not torch.isfinite(
        last_hidden_states
    ).all():
        raise FloatingPointError("EAGLE3 capture produced non-finite hidden states")


def _save_record(path: Path, payload: dict[str, torch.Tensor], compress: bool) -> None:
    """Atomically save one EAGLE3 feature record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            if compress:
                with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=6) as output:
                    torch.save(payload, output)
            else:
                torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _output_path(root: Path, index: int, compress: bool) -> Path:
    group_start = (index // 2000) * 2000
    suffix = ".ckpt.gz" if compress else ".ckpt"
    return root / f"rows_{group_start}-{group_start + 2000}" / f"data_{index}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--expected-records", type=int, default=68000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-preprocess-workers", type=int, default=4)
    parser.add_argument("--preprocess-queue-size", type=int, default=32)
    parser.add_argument("--num-io-threads", type=int, default=4)
    parser.add_argument("--io-queue-size", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dist-timeout", type=int, default=2000)
    parser.add_argument("--sglang-attention-backend", default="flashinfer")
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
    parser.add_argument("--sglang-context-length", type=int, default=None)
    return parser.parse_args()


def _collate_prepared(
    prepared_batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[int]]:
    """Right-pad variable-length prepared examples for one capture batch."""

    if not prepared_batch:
        raise ValueError("prepared_batch must not be empty")
    lengths = [int(item["input_ids"].shape[-1]) for item in prepared_batch]
    max_length = max(lengths)
    batch_size = len(prepared_batch)
    input_template = prepared_batch[0]["input_ids"]
    input_ids = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=input_template.dtype,
    )
    attention_mask = torch.zeros_like(input_ids)
    loss_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)
    position_ids = torch.zeros(
        (3, batch_size, max_length),
        dtype=prepared_batch[0]["position_ids"].dtype,
    )
    media_batch: list[dict[str, torch.Tensor]] = []
    for row, (prepared, length) in enumerate(zip(prepared_batch, lengths)):
        input_ids[row, :length] = prepared["input_ids"][0]
        attention_mask[row, :length] = prepared["attention_mask"][0]
        loss_mask[row, :length] = prepared["loss_mask"][0]
        positions = prepared["position_ids"]
        if positions.ndim == 3:
            positions = positions[:, 0]
        if positions.ndim != 2 or positions.shape[0] != 3:
            raise ValueError("prepared position_ids must have shape [3, sequence]")
        position_ids[:, row, :length] = positions
        media_batch.append(prepared["multimodal_inputs"])
    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        },
        media_batch,
        lengths,
    )


def _iter_prepared_records(
    *,
    indices: range,
    records: list[dict[str, Any]],
    processor: Any,
    processor_factory: Callable[[], Any],
    target_config: Any,
    output_root: Path,
    compress: bool,
    overwrite: bool,
    image_root: str,
    max_length: int,
    image_min_pixels: int,
    image_max_pixels: int,
    num_workers: int,
    queue_size: int,
) -> Iterator[tuple[int, dict[str, Any], dict[str, Any] | None, Exception | None, bool]]:
    worker_state = threading.local()

    def get_processor():
        if num_workers <= 1:
            return processor
        if not hasattr(worker_state, "processor"):
            worker_state.processor = processor_factory()
        return worker_state.processor

    def prepare(index: int):
        from specforge.qwen25vl import prepare_training_example

        record = records[index]
        output_file = _output_path(output_root, index, compress)
        alternate = output_file.with_suffix("") if compress else output_file.with_suffix(".ckpt.gz")
        existing = output_file if output_file.exists() else alternate
        if existing.exists():
            if _feature_record_is_complete(existing):
                return index, record, None, None, True
            if not overwrite:
                raise RuntimeError(
                    f"existing feature record is incomplete or corrupt: {existing}"
                )
        try:
            prepared = prepare_training_example(
                get_processor(),
                target_config,
                record,
                image_root=image_root,
                max_length=max_length,
                image_min_pixels=image_min_pixels,
                image_max_pixels=image_max_pixels,
            )
            return index, record, prepared, None, False
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return index, record, None, exc, False

    if num_workers <= 1:
        for index in indices:
            yield prepare(index)
        return
    index_iterator = iter(indices)
    pending: deque[Future] = deque()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        while len(pending) < queue_size:
            try:
                pending.append(executor.submit(prepare, next(index_iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(prepare, next(index_iterator)))
            except StopIteration:
                pass


def _sglang_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "attention_backend": args.sglang_attention_backend,
        "mem_fraction_static": args.sglang_mem_fraction_static,
        "max_running_requests": args.batch_size,
        "max_total_tokens": args.batch_size * args.max_length,
    }
    if args.sglang_context_length is not None:
        values["context_length"] = args.sglang_context_length
    return values


def main() -> int:
    args = parse_args()
    if args.max_length < 2:
        raise ValueError("--max-length must be at least two")
    if args.expected_records is not None and args.expected_records < 0:
        raise ValueError("--expected-records must be non-negative")
    if not 0 < args.sglang_mem_fraction_static <= 1:
        raise ValueError("--sglang-mem-fraction-static must be in (0, 1]")
    if args.batch_size < 1 or args.tp_size < 1:
        raise ValueError("--batch-size and --tp-size must be positive")
    if args.num_preprocess_workers < 0 or args.preprocess_queue_size < 1:
        raise ValueError("preprocessing worker/queue settings are invalid")
    if args.num_io_threads < 1 or args.io_queue_size < 1:
        raise ValueError("I/O worker/queue settings are invalid")

    records = _load_jsonl(args.manifest)
    if args.expected_records is not None and len(records) != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} manifest records, found {len(records)}"
        )

    from transformers import AutoConfig, AutoProcessor

    from specforge.application import resolve_offline_capture
    from specforge.config import Config
    from specforge.distributed import (
        destroy_distributed,
        get_dp_group,
        init_distributed,
    )
    from specforge.offline_capture import load_offline_capture

    target_config = AutoConfig.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    with Path(args.draft_model_config).open(encoding="utf-8") as handle:
        draft_config = json.load(handle)
    cfg = Config(
        model={
            "target_model_path": args.target_model_path,
            "draft_model_config": args.draft_model_config,
            "input_modality": "qwen2_5_vl",
            "trust_remote_code": args.trust_remote_code,
        },
        data={
            "hidden_states_path": str(Path(args.output_path).resolve()),
            "max_length": args.max_length,
        },
        training={"strategy": "eagle3"},
    )
    resolved = resolve_offline_capture(cfg, target_config=target_config)
    if resolved.capture_method != "eagle3":
        raise ValueError(f"unexpected capture method {resolved.capture_method!r}")
    if draft_config.get("architectures") != ["LlamaForCausalLMEagle3"]:
        raise ValueError("--draft-model-config must be an EAGLE3 draft config")
    target_text_config = getattr(target_config, "text_config", target_config)
    target_hidden_size = int(target_text_config.hidden_size)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    dp_group = get_dp_group()
    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)
    output_root = Path(args.output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processor = None
    capture = None
    try:
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )
        capture = load_offline_capture(
            args.target_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            chunked_prefill_size=-1,
            tp_size=args.tp_size,
            **_sglang_kwargs(args),
        )
        capture.set_capture_layers(
            list(resolved.capture_layers),
            capture_method=resolved.capture_method,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Qwen tokenizer has no pad or EOS token")

        processed = 0
        skipped = 0
        local_indices = range(rank, len(records), world_size)
        progress = tqdm(
            total=len(local_indices),
            desc=f"EAGLE3 Qwen2.5-VL capture (rank {rank})",
            unit="sample",
            disable=rank != 0,
        )
        pending_saves: deque[Future] = deque()
        prepared_batch: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=args.num_io_threads) as io_executor:
            def submit_save(path: Path, payload: dict[str, torch.Tensor]) -> None:
                while len(pending_saves) >= args.io_queue_size:
                    pending_saves.popleft().result()
                pending_saves.append(
                    io_executor.submit(_save_record, path, payload, args.compress)
                )

            def capture_batch() -> None:
                nonlocal processed
                if not prepared_batch:
                    return
                values = [item[2] for item in prepared_batch]
                cpu_batch, media_batch, lengths = _collate_prepared(
                    values,
                    pad_token_id=int(pad_token_id),
                )
                gpu_batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in cpu_batch.items()
                }
                gpu_media = [
                    {
                        key: value.to(device, non_blocking=True)
                        for key, value in media.items()
                        if isinstance(value, torch.Tensor)
                    }
                    for media in media_batch
                ]
                captured = capture.capture(
                    **gpu_batch,
                    multimodal_inputs=gpu_media,
                )
                _validate_capture_shapes(
                    aux_hidden_states=captured.hidden_states,
                    last_hidden_states=captured.last_hidden_states,
                    target_hidden_size=target_hidden_size,
                    capture_layers=list(resolved.capture_layers),
                )
                for row, ((index, _, prepared), length) in enumerate(
                    zip(prepared_batch, lengths)
                ):
                    positions = prepared["position_ids"]
                    if positions.ndim == 3:
                        positions = positions[:, 0]
                    payload = {
                        "input_ids": prepared["input_ids"][0].to(torch.int32).cpu(),
                        "loss_mask": prepared["loss_mask"][0].to(torch.float32).cpu(),
                        "aux_hidden_state": captured.hidden_states[row, :length]
                        .detach()
                        .to(torch.bfloat16)
                        .cpu()
                        .contiguous(),
                        "hidden_state": captured.last_hidden_states[row, :length]
                        .detach()
                        .to(torch.bfloat16)
                        .cpu()
                        .contiguous(),
                        "position_ids": positions[:3, :length].to(torch.int32).cpu(),
                    }
                    submit_save(_output_path(output_root, index, args.compress), payload)
                processed += len(prepared_batch)
                prepared_batch.clear()
                del gpu_batch, gpu_media, captured

            for index, record, prepared, error, exists in _iter_prepared_records(
                indices=local_indices,
                records=records,
                processor=processor,
                processor_factory=lambda: AutoProcessor.from_pretrained(
                    args.target_model_path,
                    trust_remote_code=args.trust_remote_code,
                ),
                target_config=target_config,
                output_root=output_root,
                compress=args.compress,
                overwrite=args.overwrite,
                image_root=args.image_root,
                max_length=args.max_length,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
                num_workers=args.num_preprocess_workers,
                queue_size=args.preprocess_queue_size,
            ):
                if exists:
                    processed += 1
                elif error is not None:
                    skipped += 1
                    print(f"[rank={rank}] skip id={record.get('id', index)}: {error}")
                else:
                    prepared_batch.append((index, record, prepared))
                    if len(prepared_batch) >= args.batch_size:
                        capture_batch()
                progress.update(1)
            capture_batch()
            while pending_saves:
                pending_saves.popleft().result()
        progress.close()
        counts = torch.tensor([processed, skipped], device=device, dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dp_group)
        if rank == 0:
            print(
                f"captured={int(counts[0])} skipped={int(counts[1])} "
                f"records={len(records)} output={output_root}"
            )
        dist.barrier(group=dp_group)
        if int(counts[1]) != 0:
            raise RuntimeError(
                f"capture produced {int(counts[1])} skipped records; refusing partial features"
            )
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
