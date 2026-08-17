#!/usr/bin/env python3
"""Capture Qwen2.5-VL LLaVA caption features for SpecForge DFlash."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import gzip
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.offline_capture import load_offline_capture
from specforge.qwen25vl import prepare_training_example


def _load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            records.append(record)
    return records


def _save_record(path: Path, record: dict[str, torch.Tensor], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        if compress:
            with gzip.open(temporary, "wb", compresslevel=6) as handle:
                torch.save(record, handle)
        else:
            torch.save(record, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    parser.add_argument(
        "--sglang-mem-fraction-static",
        type=float,
        default=0.4,
        help="GPU memory fraction reserved for SGLang weights and KV cache",
    )
    return parser.parse_args()


def _collate_prepared(
    prepared_batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[int]]:
    """Right-pad variable-length multimodal samples for one SGLang batch."""

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
    multimodal_inputs: list[dict[str, torch.Tensor]] = []

    for row, (prepared, length) in enumerate(zip(prepared_batch, lengths)):
        input_ids[row, :length] = prepared["input_ids"][0]
        attention_mask[row, :length] = prepared["attention_mask"][0]
        loss_mask[row, :length] = prepared["loss_mask"][0]
        positions = prepared["position_ids"]
        if positions.ndim == 3:
            positions = positions[:, 0]
        position_ids[:, row, :length] = positions
        multimodal_inputs.append(prepared["multimodal_inputs"])

    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        },
        multimodal_inputs,
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
    """Prepare records concurrently with bounded CPU/RAM prefetch."""

    worker_state = threading.local()

    def get_processor():
        if num_workers <= 1:
            return processor
        if not hasattr(worker_state, "processor"):
            worker_state.processor = processor_factory()
        return worker_state.processor

    def prepare(index: int):
        record = records[index]
        output_file = _output_path(output_root, index, compress)
        if output_file.exists() and not overwrite:
            return index, record, None, None, True
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
        except (FileNotFoundError, ValueError) as exc:
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


def main() -> int:
    args = parse_args()
    if not 0 < args.sglang_mem_fraction_static <= 1:
        raise ValueError("--sglang-mem-fraction-static must be in (0, 1]")
    for name in (
        "batch_size",
        "preprocess_queue_size",
        "num_io_threads",
        "io_queue_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_preprocess_workers < 0:
        raise ValueError("--num-preprocess-workers must be non-negative")
    records = _load_jsonl(args.manifest)
    if args.expected_records and len(records) != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} manifest records, found {len(records)}"
        )
    with open(args.draft_model_config, encoding="utf-8") as handle:
        draft_config = json.load(handle)
    layer_ids = list(draft_config["dflash_config"]["target_layer_ids"])

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    dp_group = get_dp_group()
    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)
    output_root = Path(args.output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoConfig, AutoProcessor

        target_config = AutoConfig.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
        )
        capture = load_offline_capture(
            args.target_model_path,
            trust_remote_code=args.trust_remote_code,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            chunked_prefill_size=-1,
            tp_size=args.tp_size,
            max_running_requests=args.batch_size,
            max_total_tokens=args.batch_size * args.max_length,
            mem_fraction_static=args.sglang_mem_fraction_static,
        )
        capture.set_capture_layers(layer_ids, capture_method="dflash")

        processed = 0
        skipped = 0
        local_indices = range(rank, len(records), world_size)
        progress = tqdm(
            total=len(local_indices),
            desc=f"Capture LLaVA (rank {rank} shard)",
            unit="sample",
            dynamic_ncols=True,
            mininterval=0.5,
            disable=rank != 0,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = processor.tokenizer.eos_token_id or 0
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
                prepared_values = [item[2] for item in prepared_batch]
                cpu_batch, media_batch, lengths = _collate_prepared(
                    prepared_values,
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
                        if torch.is_tensor(value)
                    }
                    for media in media_batch
                ]
                captured = capture.capture(
                    **gpu_batch,
                    multimodal_inputs=gpu_media,
                )
                for row, ((index, _, prepared), length) in enumerate(
                    zip(prepared_batch, lengths)
                ):
                    hidden = captured.hidden_states[row, :length]
                    if hidden.ndim != 2:
                        raise ValueError(
                            f"captured hidden states have shape {tuple(hidden.shape)}"
                        )
                    positions = prepared["position_ids"]
                    if positions.ndim == 3:
                        positions = positions[:, 0]
                    submit_save(
                        _output_path(output_root, index, args.compress),
                        {
                            "input_ids": prepared["input_ids"][0].to(torch.int32),
                            "loss_mask": prepared["loss_mask"][0].to(torch.float32),
                            "hidden_states": hidden.cpu().contiguous(),
                            "position_ids": positions.to(torch.int32),
                        },
                    )
                processed += len(prepared_batch)
                progress.update(len(prepared_batch))
                prepared_batch.clear()

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
                    progress.update(1)
                elif error is not None:
                    skipped += 1
                    progress.update(1)
                    print(f"[rank={rank}] skip id={record.get('id', index)}: {error}")
                else:
                    prepared_batch.append((index, record, prepared))
                    if len(prepared_batch) >= args.batch_size:
                        capture_batch()
                if rank == 0:
                    progress.set_postfix(
                        processed=processed,
                        skipped=skipped,
                        pending_io=len(pending_saves),
                        refresh=False,
                    )
            capture_batch()
            while pending_saves:
                pending_saves.popleft().result()
        progress.close()
        counts = torch.tensor([processed, skipped], device="cuda", dtype=torch.long)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dp_group)
        if rank == 0:
            print(
                f"captured={int(counts[0])} skipped={int(counts[1])} "
                f"records={len(records)} output={output_root}"
            )
        dist.barrier(group=dp_group)
        if int(counts[1]) != 0:
            raise RuntimeError(
                f"capture produced {int(counts[1])} skipped records; "
                "the strict 68k pipeline refuses a partial feature set"
            )
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
