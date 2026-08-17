#!/usr/bin/env python3
"""Capture Qwen2.5-VL LLaVA caption features for SpecForge DFlash."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

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
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dist-timeout", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
            max_running_requests=1,
            max_total_tokens=args.max_length,
        )
        capture.set_capture_layers(layer_ids, capture_method="dflash")

        processed = 0
        skipped = 0
        local_indices = range(rank, len(records), world_size)
        progress = tqdm(
            local_indices,
            total=len(local_indices),
            desc=f"Capture LLaVA (rank {rank} shard)",
            unit="sample",
            dynamic_ncols=True,
            mininterval=0.5,
            disable=rank != 0,
        )
        for index in progress:
            output_file = _output_path(output_root, index, args.compress)
            if output_file.exists() and not args.overwrite:
                processed += 1
                continue
            record = records[index]
            try:
                prepared = prepare_training_example(
                    processor,
                    target_config,
                    record,
                    image_root=args.image_root,
                    max_length=args.max_length,
                    image_min_pixels=args.image_min_pixels,
                    image_max_pixels=args.image_max_pixels,
                )
                device = torch.device("cuda", torch.cuda.current_device())
                input_ids = prepared["input_ids"].to(device)
                attention_mask = prepared["attention_mask"].to(device)
                loss_mask = prepared["loss_mask"].to(device)
                position_ids = prepared["position_ids"].to(device)
                media = {
                    key: value.to(device)
                    for key, value in prepared["multimodal_inputs"].items()
                    if torch.is_tensor(value)
                }
                captured = capture.capture(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    position_ids=position_ids,
                    multimodal_inputs=[media],
                )
                hidden = captured.hidden_states[0]
                if hidden.ndim != 2:
                    raise ValueError(f"captured hidden states have shape {tuple(hidden.shape)}")
                _save_record(
                    output_file,
                    {
                        "input_ids": input_ids[0].cpu().to(torch.int32),
                        "loss_mask": loss_mask[0].cpu().to(torch.float32),
                        "hidden_states": hidden.cpu().contiguous(),
                        "position_ids": position_ids[:, 0].cpu().to(torch.int32),
                    },
                    args.compress,
                )
                processed += 1
            except (FileNotFoundError, ValueError) as exc:
                skipped += 1
                print(f"[rank={rank}] skip id={record.get('id', index)}: {exc}")
            if rank == 0:
                progress.set_postfix(
                    processed=processed,
                    skipped=skipped,
                    refresh=False,
                )
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
