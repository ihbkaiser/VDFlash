from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

import torch
import torch.distributed as dist

from .config import DFlashTrainConfig
from .data import select_context_positions
from .distributed import initialize_distributed
from .model import build_target_layer_ids
from .real_data import sha256_file
from .target import Qwen25VLTargetAdapter, load_jsonl


def _response_token_ids(adapter: Qwen25VLTargetAdapter, messages: list[dict[str, Any]], text: str) -> list[int]:
    prompt_rendered = adapter.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_rendered = adapter.processor.apply_chat_template(
        messages
        + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_rendered.startswith(prompt_rendered):
        raise RuntimeError("Qwen chat template full conversation does not preserve the prompt prefix")
    suffix = full_rendered[len(prompt_rendered) :]
    return list(adapter.processor.tokenizer(suffix, add_special_tokens=False).input_ids)


def _drop_oldest_exchange(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if len(messages) <= 1:
        return None
    copied = copy.deepcopy(messages)
    copied.pop(0)
    while copied and copied[0].get("role") != "user":
        copied.pop(0)
    return copied or None


def _eos_token_ids(adapter: Qwen25VLTargetAdapter) -> set[int]:
    value = getattr(getattr(adapter.model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(adapter.processor.tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(token_id) for token_id in value}
    return {int(value)}


def _fit_prompt(
    adapter: Qwen25VLTargetAdapter,
    record: dict[str, Any],
    config: DFlashTrainConfig,
    *,
    preparation_device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    messages = copy.deepcopy(record["messages"])
    dropped_turns = 0
    while True:
        if preparation_device is None:
            prompt_inputs, _ = adapter.prepare_messages(messages)
        else:
            prompt_inputs, _ = adapter.prepare_messages(
                messages,
                device=preparation_device,
            )
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        if prompt_length <= config.max_seq_length - config.block_size:
            break
        next_messages = _drop_oldest_exchange(messages)
        if next_messages is None:
            raise ValueError(
                f"prompt has {prompt_length} tokens and cannot fit max_seq_length={config.max_seq_length}"
            )
        dropped_turns += len(messages) - len(next_messages)
        messages = next_messages
    return messages, prompt_inputs, prompt_length, dropped_turns


def _generate_target_response_ids(
    adapter: Qwen25VLTargetAdapter,
    prompt_inputs: dict[str, Any],
    *,
    prompt_length: int,
    config: DFlashTrainConfig,
) -> tuple[list[int], bool]:
    available = config.max_seq_length - prompt_length
    max_new_tokens = min(config.response_max_new_tokens, available)
    if max_new_tokens < config.block_size:
        raise ValueError(
            f"only {max_new_tokens} response tokens fit, fewer than block_size={config.block_size}"
        )
    with torch.inference_mode():
        generated = adapter.model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            temperature=None,
            use_cache=True,
        )
    response_ids = generated[0, prompt_length:].detach().cpu().tolist()
    eos_ids = _eos_token_ids(adapter)
    if config.teacher_require_eos and not eos_ids:
        raise ValueError("teacher_require_eos is enabled but the target defines no EOS token")
    contains_eos = any(int(token_id) in eos_ids for token_id in response_ids)
    reached_limit = len(response_ids) >= max_new_tokens and not contains_eos
    if config.teacher_require_eos and eos_ids and not contains_eos:
        raise ValueError(
            f"raw-greedy target response reached {max_new_tokens} tokens without EOS"
        )
    return [int(token_id) for token_id in response_ids], reached_limit


def _fit_clean_sequence(
    adapter: Qwen25VLTargetAdapter,
    record: dict[str, Any],
    config: DFlashTrainConfig,
) -> tuple[dict[str, Any], torch.Tensor, int, int, bool, int]:
    preparation_device = (
        torch.device("cpu")
        if config.stage == "text"
        and config.teacher_response_mode == "dataset"
        and config.teacher_batch_size > 1
        else None
    )
    messages, prompt_inputs, prompt_length, dropped_turns = _fit_prompt(
        adapter,
        record,
        config,
        preparation_device=preparation_device,
    )

    if config.teacher_response_mode == "target_generate":
        response_ids, response_truncated = _generate_target_response_ids(
            adapter,
            prompt_inputs,
            prompt_length=prompt_length,
            config=config,
        )
    else:
        target_text = record.get("target_text")
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("dataset response mode requires a non-empty target_text")
        response_ids = _response_token_ids(adapter, messages, target_text)
        available = config.max_seq_length - prompt_length
        response_truncated = len(response_ids) > available
        if response_truncated:
            # Qwen's rendered assistant suffix ends with <|im_end|> plus a newline.
            terminal = response_ids[-2:] if len(response_ids) >= 2 else response_ids[-1:]
            content_budget = available - len(terminal)
            if content_budget < 1:
                raise ValueError("no room remains for a real assistant response and its terminator")
            response_ids = response_ids[:content_budget] + terminal
    if len(response_ids) < config.block_size:
        raise ValueError(
            f"assistant response has {len(response_ids)} tokens, fewer than block_size={config.block_size}"
        )
    response_tensor = torch.tensor(
        response_ids,
        dtype=prompt_inputs["input_ids"].dtype,
        device=prompt_inputs["input_ids"].device,
    ).view(1, -1)
    full_ids = torch.cat([prompt_inputs["input_ids"], response_tensor], dim=1)
    inputs = dict(prompt_inputs)
    adapter._set_input_sequence(inputs, full_ids)
    inputs.pop("position_ids", None)
    inputs.pop("cache_position", None)
    if preparation_device is not None:
        # Pure text uses identical temporal/height/width positions. Avoid a
        # model API call so CPU preprocessing remains cheap and thread-safe.
        position_ids = torch.arange(
            full_ids.shape[1],
            device=full_ids.device,
        ).view(1, 1, -1).expand(3, 1, -1)
    else:
        position_ids = adapter._compute_position_ids(inputs)
    inputs["position_ids"] = position_ids
    return inputs, position_ids, prompt_length, int(full_ids.shape[1]), response_truncated, dropped_turns


@dataclass
class _PreparedTeacherExample:
    source_offset: int
    record: dict[str, Any]
    inputs: dict[str, Any]
    position_ids: torch.Tensor
    response_start: int
    response_end: int
    response_truncated: bool
    dropped_turns: int


def _pad_last_dimension(
    value: torch.Tensor,
    length: int,
    *,
    pad_value: int | float,
) -> torch.Tensor:
    missing = length - int(value.shape[-1])
    if missing < 0:
        raise ValueError("cannot pad a tensor to a shorter sequence length")
    if missing == 0:
        return value
    padding = value.new_full((*value.shape[:-1], missing), pad_value)
    return torch.cat([value, padding], dim=-1)


def _batch_text_inputs(
    examples: list[_PreparedTeacherExample],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    """Right-pad pure-text Qwen inputs without changing valid-token outputs."""

    if not examples:
        raise ValueError("cannot batch an empty teacher example list")
    if len(examples) == 1:
        return examples[0].inputs
    visual_keys = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "second_per_grid_ts",
    }
    if any(key in example.inputs for example in examples for key in visual_keys):
        raise ValueError("batched teacher caching currently supports pure-text records only")

    lengths = [example.response_end for example in examples]
    padded_length = max(lengths)
    keys = set(examples[0].inputs)
    if any(set(example.inputs) != keys for example in examples[1:]):
        raise ValueError("teacher examples in one batch must expose identical input keys")

    result: dict[str, Any] = {}
    for key in sorted(keys):
        values = [example.inputs[key] for example in examples]
        first = values[0]
        if not torch.is_tensor(first):
            if any(value != first for value in values[1:]):
                raise ValueError(f"non-tensor teacher input {key!r} differs within a batch")
            result[key] = first
            continue
        if not all(torch.is_tensor(value) for value in values):
            raise ValueError(f"teacher input {key!r} mixes tensor and non-tensor values")

        if key == "position_ids":
            if any(
                value.ndim != 3
                or value.shape[0] != 3
                or value.shape[1] != 1
                or value.shape[-1] != length
                for value, length in zip(values, lengths)
            ):
                raise ValueError("position_ids must have shape [3, 1, sequence_length]")
            result[key] = torch.cat(
                [
                    _pad_last_dimension(value, padded_length, pad_value=0)
                    for value in values
                ],
                dim=1,
            )
            continue

        if all(
            value.ndim >= 2
            and value.shape[0] == 1
            and value.shape[-1] == length
            for value, length in zip(values, lengths)
        ):
            pad_value = pad_token_id if key == "input_ids" else 0
            result[key] = torch.cat(
                [
                    _pad_last_dimension(value, padded_length, pad_value=pad_value)
                    for value in values
                ],
                dim=0,
            )
            continue

        if all(value.shape[1:] == first.shape[1:] for value in values):
            result[key] = torch.cat(values, dim=0)
            continue
        raise ValueError(f"cannot batch teacher input {key!r} with varying shapes")
    return result


def _iter_teacher_forwards(
    adapter: Qwen25VLTargetAdapter,
    examples: list[_PreparedTeacherExample],
    layer_ids: list[int],
    *,
    pad_token_id: int,
):
    """Run one batch, recursively backing off only when CUDA reports OOM."""

    host_inputs: dict[str, Any] | None = None
    batched_inputs: dict[str, Any] | None = None
    try:
        host_inputs = _batch_text_inputs(examples, pad_token_id=pad_token_id)
        batched_inputs = {
            key: value.to(adapter.device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in host_inputs.items()
        }
        selected = adapter.forward_selected_hidden(batched_inputs, layer_ids)
    except torch.OutOfMemoryError:
        del host_inputs, batched_inputs
        if adapter.device.type == "cuda":
            torch.cuda.empty_cache()
        if len(examples) == 1:
            raise
        midpoint = len(examples) // 2
        print(
            f"[teacher-oom] reducing batch {len(examples)} -> "
            f"{midpoint}+{len(examples) - midpoint}"
        )
        yield from _iter_teacher_forwards(
            adapter,
            examples[:midpoint],
            layer_ids,
            pad_token_id=pad_token_id,
        )
        yield from _iter_teacher_forwards(
            adapter,
            examples[midpoint:],
            layer_ids,
            pad_token_id=pad_token_id,
        )
        return
    yield examples, batched_inputs, selected


def _tensor_prefix(index: int) -> str:
    return f"sample_{index:06d}"


def _shard_filename(shard_index: int, rank: int | None) -> str:
    return (
        f"rank-{rank:05d}-shard-{shard_index:05d}.safetensors"
        if rank is not None
        else f"shard-{shard_index:05d}.safetensors"
    )


def _write_shard(
    directory: Path,
    shard_index: int,
    records: list[tuple[int, dict[str, torch.Tensor]]],
    *,
    rank: int | None = None,
) -> tuple[str, dict[int, str]]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for teacher feature shards") from exc
    filename = _shard_filename(shard_index, rank)
    tensors: dict[str, torch.Tensor] = {}
    prefixes: dict[int, str] = {}
    for sample_index, values in records:
        prefix = _tensor_prefix(sample_index)
        prefixes[sample_index] = prefix
        for name, value in values.items():
            tensors[f"{prefix}.{name}"] = value.detach().cpu().contiguous()
    save_file(tensors, str(directory / filename))
    return filename, prefixes


class _ShardWriter:
    """Bounded single-writer queue that overlaps safetensors I/O with inference."""

    def __init__(self, queue_depth: int):
        self.queue_depth = int(queue_depth)
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="teacher-cache-writer")
            if self.queue_depth > 0
            else None
        )
        self.futures: deque[Future[tuple[str, dict[int, str]]]] = deque()

    def submit(
        self,
        directory: Path,
        shard_index: int,
        records: list[tuple[int, dict[str, torch.Tensor]]],
        *,
        rank: int | None,
    ) -> str:
        filename = _shard_filename(shard_index, rank)
        if self.executor is None:
            _write_shard(directory, shard_index, records, rank=rank)
            return filename
        while len(self.futures) >= self.queue_depth:
            self.futures.popleft().result()
        self.futures.append(
            self.executor.submit(
                _write_shard,
                directory,
                shard_index,
                records,
                rank=rank,
            )
        )
        return filename

    def close(self) -> None:
        try:
            while self.futures:
                self.futures.popleft().result()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=False)


def _save_static_target_io(directory: Path, adapter: Qwen25VLTargetAdapter) -> tuple[str, bool]:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required for cached target token I/O") from exc
    embedding = adapter.input_embeddings.weight.detach()
    lm_head = adapter.lm_head.weight.detach()
    tied = embedding.data_ptr() == lm_head.data_ptr() and embedding.shape == lm_head.shape
    tensors = {"token_embedding": embedding.cpu().contiguous()}
    if not tied:
        tensors["lm_head"] = lm_head.cpu().contiguous()
    filename = "target_token_io.safetensors"
    save_file(tensors, str(directory / filename))
    return filename, tied


def cache_teacher_features(config: DFlashTrainConfig) -> Path:
    distributed = initialize_distributed(config.device, config.distributed_backend)
    device = distributed.device
    manifest_path = Path(config.prepared_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest_metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    if not manifest_metadata_path.is_file():
        raise FileNotFoundError(manifest_metadata_path)
    records = load_jsonl(manifest_path)
    if not records:
        raise ValueError("prepared real-data manifest is empty")

    cache_dir = Path(config.teacher_cache_dir).expanduser().resolve()
    temporary = cache_dir.with_name(cache_dir.name + ".tmp")
    if distributed.is_main:
        if cache_dir.exists():
            if not config.overwrite:
                raise FileExistsError(
                    f"teacher cache already exists: {cache_dir}; pass --overwrite"
                )
            if cache_dir == cache_dir.parent:
                raise RuntimeError("refusing to overwrite a filesystem root")
            shutil.rmtree(cache_dir)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
    distributed.barrier()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        config.mixed_precision
    ]
    adapter = Qwen25VLTargetAdapter.from_pretrained(config, device=device, dtype=dtype)
    adapter.freeze()
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("target model must be completely frozen during teacher caching")
    layer_ids = list(
        config.selected_target_layers
        or build_target_layer_ids(
            int(adapter.text_config.num_hidden_layers),
            config.num_target_features,
        )
    )
    if max(layer_ids) >= int(adapter.text_config.num_hidden_layers):
        raise ValueError("selected_target_layers contains an index outside the target model")
    if distributed.is_main:
        print(
            f"[teacher-setup] model={config.target_model} world_size={distributed.world_size} "
            f"dtype={config.mixed_precision} frozen=True layers={layer_ids} "
            f"response_mode={config.teacher_response_mode} batch={config.teacher_batch_size} "
            f"bucket={config.teacher_length_bucket_size} "
            f"preprocess_workers={config.teacher_preprocess_workers} "
            f"shard={config.cache_shard_size} "
            f"write_queue={config.teacher_write_queue_depth}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    image_ids, video_ids = adapter.visual_token_ids
    index_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, torch.Tensor]]] = []
    entry_by_offset: dict[int, dict[str, Any]] = {}
    shard_index = 0
    writer = _ShardWriter(config.teacher_write_queue_depth)
    preprocess_executor = (
        ThreadPoolExecutor(
            max_workers=config.teacher_preprocess_workers,
            thread_name_prefix="teacher-preprocess",
        )
        if config.teacher_preprocess_workers > 1
        else None
    )
    shard_rank = distributed.rank if distributed.enabled else None
    tokenizer = adapter.processor.tokenizer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else 0
    pad_token_id = int(pad_token_id or 0)
    started_at = time.perf_counter()
    processed_tokens = 0
    last_log_group = -1

    def flush_pending() -> None:
        nonlocal pending, shard_index
        if not pending:
            return
        records_to_write = pending
        pending = []
        filename = writer.submit(
            temporary,
            shard_index,
            records_to_write,
            rank=shard_rank,
        )
        for source_offset, _ in records_to_write:
            entry_by_offset[source_offset]["shard"] = filename
        shard_index += 1

    def prepare_offset(
        source_offset: int,
    ) -> tuple[_PreparedTeacherExample | None, dict[str, Any] | None]:
        record = records[source_offset]
        try:
            (
                inputs,
                position_ids,
                response_start,
                response_end,
                response_truncated,
                dropped_turns,
            ) = _fit_clean_sequence(adapter, record, config)
        except ValueError as exc:
            return None, {
                "source_offset": source_offset,
                "manifest_id": str(record.get("id", source_offset)),
                "reason": str(exc),
            }
        return (
            _PreparedTeacherExample(
                source_offset=source_offset,
                record=record,
                inputs=inputs,
                position_ids=position_ids,
                response_start=response_start,
                response_end=response_end,
                response_truncated=response_truncated,
                dropped_turns=dropped_turns,
            ),
            None,
        )

    local_offsets = list(range(distributed.rank, len(records), distributed.world_size))
    try:
        for bucket_start in range(0, len(local_offsets), config.teacher_length_bucket_size):
            bucket_offsets = local_offsets[
                bucket_start : bucket_start + config.teacher_length_bucket_size
            ]
            prepared: list[_PreparedTeacherExample] = []
            outcomes = (
                preprocess_executor.map(prepare_offset, bucket_offsets)
                if preprocess_executor is not None
                else map(prepare_offset, bucket_offsets)
            )
            for example, skipped_entry in outcomes:
                if skipped_entry is not None:
                    skipped.append(skipped_entry)
                    print(
                        f"[teacher-skip rank={distributed.rank}] "
                        f"id={skipped_entry['manifest_id']} "
                        f"reason={skipped_entry['reason']}"
                    )
                elif example is not None:
                    prepared.append(example)

            prepared.sort(key=lambda example: (example.response_end, example.source_offset))
            for batch_start in range(0, len(prepared), config.teacher_batch_size):
                requested_batch = prepared[
                    batch_start : batch_start + config.teacher_batch_size
                ]
                for actual_batch, batched_inputs, selected in _iter_teacher_forwards(
                    adapter,
                    requested_batch,
                    layer_ids,
                    pad_token_id=pad_token_id,
                ):
                    for batch_index, example in enumerate(actual_batch):
                        context_original = select_context_positions(
                            example.inputs["input_ids"][0],
                            context_mode=config.context_mode,
                            image_token_ids=image_ids,
                            video_token_ids=video_ids,
                        )
                        sample_hidden = selected[batch_index, : example.response_end]
                        if context_original.numel() != example.response_end:
                            sample_hidden = sample_hidden[
                                context_original.to(sample_hidden.device)
                            ]
                        cached_tensors = {
                            "input_ids": example.inputs["input_ids"][0]
                            .detach()
                            .to(device="cpu", dtype=torch.int32)
                            .contiguous(),
                            "position_ids": example.position_ids[:, 0]
                            .detach()
                            .to(device="cpu", dtype=torch.int32)
                            .contiguous(),
                            "context_hidden": sample_hidden
                            .detach()
                            .to(device="cpu", dtype=dtype)
                            .contiguous(),
                            "context_original_positions": context_original
                            .detach()
                            .to(device="cpu", dtype=torch.int32)
                            .contiguous(),
                        }
                        pending.append((example.source_offset, cached_tensors))
                        entry = {
                            "source_offset": example.source_offset,
                            "manifest_id": str(example.record["id"]),
                            "source": example.record.get("source", {}),
                            "response_start": example.response_start,
                            "response_end": example.response_end,
                            "sequence_length": example.response_end,
                            "context_length": int(context_original.numel()),
                            "response_truncated": example.response_truncated,
                            "teacher_response_mode": config.teacher_response_mode,
                            "dropped_prompt_turns": example.dropped_turns,
                            "shard": None,
                            "tensor_prefix": _tensor_prefix(example.source_offset),
                        }
                        index_entries.append(entry)
                        entry_by_offset[example.source_offset] = entry
                        processed_tokens += example.response_end
                        if len(pending) >= config.cache_shard_size:
                            flush_pending()

                    local_done = len(index_entries) + len(skipped)
                    log_group = local_done // config.cache_log_every
                    if local_done == len(actual_batch) or log_group > last_log_group:
                        elapsed = max(time.perf_counter() - started_at, 1e-6)
                        print(
                            f"[teacher rank={distributed.rank} "
                            f"done={local_done}/{len(local_offsets)} "
                            f"batch={len(actual_batch)} padded={selected.shape[1]}] "
                            f"samples_per_s={len(index_entries) / elapsed:.2f} "
                            f"tokens_per_s={processed_tokens / elapsed:.0f}"
                        )
                        last_log_group = log_group
                    del selected, batched_inputs
            del prepared
        flush_pending()
    finally:
        if preprocess_executor is not None:
            preprocess_executor.shutdown(wait=True, cancel_futures=False)
        writer.close()

    local_index_path = temporary / f"index.rank-{distributed.rank:05d}.json"
    local_skipped_path = temporary / f"skipped.rank-{distributed.rank:05d}.json"
    local_index_path.write_text(json.dumps(index_entries, ensure_ascii=False, sort_keys=True))
    local_skipped_path.write_text(json.dumps(skipped, ensure_ascii=False, sort_keys=True))
    distributed.barrier()

    merged_entries: list[dict[str, Any]] = []
    merged_skipped: list[dict[str, Any]] = []
    if distributed.is_main:
        for rank in range(distributed.world_size):
            rank_index = temporary / f"index.rank-{rank:05d}.json"
            rank_skipped = temporary / f"skipped.rank-{rank:05d}.json"
            merged_entries.extend(json.loads(rank_index.read_text()))
            merged_skipped.extend(json.loads(rank_skipped.read_text()))
            rank_index.unlink()
            rank_skipped.unlink()
        merged_entries.sort(key=lambda entry: int(entry["source_offset"]))
        merged_skipped.sort(key=lambda entry: int(entry["source_offset"]))
        for cache_index, entry in enumerate(merged_entries):
            entry["cache_index"] = cache_index
            entry.pop("source_offset", None)
        for entry in merged_skipped:
            entry.pop("source_offset", None)
        if not merged_entries:
            raise RuntimeError("no real dataset records could be cached")

        static_filename, tied_token_io = _save_static_target_io(temporary, adapter)
        manifest_sha = sha256_file(manifest_path)
        manifest_metadata = json.loads(manifest_metadata_path.read_text())
        target_provenance = adapter.target_provenance()
        cache_metadata = {
            "format": "video-dflash-teacher-cache-v2",
            "stage": config.stage,
            "target_model": config.target_model,
            "target_revision": config.target_revision,
            "target_provenance": target_provenance,
            "target_commit": target_provenance.get("target_commit"),
            "target_text_config": adapter.text_config.to_dict(),
            "target_hidden_size": adapter.hidden_size,
            "target_vocab_size": adapter.vocab_size,
            "target_layer_ids": layer_ids,
            "num_target_features": len(layer_ids),
            "teacher_response_mode": config.teacher_response_mode,
            "teacher_generation": (
                {
                    "do_sample": False,
                    "temperature": 0.0,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": config.response_max_new_tokens,
                    "require_eos": config.teacher_require_eos,
                    "eos_token_ids": sorted(_eos_token_ids(adapter)),
                }
                if config.teacher_response_mode == "target_generate"
                else None
            ),
            "mask_token_id": adapter.resolve_mask_token_id(),
            "tokenizer_fingerprint": adapter.tokenizer_fingerprint(),
            "processor_fingerprint": adapter.processor_fingerprint(),
            "context_mode": config.context_mode,
            "dtype": config.mixed_precision,
            "max_seq_length": config.max_seq_length,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": manifest_sha,
            "source_manifest_metadata": manifest_metadata,
            "cache_world_size": distributed.world_size,
            "cache_execution": {
                "teacher_batch_size": config.teacher_batch_size,
                "teacher_length_bucket_size": config.teacher_length_bucket_size,
                "teacher_preprocess_workers": config.teacher_preprocess_workers,
                "cache_shard_size": config.cache_shard_size,
                "teacher_write_queue_depth": config.teacher_write_queue_depth,
                "selected_layer_hooks": True,
                "lm_head_skipped": True,
                "decoder_early_stop_layer": max(layer_ids),
            },
            "cached_count": len(merged_entries),
            "cached_sample_ids": [entry["manifest_id"] for entry in merged_entries],
            "skipped": merged_skipped,
            "static_token_io": static_filename,
            "tied_token_io": tied_token_io,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(cache_metadata, indent=2, ensure_ascii=False, sort_keys=True)
        )
        (temporary / "index.json").write_text(
            json.dumps(merged_entries, indent=2, ensure_ascii=False, sort_keys=True)
        )

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        torch.cuda.empty_cache()
    else:
        peak = 0.0
    if distributed.enabled:
        peak_tensor = torch.tensor(peak, device=device)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        peak = float(peak_tensor.item())
    distributed.barrier()
    if distributed.is_main:
        temporary.replace(cache_dir)
        print(
            f"[teacher-cache] cached={len(merged_entries)} skipped={len(merged_skipped)} "
            f"layers={layer_ids} world_size={distributed.world_size} "
            f"peak_vram={peak:.2f}GiB path={cache_dir}"
        )
    distributed.barrier()
    del adapter
    distributed.close()
    return cache_dir
