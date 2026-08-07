from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
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
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    messages = copy.deepcopy(record["messages"])
    dropped_turns = 0
    while True:
        prompt_inputs, _ = adapter.prepare_messages(messages)
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
    messages, prompt_inputs, prompt_length, dropped_turns = _fit_prompt(
        adapter, record, config
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
        device=adapter.device,
    ).view(1, -1)
    full_ids = torch.cat([prompt_inputs["input_ids"], response_tensor], dim=1)
    inputs = dict(prompt_inputs)
    adapter._set_input_sequence(inputs, full_ids)
    inputs.pop("position_ids", None)
    inputs.pop("cache_position", None)
    position_ids = adapter._compute_position_ids(inputs)
    inputs["position_ids"] = position_ids
    return inputs, position_ids, prompt_length, int(full_ids.shape[1]), response_truncated, dropped_turns


def _tensor_prefix(index: int) -> str:
    return f"sample_{index:06d}"


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
    filename = (
        f"rank-{rank:05d}-shard-{shard_index:05d}.safetensors"
        if rank is not None
        else f"shard-{shard_index:05d}.safetensors"
    )
    tensors: dict[str, torch.Tensor] = {}
    prefixes: dict[int, str] = {}
    for sample_index, values in records:
        prefix = _tensor_prefix(sample_index)
        prefixes[sample_index] = prefix
        for name, value in values.items():
            tensors[f"{prefix}.{name}"] = value.detach().cpu().contiguous()
    save_file(tensors, str(directory / filename))
    return filename, prefixes


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
            f"response_mode={config.teacher_response_mode}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    image_ids, video_ids = adapter.visual_token_ids
    index_entries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, torch.Tensor]]] = []
    shard_index = 0
    for source_offset in range(distributed.rank, len(records), distributed.world_size):
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
            with torch.inference_mode():
                outputs = adapter.forward_clean(inputs)
                selected = adapter.selected_hidden_features(outputs, layer_ids)[0]
            context_original = select_context_positions(
                inputs["input_ids"][0],
                context_mode=config.context_mode,
                image_token_ids=image_ids,
                video_token_ids=video_ids,
            )
            cached_tensors = {
                "input_ids": inputs["input_ids"][0].to(dtype=torch.int32),
                "position_ids": position_ids[:, 0].to(dtype=torch.int32),
                "context_hidden": selected[context_original].to(dtype=dtype),
                "context_original_positions": context_original.to(dtype=torch.int32),
            }
            pending.append((source_offset, cached_tensors))
            index_entries.append(
                {
                    "source_offset": source_offset,
                    "manifest_id": str(record["id"]),
                    "source": record.get("source", {}),
                    "response_start": response_start,
                    "response_end": response_end,
                    "sequence_length": response_end,
                    "context_length": int(context_original.numel()),
                    "response_truncated": response_truncated,
                    "teacher_response_mode": config.teacher_response_mode,
                    "dropped_prompt_turns": dropped_turns,
                    "shard": None,
                    "tensor_prefix": _tensor_prefix(source_offset),
                }
            )
            if len(pending) >= config.cache_shard_size:
                filename, _ = _write_shard(
                    temporary,
                    shard_index,
                    pending,
                    rank=distributed.rank if distributed.enabled else None,
                )
                for entry in index_entries[-len(pending) :]:
                    entry["shard"] = filename
                pending.clear()
                shard_index += 1
            local_done = len(index_entries) + len(skipped)
            if local_done == 1 or local_done % config.cache_log_every == 0:
                print(
                    f"[teacher rank={distributed.rank} source={source_offset + 1}/{len(records)}] "
                    f"id={record['id']} seq={response_end} "
                    f"context={int(context_original.numel())} truncated={response_truncated} "
                    f"dropped_turns={dropped_turns}"
                )
            del outputs, selected, inputs, cached_tensors
        except ValueError as exc:
            skipped.append(
                {
                    "source_offset": source_offset,
                    "manifest_id": str(record.get("id", source_offset)),
                    "reason": str(exc),
                }
            )
            print(
                f"[teacher-skip rank={distributed.rank}] "
                f"id={record.get('id', source_offset)} reason={exc}"
            )
    if pending:
        filename, _ = _write_shard(
            temporary,
            shard_index,
            pending,
            rank=distributed.rank if distributed.enabled else None,
        )
        for entry in index_entries[-len(pending) :]:
            entry["shard"] = filename

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
