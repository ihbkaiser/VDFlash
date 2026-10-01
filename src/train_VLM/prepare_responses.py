from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .config import DFlashTrainConfig
from .target import Qwen25VLTargetAdapter, load_jsonl, validate_manifest_record
from .video import prepare_qwen_message_batch


def _eos_token_ids(adapter: Qwen25VLTargetAdapter) -> set[int]:
    value = getattr(getattr(adapter.model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(adapter.processor.tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(token_id) for token_id in value}
    return {int(value)}


def prepare_responses(
    adapter: Qwen25VLTargetAdapter,
    records: list[dict],
    *,
    max_new_tokens: int,
    max_seq_length: int,
    output_path: str | Path,
) -> None:
    """Generate and persist exact target token IDs for a multimodal manifest."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    adapter.model.eval()
    eos_token_ids = _eos_token_ids(adapter)
    written = 0
    skipped_without_eos = 0
    skipped_length = 0
    with tmp_path.open("w") as writer:
        progress = tqdm(
            enumerate(records),
            total=len(records),
            desc="Generate target responses",
            unit="samples",
            dynamic_ncols=True,
        )
        for index, record in progress:
            validate_manifest_record(record, require_target_response=False)
            messages = record["messages"]
            inputs, media_metadata = adapter.prepare_messages(messages)
            prompt_length = int(inputs["input_ids"].shape[1])
            if prompt_length >= max_seq_length:
                skipped_length += 1
                tqdm.write(
                    f"[skip] {record.get('id', index)}: prompt has {prompt_length} tokens, "
                    f"leaving no response room under max_seq_length={max_seq_length}"
                )
                progress.set_postfix(prepared=written, skipped_eos=skipped_without_eos, skipped_length=skipped_length)
                continue
            with torch.inference_mode():
                generated = adapter.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.0,
                    temperature=None,
                    use_cache=True,
                )
            response_ids = generated[0, prompt_length:].detach().cpu().tolist()
            if eos_token_ids and not any(token_id in eos_token_ids for token_id in response_ids):
                skipped_without_eos += 1
                tqdm.write(
                    f"[skip] {record.get('id', index)}: target response reached max_new_tokens "
                    "without an EOS token"
                )
                progress.set_postfix(prepared=written, skipped_eos=skipped_without_eos, skipped_length=skipped_length)
                continue
            if prompt_length + len(response_ids) > max_seq_length:
                skipped_length += 1
                tqdm.write(
                    f"[skip] {record.get('id', index)}: clean sequence has "
                    f"{prompt_length + len(response_ids)} tokens, exceeding max_seq_length={max_seq_length}"
                )
                progress.set_postfix(prepared=written, skipped_eos=skipped_without_eos, skipped_length=skipped_length)
                continue
            response_text = adapter.processor.tokenizer.decode(
                response_ids, skip_special_tokens=False
            )
            output = copy.deepcopy(record)
            output["target_response"] = {
                "token_ids": response_ids,
                "text": response_text,
                "generation": {
                    "do_sample": False,
                    "temperature": 0.0,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,
                    "eos_token_ids": sorted(eos_token_ids),
                },
            }
            output["provenance"] = adapter.target_provenance()
            if media_metadata.frame_counts or media_metadata.video_grid_thw:
                output["video_preprocessing"] = {
                    "frame_counts": list(media_metadata.frame_counts),
                    "video_grid_thw": [list(row) for row in media_metadata.video_grid_thw],
                    "video_reader": getattr(adapter, "video_reader", "torchvision"),
                }
            writer.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
            progress.set_postfix(
                sample=record.get("id", index),
                tokens=len(response_ids),
                prepared=written,
                skipped_eos=skipped_without_eos,
                skipped_length=skipped_length,
            )
    tmp_path.replace(output_path)
    print(
        f"[summary] prepared={written} skipped_without_eos={skipped_without_eos} "
        f"skipped_length={skipped_length}"
    )


def prepare_responses_batched(
    adapter: Qwen25VLTargetAdapter,
    records: list[dict],
    *,
    max_new_tokens: int,
    max_seq_length: int,
    output_path: str | Path,
    batch_size: int = 8,
    resume: bool = False,
) -> None:
    """Generate image-caption responses with padded HF batches (no vLLM needed)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"output exists: {output_path}; pass --resume-output or --overwrite-output"
            )
        with output_path.open("rb+") as reader:
            raw = reader.read()
            if raw and not raw.endswith(b"\n"):
                reader.seek(raw.rfind(b"\n") + 1 if b"\n" in raw else 0)
                reader.truncate()
        with output_path.open(encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in resumable output {output_path}:{line_number}") from exc
                if row.get("id") is not None and isinstance(row.get("target_response"), dict):
                    completed_ids.add(str(row["id"]))

    pending = [row for row in records if str(row.get("id")) not in completed_ids]
    print(
        f"[HF batch] pending={len(pending)} completed={len(completed_ids)} batch_size={batch_size}",
        flush=True,
    )
    model = adapter.model
    model.eval()
    tokenizer = adapter.processor.tokenizer
    eos_token_ids = _eos_token_ids(adapter)
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    output_mode = "a" if output_path.exists() and resume else "w"
    written = 0
    skipped_eos = 0
    skipped_length = 0
    try:
        with output_path.open(output_mode, encoding="utf-8") as writer:
            progress = tqdm(
                range(0, len(pending), batch_size),
                desc="Generate target responses (HF batch)",
                unit="batches",
                dynamic_ncols=True,
            )
            for start in progress:
                batch = pending[start : start + batch_size]
                for record in batch:
                    validate_manifest_record(record, require_target_response=False)
                messages_batch = [record["messages"] for record in batch]
                inputs = prepare_qwen_message_batch(
                    adapter.processor,
                    messages_batch,
                    processor_kwargs=getattr(adapter.processor, "dflash_processor_kwargs", {}),
                    video_reader=getattr(adapter, "video_reader", "torchvision"),
                    image_min_pixels=getattr(adapter, "image_min_pixels", None),
                    image_max_pixels=getattr(adapter, "image_max_pixels", None),
                    video_num_frames=getattr(adapter, "video_num_frames", None),
                    video_min_pixels=getattr(adapter, "video_min_pixels", None),
                    video_max_pixels=getattr(adapter, "video_max_pixels", None),
                )
                inputs = {
                    key: value.to(adapter.device) if torch.is_tensor(value) else value
                    for key, value in inputs.items()
                }
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    input_ids = inputs["input_ids"]
                    pad_id = getattr(tokenizer, "pad_token_id", None)
                    prompt_lengths = (
                        torch.full((len(batch),), input_ids.shape[1], device=input_ids.device)
                        if pad_id is None
                        else input_ids.ne(int(pad_id)).sum(dim=1)
                    )
                else:
                    prompt_lengths = attention_mask.sum(dim=1)
                available = [max_seq_length - int(length) for length in prompt_lengths.tolist()]
                valid = [index for index, count in enumerate(available) if count > 0]
                skipped_length += len(batch) - len(valid)
                if not valid:
                    progress.set_postfix(written=written, skipped_length=skipped_length)
                    continue
                generation_limit = min(max_new_tokens, min(available[index] for index in valid))
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=generation_limit,
                        do_sample=False,
                        repetition_penalty=1.0,
                        use_cache=True,
                    )
                padded_prompt_width = int(inputs["input_ids"].shape[1])
                for index, record in enumerate(batch):
                    if index not in valid:
                        continue
                    response_ids = generated[index, padded_prompt_width:].detach().cpu().tolist()
                    if eos_token_ids:
                        eos_position = next(
                            (offset for offset, token_id in enumerate(response_ids) if token_id in eos_token_ids),
                            None,
                        )
                        if eos_position is None:
                            skipped_eos += 1
                            continue
                        response_ids = response_ids[: eos_position + 1]
                    if int(prompt_lengths[index]) + len(response_ids) > max_seq_length:
                        skipped_length += 1
                        continue
                    output = copy.deepcopy(record)
                    output["target_response"] = {
                        "token_ids": [int(token_id) for token_id in response_ids],
                        "text": tokenizer.decode(response_ids, skip_special_tokens=False),
                        "generation": {
                            "backend": "transformers_batched",
                            "do_sample": False,
                            "temperature": 0.0,
                            "repetition_penalty": 1.0,
                            "max_new_tokens": generation_limit,
                            "use_cache": True,
                            "eos_token_ids": sorted(eos_token_ids),
                        },
                    }
                    output["provenance"] = adapter.target_provenance()
                    writer.write(json.dumps(output, ensure_ascii=False) + "\n")
                    written += 1
                writer.flush()
                progress.set_postfix(
                    batch=len(batch), written=written, skipped_eos=skipped_eos,
                    skipped_length=skipped_length, refresh=False,
                )
    finally:
        tokenizer.padding_side = old_padding_side
    print(
        f"[HF batch summary] newly_written={written} total_completed={len(completed_ids) + written} "
        f"skipped_without_eos={skipped_eos} skipped_length={skipped_length}",
        flush=True,
    )


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate DFlash target responses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="JSONL manifest without target_response")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = DFlashTrainConfig.from_file(args.config)
    adapter = Qwen25VLTargetAdapter.from_pretrained(config)
    prepare_responses(
        adapter,
        load_jsonl(args.input),
        max_new_tokens=config.response_max_new_tokens,
        max_seq_length=config.max_seq_length,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
