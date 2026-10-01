from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from tqdm.auto import tqdm


def _image_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"vLLM held-out generation expects a local image, got {value!r}")
    return Path(value)


def _chat_for_record(record: dict[str, Any], image: Any) -> list[dict[str, Any]]:
    """Convert this project's manifest shape to vLLM's offline chat format."""
    conversation = []
    inserted_image = False
    for message in record["messages"]:
        content = message.get("content", [])
        if isinstance(content, str):
            conversation.append({"role": message["role"], "content": content})
            continue
        converted = []
        for part in content:
            if part.get("type") == "image":
                if inserted_image:
                    raise ValueError(f"record {record.get('id')} contains more than one image")
                converted.append({"type": "image_pil", "image_pil": image})
                inserted_image = True
            elif part.get("type") == "text":
                converted.append({"type": "text", "text": part.get("text", "")})
            else:
                raise ValueError(f"unsupported vLLM chat content in record {record.get('id')}: {part!r}")
        conversation.append({"role": message["role"], "content": converted})
    if not inserted_image:
        raise ValueError(f"record {record.get('id')} has no image content")
    return conversation


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def _recover_partial_final_line(path: Path) -> None:
    """Drop only an incomplete trailing JSONL record left by an interrupted write."""
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last_newline = raw.rfind(b"\n")
        with path.open("r+b") as output:
            output.truncate(last_newline + 1 if last_newline >= 0 else 0)


def prepare_responses_vllm(
    records: list[dict[str, Any]],
    *,
    model: str,
    output_path: str | Path,
    max_new_tokens: int,
    max_seq_length: int,
    batch_size: int = 64,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 16384,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    processor_kwargs: dict[str, Any] | None = None,
    resume: bool = False,
) -> None:
    """Generate with vLLM continuous batching and append each completed batch.

    ``batch_size`` bounds host-side image loading. The vLLM engine then schedules
    requests continuously up to ``max_num_seqs`` and ``max_num_batched_tokens``.
    In resume mode, completed IDs already present in output_path are retained.
    """
    if batch_size < 1 or max_num_seqs < 1 or max_num_batched_tokens < 1:
        raise ValueError("batch_size, max_num_seqs, and max_num_batched_tokens must be positive")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    try:
        from PIL import Image
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - depends on optional vLLM environment
        raise RuntimeError(
            "The vLLM backend requires vllm and Pillow. Install vLLM in a separate compatible "
            "environment, then run this command with that environment's Python."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if output_path.exists():
        if not resume:
            raise FileExistsError(f"output exists: {output_path}; pass --resume-output or --overwrite-output")
        _recover_partial_final_line(output_path)
        for row in _read_jsonl(output_path):
            if row.get("id") is not None and isinstance(row.get("target_response"), dict):
                completed_ids.add(str(row["id"]))
    pending = [row for row in records if str(row.get("id")) not in completed_ids]
    print(
        f"[vLLM] pending={len(pending)} completed={len(completed_ids)} batch_size={batch_size} "
        f"max_num_seqs={max_num_seqs} max_num_batched_tokens={max_num_batched_tokens} "
        f"gpu_memory_utilization={gpu_memory_utilization:.2f}",
        flush=True,
    )

    mm_processor_kwargs = dict(processor_kwargs or {})
    if image_min_pixels is not None:
        mm_processor_kwargs.setdefault("min_pixels", image_min_pixels)
    if image_max_pixels is not None:
        mm_processor_kwargs.setdefault("max_pixels", image_max_pixels)
    llm = LLM(
        model=model,
        tokenizer=model,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_seq_length,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs=mm_processor_kwargs or None,
        trust_remote_code=False,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        repetition_penalty=1.0,
    )
    mode = "a" if output_path.exists() and resume else "w"
    written = 0
    with output_path.open(mode, encoding="utf-8") as writer:
        progress = tqdm(range(0, len(pending), batch_size), desc="vLLM held-out batches", unit="batch", dynamic_ncols=True)
        for start in progress:
            batch = pending[start : start + batch_size]
            images = []
            try:
                chats = []
                for row in batch:
                    image_part = next(
                        (part for msg in row["messages"] for part in msg.get("content", [])
                         if isinstance(part, dict) and part.get("type") == "image"),
                        None,
                    )
                    if image_part is None:
                        raise ValueError(f"record {row.get('id')} has no image content")
                    path = _image_path(image_part["image"])
                    with Image.open(path) as opened:
                        image = opened.convert("RGB")
                    images.append(image)
                    chats.append(_chat_for_record(row, image))
                outputs = llm.chat(chats, sampling_params=sampling_params, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(batch)} inputs")
                for row, result in zip(batch, outputs):
                    candidate = result.outputs[0]
                    output = copy.deepcopy(row)
                    output["target_response"] = {
                        "token_ids": [int(token_id) for token_id in candidate.token_ids],
                        "text": candidate.text,
                        "generation": {
                            "backend": "vllm",
                            "do_sample": False,
                            "temperature": 0.0,
                            "repetition_penalty": 1.0,
                            "max_new_tokens": max_new_tokens,
                            "finish_reason": candidate.finish_reason,
                            "max_seq_length": max_seq_length,
                        },
                    }
                    output["provenance"] = {
                        "backend": "vllm",
                        "model": model,
                        "dtype": dtype,
                        "tensor_parallel_size": tensor_parallel_size,
                    }
                    writer.write(json.dumps(output, ensure_ascii=False) + "\n")
                    written += 1
                writer.flush()
            finally:
                for image in images:
                    image.close()
            progress.set_postfix(written=written, total=len(pending), refresh=False)
    print(f"[vLLM summary] newly_written={written} total_completed={len(completed_ids) + written}", flush=True)
