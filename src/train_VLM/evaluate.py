from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .config import DFlashTrainConfig
from .data import build_masked_blocks, make_anchor_generator, sample_anchor_positions
from .target import Qwen25VLTargetAdapter, load_jsonl, validate_manifest_record
from .trainer import extract_training_context, load_draft_checkpoint
from .vlm_decode import Qwen25VLDFlashDecoder


def _now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def teacher_forced_acceptance(
    adapter: Qwen25VLTargetAdapter,
    draft_model,
    record: dict,
    config: DFlashTrainConfig,
    *,
    epoch: int = 0,
) -> dict[str, float]:
    """Measure accepted prefixes against exact clean target token IDs."""

    example = adapter.prepare_record(record, max_seq_length=config.max_seq_length)
    layer_ids = list(getattr(draft_model, "target_layer_ids"))
    context_hidden, context_positions, context_original_positions = extract_training_context(
        adapter, example, config, layer_ids
    )
    generator = make_anchor_generator(config.seed, epoch, example.sample_id, device=adapter.device)
    anchors = sample_anchor_positions(
        example.response_start,
        example.response_end,
        config.block_size,
        config.num_anchors,
        generator=generator,
        device=example.input_ids.device,
    )
    blocks = build_masked_blocks(
        example.input_ids,
        anchors,
        block_size=config.block_size,
        mask_token_id=int(draft_model.mask_token_id),
        position_ids=example.position_ids,
    )
    embeddings = adapter.input_embeddings(blocks.block_input_ids.reshape(1, -1))
    hidden = draft_model(
        noise_embeddings=embeddings,
        target_context=context_hidden,
        context_position_ids=context_positions,
        block_position_ids=blocks.block_position_ids,
        anchors=blocks.anchors,
        context_original_positions=context_original_positions,
        use_flex_attention=config.use_flex_attention,
    )
    logits = adapter.lm_head(hidden).reshape(1, anchors.numel(), config.block_size, -1)
    predictions = logits[:, :, 1:, :].argmax(-1)[0]
    labels = blocks.labels[:, 1:]
    matches = predictions.eq(labels)
    prefix = []
    for row in matches:
        correct = 0
        while correct < row.numel() and bool(row[correct]):
            correct += 1
        prefix.append(correct + 1)
    return {
        "accepted_prefix": float(sum(prefix) / max(1, len(prefix))),
        "token_accuracy": float(matches.float().mean().cpu()),
        "blocks": float(len(prefix)),
    }


def evaluate_records(
    adapter: Qwen25VLTargetAdapter,
    draft_model,
    records: list[dict],
    config: DFlashTrainConfig,
    *,
    epoch: int = 0,
) -> dict[str, float]:
    values = []
    for record in records:
        try:
            values.append(teacher_forced_acceptance(adapter, draft_model, record, config, epoch=epoch))
        except ValueError as exc:
            print(f"[skip] {record.get('id')}: {exc}")
    if not values:
        raise RuntimeError("No valid evaluation records")
    return {key: sum(value[key] for value in values) / len(values) for key in values[0]} | {
        "records": float(len(values))
    }


def _prompt_inputs(adapter: Qwen25VLTargetAdapter, record: dict[str, Any]) -> dict[str, Any]:
    validate_manifest_record(record, require_target_response=False)
    inputs, _ = adapter.prepare_messages(record["messages"])
    return inputs


@torch.inference_mode()
def benchmark_decode_records(
    adapter: Qwen25VLTargetAdapter,
    draft_model,
    records: list[dict[str, Any]],
    config: DFlashTrainConfig,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> dict[str, float]:
    """Benchmark target autoregressive generation and lossless DFlash decoding."""

    if temperature > 1e-5:
        raise NotImplementedError("The Qwen2.5-VL DFlash decoder currently supports greedy mode only")
    decoder = Qwen25VLDFlashDecoder(adapter, draft_model, config)
    eos = getattr(adapter.processor.tokenizer, "eos_token_id", None)
    stop_token_ids = [int(eos)] if eos is not None else None
    baseline_latency = 0.0
    baseline_tokens = 0
    dflash_results = []
    for record in records:
        inputs = _prompt_inputs(adapter, record)
        prompt_length = int(inputs["input_ids"].shape[1])
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        started = _now(adapter.device)
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "temperature": None,
            "use_cache": True,
        }
        baseline_ids = adapter.model.generate(**generation_kwargs)
        baseline_latency += _now(adapter.device) - started
        baseline_tokens += int(baseline_ids.shape[1] - prompt_length)
        dflash_results.append(
            decoder.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=stop_token_ids,
            )
        )
    if not dflash_results:
        raise RuntimeError("No records supplied for decode benchmark")
    dflash_tokens = sum(result.num_output_tokens for result in dflash_results)
    dflash_end_to_end = sum(result.end_to_end_latency_s for result in dflash_results)
    dflash_decode = sum(result.decode_latency_s for result in dflash_results)
    return {
        "records": float(len(dflash_results)),
        "baseline_output_tokens": float(baseline_tokens),
        "baseline_end_to_end_latency_s": baseline_latency,
        "baseline_tokens_per_second": baseline_tokens / max(baseline_latency, 1e-12),
        "dflash_output_tokens": float(dflash_tokens),
        "dflash_prefill_latency_s": sum(result.prefill_latency_s for result in dflash_results),
        "dflash_draft_latency_s": sum(result.draft_latency_s for result in dflash_results),
        "dflash_verify_latency_s": sum(result.verify_latency_s for result in dflash_results),
        "dflash_decode_latency_s": dflash_decode,
        "dflash_end_to_end_latency_s": dflash_end_to_end,
        "dflash_tokens_per_second": dflash_tokens / max(dflash_end_to_end, 1e-12),
        "decoding_speedup": baseline_latency / max(dflash_end_to_end, 1e-12),
        "accepted_prefix": sum(result.mean_acceptance_length for result in dflash_results)
        / len(dflash_results),
        "peak_memory_bytes": float(max(result.peak_memory_bytes for result in dflash_results)),
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate and benchmark a DFlash Qwen2.5-VL draft")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--decode-benchmark", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    config = DFlashTrainConfig.from_file(args.config)
    adapter = Qwen25VLTargetAdapter.from_pretrained(config)
    draft = load_draft_checkpoint(args.checkpoint, adapter, config)
    records = load_jsonl(args.manifest)
    if args.max_samples:
        records = records[: args.max_samples]
    if args.decode_benchmark:
        metrics = benchmark_decode_records(
            adapter,
            draft,
            records,
            config,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        metrics = evaluate_records(adapter, draft, records, config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
