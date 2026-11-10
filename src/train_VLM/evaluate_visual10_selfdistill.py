from __future__ import annotations

"""Evaluate video DFlash checkpoints for greedy parity, acceptance, and latency."""

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .cache_visual10_selfdistill_features import (
    _atomic_json,
    _load_target_config,
)
from .target import Qwen25VLTargetAdapter, load_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Prepared responses/{split}.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--visual-retention", type=float, default=0.1)
    parser.add_argument(
        "--visual-selector",
        choices=("prefix", "temporal_spatial"),
        default="temporal_spatial",
    )
    parser.add_argument(
        "--full-visual",
        action="store_true",
        help="Evaluate without context pruning",
    )
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stop_token_ids(tokenizer: Any) -> list[int]:
    values: set[int] = set()
    for name in ("eos_token_id", "im_end_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, (list, tuple, set)):
            values.update(int(item) for item in value if item is not None)
        elif value is not None:
            values.add(int(value))
    return sorted(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _prepare_inputs(adapter: Qwen25VLTargetAdapter, record: dict[str, Any]):
    inputs, _metadata = adapter.prepare_messages(record["messages"])
    position_ids = adapter._compute_position_ids(inputs)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    media = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids"}
    }
    return input_ids, attention_mask, position_ids, media


def _evaluate_record(
    *,
    adapter: Qwen25VLTargetAdapter,
    draft: torch.nn.Module,
    record: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
    stop_ids: list[int],
    visual_retention: float | None,
    visual_selector: str,
) -> dict[str, Any]:
    prepare_started = time.perf_counter()
    input_ids, attention_mask, position_ids, media = _prepare_inputs(adapter, record)
    _sync(device)
    prepare_seconds = time.perf_counter() - prepare_started
    visual_ids = set().union(*adapter.visual_token_ids)
    visual_count = sum(
        int((input_ids == token_id).sum().item()) for token_id in visual_ids
    )
    if visual_count == 0:
        raise ValueError(f"record {record.get('id')} has no visual placeholder tokens")

    target = adapter.model
    with torch.inference_mode():
        _sync(device)
        target_started = time.perf_counter()
        target_output = target.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **media,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            **({"eos_token_id": stop_ids} if stop_ids else {}),
        )
        _sync(device)
        target_seconds = time.perf_counter() - target_started

        _sync(device)
        speculative_started = time.perf_counter()
        speculative_output = draft.spec_generate(
            target,
            input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=0.0,
            position_ids=position_ids,
            target_kwargs=media,
            visual_retention=visual_retention,
            visual_selector=visual_selector,
            profile_timing=True,
        )
        _sync(device)
        speculative_seconds = time.perf_counter() - speculative_started

    target_new = target_output[:, input_ids.shape[1] :]
    speculative_new = speculative_output[:, input_ids.shape[1] :]
    parity = torch.equal(target_new, speculative_new)
    metrics = dict(getattr(draft, "last_speculative_metrics", {}))
    accepted = [int(value) for value in metrics.get("accepted_draft_tokens_per_round", [])]
    full_accepted = [
        int(value)
        for value in metrics.get("full_accepted_draft_tokens_per_round", accepted)
    ]
    capacities = [
        int(value) for value in metrics.get("proposal_capacity_per_round", accepted)
    ]
    effective = [
        int(value) for value in metrics.get("effective_tokens_per_round", accepted)
    ]
    return {
        "id": str(record.get("id", "")),
        "prompt_tokens": int(input_ids.shape[1]),
        "target_output_tokens": int(target_new.shape[1]),
        "visual_tokens_total": visual_count,
        "visual_tokens_retained": int(metrics.get("visual_tokens_retained", visual_count)),
        "visual_retention_actual": float(
            metrics.get("visual_retention_actual", 1.0)
        ),
        "parity": bool(parity),
        "rounds": len(accepted),
        "accepted_draft_tokens": sum(accepted),
        "accepted_draft_tokens_per_round": accepted,
        "full_accepted_draft_tokens_per_round": full_accepted,
        "proposal_capacity_per_round": capacities,
        "effective_tokens_per_round": effective,
        "tau_proposal": (
            sum(accepted) / len(accepted) if accepted else 0.0
        ),
        "tau_effective": (
            sum(effective) / len(effective) if effective else 0.0
        ),
        "input_prepare_time_s": prepare_seconds,
        "target_generate_time_s": target_seconds,
        "speculative_generate_time_s": speculative_seconds,
        "target_prefill_time_s": float(metrics.get("target_prefill_time_s", 0.0)),
        "visual_selection_time_s": float(metrics.get("visual_selection_time_s", 0.0)),
        "draft_forward_time_s": float(metrics.get("draft_forward_time_s", 0.0)),
        "target_verifier_time_s": float(metrics.get("target_verifier_time_s", 0.0)),
        "target_kv_bytes_peak": int(metrics.get("target_kv_bytes_peak", 0)),
        "draft_kv_bytes_peak": int(metrics.get("draft_kv_bytes_peak", 0)),
    }


def main() -> int:
    args = _parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples must be non-negative")
    if not args.full_visual and not 0.0 < args.visual_retention <= 1.0:
        raise ValueError("--visual-retention must be in (0, 1]")

    target_config = _load_target_config(args.target_config)
    records = load_jsonl(args.manifest)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("evaluation manifest is empty")

    device = torch.device(args.device)
    adapter = Qwen25VLTargetAdapter.from_pretrained(
        target_config,
        device=device,
        dtype=(
            torch.bfloat16 if target_config.mixed_precision == "bf16" else None
        ),
    )
    adapter.freeze()
    for record in records:
        adapter.validate_record_provenance(record)

    # Import the local trainer package only after the standalone VLM package has
    # loaded its target/processor configuration.
    import sys

    workspace = Path(__file__).resolve().parents[2]
    specforge_root = workspace / "src" / "train_Dflash_SpecForge"
    for import_root in (workspace, specforge_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from specforge.export.checkpoint_io import materialize_draft, resolve_training_state

    dtype = {
        "no": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[target_config.mixed_precision]
    draft_config_path = Path(args.draft_config).expanduser()
    draft_config_source = (
        str(draft_config_path.resolve())
        if draft_config_path.exists()
        else args.draft_config
    )
    checkpoint_candidate = Path(args.checkpoint).expanduser()
    checkpoint_source = (
        str(checkpoint_candidate.resolve())
        if checkpoint_candidate.exists()
        else args.checkpoint
    )
    from specforge.modeling.auto import AutoDraftModel
    from specforge.training.model_loading import (
        load_draft_config_source,
        warm_start_draft_model,
    )

    resolved_draft_config = load_draft_config_source(draft_config_source)
    resolved_draft_config._attn_implementation = "sdpa"
    draft = AutoDraftModel.from_config(resolved_draft_config, torch_dtype=dtype)
    warm_start_draft_model(
        draft,
        checkpoint_source,
        draft_config=resolved_draft_config,
        strategy="dflash",
        warm_start_mode="strict",
    )
    draft.to(device=device, dtype=dtype).eval()

    tokenizer = adapter.processor.tokenizer
    stop_ids = _stop_token_ids(tokenizer)
    retention = None if args.full_visual else args.visual_retention
    warmup_count = min(args.warmup_samples, len(records))
    for record in records[:warmup_count]:
        _evaluate_record(
            adapter=adapter,
            draft=draft,
            record=record,
            device=device,
            max_new_tokens=min(args.max_new_tokens, 16),
            stop_ids=stop_ids,
            visual_retention=retention,
            visual_selector=args.visual_selector,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    per_sample = []
    for record in records:
        result = _evaluate_record(
            adapter=adapter,
            draft=draft,
            record=record,
            device=device,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
            visual_retention=retention,
            visual_selector=args.visual_selector,
        )
        per_sample.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    total_rounds = sum(row["rounds"] for row in per_sample)
    total_accepted = sum(row["accepted_draft_tokens"] for row in per_sample)
    total_effective = sum(
        sum(row["effective_tokens_per_round"]) for row in per_sample
    )
    total_visual = sum(row["visual_tokens_total"] for row in per_sample)
    total_retained = sum(row["visual_tokens_retained"] for row in per_sample)
    tau_proposal = total_accepted / total_rounds if total_rounds else 0.0
    target_times = [row["target_generate_time_s"] for row in per_sample]
    speculative_times = [row["speculative_generate_time_s"] for row in per_sample]
    prepare_times = [row["input_prepare_time_s"] for row in per_sample]
    parity_failures = [row["id"] for row in per_sample if not row["parity"]]
    block_size = int(getattr(draft, "block_size", 16))
    rejection_histogram: dict[str, int] = {}
    proposal_capacity_totals = [0] * (block_size - 1)
    accepted_prefix_totals = [0] * (block_size - 1)
    for row in per_sample:
        for accepted, full_accepted, capacity in zip(
            row["accepted_draft_tokens_per_round"],
            row["full_accepted_draft_tokens_per_round"],
            row["proposal_capacity_per_round"],
        ):
            if full_accepted < capacity:
                key = f"first_rejection_offset_{full_accepted + 1}"
            elif capacity < block_size - 1:
                key = "round_end_truncated"
            else:
                key = "all_proposals_accepted"
            rejection_histogram[key] = rejection_histogram.get(key, 0) + 1
            for offset in range(1, min(capacity, block_size - 1) + 1):
                proposal_capacity_totals[offset - 1] += 1
                if accepted >= offset:
                    accepted_prefix_totals[offset - 1] += 1
    max_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else None
    )
    summary = {
        "status": "complete" if not parity_failures else "parity_failed",
        "checkpoint": checkpoint_source,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "num_samples": len(per_sample),
        "max_new_tokens": args.max_new_tokens,
        "visual_retention_requested": retention,
        "visual_selector": args.visual_selector,
        "visual_retention_actual_weighted": (
            total_retained / total_visual if total_visual else 1.0
        ),
        "visual_tokens_total": total_visual,
        "visual_tokens_retained": total_retained,
        "tau_proposal": tau_proposal,
        "tau_effective": total_effective / total_rounds if total_rounds else 0.0,
        "first_rejection_histogram": rejection_histogram,
        "accepted_prefix_survival_by_offset": {
            str(offset + 1): (
                accepted_prefix_totals[offset] / proposal_capacity_totals[offset]
                if proposal_capacity_totals[offset]
                else None
            )
            for offset in range(block_size - 1)
        },
        "mean_rounds_per_sample": statistics.mean(row["rounds"] for row in per_sample),
        "greedy_parity_rate": 1.0 - len(parity_failures) / len(per_sample),
        "parity_failure_ids": parity_failures,
        "input_prepare_time_s_mean": statistics.mean(prepare_times),
        "target_generate_time_s_mean": statistics.mean(target_times),
        "target_generate_time_s_p50": _percentile(target_times, 0.50),
        "target_generate_time_s_p95": _percentile(target_times, 0.95),
        "speculative_generate_time_s_mean": statistics.mean(speculative_times),
        "speculative_generate_time_s_p50": _percentile(speculative_times, 0.50),
        "speculative_generate_time_s_p95": _percentile(speculative_times, 0.95),
        "visual_selection_time_s_mean": statistics.mean(
            row["visual_selection_time_s"] for row in per_sample
        ),
        "draft_forward_time_s_mean": statistics.mean(
            row["draft_forward_time_s"] for row in per_sample
        ),
        "target_verifier_time_s_mean": statistics.mean(
            row["target_verifier_time_s"] for row in per_sample
        ),
        "target_kv_bytes_peak_mean": statistics.mean(
            row["target_kv_bytes_peak"] for row in per_sample
        ),
        "draft_kv_bytes_peak_mean": statistics.mean(
            row["draft_kv_bytes_peak"] for row in per_sample
        ),
        "peak_allocated_memory_gb": max_memory_gb,
        "per_sample": per_sample,
    }
    output_path = Path(args.output).expanduser().resolve()
    _atomic_json(output_path, summary)
    aggregate = {key: value for key, value in summary.items() if key != "per_sample"}
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0 if not parity_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
