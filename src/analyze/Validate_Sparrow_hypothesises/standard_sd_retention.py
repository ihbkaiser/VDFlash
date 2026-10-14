"""Standard independent-draft speculative decoding with visual retention.

This runner is deliberately separate from the MSD/DFlash paths.  The target
and draft are two independent Qwen2.5-VL checkpoints; only the draft-side
visual context is compacted.  The target verifier always receives the full
video context, so exact speculative decoding remains lossless when the cache
bookkeeping is correct.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .dataset import load_vdc_manifest, write_jsonl
from .model_analysis import (
    capture_query_attention,
    find_instruction_masks,
    load_qwen_model,
    prepare_qwen2vl_prefill,
)
from .runtime import (
    build_qwen2vl_video_processor,
    last_position_lm_head,
    model_device,
    move_batch_to_device,
    process_video,
    select_visual_positions,
)


def _new_cache() -> Any:
    from transformers import DynamicCache

    return DynamicCache()


def _cache_length(cache: Any) -> int:
    getter = getattr(cache, "get_seq_length", None)
    return int(getter()) if callable(getter) else 0


def _crop_cache(cache: Any, length: int) -> None:
    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(int(length))


def _extend_position_ids(position_ids: torch.Tensor, total_length: int, device: torch.device) -> torch.Tensor:
    position_ids = position_ids.to(device=device, dtype=torch.long)
    if total_length < position_ids.shape[-1]:
        raise ValueError("total_length cannot be shorter than position_ids")
    if position_ids.ndim == 2:
        starts = position_ids[:, -1:] + 1
        offsets = torch.arange(total_length - position_ids.shape[1], device=device).view(1, -1)
        return torch.cat([position_ids, starts + offsets], dim=-1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 3:
        starts = position_ids[:, :, -1:] + 1
        offsets = torch.arange(total_length - position_ids.shape[-1], device=device).view(1, 1, -1)
        return torch.cat([position_ids, starts + offsets], dim=-1)
    raise ValueError("position_ids must have shape [1, sequence] or [3, 1, sequence]")


def _eos_token_ids(processor: Any, target: Any) -> list[int]:
    values = (
        getattr(processor.tokenizer, "eos_token_id", None),
        getattr(processor.tokenizer, "im_end_id", None),
        getattr(getattr(target, "generation_config", None), "eos_token_id", None),
    )
    result: set[int] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        elif value is not None:
            result.add(int(value))
    return sorted(result)


@dataclass
class DecodeResult:
    output_ids: list[int]
    rounds: list[dict[str, Any]]
    prefill_s: float
    decode_s: float
    end_to_end_s: float

    @property
    def accepted_prefix_tokens(self) -> int:
        return sum(int(row["accepted_proposals"]) for row in self.rounds)

    @property
    def tau_proposal(self) -> float | None:
        if not self.rounds:
            return None
        return sum(int(row["accepted_proposals"]) for row in self.rounds) / len(self.rounds)

    @property
    def tau_effective(self) -> float | None:
        if not self.rounds:
            return None
        return sum(int(row["effective_emitted_tokens"]) for row in self.rounds) / len(self.rounds)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clock() -> float:
    if torch.cuda.is_available():
        _sync()
    return time.perf_counter()


def _last_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("model forward returned no logits")
    return logits[:, -1, :]


def _forward(
    model: Any,
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any,
    cache_position: torch.Tensor,
    logits_to_keep: int | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
        "return_dict": True,
    }
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = inputs_embeds
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = int(logits_to_keep)
    try:
        return model(**kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        kwargs.pop("logits_to_keep", None)
        return model(**kwargs)


def _prefill(model: Any, prepared: Any) -> tuple[Any, Any, torch.Tensor]:
    device = prepared.inputs_embeds.device
    cache = _new_cache()
    output = _forward(
        model,
        input_ids=None,
        inputs_embeds=prepared.inputs_embeds,
        position_ids=prepared.position_ids,
        attention_mask=prepared.attention_mask,
        past_key_values=cache,
        cache_position=torch.arange(prepared.inputs_embeds.shape[1], device=device),
        logits_to_keep=1,
    )
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise RuntimeError("model did not return a KV cache during prefill")
    return output, cache, _last_logits(output)


def _step(
    model: Any,
    token: torch.Tensor,
    cache: Any,
    positions: torch.Tensor,
    *,
    cache_position: int,
) -> tuple[Any, Any, torch.Tensor]:
    output = _forward(
        model,
        input_ids=token,
        inputs_embeds=None,
        position_ids=positions,
        attention_mask=None,
        past_key_values=cache,
        cache_position=torch.tensor([cache_position], device=token.device),
        logits_to_keep=1,
    )
    next_cache = getattr(output, "past_key_values", None)
    if next_cache is None:
        raise RuntimeError("model did not return a KV cache during decode")
    return output, next_cache, _last_logits(output)


def _rebuild_cache(
    model: Any,
    prepared: Any,
    generated_tokens: Sequence[int],
    all_positions: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    """Rebuild a clean cache after a rejected verification block.

    Some Transformers versions expose ``DynamicCache.crop`` but leave
    implementation-specific cache metadata inconsistent after a multimodal
    verification block.  Rebuilding only on rejection is slower than a crop,
    but makes the lossless gate unambiguous and keeps the all-accepted fast
    path on the normal cache.
    """

    _, cache, next_logits = _prefill(model, prepared)
    if not generated_tokens:
        return cache, next_logits
    # Replay one token at a time.  In this multimodal Qwen implementation,
    # feeding the whole generated prefix in one cached call can construct a
    # different causal-mask path from the normal greedy decoder.  Sequential
    # replay is slower, but it is the exact state transition used by the
    # lossless reference decoder.
    next_logits: torch.Tensor | None = None
    for token_id in generated_tokens:
        cache_len = _cache_length(cache)
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=prepared.input_ids.device)
        _, cache, next_logits = _step(
            model,
            token,
            cache,
            all_positions[..., cache_len : cache_len + 1],
            cache_position=cache_len,
        )
    if next_logits is None:
        raise RuntimeError("cache rebuild produced no logits")
    return cache, next_logits


def compact_prefill(prepared: Any, model: Any, percentage: float, scores: Sequence[float] | None) -> Any:
    """Compact a Qwen2.5 prepared prompt while preserving M-RoPE positions."""

    if not 0.0 <= float(percentage) <= 100.0:
        raise ValueError("retention percentage must be in [0, 100]")
    visual = prepared.video_positions.detach().to("cpu")
    selected_local = select_visual_positions(int(visual.numel()), float(percentage), scores)
    selected_global = visual[selected_local] if selected_local.numel() else visual[:0]
    total = prepared.input_ids.shape[1]
    keep = torch.ones(total, dtype=torch.bool, device=prepared.input_ids.device)
    visual_mask = torch.zeros_like(keep)
    if visual.numel():
        visual_mask[visual.to(keep.device)] = True
    if visual.numel() and selected_global.numel() < visual.numel():
        selected_mask = torch.zeros_like(keep)
        selected_mask[selected_global.to(keep.device)] = True
        keep &= ~visual_mask | selected_mask
    if float(percentage) == 0.0:
        marker_ids = {
            getattr(model.config, name, None)
            for name in ("vision_start_token_id", "vision_end_token_id", "video_token_id")
        }
        marker_ids.discard(None)
        for marker_id in marker_ids:
            keep &= prepared.input_ids[0] != int(marker_id)

    original_positions = torch.nonzero(keep, as_tuple=False).flatten()
    input_ids = prepared.input_ids[:, keep]
    inputs_embeds = prepared.inputs_embeds[:, keep, :]
    position_ids = prepared.position_ids[..., keep]
    attention_mask = prepared.attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, keep]
    compact_visual = torch.nonzero(visual_mask & keep, as_tuple=False).flatten()
    compact_visual = torch.searchsorted(original_positions, compact_visual)

    # Keep the same light-weight object shape expected by the decoder.
    return type(prepared)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=getattr(prepared, "rope_deltas", None),
        video_positions=compact_visual.to(input_ids.device),
        input_fingerprint=_hash_tensor(input_ids),
    )


def _hash_tensor(value: torch.Tensor) -> str:
    cpu = value.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.numpy().tobytes() + str(tuple(cpu.shape)).encode()).hexdigest()[:16]


def attention_scores(model: Any, batch: Any, processor: Any, prepared: Any) -> dict[str, list[float]]:
    """Capture target query-only attention once and derive both Sparrow policies."""

    visual = [int(value) for value in prepared.video_positions.detach().cpu().tolist()]
    masks = find_instruction_masks(prepared.input_ids, processor, visual)
    last_query = int(masks["query_index"])
    all_queries = sorted(set(masks["instruction_positions"]) | set(masks["text_positions"]))
    if not all_queries:
        all_queries = [last_query]
    with capture_query_attention(model, all_queries) as captured:
        with last_position_lm_head(model):
            with torch.inference_mode():
                model(**batch, use_cache=False, output_attentions=False, return_dict=True)
    if not captured:
        raise RuntimeError("query-only attention capture returned no layers")
    layers = torch.stack([captured[index] for index in sorted(captured)])
    if layers.ndim != 4:
        raise RuntimeError(f"unexpected captured attention shape: {tuple(layers.shape)}")
    # [layers, queries, heads, sequence] -> [queries, sequence]
    averaged = layers.float().mean(dim=(0, 2))
    query_to_row = {query: index for index, query in enumerate(all_queries)}
    last = averaged[query_to_row[last_query], visual]
    text_queries = [query for query in all_queries if query in set(masks["instruction_positions"]) | set(masks["text_positions"])]
    all_text = averaged[[query_to_row[query] for query in text_queries]][:, visual].mean(dim=0)
    return {
        "last_instruction": [float(value) for value in last.detach().cpu().tolist()],
        "all_text": [float(value) for value in all_text.detach().cpu().tolist()],
        "visual_token_count": len(visual),
        "instruction_positions": masks["instruction_positions"],
        "text_positions": masks["text_positions"],
        "query_index": last_query,
    }


@torch.inference_mode()
def greedy_decode(model: Any, prepared: Any, max_new_tokens: int, stop_ids: set[int]) -> tuple[list[int], dict[str, float]]:
    started = _clock()
    output, cache, next_logits = _prefill(model, prepared)
    prefill_s = _clock() - started
    all_positions = _extend_position_ids(
        prepared.position_ids,
        int(prepared.input_ids.shape[1]) + int(max_new_tokens) + 2,
        prepared.input_ids.device,
    )
    tokens: list[int] = []
    decode_start = _clock()
    for _ in range(max_new_tokens):
        token_id = int(next_logits.argmax(dim=-1).item())
        tokens.append(token_id)
        if token_id in stop_ids:
            break
        cache_len = _cache_length(cache)
        token = torch.tensor([[token_id]], dtype=torch.long, device=prepared.input_ids.device)
        _, cache, next_logits = _step(
            model,
            token,
            cache,
            all_positions[..., cache_len : cache_len + 1],
            cache_position=cache_len,
        )
    decode_s = _clock() - decode_start
    return tokens, {
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "end_to_end_s": float(_clock() - started),
    }


@torch.inference_mode()
def speculative_decode(
    target: Any,
    draft: Any,
    target_prepared: Any,
    draft_prepared: Any,
    *,
    max_new_tokens: int,
    block_size: int,
    stop_ids: set[int],
) -> DecodeResult:
    """Exact greedy independent-draft speculative decoding."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    started = _clock()
    target_output, target_cache, target_next = _prefill(target, target_prepared)
    prefill_s = _clock() - started
    _, draft_cache, draft_next = _prefill(draft, draft_prepared)
    target_positions = _extend_position_ids(
        target_prepared.position_ids,
        int(target_prepared.input_ids.shape[1]) + int(max_new_tokens) + block_size + 2,
        target_prepared.input_ids.device,
    )
    draft_positions = _extend_position_ids(
        draft_prepared.position_ids,
        int(draft_prepared.input_ids.shape[1]) + int(max_new_tokens) + block_size + 2,
        draft_prepared.input_ids.device,
    )
    target_prompt = int(target_prepared.input_ids.shape[1])
    draft_prompt = int(draft_prepared.input_ids.shape[1])
    tokens: list[int] = []
    rounds: list[dict[str, Any]] = []
    decode_start = _clock()

    while len(tokens) < max_new_tokens:
        draft_start_len = _cache_length(draft_cache)
        target_start_len = _cache_length(target_cache)
        proposals: list[int] = []
        for _ in range(min(block_size, max_new_tokens - len(tokens))):
            token_id = int(draft_next.argmax(dim=-1).item())
            proposals.append(token_id)
            token = torch.tensor([[token_id]], dtype=torch.long, device=draft_prepared.input_ids.device)
            cache_len = _cache_length(draft_cache)
            _, draft_cache, draft_next = _step(
                draft,
                token,
                draft_cache,
                draft_positions[..., cache_len : cache_len + 1],
                cache_position=cache_len,
            )

        # Verify sequentially through the same one-token cache transition as
        # the greedy reference.  A batched cached call is not reliable for
        # this Qwen2.5-VL multimodal cache: its logits can depend on later
        # proposal tokens even when an attention mask is supplied.  Sequential
        # verification is slower, but it gives an exact lossless gate.
        verify_cache = target_cache
        verify_next = target_next
        target_predictions: list[int] = []
        for proposal in proposals:
            target_predictions.append(int(verify_next.argmax(dim=-1).item()))
            token = torch.tensor(
                [[int(proposal)]],
                dtype=torch.long,
                device=target_prepared.input_ids.device,
            )
            cache_len = _cache_length(verify_cache)
            _, verify_cache, verify_next = _step(
                target,
                token,
                verify_cache,
                target_positions[..., cache_len : cache_len + 1],
                cache_position=cache_len,
            )
        target_next_before = int(target_next.argmax(dim=-1).item())
        accepted = 0
        while accepted < len(proposals) and proposals[accepted] == int(target_predictions[accepted]):
            accepted += 1

        if accepted < len(proposals):
            emitted = proposals[:accepted] + [int(target_predictions[accepted])]
            prefix = tokens + emitted
            target_cache, target_next = _rebuild_cache(
                target,
                target_prepared,
                prefix,
                target_positions,
            )
            draft_cache, draft_next = _rebuild_cache(
                draft,
                draft_prepared,
                prefix,
                draft_positions,
            )
        else:
            bonus = int(verify_next.argmax(dim=-1).item())
            emitted = proposals + [bonus]
            target_cache = verify_cache
            target_next = verify_next
            if bonus not in stop_ids and len(tokens) + len(emitted) < max_new_tokens:
                cache_len = _cache_length(target_cache)
                bonus_tensor = torch.tensor([[bonus]], dtype=torch.long, device=target_prepared.input_ids.device)
                _, target_cache, target_next = _step(
                    target,
                    bonus_tensor,
                    target_cache,
                    target_positions[..., cache_len : cache_len + 1],
                    cache_position=cache_len,
                )
                d_cache_len = _cache_length(draft_cache)
                _, draft_cache, draft_next = _step(
                    draft,
                    bonus_tensor.to(draft_prepared.input_ids.device),
                    draft_cache,
                    draft_positions[..., d_cache_len : d_cache_len + 1],
                    cache_position=d_cache_len,
                )
            else:
                # The bonus is terminal or the output budget is exhausted;
                # no next-round cache is needed.
                pass

        emitted = emitted[: max_new_tokens - len(tokens)]
        tokens.extend(emitted)
        rounds.append(
            {
                "proposal_count": len(proposals),
                "accepted_proposals": accepted,
                "effective_emitted_tokens": len(emitted),
                "is_partial_block": len(proposals) < block_size,
                "is_terminal": bool(any(token in stop_ids for token in emitted) or len(tokens) >= max_new_tokens),
                "proposal_token_ids": proposals,
                "emitted_token_ids": emitted,
                "debug_target_next_before": target_next_before,
                "debug_target_predictions": target_predictions,
                "debug_target_cache_start": target_start_len,
                "debug_target_cache_after_verify": _cache_length(target_cache),
                "debug_target_next_after": int(target_next.argmax(dim=-1).item()) if target_next is not None else None,
            }
        )
        if any(token in stop_ids for token in emitted):
            break

    decode_s = _clock() - decode_start
    return DecodeResult(
        output_ids=tokens,
        rounds=rounds,
        prefill_s=float(prefill_s),
        decode_s=float(decode_s),
        end_to_end_s=float(_clock() - started),
    )


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _common_row(
    *,
    sample: Any,
    policy: str,
    retention: float,
    target_prepared: Any,
    draft_prepared: Any,
    target_tokens: list[int],
    target_timing: dict[str, float],
    result: DecodeResult,
    processor: Any,
    model_ids: tuple[str, str],
) -> dict[str, Any]:
    prefix = 0
    for left, right in zip(target_tokens, result.output_ids):
        if left != right:
            break
        prefix += 1
    lossless = prefix == len(target_tokens) == len(result.output_ids)
    tokenizer = getattr(processor, "tokenizer", processor)
    prediction = tokenizer.decode(result.output_ids, skip_special_tokens=True).strip()
    return {
        "row_id": f"{sample.sample_id}:{policy}:{retention:g}",
        "status": "ok",
        "dataset": "VDC",
        "sample_id": sample.sample_id,
        "video_name": sample.video_name,
        "question": sample.question,
        "reference": sample.answer,
        "target_model": model_ids[0],
        "draft_model": model_ids[1],
        "decoder": "standard_independent_draft",
        "retention_percentage": float(retention),
        "selection_policy": policy,
        "target_visual_tokens": int(target_prepared.video_positions.numel()),
        "draft_visual_tokens": int(draft_prepared.video_positions.numel()),
        "target_prompt_tokens": int(target_prepared.input_ids.shape[1]),
        "draft_prompt_tokens": int(draft_prepared.input_ids.shape[1]),
        "target_output_ids": target_tokens,
        "speculative_output_ids": result.output_ids,
        "target_output_hash": _hash_tokens(target_tokens),
        "speculative_output_hash": _hash_tokens(result.output_ids),
        "prediction": prediction,
        "lossless": bool(lossless),
        "lossless_prefix_length": int(prefix),
        "accepted_prefix_tokens": int(result.accepted_prefix_tokens),
        "tau_proposal": result.tau_proposal,
        "tau_effective": result.tau_effective,
        "acceptance_rounds": result.rounds,
        "prefill_seconds": result.prefill_s,
        "decode_seconds": result.decode_s,
        "end_to_end_seconds": result.end_to_end_s,
        "target_prefill_seconds": target_timing["prefill_s"],
        "target_end_to_end_seconds": target_timing["end_to_end_s"],
        "decode_speedup": target_timing["decode_s"] / max(result.decode_s, 1e-12),
        "end_to_end_speedup": target_timing["end_to_end_s"] / max(result.end_to_end_s, 1e-12),
    }


def _hash_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:16]


def _write_report(output: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["selection_policy"]), float(row["retention_percentage"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (policy, retention), values in sorted(groups.items(), key=lambda item: (item[0][0], -item[0][1])):
        def mean(name: str) -> float | None:
            xs = [float(row[name]) for row in values if row.get(name) is not None and math.isfinite(float(row[name]))]
            return sum(xs) / len(xs) if xs else None
        summary.append({
            "policy": policy,
            "retention_percentage": retention,
            "samples": len(values),
            "lossless_rate": sum(bool(row["lossless"]) for row in values) / len(values),
            "mean_tau_proposal": mean("tau_proposal"),
            "mean_tau_effective": mean("tau_effective"),
            "mean_decode_speedup": mean("decode_speedup"),
            "mean_end_to_end_speedup": mean("end_to_end_speedup"),
            "mean_draft_visual_tokens": mean("draft_visual_tokens"),
        })
    summary_path = output.with_name("summary.json")
    summary_path.write_text(json.dumps({"config": vars(args), "groups": summary}, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Standard speculative decoding visual-retention study",
        "",
        "This report evaluates an independent Qwen2.5-VL-3B draft against a full Qwen2.5-VL-7B target. The target keeps the full video context; only the draft context is reduced. Exact greedy speculative decoding is required to remain lossless.",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Target: `{args.target_model}`",
        f"- Draft: `{args.draft_model}`",
        f"- Retentions: `{args.retentions}`",
        f"- Policies: `{args.selection_policies}`",
        "",
        "| Policy | Retention | n | Lossless | mean τ proposal | mean τ effective | decode speedup | E2E speedup | draft visual tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['policy']} | {row['retention_percentage']:g}% | {row['samples']} | "
            f"{row['lossless_rate']:.1%} | {row['mean_tau_proposal'] if row['mean_tau_proposal'] is not None else 'n/a'} | "
            f"{row['mean_tau_effective'] if row['mean_tau_effective'] is not None else 'n/a'} | "
            f"{row['mean_decode_speedup'] if row['mean_decode_speedup'] is not None else 'n/a'} | "
            f"{row['mean_end_to_end_speedup'] if row['mean_end_to_end_speedup'] is not None else 'n/a'} | "
            f"{row['mean_draft_visual_tokens'] if row['mean_draft_visual_tokens'] is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "The Sparrow-like claim is supported for this standard independent-draft control only if τ increases reproducibly as retained visual tokens decrease while losslessness remains 100% (or the exact mismatch rate is explicitly reported). A flat or decreasing curve would support the hypothesis that the negative visual-retention effect is specific to MSD rather than generic speculative decoding.",
        "",
        "The comparison with the repository's MSD curve is directional rather than an apples-to-apples absolute comparison: the existing MSD artifact uses Qwen2-VL-7B, whereas this control uses Qwen2.5-VL-7B with an independent Qwen2.5-VL-3B draft.",
    ])
    output.with_name("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = load_vdc_manifest(args.manifest, args.video_root)
    if args.sample_ids:
        wanted = {item.strip() for item in args.sample_ids.split(",") if item.strip()}
        samples = [sample for sample in samples if sample.sample_id in wanted]
    if args.limit is not None:
        samples = samples[: args.limit]
    device = torch.device(args.device)
    processor = build_qwen2vl_video_processor(args.target_model, args.min_pixels, args.max_pixels)
    target = load_qwen_model(args.target_model, device_map=args.device_map, dtype=args.dtype, attn_implementation="eager")
    draft = load_qwen_model(args.draft_model, device_map=args.device_map, dtype=args.dtype, attn_implementation="eager")
    target_device = model_device(target)
    if model_device(draft) != target_device:
        raise RuntimeError(f"target and draft must share one device for this runner: {target_device} vs {model_device(draft)}")
    stop_ids = set(_eos_token_ids(processor, target))
    retentions = _parse_float_list(args.retentions)
    policies = [item.strip() for item in args.selection_policies.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    if args.fresh and output.exists():
        output.unlink()
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] {sample.sample_id}", flush=True)
        try:
            batch = process_video(
                processor,
                sample.resolved_path(args.video_root),
                sample.question,
                args.fps,
                max_pixels=args.max_pixels,
                max_frames=args.num_frames,
            )
            batch = move_batch_to_device(batch, target_device)
            target_prepared = prepare_qwen2vl_prefill(target, batch, target_device)
            draft_full = prepare_qwen2vl_prefill(draft, batch, target_device)
            if not torch.equal(target_prepared.input_ids, draft_full.input_ids):
                raise RuntimeError("target and draft tokenizers produced different input_ids")
            scores = attention_scores(target, batch, processor, target_prepared)
            target_tokens, target_timing = greedy_decode(target, target_prepared, args.max_new_tokens, stop_ids)
            for policy in policies:
                if policy not in {"last_instruction", "all_text", "uniform"}:
                    raise ValueError(f"unknown selection policy: {policy}")
                policy_scores = None if policy == "uniform" else scores[policy]
                for retention in retentions:
                    draft_prepared = compact_prefill(draft_full, draft, retention, policy_scores)
                    result = speculative_decode(
                        target,
                        draft,
                        target_prepared,
                        draft_prepared,
                        max_new_tokens=args.max_new_tokens,
                        block_size=args.block_size,
                        stop_ids=stop_ids,
                    )
                    row = _common_row(
                        sample=sample,
                        policy=policy,
                        retention=retention,
                        target_prepared=target_prepared,
                        draft_prepared=draft_prepared,
                        target_tokens=target_tokens,
                        target_timing=target_timing,
                        result=result,
                        processor=processor,
                        model_ids=(args.target_model, args.draft_model),
                    )
                    row.update({
                        "visual_selection_scores_sha256": hashlib.sha256(json.dumps(policy_scores or []).encode()).hexdigest()[:16],
                        "attention_visual_token_count": int(scores["visual_token_count"]),
                        "query_index": scores["query_index"],
                        "target_input_fingerprint": target_prepared.input_fingerprint,
                        "draft_input_fingerprint": draft_prepared.input_fingerprint,
                        "max_new_tokens": args.max_new_tokens,
                        "block_size": args.block_size,
                        "status": "ok" if row["lossless"] else "lossless_mismatch",
                    })
                    rows.append(row)
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(
                        f"  {policy} {retention:g}%: visual={row['draft_visual_tokens']} "
                        f"tau={row['tau_effective']} lossless={row['lossless']}",
                        flush=True,
                    )
        except Exception as exc:  # keep paired coverage visible
            row = {
                "row_id": f"{sample.sample_id}:error",
                "status": "error",
                "dataset": "VDC",
                "sample_id": sample.sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  ERROR: {row['error']}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _write_report(output, rows, args)
    ok = sum(row.get("status") in {"ok", "lossless_mismatch"} for row in rows)
    print(json.dumps({"STANDARD_SD_SUMMARY": {"rows": len(rows), "completed": ok, "errors": len(rows) - ok}}), flush=True)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Optional comma-separated sample IDs to run after loading the manifest.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--retentions", default="0,1,5,10,25,50,75,100")
    parser.add_argument("--selection-policies", default="last_instruction,all_text")
    parser.add_argument("--fresh", action="store_true")
    parser.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
