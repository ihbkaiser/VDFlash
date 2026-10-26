#!/usr/bin/env python3
"""Reproduce the MSD visual-context verification on the author 7B checkpoint.

The target always verifies against the full video. The paired draft conditions
are full video-pad context and zero retained video-pad positions. Target
attention is captured on predicting query rows in a separate teacher-forced
forward, avoiding a dense attention matrix over the entire prompt.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.workspace import resolve_namespace_paths
from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    build_prompt_question,
    load_vdc_manifest,
    write_jsonl,
)
from src.analyze.Validate_Sparrow_hypothesises.model_analysis import (
    _layers,
    capture_query_attention,
)
from src.analyze.Validate_Sparrow_hypothesises.run_msd import _target_reference, _hash_tokens
from src.analyze.Validate_Sparrow_hypothesises.runtime import (
    clear_msd_runtime_state,
    compact_qwen2vl_prefill,
    generate_msd_full_video,
    generate_msd_retention_video,
    load_msd_qwen2vl,
    model_device,
    move_batch_to_device,
    prepare_qwen2vl_prefill,
    process_video,
    require_cuda,
    validate_native_prefill_parity,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def _token_text(processor: Any, token_id: int) -> str:
    return str(processor.tokenizer.decode([int(token_id)], skip_special_tokens=True))


def _capture_target_visual_attention(
    base_model: Any,
    batch: Any,
    prepared: Any,
    processor: Any,
    target_tokens: list[int],
) -> tuple[list[dict[str, Any]], int]:
    """Capture causal target attention for the query predicting each token."""

    prompt_length = int(prepared.input_ids.shape[1])
    token_count = len(target_tokens)
    full_ids = torch.cat(
        (
            prepared.input_ids,
            torch.as_tensor(target_tokens, dtype=prepared.input_ids.dtype, device=prepared.input_ids.device)[None, :],
        ),
        dim=1,
    )
    full_batch = dict(batch)
    full_batch["input_ids"] = full_ids
    full_batch["attention_mask"] = torch.ones_like(full_ids, dtype=torch.bool)
    get_rope_index = getattr(base_model, "get_rope_index", None)
    if get_rope_index is None:
        get_rope_index = getattr(base_model.model, "get_rope_index")
    position_ids, _rope_deltas = get_rope_index(
        input_ids=full_ids,
        image_grid_thw=full_batch.get("image_grid_thw"),
        video_grid_thw=full_batch.get("video_grid_thw"),
        attention_mask=full_batch["attention_mask"],
    )
    full_batch["position_ids"] = position_ids

    # Token i is predicted by the preceding target query, not by its own row.
    query_positions = [prompt_length - 1 + index for index in range(token_count)]
    visual_positions = [int(value) for value in prepared.video_positions.detach().cpu().tolist()]
    if not visual_positions:
        raise RuntimeError("prepared prompt has no video-token positions")
    with capture_query_attention(base_model, query_positions) as captured:
        with torch.inference_mode():
            base_model(
                **full_batch,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
    layer_count = len(_layers(base_model))
    if len(captured) != layer_count:
        raise RuntimeError(f"captured {len(captured)} attention layers; expected {layer_count}")

    per_token_layers: list[list[float]] = [[] for _ in range(token_count)]
    per_token_heads: list[list[list[float]]] = [[] for _ in range(token_count)]
    for layer_index in range(layer_count):
        rows = captured[layer_index]
        if rows.ndim == 2:
            rows = rows.unsqueeze(0)
        visual_index = torch.as_tensor(visual_positions, dtype=torch.long)
        head_mass = rows.index_select(-1, visual_index).sum(dim=-1)
        layer_mass = head_mass.sum(dim=-1)
        for token_index in range(token_count):
            per_token_layers[token_index].append(round(float(layer_mass[token_index]), 6))
            per_token_heads[token_index].append(
                [round(float(value), 6) for value in head_mass[token_index].tolist()]
            )

    token_rows = []
    for index, token_id in enumerate(target_tokens):
        masses = per_token_layers[index]
        token_rows.append({
            "position": index,
            "token_id": int(token_id),
            "token_text": _token_text(processor, token_id),
            "predicting_query_position": int(query_positions[index]),
            "visual_attention_mass_mid_layer": masses[layer_count // 2],
            "visual_attention_mass_all_layers": round(float(sum(masses)), 6),
            "per_layer_visual_attention_mass": masses,
            "per_layer_head_visual_attention_mass": per_token_heads[index],
        })
    return token_rows, layer_count


def _acceptance_by_position(trace: dict[str, Any], target_length: int) -> list[bool | None]:
    """Map each emitted target token to accepted proposal / verifier fallback."""

    result: list[bool | None] = [None] * target_length
    for round_row in trace.get("acceptance_by_position", []):
        for position in round_row.get("accepted_proposal_positions", []):
            position = int(position)
            if 0 <= position < target_length:
                result[position] = True
        fallback = round_row.get("fallback_position")
        if fallback is not None and 0 <= int(fallback) < target_length:
            result[int(fallback)] = False
    return result


def _validate_acceptance_trace(trace: dict[str, Any], output_length: int) -> None:
    """Require round records to account for every emitted token exactly once."""

    cursor = 0
    for round_row in trace.get("acceptance_by_position", []):
        accepted = [int(value) for value in round_row.get("accepted_proposal_positions", [])]
        count = int(round_row.get("accepted_proposal_count", -1))
        expected_accepted = list(range(cursor + 1, cursor + count + 1))
        if int(round_row.get("answer_position_start", -1)) != cursor:
            raise RuntimeError("MSD acceptance trace has a gap or overlap between rounds")
        if int(round_row.get("fallback_position", -1)) != cursor:
            raise RuntimeError("MSD round root token must be recorded as target fallback")
        if count != len(accepted) or accepted != expected_accepted:
            raise RuntimeError("MSD accepted-proposal positions do not match accept_length")
        emitted = int(round_row.get("emitted_token_count", -1))
        if emitted != count + 1:
            raise RuntimeError("MSD round emitted-token count is inconsistent")
        cursor += emitted
    if cursor != output_length:
        raise RuntimeError(
            f"MSD trace covers {cursor} tokens but generated output has {output_length}"
        )


def _aggregate_repeated_arm_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Use median latency while requiring deterministic output and acceptance."""

    if not runs:
        raise ValueError("at least one timed arm run is required")
    identity_fields = (
        "speculative_output_ids",
        "acceptance_by_position",
        "accepted_proposal_by_token",
        "lossless",
        "lossless_prefix_length",
    )
    first = runs[0]
    for repeated in runs[1:]:
        if any(repeated.get(field) != first.get(field) for field in identity_fields):
            raise RuntimeError("repeated MSD runs are not deterministic")
    result = dict(first)
    for field in ("end_to_end_seconds", "decode_seconds", "prefill_seconds", "verification_seconds"):
        values = [float(run[field]) for run in runs if run.get(field) is not None]
        result[field] = float(np.median(values)) if values else None
    result["end_to_end_seconds_by_repeat"] = [
        float(run["end_to_end_seconds"]) for run in runs
    ]
    result["timing_repeats"] = len(runs)
    result["timing_point_estimator"] = "median"
    return result


def _run_arm(
    model: Any,
    prepared: Any,
    target_tokens: list[int],
    condition: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    input_length = int(prepared.input_ids.shape[1])
    if condition == "full":
        draft_context = prepared
        output, trace = generate_msd_full_video(model, draft_context, max_new_tokens)
        retained_visual_tokens = int(prepared.video_positions.numel())
    elif condition == "cut":
        draft_context = compact_qwen2vl_prefill(prepared, 0.0, policy="uniform")
        output, trace = generate_msd_retention_video(
            model, prepared, draft_context, max_new_tokens
        )
        retained_visual_tokens = int(draft_context.video_positions.numel())
    else:  # pragma: no cover - internal caller is fixed
        raise ValueError(f"unknown arm: {condition}")
    speculative_tokens = output[0, input_length:].detach().cpu().tolist()
    if len(speculative_tokens) > max_new_tokens:
        raise RuntimeError("MSD output exceeded the exact max_new_tokens cap")
    _validate_acceptance_trace(trace, len(speculative_tokens))
    if len(speculative_tokens) > max_new_tokens:
        raise RuntimeError("MSD output exceeded the exact max_new_tokens cap")
    common = 0
    for target_id, speculative_id in zip(target_tokens, speculative_tokens):
        if int(target_id) != int(speculative_id):
            break
        common += 1
    lossless = speculative_tokens == target_tokens
    accepted = _acceptance_by_position(trace, len(target_tokens))
    # Do not attribute post-divergence draft positions to the target trajectory.
    if not lossless:
        for position in range(common, len(accepted)):
            accepted[position] = None
    accepted_count = trace.get("accepted_prefix_tokens")
    return {
        "status": "ok",
        "condition": condition,
        "retained_visual_tokens": retained_visual_tokens,
        "speculative_output_ids": [int(value) for value in speculative_tokens],
        "speculative_output_hash": _hash_tokens(speculative_tokens),
        "lossless": bool(lossless),
        "lossless_prefix_length": int(common),
        "accepted_prefix_tokens_per_round": accepted_count,
        "tau_effective": (
            float(accepted_count + 1.0) if accepted_count is not None else None
        ),
        "verification_steps": int(trace.get("verification_steps", 0)),
        "acceptance_by_position": trace.get("acceptance_by_position", []),
        "accepted_proposal_by_token": accepted,
        "prefill_seconds": trace.get("prefill_seconds"),
        "decode_seconds": trace.get("decode_seconds"),
        "end_to_end_seconds": trace.get("end_to_end_seconds"),
        "target_ar_end_to_end_seconds": None,
    }


def _sample_record(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    sample: Any,
) -> dict[str, Any]:
    base_model = model.base_model
    device = model_device(base_model)
    prompt_question = build_prompt_question(sample, "natural")
    batch = process_video(
        processor,
        sample.resolved_path(args.dataset_root),
        prompt_question,
        args.fps,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
    )
    batch = move_batch_to_device(batch, device)
    prepared = prepare_qwen2vl_prefill(base_model, batch, device)
    parity = validate_native_prefill_parity(base_model, batch, device)
    if not parity["valid"]:
        raise RuntimeError(f"native prefill parity failed: {parity}")

    target_runs = [
        _target_reference(
            base_model, batch, int(prepared.input_ids.shape[1]), args.max_new_tokens
        )
        for _ in range(args.timing_repeats)
    ]
    target_tokens, _, _ = target_runs[0]
    if any(tokens != target_tokens for tokens, _, _ in target_runs[1:]):
        raise RuntimeError("repeated target greedy runs are not deterministic")
    ar_prefill = float(np.median([prefill for _, prefill, _ in target_runs]))
    ar_e2e = float(np.median([elapsed for _, _, elapsed in target_runs]))
    attention_rows, layer_count = _capture_target_visual_attention(
        base_model, batch, prepared, processor, target_tokens
    )
    arm_runs: dict[str, list[dict[str, Any]]] = {"full": [], "cut": []}
    full_first = sum(sample.sample_id.encode("utf-8")) % 2 == 0
    first_order = ("full", "cut") if full_first else ("cut", "full")
    for repetition in range(args.timing_repeats):
        condition_order = first_order if repetition % 2 == 0 else tuple(reversed(first_order))
        for condition in condition_order:
            clear_msd_runtime_state(model)
            arm_runs[condition].append(
                _run_arm(model, prepared, target_tokens, condition, args.max_new_tokens)
            )
    arms = {
        condition: _aggregate_repeated_arm_runs(runs)
        for condition, runs in arm_runs.items()
    }
    for condition in ("full", "cut"):
        arms[condition]["target_ar_end_to_end_seconds"] = ar_e2e
        arms[condition]["target_ar_prefill_seconds"] = ar_prefill
    return {
        "sample_id": sample.sample_id,
        "duration_sec": sample.duration_sec,
        "planned_cohort_size": int(args.limit),
        "status": "ok",
        "target_model": args.target_model_id or args.base_model,
        "target_model_path": args.base_model,
        "target_model_revision": args.target_model_revision,
        "msd_checkpoint": args.msd_model_id or args.msd_model,
        "msd_checkpoint_path": args.msd_model,
        "msd_checkpoint_revision": args.msd_model_revision,
        "target_layer_count": layer_count,
        "attention_mid_layer_index_zero_based": layer_count // 2,
        "prompt_question": prompt_question,
        "prompt_tokens": int(prepared.input_ids.shape[1]),
        "visual_tokens": int(prepared.video_positions.numel()),
        "full_context_tokens": int(prepared.input_ids.shape[1]),
        "max_new_tokens": int(args.max_new_tokens),
        "timing_repeats_per_arm": int(args.timing_repeats),
        "target_ar_timing_repeats": int(args.timing_repeats),
        "msd_inference_mode": "author_use_msd_true",
        "draft_condition_interpretation": "raw_video_slot_cut; target hidden states remain full-video-conditioned",
        "attention_capture": "teacher-forced target pass; query index prompt_length-1+output_position",
        "target_output_ids": [int(value) for value in target_tokens],
        "target_output_token_count": len(target_tokens),
        "target_output_hash": _hash_tokens(target_tokens),
        "target_output_text": str(processor.tokenizer.decode(target_tokens, skip_special_tokens=True)),
        "native_prefill_parity": parity,
        "target_visual_attention": attention_rows,
        "arms": arms,
    }


def _bootstrap_mean(values: list[float], rng: np.random.Generator, n_boot: int = 30000) -> list[float]:
    if not values:
        return [None, None, None]
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size == 1:
        return [mean, mean, mean]
    draws = rng.choice(array, size=(n_boot, array.size), replace=True).mean(axis=1)
    return [mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict[str, Any]], output: Path, seed: int) -> dict[str, Any]:
    valid = [record for record in records if record.get("status") == "ok"]
    if not valid:
        return {"sample_count": 0, "status": "no_successful_samples"}
    token_rows = [token for record in valid for token in record["target_visual_attention"]]
    scores = np.asarray(
        [row["visual_attention_mass_mid_layer"] for row in token_rows], dtype=np.float64
    )
    p50, p75, p90 = [float(np.quantile(scores, q)) for q in (0.5, 0.75, 0.9)]
    category_for_score = lambda value: (
        "LOW_VISUAL" if value < p50 else
        "MEDIUM_VISUAL" if value < p75 else
        "HEAVY_VISUAL" if value < p90 else
        "ULTRA_HEAVY"
    )
    for row in token_rows:
        row["attention_category"] = category_for_score(float(row["visual_attention_mass_mid_layer"]))

    categories = ("LOW_VISUAL", "MEDIUM_VISUAL", "HEAVY_VISUAL", "ULTRA_HEAVY")
    strict_valid = [
        record for record in valid
        if record["arms"]["full"]["lossless"] and record["arms"]["cut"]["lossless"]
    ]
    category_summary: dict[str, Any] = {}
    for category in (*categories, "ALL_HEAVY_VISUAL_TOP25"):
        category_positions_by_sample: dict[str, set[int]] = {}
        for record in strict_valid:
            category_positions_by_sample[record["sample_id"]] = {
                row["position"] for row in record["target_visual_attention"]
                if (
                    float(row["visual_attention_mass_mid_layer"]) >= p75
                    if category == "ALL_HEAVY_VISUAL_TOP25"
                    else row["attention_category"] == category
                )
            }
        selected = [
            row for record in strict_valid for row in record["target_visual_attention"]
            if row["position"] in category_positions_by_sample[record["sample_id"]]
        ]
        category_summary[category] = {"token_count": len(selected)}
        for arm in ("full", "cut"):
            values = [
                bool(record["arms"][arm]["accepted_proposal_by_token"][position])
                for record in strict_valid
                for position in category_positions_by_sample[record["sample_id"]]
                if position < len(record["arms"][arm]["accepted_proposal_by_token"])
                and record["arms"][arm]["accepted_proposal_by_token"][position] is not None
            ]
            category_summary[category][f"{arm}_acceptance_pct"] = (
                100.0 * float(np.mean(values)) if values else None
            )

    rng = np.random.default_rng(seed)
    per_video_heavy_delta: list[float] = []
    per_video_all_delta: list[float] = []
    per_video_tau_delta: list[float] = []
    per_video_e2e_ratio_cut: list[float] = []
    per_video_e2e_ratio_full: list[float] = []
    per_video_e2e_ratio_full_over_cut: list[float] = []
    heavy_full_pooled: list[bool] = []
    heavy_cut_pooled: list[bool] = []
    all_full_pooled: list[bool] = []
    all_cut_pooled: list[bool] = []
    for record in valid:
        token_by_pos = {row["position"]: row for row in record["target_visual_attention"]}
        heavy_positions = {
            position for position, row in token_by_pos.items()
            if float(row["visual_attention_mass_mid_layer"]) >= p75
        }
        rates: dict[str, float | None] = {}
        all_rates: dict[str, float | None] = {}
        for arm in ("full", "cut"):
            result = record["arms"][arm]
            accepted = result["accepted_proposal_by_token"]
            values = [accepted[position] for position in heavy_positions if position < len(accepted) and accepted[position] is not None]
            all_values = [value for value in accepted if value is not None]
            rates[arm] = float(np.mean(values)) if values else None
            all_rates[arm] = float(np.mean(all_values)) if all_values else None
        if rates["full"] is not None and rates["cut"] is not None:
            if record in strict_valid:
                per_video_heavy_delta.append(rates["cut"] - rates["full"])
                heavy_full_pooled.extend(
                    bool(record["arms"]["full"]["accepted_proposal_by_token"][position])
                    for position in heavy_positions
                    if record["arms"]["full"]["accepted_proposal_by_token"][position] is not None
                )
                heavy_cut_pooled.extend(
                    bool(record["arms"]["cut"]["accepted_proposal_by_token"][position])
                    for position in heavy_positions
                    if record["arms"]["cut"]["accepted_proposal_by_token"][position] is not None
                )
        if record in strict_valid:
            if all_rates["full"] is not None and all_rates["cut"] is not None:
                per_video_all_delta.append(all_rates["cut"] - all_rates["full"])
            for arm, pooled in (("full", all_full_pooled), ("cut", all_cut_pooled)):
                pooled.extend(bool(value) for value in record["arms"][arm]["accepted_proposal_by_token"] if value is not None)
            full_tau = record["arms"]["full"]["tau_effective"]
            cut_tau = record["arms"]["cut"]["tau_effective"]
            if full_tau is not None and cut_tau is not None:
                per_video_tau_delta.append(float(cut_tau) - float(full_tau))
            for arm, ratios in (("full", per_video_e2e_ratio_full), ("cut", per_video_e2e_ratio_cut)):
                ar_time = record["arms"][arm]["target_ar_end_to_end_seconds"]
                msd_time = record["arms"][arm]["end_to_end_seconds"]
                if ar_time and msd_time:
                    ratios.append(float(ar_time) / float(msd_time))
            full_time = record["arms"]["full"]["end_to_end_seconds"]
            cut_time = record["arms"]["cut"]["end_to_end_seconds"]
            if full_time and cut_time:
                per_video_e2e_ratio_full_over_cut.append(float(full_time) / float(cut_time))

    heavy_delta_ci = _bootstrap_mean(per_video_heavy_delta, rng)
    all_delta_ci = _bootstrap_mean(per_video_all_delta, rng)
    tau_delta_ci = _bootstrap_mean(per_video_tau_delta, rng)
    speed_full_ci = _bootstrap_mean(per_video_e2e_ratio_full, rng)
    speed_cut_ci = _bootstrap_mean(per_video_e2e_ratio_cut, rng)
    full_over_cut_ci = _bootstrap_mean(per_video_e2e_ratio_full_over_cut, rng)
    planned_count = int(valid[0].get("planned_cohort_size", len(valid)))
    exact_pair_count = sum(
        record["arms"]["full"]["lossless"]
        and record["arms"]["cut"]["lossless"]
        and len(record["arms"]["full"]["speculative_output_ids"])
        == len(record["target_output_ids"])
        and len(record["arms"]["cut"]["speculative_output_ids"])
        == len(record["target_output_ids"])
        for record in valid
    )
    success_gate_met = (
        len(valid) == planned_count
        and exact_pair_count == planned_count
        and heavy_delta_ci[1] is not None
        and heavy_delta_ci[1] > 0.0
    )
    summary = {
        "status": "ok",
        "target_model": valid[0]["target_model"],
        "target_model_revision": valid[0].get("target_model_revision"),
        "msd_checkpoint": valid[0]["msd_checkpoint"],
        "msd_checkpoint_revision": valid[0].get("msd_checkpoint_revision"),
        "layer_count": valid[0]["target_layer_count"],
        "attention_mid_layer_index_zero_based": valid[0]["attention_mid_layer_index_zero_based"],
        "sample_count": len(valid),
        "planned_cohort_size": planned_count,
        "exact_target_output_pair_count": exact_pair_count,
        "success_gate_met": success_gate_met,
        "max_new_tokens": int(valid[0].get("max_new_tokens", 0)),
        "timing_repeats_per_arm": int(valid[0].get("timing_repeats_per_arm", 1)),
        "target_ar_timing_repeats": int(valid[0].get("target_ar_timing_repeats", 1)),
        "timing_point_estimator": "median of repeated paired arm decodes",
        "msd_inference_mode": valid[0].get("msd_inference_mode"),
        "draft_condition_interpretation": valid[0].get("draft_condition_interpretation"),
        "attention_capture": valid[0].get("attention_capture"),
        "strict_lossless_pair_count": len(strict_valid),
        "output_token_count": len(token_rows),
        "visual_token_count_range": [min(row["visual_tokens"] for row in valid), max(row["visual_tokens"] for row in valid)],
        "attention_thresholds": {"p50": p50, "p75": p75, "p90": p90},
        "category_summary": category_summary,
        "proposal_acceptance_pct_pooled": {
            "full": 100.0 * float(np.mean(all_full_pooled)) if all_full_pooled else None,
            "cut": 100.0 * float(np.mean(all_cut_pooled)) if all_cut_pooled else None,
        },
        "heavy_top_quartile_acceptance_pct_pooled": {
            "full": 100.0 * float(np.mean(heavy_full_pooled)) if heavy_full_pooled else None,
            "cut": 100.0 * float(np.mean(heavy_cut_pooled)) if heavy_cut_pooled else None,
        },
        "paired_video_bootstrap": {
            "heavy_top_quartile_cut_minus_full_rate": heavy_delta_ci,
            "all_output_cut_minus_full_rate": all_delta_ci,
            "tau_effective_cut_minus_full": tau_delta_ci,
            "end_to_end_speedup_vs_ar_full": speed_full_ci,
            "end_to_end_speedup_vs_ar_cut": speed_cut_ci,
            "end_to_end_time_ratio_full_over_cut": full_over_cut_ci,
            "bootstrap_seed": int(seed),
            "bootstrap_samples": 30000,
            "unit": "video sample; strict-lossless pairs only",
        },
        "lossless_sample_count": {
            "full": sum(bool(row["arms"]["full"]["lossless"]) for row in valid),
            "cut": sum(bool(row["arms"]["cut"]["lossless"]) for row in valid),
        },
        "interpretation_limit": "Acceptance measures actual MSD proposals accepted by the Qwen2-VL verifier; it is not answer accuracy. The Cut arm removes raw video-pad slots from the draft sequence but retains full-video-conditioned target hidden states at the text positions it keeps.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--base-model", default="/path/to/local-resource")
    parser.add_argument("--msd-model", default="/path/to/local-resource")
    parser.add_argument("--target-model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--target-model-revision", default=None)
    parser.add_argument("--msd-model-id", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--msd-model-revision", default=None)
    parser.add_argument("--output", default="results/22_10_26/msd_qwen2vl7b_author_visual_verification_colab_corrected.jsonl")
    parser.add_argument("--summary", default="results/22_10_26/msd_qwen2vl7b_author_visual_verification_colab_corrected_summary.json")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="model_parallel")
    parser.add_argument("--max-memory", default="0:20GiB,1:14GiB")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--fresh", action="store_true", help="start a new output file instead of resuming")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timing_repeats < 1:
        raise ValueError("--timing-repeats must be at least 1")
    resolve_namespace_paths(args, "manifest", "dataset_root", "output", "summary")
    args.base_model = str(Path(args.base_model).expanduser().resolve())
    args.msd_model = str(Path(args.msd_model).expanduser().resolve())
    require_cuda()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    existing = [] if args.fresh else _read_jsonl(output_path)
    done = {str(row.get("sample_id")) for row in existing if row.get("status") == "ok"}
    if args.fresh and output_path.exists():
        output_path.unlink()
    samples = load_vdc_manifest(args.manifest, args.dataset_root)[: args.limit]
    pending = [sample for sample in samples if sample.sample_id not in done]
    print(f"Target: {args.base_model}", flush=True)
    print(f"Author MSD checkpoint: {args.msd_model}", flush=True)
    print(f"Cohort: {len(samples)} videos; pending {len(pending)}; cap={args.max_new_tokens}", flush=True)
    print(f"Preprocessing: max_frames={args.max_frames}; max_pixels={args.max_pixels}", flush=True)
    processor = None
    model = None
    try:
        from src.analyze.Validate_Sparrow_hypothesises.runtime import build_qwen2vl_video_processor

        processor = build_qwen2vl_video_processor(args.base_model, args.min_pixels, args.max_pixels)
        model = load_msd_qwen2vl(
            args.base_model,
            args.msd_model,
            device_map=args.device_map,
            max_memory=args.max_memory,
        )
        records = list(existing)
        for index, sample in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {sample.sample_id}", flush=True)
            start = time.perf_counter()
            try:
                row = _sample_record(args, model, processor, sample)
                row["wall_seconds"] = time.perf_counter() - start
                print(
                    f"  tokens={len(row['target_output_ids'])}; prompt={row['prompt_tokens']}; "
                    f"visual={row['visual_tokens']}; lossless="
                    f"{row['arms']['full']['lossless']}/{row['arms']['cut']['lossless']}; "
                    f"tau={row['arms']['full']['tau_effective']}/"
                    f"{row['arms']['cut']['tau_effective']}",
                    flush=True,
                )
            except Exception as exc:  # preserve a row-level failure and continue
                row = {
                    "sample_id": sample.sample_id,
                    "duration_sec": sample.duration_sec,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.perf_counter() - start,
                }
                print(f"  ERROR {row['error']}", flush=True)
                if torch.cuda.is_available():
                    clear_msd_runtime_state(model)
                    torch.cuda.empty_cache()
                gc.collect()
            records = [old for old in records if old.get("sample_id") != sample.sample_id]
            records.append(row)
            write_jsonl(output_path, records)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = summarize(records, summary_path, args.seed)
        print(json.dumps({"output": str(output_path), "summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))
    finally:
        if model is not None:
            clear_msd_runtime_state(model)
            del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
