"""Run Figure 2's attention-dilution probe on the MSD *draft* model.

The Sparrow paper attributes attention dilution to the limited attention
capacity of the draft model ("attention resources are involuntarily scattered
across irrelevant details").  ``run_attention.py`` measures the target model's
attention as a proxy; this runner captures the attention that the MSD draft
model actually computes during its full-context prefill (the first ``ea_layer``
forward inside ``topK_genrate``, which is the only draft forward with an empty
KV cache).

Rows use the same Figure 2 schema as ``run_attention.py`` and additionally
carry ``attention_source="msd_draft"`` so audit/report/plots can keep the
target-proxy and draft measurements separate.
"""

from __future__ import annotations

import argparse
import gc
import os
import traceback
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch

from src.workspace import resolve_namespace_paths

from .dataset import (
    build_prompt_question,
    load_mvbench_manifest,
    load_vdc_manifest,
    write_jsonl,
)
from .metrics import normalized_entropy
from .model_analysis import find_instruction_masks
from .paper_contract import load_contract
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    compact_qwen2vl_prefill,
    load_msd_qwen2vl,
    model_device,
    move_batch_to_device,
    patched_msd_video_path,
    prepare_qwen2vl_prefill,
    process_video,
    require_cuda,
    zero_msd_draft_visual_values,
)
from .run_attention import _calibration_jobs, _fingerprint


def _query_attention_weights(
    module: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    query_positions: Sequence[int],
) -> torch.Tensor:
    """Compute only selected draft attention rows.

    The vendored eager EAGLE attention normally creates ``[S, S]`` weights
    even when the caller needs one final-instruction query.  At 25K visual
    tokens that tensor is tens of GiB.  This compact path computes Q/K and
    softmax only for the requested rows; the actual draft forward remains
    SDPA-backed and therefore has the same model output.
    """

    from eagle.model import ea_qwen2vl_model as ea_module

    batch, sequence_length, _ = hidden_states.shape
    query = module.q_proj(hidden_states).view(
        batch, sequence_length, -1, module.head_dim
    ).transpose(1, 2)
    key = module.k_proj(hidden_states).view(
        batch, sequence_length, -1, module.head_dim
    ).transpose(1, 2)
    if position_embeddings is None:
        if position_ids is None:
            raise ValueError("draft attention requires position IDs for rotary embeddings")
        position_embeddings = module.rotary_emb(hidden_states, position_ids)
    query, key = ea_module.apply_multimodal_rotary_pos_emb(
        query,
        key,
        position_embeddings[0],
        position_embeddings[1],
        module.rope_scaling["mrope_section"],
    )
    key = ea_module.repeat_kv(key, module.num_key_value_groups)
    selected = [int(value) for value in query_positions]
    if not selected or min(selected) < 0 or max(selected) >= sequence_length:
        raise IndexError("draft attention query position is outside the prefill sequence")
    scores = torch.matmul(query[:, :, selected, :], key.transpose(-1, -2))
    scores = scores / (float(module.head_dim) ** 0.5)
    key_length = scores.shape[-1]
    if attention_mask is not None:
        mask = attention_mask[..., :key_length]
        if mask.ndim == 4:
            mask = mask[:, :, selected, :]
        elif mask.ndim == 3:
            mask = mask[:, selected, :].unsqueeze(1)
        elif mask.ndim == 2:
            mask = mask[:, None, None, :]
        else:
            raise ValueError(f"unsupported draft attention mask rank: {mask.ndim}")
        scores = scores + mask.to(scores.dtype)
    else:
        causal = torch.arange(key_length, device=scores.device)[None, :] > torch.as_tensor(
            selected, device=scores.device
        )[:, None]
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
    return torch.softmax(scores.float(), dim=-1)[0].detach().to("cpu")


def _strict_preceding_attention(
    captured: torch.Tensor,
    query_positions: Sequence[int],
) -> torch.Tensor:
    """Keep and renormalize only keys strictly preceding each query."""
    if captured.ndim != 3:
        raise ValueError("captured attention must have shape [heads, queries, keys]")
    if captured.shape[1] != len(query_positions):
        raise ValueError("query_positions must match the captured query dimension")
    result = torch.zeros_like(captured)
    key_length = int(captured.shape[-1])
    for query_slot, query_position in enumerate(query_positions):
        key_end = min(max(int(query_position), 0), key_length)
        if key_end == 0:
            continue
        preceding = torch.nan_to_num(
            captured[:, query_slot, :key_end],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalizer = preceding.sum(dim=-1, keepdim=True)
        normalized = torch.where(
            normalizer > 0,
            preceding / normalizer.clamp_min(torch.finfo(preceding.dtype).tiny),
            torch.zeros_like(preceding),
        )
        result[:, query_slot, :key_end] = normalized
    return result


def _logit_kl(reference_logits: torch.Tensor, variant_logits: torch.Tensor) -> float:
    """Return ``KL(reference || variant)`` for one next-token distribution."""

    # Draft logits can be extremely peaked.  Computing the probability-weighted
    # difference in float32 can underflow to an apparent zero even when the
    # logits differ; use float64 for the diagnostic only.
    reference = reference_logits.detach().double()
    variant = variant_logits.detach().double()
    if reference.shape != variant.shape:
        raise ValueError("reference and variant logits must have identical shapes")
    reference_log_prob = torch.log_softmax(reference, dim=-1)
    variant_log_prob = torch.log_softmax(variant, dim=-1)
    reference_prob = reference_log_prob.exp()
    return float((reference_prob * (reference_log_prob - variant_log_prob)).sum().item())


def _attention_density_metrics(
    *,
    attention: torch.Tensor,
    groups: dict[str, torch.Tensor],
    query_positions: Sequence[int],
) -> dict[str, float]:
    """Compute per-eligible-key attention density for an averaged trace.

    ``attention`` is already averaged over layers, heads and (for the
    ``all_text`` policy) query rows.  A raw modality mass therefore increases
    mechanically when a modality owns more keys.  The effective key count is
    the mean number of strict-preceding keys available to that modality over
    the query rows.  The density ratio compares the resulting density with
    uniform attention over all strict-preceding keys.

    The helper deliberately returns zero for an empty modality.  This makes
    Deleted-visual rows aggregatable while the accompanying zero count keeps
    the interpretation explicit.
    """

    if attention.ndim != 1:
        raise ValueError("attention must be a one-dimensional averaged trace")
    if not query_positions:
        raise ValueError("query_positions must be non-empty")
    key_length = int(attention.numel())
    finite_attention = torch.nan_to_num(
        attention.detach().to(dtype=torch.float64, device="cpu"),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    preceding_counts = [
        float(min(max(int(position), 0), key_length)) for position in query_positions
    ]
    metrics: dict[str, float] = {}
    for modality in ("visual", "text", "instruction"):
        positions = groups.get(modality, torch.empty(0, dtype=torch.long))
        positions = torch.as_tensor(positions, dtype=torch.long, device="cpu").flatten()
        valid_positions = positions[(positions >= 0) & (positions < key_length)]
        mass = float(finite_attention[valid_positions].sum().item()) if valid_positions.numel() else 0.0
        eligible_counts = []
        for query_position in query_positions:
            eligible_counts.append(
                float(
                    sum(
                        1
                        for position in valid_positions.tolist()
                        if int(position) < int(query_position)
                    )
                )
            )
        effective_count = sum(eligible_counts) / len(eligible_counts)
        density = mass / effective_count if effective_count > 0 else 0.0
        # The average trace is an average over query rows.  The corresponding
        # uniform per-key reference must therefore first average each row's
        # *mass* (eligible_keys / preceding_keys), then divide by the average
        # eligible-key count.  Conditioning only on rows where this modality
        # is eligible would incorrectly renormalize the averaged trace.
        uniform_mass = sum(
            eligible / preceding
            for eligible, preceding in zip(eligible_counts, preceding_counts)
            if preceding > 0
        ) / len(preceding_counts)
        uniform_density = uniform_mass / effective_count if effective_count > 0 else 0.0
        ratio = density / uniform_density if uniform_density > 0 else 0.0
        metrics[f"{modality}_effective_key_count"] = effective_count
        metrics[f"{modality}_attention_density"] = density
        metrics[f"{modality}_uniform_attention_density"] = uniform_density
        metrics[f"{modality}_density_ratio"] = ratio
    return metrics


def _remap_instruction_masks(
    masks: dict[str, Any],
    keep_mask: Sequence[bool] | torch.Tensor,
) -> dict[str, Any]:
    """Remap full-context modality positions into a compacted context.

    Recomputing masks after deleting all visual tokens would classify the
    remaining user question as ``instruction``.  This helper preserves the
    original text/instruction roles and only changes token indices.
    """

    keep = torch.as_tensor(keep_mask, dtype=torch.bool).to("cpu")
    kept_indices = torch.nonzero(keep, as_tuple=False).flatten().tolist()
    position_map = {int(old): index for index, old in enumerate(kept_indices)}

    def remap(values: Sequence[int]) -> list[int]:
        return [position_map[int(value)] for value in values if int(value) in position_map]

    query_index = int(masks["query_index"])
    if query_index not in position_map:
        raise ValueError("query_index was removed by the visual-context intervention")
    marker = masks.get("assistant_marker_start")
    return {
        **masks,
        "visual_positions": remap(masks.get("visual_positions", [])),
        "instruction_positions": remap(masks.get("instruction_positions", [])),
        "text_positions": remap(masks.get("text_positions", [])),
        "query_index": position_map[query_index],
        "assistant_marker_start": position_map.get(int(marker)) if marker is not None else None,
    }


@contextmanager
def capture_draft_query_attention(
    model: Any,
    query_positions: Sequence[int],
):
    """Capture one attention row per draft layer during the draft prefill.

    The MSD draft is invoked by ``topK_genrate``; the first call has no KV
    cache and processes the full context.  Later calls (tree expansion and
    verification) pass ``past_key_value`` and are skipped so the captured row
    always describes the full-context draft attention.
    """

    layers = model.ea_layer.layers
    captured: dict[int, torch.Tensor] = {}
    originals: list[tuple[Any, Any]] = []
    positions = [int(value) for value in query_positions]
    if not positions:
        raise ValueError("query_positions must be non-empty")
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        original = module.forward

        def wrapped(*args: Any, _module=module, _index=layer_index, _original=original, **kwargs: Any):
            past_key_value = kwargs.get("past_key_value", args[3] if len(args) > 3 else None)
            is_prefill = past_key_value is None
            if is_prefill:
                hidden_states = kwargs.get("hidden_states", args[0] if args else None)
                if hidden_states is None:
                    raise ValueError("draft attention hook received no hidden states")
                captured[_index] = _query_attention_weights(
                    _module,
                    hidden_states,
                    kwargs.get("attention_mask", args[1] if len(args) > 1 else None),
                    kwargs.get("position_ids", args[2] if len(args) > 2 else None),
                    kwargs.get("position_embeddings"),
                    positions,
                )
                # Keep the underlying runtime on its memory-efficient SDPA
                # path; requesting output_attentions would reintroduce S².
                kwargs["output_attentions"] = False
            result = _original(*args, **kwargs)
            return result

        originals.append((module, original))
        module.forward = wrapped
    try:
        yield captured
    finally:
        for module, original in originals:
            module.forward = original


def run_draft_prefill(model: Any, prepared: Any) -> tuple[dict[str, Any], torch.Tensor]:
    """Run exactly the MSD prefill (target forward + draft tree build).

    Mirrors the initialization of ``EaModel.msdgenerate`` and stops after
    ``initialize_tree``, which is enough to observe the draft's full-context
    attention and avoids the expensive verify loop.
    """

    from eagle.model import ea_model as ea_module

    input_ids = prepared.input_ids.clone()
    draft_logits: list[torch.Tensor] = []

    def capture_draft_head(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        # The target prefill calls lm_head with [B, S, H].  The first draft
        # prediction in topK_genrate calls the same head with [B, H]; that is
        # the distribution affected by the value ablation.
        hidden_states = inputs[0] if inputs else None
        if (
            not draft_logits
            and isinstance(hidden_states, torch.Tensor)
            and hidden_states.ndim == 2
            and isinstance(output, torch.Tensor)
        ):
            draft_logits.append(output[:, -1].detach().float().to("cpu"))

    hook = model.base_model.lm_head.register_forward_hook(capture_draft_head)
    try:
        with patched_msd_video_path(model, prepared) as capture:
            model.ea_layer.reset_kv()
            past_key_values, _past_key_values_data, _current_length_data = ea_module.initialize_past_key_values(
                model.base_model
            )
            ea_module.reset_tree_mode(model)
            ea_module.initialize_tree(
                input_ids,
                model,
                past_key_values,
                None,
                inputs_embeds=prepared.inputs_embeds,
            )
        del past_key_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        hook.remove()
    if not draft_logits:
        raise RuntimeError("could not capture the initial draft next-token logits")
    return capture, draft_logits[0]


def _rows_for_policy(
    *,
    sample: Any,
    args: argparse.Namespace,
    point: dict[str, Any] | None,
    prepared: Any,
    masks: dict[str, Any],
    policy: str,
    query_positions: list[int],
    captured: dict[int, torch.Tensor],
    fps: float,
    max_pixels: int | None,
    visual_condition: str = "full",
    retention_percentage: float = 100.0,
    visual_value_mode: str = "real",
    value_ablation_kl_forward: float | None = None,
    value_ablation_kl_reverse: float | None = None,
    value_ablation_top1_match: bool | None = None,
    value_ablation_logit_max_abs_delta: float | None = None,
) -> list[dict[str, Any]]:
    if not captured:
        raise RuntimeError(f"draft attention capture returned no layers for {sample.sample_id}")
    visual_positions = [int(value) for value in masks["visual_positions"]]
    instruction_positions = [int(value) for value in masks["instruction_positions"]]
    text_positions = [int(value) for value in masks["text_positions"]]
    if policy == "last_instruction":
        query_position = int(query_positions[0])
        visual_positions = [value for value in visual_positions if value < query_position]
        instruction_positions = [value for value in instruction_positions if value < query_position]
        text_positions = [value for value in text_positions if value < query_position]
    visual = torch.as_tensor(visual_positions, dtype=torch.long)
    instruction = torch.as_tensor(instruction_positions, dtype=torch.long)
    text = torch.as_tensor(text_positions, dtype=torch.long)
    strict_captured = {
        layer: _strict_preceding_attention(values, query_positions)
        for layer, values in captured.items()
    }
    # Per layer: average over query rows (all_text) -> [heads, key].
    per_layer = [strict_captured[layer].mean(dim=1) if strict_captured[layer].shape[1] > 1 else strict_captured[layer][:, 0, :]
                 for layer in sorted(strict_captured)]
    layer_values = torch.stack(per_layer)  # [layers, heads, key]
    attention = layer_values.mean(dim=(0, 1))  # mean over layers and heads -> [key]
    visual_mass = float(attention[visual].sum().item())
    instruction_mass = float(attention[instruction].sum().item()) if instruction.numel() else 0.0
    text_mass = float(attention[text].sum().item()) if text.numel() else 0.0
    visual_values = attention[visual].float()
    entropy = normalized_entropy(visual_values.tolist()) if visual_values.numel() > 1 else 0.0
    density_metrics = _attention_density_metrics(
        attention=attention,
        groups={
            "visual": visual,
            "instruction": instruction,
            "text": text,
        },
        query_positions=query_positions,
    )
    per_head_visual_mass = [
        float(layer_values[:, head, visual].mean().item()) for head in range(int(layer_values.shape[1]))
    ]
    layer_visual_masses = [
        float(per_layer[index][:, visual].mean().item()) for index in range(len(per_layer))
    ]
    input_ids = prepared.input_ids.detach().to("cpu")
    common = {
        "sample_id": sample.sample_id,
        "dataset_kind": getattr(args, "dataset_kind", "vdc"),
        "task": getattr(sample, "task", None),
        "prompt_variant": getattr(args, "prompt_variant", "natural"),
        "visual_value_mode": visual_value_mode,
        "target_model": args.base_model,
        "draft_model": args.msd_model,
        "temperature": 0.0,
        "paper_figure": "Figure 2",
        "attention_source": "msd_draft",
        "attention_query": policy,
        "attention_policy": policy,
        "visual_condition": visual_condition,
        "visual_retention_percentage": float(retention_percentage),
        "attention_key_scope": "strict_preceding",
        "query_position": int(masks["query_index"]) if policy == "last_instruction" else None,
        "query_positions": query_positions,
        "instruction_positions": instruction_positions,
        "visual_positions": visual_positions,
        "text_positions": text_positions,
        "visual_token_count": int(visual.numel()),
        "attention_density_definition": "modality_mass / mean_eligible_strict_preceding_key_count",
        "target_visual_tokens": point.get("target_visual_tokens") if point else int(visual.numel()),
        "actual_visual_tokens": int(visual.numel()),
        "target_input_fingerprint": _fingerprint(input_ids),
        "draft_input_fingerprint": _fingerprint(input_ids),
        "heads": int(layer_values.shape[1]),
        "layers": int(layer_values.shape[0]),
        "layer_visual_masses": layer_visual_masses,
        "instruction_mass": instruction_mass,
        "visual_mass": visual_mass,
        "text_mass": text_mass,
        "visual_entropy": entropy,
        **density_metrics,
        "value_ablation_kl_forward": value_ablation_kl_forward,
        "value_ablation_kl_reverse": value_ablation_kl_reverse,
        "value_ablation_top1_match": value_ablation_top1_match,
        "value_ablation_logit_max_abs_delta": value_ablation_logit_max_abs_delta,
        "value_ablation_logit_scope": "draft_initial_next_token",
        "calibration_target_visual_tokens": point.get("target_visual_tokens") if point else None,
        "calibration_status": point.get("status") if point else "not_requested",
        "calibration_relative_error": point.get("relative_error") if point else None,
        "fps": fps,
        "max_pixels": max_pixels,
        "max_frames": getattr(args, "max_frames", None),
    }
    rows: list[dict[str, Any]] = []
    if os.environ.get("SPARROW_COMPACT_ATTENTION") != "1":
        for position, weight in enumerate(attention[visual].tolist()):
            row = dict(common)
            row.update({
                "row_id": f"{sample.sample_id}:{visual.numel()}:{visual_condition}:{policy}:draft:visual:{position}",
                "modality": "visual",
                "token_position": int(visual[position].item()),
                "visual_index": position,
                "attention_weight": float(weight),
            })
            rows.append(row)
        for modality, positions in (("instruction", instruction), ("text", text)):
            for position in positions.tolist():
                row = dict(common)
                row.update({
                    "row_id": f"{sample.sample_id}:{visual.numel()}:{visual_condition}:{policy}:draft:{modality}:{position}",
                    "modality": modality,
                    "token_position": int(position),
                    "attention_weight": float(attention[position].item()),
                })
                rows.append(row)
    summary_row = dict(common)
    summary_row.update({
        "row_id": f"{sample.sample_id}:{visual.numel()}:{visual_condition}:{policy}:draft:summary",
        "modality": "summary",
        "token_position": int(masks["query_index"]),
        "attention_weight": None,
        "per_head_visual_mass": per_head_visual_mass,
        "record_type": "attention_trace",
        "attention_weights": [float(value) for value in attention.tolist()],
        "visual_attention_weights": [float(value) for value in attention[visual].tolist()],
        "instruction_attention_weights": [float(value) for value in attention[instruction].tolist()],
        "text_attention_weights": [float(value) for value in attention[text].tolist()],
    })
    rows.append(summary_row)
    return rows


def run(args: argparse.Namespace) -> int:
    resolve_namespace_paths(
        args,
        "contract",
        "manifest",
        "dataset_root",
        "output",
        "calibration",
    )
    try:
        require_cuda()
    except RuntimeUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    contract = load_contract(args.contract)
    if args.dataset_kind == "mvbench":
        samples = load_mvbench_manifest(
            args.manifest,
            args.dataset_root,
            limit_per_task=args.limit_per_task,
        )
    else:
        samples = load_vdc_manifest(args.manifest, args.dataset_root)
    start_index = max(0, int(args.start_index or 0))
    end_index = int(args.end_index) if args.end_index is not None else None
    if args.limit is not None:
        end_index = min(start_index + int(args.limit), len(samples))
    samples = samples[start_index:end_index]
    if not samples:
        raise SystemExit("Selected manifest slice is empty")
    targets = list(args.visual_targets or (
        contract.attention_short_tokens,
        contract.attention_long_tokens,
    ))
    jobs = _calibration_jobs(samples, args.calibration, targets, args.allow_out_of_tolerance)
    processor = build_qwen2vl_video_processor(args.base_model, args.min_pixels, args.max_pixels)
    model = load_msd_qwen2vl(
        args.base_model,
        args.msd_model,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )
    device = model_device(model.base_model)
    rows: list[dict[str, Any]] = []
    for index, (sample, point) in enumerate(jobs, start=1):
        try:
            if point and point.get("candidate_settings"):
                settings = point["candidate_settings"]
                fps = float(settings["frames"]) / max(float(sample.duration_sec or 1.0), 1e-3)
                max_pixels = int(settings["max_pixels"])
            else:
                fps = args.fps
                # Bound the native path by the explicit pixel budget (see run_msd).
                max_pixels = args.max_pixels
            print(f"[{index}/{len(jobs)}] {sample.sample_id} target={point.get('target_visual_tokens') if point else 'native'}")
            batch = process_video(
                processor,
                sample.resolved_path(args.dataset_root),
                build_prompt_question(sample, args.prompt_variant),
                fps,
                max_pixels=max_pixels,
                max_frames=args.max_frames,
            )
            batch = move_batch_to_device(batch, device)
            prepared = prepare_qwen2vl_prefill(model.base_model, batch, device)
            input_ids = prepared.input_ids.detach().to("cpu")
            full_masks = find_instruction_masks(input_ids, processor, prepared.video_positions.tolist())
            for visual_condition in args.visual_conditions:
                if visual_condition == "full":
                    draft_prepared = prepared
                    masks = full_masks
                    retention_percentage = 100.0
                else:
                    retention_percentage = 0.0 if visual_condition == "deleted" else args.retention_percentage
                    draft_prepared = compact_qwen2vl_prefill(
                        prepared,
                        retention_percentage,
                    )
                    masks = _remap_instruction_masks(full_masks, draft_prepared.keep_mask)
                query_specs = [
                    ("last_instruction", [int(masks["query_index"])]),
                    ("all_text", sorted(set(masks["instruction_positions"]) | set(masks["text_positions"]))),
                ]
                print(
                    f"  attention condition={visual_condition}"
                    f" retained_visual={draft_prepared.video_positions.numel()}",
                    flush=True,
                )
                for policy, query_positions in query_specs:
                    if not query_positions:
                        continue
                    value_modes = ("real", "zero") if args.compare_value_ablation else (args.visual_value_mode,)
                    value_runs: dict[str, tuple[dict[int, torch.Tensor], torch.Tensor]] = {}
                    for value_mode in value_modes:
                        value_context = (
                            zero_msd_draft_visual_values(
                                model,
                                draft_prepared.video_positions.tolist(),
                            )
                            if value_mode == "zero"
                            else nullcontext()
                        )
                        with value_context:
                            with capture_draft_query_attention(model, query_positions) as captured:
                                _runtime_capture, logits = run_draft_prefill(model, draft_prepared)
                        value_runs[value_mode] = (captured, logits)
                    real_logits = value_runs.get("real", (None, None))[1]
                    zero_logits = value_runs.get("zero", (None, None))[1]
                    kl_forward = _logit_kl(real_logits, zero_logits) if real_logits is not None and zero_logits is not None else None
                    kl_reverse = _logit_kl(zero_logits, real_logits) if real_logits is not None and zero_logits is not None else None
                    top1_match = (
                        bool(real_logits.argmax().item() == zero_logits.argmax().item())
                        if real_logits is not None and zero_logits is not None
                        else None
                    )
                    logit_max_abs_delta = (
                        float((real_logits - zero_logits).abs().max().item())
                        if real_logits is not None and zero_logits is not None
                        else None
                    )
                    for value_mode, (captured, _logits) in value_runs.items():
                        rows.extend(_rows_for_policy(
                            sample=sample,
                            args=args,
                            point=point,
                            prepared=draft_prepared,
                            masks=masks,
                            policy=policy,
                            query_positions=query_positions,
                            captured=captured,
                            fps=fps,
                            max_pixels=max_pixels,
                            visual_condition=visual_condition,
                            retention_percentage=retention_percentage,
                            visual_value_mode=value_mode,
                            value_ablation_kl_forward=kl_forward,
                            value_ablation_kl_reverse=kl_reverse,
                            value_ablation_top1_match=top1_match,
                            value_ablation_logit_max_abs_delta=logit_max_abs_delta,
                        ))




        except Exception as exc:  # noqa: BLE001 - transient video/OOM errors
            print(f"  ERROR {sample.sample_id}: {exc}", flush=True)
            if os.environ.get("HYPOTHESIS_DEBUG_TRACEBACK") == "1":
                traceback.print_exc()
            rows.append({
                "row_id": f"{sample.sample_id}:error",
                "paper_figure": "Figure 1(a)",
                "sample_id": sample.sample_id,
                "target_model": args.base_model,
                "temperature": 0.0,
                "target_visual_tokens": point.get("target_visual_tokens") if point else None,
                "actual_visual_tokens": None,
                "target_input_fingerprint": "unavailable",
                "draft_input_fingerprint": "unavailable",
                "condition": "error",
                "status": "error",
                "error": str(exc),
            })
            continue
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} Figure 2 (draft) rows to {args.output}")
    return 0
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default="src/analyze/Validate_Sparrow_hypothesises/configs/local_insight_vdc50.yaml",
    )
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--dataset-kind", choices=("vdc", "mvbench"), default="vdc")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-model", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--output", default="results/sparrow_validation/figure2_draft_attention.jsonl")
    parser.add_argument("--calibration")
    parser.add_argument("--visual-targets", type=int, nargs="+")
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument(
        "--device-map",
        choices=("cuda", "auto", "model_parallel"),
        default="cuda",
        help="MSD draft placement; use model_parallel on the 3090+A4000 pair.",
    )
    parser.add_argument(
        "--max-memory",
        help="Per-device budgets for sharded placement, for example 0:22GiB,1:14GiB.",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional hard cap on decoded video frames for long-video fallback runs.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--limit-per-task",
        type=int,
        help="For MVBench, select this many records per task before slicing.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--prompt-variant",
        choices=("natural", "answer_hint"),
        default="natural",
        help="Use the natural VDC question or append the reference as an oracle hint.",
    )
    parser.add_argument(
        "--visual-conditions",
        nargs="+",
        choices=("full", "reduced", "deleted"),
        default=["full"],
        help="Attention contexts to compare; reduced uses --retention-percentage.",
    )
    parser.add_argument("--retention-percentage", type=float, default=25.0)
    parser.add_argument(
        "--visual-value-mode",
        choices=("real", "zero"),
        default="real",
        help="Keep real draft visual values or zero value projections at visual positions.",
    )
    parser.add_argument(
        "--compare-value-ablation",
        action="store_true",
        help="Run real and zero visual values in one job and attach KL/top-1 diagnostics.",
    )
    return parser
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
