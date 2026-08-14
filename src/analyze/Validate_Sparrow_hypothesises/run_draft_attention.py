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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import torch

from .dataset import load_vdc_manifest, write_jsonl
from .metrics import normalized_entropy
from .model_analysis import find_instruction_masks
from .paper_contract import DEFAULT_CONTRACT
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    load_msd_qwen2vl,
    model_device,
    move_batch_to_device,
    patched_msd_video_path,
    prepare_qwen2vl_prefill,
    process_video,
    require_cuda,
)
from .run_attention import _calibration_jobs, _fingerprint


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

        def wrapped(*args: Any, _index=layer_index, _original=original, **kwargs: Any):
            is_prefill = kwargs.get("past_key_value") is None
            if is_prefill:
                kwargs["output_attentions"] = True
            result = _original(*args, **kwargs)
            if is_prefill:
                # Post-softmax weights: [1, heads, q_len, key_len].
                weights = result[1]
                if weights is not None and weights.shape[2] > max(positions):
                    captured[_index] = weights[0, :, positions, :].float().detach().to("cpu")
            return result

        originals.append((module, original))
        module.forward = wrapped
    try:
        yield captured
    finally:
        for module, original in originals:
            module.forward = original


def run_draft_prefill(model: Any, prepared: Any) -> dict[str, Any]:
    """Run exactly the MSD prefill (target forward + draft tree build).

    Mirrors the initialization of ``EaModel.msdgenerate`` and stops after
    ``initialize_tree``, which is enough to observe the draft's full-context
    attention and avoids the expensive verify loop.
    """

    from eagle.model import ea_model as ea_module

    input_ids = prepared.input_ids.clone()
    with patched_msd_video_path(model, prepared) as capture:
        model.ea_layer.reset_kv()
        past_key_values, _past_key_values_data, _current_length_data = ea_module.initialize_past_key_values(
            model.base_model
        )
        ea_module.reset_tree_mode(model)
        _draft_tokens, _retrieve_indices, _tree_mask, _tree_position_ids, _logits, _hidden, _token = (
            ea_module.initialize_tree(
                input_ids,
                model,
                past_key_values,
                None,
                inputs_embeds=prepared.inputs_embeds,
            )
        )
        del past_key_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return capture


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
) -> list[dict[str, Any]]:
    if not captured:
        raise RuntimeError(f"draft attention capture returned no layers for {sample.sample_id}")
    visual = torch.as_tensor(masks["visual_positions"], dtype=torch.long)
    instruction = torch.as_tensor(masks["instruction_positions"], dtype=torch.long)
    text = torch.as_tensor(masks["text_positions"], dtype=torch.long)
    # Per layer: average over query rows (all_text) -> [heads, key].
    per_layer = [captured[layer].mean(dim=1) if captured[layer].shape[1] > 1 else captured[layer][:, 0, :]
                 for layer in sorted(captured)]
    layer_values = torch.stack(per_layer)  # [layers, heads, key]
    attention = layer_values.mean(dim=(0, 1))  # mean over layers and heads -> [key]
    visual_mass = float(attention[visual].sum().item())
    instruction_mass = float(attention[instruction].sum().item()) if instruction.numel() else 0.0
    text_mass = float(attention[text].sum().item()) if text.numel() else 0.0
    visual_values = attention[visual].float()
    entropy = normalized_entropy(visual_values.tolist()) if visual_values.numel() > 1 else 0.0
    per_head_visual_mass = [
        float(layer_values[:, head, visual].mean().item()) for head in range(int(layer_values.shape[1]))
    ]
    layer_visual_masses = [
        float(per_layer[index][:, visual].mean().item()) for index in range(len(per_layer))
    ]
    input_ids = prepared.input_ids.detach().to("cpu")
    common = {
        "sample_id": sample.sample_id,
        "target_model": args.base_model,
        "draft_model": args.msd_model,
        "temperature": 0.0,
        "paper_figure": "Figure 2",
        "attention_source": "msd_draft",
        "attention_query": policy,
        "attention_policy": policy,
        "query_position": int(masks["query_index"]) if policy == "last_instruction" else None,
        "query_positions": query_positions,
        "instruction_positions": masks["instruction_positions"],
        "visual_positions": masks["visual_positions"],
        "text_positions": masks["text_positions"],
        "visual_token_count": int(visual.numel()),
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
        "calibration_target_visual_tokens": point.get("target_visual_tokens") if point else None,
        "calibration_status": point.get("status") if point else "not_requested",
        "calibration_relative_error": point.get("relative_error") if point else None,
        "fps": fps,
        "max_pixels": max_pixels,
    }
    rows: list[dict[str, Any]] = []
    for position, weight in enumerate(attention[visual].tolist()):
        row = dict(common)
        row.update({
            "row_id": f"{sample.sample_id}:{visual.numel()}:{policy}:draft:visual:{position}",
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
                "row_id": f"{sample.sample_id}:{visual.numel()}:{policy}:draft:{modality}:{position}",
                "modality": modality,
                "token_position": int(position),
                "attention_weight": float(attention[position].item()),
            })
            rows.append(row)
    summary_row = dict(common)
    summary_row.update({
        "row_id": f"{sample.sample_id}:{visual.numel()}:{policy}:draft:summary",
        "modality": "summary",
        "token_position": int(masks["query_index"]),
        "attention_weight": None,
        "per_head_visual_mass": per_head_visual_mass,
    })
    rows.append(summary_row)
    return rows


def run(args: argparse.Namespace) -> int:
    try:
        require_cuda()
    except RuntimeUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    samples = load_vdc_manifest(args.manifest, args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    targets = list(args.visual_targets or (
        DEFAULT_CONTRACT.attention_short_tokens,
        DEFAULT_CONTRACT.attention_long_tokens,
    ))
    jobs = _calibration_jobs(samples, args.calibration, targets, args.allow_out_of_tolerance)
    processor = build_qwen2vl_video_processor(args.base_model, args.min_pixels, args.max_pixels)
    model = load_msd_qwen2vl(args.base_model, args.msd_model)
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
                sample.question,
                fps,
                max_pixels=max_pixels,
            )
            batch = move_batch_to_device(batch, device)
            prepared = prepare_qwen2vl_prefill(model.base_model, batch, device)
            input_ids = prepared.input_ids.detach().to("cpu")
            masks = find_instruction_masks(input_ids, processor, prepared.video_positions.tolist())
            query_specs = [
                ("last_instruction", [int(masks["query_index"])]),
                ("all_text", sorted(set(masks["instruction_positions"]) | set(masks["text_positions"]))),
            ]
            for policy, query_positions in query_specs:
                if not query_positions:
                    continue
                with capture_draft_query_attention(model, query_positions) as captured:
                    run_draft_prefill(model, prepared)
                rows.extend(_rows_for_policy(
                    sample=sample,
                    args=args,
                    point=point,
                    prepared=prepared,
                    masks=masks,
                    policy=policy,
                    query_positions=query_positions,
                    captured=captured,
                    fps=fps,
                    max_pixels=max_pixels,
                ))




        except Exception as exc:  # noqa: BLE001 - transient video/OOM errors
            print(f"  ERROR {sample.sample_id}: {exc}", flush=True)
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
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-model", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--output", default="results/sparrow_validation/figure2_draft_attention.jsonl")
    parser.add_argument("--calibration")
    parser.add_argument("--visual-targets", type=int, nargs="+")
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--limit", type=int)
    return parser
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
