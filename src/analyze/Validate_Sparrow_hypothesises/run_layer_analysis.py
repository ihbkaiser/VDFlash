"""Run Figure 3 layer ablations and Figure 6 information-retention curves."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from .dataset import load_vdc_manifest, write_jsonl
from .metrics import rouge_l
from .model_analysis import (
    capture_query_attention,
    find_instruction_masks,
    hash_tokens,
    layerwise_input_cosine,
    load_qwen_model,
    mask_visual_keys,
    prepare_qwen25_prefill,
    prefix_length,
    target_generation,
)
from .paper_contract import DEFAULT_CONTRACT
from .run_attention import _calibration_jobs
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    move_batch_to_device,
    model_device,
    process_video,
    require_cuda,
)


def _decode(processor: Any, tokens: list[int]) -> str:
    try:
        return processor.batch_decode(
            [tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except TypeError:
        return processor.batch_decode([tokens], skip_special_tokens=True)[0]


def _base_row(
    sample: Any,
    args: argparse.Namespace,
    prepared: Any,
    point: dict[str, Any] | None,
    fps: float,
    max_pixels: int | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "target_model": args.model,
        "temperature": 0.0,
        "target_visual_tokens": int(prepared.video_positions.numel()),
        "actual_visual_tokens": int(prepared.video_positions.numel()),
        "full_target_visual_tokens": int(prepared.video_positions.numel()),
        "target_input_fingerprint": prepared.input_fingerprint,
        "target_input_fingerprint_reference": prepared.input_fingerprint,
        "draft_input_fingerprint": prepared.input_fingerprint,
        "calibration_target_visual_tokens": point.get("target_visual_tokens") if point else None,
        "calibration_status": point.get("status") if point else "not_requested",
        "calibration_relative_error": point.get("relative_error") if point else None,
        "calibration_candidate_id": point.get("candidate_id") if point else None,
        "fps": fps,
        "max_pixels": max_pixels,
    }


def _run_figure3(
    model: Any,
    processor: Any,
    batch: Any,
    sample: Any,
    args: argparse.Namespace,
    prepared: Any,
    point: dict[str, Any] | None,
    fps: float,
    max_pixels: int | None,
    target_tokens: list[int],
    target_text: str,
    masks: dict[str, Any],
) -> list[dict[str, Any]]:
    layers = len(model.model.language_model.layers)
    cuts = [int(value) for value in args.layer_cut_points]
    if any(cut < 0 or cut > layers for cut in cuts):
        raise SystemExit(f"layer cut points must be in [0, {layers}], got {cuts}")
    input_ids = batch["input_ids"] if isinstance(batch, dict) else batch.input_ids
    input_length = int(input_ids.shape[1])
    rows: list[dict[str, Any]] = []
    for cut in cuts:
        with mask_visual_keys(model, prepared.video_positions.tolist(), cut):
            with torch.inference_mode():
                output = model.generate(
                    **batch,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
        candidate = output[0, input_length:].detach().to("cpu").tolist()
        common = prefix_length(target_tokens, candidate)
        candidate_text = _decode(processor, candidate)
        row = _base_row(sample, args, prepared, point, fps, max_pixels)
        row.update({
            "row_id": f"{sample.sample_id}:{prepared.video_positions.numel()}:layer-cut-{cut}",
            "paper_figure": "Figure 3",
            "layer_cut": cut,
            "visual_kv_masked_from": cut,
            "target_output_ids": target_tokens,
            "speculative_output_ids": candidate,
            "target_output_hash": hash_tokens(target_tokens),
            "speculative_output_hash": hash_tokens(candidate),
            "lossless": common == len(target_tokens),
            "lossless_prefix_length": common,
            "prefix_agreement": common / max(1, len(target_tokens)),
            "target_text": target_text,
            "speculative_text": candidate_text,
            "rouge_l": rouge_l(target_text, candidate_text),
            "condition": "layer_visual_kv_ablation",
        })
        rows.append(row)
    return rows


def _run_figure3_attention(
    model: Any,
    processor: Any,
    batch: Any,
    sample: Any,
    args: argparse.Namespace,
    prepared: Any,
    point: dict[str, Any] | None,
    fps: float,
    max_pixels: int | None,
    masks: dict[str, Any],
) -> list[dict[str, Any]]:
    with capture_query_attention(model, int(masks["query_index"])) as captured:
        with torch.inference_mode():
            model(**batch, use_cache=False, output_attentions=False, return_dict=True)
    rows: list[dict[str, Any]] = []
    visual = torch.as_tensor(masks["visual_positions"], dtype=torch.long)
    for layer, weights in sorted(captured.items()):
        row = _base_row(sample, args, prepared, point, fps, max_pixels)
        row.update({
            "row_id": f"{sample.sample_id}:{prepared.video_positions.numel()}:attention-layer-{layer}",
            "paper_figure": "Figure 3(b)",
            "layer": int(layer),
            "attention_query": "last_instruction",
            "query_position": int(masks["query_index"]),
            "instruction_positions": masks["instruction_positions"],
            "visual_positions": masks["visual_positions"],
            "text_positions": masks["text_positions"],
            "visual_mass": float(weights[:, visual].sum().item()),
            "per_head_visual_mass": [float(value) for value in weights[:, visual].sum(dim=-1).tolist()],
            "attention_heads": int(weights.shape[0]),
            "attention_key_length": int(weights.shape[-1]),
            "condition": "layer_attention_probe",
        })
        rows.append(row)
    if not rows:
        raise RuntimeError(f"attention capture returned no layers for {sample.sample_id}")
    return rows


def _run_figure6(
    model: Any,
    processor: Any,
    batch: Any,
    sample: Any,
    args: argparse.Namespace,
    prepared: Any,
    point: dict[str, Any] | None,
    fps: float,
    max_pixels: int | None,
    masks: dict[str, Any],
) -> list[dict[str, Any]]:
    text_positions = list(masks["instruction_positions"]) + list(masks["text_positions"])
    curves = layerwise_input_cosine(
        model,
        batch,
        prepared,
        masks["visual_positions"],
        text_positions,
    )
    rows: list[dict[str, Any]] = []
    for curve in curves:
        row = _base_row(sample, args, prepared, point, fps, max_pixels)
        row.update({
            "row_id": f"{sample.sample_id}:{prepared.video_positions.numel()}:cosine-layer-{curve['layer']}",
            "paper_figure": "Figure 6 / Appendix D",
            "layer": int(curve["layer"]),
            "visual_cosine": float(curve["visual_cosine"]),
            "text_cosine": float(curve["text_cosine"]),
            "visual_positions": masks["visual_positions"],
            "text_positions": text_positions,
            "condition": "layerwise_input_cosine",
        })
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    try:
        require_cuda()
    except RuntimeUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    samples = load_vdc_manifest(args.manifest, args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    targets = list(args.visual_targets or DEFAULT_CONTRACT.visual_token_milestones)
    jobs = _calibration_jobs(samples, args.calibration, targets, args.allow_out_of_tolerance)
    processor = build_qwen2vl_video_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_qwen_model(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        quantized=args.quantized,
    )
    device = model_device(model)
    rows: list[dict[str, Any]] = []
    for index, (sample, point) in enumerate(jobs, start=1):
        if point and point.get("candidate_settings"):
            settings = point["candidate_settings"]
            fps = float(settings["frames"]) / max(float(sample.duration_sec or 1.0), 1e-3)
            max_pixels = int(settings["max_pixels"])
        else:
            fps = args.fps
            max_pixels = None
        print(f"[{index}/{len(jobs)}] {sample.sample_id} target={point.get('target_visual_tokens') if point else 'native'}")
        batch = process_video(
            processor,
            sample.resolved_path(args.dataset_root),
            sample.question,
            fps,
            max_pixels=max_pixels,
        )
        batch = move_batch_to_device(batch, device)
        prepared = prepare_qwen25_prefill(model, batch, device)
        masks = find_instruction_masks(batch["input_ids"], processor, prepared.video_positions.tolist())
        target_tokens, timing, _target_output = target_generation(model, batch, args.max_new_tokens)
        target_text = _decode(processor, target_tokens)
        if args.experiments in {"figure3", "both"}:
            rows.extend(_run_figure3(
                model, processor, batch, sample, args, prepared, point, fps, max_pixels,
                target_tokens, target_text, masks,
            ))
            rows.extend(_run_figure3_attention(
                model, processor, batch, sample, args, prepared, point, fps, max_pixels, masks,
            ))
        if args.experiments in {"figure6", "both"}:
            rows.extend(_run_figure6(
                model, processor, batch, sample, args, prepared, point, fps, max_pixels, masks,
            ))
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} layer-analysis rows to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", default="results/sparrow_validation/layer_analysis.jsonl")
    parser.add_argument("--experiments", choices=("figure3", "figure6", "both"), default="both")
    parser.add_argument("--calibration")
    parser.add_argument("--visual-targets", type=int, nargs="+")
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument("--layer-cut-points", type=int, nargs="+", default=list(DEFAULT_CONTRACT.layer_cut_points))
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--quantized", action="store_true")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
