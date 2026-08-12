"""Run Figure 2's final-instruction attention-dilution probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .dataset import load_vdc_manifest, qwen2vl_video_token_count, write_jsonl
from .model_analysis import (
    capture_query_attention,
    find_instruction_masks,
    load_qwen_model,
)
from .paper_contract import DEFAULT_CONTRACT
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    move_batch_to_device,
    model_device,
    process_video,
    require_cuda,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _calibration_jobs(
    samples: list[Any],
    calibration_path: str | None,
    targets: list[int],
    allow_out_of_tolerance: bool,
) -> list[tuple[Any, dict[str, Any] | None]]:
    if calibration_path is None:
        return [(sample, None) for sample in samples]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(calibration_path):
        if int(row.get("target_visual_tokens", -1)) in targets:
            grouped.setdefault(str(row["sample_id"]), []).append(row)
    jobs: list[tuple[Any, dict[str, Any]]] = []
    for sample in samples:
        points = grouped.get(sample.sample_id, [])
        if not points:
            raise SystemExit(f"No calibration rows for {sample.sample_id} in {calibration_path}")
        for point in sorted(points, key=lambda row: int(row["target_visual_tokens"])):
            if point.get("status") != "ok" and not allow_out_of_tolerance:
                raise SystemExit(
                    f"Calibration point {sample.sample_id}:{point['target_visual_tokens']} is "
                    f"{point.get('status')}; pass --allow-out-of-tolerance to use it"
                )
            jobs.append((sample, point))
    return jobs


def _grid_count(batch: Any) -> int | None:
    grid = batch.get("video_grid_thw") if isinstance(batch, dict) else getattr(batch, "video_grid_thw", None)
    return qwen2vl_video_token_count(grid.detach().to("cpu").tolist()) if grid is not None else None


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
    processor = build_qwen2vl_video_processor(args.model, args.min_pixels, args.max_pixels)
    model = load_qwen_model(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        quantized=args.quantized,
    )
    device = model_device(model)
    video_token_id = int(model.config.video_token_id)
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
        input_ids = batch["input_ids"] if isinstance(batch, dict) else batch.input_ids
        visual_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=False).flatten().tolist()
        actual_count = len(visual_positions) or _grid_count(batch)
        if not visual_positions:
            raise RuntimeError(f"no video placeholder positions found for {sample.sample_id}")
        masks = find_instruction_masks(input_ids, processor, visual_positions)
        visual = torch.as_tensor(masks["visual_positions"], dtype=torch.long)
        instruction = torch.as_tensor(masks["instruction_positions"], dtype=torch.long)
        text = torch.as_tensor(masks["text_positions"], dtype=torch.long)
        query_specs = [
            ("last_instruction", [int(masks["query_index"])]),
            ("all_text", sorted(set(masks["instruction_positions"]) | set(masks["text_positions"]))),
        ]
        for attention_policy, query_positions in query_specs:
            if not query_positions:
                continue
            with capture_query_attention(model, query_positions[0] if attention_policy == "last_instruction" else query_positions) as captured:
                with torch.inference_mode():
                    model(**batch, use_cache=False, output_attentions=False, return_dict=True)
            if not captured:
                raise RuntimeError(f"attention capture returned no layers for {sample.sample_id}")
            layer_values = torch.stack([captured[layer] for layer in sorted(captured)])
            if layer_values.ndim == 3:
                attention = layer_values.mean(dim=(0, 1))
                head_count = int(layer_values.shape[1])
            else:
                attention = layer_values.mean(dim=(0, 1, 2))
                head_count = int(layer_values.shape[2])
            visual_mass = float(attention[visual].sum().item())
            instruction_mass = float(attention[instruction].sum().item()) if instruction.numel() else 0.0
            text_mass = float(attention[text].sum().item()) if text.numel() else 0.0
            visual_values = attention[visual].float()
            if visual_values.numel() > 1 and float(visual_values.sum()) > 0:
                probs = visual_values / visual_values.sum()
                entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(float(probs.numel())))).item())
            else:
                entropy = 0.0
            common = {
                "sample_id": sample.sample_id,
                "target_model": args.model,
                "temperature": 0.0,
                "paper_figure": "Figure 2",
                "attention_query": attention_policy,
                "attention_policy": attention_policy,
                "query_position": int(masks["query_index"]) if attention_policy == "last_instruction" else None,
                "query_positions": query_positions,
                "instruction_positions": masks["instruction_positions"],
                "visual_positions": masks["visual_positions"],
                "text_positions": masks["text_positions"],
                "visual_token_count": int(actual_count or 0),
                "target_visual_tokens": point.get("target_visual_tokens") if point else int(actual_count or 0),
                "actual_visual_tokens": int(actual_count or 0),
                "target_input_fingerprint": _fingerprint(input_ids),
                "draft_input_fingerprint": _fingerprint(input_ids),
                "heads": head_count,
                "layers": int(layer_values.shape[0]),
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
            for position, weight in enumerate(attention[visual].tolist()):
                row = dict(common)
                row.update({
                    "row_id": f"{sample.sample_id}:{actual_count}:{attention_policy}:visual:{position}",
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
                        "row_id": f"{sample.sample_id}:{actual_count}:{attention_policy}:{modality}:{position}",
                        "modality": modality,
                        "token_position": int(position),
                        "attention_weight": float(attention[position].item()),
                    })
                    rows.append(row)
            summary = dict(common)
            summary.update({
                "row_id": f"{sample.sample_id}:{actual_count}:{attention_policy}:summary",
                "modality": "summary",
                "token_position": int(masks["query_index"]),
                "attention_weight": None,
            })
            rows.append(summary)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} Figure 2 rows to {args.output}")
    return 0


def _fingerprint(input_ids: torch.Tensor) -> str:
    values = input_ids.detach().to("cpu").contiguous()
    return __import__("hashlib").sha256(values.numpy().tobytes() + str(values.shape).encode()).hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output", default="results/sparrow_validation/figure2_attention.jsonl")
    parser.add_argument("--calibration")
    parser.add_argument("--visual-targets", type=int, nargs="+")
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--quantized", action="store_true")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
