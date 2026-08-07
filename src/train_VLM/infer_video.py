"""Run greedy, lossless Video-DFlash inference on one real video."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import torch

from .config import DFlashTrainConfig
from .target import Qwen25VLTargetAdapter
from .trainer import load_draft_checkpoint
from .vlm_decode import Qwen25VLDFlashDecoder, VLMDecodeStep


def _config_from_checkpoint(checkpoint: Path) -> DFlashTrainConfig:
    metadata = json.loads((checkpoint / "dflash_config.json").read_text())
    valid = {field.name for field in fields(DFlashTrainConfig)}
    return DFlashTrainConfig(**{key: value for key, value in metadata.items() if key in valid})


def _eos_token_ids(adapter: Qwen25VLTargetAdapter) -> list[int]:
    value = getattr(getattr(adapter.model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(adapter.processor.tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({int(token_id) for token_id in value})
    return [int(value)]


def infer_video(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    video = Path(args.video).expanduser().resolve(strict=True)
    if not video.is_file():
        raise FileNotFoundError(video)
    config = _config_from_checkpoint(checkpoint)
    config.device = args.device
    if args.dtype is not None:
        config.mixed_precision = args.dtype
    if args.target_attention is not None:
        config.target_attn_implementation = args.target_attention
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Video-DFlash inference requires the requested CUDA device")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[config.mixed_precision]

    adapter = Qwen25VLTargetAdapter.from_pretrained(config, device=device, dtype=dtype)
    adapter.freeze()
    draft = load_draft_checkpoint(checkpoint, adapter, config).eval()

    # These are inference-only media choices. Load/validate the checkpoint
    # against its original processor contract before applying them.
    if args.num_frames is not None:
        if args.num_frames < 2 or args.num_frames % 2:
            raise ValueError("--num-frames must be an even integer >= 2")
        adapter.video_num_frames = args.num_frames
    if args.video_min_pixels is not None:
        adapter.video_min_pixels = args.video_min_pixels
    if args.video_max_pixels is not None:
        adapter.video_max_pixels = args.video_max_pixels
    if args.video_reader is not None:
        adapter.video_reader = args.video_reader

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video.as_uri()},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    inputs, media = adapter.prepare_messages(messages)

    def trace(step: VLMDecodeStep) -> None:
        print(
            f"[decode step={step.iteration}] proposed={len(step.proposed_token_ids)} "
            f"accepted={step.accepted_proposals} emitted={len(step.emitted_token_ids)} "
            f"target_cache={step.target_cache_length}"
        )

    decoder = Qwen25VLDFlashDecoder(adapter, draft, config)
    with torch.inference_mode():
        result = decoder.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            stop_token_ids=_eos_token_ids(adapter),
            trace_callback=trace if args.trace else None,
        )
    new_ids = result.output_ids[0, result.num_input_tokens :].detach().cpu()
    text = adapter.processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    metrics: dict[str, Any] = result.metrics()
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "video": str(video),
            "frame_counts": list(media.frame_counts),
            "video_grid_thw": [list(item) for item in media.video_grid_thw],
            "text": text,
        }
    )
    print("[metrics] " + json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    print("\n[output]\n" + text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
        temporary.replace(output)
    return metrics


def main() -> None:  # pragma: no cover - GPU integration CLI
    parser = argparse.ArgumentParser(description="Run Video-DFlash on one real video")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default="Describe this video in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "no"], default=None)
    parser.add_argument("--target-attention", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument(
        "--video-reader", choices=["torchvision", "decord", "torchcodec"], default=None
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", default="", help="Optional JSON result path")
    infer_video(parser.parse_args())


if __name__ == "__main__":
    main()
