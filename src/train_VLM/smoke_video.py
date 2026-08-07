"""One-video, one-GPU end-to-end smoke test for vanilla Video-DFlash.

The script intentionally uses an untrained DFlash drafter. Exact greedy output
must still match target autoregressive generation because every proposal is
verified by the frozen Qwen2.5-VL target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .config import DFlashTrainConfig
from .target import Qwen25VLTargetAdapter
from .trainer import load_draft_checkpoint, make_draft_model
from .vlm_decode import Qwen25VLDFlashDecoder, VLMDecodeStep


def _make_synthetic_video(path: Path, *, frames: int, size: int, fps: int = 8) -> None:
    """Write a tiny deterministic MP4 so the smoke test needs no dataset."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - real CLI dependency
        raise RuntimeError("PyAV is required to create the synthetic smoke video") from exc

    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = size
    stream.height = size
    stream.pix_fmt = "yuv420p"
    square = max(8, size // 5)
    try:
        for index in range(frames):
            image = np.zeros((size, size, 3), dtype=np.uint8)
            image[..., 0] = 20 + index * 10
            image[..., 1] = 40
            image[..., 2] = 90
            x = int(index * max(1, size - square) / max(1, frames - 1))
            y = size // 2 - square // 2
            image[y : y + square, x : x + square] = (240, 210, 30)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _eos_token_ids(adapter: Qwen25VLTargetAdapter) -> list[int]:
    value = getattr(getattr(adapter.model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(adapter.processor.tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({int(token_id) for token_id in value})
    return [int(value)]


def _find_visual_module(model: Any) -> Any:
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "visual", None),
    ):
        visual = getattr(candidate, "visual", None) if candidate is not None else None
        if visual is not None:
            return visual
    raise RuntimeError("Loaded Qwen2.5-VL model does not expose its visual module")


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _gib(value: int) -> float:
    return value / (1024**3)


def run_smoke(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU")
    if args.num_frames < 4 or args.num_frames > 8 or args.num_frames % 2:
        raise ValueError("--num-frames must be one of 4, 6, or 8")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.checkpoint and not args.video:
        raise ValueError("a trained --checkpoint smoke run requires a real --video path")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    config = DFlashTrainConfig(
        target_model=args.model,
        max_seq_length=args.max_seq_length,
        block_size=args.block_size,
        num_draft_layers=args.draft_layers,
        num_target_features=args.target_features,
        selected_target_layers=(
            [int(value) for value in args.selected_target_layers.split(",")]
            if args.selected_target_layers
            else None
        ),
        context_mode=args.context_mode,
        mixed_precision=args.dtype,
        use_flex_attention=args.flex_attention,
        compile_flex_attention=False,
        gradient_checkpointing=False,
        video_num_frames=args.num_frames,
        video_min_pixels=args.size * args.size,
        video_max_pixels=args.size * args.size,
        video_reader=args.video_reader,
        target_attn_implementation=args.target_attention,
    )

    print(
        f"[setup] model={args.model} device={device} dtype={args.dtype} "
        f"draft_layers={args.draft_layers} block={args.block_size}"
    )
    adapter = Qwen25VLTargetAdapter.from_pretrained(
        config, device=device, dtype=dtype
    )
    adapter.freeze()
    draft = (
        load_draft_checkpoint(args.checkpoint, adapter, config)
        if args.checkpoint
        else make_draft_model(adapter, config).eval()
    )
    visual = _find_visual_module(adapter.model)
    visual_calls = {"count": 0}

    def count_visual_call(_module: Any, _inputs: Any) -> None:
        visual_calls["count"] += 1

    hook = visual.register_forward_pre_hook(count_visual_call)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.video:
            video_path = Path(args.video).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="video-dflash-smoke-")
            video_path = Path(temporary.name) / "moving-square.mp4"
            _make_synthetic_video(
                video_path, frames=args.num_frames, size=args.size, fps=args.video_fps
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path.as_uri(),
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        prompt_inputs, media = adapter.prepare_messages(messages)
        required = {"input_ids", "pixel_values_videos", "video_grid_thw"}
        missing = required.difference(prompt_inputs)
        if missing:
            raise RuntimeError(f"Video processor omitted required keys: {sorted(missing)}")
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        _, video_ids = adapter.visual_token_ids
        visual_tokens = sum(
            int(prompt_inputs["input_ids"].eq(token_id).sum()) for token_id in video_ids
        )
        frame_count = media.frame_counts[0] if media.frame_counts else args.num_frames
        print(
            f"[input] file={video_path.name} frames={frame_count} size={args.size}x{args.size} "
            f"visual_tokens={visual_tokens} prompt_tokens={prompt_length} "
            f"grid={list(media.video_grid_thw)}"
        )

        eos_ids = _eos_token_ids(adapter)
        visual_calls["count"] = 0
        with torch.inference_mode():
            ar_output = adapter.model.generate(
                **_clone_inputs(prompt_inputs),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                # Qwen's checkpoint generation config enables a 1.05
                # repetition penalty.  Vanilla greedy DFlash verifies raw
                # target argmax logits, so make the AR oracle raw greedy too.
                repetition_penalty=1.0,
                temperature=None,
                use_cache=True,
            )
        ar_visual_calls = visual_calls["count"]
        if ar_visual_calls != 1:
            raise RuntimeError(
                f"AR reference invoked the visual encoder {ar_visual_calls} times; expected once"
            )
        ar_new = ar_output[:, prompt_length:]
        print(f"[ar] output_tokens={ar_new.shape[1]} vision_prefills={ar_visual_calls}")

        del ar_output
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        visual_calls["count"] = 0

        def log_step(step: VLMDecodeStep) -> None:
            print(
                f"[step {step.iteration:02d}] proposed={len(step.proposed_token_ids)} "
                f"accepted={step.accepted_proposals} emitted={list(step.emitted_token_ids)} "
                f"cache_len={step.target_cache_length} cache_k={step.target_cache_key_shape}"
            )

        decoder = Qwen25VLDFlashDecoder(adapter, draft, config)
        result = decoder.generate(
            _clone_inputs(prompt_inputs),
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            stop_token_ids=eos_ids,
            trace_callback=log_step,
        )
        dflash_visual_calls = visual_calls["count"]
        if dflash_visual_calls != 1:
            raise RuntimeError(
                "Video-DFlash must invoke the visual encoder exactly once during prefill; "
                f"observed {dflash_visual_calls} calls"
            )
        dflash_new = result.output_ids[:, prompt_length:]
        equal = torch.equal(ar_new, dflash_new)
        print(
            f"[memory] peak_vram={_gib(result.peak_memory_bytes):.2f}GiB "
            f"final_cache_len={result.final_cache_length} vision_prefills={dflash_visual_calls}"
        )
        print(
            f"[result] equality={equal} ar_tokens={ar_new.shape[1]} "
            f"dflash_tokens={dflash_new.shape[1]} target_calls={result.target_forward_calls}"
        )
        if not equal:
            ar_values = ar_new[0].detach().cpu().tolist()
            draft_values = dflash_new[0].detach().cpu().tolist()
            mismatch = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(ar_values, draft_values))
                    if left != right
                ),
                min(len(ar_values), len(draft_values)),
            )
            raise AssertionError(
                f"Greedy AR/Video-DFlash mismatch at output token {mismatch}: "
                f"ar={ar_values} dflash={draft_values}"
            )
    finally:
        hook.remove()
        if temporary is not None:
            temporary.cleanup()


def main() -> None:  # pragma: no cover - GPU integration entry point
    parser = argparse.ArgumentParser(description="Run one tiny Video-DFlash smoke test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--video", default="", help="Optional local MP4; synthetic video by default")
    parser.add_argument("--prompt", default="Briefly describe the moving object in this video.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument(
        "--video-reader", choices=["torchvision", "decord", "torchcodec"], default="torchvision"
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--draft-layers", type=int, default=5)
    parser.add_argument("--target-features", type=int, default=5)
    parser.add_argument("--selected-target-layers", default="")
    parser.add_argument("--context-mode", choices=["full", "text_only"], default="full")
    parser.add_argument("--checkpoint", default="", help="Optional trained draft checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flex-attention", action="store_true")
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
