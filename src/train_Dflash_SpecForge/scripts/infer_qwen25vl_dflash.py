#!/usr/bin/env python3
"""HF end-to-end smoke inference for a Qwen2.5-VL DFlash checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from specforge.export.checkpoint_io import materialize_draft, resolve_training_state
from specforge.qwen25vl import prepare_inference_prompt


def _load_target(path: str):
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    except (AttributeError, ImportError, ValueError):
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--draft-model-config", default="configs/qwen2.5-vl-3b-dflash.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    return parser.parse_args()


def _move(value, device):
    return value.to(device) if torch.is_tensor(value) else value


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 <= args.sample_index < len(records):
        raise IndexError(f"sample-index {args.sample_index} is outside [0, {len(records)})")

    from transformers import AutoConfig, AutoProcessor

    target_config = AutoConfig.from_pretrained(args.target_model_path)
    processor = AutoProcessor.from_pretrained(args.target_model_path)
    prepared = prepare_inference_prompt(
        processor,
        target_config,
        records[args.sample_index],
        image_root=args.image_root,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    target = _load_target(args.target_model_path).eval()
    device = next(target.parameters()).device
    input_ids = prepared["input_ids"].to(device)
    attention_mask = prepared["attention_mask"].to(device)
    position_ids = prepared["position_ids"].to(device)
    media = {
        key: _move(value, device)
        for key, value in prepared["multimodal_inputs"].items()
    }
    target_inputs = {
        **media,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    with torch.inference_mode():
        target_output = target.generate(
            input_ids=input_ids,
            **target_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        state = resolve_training_state(args.checkpoint)
        draft = materialize_draft(state, args.draft_model_config)
        draft.to(device=device, dtype=torch.bfloat16).eval()
        speculative = draft.spec_generate(
            target,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[
                int(value)
                for value in {
                    getattr(processor.tokenizer, "eos_token_id", None),
                    getattr(processor.tokenizer, "im_end_id", None),
                }
                if value is not None
            ],
            temperature=0.0,
            position_ids=position_ids,
            target_kwargs=media,
        )

    target_new = target_output[:, input_ids.shape[1] :]
    speculative_new = speculative[:, input_ids.shape[1] :]
    if not torch.equal(target_new, speculative_new):
        raise RuntimeError(
            "DFlash HF smoke output differs from target-only greedy output: "
            f"target_shape={tuple(target_new.shape)} speculative_shape={tuple(speculative_new.shape)}"
        )
    text = processor.tokenizer.decode(target_new[0], skip_special_tokens=True)
    print(json.dumps({"sample_id": records[args.sample_index]["id"], "text": text}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
