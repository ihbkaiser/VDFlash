"""Paired Video-DFlash experiment using target verification attention selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from src.infer.qwen25vl_dflash_compare import (
    InstrumentedDFlashDecoder,
    _decode_text,
    _eos_token_ids,
    _load_draft,
    _load_target,
    _resolve_dtype,
    _sha256_tokens,
    _target_greedy,
    load_manifest_sample,
    prepare_video_prompt,
    resolve_dataset_format,
    resolve_video_path,
    score_caption,
)
from src.infer.verify_attention_selectors import ATTENTION_METHODS
from src.workspace import workspace_path


DEFAULT_CHECKPOINT = workspace_path(
    "dataset",
    "qwen25vl-3b-depth1-depth3",
    "qwen25vl-3b-dflash-llava68k-h32-depth3-latest",
    "training_state.pt",
)
DEFAULT_CONFIG = workspace_path(
    "src",
    "train_Dflash_SpecForge",
    "configs",
    "qwen2.5-vl-3b-dflash-h32-depth3.json",
)
DEFAULT_MANIFEST = workspace_path("dataset", "VideoDetailCaption", "test_100.jsonl")
DEFAULT_VIDEO_ROOT = workspace_path("dataset", "VideoDetailCaption")
DEFAULT_OUTPUT = workspace_path(
    "results",
    "infer",
    "verify_attention_context",
    "qwen25vl3b_vdc_descriptions_b128_pilot.json",
)
DEFAULT_SAMPLE_INDICES = (0, 9, 19, 29, 39, 49)
ORDER_SEED = 20261021


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _arm_order(sample_slot: int) -> list[str]:
    arms = ["dense", *ATTENTION_METHODS]
    random.Random(ORDER_SEED + int(sample_slot)).shuffle(arms)
    return arms


def _first_token_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _sample_result(
    *,
    args: argparse.Namespace,
    sample_index: int,
    sample_slot: int,
    processor: Any,
    target: Any,
    draft: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    record = load_manifest_sample(args.manifest, sample_index)
    dataset_format = resolve_dataset_format(record, args.dataset_format)
    video_path = resolve_video_path(record, args.video_root)
    prompt = prepare_video_prompt(
        processor,
        target,
        record,
        video_root=args.video_root,
        device=device,
        num_frames=args.num_frames,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_reader=args.video_reader,
        dataset_format=dataset_format,
    )
    prompt_length = int(prompt.inputs["input_ids"].shape[1])
    target_ids, target_timing = _target_greedy(
        target,
        prompt,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=_eos_token_ids(processor, target),
        device=device,
    )
    target_tokens = target_ids[0, prompt_length:].detach().cpu().tolist()
    target_text = _decode_text(processor, target_ids[0, prompt_length:])
    reference = str(record.get("answer") or record.get("reference") or "")
    order = _arm_order(sample_slot)
    arm_rows: dict[str, dict[str, Any]] = {}
    arm_tokens: dict[str, list[int]] = {}

    for arm in order:
        selector = None if arm == "dense" else arm
        decoder = InstrumentedDFlashDecoder(
            target,
            draft,
            device=device,
            token_decoder=lambda values: processor.tokenizer.decode(
                values, skip_special_tokens=True
            ).strip(),
            attention_selector=selector,
            context_budget=args.context_budget,
        )
        started = time.perf_counter()
        try:
            speculative = decoder.decode(
                input_ids=prompt.inputs["input_ids"],
                position_ids=prompt.position_ids,
                target_kwargs=prompt.target_kwargs,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=_eos_token_ids(processor, target),
            )
            output_tokens = speculative.output_ids[
                0, prompt_length:
            ].detach().cpu().tolist()
            prediction = _decode_text(processor, speculative.output_ids[0, prompt_length:])
            metrics = speculative.as_dict()
            rounds = metrics["acceptance_rounds"]
            selected_fractions = [
                item["context_selection"]["context_tokens_used"]
                / max(1, item["context_selection"]["context_tokens_available"])
                for item in rounds
                if not item["context_selection"]["dense_bootstrap"]
            ]
            arm_rows[arm] = {
                "status": "ok",
                "prediction": prediction,
                "output_token_count": len(output_tokens),
                "output_token_ids": output_tokens,
                "output_hash": _sha256_tokens(output_tokens),
                "matches_target_greedy": output_tokens == target_tokens,
                "text_metrics": score_caption(prediction, reference),
                "mean_accepted_proposals_per_round": metrics[
                    "mean_accepted_proposals"
                ],
                "mean_effective_emitted_tokens_per_round": metrics[
                    "mean_effective_emitted_tokens"
                ],
                "tau_proposal": metrics["tau_proposal"],
                "tau_effective": metrics["tau_effective"],
                "round_count": metrics["rounds"],
                "target_forward_calls": metrics["target_forward_calls"],
                "selected_context_fraction_mean": (
                    sum(selected_fractions) / len(selected_fractions)
                    if selected_fractions
                    else None
                ),
                "selection_round_count": len(selected_fractions),
                "timing": metrics["timing"],
                "wall_e2e_s": float(time.perf_counter() - started),
                "peak_memory_bytes": metrics["peak_memory_bytes"],
                "rounds": [
                    {
                        "matched_proposals": item["matched_proposals"],
                        "effective_emitted_tokens": item[
                            "effective_emitted_tokens"
                        ],
                        "is_partial_block": item["is_partial_block"],
                        "is_terminal": item["is_terminal"],
                        "context_selection": item["context_selection"],
                    }
                    for item in rounds
                ],
            }
            arm_tokens[arm] = output_tokens
        except Exception as exc:
            arm_rows[arm] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "wall_e2e_s": float(time.perf_counter() - started),
            }

    dense_tokens = arm_tokens.get("dense")
    for arm, row in arm_rows.items():
        row["matches_dense"] = (
            arm_tokens[arm] == dense_tokens
            if arm in arm_tokens and dense_tokens is not None
            else None
        )
        row["first_target_mismatch_index"] = (
            _first_token_mismatch(arm_tokens[arm], target_tokens)
            if arm in arm_tokens
            else None
        )

    manifest_row_hash = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "sample_index": int(sample_index),
        "sample_id": str(
            record.get("sample_id") or record.get("video_name") or record.get("id") or sample_index
        ),
        "task_type": str(record.get("task_type") or record.get("task") or ""),
        "dataset_format": dataset_format,
        "video_sha256": _sha256_file(video_path),
        "manifest_row_sha256": manifest_row_hash,
        "question": str(record.get("question") or ""),
        "reference": reference,
        "target_greedy": {
            "prediction": target_text,
            "output_token_ids": target_tokens,
            "output_hash": _sha256_tokens(target_tokens),
            "output_token_count": len(target_tokens),
            "text_metrics": score_caption(target_text, reference),
            "timing": target_timing,
        },
        "prompt": {
            "input_tokens": prompt_length,
            "num_frames": args.num_frames,
            "frame_counts": list(prompt.frame_counts),
            "video_grid_thw": [list(row) for row in prompt.video_grid_thw],
        },
        "arm_order": order,
        "arms": arm_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--draft-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-indices", type=int, nargs="+", default=list(DEFAULT_SAMPLE_INDICES)
    )
    parser.add_argument("--dataset-format", choices=("auto", "vdc", "mvbench"), default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-budget", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--video-min-pixels", type=int, default=50176)
    parser.add_argument("--video-max-pixels", type=int, default=50176)
    parser.add_argument("--video-reader", default="decord")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.manifest = Path(args.manifest).expanduser().resolve(strict=True)
    args.video_root = Path(args.video_root).expanduser().resolve(strict=True)
    args.checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    args.draft_config = Path(args.draft_config).expanduser().resolve(strict=True)
    output_path = Path(args.output).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this experiment on a GPU host")
    dtype = _resolve_dtype(args.dtype, device)

    report: dict[str, Any] = {
        "experiment": "qwen25vl_video_dflash_verify_attention_context",
        "target_model": Path(args.target_model).name,
        "checkpoint": str(args.checkpoint.relative_to(Path.cwd()))
        if args.checkpoint.is_relative_to(Path.cwd())
        else args.checkpoint.name,
        "draft_config": args.draft_config.name,
        "device": str(device),
        "dtype": str(dtype),
        "manifest": str(args.manifest.relative_to(Path.cwd()))
        if args.manifest.is_relative_to(Path.cwd())
        else args.manifest.name,
        "sample_indices": [int(value) for value in args.sample_indices],
        "methods": ["dense", *ATTENTION_METHODS],
        "context_budget": int(args.context_budget),
        "max_new_tokens": int(args.max_new_tokens),
        "num_frames": int(args.num_frames),
        "target_attention_backend": args.target_attention,
        "order_seed": ORDER_SEED,
        "selector_layers_zero_based": {"M0": 18, "M1": 18, "M2": 18, "M3": 18, "M5": 35},
        "selection_policy": "dense first draft call; verifier-selected context applies starting next round",
        "timing_scope": "each decode includes target prefill, draft, verify, and selector work; shared video preprocessing and model load excluded",
        "samples": [],
    }
    completed: dict[int, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "experiment",
            "checkpoint",
            "draft_config",
            "context_budget",
            "max_new_tokens",
            "num_frames",
            "target_attention_backend",
        )
        if any(prior.get(key) != report.get(key) for key in invariant_keys):
            raise ValueError("existing report protocol does not match this resumed run")
        completed = {
            int(row["sample_index"]): row for row in prior.get("samples", [])
        }

    processor, target, target_load_s = _load_target(
        args.target_model,
        device=device,
        dtype=dtype,
        attention=args.target_attention,
    )
    draft, draft_load_s = _load_draft(
        args.checkpoint,
        args.draft_config,
        device=device,
        dtype=dtype,
    )
    report["model_load_s"] = {
        "target": float(target_load_s),
        "draft": float(draft_load_s),
    }
    report["samples"] = [completed[index] for index in sorted(completed)]
    _atomic_write_json(output_path, report)

    for sample_slot, sample_index in enumerate(args.sample_indices):
        if sample_index in completed:
            continue
        row = _sample_result(
            args=args,
            sample_index=sample_index,
            sample_slot=sample_slot,
            processor=processor,
            target=target,
            draft=draft,
            device=device,
            dtype=dtype,
        )
        completed[sample_index] = row
        report["samples"] = [completed[index] for index in sorted(completed)]
        _atomic_write_json(output_path, report)
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_experiment(args)
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "sample_count": len(report["samples"]),
        "sample_statuses": [
            {
                "sample_id": row["sample_id"],
                "arms_ok": sum(arm.get("status") == "ok" for arm in row["arms"].values()),
                "arms_total": len(row["arms"]),
            }
            for row in report["samples"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
