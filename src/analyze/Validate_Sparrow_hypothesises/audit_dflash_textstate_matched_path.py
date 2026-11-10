"""Diagnostic audit for text-state feature capture: matched-path, no-op, and previsual integrity."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from src.analyze.Validate_Sparrow_hypothesises.run_dflash_distribution_replay import (
    _build_prompt_record,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_experiments import (
    _prepare_prompt,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import _load_records
from src.infer.qwen25vl_dflash_compare import (
    _capture_target_hidden_states,
    _concat_captured_target_hidden,
    _compute_position_ids,
    _load_target,
    _resolve_dtype,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEDULE = ROOT / "results/dflash_textstate_intervention/pilot12_3k_20261106/design/block_schedule.json"
DEFAULT_TARGET = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
DEFAULT_CALIBRATION = ROOT / "results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/calibration/calibration.jsonl"
DEFAULT_SAMPLES = ["v_1YU4MSK80cQ", "v_ej9TkJVjK9M", "v_zRqJZ0u6f88", "v_aLb79wtvxoQ"]
LAYER_IDS = [1, 9, 17, 25, 33]


def capture_target_features(
    *,
    target: Any,
    prompt: Any,
    gold_ids: list[int],
    layer_ids: list[int],
    device: torch.device,
    hook_mode: str = "none",  # "none", "noop", "ablation"
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Capture target features using a full teacher-forced forward pass over prompt + answer."""

    input_ids = prompt.inputs["input_ids"]
    answer = torch.as_tensor(gold_ids, device=device, dtype=input_ids.dtype).view(1, -1)
    full_input_ids = torch.cat([input_ids.to(device), answer], dim=1)
    total_length = int(full_input_ids.shape[1])

    inputs_for_positions = dict(prompt.inputs)
    inputs_for_positions["input_ids"] = full_input_ids
    attention = inputs_for_positions.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(full_input_ids)
    else:
        attention = attention.to(device=device)
        if attention.shape[1] != total_length:
            extension = torch.ones(
                (attention.shape[0], total_length - attention.shape[1]),
                dtype=attention.dtype,
                device=device,
            )
            attention = torch.cat([attention, extension], dim=1)
    inputs_for_positions["attention_mask"] = attention
    if "mm_token_type_ids" in inputs_for_positions:
        mm = inputs_for_positions["mm_token_type_ids"].to(device=device)
        if mm.shape[1] != total_length:
            extension = torch.zeros(
                (mm.shape[0], total_length - mm.shape[1]),
                dtype=mm.dtype,
                device=device,
            )
            inputs_for_positions["mm_token_type_ids"] = torch.cat([mm, extension], dim=1)

    position_ids = _compute_position_ids(target, inputs_for_positions)

    forward_kwargs = {
        key: value
        for key, value in dict(prompt.target_kwargs).items()
        if key not in {"input_ids", "position_ids", "past_key_values", "cache_position"}
    }
    if "mm_token_type_ids" in forward_kwargs and isinstance(forward_kwargs["mm_token_type_ids"], torch.Tensor):
        mm = forward_kwargs["mm_token_type_ids"].to(device=device)
        if mm.shape[1] != total_length:
            extension = torch.zeros(
                (mm.shape[0], total_length - mm.shape[1]),
                dtype=mm.dtype,
                device=device,
            )
            forward_kwargs["mm_token_type_ids"] = torch.cat([mm, extension], dim=1)

    forward_kwargs.update(
        {
            "input_ids": full_input_ids,
            "attention_mask": attention,
            "position_ids": position_ids,
            "cache_position": torch.arange(total_length, device=device),
            "past_key_values": None,
            "use_cache": False,
            "output_hidden_states": False,
            "return_dict": True,
            "logits_to_keep": 1,
        }
    )

    model_core = getattr(target, "model", None)
    if model_core is None or not callable(getattr(model_core, "get_video_features", None)):
        raise RuntimeError("cannot locate Qwen video-feature merger before language-model scatter")

    had_override = "get_video_features" in getattr(model_core, "__dict__", {})
    old_override = model_core.__dict__.get("get_video_features") if had_override else None
    original_get_video_features = model_core.get_video_features
    hook_stats = {"calls": 0, "rows": 0, "nonzero_rows": 0}

    def noop_hook(pixel_values_videos, video_grid_thw=None, **kwargs):
        outputs = original_get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)
        hook_stats["calls"] += 1
        pooler = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        hook_stats["rows"] += sum(int(v.shape[0]) for v in pooler)
        hook_stats["nonzero_rows"] += sum(int(torch.count_nonzero(v).item()) for v in pooler)
        return outputs

    def zero_video_features(pixel_values_videos, video_grid_thw=None, **kwargs):
        outputs = original_get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)
        hook_stats["calls"] += 1
        if hasattr(outputs, "pooler_output"):
            pooler = outputs.pooler_output
            zeros = tuple(torch.zeros_like(v) for v in pooler)
            outputs.pooler_output = zeros
            hook_stats["rows"] += sum(int(v.shape[0]) for v in zeros)
            hook_stats["nonzero_rows"] += sum(int(torch.count_nonzero(v).item()) for v in zeros)
            return outputs
        else:
            zeros = tuple(torch.zeros_like(v) for v in outputs)
            hook_stats["rows"] += sum(int(v.shape[0]) for v in zeros)
            hook_stats["nonzero_rows"] += sum(int(torch.count_nonzero(v).item()) for v in zeros)
            return zeros

    if hook_mode == "noop":
        model_core.get_video_features = noop_hook
    elif hook_mode == "ablation":
        model_core.get_video_features = zero_video_features

    try:
        with torch.inference_mode(), _capture_target_hidden_states(
            target, layer_ids, device=device
        ) as captured:
            target(**forward_kwargs)
        bank = _concat_captured_target_hidden(captured, layer_ids)
    finally:
        if hook_mode in {"noop", "ablation"}:
            if had_override:
                model_core.get_video_features = old_override
            else:
                delattr(model_core, "get_video_features")

    return bank, hook_stats


def capture_prefill_cached(
    *,
    target: Any,
    prompt: Any,
    layer_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Capture prompt hidden states via cached prefill forward (matching old G-prefill codepath)."""

    input_ids = prompt.inputs["input_ids"].to(device)
    prompt_length = int(input_ids.shape[1])
    forward_kwargs = {
        key: value
        for key, value in dict(prompt.target_kwargs).items()
        if key not in {"input_ids", "position_ids", "past_key_values", "cache_position"}
    }
    forward_kwargs.update(
        {
            "input_ids": input_ids,
            "attention_mask": prompt.inputs.get("attention_mask", torch.ones_like(input_ids)).to(device),
            "position_ids": prompt.position_ids.to(device) if hasattr(prompt, "position_ids") and prompt.position_ids is not None else None,
            "cache_position": torch.arange(prompt_length, device=device),
            "past_key_values": None,
            "use_cache": True,
            "output_hidden_states": False,
            "return_dict": True,
        }
    )
    with torch.inference_mode(), _capture_target_hidden_states(
        target, layer_ids, device=device
    ) as captured:
        target(**forward_kwargs)
    return _concat_captured_target_hidden(captured, layer_ids)


def run_diagnostic_audit(
    *,
    target_model_path: Path,
    schedule_path: Path,
    video_root: Path,
    calibration_path: Path = DEFAULT_CALIBRATION,
    sample_ids: list[str],
    device_str: str = "cuda:0",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Audit] Device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    print(f"[Audit] Loading target model from {target_model_path} ...")
    processor, target, target_load_s = _load_target(
        str(target_model_path),
        device=device,
        dtype=_resolve_dtype("bf16", device),
        attention="sdpa",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    target.eval()

    with open(schedule_path, "r", encoding="utf-8") as f:
        schedule_doc = json.load(f)

    # Load prompt records
    manifest_path = ROOT / "dataset/VideoDetailCaption/test.jsonl"
    records = _load_records(manifest_path, video_root, calibration_path, "vdc")
    record_by_id = {str(row["sample_id"]): row for row in records}

    results = {}
    print("=" * 80)
    print(f"{'Sample ID':<16} | {'Tol (No-op)':<12} | {'Matched Pre-err':<16} | {'Old Path Pre-err':<17} | {'Suffix Leak':<12} | {'Status'}")
    print("-" * 80)

    for sample_id in sample_ids:
        if sample_id not in schedule_doc["samples"]:
            print(f"Sample {sample_id} not found in schedule, skipping.")
            continue

        sample_info = schedule_doc["samples"][sample_id]
        gold_ids = [int(v) for v in sample_info["blocks"][0]["gold_ids"]] if "blocks" in sample_info and sample_info["blocks"] and "gold_ids" in sample_info["blocks"][0] else []
        if not gold_ids:
            # Fall back to source pair
            source_pair_path = ROOT / sample_info["source_pair"]
            with gzip.open(source_pair_path, "rt", encoding="utf-8") as f:
                meta = json.loads(f.readline())
            gold_ids = [int(v) for v in meta["target_greedy_ids"]]

        record = _build_prompt_record(record_by_id[sample_id])
        prompt, visual_positions, prompt_fp, _ = _prepare_prompt(
            processor=processor,
            target=target,
            sample=record,
            video_root=str(video_root),
            device=device,
            target_visual_tokens=3000,
            allow_out_of_tolerance=False,
            video_reader="decord",
        )
        first_visual = min(visual_positions)

        # 1. Standard forward (G matched-path)
        bank_std, _ = capture_target_features(
            target=target,
            prompt=prompt,
            gold_ids=gold_ids,
            layer_ids=LAYER_IDS,
            device=device,
            hook_mode="none",
        )

        # 2. No-op hook forward
        bank_noop, _ = capture_target_features(
            target=target,
            prompt=prompt,
            gold_ids=gold_ids,
            layer_ids=LAYER_IDS,
            device=device,
            hook_mode="noop",
        )
        tol_noop = float((bank_std.float() - bank_noop.float()).abs().max().item())

        # 3. Ablation forward (A matched-path)
        bank_abl, _ = capture_target_features(
            target=target,
            prompt=prompt,
            gold_ids=gold_ids,
            layer_ids=LAYER_IDS,
            device=device,
            hook_mode="ablation",
        )
        # Previsual error under matched-path:
        matched_pre_err = float(
            (bank_std[:, :first_visual].float() - bank_abl[:, :first_visual].float())
            .abs()
            .max()
            .item()
        )

        # 4. Old-path cached prefill comparison
        bank_prefill = capture_prefill_cached(
            target=target,
            prompt=prompt,
            layer_ids=LAYER_IDS,
            device=device,
        )
        old_path_pre_err = float(
            (bank_prefill[:, :first_visual].float() - bank_abl[:, :first_visual].float())
            .abs()
            .max()
            .item()
        )

        # 5. Suffix perturbation leakage check
        altered_ids = list(gold_ids)
        vocab_size = int(getattr(target.config, "vocab_size", 152064))
        altered_ids[-1] = (altered_ids[-1] + 1) % vocab_size
        bank_abl_altered, _ = capture_target_features(
            target=target,
            prompt=prompt,
            gold_ids=altered_ids,
            layer_ids=LAYER_IDS,
            device=device,
            hook_mode="ablation",
        )
        prompt_len = int(prompt.inputs["input_ids"].shape[1])
        leak_anchor = min(31, max(0, len(gold_ids) - 1))
        leak_prefix_len = prompt_len + leak_anchor
        leak_err = float(
            (bank_abl[:, :leak_prefix_len].float() - bank_abl_altered[:, :leak_prefix_len].float())
            .abs()
            .max()
            .item()
        )

        # Verdict
        passed_matched = matched_pre_err <= max(1e-4, tol_noop * 2) and leak_err <= 1e-4
        status = "PASSED" if passed_matched else "FAILED"

        print(
            f"{sample_id:<16} | {tol_noop:<12.6f} | {matched_pre_err:<16.6f} | "
            f"{old_path_pre_err:<17.6f} | {leak_err:<12.6f} | {status}"
        )

        results[sample_id] = {
            "first_visual": first_visual,
            "prompt_length": prompt_len,
            "answer_length": len(gold_ids),
            "no_op_tolerance": tol_noop,
            "matched_previsual_max_abs_error": matched_pre_err,
            "old_path_previsual_max_abs_error": old_path_pre_err,
            "suffix_leakage_error": leak_err,
            "passed_matched_audit": passed_matched,
        }

    print("=" * 80)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "matched_path_audit_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Audit] Results saved to {out_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic audit for text-state feature capture")
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--video-root", type=Path, default=ROOT / "dataset/VideoDetailCaption/Test_Videos")
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dflash_textstate_intervention/audit_matched_path")
    args = parser.parse_args()

    run_diagnostic_audit(
        target_model_path=args.target_model,
        schedule_path=args.schedule,
        video_root=args.video_root,
        sample_ids=args.samples,
        device_str=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
