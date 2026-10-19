"""Collect and score free-running DFlash visual-retention rollouts.

Generate phase: draft proposals form the scored output tokens. The target
refreshes conditioning features on the draft prefix. An optional seed-control
mode uses one common target-greedy token and excludes it from scoring. The
sensitivity phase compares proposals on identical teacher-forced prefixes;
acceptance remains a separate verified-decoding phase.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import torch

from src.analyze.Validate_Sparrow_hypothesises.free_running_experiment import (
    CONDITION_NAMES,
    RETENTION_PERCENTAGES,
    append_jsonl,
    build_blind_pairs,
    build_matched_prefix_pairs,
    build_visual_retention_mask,
    checkpoint_fingerprint,
    classify_token_surface,
    make_judge_prompt,
    make_protocol,
    parse_judge_response,
    protocol_fingerprint,
    read_jsonl,
    verify_or_create_protocol,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import (
    _load_records,
    _prepare_prompt,
)
from src.infer.dflash_free_running import FreeRunningDFlashDecoder
from src.infer.qwen25vl_dflash_compare import (
    PreparedVideoPrompt,
    _compute_position_ids,
    _decode_text,
    _eos_token_ids,
    InstrumentedDFlashDecoder,
    _load_draft,
    _load_target,
    _resolve_dtype,
    _sha256_tokens,
    _target_greedy,
)
from src.workspace import resolve_workspace_path, workspace_path


DEFAULT_CHECKPOINT = workspace_path(
    "dataset", "qwen25vl-3b-depth1-depth3",
    "qwen25vl-3b-dflash-llava68k-h32-depth3-latest", "training_state.pt",
)
DEFAULT_DRAFT_CONFIG = workspace_path(
    "src", "train_Dflash_SpecForge", "configs",
    "qwen2.5-vl-3b-dflash-h32-depth3.json",
)
DEFAULT_MANIFEST = workspace_path("dataset", "VideoDetailCaption", "test.jsonl")
DEFAULT_VIDEO_ROOT = workspace_path("dataset", "VideoDetailCaption")
DEFAULT_CALIBRATION = workspace_path(
    "dataset", "VideoDetailCaption", "calibration_complete_20260823_currentenv.jsonl"
)
DEFAULT_OUTPUT_DIR = workspace_path("results", f"dflash_free_running_{date.today().isoformat()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "generate", "sensitivity", "acceptance", "score", "analyze", "all"),
        default="preflight",
    )
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--draft-config", default=str(DEFAULT_DRAFT_CONFIG))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=50, help="Cohort size; use 2 for the pilot.")
    parser.add_argument("--all-samples", action="store_true", help="Use every row in the VDC manifest.")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--target-visual-tokens", type=int, default=3000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--judge-max-new-tokens", type=int, default=160)
    parser.add_argument("--matched-prefix-tokens", type=int, default=128)
    parser.add_argument("--sensitivity-limit", type=int, default=10)
    parser.add_argument("--feature-refresh", choices=("refresh", "frozen"), default="refresh")
    parser.add_argument(
        "--target-bonus-per-block",
        action="store_true",
        help="Append one greedy target token after each full 15-token draft block; do not verify proposals.",
    )
    parser.add_argument(
        "--initial-anchor-control",
        choices=("prompt_tail", "target_greedy_seed"),
        default="prompt_tail",
        help="target_greedy_seed runs only Full/Cut and excludes the common target seed from coherence text.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="model_parallel")
    parser.add_argument("--max-memory", default="0:22GiB,1:14GiB")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "no"),
        default="fp16",
        help="Target/draft precision; FP16 avoids the BF16 SDPA near-tie parity failure seen in the pilot.",
    )
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--review-samples", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "draft_config", "manifest", "video_root", "calibration", "output_dir"):
        setattr(args, name, resolve_workspace_path(getattr(args, name)))


def _selected_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = _load_records(args.manifest, args.video_root, args.calibration, "vdc")
    if args.sample_id:
        wanted = {str(value) for value in args.sample_id}
        records = [row for row in records if str(row["sample_id"]) in wanted]
        missing = wanted - {str(row["sample_id"]) for row in records}
        if missing:
            raise ValueError(f"sample_id values are absent from the selected VDC manifest: {sorted(missing)}")
    if not args.all_samples:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[:args.limit]
    if not records:
        raise ValueError("no VDC samples were selected")
    sample_ids = [str(row["sample_id"]) for row in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("the selected cohort contains duplicate sample_id values")
    return records


def _latest_by_row_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get(key) or "")
        if not identifier:
            raise ValueError(f"row is missing {key}")
        latest[identifier] = row
    return latest


def _free_running_retentions(args: argparse.Namespace) -> tuple[int, ...]:
    return (100, 0) if args.initial_anchor_control == "target_greedy_seed" else RETENTION_PERCENTAGES


def _write_protocol_and_check(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_run_protocol_cache", None)
    protocol_path = Path(args.output_dir) / "protocol.json"
    if cached is not None:
        verify_or_create_protocol(protocol_path, cached)
        return cached
    records = _selected_records(args)
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    protocol = make_protocol(
        target_model=args.target_model,
        checkpoint=checkpoint,
        draft_config=args.draft_config,
        manifest=args.manifest,
        calibration=args.calibration,
        sample_ids=[str(row["sample_id"]) for row in records],
        sensitivity_limit=args.sensitivity_limit,
        review_samples=args.review_samples,
        initial_anchor=(
            "target_greedy_seed"
            if args.initial_anchor_control == "target_greedy_seed"
            else "last_prompt_token"
        ),
        free_running_retention_percentages=_free_running_retentions(args),
        max_new_tokens=args.max_new_tokens,
        judge_max_new_tokens=args.judge_max_new_tokens,
        matched_prefix_tokens=args.matched_prefix_tokens,
        feature_refresh=args.feature_refresh,
        target_visual_tokens=args.target_visual_tokens,
        seed=args.seed,
        dtype=args.dtype,
        target_attention=args.target_attention,
        device_map=args.device_map,
        allow_out_of_tolerance=args.allow_out_of_tolerance,
        target_bonus_per_block=bool(getattr(args, "target_bonus_per_block", False)),
    )
    verify_or_create_protocol(protocol_path, protocol)
    args._run_protocol_cache = protocol
    return protocol


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    records = _selected_records(args)
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        state = checkpoint / "training_state.pt"
        if not state.is_file():
            state = next(iter(sorted(checkpoint.glob("*-latest/training_state.pt"))), state)
    else:
        state = checkpoint
    if not state.is_file():
        raise FileNotFoundError(f"DFlash checkpoint is not a training_state.pt: {checkpoint.name}")
    config = Path(args.draft_config)
    manifest = Path(args.manifest)
    calibration = Path(args.calibration)
    for label, path in (("draft config", config), ("manifest", manifest), ("calibration", calibration)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (output_dir / "protocol.json").is_file():
        protocol = _write_protocol_and_check(args)
    else:
        protocol = None
    result = {
        "status": "preflight_ok",
        "phase": "preflight",
        "selected_samples": len(records),
        "retention_percentages": list(_free_running_retentions(args)),
        "planned_rollouts": len(records) * len(_free_running_retentions(args)),
        "planned_acceptance_runs": len(records) * len(RETENTION_PERCENTAGES),
        "planned_sensitivity_probes": min(len(records), args.sensitivity_limit) * len(RETENTION_PERCENTAGES),
        "checkpoint": checkpoint_fingerprint(args.checkpoint),
        "output_dir": str(output_dir),
        "existing_protocol_matches": protocol is not None,
        "cuda_available": bool(torch.cuda.is_available()),
        "warning": "Preflight loads no models and does not establish inference compatibility.",
    }
    return result


def _error_rollout(
    *, sample: dict[str, Any], retention: int, row_id: str, attempt: int,
    error: Exception, target_bonus_per_block: bool = False,
) -> dict[str, Any]:
    # Keep paths, prompt content, and exception text out of durable artifacts.
    return {
        "schema_version": 1,
        "experiment": "dflash_free_running",
        "row_id": row_id,
        "attempt": attempt,
        "sample_id": str(sample["sample_id"]),
        "retention_percentage": retention,
        "condition": CONDITION_NAMES[retention],
        "status": "error",
        "error_type": type(error).__name__,
        "generation_mode": (
            "unverified_draft_rollout_with_target_bonus"
            if target_bonus_per_block
            else "unverified_draft_rollout"
        ),
        "target_bonus_per_block": bool(target_bonus_per_block),
        "target_bonus_tokens": 0,
        "verification_calls": 0,
    }


def _clean_prompt_record(sample: dict[str, Any]) -> dict[str, Any]:
    record = dict(sample)
    for key in (
        "answer", "reference_answer", "correct_option", "candidates",
        "distractor_provenance", "option_provenance",
    ):
        record.pop(key, None)
    return record


def _load_models(args: argparse.Namespace, *, with_draft: bool) -> tuple[Any, Any, Any | None]:
    device = torch.device(args.device)
    draft_device = torch.device(args.draft_device)
    dtype = _resolve_dtype(args.dtype, device)
    processor, target, _ = _load_target(
        args.target_model,
        device=device,
        dtype=dtype,
        attention=args.target_attention,
        device_map=args.device_map,
        max_memory=args.max_memory,
    )
    draft = None
    if with_draft:
        draft, _ = _load_draft(
            args.checkpoint,
            args.draft_config,
            device=draft_device,
            dtype=dtype,
        )
        # DFlash's eager mask and GQA paths differ from the target model.
        if getattr(draft, "config", None) is not None:
            draft.config._attn_implementation = "sdpa"
    return processor, target, draft


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    records = _selected_records(args)
    protocol = _write_protocol_and_check(args)
    protocol_sha256 = protocol_fingerprint(protocol)
    output_path = Path(args.output_dir) / "raw_rollouts.jsonl"
    target_bonus_per_block = bool(getattr(args, "target_bonus_per_block", False))
    existing_rows = read_jsonl(output_path) if output_path.is_file() else []
    if existing_rows and not args.resume:
        raise FileExistsError("raw_rollouts.jsonl exists; pass --resume or use a new output directory")
    latest = _latest_by_row_id(existing_rows, "row_id")
    processor, target, draft = _load_models(args, with_draft=True)
    device, draft_device = torch.device(args.device), torch.device(args.draft_device)
    decoder = FreeRunningDFlashDecoder(
        target, draft, device=device, draft_device=draft_device,
        token_decoder=lambda token_ids: processor.tokenizer.decode(
            token_ids, skip_special_tokens=True
        ),
    )
    stop_ids = _eos_token_ids(processor, target)
    written = 0
    skipped = 0
    errors = 0
    empty_outputs = 0

    try:
        for sample in records:
            sample_id = str(sample["sample_id"])
            prompt = None
            prompt_error = None
            try:
                # References are removed before prompt construction so they
                # cannot enter target prefill or subsequent feature refresh.
                prompt = _prepare_prompt(
                    processor=processor,
                    target=target,
                    sample=_clean_prompt_record(sample),
                    video_root=str(args.video_root),
                    device=device,
                    target_visual_tokens=args.target_visual_tokens,
                    allow_out_of_tolerance=args.allow_out_of_tolerance,
                )
            except Exception as exc:
                prompt_error = exc

            prompt_obj = prompt[0] if prompt else None
            positions = prompt[1] if prompt else []
            fingerprint = prompt[2] if prompt else str(sample.get("input_fingerprint", ""))
            for retention in _free_running_retentions(args):
                bonus_tag = ":target-bonus" if target_bonus_per_block else ""
                row_id = f"{sample_id}:free-running:{args.feature_refresh}:{args.initial_anchor_control}:retention-{retention}{bonus_tag}"
                previous = latest.get(row_id)
                if previous and (previous.get("status") == "ok" or not args.retry_errors):
                    skipped += 1
                    continue
                attempt = int(previous.get("attempt", 1)) + 1 if previous else 1
                if prompt_error is not None:
                    row = _error_rollout(
                        sample=sample, retention=retention, row_id=row_id,
                        attempt=attempt, error=prompt_error,
                        target_bonus_per_block=target_bonus_per_block,
                    )
                    row["protocol_sha256"] = protocol_sha256
                else:
                    try:
                        keep_mask, retained_positions = build_visual_retention_mask(
                            int(prompt_obj.inputs["input_ids"].shape[1]), positions, retention,
                        )
                        result = decoder.decode(
                            input_ids=prompt_obj.inputs["input_ids"],
                            position_ids=prompt_obj.position_ids,
                            target_kwargs=prompt_obj.target_kwargs,
                            max_new_tokens=args.max_new_tokens,
                            stop_token_ids=stop_ids,
                            context_keep_mask=keep_mask,
                            feature_refresh=args.feature_refresh,
                            target_bonus_per_block=target_bonus_per_block,
                            initial_anchor=(
                                "target_greedy_seed"
                                if args.initial_anchor_control == "target_greedy_seed"
                                else "last_prompt_token"
                            ),
                        )
                        output_start = result.num_input_tokens + result.excluded_prefix_tokens
                        output_ids = result.output_ids[0, output_start:]
                        prediction = _decode_text(processor, output_ids)
                        rollout_metadata = result.as_dict()
                        row = {
                            "schema_version": 2 if target_bonus_per_block else 1,
                            "experiment": "dflash_free_running",
                            "row_id": row_id,
                            "attempt": attempt,
                            "sample_id": sample_id,
                            "question": str(sample.get("question", "")),
                            "input_fingerprint": fingerprint,
                            "status": "ok" if prediction else "empty_output",
                            "generation_mode": rollout_metadata["generation_mode"],
                            "target_bonus_per_block": target_bonus_per_block,
                            "target_bonus_tokens": result.target_bonus_tokens,
                            "feature_refresh": args.feature_refresh,
                            "initial_anchor": result.initial_anchor,
                            "target_seed_token_id": result.target_seed_token_id,
                            "retention_percentage": retention,
                            "condition": CONDITION_NAMES[retention],
                            "prompt_tokens": int(prompt_obj.inputs["input_ids"].shape[1]),
                            "visual_token_count": len(positions),
                            "retained_visual_token_count": len(retained_positions),
                            "retained_visual_positions": retained_positions,
                            "max_new_tokens": int(args.max_new_tokens),
                            "num_output_tokens": result.num_output_tokens,
                            "stop_reason": result.stop_reason,
                            "target_feature_calls": result.target_feature_calls,
                            "target_forward_logits_used_for_proposals": 0,
                            "target_forward_logits_used_for_target_bonus": result.target_bonus_tokens,
                            "target_seed_selected_from_target_logits": result.target_seed_token_id is not None,
                            "proposal_logits_source": "draft_hidden_through_shared_target_lm_head",
                            "verification_calls": 0,
                            "target_selected_output_tokens": (
                                result.excluded_prefix_tokens + result.target_bonus_tokens
                            ),
                            "prediction_token_ids": output_ids.detach().cpu().tolist(),
                            "prediction": prediction,
                            "blocks": result.blocks,
                            "timing": rollout_metadata["timing"],
                            "target_model": protocol["target_model"],
                            "checkpoint": protocol["checkpoint"],
                            "draft_config_sha256": protocol["draft_config_sha256"],
                            "protocol_sha256": protocol_sha256,
                        }
                    except Exception as exc:
                        row = _error_rollout(
                            sample=sample, retention=retention, row_id=row_id,
                            attempt=attempt, error=exc,
                            target_bonus_per_block=target_bonus_per_block,
                        )
                row.setdefault("protocol_sha256", protocol_sha256)
                append_jsonl(output_path, row)
                latest[row_id] = row
                written += 1
                if row.get("status") == "error":
                    errors += 1
                    print(f"[free-running] {sample_id} retention={retention} status={row['status']}", file=sys.stderr)
                elif row.get("status") == "empty_output":
                    empty_outputs += 1
                    print(f"[free-running] {sample_id} retention={retention} status={row['status']}", file=sys.stderr)
                else:
                    print(f"[free-running] {sample_id} retention={retention} tokens={row['num_output_tokens']}", file=sys.stderr)
                del row
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del decoder
        del draft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "phase": "generate",
        "samples_selected": len(records),
        "rollouts_written": written,
        "rollouts_skipped": skipped,
        "rollout_errors": errors,
        "empty_outputs": empty_outputs,
        "raw_journal": str(output_path),
        "output_dir": str(args.output_dir),
    }


def _error_acceptance(
    *, sample: dict[str, Any], retention: int, row_id: str, attempt: int, error: Exception
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "dflash_free_running_acceptance",
        "row_id": row_id,
        "attempt": attempt,
        "sample_id": str(sample["sample_id"]),
        "retention_percentage": retention,
        "condition": CONDITION_NAMES[retention],
        "status": "error",
        "error_type": type(error).__name__,
        "generation_mode": "verified_speculative",
    }


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    """Run normal greedy speculative decoding on the same retention grid."""
    records = _selected_records(args)
    protocol = _write_protocol_and_check(args)
    protocol_sha256 = protocol_fingerprint(protocol)
    output_path = Path(args.output_dir) / "raw_acceptance.jsonl"
    existing_rows = read_jsonl(output_path) if output_path.is_file() else []
    if existing_rows and not args.resume:
        raise FileExistsError(
            "raw_acceptance.jsonl exists; pass --resume or use a new output directory"
        )
    latest = _latest_by_row_id(existing_rows, "row_id")
    processor, target, draft = _load_models(args, with_draft=True)
    device = torch.device(args.device)
    decoder = InstrumentedDFlashDecoder(
        target, draft, device=device, draft_device=torch.device(args.draft_device)
    )
    stop_ids = _eos_token_ids(processor, target)
    written = skipped = errors = mismatches = 0
    try:
        for sample in records:
            sample_id = str(sample["sample_id"])
            prompt_obj = None
            positions: list[int] = []
            fingerprint = ""
            target_ids = None
            target_timing = None
            prompt_error = None
            try:
                prompt_obj, positions, fingerprint, _settings = _prepare_prompt(
                    processor=processor,
                    target=target,
                    sample=_clean_prompt_record(sample),
                    video_root=str(args.video_root),
                    device=device,
                    target_visual_tokens=args.target_visual_tokens,
                    allow_out_of_tolerance=args.allow_out_of_tolerance,
                )
                target_ids, target_timing = _target_greedy(
                    target,
                    prompt_obj,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_ids,
                    device=device,
                )
            except Exception as exc:
                prompt_error = exc

            for retention in RETENTION_PERCENTAGES:
                row_id = f"{sample_id}:verified:retention-{retention}"
                previous = latest.get(row_id)
                if previous and (previous.get("status") in {"ok", "mismatch"} or not args.retry_errors):
                    skipped += 1
                    continue
                attempt = int(previous.get("attempt", 1)) + 1 if previous else 1
                if prompt_error is not None:
                    row = _error_acceptance(
                        sample=sample, retention=retention, row_id=row_id,
                        attempt=attempt, error=prompt_error,
                    )
                    row["protocol_sha256"] = protocol_sha256
                else:
                    try:
                        keep_mask, retained_positions = build_visual_retention_mask(
                            int(prompt_obj.inputs["input_ids"].shape[1]), positions, retention,
                        )
                        result = decoder.decode(
                            input_ids=prompt_obj.inputs["input_ids"],
                            position_ids=prompt_obj.position_ids,
                            target_kwargs=prompt_obj.target_kwargs,
                            max_new_tokens=args.max_new_tokens,
                            stop_token_ids=stop_ids,
                            prefill_target_context_keep_mask=keep_mask,
                        )
                        prompt_length = int(prompt_obj.inputs["input_ids"].shape[1])
                        predicted_ids = result.output_ids[0, prompt_length:]
                        reference_ids = target_ids[0, prompt_length:]
                        predicted_list = predicted_ids.detach().cpu().tolist()
                        reference_list = reference_ids.detach().cpu().tolist()
                        outputs_match = predicted_list == reference_list
                        metrics = result.as_dict()
                        row = {
                            "schema_version": 1,
                            "experiment": "dflash_free_running_acceptance",
                            "row_id": row_id,
                            "attempt": attempt,
                            "sample_id": sample_id,
                            "question": str(sample.get("question", "")),
                            "input_fingerprint": fingerprint,
                            "status": "ok" if outputs_match and predicted_list else (
                                "empty_output" if outputs_match else "mismatch"
                            ),
                            "generation_mode": "verified_speculative",
                            "retention_percentage": retention,
                            "condition": CONDITION_NAMES[retention],
                            "prompt_tokens": prompt_length,
                            "visual_token_count": len(positions),
                            "retained_visual_token_count": len(retained_positions),
                            "retained_visual_positions": retained_positions,
                            "max_new_tokens": int(args.max_new_tokens),
                            "num_output_tokens": len(predicted_list),
                            "prediction_token_ids": predicted_list,
                            "prediction": _decode_text(processor, predicted_ids),
                            "target_reference_num_tokens": len(reference_list),
                            "target_reference_token_sha256": _sha256_tokens(reference_list),
                            "outputs_match_target_greedy": outputs_match,
                            "target_forward_calls": int(metrics["target_forward_calls"]),
                            "verification_calls": max(0, int(metrics["target_forward_calls"]) - 1),
                            "tau_proposal": metrics["tau_proposal"],
                            "tau_effective": metrics["tau_effective"],
                            "acceptance_rounds": metrics["acceptance_rounds"],
                            "target_reference_timing": target_timing,
                            "timing": metrics["timing"],
                            "target_model": protocol["target_model"],
                            "checkpoint": protocol["checkpoint"],
                            "draft_config_sha256": protocol["draft_config_sha256"],
                            "protocol_sha256": protocol_sha256,
                        }
                    except Exception as exc:
                        row = _error_acceptance(
                            sample=sample, retention=retention, row_id=row_id,
                            attempt=attempt, error=exc,
                        )
                row.setdefault("protocol_sha256", protocol_sha256)
                append_jsonl(output_path, row)
                latest[row_id] = row
                written += 1
                if row.get("status") == "error":
                    errors += 1
                elif row.get("status") == "mismatch":
                    mismatches += 1
                print(
                    f"[acceptance] {sample_id} retention={retention} status={row['status']}",
                    file=sys.stderr,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del decoder
        del draft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "phase": "acceptance",
        "samples_selected": len(records),
        "rows_written": written,
        "rows_skipped": skipped,
        "runtime_errors": errors,
        "target_parity_mismatches": mismatches,
        "journal": str(output_path),
        "output_dir": str(args.output_dir),
    }


def run_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    """Compare per-token proposals on fixed target-greedy prefixes."""
    if args.sensitivity_limit <= 0:
        raise ValueError("--sensitivity-limit must be positive")
    records = _selected_records(args)[:args.sensitivity_limit]
    protocol = _write_protocol_and_check(args)
    protocol_sha256 = protocol_fingerprint(protocol)
    output_path = Path(args.output_dir) / "raw_sensitivity.jsonl"
    existing_rows = read_jsonl(output_path) if output_path.is_file() else []
    if existing_rows and not args.resume:
        raise FileExistsError(
            "raw_sensitivity.jsonl exists; pass --resume or use a new output directory"
        )
    latest = _latest_by_row_id(existing_rows, "row_id")
    processor, target, draft = _load_models(args, with_draft=True)
    decoder = FreeRunningDFlashDecoder(
        target, draft,
        device=torch.device(args.device),
        draft_device=torch.device(args.draft_device),
    )
    written = skipped = errors = 0
    try:
        for sample in records:
            sample_id = str(sample["sample_id"])
            prompt_obj = None
            positions: list[int] = []
            fingerprint = ""
            answer_ids = None
            answer_hash = None
            prompt_error = None
            try:
                prompt_obj, positions, fingerprint, _settings = _prepare_prompt(
                    processor=processor,
                    target=target,
                    sample=_clean_prompt_record(sample),
                    video_root=str(args.video_root),
                    device=torch.device(args.device),
                    target_visual_tokens=args.target_visual_tokens,
                    allow_out_of_tolerance=args.allow_out_of_tolerance,
                )
                target_output, _target_timing = _target_greedy(
                    target,
                    prompt_obj,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=_eos_token_ids(processor, target),
                    device=torch.device(args.device),
                )
                answer_ids = target_output[0, prompt_obj.inputs["input_ids"].shape[1]:]
                answer_hash = _sha256_tokens(answer_ids.detach().cpu().tolist())
            except Exception as exc:
                prompt_error = exc

            for retention in RETENTION_PERCENTAGES:
                row_id = f"{sample_id}:sensitivity:{args.feature_refresh}:retention-{retention}"
                previous = latest.get(row_id)
                if previous and (previous.get("status") == "ok" or not args.retry_errors):
                    skipped += 1
                    continue
                attempt = int(previous.get("attempt", 1)) + 1 if previous else 1
                if prompt_error is not None:
                    row = {
                        "schema_version": 1,
                        "experiment": "dflash_teacher_forced_visual_sensitivity",
                        "row_id": row_id,
                        "attempt": attempt,
                        "sample_id": sample_id,
                        "retention_percentage": retention,
                        "condition": CONDITION_NAMES[retention],
                        "status": "error",
                        "error_type": type(prompt_error).__name__,
                        "protocol_sha256": protocol_sha256,
                    }
                else:
                    try:
                        keep_mask, retained_positions = build_visual_retention_mask(
                            int(prompt_obj.inputs["input_ids"].shape[1]), positions, retention,
                        )
                        probe = decoder.probe_teacher_forced(
                            input_ids=prompt_obj.inputs["input_ids"],
                            position_ids=prompt_obj.position_ids,
                            target_kwargs=prompt_obj.target_kwargs,
                            answer_ids=answer_ids,
                            context_keep_mask=keep_mask,
                            feature_refresh=args.feature_refresh,
                        )
                        token_rows = []
                        tokenizer = processor.tokenizer
                        special_ids = set(getattr(tokenizer, "all_special_ids", []))
                        for block in probe["probe_positions"]:
                            start = int(block["answer_position_start"])
                            distribution_statistics = block.get(
                                "proposal_distribution_statistics", []
                            )
                            for offset, (reference_id, proposal_id, matches) in enumerate(zip(
                                block["reference_token_ids"],
                                block["proposed_token_ids"],
                                block["proposal_matches_reference"],
                            )):
                                reference_id = int(reference_id)
                                proposal_id = int(proposal_id)
                                reference_text = tokenizer.decode(
                                    [reference_id], skip_special_tokens=False
                                )
                                proposal_text = tokenizer.decode(
                                    [proposal_id], skip_special_tokens=False
                                )
                                token_row = {
                                    "answer_position": start + offset,
                                    "target_token_id": reference_id,
                                    "draft_proposal_token_id": proposal_id,
                                    "target_token": reference_text,
                                    "draft_proposal_token": proposal_text,
                                    "target_surface_class": classify_token_surface(
                                        reference_text, is_special=reference_id in special_ids
                                    ),
                                    "proposal_surface_class": classify_token_surface(
                                        proposal_text, is_special=proposal_id in special_ids
                                    ),
                                    "proposal_matches_target_prefix": bool(matches),
                                }
                                if offset < len(distribution_statistics):
                                    token_row.update(distribution_statistics[offset])
                                token_rows.append(token_row)
                        row = {
                            "schema_version": 1,
                            "experiment": "dflash_teacher_forced_visual_sensitivity",
                            "row_id": row_id,
                            "attempt": attempt,
                            "sample_id": sample_id,
                            "input_fingerprint": fingerprint,
                            "status": "ok" if token_rows else "empty_output",
                            "generation_mode": "teacher_forced_draft_sensitivity",
                            "feature_refresh": args.feature_refresh,
                            "retention_percentage": retention,
                            "condition": CONDITION_NAMES[retention],
                            "visual_token_count": len(positions),
                            "retained_visual_token_count": len(retained_positions),
                            "retained_visual_positions": retained_positions,
                            "teacher_forced_answer_sha256": answer_hash,
                            "teacher_forced_answer_tokens": len(token_rows),
                            "max_new_tokens": int(args.max_new_tokens),
                            "proposal_logits_source": "draft_hidden_through_shared_target_lm_head",
                            "target_feature_calls": probe["target_feature_calls"],
                            "probe_tokens": token_rows,
                            "target_model": protocol["target_model"],
                            "checkpoint": protocol["checkpoint"],
                            "draft_config_sha256": protocol["draft_config_sha256"],
                            "protocol_sha256": protocol_sha256,
                        }
                    except Exception as exc:
                        row = {
                            "schema_version": 1,
                            "experiment": "dflash_teacher_forced_visual_sensitivity",
                            "row_id": row_id,
                            "attempt": attempt,
                            "sample_id": sample_id,
                            "retention_percentage": retention,
                            "condition": CONDITION_NAMES[retention],
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "protocol_sha256": protocol_sha256,
                        }
                row.setdefault("protocol_sha256", protocol_sha256)
                append_jsonl(output_path, row)
                latest[row_id] = row
                written += 1
                if row.get("status") == "error":
                    errors += 1
                print(
                    f"[visual sensitivity] {sample_id} retention={retention} status={row['status']}",
                    file=sys.stderr,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del decoder
        del draft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "phase": "sensitivity",
        "samples_selected": len(records),
        "rows_written": written,
        "rows_skipped": skipped,
        "runtime_errors": errors,
        "journal": str(output_path),
    }


def _tokenize_judge_prompt(processor: Any, text: str) -> dict[str, torch.Tensor]:
    encoded = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if torch.is_tensor(encoded):
        return {"input_ids": encoded}
    return {key: value for key, value in encoded.items() if torch.is_tensor(value)}


def _judge_pair(
    *, processor: Any, target: Any, pair: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    text = make_judge_prompt(
        pair["text_a"], pair["text_b"],
        capped_a=bool(pair.get("capped_a")),
        capped_b=bool(pair.get("capped_b")),
        truncated_a=bool(pair.get("truncated_a")),
        truncated_b=bool(pair.get("truncated_b")),
    )
    encoded = _tokenize_judge_prompt(processor, text)
    device = torch.device(args.device)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    position_ids = _compute_position_ids(target, encoded)
    inputs = dict(encoded)
    target_kwargs = {
        key: value for key, value in encoded.items() if key not in {"input_ids", "position_ids"}
    }
    judge_prompt = PreparedVideoPrompt(
        record={"sample_id": pair["sample_id"], "judge_only": True},
        video_path=Path("."),
        inputs=inputs,
        position_ids=position_ids,
        target_kwargs=target_kwargs,
        frame_counts=(),
        video_grid_thw=None,
    )
    ids, _ = _target_greedy(
        target,
        judge_prompt,
        max_new_tokens=args.judge_max_new_tokens,
        stop_token_ids=_eos_token_ids(processor, target),
        device=device,
    )
    generated_ids = ids[0, encoded["input_ids"].shape[1]:]
    answer = _decode_text(processor, generated_ids)
    generated = generated_ids.detach().cpu().tolist()
    stop_ids = set(_eos_token_ids(processor, target))
    ended_at_cap = (
        len(generated) >= int(args.judge_max_new_tokens)
        and (not generated or int(generated[-1]) not in stop_ids)
    )
    parse_error = None
    parsed = None
    try:
        parsed = parse_judge_response(answer)
    except ValueError as exc:
        parse_error = type(exc).__name__
    return {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "sample_id": pair["sample_id"],
        "retention_percentage": int(pair["retention_percentage"]),
        "analysis_variant": pair.get("analysis_variant", "full_output"),
        "status": "ok" if parsed else "judge_parse_error",
        "judge": parsed,
        "judge_response": answer,
        "pair_fingerprint": pair.get("pair_fingerprint"),
        "judge_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "judge_response_truncated_at_cap": ended_at_cap,
        "protocol_sha256": getattr(args, "_run_protocol_sha256", None),
        "error_type": parse_error,
    }


def run_judging(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir)
    raw_path = root / "raw_rollouts.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError("run --phase generate before --phase score")
    protocol = _write_protocol_and_check(args)
    args._run_protocol_sha256 = protocol_fingerprint(protocol)
    raw = read_jsonl(raw_path)
    latest_raw = _latest_by_row_id(raw, "row_id")
    pairs, blind_key = build_blind_pairs(latest_raw.values(), seed=args.seed)
    if not pairs:
        if not args.resume and any(
            (root / name).is_file()
            for name in ("judge_pairs.jsonl", "judge_matched_prefix_pairs.jsonl")
        ):
            raise FileExistsError(
                "judge journals exist; pass --resume or use a new output directory"
            )
        root.mkdir(parents=True, exist_ok=True)
        (root / "blind_key.json").write_text("{}\n", encoding="utf-8")
        (root / "matched_prefix_key.json").write_text("{}\n", encoding="utf-8")
        coverage = {
            "prefix_tokens": int(args.matched_prefix_tokens),
            "eligible_pairs": 0,
            "coverage": {str(value): {"eligible": 0, "excluded_short": 0}
                         for value in (75, 50, 25, 0)},
        }
        (root / "matched_prefix_coverage.json").write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "phase": "score",
            "candidate_pairs": 0,
            "pairs_to_score": 0,
            "judge_rows_written": 0,
            "judge_rows_skipped": 0,
            "judge_errors": 0,
            "matched_prefix_pairs": 0,
            "matched_prefix_rows_written": 0,
            "matched_prefix_rows_skipped": 0,
            "matched_prefix_errors": 0,
            "judge_journal": str(root / "judge_pairs.jsonl"),
            "blind_key": str(root / "blind_key.json"),
        }
    output_path = root / "judge_pairs.jsonl"
    old_rows = read_jsonl(output_path) if output_path.is_file() else []
    if old_rows and not args.resume:
        raise FileExistsError("judge_pairs.jsonl exists; pass --resume or use a new output directory")
    matched_path = root / "judge_matched_prefix_pairs.jsonl"
    old_matched = read_jsonl(matched_path) if matched_path.is_file() else []
    if old_matched and not args.resume:
        raise FileExistsError(
            "judge_matched_prefix_pairs.jsonl exists; pass --resume or use a new output directory"
        )
    latest_scores = _latest_by_row_id(old_rows, "pair_id") if old_rows else {}
    missing = [
        pair for pair in pairs
        if pair["pair_id"] not in latest_scores
        or latest_scores[pair["pair_id"]].get("pair_fingerprint") != pair.get("pair_fingerprint")
        or latest_scores[pair["pair_id"]].get("protocol_sha256") != args._run_protocol_sha256
        or (args.retry_errors and latest_scores[pair["pair_id"]].get("status") != "ok")
    ]
    processor, target, _ = _load_models(args, with_draft=False)
    completed = failed = skipped = 0
    matched_written = matched_failed = matched_skipped = 0
    matched_coverage: dict[str, Any] = {}
    try:
        for pair in pairs:
            old = latest_scores.get(pair["pair_id"])
            if old and old.get("pair_fingerprint") == pair.get("pair_fingerprint") and (
                old.get("protocol_sha256") == args._run_protocol_sha256
                and (old.get("status") == "ok" or not args.retry_errors)
            ):
                skipped += 1
                continue
            attempt = int(old.get("attempt", 1)) + 1 if old else 1
            try:
                row = _judge_pair(processor=processor, target=target, pair=pair, args=args)
                row["attempt"] = attempt
            except Exception as exc:
                row = {
                    "schema_version": 1,
                    "pair_id": pair["pair_id"],
                    "pair_fingerprint": pair.get("pair_fingerprint"),
                    "protocol_sha256": args._run_protocol_sha256,
                    "sample_id": pair["sample_id"],
                    "retention_percentage": pair["retention_percentage"],
                    "attempt": attempt,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            append_jsonl(output_path, row)
            latest_scores[pair["pair_id"]] = row
            if row.get("status") == "ok":
                completed += 1
            else:
                failed += 1
            print(f"[coherence judge] pair={pair['pair_id']} status={row['status']}", file=sys.stderr)

        matched_pairs, matched_key, matched_coverage = build_matched_prefix_pairs(
            latest_raw.values(),
            seed=args.seed,
            prefix_tokens=args.matched_prefix_tokens,
            decode_token_ids=lambda token_ids: processor.tokenizer.decode(
                list(token_ids), skip_special_tokens=True
            ).strip(),
        )
        latest_matched = _latest_by_row_id(old_matched, "pair_id") if old_matched else {}
        for pair in matched_pairs:
            old = latest_matched.get(pair["pair_id"])
            if old and old.get("pair_fingerprint") == pair.get("pair_fingerprint") and (
                old.get("protocol_sha256") == args._run_protocol_sha256
                and (old.get("status") == "ok" or not args.retry_errors)
            ):
                matched_skipped += 1
                continue
            attempt = int(old.get("attempt", 1)) + 1 if old else 1
            try:
                row = _judge_pair(processor=processor, target=target, pair=pair, args=args)
                row["attempt"] = attempt
            except Exception as exc:
                row = {
                    "schema_version": 1,
                    "pair_id": pair["pair_id"],
                    "pair_fingerprint": pair.get("pair_fingerprint"),
                    "protocol_sha256": args._run_protocol_sha256,
                    "sample_id": pair["sample_id"],
                    "retention_percentage": pair["retention_percentage"],
                    "analysis_variant": pair["analysis_variant"],
                    "attempt": attempt,
                    "status": "error",
                    "error_type": type(exc).__name__,
                }
            append_jsonl(matched_path, row)
            latest_matched[pair["pair_id"]] = row
            if row.get("status") == "ok":
                matched_written += 1
            else:
                matched_failed += 1
            print(
                f"[matched-prefix judge] pair={pair['pair_id']} status={row['status']}",
                file=sys.stderr,
            )
    finally:
        del target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Save a condition map away from judge inputs; judges see only A/B texts.
    (root / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "matched_prefix_key.json").write_text(
        json.dumps(matched_key, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "matched_prefix_coverage.json").write_text(
        json.dumps(matched_coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "phase": "score",
        "candidate_pairs": len(pairs),
        "pairs_to_score": len(missing),
        "judge_rows_written": completed + failed,
        "judge_rows_skipped": skipped,
        "judge_errors": failed,
        "matched_prefix_pairs": int(matched_coverage.get("eligible_pairs", 0)),
        "matched_prefix_rows_written": matched_written,
        "matched_prefix_rows_skipped": matched_skipped,
        "matched_prefix_errors": matched_failed,
        "judge_journal": str(output_path),
        "blind_key": str(root / "blind_key.json"),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_free_running import (
        analyze_free_running,
    )

    protocol = _write_protocol_and_check(args)
    return analyze_free_running(
        output_dir=args.output_dir,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        review_samples=args.review_samples,
        protocol=protocol,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _resolve_paths(args)
    if args.max_new_tokens < 1 or args.judge_max_new_tokens < 1:
        raise ValueError("generation and judge token budgets must be positive")
    if args.sensitivity_limit < 1:
        raise ValueError("--sensitivity-limit must be positive")
    if args.matched_prefix_tokens < 1:
        raise ValueError("--matched-prefix-tokens must be positive")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")
    if args.phase == "preflight":
        return run_preflight(args)
    result: dict[str, Any] = {"output_dir": str(args.output_dir)}
    if args.phase in {"generate", "all"}:
        result["generate"] = run_generation(args)
    if args.phase in {"sensitivity", "all"}:
        result["sensitivity"] = run_sensitivity(args)
    if args.phase in {"acceptance", "all"}:
        result["acceptance"] = run_acceptance(args)
    if args.phase in {"score", "all"}:
        result["score"] = run_judging(args)
    if args.phase in {"analyze", "all"}:
        result["analysis"] = run_analysis(args)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
