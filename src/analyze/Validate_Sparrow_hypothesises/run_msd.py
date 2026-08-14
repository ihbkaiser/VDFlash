"""Run the MSD visual-length and draft-retention experiments from Figure 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from .dataset import load_vdc_manifest, write_jsonl
from .paper_contract import DEFAULT_CONTRACT
from .runtime import (
    RuntimeUnavailableError,
    build_qwen2vl_video_processor,
    compact_qwen2vl_prefill,
    generate_msd_full_video,
    generate_msd_retention_video,
    move_batch_to_device,
    prepare_qwen2vl_prefill,
    process_video,
    load_msd_qwen2vl,
    model_device,
    require_cuda,
    validate_native_prefill_parity,
)


def _new_tokens(output: torch.Tensor, input_length: int) -> list[int]:
    return output[0, input_length:].detach().to("cpu").tolist()


def _hash_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()[:16]


def _target_reference(base_model: Any, batch: Any, input_length: int, max_new_tokens: int) -> tuple[list[int], float, float]:
    start = time.perf_counter()
    with torch.inference_mode():
        base_model(**batch, use_cache=True, return_dict=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with torch.inference_mode():
        output = base_model.generate(
            **batch,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_to_end_seconds = time.perf_counter() - start
    return _new_tokens(output, input_length), prefill_seconds, end_to_end_seconds


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_calibration(path: str | None, targets: list[int]) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    rows = _read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if int(row.get("target_visual_tokens", -1)) not in targets:
            continue
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    return grouped


def _load_visual_scores(path: str | None) -> dict[tuple[str, int, str], list[float]]:
    """Load Figure 2 visual scores for Last-Instr. or All-Text retention."""

    if path is None:
        return {}
    grouped: dict[tuple[str, int, str], list[tuple[int, float]]] = {}
    for row in _read_jsonl(path):
        if row.get("paper_figure") != "Figure 2" or row.get("modality") != "visual":
            continue
        policy = str(row.get("attention_policy") or "last_instruction")
        key = (str(row["sample_id"]), int(row.get("visual_token_count", -1)), policy)
        grouped.setdefault(key, []).append((int(row["visual_index"]), float(row["attention_weight"])))
    return {
        key: [value for _index, value in sorted(values)]
        for key, values in grouped.items()
    }


def _load_existing_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load previously written rows for resume; tolerate a truncated tail.

    The runner writes incrementally so a killed stage can be resumed without
    losing completed jobs.  A partially written final line (crash mid-write)
    is dropped.
    """

    existing: dict[str, dict[str, Any]] = {}
    if not Path(path).exists():
        return existing
    with Path(path).open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated tail from a previous kill
        if isinstance(row, dict) and row.get("row_id"):
            existing[str(row["row_id"])] = row
    return existing


def run(args: argparse.Namespace) -> int:
    try:
        require_cuda()
    except RuntimeUnavailableError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_path = Path(args.manifest)
    samples = load_vdc_manifest(manifest_path, args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    targets = list(args.visual_targets or DEFAULT_CONTRACT.visual_token_milestones)
    calibration = _load_calibration(args.calibration, targets)
    visual_scores = _load_visual_scores(args.selection_scores)
    if args.calibration and not calibration:
        raise SystemExit("Calibration file contains no requested visual-token targets")
    existing = _load_existing_rows(args.output)
    processor = build_qwen2vl_video_processor(args.base_model, args.min_pixels, args.max_pixels)
    model = load_msd_qwen2vl(args.base_model, args.msd_model)
    base_model = model.base_model
    device = model_device(base_model)
    rows: list[dict[str, Any]] = list(existing.values())
    jobs: list[tuple[Any, dict[str, Any] | None]] = []
    for sample in samples:
        points = calibration.get(sample.sample_id) if args.calibration else None
        if points:
            for point in sorted(points, key=lambda row: int(row["target_visual_tokens"])):
                if point.get("status") != "ok" and not args.allow_out_of_tolerance:
                    raise SystemExit(
                        f"Calibration point {sample.sample_id}:{point['target_visual_tokens']} is "
                        f"{point.get('status')}; pass --allow-out-of-tolerance to run it"
                    )
                jobs.append((sample, point))
        else:
            jobs.append((sample, None))
    # Resume: skip jobs whose complete row set is already in the output file.
    def _job_row_ids(sample: Any, point: dict[str, Any] | None) -> set[str]:
        row_ids = {f"{sample.sample_id}:error"}
        conditions = ["full"] if args.condition == "full" else ["retention"] if args.condition == "retention" else ["full", "retention"]
        for condition in conditions:
            percentages = [100.0] if condition == "full" else list(args.retention_percentages)
            for percentage in percentages:
                row_ids.add(
                    f"{sample.sample_id}:{'full' if condition == 'full' else f'retention-{percentage:g}'}"
                )
        return row_ids

    pending: list[tuple[Any, dict[str, Any] | None]] = []
    for sample, point in jobs:
        if _job_row_ids(sample, point).issubset(existing):
            continue
        pending.append((sample, point))
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"Resume: {skipped}/{len(jobs)} jobs already complete, {len(pending)} pending")
    total_jobs = len(pending)
    for index, (sample, point) in enumerate(pending, start=1):
        target_text = point.get("target_visual_tokens") if point else "native"
        print(f"[{index}/{total_jobs}] {sample.sample_id} target={target_text}", flush=True)
        job_rows: list[dict[str, Any]] = []
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                job_rows = _run_job(
                    args, model, base_model, processor, device, sample, point, visual_scores
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - transient video/OOM errors
                last_exc = exc
                print(f"  ERROR {sample.sample_id} attempt {attempt}: {exc}", flush=True)
                if attempt == 1:
                    time.sleep(5)
        if last_exc is not None:
            # Record an explicit error row (audit-compatible) and continue with
            # the next sample instead of aborting the whole multi-hour stage.
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
                "error": str(last_exc),
                "calibration_target_visual_tokens": point.get("target_visual_tokens") if point else None,
                "calibration_status": point.get("status") if point else "not_requested",
            })
            continue
        rows.extend(job_rows)
        # Incremental write: a killed stage can resume without losing work.
        write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} MSD rows to {args.output}")
    return 0


def _run_job(
    args: argparse.Namespace,
    model: Any,
    base_model: Any,
    processor: Any,
    device: torch.device,
    sample: Any,
    point: dict[str, Any] | None,
    visual_scores: dict[tuple[str, int, str], list[float]],
) -> list[dict[str, Any]]:
    """Run every Figure 1(a)/1(b) condition for one (sample, milestone) job."""

    if point and point.get("candidate_settings"):
        settings = point["candidate_settings"]
        fps = float(settings["frames"]) / max(float(sample.duration_sec or 1.0), 1e-3)
        max_pixels = int(settings["max_pixels"])
    else:
        fps = args.fps
        # Bound the native (non-calibrated) path by the explicit pixel
        # budget instead of letting the processor pick an unbounded
        # resolution that OOMs the T4 prefill.
        max_pixels = args.max_pixels
    batch = process_video(
        processor,
        sample.resolved_path(args.dataset_root),
        sample.question,
        fps,
        max_pixels=max_pixels,
    )
    batch = move_batch_to_device(batch, device)
    prepared = prepare_qwen2vl_prefill(base_model, batch, device)
    parity = validate_native_prefill_parity(base_model, batch, device)
    if not parity["valid"]:
        raise RuntimeError(f"native prefill parity failed for {sample.sample_id}: {parity}")
    target_tokens, ar_prefill, ar_e2e = _target_reference(base_model, batch, prepared.input_ids.shape[1], args.max_new_tokens)
    rows: list[dict[str, Any]] = []
    conditions = ["full"] if args.condition == "full" else ["retention"] if args.condition == "retention" else ["full", "retention"]
    for condition in conditions:
        percentages = [100.0] if condition == "full" else list(args.retention_percentages)
        for percentage in percentages:
            scores = None
            score_policy = {
                "top_attention": "last_instruction",
                "last_instruction": "last_instruction",
                "all_text": "all_text",
            }.get(args.selection)
            score_key = (sample.sample_id, int(prepared.video_positions.numel()), score_policy) if score_policy else None
            if score_policy:
                scores = visual_scores.get(score_key)
                if scores is None:
                    raise SystemExit(
                        f"Missing attention scores for {score_key}; run the Figure 2 runner first "
                        "or use --selection uniform"
                    )
            draft = prepared if condition == "full" else compact_qwen2vl_prefill(prepared, percentage, scores)
            if condition == "full":
                speculative_ids, trace = generate_msd_full_video(model, draft, args.max_new_tokens)
            else:
                # The target keeps the full video; only the draft context is
                # compacted (paper Figure 1(b) setup).
                speculative_ids, trace = generate_msd_retention_video(
                    model, prepared, draft, args.max_new_tokens
                )
            # The decoded sequence always starts from the full target
            # context, so new tokens are sliced from the full length.
            speculative_tokens = _new_tokens(speculative_ids, prepared.input_ids.shape[1])
            common = 0
            for _target, _spec in zip(target_tokens, speculative_tokens):
                if _target != _spec:
                    break
                common += 1
            lossless = common == len(target_tokens)
            row_id = f"{sample.sample_id}:{'full' if condition == 'full' else f'retention-{percentage:g}'}"
            row = {
                "row_id": row_id,
                "paper_figure": "Figure 1(a)" if condition == "full" else "Figure 1(b)",
                "sample_id": sample.sample_id,
                "target_model": args.base_model,
                "temperature": 0.0,
                "target_visual_tokens": int(prepared.video_positions.numel()),
                "actual_visual_tokens": int(draft.video_positions.numel()),
                "full_target_visual_tokens": int(prepared.video_positions.numel()),
                "target_input_fingerprint": prepared.input_fingerprint,
                "target_input_fingerprint_reference": prepared.input_fingerprint,
                "draft_input_fingerprint": draft.input_fingerprint,
                "target_output_ids": target_tokens,
                "speculative_output_ids": speculative_tokens,
                "lossless": lossless,
                "lossless_prefix_length": common,
                "target_output_hash": _hash_tokens(target_tokens),
                "speculative_output_hash": _hash_tokens(speculative_tokens),
                "accepted_prefix_tokens": trace["accepted_prefix_tokens"],
                "acceptance_trace": trace["acceptance_trace"],
                "verification_steps": trace["verification_steps"],
                "prefill_seconds": trace["prefill_seconds"],
                "decode_seconds": trace["decode_seconds"],
                "end_to_end_seconds": trace["end_to_end_seconds"],
                "ar_prefill_seconds": ar_prefill,
                "ar_decode_seconds": max(0.0, ar_e2e - ar_prefill),
                "ar_end_to_end_seconds": ar_e2e,
                "decode_speedup": (
                    max(0.0, ar_e2e - ar_prefill) / trace["decode_seconds"]
                    if trace["decode_seconds"]
                    else None
                ),
                "end_to_end_speedup": (
                    ar_e2e / trace["end_to_end_seconds"]
                    if trace["end_to_end_seconds"]
                    else None
                ),
                "video_token_count": trace["video_token_count"],
                "native_prefill_parity": parity,
                "condition": condition,
                "retention_percentage": float(percentage) if condition == "retention" else 100.0,
                "selection_policy": args.selection,
                "calibration_target_visual_tokens": point.get("target_visual_tokens") if point else None,
                "calibration_status": point.get("status") if point else "not_requested",
                "calibration_relative_error": point.get("relative_error") if point else None,
                "calibration_candidate_id": point.get("candidate_id") if point else None,
                "fps": fps,
                "max_pixels": max_pixels,
            }
            rows.append(row)
            if lossless:
                print(f"  {condition} {percentage:g}% lossless: yes ({common}/{len(target_tokens)} tokens)", flush=True)
            else:
                print(f"  {condition} {percentage:g}% lossless: NO (prefix {common}/{len(target_tokens)})", flush=True)
                if args.strict_losslessness:
                    raise RuntimeError(f"losslessness failed for {row_id}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    parser.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    parser.add_argument("--base-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-model", default="lucylyn/MSD-Qwen2VL-7B-Instruct")
    parser.add_argument("--output", default="results/sparrow_validation/msd_full.jsonl")
    parser.add_argument("--condition", choices=("full", "retention", "both"), default="full")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--calibration",
        help="Measured calibration JSONL. With this flag, one job is run for each requested milestone.",
    )
    parser.add_argument(
        "--visual-targets",
        type=int,
        nargs="+",
        help="Milestones to select from --calibration (default: 400 3000 13000 25000).",
    )
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument(
        "--retention-percentages",
        type=float,
        nargs="+",
        default=list(DEFAULT_CONTRACT.retention_percentages),
    )
    parser.add_argument(
        "--selection",
        choices=("uniform", "top_attention", "last_instruction", "all_text"),
        default="uniform",
        help="How retention keeps visual tokens. last_instruction/all_text consume Figure 2 scores; top_attention is an alias for last_instruction.",
    )
    parser.add_argument("--selection-scores", help="Figure 2 JSONL used by --selection top_attention.")
    parser.add_argument(
        "--strict-losslessness",
        action="store_true",
        help="Fail hard (exit 1) when the target output is not a strict prefix "
        "of the speculative output. Off by default so a near-tie divergence at "
        "one token records lossless_prefix_length instead of aborting.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))
