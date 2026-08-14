"""CLI for preparing, auditing and reporting Sparrow validation runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .audit import audit_coverage, audit_losslessness, audit_rows
from .calibrate import calibrate_sample, candidate_grid, write_calibration
from .dataset import load_vdc_manifest, planned_calibration, write_jsonl
from .paper_contract import load_contract, validate_contract
from .preflight import run_preflight, write_preflight
from .report import build_report, read_jsonl, write_report


def _contract(args: argparse.Namespace):
    contract = load_contract(args.contract)
    errors = validate_contract(contract)
    if errors:
        raise SystemExit("Invalid paper contract: " + "; ".join(errors))
    return contract


def _cmd_preflight(args: argparse.Namespace) -> int:
    contract = _contract(args)
    result = run_preflight(contract, args.repo_root, args.require_gpu, args.require_models)
    write_preflight(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 2


def _cmd_prepare(args: argparse.Namespace) -> int:
    contract = _contract(args)
    samples = load_vdc_manifest(args.manifest, args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    rows = []
    for sample in samples:
        for point in planned_calibration(sample.sample_id, contract.visual_token_milestones):
            rows.append({
                "row_id": f"{sample.sample_id}:{point.target_visual_tokens}",
                "paper_figure": "Figure 1(a)",
                "sample_id": sample.sample_id,
                "sample_fingerprint": sample.fingerprint(),
                "video_path": str(sample.resolved_path(args.dataset_root)),
                "question": sample.question,
                "reference_answer": sample.answer,
                **asdict(point),
                "calibration_status": "pending",
                "runtime_status": "not_run",
            })
    write_jsonl(args.output, rows)
    print(f"Prepared {len(rows)} planned rows for {len(samples)} samples: {args.output}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    contract = _contract(args)
    samples = load_vdc_manifest(args.manifest, args.dataset_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    try:
        from .runtime import build_qwen2vl_video_processor

        processor = build_qwen2vl_video_processor(args.model, args.min_pixels, args.max_pixels)
    except Exception as exc:
        raise SystemExit(f"Cannot initialize Qwen2-VL processor: {exc}") from exc
    candidates = candidate_grid()
    if args.grid_frames or args.grid_pixels:
        frames = (
            [int(value) for value in args.grid_frames.split(",")]
            if args.grid_frames
            else tuple(candidate.frames for candidate in candidates)
        )
        pixels = (
            [int(value) for value in args.grid_pixels.split(",")]
            if args.grid_pixels
            else tuple(candidate.max_pixels for candidate in candidates)
        )
        from .calibrate import VideoCandidate

        candidates = [VideoCandidate(int(frames_value), int(pixels_value))
                      for frames_value in frames for pixels_value in pixels]
        print(f"Using reduced grid: {len(candidates)} candidates", flush=True)
    rows = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] calibrating {sample.sample_id}", flush=True)
        try:
            sample_rows = calibrate_sample(
                sample,
                args.dataset_root,
                processor,
                contract.visual_token_milestones,
                contract.calibration_tolerance,
                candidates,
            )
        except Exception as exc:  # pragma: no cover - transient video read failures
            # A broken/transient video must not abort the whole calibration;
            # record an explicit error row and continue with the next sample.
            print(f"  ERROR {sample.sample_id}: {exc}", flush=True)
            sample_rows = []
            for target in contract.visual_token_milestones:
                sample_rows.append({
                    "row_id": f"{sample.sample_id}:{target}",
                    "paper_figure": "Figure 1(a)",
                    "sample_id": sample.sample_id,
                    "sample_fingerprint": sample.fingerprint(),
                    "video_path": str(sample.resolved_path(args.dataset_root)),
                    "question": sample.question,
                    "reference_answer": sample.answer,
                    "target_visual_tokens": int(target),
                    "actual_visual_tokens": None,
                    "candidate_id": None,
                    "status": "error",
                    "relative_error": None,
                    "source": "processor",
                    "calibration_status": "error",
                    "runtime_status": "not_run",
                    "error": str(exc),
                })
        rows.extend(sample_rows)
        # Write incrementally so a crash never discards completed samples.
        write_calibration(args.output, rows)
    print(f"Wrote {len(rows)} measured calibration rows: {args.output}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    contract = _contract(args)
    rows = read_jsonl(args.input)
    report = audit_rows(rows, contract)
    coverage = audit_coverage(rows, contract)
    if any("target_output_ids" in row or "speculative_output_ids" in row for row in rows):
        lossless = audit_losslessness(rows)
        report.issues.extend(lossless.issues)
        report.valid = report.valid and lossless.valid
    payload = report.to_dict()
    payload["coverage"] = coverage.to_dict()
    report.valid = report.valid and coverage.valid
    payload["valid"] = report.valid
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.valid else 2


def _cmd_report(args: argparse.Namespace) -> int:
    contract = _contract(args)
    report = build_report(read_jsonl(args.input), contract)
    write_report(args.output_dir, report)
    print(f"Wrote report to {args.output_dir}")
    return 0 if report["valid"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "GPU runners are delegated as: `msd`, `attention`, `layers`, `all`. "
            "For example: python -m src.analyze.Validate_Sparrow_hypothesises "
            "attention --help"
        ),
    )
    parser.add_argument("--contract", default="src/analyze/Validate_Sparrow_hypothesises/configs/local_insight_vdc50.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--repo-root", default=".")
    preflight.add_argument("--output", default="results/sparrow_validation/preflight.json")
    preflight.add_argument("--require-gpu", action="store_true")
    preflight.add_argument("--require-models", action="store_true")
    preflight.set_defaults(function=_cmd_preflight)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    prepare.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    prepare.add_argument("--output", default="results/sparrow_validation/planned_manifest.jsonl")
    prepare.add_argument("--limit", type=int)
    prepare.set_defaults(function=_cmd_prepare)

    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--manifest", default="dataset/VideoDetailCaption/subset_manifest.jsonl")
    calibrate.add_argument("--dataset-root", default="dataset/VideoDetailCaption")
    calibrate.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    calibrate.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    calibrate.add_argument("--max-pixels", type=int, default=1024 * 28 * 28)
    calibrate.add_argument("--output", default="results/sparrow_validation/calibration.jsonl")
    calibrate.add_argument("--limit", type=int)
    calibrate.add_argument(
        "--grid-frames",
        help="Comma-separated frame counts replacing the default calibration grid "
        "(e.g. --grid-frames 2,4,8,16,32,64 --grid-pixels 200704,401408 for a fast T4 calibration).",
    )
    calibrate.add_argument(
        "--grid-pixels",
        help="Comma-separated max_pixels values replacing the default calibration grid.",
    )
    calibrate.set_defaults(function=_cmd_calibrate)

    audit = sub.add_parser("audit")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", default="results/sparrow_validation/audit.json")
    audit.set_defaults(function=_cmd_audit)

    report = sub.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output-dir", default="results/sparrow_validation/report")
    report.set_defaults(function=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in {"msd", "attention", "layers", "draft_attention", "all"}:
        command = values.pop(0)
        if command == "msd":
            from .run_msd import build_parser as delegated_parser, run as delegated_run
        elif command == "attention":
            from .run_attention import build_parser as delegated_parser, run as delegated_run
        elif command == "draft_attention":
            from .run_draft_attention import build_parser as delegated_parser, run as delegated_run
        elif command == "layers":
            from .run_layer_analysis import build_parser as delegated_parser, run as delegated_run
        else:
            from .run_paper_experiments import build_parser as delegated_parser, run as delegated_run
        return delegated_run(delegated_parser().parse_args(values))
    args = build_parser().parse_args(values)
    return args.function(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
