"""Orchestrate the complete Sparrow paper-conformance experiment matrix.

Each GPU-heavy stage runs in a separate Python process so the 7B checkpoints
are released before the next analysis stage.  The final merged JSONL is then
audited and rendered into the same report format used by the individual CLI
commands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .dataset import write_jsonl


PACKAGE = "src.analyze.Validate_Sparrow_hypothesises"


def _run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def _merge(paths: list[Path], output: Path) -> None:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            import json

            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit("No measured rows were produced; refusing to build a report")
    write_jsonl(output, rows)
    print(f"Merged {len(rows)} rows into {output}")


def run(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    preflight = output_dir / "preflight.json"
    calibration = output_dir / "calibration.jsonl"
    msd = output_dir / "msd.jsonl"
    attention = output_dir / "figure2_attention.jsonl"
    draft_attention = output_dir / "figure2_draft_attention.jsonl"
    layers = output_dir / "layer_analysis.jsonl"
    combined = output_dir / "results.jsonl"
    audit = output_dir / "audit.json"
    report = output_dir / "report"

    base = [python, "-m", PACKAGE]
    if not args.skip_preflight:
        _run(base + ["preflight", "--require-gpu", "--require-models", "--output", str(preflight)], root)
    if not args.skip_calibration:
        _run(base + [
            "calibrate",
            "--output", str(calibration),
            "--model", args.calibration_model,
        ] + (["--limit", str(args.limit)] if args.limit is not None else []), root)

    calibration_arg = [] if args.skip_calibration else ["--calibration", str(calibration)]
    if args.allow_out_of_tolerance:
        calibration_arg.append("--allow-out-of-tolerance")
    common = ["--limit", str(args.limit)] if args.limit is not None else []
    model_flags = ["--device-map", args.device_map, "--dtype", args.dtype]
    if args.quantized:
        model_flags.append("--quantized")
    produced: list[Path] = []
    if not args.skip_msd:
        _run(base + [
            "msd",
            "--condition", args.msd_condition,
            "--output", str(msd),
            *calibration_arg,
            *common,
        ], root)
        produced.append(msd)
    if not args.skip_attention:
        _run(base + [
            "attention",
            "--output", str(attention),
            *calibration_arg,
            "--visual-targets", "400", "3000",
            *model_flags,
            *common,
        ], root)
        produced.append(attention)
    if not args.skip_draft_attention:
        _run(base + [
            "draft_attention",
            "--output", str(draft_attention),
            *calibration_arg,
            "--visual-targets", "400", "3000",
            *common,
        ], root)
        produced.append(draft_attention)
    if not args.skip_layers:
        _run(base + [
            "layers",
            "--output", str(layers),
            "--experiments", args.layer_experiments,
            *calibration_arg,
            "--visual-targets", *[str(value) for value in args.layer_visual_targets],
            *model_flags,
            *common,
        ], root)
        produced.append(layers)
    _merge(produced, combined)
    _run(base + ["audit", "--input", str(combined), "--output", str(audit)], root)
    _run(base + ["report", "--input", str(combined), "--output-dir", str(report)], root)
    print(f"Complete report: {report / 'REPORT.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="results/sparrow_validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calibration-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-condition", choices=("full", "retention", "both"), default="both")
    parser.add_argument("--layer-experiments", choices=("figure3", "figure6", "both"), default="both")
    parser.add_argument("--layer-visual-targets", type=int, nargs="+", default=[3000])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument(
        "--allow-out-of-tolerance",
        action="store_true",
        help="Run calibration points whose measured visual-token count is outside "
        "the 10 percent tolerance (needed when a video is too short to reach the 25k milestone).",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-msd", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-draft-attention", action="store_true")
    parser.add_argument("--skip-layers", action="store_true")
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(build_parser().parse_args()))

