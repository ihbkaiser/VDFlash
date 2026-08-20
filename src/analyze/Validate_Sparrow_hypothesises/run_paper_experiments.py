"""Orchestrate the complete Sparrow paper-conformance experiment matrix.

Each GPU-heavy stage runs in a separate Python process so the 7B checkpoints
are released before the next analysis stage.  The final merged JSONL is then
audited and rendered into the same report format used by the individual CLI
commands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .audit import audit_figure2_homogeneous
from .evidence import build_final_evidence, collect_evidence, write_evidence
from .dataset import write_jsonl
from .paper_contract import load_contract


PACKAGE = "src.analyze.Validate_Sparrow_hypothesises"


def _run(command: list[str], cwd: Path, *, allow_failure: bool = False) -> int:
    print("$", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode and not allow_failure:
        raise SystemExit(result.returncode)
    return result.returncode


def _merge(paths: list[Path], output: Path, diagnostic: Path):
    """Strictly merge stage files and separate evidence from diagnostics."""

    result = collect_evidence(paths)
    if not result.evidence_rows:
        raise SystemExit("No evidence rows were produced; refusing to build a report")
    write_evidence(result, output, diagnostic)
    if result.malformed_files:
        print("Excluded malformed stage files: " + ", ".join(result.malformed_files), flush=True)
    print(
        f"Merged {len(result.evidence_rows)} evidence rows and "
        f"{len(result.diagnostic_rows)} diagnostic rows into {output}",
        flush=True,
    )
    return result


def _finalize_evidence(
    full_result,
    output: Path,
    diagnostic: Path,
    selection_path: Path,
    figure2_audit_path: Path,
    contract,
    *,
    enabled: bool = True,
):
    """Create final report evidence from the strict full stage merge."""

    if not enabled:
        write_evidence(full_result, output, diagnostic)
        return None

    selected = build_final_evidence(
        full_result.evidence_rows,
        figure2_targets=(contract.attention_short_tokens, contract.attention_long_tokens),
        minimum_samples=contract.minimum_paired_samples,
    )
    write_evidence(full_result, output, diagnostic)
    figure2_rows_path = output.parent / "figure2_homogeneous_results.jsonl"
    figure2_diagnostic_path = output.parent / "figure2_diagnostic_rows.jsonl"
    write_jsonl(figure2_rows_path, selected.figure2_rows)
    write_jsonl(figure2_diagnostic_path, selected.diagnostic_rows)
    cohort = dict(selected.cohort)
    cohort.update({
        "source_results": str(output),
        "selected_results": str(figure2_rows_path),
        "selection_reason": "Exact actual visual-token signature per target; removes absolute-position boundary mixing.",
    })
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure2_audit = audit_figure2_homogeneous(
        selected.figure2_rows,
        contract,
        (contract.attention_short_tokens, contract.attention_long_tokens),
        minimum_samples=contract.minimum_paired_samples,
    )
    figure2_audit["cohort"] = cohort
    figure2_audit_path.write_text(
        json.dumps(figure2_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selected


def run(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    output_dir = root / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. "
            "Choose a fresh --output-dir or pass --resume explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    contract_path = root / args.contract
    contract = load_contract(contract_path)
    figure2_enabled = args.figure2_homogeneous
    if args.limit is not None and args.limit < contract.minimum_paired_samples:
        if args.auto_cohort:
            print(
                f"Skipping automatic paired cohort for smoke limit={args.limit}; "
                f"at least {contract.minimum_paired_samples} samples are required.",
                flush=True,
            )
        if args.figure2_homogeneous:
            print(
                "Skipping homogeneous Figure 2 report source for an undersized smoke run.",
                flush=True,
            )
        figure2_enabled = False
    preflight = output_dir / "preflight.json"
    calibration = root / args.calibration_input if args.calibration_input else output_dir / "calibration.jsonl"
    msd_full = output_dir / "msd_full.jsonl"
    msd_remove_all = output_dir / "msd_remove_all.jsonl"
    msd_retention_last = output_dir / "msd_retention_last_instruction.jsonl"
    msd_retention_all = output_dir / "msd_retention_all_text.jsonl"
    attention = output_dir / "figure2_attention.jsonl"
    draft_attention = output_dir / "figure2_draft_attention.jsonl"
    layers = output_dir / "layer_analysis.jsonl"
    full_combined = output_dir / "full_results.jsonl"
    full_diagnostics = output_dir / "full_diagnostic_rows.jsonl"
    combined = output_dir / "results.jsonl"
    diagnostics = output_dir / "diagnostic_rows.jsonl"
    figure2_selection = output_dir / "figure2_homogeneous_cohort.json"
    figure2_audit = output_dir / "audit_figure2_homogeneous.json"
    full_audit = output_dir / "audit_full.json"
    audit = output_dir / "audit.json"
    report = root / args.report_output_dir if args.report_output_dir else output_dir / "report"

    base = [python, "-m", PACKAGE]
    global_flags = ["--contract", str(contract_path)]
    if not args.skip_preflight:
        _run(base + global_flags + ["preflight", "--require-gpu", "--require-models", "--output", str(preflight)], root)
    if not args.skip_calibration:
        calibration_command = base + global_flags + [
            "calibrate",
            "--output", str(calibration),
            "--model", args.calibration_model,
        ]
        if args.reuse_calibration:
            calibration_command += ["--reuse-calibration", str(root / args.reuse_calibration)]
        if args.limit is not None:
            calibration_command += ["--limit", str(args.limit)]
        _run(calibration_command, root)

    if args.skip_calibration:
        if not calibration.is_file() or calibration.stat().st_size == 0:
            raise SystemExit(
                f"--skip-calibration was requested, but calibration is missing or empty: {calibration}"
            )
        # Skipping recalibration must still feed the existing measurements to
        # every GPU stage; otherwise attention/MSD silently fall back to the
        # unbounded native video path.
        calibration_arg = ["--calibration", str(calibration)]
    else:
        calibration_arg = ["--calibration", str(calibration)]
    if args.allow_out_of_tolerance:
        calibration_arg.append("--allow-out-of-tolerance")
    if args.cohort_manifest:
        stage_manifest = root / args.cohort_manifest
    elif args.auto_cohort and (args.limit is None or args.limit >= contract.minimum_paired_samples):
        auto_manifest = output_dir / "cohort_manifest.jsonl"
        auto_selection = output_dir / "cohort_selection.json"
        if not (args.resume and auto_manifest.is_file() and auto_manifest.stat().st_size):
            _run(base + global_flags + [
                "cohort",
                "--input", str(calibration),
                "--output-manifest", str(auto_manifest),
                "--output-selection", str(auto_selection),
                "--targets", *[str(value) for value in contract.visual_token_milestones],
            ], root)
        stage_manifest = auto_manifest
    else:
        stage_manifest = root / "dataset/VideoDetailCaption/subset_manifest.jsonl"
    common = ["--manifest", str(stage_manifest)]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]
    model_flags = ["--device-map", args.device_map, "--dtype", args.dtype]
    if args.quantized:
        model_flags.append("--quantized")
    msd_flags: list[str] = []
    if args.msd_device_map:
        msd_flags.extend(["--device-map", args.msd_device_map])
    if args.msd_max_memory:
        msd_flags.extend(["--max-memory", args.msd_max_memory])
    produced: list[Path] = []
    if not args.skip_attention:
        _run(base + global_flags + [
            "attention",
            "--output", str(attention),
            *calibration_arg,
            "--visual-targets", str(contract.attention_short_tokens), str(contract.attention_long_tokens),
            *model_flags,
            *common,
        ], root)
        produced.append(attention)
    elif attention.exists():
        # Resume runs may intentionally skip the expensive target probe while
        # still needing its completed output in the final merged report.
        produced.append(attention)
    if not args.skip_draft_attention:
        _run(base + global_flags + [
            "draft_attention",
            "--output", str(draft_attention),
            *calibration_arg,
            "--visual-targets", str(contract.attention_short_tokens), str(contract.attention_long_tokens), str(contract.retention_anchor_visual_tokens),
            *common, *msd_flags,
        ], root)
        produced.append(draft_attention)
    elif draft_attention.exists():
        produced.append(draft_attention)
    if not args.skip_msd:
        # Run attention first: the retention selectors consume the draft
        # attention trace.  Keep the full length sweep and remove-all series
        # separate so Figure 1(a) cannot accidentally mix with Figure 1(b).
        _run(base + global_flags + [
            "msd", "--condition", "full", "--output", str(msd_full),
            "--visual-targets", *[str(value) for value in contract.visual_token_milestones],
            "--strict-losslessness", *calibration_arg, *common, *msd_flags,
        ], root)
        produced.append(msd_full)
        _run(base + global_flags + [
            "msd", "--condition", "retention", "--length-series", "remove_all",
            "--selection", "uniform", "--retention-percentages", "0",
            "--output", str(msd_remove_all), "--visual-targets",
            *[str(value) for value in contract.visual_token_milestones],
            "--strict-losslessness", *calibration_arg, *common, *msd_flags,
        ], root)
        produced.append(msd_remove_all)
        if args.skip_draft_attention:
            print("WARNING: draft attention was skipped; selector retention is omitted and coverage will remain incomplete.", flush=True)
        else:
            for selection, destination in (("last_instruction", msd_retention_last), ("all_text", msd_retention_all)):
                _run(base + global_flags + [
                    "msd", "--condition", "retention", "--selection", selection,
                    "--selection-scores", str(draft_attention),
                    "--retention-percentages", *[str(value) for value in contract.retention_percentages],
                    "--visual-targets", str(contract.retention_anchor_visual_tokens),
                    "--output", str(destination), "--strict-losslessness", *calibration_arg, *common, *msd_flags,
                ], root)
                produced.append(destination)
    else:
        for existing in (msd_full, msd_remove_all, msd_retention_last, msd_retention_all):
            if existing.exists():
                produced.append(existing)
    if not args.skip_layers:
        _run(base + global_flags + [
            "layers",
            "--output", str(layers),
            "--experiments", args.layer_experiments,
            *calibration_arg,
            "--visual-targets", *[str(value) for value in args.layer_visual_targets],
            *model_flags,
            *common,
        ], root)
        produced.append(layers)
    elif layers.exists():
        produced.append(layers)
    full_result = _merge(produced, full_combined, full_diagnostics)
    _finalize_evidence(
        full_result,
        combined,
        diagnostics,
        figure2_selection,
        figure2_audit,
        contract,
        enabled=figure2_enabled,
    )
    # Keep a separate audit of the complete stage evidence.  The final audit
    # below runs on the public full-evidence `results.jsonl`; the report-only
    # Figure 2 source is audited separately above.
    _run(base + global_flags + [
        "audit", "--input", str(full_combined), "--output", str(full_audit),
        "--calibration", str(calibration),
        "--calibration-targets", *[str(value) for value in contract.visual_token_milestones],
    ], root, allow_failure=True)
    audit_rc = _run(base + global_flags + [
        "audit", "--input", str(combined), "--output", str(audit),
        "--calibration", str(calibration),
        "--calibration-targets", *[str(value) for value in contract.visual_token_milestones],
    ], root, allow_failure=True)
    report_command = base + global_flags + [
        "report", "--input", str(combined), "--output-dir", str(report),
    ]
    if figure2_enabled:
        report_command.extend([
            "--figure2-input", str(output_dir / "figure2_homogeneous_results.jsonl"),
            "--figure2-selection", str(figure2_selection),
            "--figure2-audit", str(figure2_audit),
        ])
    report_rc = _run(report_command, root, allow_failure=True)
    print(f"Complete report: {report / 'REPORT.md'}")
    return report_rc or audit_rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="src/analyze/Validate_Sparrow_hypothesises/configs/local_insight_vdc50.yaml")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="results/sparrow_validation")
    parser.add_argument(
        "--report-output-dir",
        help="Optional separate canonical report directory; defaults to <output-dir>/report.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow an existing output directory to be reused; merge remains strict.",
    )
    parser.add_argument(
        "--calibration-input",
        help="Use an existing measured calibration JSONL, e.g. a supplementary 400-token run.",
    )
    parser.add_argument(
        "--reuse-calibration",
        help="Copy strict non-requested targets from this file when creating calibration.",
    )
    parser.add_argument(
        "--cohort-manifest",
        help="Use the manifest emitted by the paired-cohort selector for every stage.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calibration-model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--msd-condition", choices=("full", "retention", "both"), default="both", help="Legacy compatibility; the local profile runs the required series explicitly.")
    parser.add_argument("--layer-experiments", choices=("figure3", "figure6", "both"), default="both")
    parser.add_argument("--layer-visual-targets", type=int, nargs="+", default=[3000])
    parser.add_argument(
        "--auto-cohort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select a status=ok paired cohort after calibration when no manifest is supplied.",
    )
    parser.add_argument(
        "--figure2-homogeneous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the exact homogeneous summary cohort as the Figure 2 report/statistics source.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--msd-device-map",
        choices=("cuda", "auto", "model_parallel"),
        default=None,
        help="Optional device map for MSD stages; use model_parallel on the 3090+A4000 pair.",
    )
    parser.add_argument(
        "--msd-max-memory",
        help="Per-device budgets passed to MSD, for example 0:22GiB,1:14GiB.",
    )
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
