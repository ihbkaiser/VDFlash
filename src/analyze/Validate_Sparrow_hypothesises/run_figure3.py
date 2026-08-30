"""Run and compose the current Qwen2.5-VL-3B Figure 3 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .figure3_pipeline import (
    DEFAULT_MODEL,
    DEFAULT_TASKS,
    build_panel_commands,
    compose_figure3_plot,
    validate_figure3_summaries,
    write_figure3_metadata,
)
from src.workspace import resolve_namespace_paths, resolve_workspace_path, workspace_python


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # Accepted for symmetry with the package dispatcher; current Figure 3 has
    # its own MVBench manifest and does not read the VDC paper contract.
    parser.add_argument("--contract", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", default="dataset/MVBench/classified/selected.jsonl")
    parser.add_argument("--output-dir", default="results/figure3_qwen25vl3b")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=360 * 420)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    root = resolve_workspace_path(args.repo_root)
    expected_python = workspace_python(root)
    actual_python = Path(sys.executable)
    if not expected_python.is_file():
        raise SystemExit(f"The shared workspace environment is missing: {expected_python}")
    if not actual_python.samefile(expected_python):
        raise SystemExit(
            "This Figure 3 runner must use the shared workspace interpreter "
            f"{expected_python}; got {actual_python}"
        )
    resolve_namespace_paths(args, "manifest", "output_dir")
    output = (root / args.output_dir).resolve()
    manifest = (root / args.manifest).resolve()
    if not manifest.is_file():
        raise SystemExit(f"Figure 3 manifest does not exist: {manifest}")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"Output directory is not empty: {output}; use --resume explicitly")
    if args.resume and output.exists() and any(output.iterdir()):
        metadata_path = output / "figure3_metadata.json"
        if not metadata_path.is_file():
            raise SystemExit(f"Cannot resume without current Figure 3 metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model") != args.model:
            raise SystemExit("Current Figure 3 resume model mismatch")
        if metadata.get("manifest_sha256") != _sha256_file(manifest):
            raise SystemExit("Current Figure 3 resume manifest mismatch")
    output.mkdir(parents=True, exist_ok=True)
    commands = build_panel_commands(
        python=sys.executable,
        model=args.model,
        manifest=manifest,
        output_dir=output,
        tasks=args.tasks,
        limit_per_task=args.limit_per_task,
        fps=args.fps,
        max_frames=args.max_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
        quantized=args.quantized,
    )
    for panel in ("a", "b"):
        print("$", " ".join(commands[panel]), flush=True)
        result = subprocess.run(commands[panel], cwd=str(root), check=False)
        if result.returncode:
            return result.returncode
    a_summary = json.loads((output / "figure3a.summary.json").read_text(encoding="utf-8"))
    b_summary = json.loads((output / "figure3b.summary.json").read_text(encoding="utf-8"))
    validation = validate_figure3_summaries(a_summary, b_summary, expected_model=args.model)
    a_rows = [json.loads(line) for line in (output / "figure3a.jsonl").read_text().splitlines() if line.strip()]
    b_rows = [json.loads(line) for line in (output / "figure3b.jsonl").read_text().splitlines() if line.strip()]
    compose_figure3_plot(output, a_summary, b_summary)
    write_figure3_metadata(
        output,
        a_summary,
        b_summary,
        a_rows,
        b_rows,
        audit=validation,
        expected_model=args.model,
    )
    if not validation["valid"]:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 2
    print(f"Current Figure 3 bundle: {output}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
