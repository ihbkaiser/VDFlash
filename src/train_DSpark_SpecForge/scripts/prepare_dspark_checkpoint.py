#!/usr/bin/env python3
"""Select and extract one resumable DSpark checkpoint from the MSD tar.zst."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


_CHECKPOINT_MEMBER = re.compile(
    r"^(outputs/.+?)/(phase[12])/([^/]+)/([^/]+-step([0-9]+))/"
    r"(training_state(?:_rank([0-9]+))?\.pt)$"
)


def _parse_step(value: str) -> str:
    if value == "latest":
        return value
    if re.fullmatch(r"[0-9]+", value):
        return value
    raise argparse.ArgumentTypeError("step must be 'latest' or a non-negative integer")


def _archive_entries(archive: Path):
    process = subprocess.Popen(
        ["tar", "--use-compress-program=zstd", "--list", "--verbose", "--file", str(archive)],
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    candidates: dict[str, dict] = {}
    entry_types: dict[str, str] = {}
    for line in process.stdout:
        fields = line.rstrip("\n").split(None, 5)
        if len(fields) < 6:
            continue
        mode = fields[0]
        try:
            size = int(fields[2])
        except ValueError:
            continue
        member = fields[5]
        normalized = member.rstrip("/")
        parsed = PurePosixPath(normalized)
        if parsed.is_absolute() or ".." in parsed.parts:
            continue
        entry_types[normalized] = mode[:1]

        match = _CHECKPOINT_MEMBER.match(normalized)
        if not match or mode[:1] != "-":
            continue
        prefix, phase, run_id, checkpoint_name, step, filename, rank = match.groups()
        checkpoint_dir = "/".join(
            (prefix, phase, run_id, checkpoint_name)
        )
        candidate = candidates.setdefault(
            checkpoint_dir,
            {
                "phase": phase,
                "run_id": run_id,
                "step": int(step),
                "files": {},
                "prefix": prefix,
            },
        )
        if candidate["step"] != int(step) or candidate["run_id"] != run_id:
            raise RuntimeError(f"inconsistent checkpoint members under {checkpoint_dir}")
        if filename in candidate["files"]:
            raise RuntimeError(f"duplicate archive member: {normalized}")
        candidate["files"][filename] = {"path": normalized, "size": size}

    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"tar listing failed with exit code {return_code}")
    return candidates, entry_types


def _select_checkpoint(
    candidates: dict[str, dict], phase: str, run_id: str | None, step: str
) -> tuple[str, dict]:
    selected_phase = phase
    available_phases = {
        candidate["phase"]
        for candidate in candidates.values()
        if "training_state.pt" in candidate["files"]
    }
    if selected_phase == "auto":
        selected_phase = "phase2" if "phase2" in available_phases else "phase1"
    choices = [
        (path, item)
        for path, item in candidates.items()
        if item["phase"] == selected_phase
        and (run_id is None or item["run_id"] == run_id)
        and "training_state.pt" in item["files"]
    ]
    if not choices:
        raise FileNotFoundError(
            f"no complete training_state.pt checkpoint for phase={selected_phase}, "
            f"run_id={run_id!r}; available phases={sorted(available_phases)}"
        )

    runs = sorted({item["run_id"] for _, item in choices})
    if len(runs) > 1 and run_id is None:
        raise ValueError(
            f"multiple runs found for {selected_phase}: {runs}; pass --run-id"
        )
    if step == "latest":
        path, candidate = max(choices, key=lambda pair: pair[1]["step"])
    else:
        requested_step = int(step)
        matches = [pair for pair in choices if pair[1]["step"] == requested_step]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one checkpoint at step {requested_step}, found {len(matches)}"
            )
        path, candidate = matches[0]
    return path, candidate


def _extract(archive: Path, output_root: Path, checkpoint_path: str, candidate: dict) -> Path:
    files = candidate["files"]
    members = [files["training_state.pt"]["path"]]
    members.extend(
        files[name]["path"]
        for name in sorted(files)
        if re.fullmatch(r"training_state_rank[0-9]+\.pt", name)
    )

    for member in members:
        parts = PurePosixPath(member).parts
        for index in range(1, len(parts)):
            ancestor = "/".join(parts[:index])
            # Archive path components must be real directories, never links.
            # Missing explicit directory entries are safe: tar creates them.
            kind = candidate.get("entry_types", {}).get(ancestor)
            if kind is not None and kind != "d":
                raise ValueError(f"unsafe non-directory archive parent: {ancestor}")

    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        "tar",
        "--use-compress-program=zstd",
        "--extract",
        "--file",
        str(archive),
        "--directory",
        str(output_root),
        "--no-same-owner",
        "--no-same-permissions",
        "--occurrence=1",
        "--",
        *members,
    ]
    subprocess.run(command, check=True)

    local_checkpoint = output_root / Path(*PurePosixPath(checkpoint_path).parts)
    for member in members:
        local_path = output_root / Path(*PurePosixPath(member).parts)
        expected_size = next(
            info["size"] for info in files.values() if info["path"] == member
        )
        if not local_path.is_file() or local_path.stat().st_size != expected_size:
            raise IOError(
                f"extracted file size mismatch for {local_path}: "
                f"expected {expected_size} bytes"
            )
    return local_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=("auto", "phase1", "phase2"),
        default="auto",
        help="auto prefers phase2 and falls back to phase1",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--step",
        type=_parse_step,
        default="latest",
        help="latest or an integer step",
    )
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args(argv)

    archive = args.archive.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not archive.is_file():
        parser.error(f"archive does not exist: {archive}")

    print(f"Scanning archive entries in {archive}", file=sys.stderr, flush=True)
    candidates, entry_types = _archive_entries(archive)
    if not candidates:
        raise FileNotFoundError("no DSpark training_state.pt checkpoints found")
    for path, item in sorted(
        candidates.items(), key=lambda pair: (pair[1]["phase"], pair[1]["step"])
    ):
        print(
            f"{item['phase']} {item['run_id']} step={item['step']} "
            f"files={','.join(sorted(item['files']))}"
        )

    checkpoint_path, candidate = _select_checkpoint(
        candidates, args.phase, args.run_id, args.step
    )
    if args.list_only:
        print(f"SELECTED {checkpoint_path}")
        return 0

    candidate["entry_types"] = entry_types
    local_path = _extract(archive, output_root, checkpoint_path, candidate)
    selection_path = output_root / "selected_checkpoint.json"
    temporary_path = selection_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "checkpoint_dir": checkpoint_path,
                "phase": candidate["phase"],
                "run_id": candidate["run_id"],
                "step": candidate["step"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(selection_path)
    print(f"EXTRACTED {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
