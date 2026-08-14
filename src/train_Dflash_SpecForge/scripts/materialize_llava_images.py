#!/usr/bin/env python3
"""Safely materialize only the images referenced by a normalized manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def _safe(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe manifest image path: {value!r}")
    return str(path)


def materialize(manifest: str, archive: str, output_root: str) -> int:
    rows = [
        json.loads(line)
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = sorted({_safe(str(row["image"])) for row in rows})
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    copied = 0
    with zipfile.ZipFile(Path(archive).expanduser().resolve()) as source:
        archive_names = {}
        for name in source.namelist():
            if name and not name.endswith("/"):
                try:
                    archive_names[_safe(name)] = name
                except ValueError:
                    # Never extract unreferenced unsafe archive members.
                    continue
        missing = [
            path
            for path in wanted
            if path not in archive_names and not (root / path).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{missing[0]} is absent from {archive}")
        for relative in wanted:
            destination = (root / relative).resolve()
            if destination.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=str(destination.parent)
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                with source.open(archive_names[relative]) as reader, temporary.open(
                    "wb"
                ) as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(materialize(args.manifest, args.archive, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
