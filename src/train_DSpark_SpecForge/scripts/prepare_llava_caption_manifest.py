#!/usr/bin/env python3
"""Validate and normalize the flat LLaVA caption JSONL format.

The source file is intentionally treated as strict JSONL.  A partial final
line is an error: silently training on a smaller set makes the 68k run
irreproducible.  The emitted manifest is still JSONL, but has the structured
Qwen2.5-VL message shape consumed by the multimodal capture script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


def safe_relative_image_path(value: str) -> str:
    """Return a normalized relative image path or raise for unsafe input."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("image must be a non-empty string")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe image path: {value!r}")
    return str(path)


def _caption_content(prompt: str) -> list[dict[str, str]]:
    """Build one image plus text content, preserving prompt semantics."""

    # The source records are single-turn captions.  Qwen receives the image
    # as a structured content item; any legacy <image> marker is removed from
    # the text because the structured item is the canonical image placeholder.
    cleaned = prompt.replace("<image>", "").replace("<Image>", "").strip()
    content: list[dict[str, str]] = [{"type": "image"}]
    if cleaned:
        content.append({"type": "text", "text": cleaned})
    return content


def _iter_source(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"record at {path}:{line_number} must be a JSON object"
                )
            yield line_number, value


def _validate_image_sources(
    relative_paths: list[str], *, image_root: Path | None, image_archive: Path | None
) -> None:
    if image_root is not None:
        missing = [path for path in relative_paths if not (image_root / path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images are missing under {image_root}; "
                f"first missing path: {missing[0]}"
            )
    if image_archive is not None:
        wanted = set(relative_paths)
        with zipfile.ZipFile(image_archive) as archive:
            names = set()
            for name in archive.namelist():
                if not name or name.endswith("/"):
                    continue
                try:
                    names.add(safe_relative_image_path(name))
                except ValueError:
                    # Unreferenced unsafe members are never extracted.
                    continue
            missing = sorted(wanted - names)
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} images are absent from {image_archive}; "
                    f"first missing path: {missing[0]}"
                )


def prepare_manifest(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    expected_records: int = 68_000,
    image_root: str | os.PathLike[str] | None = None,
    image_archive: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if image_root is not None and image_archive is not None:
        raise ValueError("pass at most one of image_root or image_archive")

    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    image_paths: list[str] = []
    with source.open("rb") as raw:
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            digest.update(chunk)
    for line_number, source_record in _iter_source(source):
        for key in ("id", "image", "prompt", "response"):
            if key not in source_record:
                raise ValueError(f"record at line {line_number} is missing {key!r}")
        record_id = str(source_record["id"])
        if not record_id:
            raise ValueError(f"record at line {line_number} has an empty id")
        if record_id in ids:
            raise ValueError(f"duplicate id {record_id!r} at line {line_number}")
        image = safe_relative_image_path(source_record["image"])
        prompt = source_record["prompt"]
        response = source_record["response"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"record {record_id} has an empty prompt")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"record {record_id} has an empty response")
        ids.add(record_id)
        image_paths.append(image)
        records.append(
            {
                "id": record_id,
                "image": image,
                "messages": [
                    {
                        "role": "user",
                        "content": _caption_content(prompt),
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}],
                    },
                ],
                "prompt": prompt,
                "response": response,
                "source_line": line_number,
            }
        )

    if expected_records and len(records) != expected_records:
        raise ValueError(
            f"expected exactly {expected_records} valid records, found {len(records)}"
        )
    _validate_image_sources(
        image_paths,
        image_root=Path(image_root).expanduser().resolve() if image_root else None,
        image_archive=Path(image_archive).expanduser().resolve()
        if image_archive
        else None,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "llava_caption_v1",
        "source": str(source),
        "source_sha256": digest.hexdigest(),
        "records": len(records),
        "unique_images": len(set(image_paths)),
        "image_root": str(Path(image_root).expanduser().resolve())
        if image_root
        else None,
        "image_archive": str(Path(image_archive).expanduser().resolve())
        if image_archive
        else None,
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
        metadata_path = output.with_suffix(output.suffix + ".meta.json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, dest="source_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--image-root")
    parser.add_argument("--image-archive")
    parser.add_argument("--expected-records", type=int, default=68_000)
    args = parser.parse_args(argv)
    metadata = prepare_manifest(
        args.source_path,
        args.output_path,
        expected_records=args.expected_records,
        image_root=args.image_root,
        image_archive=args.image_archive,
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
