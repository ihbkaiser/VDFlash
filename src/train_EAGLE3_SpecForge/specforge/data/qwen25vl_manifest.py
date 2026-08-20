"""Standalone Qwen2.5-VL caption manifest and image-asset utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import shutil
import stat
import tarfile
import tempfile
from typing import Any, Mapping
import zipfile


MANIFEST_SCHEMA = "eagle3_qwen25vl_caption_manifest_v1"
REQUIRED_FIELDS = ("id", "image", "prompt", "response")


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def safe_relative_image_path(
    image_root: str | os.PathLike[str],
    relative_image: str,
) -> Path:
    """Resolve one image below ``image_root`` without traversal or symlinks."""

    root = Path(image_root).expanduser().resolve()
    relative = _non_empty_string(relative_image, name="image")
    candidate_path = Path(relative.replace("\\", "/"))
    windows_path = PureWindowsPath(relative)
    if (
        candidate_path.is_absolute()
        or candidate_path.anchor
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(f"image path {relative!r} is outside image root")
    if any(part == ".." for part in candidate_path.parts):
        raise ValueError(f"image path {relative!r} is outside image root")

    candidate = (root / candidate_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"image path {relative!r} is outside image root") from exc
    return candidate


def normalize_caption_record(
    raw: Mapping[str, Any],
    *,
    source_line: int,
) -> dict[str, Any]:
    """Validate and convert one flat caption row to Qwen chat messages."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"source line {source_line} must be a JSON object")
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(
            f"source line {source_line} is missing required fields: {missing}"
        )
    record_id = _non_empty_string(raw["id"], name=f"id at source line {source_line}")
    image = _non_empty_string(
        raw["image"], name=f"image at source line {source_line}"
    )
    prompt = _non_empty_string(
        raw["prompt"], name=f"prompt at source line {source_line}"
    )
    response = _non_empty_string(
        raw["response"], name=f"response at source line {source_line}"
    )
    prompt = prompt.replace("<image>", "").strip()
    if not prompt:
        raise ValueError(f"prompt at source line {source_line} is empty after cleanup")
    return {
        "id": record_id,
        "image": image,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ],
        "prompt": prompt,
        "response": response,
        "source_line": source_line,
    }


def _atomic_jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def normalize_caption_jsonl(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    expected_records: int | None = None,
    image_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Normalize a caption JSONL file, refusing partial output on failure."""

    if expected_records is not None and expected_records < 0:
        raise ValueError("expected_records must be non-negative")
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"empty source line {source_line}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at source line {source_line}") from exc
            record = normalize_caption_record(raw, source_line=source_line)
            if record["id"] in seen_ids:
                raise ValueError(f"duplicate id {record['id']!r}")
            seen_ids.add(record["id"])
            if image_root is not None:
                image_path = safe_relative_image_path(image_root, record["image"])
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"image for record {record['id']!r} does not exist: {image_path}"
                    )
            records.append(record)

    record_count = len(records)
    if expected_records is not None and record_count != expected_records:
        raise ValueError(
            f"expected {expected_records} records, found {record_count}"
        )
    _atomic_jsonl_write(destination, records)
    metadata = {
        "schema": MANIFEST_SCHEMA,
        "input_path": str(source.resolve()),
        "output_path": str(destination.resolve()),
        "record_count": record_count,
        "expected_records": expected_records,
        "image_root": str(Path(image_root).expanduser().resolve())
        if image_root is not None
        else None,
    }
    _atomic_json_write(Path(f"{destination}.meta.json"), metadata)
    return metadata


def _safe_archive_target(root: Path, member_name: str) -> Path:
    return safe_relative_image_path(root, member_name)


def _extract_zip(archive: zipfile.ZipFile, root: Path) -> None:
    for info in archive.infolist():
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"archive member {info.filename!r} is a link")
        target = _safe_archive_target(root, info.filename)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _extract_tar(archive: tarfile.TarFile, root: Path) -> None:
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"archive member {member.name!r} is a link")
        target = _safe_archive_target(root, member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"unsupported archive member {member.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"could not read archive member {member.name!r}")
        with source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def materialize_image_archive(
    archive_path: str | os.PathLike[str],
    image_root: str | os.PathLike[str],
) -> Path:
    """Extract a zip/tar image archive into a controlled root."""

    archive_path = Path(archive_path).expanduser()
    root = Path(image_root).expanduser().resolve()
    root_preexisted = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    try:
        if any(root.iterdir()):
            raise FileExistsError(f"image root is not empty: {root}")
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                _extract_zip(archive, root)
        else:
            try:
                archive = tarfile.open(archive_path, mode="r:*")
            except tarfile.TarError as exc:
                raise ValueError(f"unsupported image archive: {archive_path}") from exc
            with archive:
                _extract_tar(archive, root)
    except BaseException:
        if not root_preexisted:
            shutil.rmtree(root, ignore_errors=True)
        raise
    return root


__all__ = [
    "MANIFEST_SCHEMA",
    "materialize_image_archive",
    "normalize_caption_jsonl",
    "normalize_caption_record",
    "safe_relative_image_path",
]
