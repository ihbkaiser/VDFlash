from __future__ import annotations

import hashlib
import heapq
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterator
import zipfile

from .config import DFlashTrainConfig


_DATASET_FILES = {
    "text": "ShareGPT_V3_unfiltered_cleaned_split.json",
    "multimodal": "blip_laion_cc_sbu_558k.json",
}
_LLAVA_IMAGE_ARCHIVE = "images.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, value: str) -> None:
    """Durably replace a text file without sharing a temporary filename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as writer:
            temporary_path = Path(writer.name)
            writer.write(value)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_jsonl(path: Path, *, expected_records: int) -> None:
    records = 0
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"generated invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"generated non-object JSON at {path}:{line_number}")
            records += 1
    if records != expected_records:
        raise RuntimeError(
            f"generated manifest contains {records} records; expected {expected_records}"
        )


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - dependency error in CLI
        raise RuntimeError("ijson is required to stream the official dataset JSON files") from exc
    with path.open("rb") as reader:
        for value in ijson.items(reader, "item"):
            if isinstance(value, dict):
                yield value


def _normalized_turns(record: dict[str, Any]) -> tuple[list[dict[str, str]], str] | None:
    raw_turns = record.get("conversations")
    if not isinstance(raw_turns, list):
        return None
    turns: list[dict[str, str]] = []
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
    for raw in raw_turns:
        if not isinstance(raw, dict):
            continue
        role = role_map.get(str(raw.get("from", "")).lower())
        text = raw.get("value")
        if role is None or not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": text})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    if len(turns) < 2 or turns[-1]["role"] != "assistant":
        return None
    target = turns[-1]["content"]
    prompt = turns[:-1]
    if not prompt or prompt[-1]["role"] != "user":
        return None
    return prompt, target


def _valid_record(record: dict[str, Any], stage: str) -> bool:
    if not isinstance(record.get("id"), (str, int)):
        return False
    if _normalized_turns(record) is None:
        return False
    if stage == "multimodal":
        image = record.get("image")
        return isinstance(image, str) and bool(image.strip())
    return True


def _selection_score(seed: int, source_id: str, source_index: int) -> int:
    payload = f"video-dflash-data-v1:{seed}:{source_id}:{source_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def select_source_records(
    annotation_path: Path,
    *,
    stage: str,
    seed: int,
    max_samples: int,
) -> tuple[list[tuple[int, int, str]], int]:
    """Return deterministic hash-shuffled source indices without loading the JSON."""

    heap: list[tuple[int, int, str]] = []
    valid_count = 0
    for source_index, record in enumerate(_iter_json_array(annotation_path)):
        if not _valid_record(record, stage):
            continue
        valid_count += 1
        source_id = str(record["id"])
        score = _selection_score(seed, source_id, source_index)
        if max_samples == 0:
            heap.append((-score, -source_index, source_id))
        elif len(heap) < max_samples:
            heapq.heappush(heap, (-score, -source_index, source_id))
        else:
            worst_score, worst_index = -heap[0][0], -heap[0][1]
            if (score, source_index) < (worst_score, worst_index):
                heapq.heapreplace(heap, (-score, -source_index, source_id))
    selected = [(-neg_score, -neg_index, source_id) for neg_score, neg_index, source_id in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return selected, valid_count


def _safe_image_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe LLaVA image path: {value!r}")
    return str(path)


def _materialize_selected_images(
    relative_paths: list[str],
    *,
    image_root: Path,
    image_archive: str,
    repo_id: str,
    resolved_revision: str,
    allow_remote: bool,
) -> None:
    missing = [value for value in relative_paths if not (image_root / value).is_file()]
    if not missing:
        return
    if not allow_remote and not image_archive:
        raise FileNotFoundError(
            f"{len(missing)} selected LLaVA images are missing under {image_root}; "
            "provide --image-archive/--image-root or enable selective image download"
        )

    if image_archive:
        archive_context = Path(image_archive).expanduser().resolve().open("rb")
    else:
        try:
            import fsspec
            from huggingface_hub import hf_hub_url
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("fsspec and huggingface_hub are required for remote ZIP ranges") from exc
        url = hf_hub_url(
            repo_id=repo_id,
            filename=_LLAVA_IMAGE_ARCHIVE,
            repo_type="dataset",
            revision=resolved_revision,
        )
        archive_context = fsspec.open(
            url,
            mode="rb",
            block_size=1024 * 1024,
            cache_type="readahead",
        ).open()
    image_root.mkdir(parents=True, exist_ok=True)
    try:
        with archive_context as archive_reader, zipfile.ZipFile(archive_reader) as archive:
            for offset, relative in enumerate(missing, start=1):
                try:
                    archive.getinfo(relative)
                except KeyError:
                    raise FileNotFoundError(f"{relative!r} is absent from LLaVA images.zip")
                destination = image_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                with archive.open(relative) as source, temporary.open("wb") as writer:
                    shutil.copyfileobj(source, writer, length=1024 * 1024)
                temporary.replace(destination)
                print(f"[image {offset}/{len(missing)}] {relative}")
    finally:
        close = getattr(archive_context, "close", None)
        if callable(close):
            close()


def _image_user_content(text: str, image_path: Path) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    parts = re.split(r"(<image>)", text, flags=re.IGNORECASE)
    inserted = False
    for part in parts:
        if not part:
            continue
        if part.lower() == "<image>":
            if not inserted:
                content.append({"type": "image", "image": image_path.resolve().as_uri()})
                inserted = True
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})
    if not inserted:
        content.insert(0, {"type": "image", "image": image_path.resolve().as_uri()})
    return content


def _normalize_selected_record(
    record: dict[str, Any],
    *,
    stage: str,
    image_root: Path | None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    normalized = _normalized_turns(record)
    if normalized is None:
        raise ValueError("selected source record no longer satisfies conversation schema")
    prompt, target = normalized
    prompt = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in prompt
    ]
    image_relative: str | None = None
    if stage == "multimodal":
        assert image_root is not None
        image_relative = _safe_image_relative_path(str(record["image"]))
        image_path = image_root / image_relative
        inserted = False
        marker_present = any(
            message["role"] == "user"
            and "<image>" in str(message["content"][0]["text"]).lower()
            for message in prompt
        )
        converted: list[dict[str, Any]] = []
        for message in prompt:
            text = str(message["content"][0]["text"])
            should_insert = message["role"] == "user" and (
                "<image>" in text.lower() or (not marker_present and not inserted)
            )
            if should_insert and not inserted:
                content = _image_user_content(text, image_path)
                inserted = True
                converted.append({"role": "user", "content": content})
            elif message["role"] == "user" and "<image>" in text.lower():
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": re.sub(r"<image>", "", text, flags=re.IGNORECASE).strip()}
                        ],
                    }
                )
            else:
                converted.append(dict(message))
        prompt = converted
    return prompt, target, image_relative


def _resolve_source(config: DFlashTrainConfig) -> tuple[Path, str, str | None]:
    filename = _DATASET_FILES[config.stage]
    if config.data_path:
        supplied = Path(config.data_path).expanduser().resolve()
        annotation = supplied / filename if supplied.is_dir() else supplied
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        return annotation, config.dataset_revision or "local", None
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to download dataset annotations") from exc
    info = HfApi().dataset_info(
        config.dataset_repo,
        revision=config.dataset_revision,
        files_metadata=True,
    )
    resolved = str(info.sha)
    annotation = Path(
        hf_hub_download(
            repo_id=config.dataset_repo,
            filename=filename,
            repo_type="dataset",
            revision=resolved,
        )
    )
    lfs_sha = None
    for sibling in info.siblings:
        if sibling.rfilename == filename and sibling.lfs is not None:
            lfs_sha = sibling.lfs.sha256
            break
    return annotation, resolved, lfs_sha


def prepare_real_manifest(config: DFlashTrainConfig) -> tuple[Path, Path]:
    annotation_path, resolved_revision, hub_file_sha = _resolve_source(config)
    annotation_sha = sha256_file(annotation_path)
    if hub_file_sha is not None and annotation_sha != hub_file_sha:
        raise RuntimeError(
            f"downloaded annotation SHA-256 mismatch: {annotation_sha} != {hub_file_sha}"
        )
    print(
        f"[source] stage={config.stage} repo={config.dataset_repo} "
        f"revision={resolved_revision} file={annotation_path.name} sha256={annotation_sha}"
    )
    selected, valid_count = select_source_records(
        annotation_path,
        stage=config.stage,
        seed=config.seed,
        max_samples=config.max_samples,
    )
    selected_by_index = {source_index: (rank, score, source_id) for rank, (score, source_index, source_id) in enumerate(selected)}
    raw_selected: list[dict[str, Any] | None] = [None] * len(selected)
    for source_index, record in enumerate(_iter_json_array(annotation_path)):
        selection = selected_by_index.get(source_index)
        if selection is not None:
            rank, _, _ = selection
            raw_selected[rank] = record
    if any(record is None for record in raw_selected):
        raise RuntimeError("failed to recover every deterministically selected source record")

    manifest_path = Path(config.prepared_manifest).expanduser().resolve()
    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    if (manifest_path.exists() or metadata_path.exists()) and not config.overwrite:
        raise FileExistsError(
            f"prepared manifest already exists at {manifest_path}; pass --overwrite to replace it"
        )
    image_root: Path | None = None
    image_relatives: list[str] = []
    if config.stage == "multimodal":
        image_root = (
            Path(config.image_root).expanduser().resolve()
            if config.image_root
            else manifest_path.parent / "llava_images"
        )
        image_relatives = [
            _safe_image_relative_path(str(record["image"]))
            for record in raw_selected
            if record is not None
        ]
        _materialize_selected_images(
            image_relatives,
            image_root=image_root,
            image_archive=config.image_archive,
            repo_id=config.dataset_repo,
            resolved_revision=resolved_revision,
            allow_remote=config.selective_image_download,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected_ids: list[dict[str, Any]] = []
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as writer:
            temporary_path = Path(writer.name)
            for rank, ((score, source_index, source_id), raw) in enumerate(zip(selected, raw_selected)):
                assert raw is not None
                messages, target_text, image_relative = _normalize_selected_record(
                    raw,
                    stage=config.stage,
                    image_root=image_root,
                )
                manifest_id = f"{config.stage}:{source_id}:{source_index}"
                item = {
                    "id": manifest_id,
                    "messages": messages,
                    "target_text": target_text,
                    "source": {
                        "dataset_repo": config.dataset_repo,
                        "dataset_revision": resolved_revision,
                        "annotation_file": annotation_path.name,
                        "annotation_sha256": annotation_sha,
                        "split": config.split,
                        "source_id": source_id,
                        "source_index": source_index,
                        "shuffle_rank": rank,
                        "shuffle_score": f"{score:032x}",
                        "image": image_relative,
                    },
                }
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                selected_ids.append(
                    {"manifest_id": manifest_id, "source_id": source_id, "source_index": source_index}
                )
            writer.flush()
            os.fsync(writer.fileno())
        _validate_jsonl(temporary_path, expected_records=len(selected_ids))
        os.replace(temporary_path, manifest_path)
        temporary_path = None
        _fsync_directory(manifest_path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    metadata = {
        "format": "video-dflash-real-manifest-v1",
        "stage": config.stage,
        "dataset_repo": config.dataset_repo,
        "requested_revision": config.dataset_revision,
        "resolved_revision": resolved_revision,
        "annotation_file": annotation_path.name,
        "annotation_sha256": annotation_sha,
        "split": config.split,
        "seed": config.seed,
        "valid_source_records": valid_count,
        "max_samples": config.max_samples,
        "selected_count": len(selected_ids),
        "selected_sample_ids": selected_ids,
        "image_root": str(image_root) if image_root is not None else None,
        "selective_image_download": config.selective_image_download,
    }
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
    )
    print(
        f"[manifest] records={len(selected_ids)} valid_source={valid_count} "
        f"path={manifest_path} metadata={metadata_path}"
    )
    return manifest_path, metadata_path
