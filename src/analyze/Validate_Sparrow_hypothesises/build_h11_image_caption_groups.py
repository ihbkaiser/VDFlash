"""Build H1.1 Reference and genuinely held-out LLaVA image-caption manifests.

Reference is sampled from the exact 68K training manifest. Held-out evaluation
is sampled from the full LLaVA-Pretrain source after excluding every training
record ID and image path. Both outputs use the manifest shape accepted by the
SpecForge LLaVA hidden-state capture script.

The source may be the official JSON array (with ``conversations``) or flat
JSONL (with ``id``, ``image``, ``prompt``, and ``response``).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Iterator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_image(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("record has an empty image path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe image path: {value!r}")
    return str(path)


def _iter_source(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream official JSON arrays through ijson or read flat JSONL linewise."""
    with path.open("rb") as handle:
        first = b""
        while not first:
            byte = handle.read(1)
            if not byte:
                raise ValueError(f"empty source annotation: {path}")
            if not byte.isspace():
                first = byte
        handle.seek(0)

        if first == b"[":
            try:
                import ijson
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("install ijson to stream the 558K JSON array") from exc
            for index, record in enumerate(ijson.items(handle, "item")):
                if not isinstance(record, dict):
                    raise ValueError(f"source record {index} is not a JSON object")
                yield index, record
            return

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"source record at line {line_number} is not an object")
            yield line_number - 1, record


def _prompt_response(record: dict[str, Any]) -> tuple[str, str] | None:
    """Normalize flat caption JSONL and original LLaVA conversation records."""
    prompt, response = record.get("prompt"), record.get("response")
    if isinstance(prompt, str) and prompt.strip() and isinstance(response, str) and response.strip():
        return prompt.strip(), response.strip()

    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None
    turns: list[tuple[str, str]] = []
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = role_map.get(str(turn.get("from", "")).lower())
        text = turn.get("value")
        if role and isinstance(text, str) and text.strip():
            text = text.strip()
            if turns and turns[-1][0] == role:
                turns[-1] = (role, turns[-1][1] + "\n\n" + text)
            else:
                turns.append((role, text))
    while turns and turns[0][0] != "user":
        turns.pop(0)
    if len(turns) < 2 or turns[-1][0] != "assistant":
        return None
    prompt_turns = turns[:-1]
    if not prompt_turns or prompt_turns[-1][0] != "user":
        return None
    return "\n\n".join(text for _, text in prompt_turns), turns[-1][1]


def _caption_manifest_record(
    record_id: str,
    image: str,
    prompt: str,
    response: str,
    *,
    source_index: int,
) -> dict[str, Any]:
    clean_prompt = re.sub(r"<image>", "", prompt, flags=re.IGNORECASE).strip()
    user_content: list[dict[str, str]] = [{"type": "image"}]
    if clean_prompt:
        user_content.append({"type": "text", "text": clean_prompt})
    return {
        "id": record_id,
        "image": image,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ],
        "prompt": prompt,
        "response": response,
        "source_line": source_index + 1,
    }


def _read_training_manifest(path: Path) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    records: list[dict[str, Any]] = []
    ids: set[str] = set()
    images: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank training manifest line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"training manifest line {line_number} is not an object")
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            source_id = source.get("source_id")
            record_id = str(source_id if source_id is not None else row.get("id", ""))
            if not record_id:
                raise ValueError(f"training manifest line {line_number} has no ID")
            if record_id in ids:
                raise ValueError(f"duplicate training record ID {record_id!r}")
            image_value = row.get("image", source.get("image"))
            image = _safe_image(image_value)
            ids.add(record_id)
            images.add(image)
            records.append(row)
    if not records:
        raise ValueError("training manifest is empty")
    return records, ids, images


def _score(seed: int, group: str, record_id: str, image: str, index: int) -> int:
    value = f"vdflash-h11-v1:{seed}:{group}:{record_id}:{image}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:16], "big")


def _select_reference(
    records: list[dict[str, Any]], count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Sample unique images so one duplicated image cannot dominate the reference.
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    seen_images: set[str] = set()
    for index, row in enumerate(records):
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        record_id = str(source.get("source_id", row.get("id", "")))
        image = _safe_image(row.get("image", source.get("image")))
        if image in seen_images:
            continue
        seen_images.add(image)
        candidates.append((_score(seed, "reference", record_id, image, index), index, row))
    if len(candidates) < count:
        raise ValueError(f"requested {count} references but only {len(candidates)} unique train images exist")
    selected = sorted(candidates)[:count]
    selected_indices = {index for _, index, _ in selected}
    reference = []
    non_reference = []
    for index, row in enumerate(records):
        if index in selected_indices:
            output = dict(row)
            output["h11_train_manifest_index"] = index
            reference.append(output)
        else:
            non_reference.append(row)
    return reference, non_reference


def build_groups(
    *,
    source_annotations: Path,
    train_manifest: Path,
    output_dir: Path,
    reference_count: int = 1000,
    eval_count: int = 200,
    seed: int = 20261001,
    overwrite: bool = False,
) -> dict[str, Any]:
    if reference_count < 1 or eval_count < 1:
        raise ValueError("reference_count and eval_count must be positive")
    output_dir = output_dir.expanduser().resolve()
    output_files = [
        output_dir / "reference_manifest.jsonl",
        output_dir / "heldout_eval_manifest.jsonl",
        output_dir / "groups.meta.json",
    ]
    if any(path.exists() for path in output_files) and not overwrite:
        raise FileExistsError(f"group output already exists: {output_dir}; pass --overwrite")

    train_records, train_ids, train_images = _read_training_manifest(train_manifest)
    reference, _ = _select_reference(train_records, reference_count, seed)

    # Keep the best eval_count deterministic hashes in memory; the 558K source is
    # streamed, and duplicate images are rejected before entering the reservoir.
    heap: list[tuple[int, int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    stats = {
        "source_records": 0,
        "invalid_records": 0,
        "excluded_train_id": 0,
        "excluded_train_image": 0,
        "duplicate_source_id": 0,
        "duplicate_source_image": 0,
        "heldout_candidates": 0,
    }
    for source_index, raw in _iter_source(source_annotations):
        stats["source_records"] += 1
        raw_id = raw.get("id")
        image_value = raw.get("image")
        normalized = _prompt_response(raw)
        if raw_id is None or normalized is None:
            stats["invalid_records"] += 1
            continue
        record_id = str(raw_id)
        try:
            image = _safe_image(image_value)
        except ValueError:
            stats["invalid_records"] += 1
            continue
        if record_id in seen_ids:
            stats["duplicate_source_id"] += 1
            raise ValueError(f"duplicate source ID {record_id!r}; refusing ambiguous split")
        seen_ids.add(record_id)
        if record_id in train_ids:
            stats["excluded_train_id"] += 1
            continue
        if image in train_images:
            stats["excluded_train_image"] += 1
            continue
        if image in seen_images:
            stats["duplicate_source_image"] += 1
            continue
        seen_images.add(image)
        stats["heldout_candidates"] += 1
        prompt, response = normalized
        manifest_id = f"heldout:{record_id}:{source_index}"
        manifest_row = _caption_manifest_record(
            manifest_id, image, prompt, response, source_index=source_index
        )
        manifest_row["source_id"] = record_id
        rank = _score(seed, "heldout_eval", record_id, image, source_index)
        item = (-rank, -source_index, manifest_row)
        if len(heap) < eval_count:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)

    if len(heap) < eval_count:
        raise ValueError(
            f"requested {eval_count} held-out eval images but found {len(heap)} after exclusions"
        )
    heldout = [row for _, _, row in sorted(heap, key=lambda item: (-item[0], -item[1]))]

    ref_ids = {str((row.get("source") or {}).get("source_id", row.get("id"))) for row in reference}
    ref_images = {_safe_image(row.get("image", (row.get("source") or {}).get("image"))) for row in reference}
    eval_ids = {str(row["source_id"]) for row in heldout}
    eval_images = {_safe_image(row["image"]) for row in heldout}
    if ref_ids & eval_ids or ref_images & eval_images or train_images & eval_images:
        raise AssertionError("split construction produced overlapping record IDs or image paths")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in ((output_files[0], reference), (output_files[1], heldout)):
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    metadata = {
        "format": "vdflash-h11-image-caption-groups-v1",
        "seed": seed,
        "source_annotations": str(source_annotations.resolve()),
        "source_annotations_sha256": _sha256(source_annotations),
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": _sha256(train_manifest),
        "training_records": len(train_records),
        "training_unique_image_paths": len(train_images),
        "reference_records": len(reference),
        "reference_unique_image_paths": len(ref_images),
        "heldout_eval_records": len(heldout),
        "heldout_eval_unique_image_paths": len(eval_images),
        "overlap_train_eval_ids": len(train_ids & eval_ids),
        "overlap_train_eval_image_paths": len(train_images & eval_images),
        "overlap_reference_eval_ids": len(ref_ids & eval_ids),
        "overlap_reference_eval_image_paths": len(ref_images & eval_images),
        "source_scan": stats,
        "notes": [
            "Reference rows are sampled from the exact LLaVA68K training manifest.",
            "Held-out rows are from the source annotations and exclude all training record IDs and image paths.",
            "This guarantees path-level disjointness, not perceptual/near-duplicate image disjointness.",
            "Generate teacher features for heldout_eval_manifest with the same frozen target, processor, image resolution, and max length as Phase 2.",
        ],
    }
    fd, temp_name = tempfile.mkstemp(prefix=".groups.meta.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output_files[2])
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", type=Path, required=True,
                        help="Full LLaVA-Pretrain 558K JSON array or flat JSONL")
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="Exact normalized manifest used for the checkpoint's LLaVA68K training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-count", type=int, default=1000)
    parser.add_argument("--eval-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_groups(
        source_annotations=args.source_annotations,
        train_manifest=args.train_manifest,
        output_dir=args.output_dir,
        reference_count=args.reference_count,
        eval_count=args.eval_count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
