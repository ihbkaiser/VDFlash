from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from tqdm.auto import tqdm

from .config import DFlashTrainConfig
from .prepare_responses import prepare_responses
from .real_data import _iter_json_array
from .target import Qwen25VLTargetAdapter


DEFAULT_PROMPT = (
    "Write a terse but informative summary of the picture.\n\n"
    "Please answer with at least 1000 words"
)


def _clean_image_path(value: str) -> str:
    image = PurePosixPath(value.replace("\\", "/"))
    if image.is_absolute() or ".." in image.parts or not image.parts:
        raise ValueError(f"unsafe image path in dataset: {value!r}")
    return str(image)


def _read_training_exclusions(path: Path) -> tuple[set[str], set[str], str, str]:
    ids: set[str] = set()
    images: set[str] = set()
    prompts: Counter[str] = Counter()
    teachers: Counter[str] = Counter()
    with path.open(encoding="utf-8") as reader:
        for line_number, line in enumerate(
            tqdm(reader, desc="Read 68K exclusion manifest", unit="lines", dynamic_ncols=True), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                continue
            if row.get("id") is not None:
                ids.add(str(row["id"]))
            if isinstance(row.get("image"), str):
                images.add(_clean_image_path(row["image"]))
            if isinstance(row.get("prompt"), str) and row["prompt"].strip():
                prompts[row["prompt"].strip()] += 1
            if isinstance(row.get("_teacher_model"), str) and row["_teacher_model"].strip():
                teachers[row["_teacher_model"].strip()] += 1
    prompt = prompts.most_common(1)[0][0] if prompts else DEFAULT_PROMPT
    if len(prompts) > 1:
        print(f"[prompt] found {len(prompts)} training prompts; using most common ({prompts[prompt]} rows)")
    teacher = teachers.most_common(1)[0][0] if teachers else "qwen2.5-VL-3b"
    return ids, images, prompt, teacher


def _score(seed: int, sample_id: str, source_index: int) -> int:
    raw = f"vdflash-heldout-v1:{seed}:{sample_id}:{source_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")


def build_heldout_manifest(
    source_json: str | Path,
    train_jsonl: str | Path,
    image_root: str | Path,
    output_jsonl: str | Path,
    *,
    num_samples: int,
    seed: int = 2026,
    prompt: str | None = None,
    overwrite: bool = False,
    candidate_pool_multiplier: int = 10,
) -> dict[str, Any]:
    """Select unseen LLaVA-Pretrain images and write a Qwen chat manifest."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if candidate_pool_multiplier < 1:
        raise ValueError("candidate_pool_multiplier must be positive")
    source_json = Path(source_json).expanduser().resolve()
    train_jsonl = Path(train_jsonl).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output_jsonl = Path(output_jsonl).expanduser().resolve()
    excluded_ids, excluded_images, inferred_prompt, _ = _read_training_exclusions(train_jsonl)
    prompt = (prompt or inferred_prompt).strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    # Keep a deterministic oversampled pool, then check only those paths on the
    # shared filesystem. Stat-ing every unused image during the JSON scan is very
    # slow on network storage.
    candidate_pool_size = max(num_samples, num_samples * candidate_pool_multiplier)
    candidates: list[tuple[int, int, dict[str, str]]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    missing_images = 0
    with tqdm(
        _iter_json_array(source_json),
        desc="Scan/hash LLaVA annotations",
        unit="records",
        dynamic_ncols=True,
    ) as scan_bar:
        for source_index, row in enumerate(scan_bar):
            raw_id, raw_image = row.get("id"), row.get("image")
            if raw_id is None or not isinstance(raw_image, str) or not raw_image.strip():
                continue
            sample_id = str(raw_id)
            image = _clean_image_path(raw_image)
            if sample_id in excluded_ids or image in excluded_images:
                continue
            if sample_id in seen_ids or image in seen_images:
                continue
            seen_ids.add(sample_id)
            seen_images.add(image)
            image_path = image_root / image
            score = _score(seed, sample_id, source_index)
            item = {"id": sample_id, "image": image, "image_path": str(image_path)}
            candidate = (-score, -source_index, item)
            if len(candidates) < candidate_pool_size:
                heapq.heappush(candidates, candidate)
            else:
                worst_score, worst_index = -candidates[0][0], -candidates[0][1]
                if (score, source_index) < (worst_score, worst_index):
                    heapq.heapreplace(candidates, candidate)
            if (source_index + 1) % 5_000 == 0:
                scan_bar.set_postfix(
                    unique_unused=f"{len(seen_ids):,}",
                    pool=f"{len(candidates):,}/{candidate_pool_size}",
                    refresh=False,
                )

    ranked_candidates = sorted(
        ((-score, -index, item) for score, index, item in candidates),
        key=lambda item: (item[0], item[1]),
    )
    selected: list[tuple[int, int, dict[str, str]]] = []
    for score, source_index, item in tqdm(
        ranked_candidates,
        desc="Check selected image files",
        unit="paths",
        dynamic_ncols=True,
    ):
        if not Path(item["image_path"]).is_file():
            missing_images += 1
            continue
        selected.append((score, source_index, item))
        if len(selected) == num_samples:
            break
    if len(selected) < num_samples:
        raise ValueError(
            f"requested {num_samples} held-out samples, but only {len(selected)} images exist among the "
            f"top {len(ranked_candidates)} candidates ({missing_images} missing). "
            "Increase --candidate-pool-multiplier and retry."
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {output_jsonl}")
    tmp_path = output_jsonl.with_name(output_jsonl.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as writer:
            for score, source_index, item in tqdm(
                selected, desc="Write held-out manifest", unit="samples", dynamic_ncols=True
            ):
                image_uri = Path(item["image_path"]).as_uri()
                text = re.sub(r"<image>", "", prompt, flags=re.IGNORECASE).strip()
                record = {
                    "id": item["id"],
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_uri},
                            {"type": "text", "text": text},
                        ],
                    }],
                    "source": {
                        "dataset": "LLaVA-Pretrain",
                        "source_id": item["id"],
                        "source_index": source_index,
                        "image": item["image"],
                        "selection_score": f"{score:032x}",
                        "seed": seed,
                    },
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        tmp_path.replace(output_jsonl)
    finally:
        tmp_path.unlink(missing_ok=True)
    summary = {
        "source_json": str(source_json),
        "train_jsonl": str(train_jsonl),
        "image_root": str(image_root),
        "manifest": str(output_jsonl),
        "selected": len(selected),
        "candidate_pool_size": candidate_pool_size,
        "selected_records_with_images": len(selected),
        "missing_unused_images": missing_images,
        "excluded_train_ids": len(excluded_ids),
        "excluded_train_images": len(excluded_images),
        "seed": seed,
        "prompt": prompt,
    }
    print("[heldout manifest] " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _write_flat_responses(target_manifest: Path, flat_output: Path, teacher_model: str, tokenizer: Any) -> int:
    rows = 0
    flat_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = flat_output.with_name(flat_output.name + ".tmp")
    try:
        with target_manifest.open(encoding="utf-8") as reader, tmp_path.open("w", encoding="utf-8") as writer:
            lines = tqdm(reader, desc="Format flat held-out JSONL", unit="records", dynamic_ncols=True)
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = row.get("source", {})
                target = row.get("target_response", {})
                image = source.get("image")
                if not image or not isinstance(target.get("text"), str):
                    continue
                messages = row.get("messages", [])
                prompt_text = "\n\n".join(
                    part.get("text", "")
                    for message in messages
                    for part in message.get("content", [])
                    if part.get("type") == "text"
                ).strip()
                item = {
                    "id": str(row["id"]),
                    "image": image,
                    "prompt": prompt_text,
                    "response": tokenizer.decode(target.get("token_ids", []), skip_special_tokens=True),
                    "_teacher_model": teacher_model,
                }
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1
        tmp_path.replace(flat_output)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"[flat responses] records={rows} path={flat_output}", flush=True)
    return rows


def main() -> None:  # pragma: no cover - exercised through component tests
    parser = argparse.ArgumentParser(
        description="Select unused LLaVA-Pretrain images and generate held-out Qwen responses"
    )
    parser.add_argument("--config", required=True, help="same Qwen target config used for the 68K generation")
    parser.add_argument("--source-json", required=True, help="LLaVA-Pretrain/blip_laion_cc_sbu_558k.json")
    parser.add_argument("--train-jsonl", required=True, help="existing 68K JSONL; IDs and images are excluded")
    parser.add_argument("--image-root", required=True, help="extracted LLaVA-Pretrain directory")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--candidate-pool-multiplier",
        type=int,
        default=10,
        help="oversample this many candidates before checking files on shared storage",
    )
    parser.add_argument("--prompt", default=None, help="defaults to the most common prompt in train-jsonl")
    parser.add_argument("--manifest", required=True, help="selected held-out input manifest")
    parser.add_argument("--target-output", required=True, help="full target manifest with exact token IDs")
    parser.add_argument("--flat-output", required=True, help="JSONL in the same id/image/prompt/response format as 68K")
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    target_output = Path(args.target_output).expanduser().resolve()
    flat_output = Path(args.flat_output).expanduser().resolve()
    for path in (target_output, flat_output):
        if path.exists() and not args.overwrite_output:
            raise FileExistsError(f"output exists: {path}; pass --overwrite-output to replace it")
    summary = build_heldout_manifest(
        args.source_json, args.train_jsonl, args.image_root, manifest,
        num_samples=args.num_samples, seed=args.seed, prompt=args.prompt,
        overwrite=args.overwrite_output,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
    )
    config = DFlashTrainConfig.from_file(args.config)
    if config.stage != "multimodal":
        raise ValueError("held-out LLaVA generation requires a multimodal config")
    print(f"[model] loading {config.target_model} on {config.device}", flush=True)
    # Honor the configured device explicitly; otherwise the adapter defaults to
    # the process's generic "cuda" device and can follow a different device map.
    adapter = Qwen25VLTargetAdapter.from_pretrained(config, device=config.device)
    print(f"[generation] samples={args.num_samples} max_new_tokens={config.response_max_new_tokens}", flush=True)
    prepare_responses(
        adapter,
        [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()],
        max_new_tokens=config.response_max_new_tokens,
        max_seq_length=config.max_seq_length,
        output_path=target_output,
    )
    _, _, _, teacher_model = _read_training_exclusions(Path(args.train_jsonl).expanduser().resolve())
    _write_flat_responses(target_output, flat_output, teacher_model, adapter.processor.tokenizer)
    metadata_path = flat_output.with_suffix(flat_output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
