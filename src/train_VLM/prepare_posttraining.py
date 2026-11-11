from __future__ import annotations

"""Prepare DFlash visual-10% post-training manifests from the raw HF video pool.

This module has three explicit stages:
  prepare: resolve/verify source files, make video-disjoint splits, extract only
           selected videos, and write Qwen-compatible prompts;
  targets: generate exact greedy Qwen target token IDs (no hidden-state/logit cache);
  validate: check split isolation, local media, target coverage, and realized grids.

The module never writes into the raw source tree and never creates feature or
full-vocabulary logit caches.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from tqdm.auto import tqdm

from .config import DFlashTrainConfig
from .real_data import _iter_json_array
from .target import Qwen25VLTargetAdapter, load_jsonl, validate_manifest_record


DEFAULT_HF_REPO = "Tphuc15/dflash-posttraining-raw"
DEFAULT_SEED = 20261104
SPLIT_RATIOS = (("train", 0.8), ("validation", 0.1), ("test", 0.1))
LONG_GROUPS = {"1_2_m_youtube_v0_1", "2_3_m_youtube_v0_1",
               "1_2_m_academic_v0_1", "2_3_m_academic_v0_1"}
SHORT_GROUPS = {"0_30_s_youtube_v0_1", "30_60_s_youtube_v0_1"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
_VISUAL_TAG = re.compile(r"<\s*(?:image|video)(?:\s+[^>]*)?\s*>", re.IGNORECASE)
_SPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class PromptCandidate:
    video_key: str
    group: str
    duration_bucket: str
    task: str
    annotation_file: str
    row_index: int
    turn_index: int
    source_id: str
    question: str

    @property
    def sample_id(self) -> str:
        return (f"{self.video_key}-{self.task}-"
                f"{Path(self.annotation_file).stem}-{self.row_index}-{self.turn_index}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as writer:
        json.dump(value, writer, ensure_ascii=False, indent=2, sort_keys=True)
        writer.write("\n")
        writer.flush()
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as writer:
            for record in records:
                writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            writer.flush()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield value


def _stable_score(seed: int, *parts: object) -> int:
    payload = "dflash-posttraining-v1:" + str(seed) + ":" + ":".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:16], "big")


def _video_key(value: str) -> str:
    stem = PurePosixPath(value.replace("\\", "/")).stem
    for prefix in ("ytb_", "v_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def _duration_bucket(group: str) -> str:
    if group in LONG_GROUPS:
        return "long_1_to_3_min_source_group"
    if group in SHORT_GROUPS:
        return "short_0_to_60_sec_source_group"
    raise ValueError(f"unsupported source duration group: {group!r}")


def _group_priority(group: str) -> tuple[int, int]:
    # If the same normalized video ID occurs in multiple source archives, retain
    # its longest annotated clip for the long-context pool.
    if group.startswith("2_3_m_"):
        duration_priority = 4
    elif group.startswith("1_2_m_"):
        duration_priority = 3
    elif group.startswith("30_60_s_"):
        duration_priority = 2
    else:
        duration_priority = 1
    return duration_priority, 1


def _normalise_repo_id(value: str) -> str:
    value = value.strip()
    for prefix in ("hf://datasets/", "https://huggingface.co/datasets/"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip("/")
            break
    if value.count("/") != 1 or not all(value.split("/", 1)):
        raise ValueError("source must be a local dataset directory or HF dataset repo_id")
    return value


def resolve_source(source: str, revision: str | None = None) -> tuple[Path, str | None]:
    """Resolve a local raw-data root or snapshot a private HF dataset repo."""
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve(), None
    repo_id = _normalise_repo_id(source)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("huggingface_hub is required for an HF repo source") from exc
    info = HfApi().dataset_info(repo_id, revision=revision)
    resolved_revision = str(info.sha)
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        token=True,
    )
    return Path(snapshot).resolve(), resolved_revision


def _find_file(root: Path, name: str) -> Path:
    candidates = (root / name, root / "metadata" / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not find {name} at dataset root or metadata/")


def _safe_source_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"unsafe relative source path: {relative!r}")
    result = root.joinpath(*posix.parts)
    # HF snapshot files are symlinks into the Hub blob cache. Check directory
    # components for escape while allowing the final file symlink.
    if not result.parent.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes dataset root: {relative!r}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Validate the raw collection contract and all source archive/JSON hashes."""
    plan = json.loads(_find_file(root, "download_plan.json").read_text(encoding="utf-8"))
    download_status = json.loads(_find_file(root, "download_status.json").read_text(encoding="utf-8"))
    summary = json.loads(_find_file(root, "collection_summary.json").read_text(encoding="utf-8"))
    inventory_path = _find_file(root, "raw_video_inventory.jsonl")

    if plan.get("stage") != "raw_collection_only":
        raise ValueError("source download plan is not a raw-collection plan")
    if plan.get("repo_id") != "lmms-lab/LLaVA-Video-178K":
        raise ValueError(f"unexpected raw source repository: {plan.get('repo_id')!r}")
    if download_status.get("status") != "complete" or download_status.get("errors"):
        raise ValueError("source download_status.json is incomplete or contains errors")
    if summary.get("videos_extracted") or summary.get("target_responses_generated") or summary.get("features_cached"):
        raise ValueError("expected the uncached raw source dataset")
    planned = {row["path"]: row for row in plan.get("files", [])}
    verified = {row["path"] for row in download_status.get("files", [])
                if row.get("status") == "verified"}
    if set(planned) != verified:
        raise ValueError("verified source files differ from the pinned download plan")
    if summary.get("verified_source_files") != len(planned):
        raise ValueError("collection summary does not cover every planned source file")

    checksum_path = _find_file(root, "SHA256SUMS")
    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError("malformed SHA256SUMS entry")
        checksum_rows[relative] = digest

    data_rows = [row for row in plan["files"] if row.get("kind") in {"archive", "annotation"}]
    progress = tqdm(data_rows, desc="Verify source SHA256", unit="files", dynamic_ncols=True)
    for row in progress:
        relative = row["path"]
        path = _safe_source_path(root, relative)
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise ValueError(f"missing or wrong-sized source file: {relative}")
        expected = checksum_rows.get(relative) or row.get("source_sha256")
        if not expected:
            raise ValueError(f"no recorded checksum for source file: {relative}")
        if expected.removeprefix("sha256:") != _sha256_file(path):
            raise ValueError(f"source SHA256 mismatch: {relative}")
    if summary.get("unique_video_keys", 0) < 1:
        raise ValueError("source catalog has no video IDs")
    return plan, download_status, summary, inventory_path


def _load_unique_video_inventory(path: Path) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in tqdm(_read_jsonl(path), desc="Read video inventory", unit="videos", dynamic_ncols=True):
        if row.get("excluded_vdc"):
            continue
        key = str(row.get("video_key", "")).strip()
        group = str(row.get("group", "")).strip()
        member = str(row.get("member", "")).strip()
        archive = str(row.get("archive", "")).strip()
        if not key or not group or not member or not archive:
            raise ValueError("raw video inventory row is missing key/group/archive/member")
        refs = row.get("annotation_refs")
        if not isinstance(refs, list) or not refs:
            continue
        candidate = dict(row)
        current = unique.get(key)
        if current is None:
            unique[key] = candidate
        else:
            duplicates += 1
            candidate_priority = (_group_priority(group), len(refs), archive, member)
            current_priority = (_group_priority(str(current["group"])),
                                len(current.get("annotation_refs", [])),
                                str(current["archive"]), str(current["member"]))
            if candidate_priority > current_priority:
                unique[key] = candidate
    if not unique:
        raise ValueError("no eligible, annotated videos remain after VDC exclusion")
    print(f"[inventory] unique={len(unique)} duplicate_rows={duplicates} VDC-excluded rows ignored", flush=True)
    return unique


def _assign_video_splits(videos: dict[str, dict[str, Any]], seed: int) -> dict[str, str]:
    by_duration: dict[str, list[str]] = defaultdict(list)
    for key, row in videos.items():
        by_duration[_duration_bucket(str(row["group"]))].append(key)
    assignment: dict[str, str] = {}
    for duration_bucket, keys in sorted(by_duration.items()):
        ordered = sorted(keys, key=lambda key: (_stable_score(seed, "video-split", key), key))
        train_end = int(len(ordered) * 0.8)
        validation_end = train_end + int(len(ordered) * 0.1)
        for index, key in enumerate(ordered):
            if index < train_end:
                assignment[key] = "train"
            elif index < validation_end:
                assignment[key] = "validation"
            else:
                assignment[key] = "test"
        print(f"[split] {duration_bucket}: train={train_end} validation={validation_end-train_end} "
              f"test={len(ordered)-validation_end}", flush=True)
    return assignment


def _normalise_question(value: Any) -> str:
    text = str(value or "")
    text = _VISUAL_TAG.sub(" ", text)
    text = _SPACE.sub(" ", text)
    return "\n".join(part.strip() for part in text.splitlines() if part.strip()).strip()


def _conversation_pairs(row: dict[str, Any]) -> Iterable[tuple[int, str]]:
    conversation = row.get("conversations")
    if not isinstance(conversation, list):
        return
    for index, message in enumerate(conversation):
        if not isinstance(message, dict) or str(message.get("from", "")).lower() not in {"human", "user"}:
            continue
        question = _normalise_question(message.get("value"))
        if not question:
            continue
        # Source assistant text is deliberately not returned: target responses
        # are generated later by the pinned Qwen target model.
        if index + 1 < len(conversation):
            following = conversation[index + 1]
            if isinstance(following, dict) and str(following.get("from", "")).lower() in {"gpt", "assistant"}:
                yield index, question


def _load_prompt_candidates(
    root: Path,
    inventory_path: Path,
    videos: dict[str, dict[str, Any]],
    split_by_video: dict[str, str],
    seed: int,
) -> dict[tuple[str, str, str], dict[str, list[PromptCandidate]]]:
    """Stream original JSON arrays and retain prompt candidates for eligible IDs."""
    selected_by_file: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    for key, video in videos.items():
        for ref in video["annotation_refs"]:
            relative = str(ref["file"])
            task = str(ref.get("task", ""))
            if task not in {"caption", "open_ended"}:
                continue
            row_index = int(ref["row_index"])
            previous = selected_by_file[relative].get(row_index)
            value = (key, task)
            if previous is not None and previous != value:
                raise ValueError(f"annotation row {relative}:{row_index} maps to multiple video IDs")
            selected_by_file[relative][row_index] = value

    prompts: dict[tuple[str, str, str], dict[str, list[PromptCandidate]]] = defaultdict(lambda: defaultdict(list))
    for relative, selected_rows in sorted(selected_by_file.items()):
        path = _safe_source_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"annotation JSON missing from source snapshot: {relative}")
        task_by_index = {row_index: value for row_index, value in selected_rows.items()}
        for row_index, row in enumerate(tqdm(
            _iter_json_array(path), desc=f"Index prompts: {Path(relative).name}", unit="rows", dynamic_ncols=True
        )):
            selected = task_by_index.get(row_index)
            if selected is None:
                continue
            key, task = selected
            source_video = row.get("video", row.get("id", ""))
            if _video_key(str(source_video)) != key:
                raise ValueError(f"annotation/video ID mismatch at {relative}:{row_index}")
            video = videos[key]
            for turn_index, question in _conversation_pairs(row):
                candidate = PromptCandidate(
                    video_key=key,
                    group=str(video["group"]),
                    duration_bucket=_duration_bucket(str(video["group"])),
                    task=task,
                    annotation_file=relative,
                    row_index=row_index,
                    turn_index=turn_index,
                    source_id=str(row.get("id", key)),
                    question=question,
                )
                prompts[(split_by_video[key], candidate.duration_bucket, task)][key].append(candidate)

    # Stable ordering and within-video deduplication prevent near-duplicate
    # question rows from overrepresenting a single clip.
    for by_video in prompts.values():
        for key, candidates in by_video.items():
            unique: dict[str, PromptCandidate] = {}
            for candidate in candidates:
                normalized = candidate.question.casefold()
                old = unique.get(normalized)
                rank = (_stable_score(seed, "prompt", candidate.sample_id), candidate.sample_id)
                if old is None or rank < (_stable_score(seed, "prompt", old.sample_id), old.sample_id):
                    unique[normalized] = candidate
            by_video[key] = sorted(
                unique.values(), key=lambda item: (_stable_score(seed, "prompt", item.sample_id), item.sample_id)
            )
    return prompts


def _round_task_targets(count: int, caption_fraction: float) -> dict[str, int]:
    caption = int(round(count * caption_fraction))
    return {"caption": caption, "open_ended": count - caption}


def _select_prompt_bucket(
    by_task: dict[str, dict[str, list[PromptCandidate]]],
    count: int,
    *,
    caption_fraction: float,
    max_per_video: int,
    seed: int,
    bucket_key: str,
) -> list[PromptCandidate]:
    targets = _round_task_targets(count, caption_fraction)
    chosen: list[PromptCandidate] = []
    used_per_video: Counter[str] = Counter()
    chosen_ids: set[str] = set()

    def task_candidates(task: str, round_index: int) -> list[PromptCandidate]:
        rows: list[PromptCandidate] = []
        for video_key, candidates in by_task.get(task, {}).items():
            if used_per_video[video_key] >= max_per_video:
                continue
            if round_index == 0:
                available = candidates[:1]
            else:
                available = candidates[round_index:round_index + 1]
            rows.extend(item for item in available if item.sample_id not in chosen_ids)
        return sorted(rows, key=lambda item: (_stable_score(seed, bucket_key, task, item.sample_id), item.sample_id))

    for task in ("caption", "open_ended"):
        remaining = targets[task]
        # Prefer one prompt/video first; use a second prompt only if task coverage
        # requires it and the training per-video limit permits it.
        for round_index in range(max_per_video):
            if remaining <= 0:
                break
            for candidate in task_candidates(task, round_index):
                if remaining <= 0:
                    break
                if used_per_video[candidate.video_key] >= max_per_video:
                    continue
                chosen.append(candidate)
                chosen_ids.add(candidate.sample_id)
                used_per_video[candidate.video_key] += 1
                remaining -= 1

    # If one task has fewer annotations than its target, backfill from the other
    # task without changing the split or exceeding the video repetition limit.
    if len(chosen) < count:
        backfill: list[PromptCandidate] = []
        for task in ("open_ended", "caption"):
            for video_key, candidates in by_task.get(task, {}).items():
                if used_per_video[video_key] >= max_per_video:
                    continue
                backfill.extend(item for item in candidates if item.sample_id not in chosen_ids)
        backfill.sort(key=lambda item: (_stable_score(seed, bucket_key, "backfill", item.sample_id), item.sample_id))
        for candidate in backfill:
            if len(chosen) >= count:
                break
            if used_per_video[candidate.video_key] >= max_per_video:
                continue
            chosen.append(candidate)
            chosen_ids.add(candidate.sample_id)
            used_per_video[candidate.video_key] += 1
    if len(chosen) != count:
        available = sum(len(rows) for task_rows in by_task.values() for rows in task_rows.values())
        raise ValueError(
            f"not enough prompts for {bucket_key}: requested {count}, selected {len(chosen)}, "
            f"candidate prompts={available}; lower sample counts or repetition requirements"
        )
    return chosen


def _parse_budget_mixture(value: str) -> list[dict[str, Any]]:
    """Parse entries as visual_tokens:fraction:nframes, preserving exact quotas."""
    result = []
    for item in value.split(","):
        fields = [field.strip() for field in item.split(":")]
        if len(fields) != 3:
            raise argparse.ArgumentTypeError("budget entries must be visual_tokens:fraction:nframes")
        try:
            tokens, fraction, frames = int(fields[0]), float(fields[1]), int(fields[2])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("invalid visual budget mixture entry") from exc
        if tokens < 1 or frames < 2 or frames % 2 or fraction <= 0:
            raise argparse.ArgumentTypeError("budgets and fractions must be positive; nframes must be even >= 2")
        result.append({"target_visual_tokens": tokens, "fraction": fraction, "nframes": frames})
    if not result or not math.isclose(sum(row["fraction"] for row in result), 1.0, abs_tol=1e-6):
        raise argparse.ArgumentTypeError("visual budget fractions must sum to 1")
    if len({row["target_visual_tokens"] for row in result}) != len(result):
        raise argparse.ArgumentTypeError("visual budget targets must be unique")
    return result


def _largest_remainder_counts(total: int, rows: list[dict[str, Any]]) -> list[int]:
    exact = [total * row["fraction"] for row in rows]
    counts = [math.floor(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(range(len(rows)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _assign_visual_budgets(
    records: list[PromptCandidate],
    *,
    budgets: list[dict[str, Any]],
    spatial_tokens_per_temporal_patch: int,
    seed: int,
    split: str,
) -> dict[str, dict[str, int]]:
    ordered = sorted(records, key=lambda row: (_stable_score(seed, "budget", split, row.sample_id), row.sample_id))
    counts = _largest_remainder_counts(len(ordered), budgets)
    cursor = 0
    assigned: dict[str, dict[str, int]] = {}
    for budget, count in zip(budgets, counts):
        pixel_budget = spatial_tokens_per_temporal_patch * 28 * 28
        for candidate in ordered[cursor:cursor + count]:
            assigned[candidate.sample_id] = {
                "target_visual_tokens": budget["target_visual_tokens"],
                "nframes": budget["nframes"],
                "min_pixels": pixel_budget,
                "max_pixels": pixel_budget,
                "spatial_tokens_per_temporal_patch": spatial_tokens_per_temporal_patch,
            }
        cursor += count
    return assigned


def _select_records(
    prompts: dict[tuple[str, str, str], dict[str, list[PromptCandidate]]],
    *,
    sample_counts: dict[str, int],
    long_fraction: float,
    caption_fraction: float,
    train_max_per_video: int,
    validation_max_per_video: int,
    test_max_per_video: int,
    seed: int,
) -> dict[str, list[PromptCandidate]]:
    selected: dict[str, list[PromptCandidate]] = {split: [] for split in sample_counts}
    for split, requested in sample_counts.items():
        long_count = int(round(requested * long_fraction))
        counts = {
            "long_1_to_3_min_source_group": long_count,
            "short_0_to_60_sec_source_group": requested - long_count,
        }
        repetition_limit = {
            "train": train_max_per_video,
            "validation": validation_max_per_video,
            "test": test_max_per_video,
        }[split]
        for duration_bucket, count in counts.items():
            task_rows = {
                task: prompts.get((split, duration_bucket, task), {})
                for task in ("caption", "open_ended")
            }
            selected[split].extend(_select_prompt_bucket(
                task_rows,
                count,
                caption_fraction=caption_fraction,
                max_per_video=repetition_limit,
                seed=seed,
                bucket_key=f"{split}:{duration_bucket}",
            ))
        selected[split].sort(key=lambda row: (_stable_score(seed, "final-order", split, row.sample_id), row.sample_id))
    return selected


def _selected_video_rows(
    videos: dict[str, dict[str, Any]], selected: dict[str, list[PromptCandidate]]
) -> dict[str, dict[str, Any]]:
    selected_keys = {candidate.video_key for rows in selected.values() for candidate in rows}
    return {key: videos[key] for key in selected_keys}


def _safe_member_name(value: str) -> PurePosixPath:
    member = PurePosixPath(value.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"unsafe tar member path: {value!r}")
    return member


def _extract_selected_videos(
    source_root: Path,
    output_dir: Path,
    videos: dict[str, dict[str, Any]],
    selected: dict[str, list[PromptCandidate]],
    *,
    resume: bool,
) -> dict[str, Path]:
    output_media = output_dir / "videos"
    output_media.mkdir(parents=True, exist_ok=True)
    wanted_keys = {candidate.video_key for rows in selected.values() for candidate in rows}
    archive_to_members: dict[str, dict[str, str]] = defaultdict(dict)
    video_paths: dict[str, Path] = {}
    expected_sizes: dict[str, int] = {}
    for key in sorted(wanted_keys):
        row = videos[key]
        suffix = PurePosixPath(str(row["member"])).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported video suffix for {key}: {suffix}")
        destination = output_media / f"{key}{suffix}"
        video_paths[key] = destination.resolve()
        expected_sizes[key] = int(row["bytes"])
        archive_to_members[str(row["archive"])][str(row["member"])] = key
        if destination.exists():
            if not resume:
                raise FileExistsError(f"video already exists; pass --resume to reuse it: {destination}")
            if destination.stat().st_size != expected_sizes[key]:
                raise ValueError(f"existing extracted video has wrong size: {destination}")

    status_path = output_dir / "media_status.json"
    completed = {
        key for key, path in video_paths.items()
        if path.is_file() and path.stat().st_size == expected_sizes[key]
    }
    remaining_bytes = sum(expected_sizes[key] for key in wanted_keys - completed)
    reserve_bytes = 5 * 1024 ** 3
    if shutil.disk_usage(output_dir).free < remaining_bytes + reserve_bytes:
        raise RuntimeError(
            f"insufficient free space for selected videos plus 5 GiB reserve; "
            f"need about {(remaining_bytes + reserve_bytes) / 1024**3:.2f} GiB"
        )
    print(f"[extract] selected_media={len(wanted_keys)} remaining_bytes={remaining_bytes / 1e9:.2f} GB",
          flush=True)
    archive_names = sorted(archive_to_members)
    progress = tqdm(archive_names, desc="Extract selected videos", unit="archives", dynamic_ncols=True)
    for archive_relative in progress:
        pending = {
            member: key for member, key in archive_to_members[archive_relative].items()
            if key not in completed
        }
        if not pending:
            progress.set_postfix(extracted=len(completed), total=len(wanted_keys))
            continue
        archive_path = _safe_source_path(source_root, archive_relative)
        found: set[str] = set()
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                key = pending.get(member.name)
                if key is None:
                    continue
                safe_name = _safe_member_name(member.name)
                if not member.isfile() or safe_name.suffix.lower() not in VIDEO_SUFFIXES:
                    raise ValueError(f"selected tar member is not a regular video file: {member.name}")
                if member.size != expected_sizes[key]:
                    raise ValueError(f"tar member size mismatch for video {key}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read selected tar member: {member.name}")
                destination = video_paths[key]
                temporary = destination.with_name(destination.name + ".part")
                if temporary.exists() and not resume:
                    raise FileExistsError(f"partial video extraction exists: {temporary}")
                with source, temporary.open("wb") as writer:
                    shutil.copyfileobj(source, writer, length=4 * 1024 * 1024)
                    writer.flush()
                if temporary.stat().st_size != expected_sizes[key]:
                    raise ValueError(f"extracted video size mismatch for {key}")
                temporary.replace(destination)
                found.add(member.name)
                completed.add(key)
                progress.set_postfix(extracted=len(completed), total=len(wanted_keys))
        missing = set(pending) - found
        if missing:
            raise ValueError(f"selected videos missing from {archive_relative}: {len(missing)} members")
        _write_json(status_path, {
            "status": "running", "completed_videos": len(completed),
            "total_videos": len(wanted_keys), "last_archive": archive_relative,
        })
    if completed != wanted_keys:
        raise RuntimeError(f"video extraction incomplete: {len(completed)}/{len(wanted_keys)}")
    _write_json(status_path, {
        "status": "complete", "completed_videos": len(completed),
        "total_videos": len(wanted_keys),
    })
    return video_paths


def _make_manifest_record(
    candidate: PromptCandidate,
    *,
    split: str,
    video_path: Path,
    budget: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    media = {
        "type": "video",
        "video": video_path.as_uri(),
        "nframes": budget["nframes"],
        "min_pixels": budget["min_pixels"],
        "max_pixels": budget["max_pixels"],
    }
    return {
        "id": candidate.sample_id,
        "messages": [{
            "role": "user",
            "content": [media, {"type": "text", "text": candidate.question}],
        }],
        "metadata": {
            "split": split,
            "video_key": candidate.video_key,
            "source_group": candidate.group,
            "duration_bucket": candidate.duration_bucket,
            "task": candidate.task,
            "annotation_file": candidate.annotation_file,
            "annotation_row_index": candidate.row_index,
            "conversation_turn_index": candidate.turn_index,
            "source_annotation_id": candidate.source_id,
            "video_path": str(video_path),
            "target_visual_tokens": budget["target_visual_tokens"],
            "requested_video_frames": budget["nframes"],
            "spatial_tokens_per_temporal_patch": budget["spatial_tokens_per_temporal_patch"],
            "sampling_seed": seed,
            "source_assistant_response_used_as_target": False,
        },
    }


def _ensure_output_dir(output_dir: Path, resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    existing = [path for path in output_dir.iterdir() if path.name not in {".DS_Store"}]
    if existing:
        raise FileExistsError(f"output directory is not empty; use a new directory or pass --resume: {output_dir}")


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_root, source_commit = resolve_source(args.source, args.source_revision)
    if output_dir == source_root or source_root in output_dir.parents or output_dir in source_root.parents:
        raise ValueError("output directory and raw source directory must be separate")
    _ensure_output_dir(output_dir, args.resume)
    plan, _download_status, source_summary, inventory_path = verify_source(source_root)
    videos = _load_unique_video_inventory(inventory_path)
    expected_eligible = source_summary.get("annotated_unique_videos_after_vdc_exclusion")
    if expected_eligible is not None and len(videos) != int(expected_eligible):
        raise ValueError(
            f"eligible video inventory count mismatch: {len(videos)} != {expected_eligible}"
        )
    preparation_contract = {
        "source_repo_id": plan["repo_id"],
        "source_revision": plan["revision"],
        "source_hf_commit": source_commit,
        "seed": args.seed,
        "records": {
            "train": args.train_records,
            "validation": args.validation_records,
            "test": args.test_records,
        },
        "long_fraction": args.long_fraction,
        "caption_fraction": args.caption_fraction,
        "train_max_per_video": args.train_max_per_video,
        "visual_budget_mixture": _parse_budget_mixture(args.visual_budget_mixture),
        "spatial_tokens_per_temporal_patch": args.spatial_tokens_per_temporal_patch,
    }
    contract_path = output_dir / "preparation_contract.json"
    if contract_path.is_file():
        previous_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous_contract != preparation_contract:
            raise ValueError(
                "resume settings/source differ from preparation_contract.json; use a new output directory "
                "to avoid mixing manifests and media"
            )
    elif args.resume and any(path.name != ".DS_Store" for path in output_dir.iterdir()):
        raise ValueError(
            "cannot safely resume a non-empty output directory without preparation_contract.json; "
            "use a new output directory"
        )
    else:
        # Persist the contract before scanning prompts or extracting media so an
        # interrupted multi-hour preparation can be resumed with identical inputs.
        _write_json(contract_path, preparation_contract)
    prior_summary = output_dir / "prepare_summary.json"
    if args.resume and prior_summary.is_file():
        previous = json.loads(prior_summary.read_text(encoding="utf-8"))
        previous_contract = previous.get("preparation_contract")
        if previous_contract != preparation_contract:
            raise ValueError(
                "resume settings/source differ from prepare_summary.json; use a new output directory "
                "to avoid mixing manifests and media"
            )
    split_by_video = _assign_video_splits(videos, args.seed)

    sample_counts = {
        "train": args.train_records,
        "validation": args.validation_records,
        "test": args.test_records,
    }
    if min(sample_counts.values()) < 1:
        raise ValueError("train, validation, and test record counts must all be positive")
    if not 0 < args.long_fraction < 1 or not 0 < args.caption_fraction < 1:
        raise ValueError("long/caption fractions must be strictly between zero and one")
    if args.train_max_per_video < 1:
        raise ValueError("train max-per-video must be positive")
    if args.validation_max_per_video != 1 or args.test_max_per_video != 1:
        raise ValueError("validation and test must use exactly one record per video")
    budgets = _parse_budget_mixture(args.visual_budget_mixture)
    prompts = _load_prompt_candidates(source_root, inventory_path, videos, split_by_video, args.seed)
    selected = _select_records(
        prompts,
        sample_counts=sample_counts,
        long_fraction=args.long_fraction,
        caption_fraction=args.caption_fraction,
        train_max_per_video=args.train_max_per_video,
        validation_max_per_video=args.validation_max_per_video,
        test_max_per_video=args.test_max_per_video,
        seed=args.seed,
    )
    all_candidates = [candidate for rows in selected.values() for candidate in rows]
    if len({candidate.sample_id for candidate in all_candidates}) != len(all_candidates):
        raise ValueError("duplicate sample IDs were selected")
    split_video_ids = {
        split: {candidate.video_key for candidate in rows}
        for split, rows in selected.items()
    }
    if (split_video_ids["train"] & split_video_ids["validation"]
            or split_video_ids["train"] & split_video_ids["test"]
            or split_video_ids["validation"] & split_video_ids["test"]):
        raise RuntimeError("video IDs leaked across train/validation/test splits")

    selected_videos = _selected_video_rows(videos, selected)
    video_paths = _extract_selected_videos(
        source_root, output_dir, selected_videos, selected, resume=args.resume
    )
    budget_counts: dict[str, dict[str, int]] = {}
    task_counts: dict[str, dict[str, int]] = {}
    duration_counts: dict[str, dict[str, int]] = {}
    for split, candidates in selected.items():
        assignments = _assign_visual_budgets(
            candidates, budgets=budgets,
            spatial_tokens_per_temporal_patch=args.spatial_tokens_per_temporal_patch,
            seed=args.seed, split=split,
        )
        records = [
            _make_manifest_record(
                candidate,
                split=split,
                video_path=video_paths[candidate.video_key],
                budget=assignments[candidate.sample_id],
                seed=args.seed,
            )
            for candidate in candidates
        ]
        records.sort(key=lambda row: str(row["id"]))
        expected_path = output_dir / "manifests" / f"{split}.jsonl"
        if expected_path.exists() and not args.resume:
            raise FileExistsError(f"manifest already exists: {expected_path}")
        _write_jsonl(expected_path, records)
        budget_counts[split] = dict(Counter(
            str(record["metadata"]["target_visual_tokens"]) for record in records
        ))
        task_counts[split] = dict(Counter(str(record["metadata"]["task"]) for record in records))
        duration_counts[split] = dict(Counter(str(record["metadata"]["duration_bucket"]) for record in records))
        print(f"[manifest] {split}: records={len(records)} unique_videos={len(split_video_ids[split])} "
              f"tasks={task_counts[split]} visual_targets={budget_counts[split]}", flush=True)

    result = {
        "status": "prepared",
        "source_repo_id": plan.get("repo_id"),
        "source_revision": plan.get("revision"),
        "source_hf_commit": source_commit,
        "source_download_bytes": plan.get("total_bytes"),
        "source_unique_videos": source_summary.get("unique_video_keys"),
        "source_vdc_overlap_entries": source_summary.get("vdc_overlap_entries"),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "split_ratios_by_video_and_duration_bucket": dict(SPLIT_RATIOS),
        "records": sample_counts,
        "unique_videos_by_split": {split: len(keys) for split, keys in split_video_ids.items()},
        "duration_bucket_counts": duration_counts,
        "task_counts": task_counts,
        "visual_budget_counts": budget_counts,
        "long_fraction": args.long_fraction,
        "caption_fraction_target": args.caption_fraction,
        "max_records_per_video": {
            "train": args.train_max_per_video,
            "validation": args.validation_max_per_video,
            "test": args.test_max_per_video,
        },
        "visual_budget_mixture": budgets,
        "spatial_tokens_per_temporal_patch": args.spatial_tokens_per_temporal_patch,
        "manifests": {split: f"manifests/{split}.jsonl" for split in sample_counts},
        "responses": {split: f"responses/{split}.jsonl" for split in sample_counts},
        "target_responses_generated": False,
        "target_hidden_features_cached": False,
        "full_visual_dflash_logits_cached": False,
        "full_visual_dflash_teacher_mode": "online_during_phase3_training",
        "visual_budget_note": "requested settings are verified against Qwen video_grid_thw after target response generation; source duration groups are not visual-token counts",
        "source_assistant_responses_used": False,
        "source_card_academic_research_only": True,
        "preparation_contract": preparation_contract,
    }
    _write_json(output_dir / "prepare_summary.json", result)
    _write_handoff(output_dir, result)
    return result


def _write_handoff(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DFlash 10% visual post-training data handoff",
        "",
        f"Source: `{summary['source_repo_id']}` at `{summary['source_revision']}`",
        f"HF snapshot commit: `{summary.get('source_hf_commit')}`",
        f"Prepared output: `{output_dir}`",
        "",
        "The upstream LLaVA-Video-178K dataset card limits use to academic research and education. "
        "This prepared copy does not broaden those terms; see https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K.",
        "",
        "## Stages",
        "",
        "1. Prepare and extract the split-specific media: `prepare_summary.json` and `manifests/*.jsonl`.",
        "2. Generate exact greedy target responses on the training server with the same Qwen2.5-VL processor configuration used by the experiment.",
        "3. Validate target-response coverage and actual `video_grid_thw` visual-token counts.",
        "4. Run the Phase 3 trainer on the training server. Target hidden states may be cached there; full-visual DFlash teacher distributions are intended to be computed online and are not serialized as full-vocabulary logits.",
        "",
        "## Target response command",
        "",
        "```bash",
        "python -m src.train_VLM.prepare_posttraining targets \\",
        f"  --data-dir '{output_dir}' \\",
        "  --config /path/to/qwen25vl3b_visual10_responses.json \\",
        "  --batch-size 1 --resume",
        "```",
        "",
        "## Final validation command",
        "",
        "```bash",
        f"python -m src.train_VLM.prepare_posttraining validate --data-dir '{output_dir}' --require-target-responses",
        "```",
        "",
        "The Qwen target response contains exact token IDs and processor provenance. The source GPT responses are not used as labels.",
        "",
    ]
    path = output_dir / "HANDOFF.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def generate_target_responses(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    summary_path = data_dir / "prepare_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"run prepare first; missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = DFlashTrainConfig.from_file(args.config)
    maximum_visual_target = max(int(row["target_visual_tokens"]) for row in summary["visual_budget_mixture"])
    minimum_sequence_length = maximum_visual_target + config.response_max_new_tokens + 1024
    if config.max_seq_length < minimum_sequence_length:
        raise ValueError(
            f"max_seq_length={config.max_seq_length} is below the largest requested visual context, "
            f"1024-token prompt reserve, and response budget ({maximum_visual_target}+1024+"
            f"{config.response_max_new_tokens}); "
            "increase it in the target config before loading the model"
        )
    config_contract = {
        "target_model": config.target_model,
        "target_revision": config.target_revision,
        "max_seq_length": config.max_seq_length,
        "response_max_new_tokens": config.response_max_new_tokens,
        "mixed_precision": config.mixed_precision,
        "video_reader": config.video_reader,
        "video_num_frames_default": config.video_num_frames,
        "video_min_pixels_default": config.video_min_pixels,
        "video_max_pixels_default": config.video_max_pixels,
        "processor_kwargs": config.processor_kwargs,
        "batch_size": args.batch_size,
    }
    output_paths = {
        split: data_dir / "responses" / f"{split}.jsonl"
        for split in args.splits
    }
    for split, output_path in output_paths.items():
        if output_path.exists() and not args.resume:
            raise FileExistsError(f"response file exists; pass --resume to continue: {output_path}")
        if output_path.exists() and args.resume:
            metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
            if not metadata_path.is_file():
                raise ValueError(f"cannot safely resume without response provenance: {metadata_path}")
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved_config = saved.get("config")
            can_extend_response = False
            if isinstance(saved_config, dict):
                saved_response_limit = saved_config.get("response_max_new_tokens")
                saved_other_settings = {
                    key: value for key, value in saved_config.items()
                    if key != "response_max_new_tokens"
                }
                current_other_settings = {
                    key: value for key, value in config_contract.items()
                    if key != "response_max_new_tokens"
                }
                can_extend_response = (
                    isinstance(saved_response_limit, int)
                    and config.response_max_new_tokens >= saved_response_limit
                    and saved_other_settings == current_other_settings
                )
            if saved_config != config_contract and not can_extend_response:
                raise ValueError(
                    f"target response config differs from saved output for {split}; "
                    "resume allows only an equal or larger response token limit with all other settings unchanged"
                )
    adapter = Qwen25VLTargetAdapter.from_pretrained(config, device=config.device)
    target_provenance = adapter.target_provenance()
    response_counts = {}
    for split in args.splits:
        manifest_path = data_dir / "manifests" / f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        records = load_jsonl(manifest_path)
        output_path = output_paths[split]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        if output_path.exists():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved.get("target_provenance") != target_provenance:
                raise ValueError(f"target model/processor fingerprint changed for {split}; use a new data directory")
        else:
            _write_json(metadata_path, {
                "status": "running",
                "config": config_contract,
                "target_provenance": target_provenance,
            })
        from .prepare_responses import prepare_responses_batched

        prepare_responses_batched(
            adapter,
            records,
            max_new_tokens=config.response_max_new_tokens,
            max_seq_length=config.max_seq_length,
            output_path=output_path,
            batch_size=args.batch_size,
            resume=args.resume,
        )
        generated = load_jsonl(output_path)
        input_ids = {str(row["id"]) for row in records}
        output_ids = {str(row["id"]) for row in generated if isinstance(row.get("target_response"), dict)}
        missing = sorted(input_ids - output_ids)
        extra = sorted(output_ids - input_ids)
        requested_input_count = len(input_ids)
        if extra:
            raise ValueError(f"{split} response output contains {len(extra)} IDs absent from its manifest")
        if missing:
            if not args.drop_incomplete_responses:
                sample = ", ".join(missing[:8])
                raise RuntimeError(
                    f"target responses incomplete for {split}: missing {len(missing)} records; examples: {sample}. "
                    "Inspect skipped length/EOS messages, adjust the target config, and resume."
                )
            missing_ids = set(missing)
            dropped_rows = [row for row in records if str(row["id"]) in missing_ids]
            records = [row for row in records if str(row["id"]) not in missing_ids]
            _write_jsonl(manifest_path, records)

            summary["records"][split] = len(records)
            summary["unique_videos_by_split"][split] = len({
                str(row["metadata"]["video_key"]) for row in records
            })
            summary["task_counts"][split] = dict(Counter(
                str(row["metadata"]["task"]) for row in records
            ))
            summary["duration_bucket_counts"][split] = dict(Counter(
                str(row["metadata"]["duration_bucket"]) for row in records
            ))
            summary["visual_budget_counts"][split] = dict(Counter(
                str(row["metadata"]["target_visual_tokens"]) for row in records
            ))
            _write_json(summary_path, summary)

            drop_path = data_dir / "incomplete_target_response_drops.jsonl"
            previous_drops = load_jsonl(drop_path) if drop_path.is_file() else []
            previous_keys = {
                (str(row.get("split")), str(row.get("id"))) for row in previous_drops
            }
            new_drops = [
                {
                    "split": split,
                    "id": str(row["id"]),
                    "reason": "no_complete_response_within_configured_limits",
                    "response_max_new_tokens": config.response_max_new_tokens,
                }
                for row in dropped_rows
                if (split, str(row["id"])) not in previous_keys
            ]
            _write_jsonl(drop_path, [*previous_drops, *new_drops])
            print(
                f"[targets] dropped {len(dropped_rows)} {split} records without EOS; "
                f"see {drop_path}",
                flush=True,
            )
            input_ids = {str(row["id"]) for row in records}
            missing = []

        response_counts[split] = {
            "input": requested_input_count,
            "generated": len(output_ids),
            "missing": len(missing),
            "dropped": requested_input_count - len(input_ids),
        }
        missing_grids = [row["id"] for row in generated if _actual_visual_tokens(row) is None]
        if missing_grids:
            raise RuntimeError(
                f"Qwen did not return video_grid_thw for {len(missing_grids)} {split} records; "
                "check the video processor and single-video-per-record batching"
            )
        provenance_mismatches = [
            row["id"] for row in generated if row.get("provenance") != target_provenance
        ]
        if provenance_mismatches:
            raise ValueError(f"{split} responses contain mixed target provenance")
        _write_json(metadata_path, {
            "status": "complete",
            "config": config_contract,
            "target_provenance": target_provenance,
            "record_count": len(generated),
        })
        print(f"[targets] {split}: {len(output_ids)}/{len(input_ids)} exact target responses", flush=True)
    result = {
        "status": "complete",
        "target_model": config.target_model,
        "target_revision": config.target_revision,
        "max_seq_length": config.max_seq_length,
        "response_max_new_tokens": config.response_max_new_tokens,
        "dtype": config.mixed_precision,
        "target_provenance": target_provenance,
        "splits": response_counts,
        "target_hidden_features_cached": False,
        "full_visual_dflash_logits_cached": False,
    }
    _write_json(data_dir / "target_generation_summary.json", result)
    return result


def _video_path_from_record(record: dict[str, Any]) -> Path | None:
    for message in record.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "video":
                value = str(item.get("video", ""))
                parsed = urlparse(value)
                if parsed.scheme == "file":
                    return Path(unquote(parsed.path))
                return Path(value).expanduser()
    return None


def _actual_visual_tokens(record: dict[str, Any]) -> int | None:
    metadata = record.get("video_preprocessing")
    grids = metadata.get("video_grid_thw") if isinstance(metadata, dict) else None
    if not isinstance(grids, list):
        return None
    total = 0
    for grid in grids:
        if not isinstance(grid, (list, tuple)) or len(grid) < 3:
            continue
        temporal, height, width = (int(value) for value in grid[-3:])
        if min(temporal, height, width) <= 0 or height % 2 or width % 2:
            raise ValueError(f"invalid Qwen video_grid_thw row: {grid}")
        total += temporal * (height // 2) * (width // 2)
    return total or None


def validate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    summary_path = data_dir / "prepare_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    prepared_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    video_split: dict[str, str] = {}
    seen_ids: set[str] = set()
    counts: dict[str, Any] = {}
    realized_budgets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ("train", "validation", "test"):
        manifest_path = data_dir / "manifests" / f"{split}.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        records = load_jsonl(manifest_path)
        records_by_split[split] = records
        task_counts: Counter[str] = Counter()
        duration_counts: Counter[str] = Counter()
        visual_targets: Counter[str] = Counter()
        response_records: list[dict[str, Any]] = []
        response_path = data_dir / "responses" / f"{split}.jsonl"
        if args.require_target_responses:
            if not response_path.is_file():
                raise FileNotFoundError(response_path)
            response_records = load_jsonl(response_path)
        response_by_id = {str(row.get("id")): row for row in response_records}
        if len(response_by_id) != len(response_records):
            raise ValueError(f"duplicate response IDs in {response_path}")
        for record in records:
            validate_manifest_record(record, require_target_response=False)
            record_id = str(record["id"])
            if record_id in seen_ids:
                raise ValueError(f"sample ID appears more than once across splits: {record_id}")
            seen_ids.add(record_id)
            metadata = record.get("metadata", {})
            video_key = str(metadata.get("video_key", ""))
            if not video_key:
                raise ValueError(f"record {record_id} has no video_key metadata")
            previous_split = video_split.setdefault(video_key, split)
            if previous_split != split:
                raise ValueError(f"video {video_key} leaks across {previous_split} and {split}")
            video_path = _video_path_from_record(record)
            if video_path is None or not video_path.is_file():
                raise FileNotFoundError(f"record {record_id} references missing video media: {video_path}")
            task_counts[str(metadata.get("task"))] += 1
            duration_counts[str(metadata.get("duration_bucket"))] += 1
            visual_target = int(metadata.get("target_visual_tokens", 0))
            visual_targets[str(visual_target)] += 1
            if args.require_target_responses:
                response = response_by_id.get(record_id)
                if response is None:
                    raise ValueError(f"target response missing for {record_id}")
                validate_manifest_record(response, require_target_response=True)
                if not isinstance(response.get("provenance"), dict):
                    raise ValueError(f"target response lacks model provenance: {record_id}")
                actual = _actual_visual_tokens(response)
                if actual is None:
                    raise ValueError(f"target response lacks Qwen video_grid_thw metadata: {record_id}")
                if actual is not None:
                    relative_error = abs(actual - visual_target) / visual_target if visual_target else None
                    realized_budgets[split].append({
                        "id": record_id,
                        "target_visual_tokens": visual_target,
                        "actual_visual_tokens": actual,
                        "relative_error": relative_error,
                    })
        expected = int(prepared_summary["records"][split])
        if len(records) != expected:
            raise ValueError(f"{split} count changed: got {len(records)}, expected {expected}")
        if args.require_target_responses and set(response_by_id) != {str(row["id"]) for row in records}:
            raise ValueError(f"target response IDs differ from the {split} manifest")
        counts[split] = {
            "records": len(records),
            "unique_videos": len({str(row["metadata"]["video_key"]) for row in records}),
            "task_counts": dict(task_counts),
            "duration_counts": dict(duration_counts),
            "requested_visual_budget_counts": dict(visual_targets),
            "target_responses": len(response_records) if args.require_target_responses else None,
        }

    for split, rows in realized_budgets.items():
        if rows:
            errors = [row["relative_error"] for row in rows if row["relative_error"] is not None]
            counts[split]["realized_visual_budget"] = {
                "records_with_processor_grid": len(rows),
                "mean_relative_error": sum(errors) / len(errors) if errors else None,
                "within_10_percent": sum(error <= 0.10 for error in errors),
                "outside_10_percent": sum(error > 0.10 for error in errors),
            }
    result = {
        "status": "valid",
        "data_dir": str(data_dir),
        "video_split_overlap": 0,
        "target_responses_required": bool(args.require_target_responses),
        "splits": counts,
    }
    _write_json(data_dir / "validation_report.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", default=DEFAULT_HF_REPO,
                        help="local raw-data directory or HF dataset repo ID (default: %(default)s)")
    parser.add_argument("--source-revision", default=None,
                        help="HF commit/revision; defaults to the current repo commit")
    parser.add_argument("--output-dir", required=True, help="new processed-data output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-records", type=int, default=3000)
    parser.add_argument("--validation-records", type=int, default=250)
    parser.add_argument("--test-records", type=int, default=250)
    parser.add_argument("--long-fraction", type=float, default=0.75,
                        help="sample fraction from 1-3 minute source groups; not a visual-token fraction")
    parser.add_argument("--caption-fraction", type=float, default=0.30,
                        help="caption share within each duration bucket; remaining prompts are open-ended QA")
    parser.add_argument("--train-max-per-video", type=int, default=2)
    parser.add_argument("--validation-max-per-video", type=int, default=1)
    parser.add_argument("--test-max-per-video", type=int, default=1)
    parser.add_argument("--visual-budget-mixture", default="3000:0.25:12,13000:0.375:50,25000:0.375:98",
                        help="comma-separated visual_tokens:fraction:nframes entries")
    parser.add_argument(
        "--spatial-tokens-per-temporal-patch", "--patches-per-frame",
        dest="spatial_tokens_per_temporal_patch", type=int, default=512,
        help="spatially merged visual tokens per Qwen temporal grid position (default: %(default)s)",
    )
    parser.add_argument("--resume", action="store_true",
                        help="reuse complete extracted media and rewrite deterministic manifests")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="download/verify raw source and build splits/manifests")
    _add_prepare_arguments(prepare_parser)

    targets_parser = subparsers.add_parser("targets", help="generate exact target responses on the training server")
    targets_parser.add_argument("--data-dir", required=True)
    targets_parser.add_argument("--config", required=True, help="DFlashTrainConfig JSON/YAML for the full-visual target")
    targets_parser.add_argument("--batch-size", type=int, default=1)
    targets_parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                                 default=["train", "validation", "test"])
    targets_parser.add_argument("--resume", action="store_true")
    targets_parser.add_argument(
        "--drop-incomplete-responses",
        action="store_true",
        help="remove manifest rows whose greedy target response reaches its token limit without EOS",
    )

    validate_parser = subparsers.add_parser("validate", help="validate data manifests and optionally generated targets")
    validate_parser.add_argument("--data-dir", required=True)
    validate_parser.add_argument("--require-target-responses", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        _parse_budget_mixture(args.visual_budget_mixture)
        if args.spatial_tokens_per_temporal_patch < 1:
            parser.error("--spatial-tokens-per-temporal-patch must be positive")
        result = prepare_dataset(args)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    elif args.command == "targets":
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        result = generate_target_responses(args)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    else:
        validate_dataset(args)


if __name__ == "__main__":
    main()
