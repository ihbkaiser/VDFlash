"""Pure data, prompt, token-scoring, and paired-statistics contracts.

This module is deliberately independent of torch and Transformers.  The GPU
runner uses these contracts, while the unit tests can validate the experiment
semantics without downloading a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
INPUT_ORDERS = ("video_text", "text_video")
CONDITIONING = ("full", "visual", "text")
EXPERIMENTS = {
    "E1": ("video_text", "full"),
    "E2": ("text_video", "full"),
    "E3": ("video_text", "visual"),
    "E4": ("text_video", "visual"),
    "E5": ("video_text", "text"),
    "E6": ("text_video", "text"),
}


@dataclass(frozen=True)
class ReverseTransferSample:
    sample_id: str
    video_id: str
    video_path: str
    question: str
    options: tuple[str, ...]
    correct_answer: str
    task_type: str
    dataset: str
    source_index: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.sample_id or not self.video_id:
            raise ValueError("sample_id and video_id must be non-empty")
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not 2 <= len(self.options) <= len(LABELS):
            raise ValueError("options must contain between two and 26 choices")
        if len(set(self.options)) != len(self.options):
            raise ValueError("options must be unique")
        if self.correct_answer not in self.options:
            raise ValueError("correct_answer must equal one option content")

    @property
    def labels(self) -> tuple[str, ...]:
        return LABELS[: len(self.options)]

    @property
    def correct_label(self) -> str:
        return self.labels[self.options.index(self.correct_answer)]

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def with_options(self, options: Sequence[str]) -> "ReverseTransferSample":
        values = tuple(str(value).strip() for value in options)
        if sorted(values) != sorted(self.options):
            raise ValueError("option permutation must preserve option contents")
        return replace(self, options=values)


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    input_order: str
    conditioning: str

    def __post_init__(self) -> None:
        if self.input_order not in INPUT_ORDERS:
            raise ValueError(f"unsupported input order: {self.input_order}")
        if self.conditioning not in CONDITIONING:
            raise ValueError(f"unsupported MSD conditioning: {self.conditioning}")


@dataclass(frozen=True)
class CandidateTokens:
    label: str
    token_id: int
    raw_token_ids: tuple[int, ...]
    spaced_token_ids: tuple[int, ...]


def condition_spec(condition: str) -> ConditionSpec:
    try:
        order, conditioning = EXPERIMENTS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown experiment condition {condition!r}") from exc
    return ConditionSpec(condition, order, conditioning)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _resolve_video_path(row: Mapping[str, Any], video_root: str | Path | None) -> Path:
    root = Path(video_root) if video_root is not None else None
    candidates: list[Path] = []
    for key in ("video_path", "local_video_path", "path", "video_file"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            candidates.append(path if path.is_absolute() or root is None else root / path)
    for key in ("video_name", "video", "video_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and root is not None:
            raw = Path(value)
            candidates.append(root / raw)
            if not raw.suffix:
                candidates.extend(root / raw.with_suffix(ext) for ext in (".mp4", ".mkv", ".mov", ".avi"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    shown = ", ".join(str(value) for value in candidates[:8]) or "<no video field>"
    raise FileNotFoundError(f"video not found; tried: {shown}")


def _answer_content(row: Mapping[str, Any], options: Sequence[str], dataset: str) -> str:
    answer = row.get(
        "correct_answer",
        row.get("answer", row.get("label_name", row.get("label"))),
    )
    if answer is None:
        raise ValueError("record has no correct_answer/answer field")
    text = str(answer).strip()
    if text in options:
        return text
    upper = text.upper().rstrip(".")
    if upper in LABELS[: len(options)]:
        return options[LABELS.index(upper)]
    if dataset in {"vidhalluc", "vdc", "videohallucination"}:
        lowered = text.lower()
        if lowered in {"supported", "true", "1"} and len(options) >= 2:
            return options[0]
        if lowered in {"not supported", "unsupported", "false", "0"} and len(options) >= 2:
            return options[1]
        if lowered in {"yes", "true", "1"}:
            return options[0]
        if lowered in {"no", "false", "0"}:
            return options[1]
    raise ValueError(f"answer {answer!r} does not match option contents or labels")


def load_reverse_manifest(
    path: str | Path,
    *,
    dataset: str,
    video_root: str | Path | None = None,
    num_samples: int | None = None,
    require_videos: bool = True,
) -> list[ReverseTransferSample]:
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive")
    dataset_name = dataset.lower()
    samples: list[ReverseTransferSample] = []
    seen: set[str] = set()
    for index, row in enumerate(_read_jsonl(path)):
        if "_manifest" in row:
            continue
        sample_id = str(row.get("sample_id", row.get("id", f"{dataset}:{index}")))
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        video_id = str(row.get("video_id", row.get("video_name", row.get("video", sample_id))))
        question = row.get("question", row.get("prompt"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{index + 1}: missing question")
        raw_options = row.get("options", row.get("candidates"))
        if raw_options is None:
            if dataset_name in {"vidhalluc", "vdc", "videohallucination"}:
                if row.get("label_name") is not None:
                    raw_options = ["Supported", "Not Supported"]
                else:
                    raw_options = ["Yes", "No"]
            else:
                raise ValueError(f"{path}:{index + 1}: missing options/candidates")
        options = tuple(str(value).strip() for value in raw_options)
        correct_answer = _answer_content(row, options, dataset_name)
        if require_videos:
            video_path = _resolve_video_path(row, video_root)
        else:
            value = row.get("video_path", row.get("local_video_path", row.get("video", video_id)))
            raw_path = Path(str(value))
            video_path = raw_path if raw_path.is_absolute() or video_root is None else Path(video_root) / raw_path
        metadata = tuple(sorted((str(key), str(value)) for key, value in row.items()))
        samples.append(ReverseTransferSample(
            sample_id=sample_id,
            video_id=video_id,
            video_path=str(video_path),
            question=question,
            options=options,
            correct_answer=correct_answer,
            task_type=str(row.get("task_type", row.get("task", row.get("category", dataset_name)))),
            dataset=dataset_name,
            source_index=index,
            metadata=metadata,
        ))
        seen.add(sample_id)
        if num_samples is not None and len(samples) >= num_samples:
            break
    if not samples:
        raise ValueError(f"manifest is empty: {path}")
    return samples


def canonical_task_text(question: str, options: Sequence[str]) -> str:
    lines = [f"Question: {question.strip()}", "", "Options:"]
    lines.extend(f"{label}. {value}" for label, value in zip(LABELS, options))
    lines.extend(("", "Answer with only the correct option letter."))
    return "\n".join(lines)


def build_user_message(
    sample: ReverseTransferSample,
    *,
    input_order: str,
    fps: float = 2.0,
    num_frames: int | None = 8,
    max_pixels: int | None = None,
    include_video: bool = True,
    include_text: bool = True,
) -> list[dict[str, Any]]:
    if input_order not in INPUT_ORDERS:
        raise ValueError(f"unsupported input order: {input_order}")
    if not include_video and not include_text:
        raise ValueError("at least one modality must be included")
    text = {"type": "text", "text": canonical_task_text(sample.question, sample.options)}
    video: dict[str, Any] = {"type": "video", "video": str(sample.video_path)}
    if num_frames is not None:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        video["nframes"] = int(num_frames)
    else:
        video["fps"] = float(fps)
    if max_pixels is not None:
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        video["min_pixels"] = int(max_pixels)
        video["max_pixels"] = int(max_pixels)
    if include_video and include_text:
        content = [video, text] if input_order == "video_text" else [text, video]
    elif include_video:
        content = [video]
    else:
        content = [text]
    return [{"role": "user", "content": content}]


def tokenize_candidates(tokenizer: Any, labels: Sequence[str]) -> tuple[CandidateTokens, ...]:
    result: list[CandidateTokens] = []
    for label in labels:
        if len(label) != 1 or label not in LABELS:
            raise ValueError(f"candidate labels must be single letters, got {label!r}")
        raw = tuple(int(value) for value in tokenizer(label, add_special_tokens=False)["input_ids"])
        spaced = tuple(int(value) for value in tokenizer(f" {label}", add_special_tokens=False)["input_ids"])
        if len(raw) != 1:
            raise ValueError(f"candidate {label!r} is not one token: {raw}")
        result.append(CandidateTokens(label, raw[0], raw, spaced))
    return tuple(result)


def restricted_probabilities(log_scores: Mapping[str, float]) -> dict[str, float]:
    if not log_scores:
        raise ValueError("at least one candidate score is required")
    values = {str(key): float(value) for key, value in log_scores.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("candidate log scores must be finite")
    maximum = max(values.values())
    weights = {key: math.exp(value - maximum) for key, value in values.items()}
    denominator = sum(weights.values())
    return {key: value / denominator for key, value in weights.items()}


def answer_metrics(log_scores: Mapping[str, float], correct_label: str) -> dict[str, Any]:
    probabilities = restricted_probabilities(log_scores)
    if correct_label not in probabilities:
        raise ValueError(f"correct label {correct_label!r} is not in candidates")
    predicted = max(probabilities, key=probabilities.get)
    incorrect = [value for key, value in log_scores.items() if key != correct_label]
    return {
        "predicted_answer": predicted,
        "correct_answer": correct_label,
        "accuracy": float(predicted == correct_label),
        "correct_option_probability": float(probabilities[correct_label]),
        "nll": float(-math.log(max(probabilities[correct_label], 1e-30))),
        "margin": float(log_scores[correct_label] - max(incorrect)) if incorrect else None,
        "prediction_entropy": float(-sum(value * math.log(max(value, 1e-30)) for value in probabilities.values())),
        "candidate_logits": {key: float(value) for key, value in log_scores.items()},
        "candidate_probabilities": probabilities,
    }


def deterministic_permutations(sample: ReverseTransferSample, seed: int = 0) -> list[tuple[str, ReverseTransferSample]]:
    identity = ("identity", sample)
    indices = list(range(len(sample.options)))
    rng = random.Random(f"{seed}:{sample.sample_id}")
    permutation = indices[:]
    rng.shuffle(permutation)
    if permutation == indices and len(permutation) > 1:
        permutation[0], permutation[1] = permutation[1], permutation[0]
    permuted = sample.with_options([sample.options[index] for index in permutation])
    return [identity, ("perm-" + "-".join(map(str, permutation)), permuted)]


def sample_to_json(sample: ReverseTransferSample) -> dict[str, Any]:
    value = asdict(sample)
    value["options"] = list(sample.options)
    value["metadata"] = dict(sample.metadata)
    value["fingerprint"] = sample.fingerprint()
    value["correct_label"] = sample.correct_label
    return value


def paired_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    condition_a: str,
    condition_b: str,
    group_key: str = "video_id",
    repeats: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        group = str(row.get(group_key, row.get("sample_id")))
        condition = str(row.get("condition"))
        value = row.get(value_key)
        if condition in {condition_a, condition_b} and value is not None:
            grouped.setdefault(group, {})[condition] = float(value)
    deltas = [values[condition_a] - values[condition_b] for values in grouped.values() if condition_a in values and condition_b in values]
    if not deltas:
        return {"n_groups": 0, "estimate": None, "ci95": [None, None]}
    rng = random.Random(seed)
    boot = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(max(1, repeats))]
    boot.sort()
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    return {"n_groups": len(deltas), "estimate": sum(deltas) / len(deltas), "ci95": [low, high]}

