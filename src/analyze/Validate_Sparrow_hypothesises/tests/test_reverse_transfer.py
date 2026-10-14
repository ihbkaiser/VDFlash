from __future__ import annotations

import json
import math

import pytest

from src.analyze.Validate_Sparrow_hypothesises.reverse_transfer import (
    EXPERIMENTS,
    ReverseTransferSample,
    answer_metrics,
    build_user_message,
    canonical_task_text,
    condition_spec,
    deterministic_permutations,
    load_reverse_manifest,
    paired_bootstrap_delta,
    restricted_probabilities,
    tokenize_candidates,
)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def sample() -> ReverseTransferSample:
    return ReverseTransferSample(
        sample_id="mvbench:0",
        video_id="clip-0",
        video_path="/data/clip.mp4",
        question="What happens?",
        options=("waves", "sits", "runs", "nothing"),
        correct_answer="sits",
        task_type="moving_attribute",
        dataset="mvbench",
    )


def test_prompt_order_is_explicit_and_text_is_identical():
    value = sample()
    assert canonical_task_text(value.question, value.options).endswith("Answer with only the correct option letter.")
    vt = build_user_message(value, input_order="video_text")
    tv = build_user_message(value, input_order="text_video")
    assert [item["type"] for item in vt[0]["content"]] == ["video", "text"]
    assert [item["type"] for item in tv[0]["content"]] == ["text", "video"]
    assert vt[0]["content"][1]["text"] == tv[0]["content"][0]["text"]


def test_e1_e6_matrix_is_exact():
    assert set(EXPERIMENTS) == {"E1", "E2", "E3", "E4", "E5", "E6"}
    assert condition_spec("E2").input_order == "text_video"
    assert condition_spec("E4").conditioning == "visual"
    assert condition_spec("E6").conditioning == "text"
    with pytest.raises(ValueError):
        condition_spec("C1")


def test_candidate_tokenization_rejects_multitoken_labels():
    candidates = tokenize_candidates(TinyTokenizer(), ("A", "B"))
    assert candidates[0].token_id != candidates[1].token_id
    assert candidates[0].spaced_token_ids
    class BadTokenizer:
        def __call__(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return {"input_ids": [1, 2]}
    with pytest.raises(ValueError):
        tokenize_candidates(BadTokenizer(), ("A",))


def test_answer_metrics_are_restricted_to_valid_labels():
    probabilities = restricted_probabilities({"A": 0.0, "B": -math.log(3)})
    assert probabilities["A"] == pytest.approx(0.75)
    metrics = answer_metrics({"A": 0.0, "B": -math.log(3)}, "A")
    assert metrics["predicted_answer"] == "A"
    assert metrics["correct_option_probability"] == pytest.approx(0.75)


def test_permutation_changes_label_but_preserves_answer_content():
    original = sample()
    variants = deterministic_permutations(original, seed=11)
    assert variants[0][0] == "identity"
    _, permuted = variants[1]
    assert sorted(permuted.options) == sorted(original.options)
    assert permuted.correct_answer == original.correct_answer
    assert permuted.correct_label != original.correct_label


def test_grouped_paired_bootstrap():
    rows = [
        {"video_id": "v0", "condition": "E4", "accuracy": 1},
        {"video_id": "v0", "condition": "E3", "accuracy": 0},
        {"video_id": "v1", "condition": "E4", "accuracy": 0},
        {"video_id": "v1", "condition": "E3", "accuracy": 0},
    ]
    result = paired_bootstrap_delta(rows, value_key="accuracy", condition_a="E4", condition_b="E3", repeats=50)
    assert result["n_groups"] == 2
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci95"][0] <= result["estimate"] <= result["ci95"][1]


def test_manifest_loader_maps_mvbench_and_vidhalluc(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({
        "sample_id": "mv-0", "video_id": "clip", "video_path": str(video),
        "question": "Which?", "candidates": ["left", "right"], "answer": "right",
    }) + "\n", encoding="utf-8")
    mv = load_reverse_manifest(path, dataset="mvbench")
    assert mv[0].correct_label == "B"
    vdc = tmp_path / "vdc.jsonl"
    vdc.write_text(json.dumps({
        "sample_id": "vdc-0", "video_id": "clip", "video_path": str(video),
        "question": "Is this true?", "answer": "No",
    }) + "\n", encoding="utf-8")
    assert load_reverse_manifest(vdc, dataset="vidhalluc")[0].correct_label == "B"
    vdc_supported = tmp_path / "vdc_supported.jsonl"
    vdc_supported.write_text(json.dumps({
        "sample_id": "vdc-supported", "video_id": "clip", "video_path": str(video),
        "question": "Is this supported?", "label": 1, "label_name": "Supported",
    }) + "\n", encoding="utf-8")
    supported = load_reverse_manifest(vdc_supported, dataset="vidhalluc")[0]
    assert supported.options == ("Supported", "Not Supported")
    assert supported.correct_answer == "Supported"
    assert supported.correct_label == "A"

    with_header = tmp_path / "vdc_with_header.jsonl"
    with_header.write_text(
        json.dumps({"_manifest": {"count": 1}}) + "\n" +
        json.dumps({
            "sample_id": "vdc-header", "video_id": "clip", "video_path": str(video),
            "question": "Is this supported?", "label": 0, "label_name": "Not Supported",
        }) + "\n",
        encoding="utf-8",
    )
    assert load_reverse_manifest(with_header, dataset="vidhalluc")[0].correct_label == "B"
