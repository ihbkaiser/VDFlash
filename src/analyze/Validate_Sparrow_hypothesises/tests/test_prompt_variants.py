from types import SimpleNamespace

import json

from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    build_prompt_question,
    load_mvbench_manifest,
)


def test_answer_hint_prompt_is_explicit_and_natural_prompt_is_unchanged():
    sample = SimpleNamespace(question="What happens?", answer="A person waves.")

    assert build_prompt_question(sample, "natural") == "What happens?"
    augmented = build_prompt_question(sample, "answer_hint")
    assert augmented.startswith("What happens?")
    assert "Reference information (oracle hint): A person waves." in augmented


def test_mvbench_loader_builds_canonical_prompt_and_checks_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"placeholder")
    manifest = tmp_path / "selected.jsonl"
    manifest.write_text(json.dumps({
        "task": "action_prediction",
        "sample_id": "action_prediction:000000",
        "question": "What happens?",
        "candidates": ["A", "B"],
        "answer": "A",
        "video_path": str(video),
    }) + "\n", encoding="utf-8")

    samples = load_mvbench_manifest(manifest, tmp_path, limit_per_task=1)

    assert len(samples) == 1
    assert samples[0].sample_id == "action_prediction:000000"
    assert samples[0].task == "action_prediction"
    assert "(A) A" in samples[0].question
    assert "Only give the best option." in samples[0].question
