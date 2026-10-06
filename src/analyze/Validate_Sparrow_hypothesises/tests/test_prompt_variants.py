from types import SimpleNamespace

import json

from src.analyze.Validate_Sparrow_hypothesises.dataset import (
    build_prompt_question,
    build_vdc_mcq_prompt,
    load_vdc_manifest,
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


def test_vdc_mcq_prompt_matches_mvbench_template():
    row = {
        "question": "Which description best matches the video?",
        "candidates": ["first", "second", "third", "fourth"],
    }

    assert build_vdc_mcq_prompt(row) == (
        "Question:Which description best matches the video?\n"
        "Option:\n"
        "(A) first\n"
        "(B) second\n"
        "(C) third\n"
        "(D) fourth\n"
        "Only give the best option.\n"
    )


def test_vdc_mcq_manifest_preserves_option_metadata(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"placeholder")
    manifest = tmp_path / "vdc_mcq.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_name": "v0",
                "question": "Which description best matches the video?",
                "answer": "first",
                "reference_answer": "first",
                "candidates": ["first", "second", "third", "fourth"],
                "correct_option": "A",
                "local_video_path": str(video),
                "dataset_format": "vdc_mcq",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_vdc_manifest(manifest, tmp_path)

    assert samples[0].candidates == ("first", "second", "third", "fourth")
    assert samples[0].correct_option == "A"
    assert build_prompt_question(samples[0], "mcq").startswith("Question:")


def test_vdc_mcq_prompt_supports_class_attribute_sample_proxy():
    sample = type(
        "SampleProxy",
        (),
        {
            "question": "Which description best matches the video?",
            "candidates": ("first", "second", "third", "fourth"),
        },
    )()

    assert "(A) first" in build_prompt_question(sample, "mcq")


def test_vdc_mcq2_keeps_options_and_requires_a_full_answer():
    sample = type(
        "SampleProxy",
        (),
        {
            "question": "Which description best matches the video?",
            "candidates": ("first", "second", "third", "fourth"),
            "answer": "oracle answer must not be included",
        },
    )()

    prompt = build_prompt_question(sample, "mcq2")

    assert prompt == (
        "Question:Which description best matches the video?\n"
        "Option:\n"
        "(A) first\n"
        "(B) second\n"
        "(C) third\n"
        "(D) fourth\n"
        "Begin with the selected option in the form (A), (B), (C), or (D). "
        "Then answer the question directly in 1–2 complete sentences using "
        "concise details visible in the video. Do not copy or restate an entire option, "
        "and do not output only the option label.\n"
    )
    assert "oracle answer must not be included" not in prompt
    assert build_prompt_question(sample, "mcq") == (
        "Question:Which description best matches the video?\n"
        "Option:\n"
        "(A) first\n(B) second\n(C) third\n(D) fourth\n"
        "Only give the best option.\n"
    )
