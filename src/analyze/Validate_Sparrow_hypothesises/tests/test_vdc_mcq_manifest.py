from __future__ import annotations

import json

from src.analyze.Validate_Sparrow_hypothesises.build_vdc_mcq_manifest import (
    build_vdc_mcq_rows,
)
from src.analyze.Validate_Sparrow_hypothesises.dataset import load_vdc_manifest


def test_vdc_mcq_builder_creates_four_options_and_auditable_answer():
    rows = [
        {"video_name": "v0", "answer": "A person chops wood near a truck."},
        {"video_name": "v1", "answer": "A rider jumps over a ramp on a bike."},
        {"video_name": "v2", "answer": "A swimmer throws a ball in a pool."},
        {"video_name": "v3", "answer": "A boxer kicks an opponent in a ring."},
    ]

    result = build_vdc_mcq_rows(rows, seed=7)

    assert len(result) == 4
    for row in result:
        assert row["dataset_format"] == "vdc_mcq"
        assert len(row["candidates"]) == 4
        assert row["candidates"][ord(row["correct_option"]) - ord("A")] == row["answer"]
        assert row["automatic_audit"]["exactly_one_answer"] is True
        assert len(row["distractor_provenance"]) == 3


def test_vdc_mcq_builder_is_deterministic_for_same_seed():
    rows = [
        {"video_name": "v0", "answer": "A person chops wood near a truck."},
        {"video_name": "v1", "answer": "A rider jumps over a ramp on a bike."},
        {"video_name": "v2", "answer": "A swimmer throws a ball in a pool."},
        {"video_name": "v3", "answer": "A boxer kicks an opponent in a ring."},
    ]

    assert build_vdc_mcq_rows(rows, seed=7) == build_vdc_mcq_rows(rows, seed=7)


def test_vdc_manifest_loader_preserves_option_provenance(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"stub")
    option_provenance = [
        {"option": "A", "is_correct": False, "sample_id": "other", "source_index": 1},
        {"option": "B", "is_correct": True, "sample_id": "sample", "source_index": 0},
    ]
    distractor_provenance = [option_provenance[0]]
    manifest_row = {
        "video_name": "sample",
        "question": "Which description matches?",
        "answer": "A person waves.",
        "reference_answer": "A person waves.",
        "local_video_path": video.name,
        "candidates": ["A person runs.", "A person waves."],
        "correct_option": "B",
        "option_provenance": option_provenance,
        "distractor_provenance": distractor_provenance,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(manifest_row), encoding="utf-8")

    sample = load_vdc_manifest(manifest, tmp_path)[0]

    assert sample.option_provenance == tuple(option_provenance)
    assert sample.distractor_provenance == tuple(distractor_provenance)
