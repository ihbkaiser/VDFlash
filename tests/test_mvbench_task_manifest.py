import json
from pathlib import Path

from src.dataset.mvbench_task_manifest import (
    TASK_VIDEO_ROOTS,
    build_task_records,
    write_task_manifests,
)


def test_build_task_records_adds_task_and_resolves_video_root(tmp_path: Path):
    annotation = {
        "video": "clip.mp4",
        "question": "What happened?",
        "candidates": ["A", "B"],
        "answer": "A",
    }

    records = build_task_records(
        "action_prediction",
        [annotation],
        dataset_root=tmp_path,
    )

    assert TASK_VIDEO_ROOTS["action_prediction"] == "star/Charades_segment"
    assert records == [
        {
            **annotation,
            "task": "action_prediction",
            "sample_id": "action_prediction:000000",
            "video_root": "star/Charades_segment",
            "video_relpath": "star/Charades_segment/clip.mp4",
            "video_path": str(tmp_path / "star/Charades_segment/clip.mp4"),
            "video_exists": False,
        }
    ]


def test_write_task_manifests_creates_per_task_and_combined_outputs(tmp_path: Path):
    annotation_dir = tmp_path / "json"
    output_dir = tmp_path / "classified"
    annotation_dir.mkdir()
    (annotation_dir / "action_prediction.json").write_text(
        json.dumps([{"video": "a.mp4", "question": "q", "candidates": ["x"], "answer": "x"}])
    )
    (annotation_dir / "moving_direction.json").write_text(
        json.dumps([{"video": "b.mp4", "question": "q", "candidates": ["x"], "answer": "x"}])
    )

    result = write_task_manifests(
        annotation_dir=annotation_dir,
        output_dir=output_dir,
        dataset_root=tmp_path,
        tasks=("action_prediction", "moving_direction"),
    )

    assert result == {"action_prediction": 1, "moving_direction": 1}
    action_rows = [json.loads(line) for line in (output_dir / "action_prediction.jsonl").read_text().splitlines()]
    direction_rows = [json.loads(line) for line in (output_dir / "moving_direction.jsonl").read_text().splitlines()]
    combined_rows = [json.loads(line) for line in (output_dir / "selected.jsonl").read_text().splitlines()]

    assert action_rows[0]["video_root"] == "star/Charades_segment"
    assert direction_rows[0]["video_root"] == "clevrer/video_validation"
    assert [row["task"] for row in combined_rows] == ["action_prediction", "moving_direction"]
