from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analyze.Validate_Sparrow_hypothesises import calibrate_visualcut_vdc50 as calibration
from src.analyze.Validate_Sparrow_hypothesises.calibrate import VideoCandidate
from src.analyze.Validate_Sparrow_hypothesises.runtime import build_video_messages


def test_build_video_messages_supports_matched_nframes_and_pixel_bounds():
    messages = build_video_messages(
        "/tmp/video.mp4",
        "describe it",
        nframes=12,
        min_pixels=512 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    video = messages[0]["content"][0]
    assert video["nframes"] == 12
    assert "fps" not in video
    assert video["min_pixels"] == 512 * 28 * 28
    assert video["max_pixels"] == 512 * 28 * 28


def test_calibration_measure_matches_inference_video_settings(monkeypatch, tmp_path):
    observed = {}

    def fake_process_video(processor, video_path, question, **kwargs):
        observed.update(kwargs)
        return {"video_grid_thw": torch.tensor([[12, 8, 8]])}

    monkeypatch.setattr(calibration, "_WORKER_PROCESSOR", object())
    monkeypatch.setattr(calibration, "_WORKER_ROOT", tmp_path)
    monkeypatch.setattr(calibration, "process_video", fake_process_video)
    sample = SimpleNamespace(
        sample_id="sample-1",
        question="describe it",
        resolved_path=lambda root: root / "video.mp4",
    )
    candidate = VideoCandidate(frames=12, max_pixels=512 * 28 * 28)

    actual = calibration._measure(sample, candidate)

    assert actual == 12 * 4 * 4
    assert observed == {
        "nframes": 12,
        "min_pixels": 512 * 28 * 28,
        "max_pixels": 512 * 28 * 28,
    }


def test_adaptive_calibration_keeps_candidate_builder_callable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "INITIAL_SETTINGS",
        {400: (4, 512), 3000: (12, 1024), 13000: (52, 1024), 25000: (100, 1024)},
    )
    monkeypatch.setattr(calibration, "_WORKER_PROCESSOR", object())
    monkeypatch.setattr(calibration, "_WORKER_ROOT", tmp_path)
    monkeypatch.setattr(calibration, "_WORKER_OUTPUT", tmp_path)
    monkeypatch.setattr(
        calibration,
        "_measure",
        lambda sample, candidate: candidate.frames * candidate.max_pixels // (28 * 28),
    )
    sample = SimpleNamespace(
        sample_id="sample-adaptive",
        question="describe it",
        total_frames=240,
        resolved_path=lambda root: root / "video.mp4",
        fingerprint=lambda: "stable-fingerprint",
    )

    rows = calibration._calibrate_sample(sample)

    assert [row["target_visual_tokens"] for row in rows] == [400, 3000, 13000, 25000]
    assert all(row["status"] == "ok" for row in rows)
    assert any(len(row["candidate_measurements"]) > 4 for row in rows)


def test_calibration_seed_settings_start_within_ten_percent_of_each_target():
    for target, (frames, pixel_patches) in calibration.INITIAL_SETTINGS.items():
        estimate = frames * pixel_patches
        assert abs(estimate - target) / target <= 0.10
