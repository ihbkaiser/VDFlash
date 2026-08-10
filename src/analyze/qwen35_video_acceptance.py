"""Compatibility helpers for the legacy Qwen3.5 acceptance tests.

The active dataset runner lives in
``Whether_they_are_appliable_for_dDrafter.qwen35_dflash_benchmark``.  These
small helpers preserve the old test-facing API without reintroducing the old
CLI or its separate experiment modes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np


def sample_frame_indices(
    total_frames: int,
    duration_sec: float,
    *,
    experiment_type: str = "natural",
    requested_count: Optional[int] = None,
) -> list[int]:
    """Return deterministic, uniformly spread frame indices.

    ``natural`` uses approximately one frame per second and caps the sample
    at 1020 frames, matching the legacy acceptance-test contract.  The active
    benchmark uses its simpler fixed ``--num-frames`` sampler instead.
    """

    total = int(total_frames)
    if total <= 0:
        raise ValueError("total_frames must be positive")
    if experiment_type == "controlled":
        if requested_count is None:
            raise ValueError("controlled sampling requires requested_count")
        count = int(requested_count)
    elif experiment_type == "natural":
        if requested_count is not None:
            count = int(requested_count)
        else:
            count = int(round(float(duration_sec)))
        count = min(count, 1020)
    else:
        raise ValueError("experiment_type must be 'natural' or 'controlled'")
    count = min(count, total)
    if count < 1:
        raise ValueError("requested sample count must be positive")
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def config_hash(**kwargs: object) -> str:
    """Hash experiment configuration while ignoring legacy context labels."""

    payload = {key: value for key, value in kwargs.items() if key != "context_mode"}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
