"""Deterministic form of MSD's 40-epoch text-to-vision curriculum."""

from __future__ import annotations

import hashlib

MSD_TOTAL_EPOCHS = 40
MSD_CURRICULUM_VERSION = "msd-linear-40-v1"


def visual_ratio_for_epoch(
    epoch_now: int,
    total_epoch: int = MSD_TOTAL_EPOCHS,
) -> float:
    """Return the visual-example probability from the public MSD schedule.

    Epochs use one-based indexing. The replication profile deliberately fixes
    ``total_epoch`` to 40 so a config cannot silently change the experiment.
    """

    if total_epoch != MSD_TOTAL_EPOCHS:
        raise ValueError(
            f"MSD replication requires total_epoch={MSD_TOTAL_EPOCHS}, "
            f"got {total_epoch}"
        )
    if isinstance(epoch_now, bool) or not isinstance(epoch_now, int):
        raise TypeError("epoch_now must be an integer")
    if epoch_now < 1 or epoch_now > total_epoch:
        raise ValueError("epoch_now must be in [1, total_epoch]")
    if epoch_now <= total_epoch // 2:
        return 0.0
    if epoch_now < total_epoch:
        return (epoch_now - total_epoch // 2) / total_epoch * 2
    return 1.0


def choose_visual_sample(
    seed: int,
    epoch_now: int,
    sample_index: int,
    total_epoch: int = MSD_TOTAL_EPOCHS,
) -> bool:
    """Choose a corpus without relying on mutable worker-local RNG state."""

    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    ratio = visual_ratio_for_epoch(epoch_now, total_epoch)
    digest = hashlib.sha256(
        f"msd:{int(seed)}:{epoch_now}:{sample_index}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    return draw < ratio


__all__ = [
    "MSD_CURRICULUM_VERSION",
    "MSD_TOTAL_EPOCHS",
    "choose_visual_sample",
    "visual_ratio_for_epoch",
]
