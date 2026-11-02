"""Video-reader setup shared by processor calibration and DFlash inference."""

from __future__ import annotations

import os


def configure_video_reader(video_reader: str = "torchvision") -> None:
    """Select a reader, adapt PyAV, and raise Qwen's per-frame pixel cap."""

    if video_reader not in {"torchvision", "decord"}:
        raise ValueError(f"unsupported Qwen video reader: {video_reader}")
    os.environ["FORCE_QWENVL_VIDEO_READER"] = video_reader
    if video_reader == "torchvision":
        try:
            import av
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("PyAV is required by the torchvision Qwen video reader") from exc
        original = av.open
        if not getattr(original, "_vdflash_metadata_errors_compat", False):
            def compatible_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return original(*args, **kwargs)

            compatible_open._vdflash_metadata_errors_compat = True
            compatible_open._vdflash_original_open = original
            av.open = compatible_open

    # Qwen's stock video utility caps each frame at 768 image patches even
    # when the experiment asks for more. Large VDC token targets then silently
    # become unreachable. Calibration measures the resulting token count after
    # applying matched min/max pixel settings, so only usable candidates enter
    # the paired cohort.
    try:
        from qwen_vl_utils import vision_process
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("qwen-vl-utils is required for Qwen video processing") from exc
    frame_patch_limit = int(os.environ.get("VD_FLASH_VIDEO_FRAME_MAX_PATCHES", "4096"))
    if frame_patch_limit <= 0:
        raise ValueError("VD_FLASH_VIDEO_FRAME_MAX_PATCHES must be positive")
    requested_pixel_limit = frame_patch_limit * 28 * 28
    vision_process.VIDEO_MAX_PIXELS = max(
        int(vision_process.VIDEO_MAX_PIXELS), requested_pixel_limit
    )


__all__ = ["configure_video_reader"]
