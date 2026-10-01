"""Multimodal Speculative Decoding algorithm support."""

from .curriculum import choose_visual_sample, visual_ratio_for_epoch

__all__ = ["choose_visual_sample", "visual_ratio_for_epoch"]
