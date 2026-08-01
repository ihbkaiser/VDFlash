"""Training and verification utilities for a Qwen2.5-VL DFlash drafter.

The package deliberately keeps the target model outside the draft checkpoint.  A
checkpoint contains only the lightweight block-diffusion adapter and a config
fingerprint for the target model it was trained with.
"""

from .config import DFlashTrainConfig
from .data import (
    MaskedBlockBatch,
    build_masked_blocks,
    make_dense_attention_mask,
    sample_anchor_positions,
)
from .losses import weighted_block_cross_entropy
from .model import DFlashVLMModel
from .vlm_decode import Qwen25VLDFlashDecoder, VLMDecodeResult

__all__ = [
    "DFlashTrainConfig",
    "DFlashVLMModel",
    "MaskedBlockBatch",
    "build_masked_blocks",
    "make_dense_attention_mask",
    "sample_anchor_positions",
    "weighted_block_cross_entropy",
    "Qwen25VLDFlashDecoder",
    "VLMDecodeResult",
]
