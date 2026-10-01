from .base import Eagle3DraftModel
from .dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)
from .domino import DominoDraftModel
from .dspark import DSparkDraftModel
try:
    from .llama3_eagle import LlamaForCausalLMEagle3
except ModuleNotFoundError as exc:
    if exc.name != "torch.nn.attention":
        raise
    # Torch < 2.4 has no FlexAttention package. Other draft architectures
    # remain importable for lightweight CPU-side configuration tests.
    LlamaForCausalLMEagle3 = None
from .msd import MSDConfig, MSDDraftModel
try:
    from .peagle import PEagleDraftModel
except ModuleNotFoundError as exc:
    if exc.name != "torch.nn.attention":
        raise
    PEagleDraftModel = None
from .registry import DRAFT_REGISTRY, available_drafts, register_draft, resolve_draft

__all__ = [
    "Eagle3DraftModel",
    "DFlashDraftModel",
    "DominoDraftModel",
    "DSparkDraftModel",
    "LlamaForCausalLMEagle3",
    "MSDConfig",
    "MSDDraftModel",
    "PEagleDraftModel",
    "build_target_layer_ids",
    "extract_context_feature",
    "sample",
    "DRAFT_REGISTRY",
    "register_draft",
    "resolve_draft",
    "available_drafts",
]
