from .base import Eagle3DraftModel
from .dflash import (
    DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
    sample,
)
from .domino import DominoDraftModel
from .dspark import DSparkDraftModel
from .registry import DRAFT_REGISTRY, available_drafts, register_draft, resolve_draft


def _register_flex_attention_placeholder(name, import_error):
    """Keep config loading available when the local Torch lacks FlexAttention."""

    from transformers.models.llama.configuration_llama import LlamaConfig

    def unavailable_init(self, *args, **kwargs):
        del self, args, kwargs
        raise RuntimeError(
            f"{name} requires torch.nn.attention.flex_attention"
        ) from import_error

    placeholder = type(
        name,
        (),
        {
            "config_class": LlamaConfig,
            "__init__": unavailable_init,
            "__module__": __name__,
        },
    )
    return register_draft(placeholder, name=name)


try:
    from .llama3_eagle import LlamaForCausalLMEagle3
except ModuleNotFoundError as exc:
    if exc.name != "torch.nn.attention":
        raise
    # Torch < 2.4 has no FlexAttention package. Other draft architectures
    # remain importable for lightweight CPU-side configuration tests.
    LlamaForCausalLMEagle3 = _register_flex_attention_placeholder(
        "LlamaForCausalLMEagle3", exc
    )
from .msd import MSDConfig, MSDDraftModel
try:
    from .peagle import PEagleDraftModel
except ModuleNotFoundError as exc:
    if exc.name != "torch.nn.attention":
        raise
    PEagleDraftModel = _register_flex_attention_placeholder("PEagleDraftModel", exc)

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
