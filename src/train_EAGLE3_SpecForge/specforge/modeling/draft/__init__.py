from .base import Eagle3DraftModel
from .llama3_eagle import LlamaForCausalLMEagle3
from .registry import DRAFT_REGISTRY, available_drafts, register_draft, resolve_draft

__all__ = [
    "Eagle3DraftModel",
    "LlamaForCausalLMEagle3",
    "DRAFT_REGISTRY",
    "register_draft",
    "resolve_draft",
    "available_drafts",
]
