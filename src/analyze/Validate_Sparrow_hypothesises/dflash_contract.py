"""Evidence contract for the Qwen2.5-VL DFlash validation path.

This module deliberately does not import the MSD paper contract.  The two
contracts share a few field names, but DFlash attention and hidden-context
retention have different semantics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class DFlashSemanticStatus(str, Enum):
    DIRECT = "direct"
    ADAPTED = "adapted"
    TARGET_SIDE_DIAGNOSTIC = "target_side_diagnostic"


class DFlashExperiment(str, Enum):
    LENGTH_SWEEP = "length_sweep"
    TARGET_HIDDEN_VISUAL_RETENTION = "target_hidden_visual_retention"
    CONTEXT_ATTENTION = "dflash_context_attention"
    TARGET_VISUAL_KV = "qwen25vl_target_visual_kv"
    TARGET_ATTENTION = "qwen25vl_target_attention"
    TARGET_HIDDEN_COSINE = "qwen25vl_target_hidden_cosine"


DFLASH_LENGTH_TARGETS = (400, 3_000, 13_000, 25_000)
DFLASH_RETENTION_PERCENTAGES = (100, 25, 10, 5, 1, 0)
DFLASH_LAYER_CUTS = (0, 4, 8, 12, 16, 20, 24)

_BASE_REQUIRED_FIELDS = (
    "backend",
    "experiment",
    "semantic_status",
    "target_model",
    "draft_checkpoint",
    "draft_config",
    "sample_id",
    "input_fingerprint",
)
_DECODE_EXPERIMENTS = {
    DFlashExperiment.LENGTH_SWEEP.value,
    DFlashExperiment.TARGET_HIDDEN_VISUAL_RETENTION.value,
}
_TARGET_DIAGNOSTICS = {
    DFlashExperiment.TARGET_VISUAL_KV.value,
    DFlashExperiment.TARGET_ATTENTION.value,
    DFlashExperiment.TARGET_HIDDEN_COSINE.value,
}
_EXPECTED_SEMANTICS = {
    DFlashExperiment.LENGTH_SWEEP.value: DFlashSemanticStatus.DIRECT.value,
    DFlashExperiment.TARGET_HIDDEN_VISUAL_RETENTION.value: DFlashSemanticStatus.ADAPTED.value,
    DFlashExperiment.CONTEXT_ATTENTION.value: DFlashSemanticStatus.ADAPTED.value,
    DFlashExperiment.TARGET_VISUAL_KV.value: DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value,
    DFlashExperiment.TARGET_ATTENTION.value: DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value,
    DFlashExperiment.TARGET_HIDDEN_COSINE.value: DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value,
}


def validate_dflash_row(row: Mapping[str, Any]) -> list[str]:
    """Return contract errors for one DFlash JSONL evidence row."""

    errors: list[str] = []
    missing = [field for field in _BASE_REQUIRED_FIELDS if not row.get(field)]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    if row.get("backend") != "dflash":
        errors.append("backend must be 'dflash'")

    experiment = row.get("experiment")
    known_experiments = {item.value for item in DFlashExperiment}
    if experiment not in known_experiments:
        errors.append(f"unknown DFlash experiment: {experiment!r}")

    semantic_status = row.get("semantic_status")
    if semantic_status not in {item.value for item in DFlashSemanticStatus}:
        errors.append(f"unknown DFlash semantic_status: {semantic_status!r}")
    expected_status = _EXPECTED_SEMANTICS.get(experiment)
    if expected_status is not None and semantic_status != expected_status:
        errors.append(f"{experiment} must be {expected_status} but was {semantic_status}")

    is_non_success = row.get("status") in {"error", "unsupported"}
    if experiment in _DECODE_EXPERIMENTS and not is_non_success:
        for field in ("target_output_ids", "speculative_output_ids"):
            if field not in row or row[field] is None:
                errors.append(f"missing {field} for decode experiment")

    if experiment == DFlashExperiment.TARGET_HIDDEN_VISUAL_RETENTION.value:
        if not row.get("full_target_input_fingerprint"):
            errors.append("missing full_target_input_fingerprint for retention")
        if not is_non_success and not row.get("target_input_fingerprint"):
            errors.append("missing target_input_fingerprint for retention")
        if (
            not is_non_success
            and row.get("target_input_fingerprint")
            != row.get("full_target_input_fingerprint")
        ):
            errors.append("target_input_fingerprint must equal full_target_input_fingerprint for retention")
        if semantic_status != DFlashSemanticStatus.ADAPTED.value:
            errors.append("target-hidden visual retention must be adapted")

    if experiment == DFlashExperiment.CONTEXT_ATTENTION.value:
        if semantic_status != DFlashSemanticStatus.ADAPTED.value:
            errors.append("DFlash context attention must be adapted")
        if row.get("attention_source") == "msd_draft":
            errors.append("DFlash context attention cannot use attention_source='msd_draft'")
        if not row.get("query_policy"):
            errors.append("missing query_policy for DFlash context attention")

    if experiment in _TARGET_DIAGNOSTICS:
        if semantic_status != DFlashSemanticStatus.TARGET_SIDE_DIAGNOSTIC.value:
            errors.append(f"{experiment} must be a target-side diagnostic")
        if row.get("target_output_ids") is not None or row.get("speculative_output_ids") is not None:
            errors.append(f"{experiment} must not claim speculative decode IDs")

    return errors


def validate_dflash_grid(config: Mapping[str, Any]) -> list[str]:
    """Validate the requested DFlash experiment milestones."""

    errors: list[str] = []
    lengths = tuple(config.get("length_targets", DFLASH_LENGTH_TARGETS))
    missing_lengths = [value for value in DFLASH_LENGTH_TARGETS if value not in lengths]
    if missing_lengths:
        errors.append(f"missing length targets: {missing_lengths}")

    retentions = tuple(config.get("retention_percentages", DFLASH_RETENTION_PERCENTAGES))
    missing_retentions = [value for value in DFLASH_RETENTION_PERCENTAGES if value not in retentions]
    if missing_retentions:
        errors.append(f"missing retention percentages: {missing_retentions}")
    return errors


def make_dflash_metadata(
    *,
    target_model: str,
    draft_checkpoint: str,
    draft_config: str,
    experiment: DFlashExperiment | str,
    semantic_status: DFlashSemanticStatus | str,
) -> dict[str, Any]:
    """Build the immutable metadata shared by every DFlash stage row."""

    return {
        "backend": "dflash",
        "experiment": getattr(experiment, "value", experiment),
        "semantic_status": getattr(semantic_status, "value", semantic_status),
        "target_model": target_model,
        "draft_checkpoint": draft_checkpoint,
        "draft_config": draft_config,
    }
