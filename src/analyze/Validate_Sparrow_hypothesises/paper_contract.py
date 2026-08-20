"""Machine-readable contract for the experiments described in Sparrow.

The contract records what must be true for an experiment to be comparable to
a paper figure. It does not encode expected numerical results because those
depend on hardware, checkpoint revision and the local VDC subset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PaperContract:
    paper: str
    paper_pdf: str
    dataset: str
    visual_token_milestones: tuple[int, ...]
    retention_percentages: tuple[float, ...]
    msd_target_model: str
    msd_weights: str
    layer_target_model: str
    tree_total_tokens: int
    tree_depth: int
    tree_top_k: int
    temperature: float
    max_new_tokens: int
    calibration_tolerance: float
    attention_short_tokens: int
    attention_long_tokens: int
    layer_cut_points: tuple[int, ...]
    profile: str = "paper"
    minimum_paired_samples: int = 1
    strict_calibration: bool = False
    retention_anchor_visual_tokens: int = 25000
    layer_count: int = 28
    required_figures: tuple[str, ...] = (
        "Figure 1(a)",
        "Figure 1(b)",
        "Figure 2",
        "Figure 3",
        "Figure 3(b)",
        "Figure 6 / Appendix D",
    )
    paper_plot_formats: tuple[str, ...] = ("png",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONTRACT = PaperContract(
    paper="Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs",
    paper_pdf="externals/Sparrow/2026.acl-long.450.pdf",
    dataset="dataset/VideoDetailCaption/subset_manifest.jsonl",
    visual_token_milestones=(400, 3000, 13000, 25000),
    retention_percentages=(100.0, 25.0, 10.0, 5.0, 1.0, 0.0),
    msd_target_model="Qwen/Qwen2-VL-7B-Instruct",
    msd_weights="lucylyn/MSD-Qwen2VL-7B-Instruct",
    # Use the cached Qwen2-VL checkpoint for all remaining local analyses.
    # This is a local substitution for the paper's Qwen2.5-VL layer model.
    layer_target_model="Qwen/Qwen2-VL-7B-Instruct",
    tree_total_tokens=30,
    tree_depth=4,
    tree_top_k=8,
    temperature=0.0,
    max_new_tokens=512,
    calibration_tolerance=0.10,
    attention_short_tokens=400,
    attention_long_tokens=3000,
    layer_cut_points=(0, 4, 8, 12, 16, 20, 24),
)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the paper contract") from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Contract must contain a mapping: {path}")
    return value


def load_contract(path: str | Path | None = None) -> PaperContract:
    """Load a contract YAML, filling omitted values from the default contract."""

    if path is None:
        return DEFAULT_CONTRACT
    values = DEFAULT_CONTRACT.to_dict()
    values.update(dict(_read_yaml(Path(path))))
    for field in (
        "visual_token_milestones",
        "retention_percentages",
        "layer_cut_points",
        "required_figures",
        "paper_plot_formats",
    ):
        values[field] = tuple(values[field])
    return PaperContract(**values)


def validate_contract(contract: PaperContract) -> list[str]:
    """Return errors; an empty list means that the contract is valid."""

    errors: list[str] = []
    if len(contract.visual_token_milestones) != 4:
        errors.append("visual_token_milestones must contain the four paper points")
    if tuple(sorted(contract.visual_token_milestones)) != contract.visual_token_milestones:
        errors.append("visual_token_milestones must be strictly increasing")
    if len(set(contract.retention_percentages)) != len(contract.retention_percentages):
        errors.append("retention_percentages contains duplicates")
    if any(value < 0 or value > 100 for value in contract.retention_percentages):
        errors.append("retention_percentages must be in [0, 100]")
    if contract.tree_total_tokens <= 0 or contract.tree_depth <= 0 or contract.tree_top_k <= 0:
        errors.append("MSD tree parameters must be positive")
    if contract.temperature != 0:
        errors.append("lossless validation requires temperature=0")
    if contract.max_new_tokens <= 0:
        errors.append("max_new_tokens must be positive")
    if not 0 < contract.calibration_tolerance <= 1:
        errors.append("calibration_tolerance must be in (0, 1]")
    if not contract.msd_target_model.startswith("Qwen/"):
        errors.append("MSD target must be the Qwen2-VL family used by Figure 1/2")
    if not contract.layer_target_model.startswith("Qwen/"):
        errors.append("layer analysis target must be a Qwen-VL family model")
    if contract.minimum_paired_samples <= 0:
        errors.append("minimum_paired_samples must be positive")
    if contract.retention_anchor_visual_tokens not in contract.visual_token_milestones:
        errors.append("retention_anchor_visual_tokens must be one of the visual milestones")
    if contract.layer_count <= 0:
        errors.append("layer_count must be positive")
    if not contract.required_figures:
        errors.append("required_figures must not be empty")
    if any(fmt not in {"png", "pdf", "svg"} for fmt in contract.paper_plot_formats):
        errors.append("paper_plot_formats must contain only png, pdf or svg")
    return errors


def paper_contract_rows(contract: PaperContract) -> list[dict[str, Any]]:
    """Return traceability rows used in the final report."""

    return [
        {
            "figure": "Figure 1(a)",
            "claim": "MSD acceptance and latency degrade as visual length grows",
            "model": contract.msd_target_model,
            "visual_tokens": list(contract.visual_token_milestones),
        "metric": "accepted_length, end_to_end_seconds",
        },
        {
            "figure": "Figure 1(b)",
            "claim": "video acceptance is robust or improves when draft visual input is reduced",
            "model": contract.msd_target_model,
            "retention_percentages": list(contract.retention_percentages),
            "metric": "accepted_length, lossless_output_match",
        },
        {
            "figure": "Figure 2",
            "claim": "long visual context dilutes draft attention",
            "model": contract.msd_target_model,
            "visual_tokens": [contract.attention_short_tokens, contract.attention_long_tokens],
            "metric": "query_only_attention_mass, normalized_visual_entropy",
        },
        {
            "figure": "Figure 3",
            "claim": "visual flow becomes less important after the middle layers",
            "model": contract.layer_target_model,
            "layer_cut_points": list(contract.layer_cut_points),
            "metric": "local_prefix_agreement_and_vdc_answer_quality_delta",
        },
        {
            "figure": "Figure 6 / Appendix D",
            "claim": "visual hidden states internalize into text representations",
            "model": contract.layer_target_model,
            "metric": "layerwise_cosine_similarity_to_input_embedding",
        },
    ]
