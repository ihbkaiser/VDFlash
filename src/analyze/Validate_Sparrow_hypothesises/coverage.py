"""Coverage gates for measured local insight figures.

Row validation answers "is this record well formed?".  Coverage answers the
different question "does this run contain the complete, paired experiment
matrix required by the selected profile?".  Keeping the two gates separate
prevents a one-row smoke test from being presented as a paper-shaped result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .paper_contract import PaperContract


@dataclass(frozen=True)
class CoverageIssue:
    code: str
    message: str
    figure: str | None = None


@dataclass
class CoverageReport:
    valid: bool
    enforced: bool
    minimum_paired_samples: int
    paired_samples: int
    required: dict[str, Any]
    observed: dict[str, Any]
    issues: list[CoverageIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "enforced": self.enforced,
            "minimum_paired_samples": self.minimum_paired_samples,
            "paired_samples": self.paired_samples,
            "required": self.required,
            "observed": self.observed,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _target(row: Mapping[str, Any]) -> int | None:
    value = row.get("calibration_target_visual_tokens")
    if value is None:
        value = row.get("target_visual_tokens")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _valid_runtime_row(row: Mapping[str, Any], contract: PaperContract) -> bool:
    if row.get("status") == "error":
        return False
    if contract.strict_calibration and row.get("calibration_status") != "ok":
        return False
    return row.get("sample_id") is not None and _target(row) is not None


def _ids(rows: Iterable[Mapping[str, Any]], predicate: Any, contract: PaperContract) -> set[str]:
    return {
        str(row["sample_id"])
        for row in rows
        if _valid_runtime_row(row, contract) and predicate(row)
    }


def _group_name(figure: str, target: int | None = None, extra: str = "") -> str:
    pieces = [figure]
    if target is not None:
        pieces.append(str(target))
    if extra:
        pieces.append(extra)
    return ":".join(pieces)


def build_coverage(rows: Iterable[Mapping[str, Any]], contract: PaperContract) -> CoverageReport:
    """Build the profile coverage matrix and paired-cohort gate."""

    rows = list(rows)
    issues: list[CoverageIssue] = []
    milestones = tuple(int(value) for value in contract.visual_token_milestones)
    retentions = tuple(float(value) for value in contract.retention_percentages)
    anchor = int(contract.retention_anchor_visual_tokens)
    cuts = tuple(int(value) for value in contract.layer_cut_points)
    layer_range = tuple(range(1, int(contract.layer_count) + 1))
    required: dict[str, Any] = {
        "figures": list(contract.required_figures),
        "visual_token_milestones": list(milestones),
        "retention_anchor_visual_tokens": anchor,
        "retention_percentages": list(retentions),
        "retention_policies": ["last_instruction", "all_text"],
        "layer_cut_points": list(cuts),
        "layers": list(layer_range),
        "minimum_paired_samples": contract.minimum_paired_samples,
    }
    observed: dict[str, Any] = {"groups": {}, "missing": []}
    group_sets: list[set[str]] = []

    def require_group(name: str, predicate: Any, *, figure: str) -> None:
        sample_ids = _ids(rows, predicate, contract)
        group_sets.append(sample_ids)
        observed["groups"][name] = {
            "sample_count": len(sample_ids),
            "sample_ids": sorted(sample_ids),
        }
        if len(sample_ids) < contract.minimum_paired_samples:
            observed["missing"].append(name)
            issues.append(CoverageIssue(
                "insufficient_samples",
                f"{name} has {len(sample_ids)} samples; requires "
                f"{contract.minimum_paired_samples}",
                figure,
            ))

    # Figure 1(a): local MSD keep-visual and remove-all series at all four
    # calibrated points.  Older rows are accepted through the condition
    # fallback, while new rows carry an explicit series_id.
    for target in milestones:
        require_group(
            _group_name("Figure 1(a)", target, "msd_keep_visual"),
            lambda row, target=target: (
                row.get("paper_figure") == "Figure 1(a)"
                and _target(row) == target
                and str(row.get("series_id") or "msd_keep_visual") == "msd_keep_visual"
                and row.get("condition") == "full"
            ),
            figure="Figure 1(a)",
        )
        require_group(
            _group_name("Figure 1(a)", target, "msd_remove_all"),
            lambda row, target=target: (
                row.get("paper_figure") == "Figure 1(a)"
                and _target(row) == target
                and str(row.get("series_id") or "") == "msd_remove_all"
            ),
            figure="Figure 1(a)",
        )

    # Figure 1(b) is anchored at the long-context point to avoid mixing
    # retention with the separate visual-length sweep.
    for policy in ("last_instruction", "all_text"):
        for retention in retentions:
            require_group(
                _group_name("Figure 1(b)", anchor, f"{policy}:{retention:g}"),
                lambda row, policy=policy, retention=retention: (
                    row.get("paper_figure") == "Figure 1(b)"
                    and _target(row) == anchor
                    and str(row.get("selection_policy") or "") == policy
                    and row.get("retention_percentage") is not None
                    and float(row.get("retention_percentage")) == retention
                ),
                figure="Figure 1(b)",
            )

    # Figure 2 uses the official MSD draft and the final instruction query.
    for target in (int(contract.attention_short_tokens), int(contract.attention_long_tokens)):
        require_group(
            _group_name("Figure 2", target, "msd_draft:last_instruction"),
            lambda row, target=target: (
                row.get("paper_figure") == "Figure 2"
                and row.get("modality") == "summary"
                and row.get("attention_source") == "msd_draft"
                and row.get("attention_policy") == "last_instruction"
                and _target(row) == target
            ),
            figure="Figure 2",
        )

    # Figure 3(a), 3(b), and Figure 6 share the local 3K layer-analysis
    # cohort.  Figure 3(a) is deliberately called prefix agreement in the
    # renderer; it is not mislabeled as MVBench accuracy.
    layer_target = int(contract.attention_long_tokens)
    for cut in cuts:
        require_group(
            _group_name("Figure 3", layer_target, f"cut:{cut}"),
            lambda row, cut=cut: (
                row.get("paper_figure") == "Figure 3"
                and _target(row) == layer_target
                and row.get("layer_cut") is not None
                and int(row.get("layer_cut")) == cut
                and row.get("prefix_agreement") is not None
                and row.get("answer_quality_delta") is not None
            ),
            figure="Figure 3",
        )
    for layer in layer_range:
        require_group(
            _group_name("Figure 3(b)", layer_target, f"layer:{layer}"),
            lambda row, layer=layer: (
                row.get("paper_figure") == "Figure 3(b)"
                and _target(row) == layer_target
                and row.get("layer") is not None
                and int(row.get("layer")) == layer
                and row.get("per_head_visual_mass") is not None
            ),
            figure="Figure 3(b)",
        )
        require_group(
            _group_name("Figure 6 / Appendix D", layer_target, f"layer:{layer}"),
            lambda row, layer=layer: (
                row.get("paper_figure") == "Figure 6 / Appendix D"
                and _target(row) == layer_target
                and row.get("layer") is not None
                and int(row.get("layer")) == layer
                and row.get("visual_cosine") is not None
                and row.get("text_cosine") is not None
            ),
            figure="Figure 6 / Appendix D",
        )

    paired = set.intersection(*group_sets) if group_sets else set()
    observed["paired_sample_ids"] = sorted(paired)
    observed["paired_sample_count"] = len(paired)
    if len(paired) < contract.minimum_paired_samples:
        issues.append(CoverageIssue(
            "paired_cohort_too_small",
            f"intersection across required groups has {len(paired)} samples; "
            f"requires {contract.minimum_paired_samples}",
        ))

    enforced = contract.profile.startswith("local_") or contract.strict_calibration
    valid = not issues if enforced else True
    return CoverageReport(
        valid=valid,
        enforced=enforced,
        minimum_paired_samples=contract.minimum_paired_samples,
        paired_samples=len(paired),
        required=required,
        observed=observed,
        issues=issues,
    )
