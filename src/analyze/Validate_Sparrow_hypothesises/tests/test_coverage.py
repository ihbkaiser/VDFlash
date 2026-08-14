from src.analyze.Validate_Sparrow_hypothesises.coverage import build_coverage
from src.analyze.Validate_Sparrow_hypothesises.paper_contract import load_contract


def test_local_profile_rejects_partial_smoke_rows():
    contract = load_contract(
        "src/analyze/Validate_Sparrow_hypothesises/configs/local_insight_vdc50.yaml"
    )
    row = {
        "paper_figure": "Figure 1(a)",
        "sample_id": "one",
        "target_visual_tokens": 400,
        "calibration_status": "ok",
        "condition": "full",
        "series_id": "msd_keep_visual",
    }
    report = build_coverage([row], contract)
    assert report.enforced
    assert not report.valid
    assert report.paired_samples == 0


def test_paper_profile_keeps_coverage_as_diagnostic():
    contract = load_contract("src/analyze/Validate_Sparrow_hypothesises/configs/paper_contract.yaml")
    report = build_coverage([], contract)
    assert not report.enforced
    assert report.valid
