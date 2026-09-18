from src.analyze.Validate_Sparrow_hypothesises.analyze_h32_depth import (
    _paired_statistics,
)


def _rows_for_depths(depths: tuple[int, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for depth in depths:
        for sample_id in ("s1", "s2"):
            for condition, value in (
                ("full", 10.0 + depth),
                ("zero", 10.0),
                ("cut", 9.0),
            ):
                rows.append(
                    {
                        "dataset": "vdc",
                        "training_corpus": "llava68k",
                        "draft_depth": depth,
                        "sample_id": sample_id,
                        "visual_condition": condition,
                        "accepted_effective_tokens": value,
                    }
                )
    return rows


def test_h32_pairing_includes_depth5_and_relative_interaction() -> None:
    paired = _paired_statistics(_rows_for_depths((1, 3, 5)))

    value_contrasts = {
        (row["draft_depth"], row["contrast"])
        for row in paired
        if row["contrast"] == "full_minus_zero"
    }
    assert value_contrasts == {
        (1, "full_minus_zero"),
        (3, "full_minus_zero"),
        (5, "full_minus_zero"),
    }

    interactions = {
        row["draft_depth"]: row
        for row in paired
        if row["contrast"] == "depth_interaction_full_minus_zero"
    }
    assert set(interactions) == {"3_minus_1", "5_minus_1", "5_minus_3"}
    assert interactions["3_minus_1"]["mean"] == 2.0
    assert interactions["5_minus_1"]["mean"] == 4.0
    assert interactions["5_minus_3"]["mean"] == 2.0
