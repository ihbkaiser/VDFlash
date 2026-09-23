"""Analyze equal-budget visual versus text deletion in cached DFlash runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_dflash_h11 import SITES, fit_reference


def choose_positions(input_ids: list[int], anchor: int, *, visual_ids: set[int],
                     special_ids: set[int], budget: int, seed: int, sample_id: str):
    """Sample exactly the same number of visual and ordinary prompt tokens."""
    visual = [i for i, token in enumerate(input_ids[:anchor]) if token in visual_ids]
    text = [i for i, token in enumerate(input_ids[:anchor])
            if token not in visual_ids and token not in special_ids]
    if len(visual) < budget or len(text) < budget:
        return None, len(visual), len(text)
    stable_seed = seed ^ int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(stable_seed)
    positions = {
        "visual_k": sorted(int(i) for i in rng.choice(visual, size=budget, replace=False)),
        "text_k": sorted(int(i) for i in rng.choice(text, size=budget, replace=False)),
    }
    return positions, len(visual), len(text)


def analyze_matched(source_run: Path, matched_run: Path, *, bootstrap: int = 2000,
                    components: int = 32) -> dict:
    with np.load(source_run / "train_full.npz", allow_pickle=False) as ref, \
            np.load(source_run / "test_full_cut.npz", allow_pickle=False) as previous, \
            np.load(matched_run / "matched_vectors.npz", allow_pickle=False) as matched:
        train = ref["vectors"]
        full, cut = previous["full"], previous["cut"]
        source_ids = [str(value) for value in previous["sample_ids"].tolist()]
        visual, text = matched["visual"], matched["text"]
        sample_ids = [str(value) for value in matched["sample_ids"].tolist()]
        budget = int(matched["budget"].item())
        checkpoints = [str(item["checkpoint"].item()) for item in (ref, previous, matched)]
    if len(set(checkpoints)) != 1:
        raise ValueError("matched study must use the same draft checkpoint as H1.1")
    if len(set(sample_ids)) != len(sample_ids) or not set(sample_ids).issubset(set(source_ids)):
        raise ValueError("matched sample IDs are not a unique subset of the original run")
    if (train.ndim != 4 or full.ndim != 4 or train.shape[1:] != full.shape[1:]
            or visual.shape != text.shape or visual.shape[1:] != train.shape[1:]
            or len(visual) != len(sample_ids) or train.shape[2] != len(SITES)):
        raise ValueError("incompatible reference, original, or paired vectors")
    if not all(np.isfinite(arr).all() for arr in (train, full, cut, visual, text)):
        raise ValueError("nonfinite matched representations")
    old_rows, new_rows = {}, {}
    with (source_run / "test_full_cut.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            old_rows.setdefault(str(row["sample_id"]), {})[row["visual_condition"]] = row
    with (matched_run / "matched_reports.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid, condition = str(row["sample_id"]), row["matched_condition"]
            if condition not in {"visual_k", "text_k"} or condition in new_rows.setdefault(sid, {}):
                raise ValueError("duplicate or invalid matched condition")
            new_rows[sid][condition] = row
    if set(new_rows) != set(sample_ids):
        raise ValueError("matched reports do not cover captured vectors")
    full_count, cut_count, visual_count, text_count = [], [], [], []
    for sid in sample_ids:
        old, new = old_rows[sid], new_rows[sid]
        if set(old) != {"full", "cut"} or set(new) != {"visual_k", "text_k"}:
            raise ValueError(f"missing Full/Cut or visual/text pair for {sid}")
        visual_row, text_row = new["visual_k"], new["text_k"]
        if (visual_row["budget"] != budget or text_row["budget"] != budget
                or len(visual_row["dropped_positions"]) != budget
                or len(text_row["dropped_positions"]) != budget
                or visual_row["context_length"] != text_row["context_length"]):
            raise ValueError(f"unequal deletion budget or context length for {sid}")
        for row in (visual_row, text_row):
            if (row["target_input_fingerprint"] != old["full"]["target_input_fingerprint"]
                    or row["target_output_hash"] != old["full"]["target_output_hash"]):
                raise ValueError(f"target prompt or continuation differs for {sid}")
        full_count.append(int(old["full"]["acceptance_rounds"][0]["matched_proposals"]))
        cut_count.append(int(old["cut"]["acceptance_rounds"][0]["matched_proposals"]))
        visual_count.append(int(visual_row["matched_proposals"]))
        text_count.append(int(text_row["matched_proposals"]))
    full_count, cut_count = np.asarray(full_count), np.asarray(cut_count)
    visual_count, text_count = np.asarray(visual_count), np.asarray(text_count)
    proposal_delta = visual_count - text_count
    rng = np.random.default_rng(1847)
    indices = rng.integers(0, len(sample_ids), size=(bootstrap, len(sample_ids)))
    proposal_ci = np.quantile(proposal_delta[indices].mean(axis=1), [0.025, 0.975]).tolist()
    fit_count = int(len(train) * 0.75)
    summaries = []
    for layer in range(train.shape[1]):
        for site_index, site in enumerate(SITES):
            print(f"[H1.1 matched analysis] layer {layer} {site}", flush=True)
            distance, cutoff = fit_reference(
                train[:fit_count, layer, site_index],
                train[fit_count:, layer, site_index], components=components,
            )
            visual_distance = distance(visual[:, layer, site_index])
            text_distance = distance(text[:, layer, site_index])
            delta = visual_distance - text_distance
            ci = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist()
            summaries.append({
                "layer": layer, "site": site,
                "mean_visual_minus_text_distance": float(delta.mean()),
                "distance_ci95": ci,
                "visual_ood_rate": float((visual_distance > cutoff).mean()),
                "text_ood_rate": float((text_distance > cutoff).mean()),
            })
    return {
        "checkpoint": checkpoints[0], "budget": budget,
        "n_source": len(source_ids), "n_eligible": len(sample_ids),
        "mean_full_matches": float(full_count.mean()),
        "mean_full_visual_cut_matches": float(cut_count.mean()),
        "mean_visual_k_matches": float(visual_count.mean()),
        "mean_text_k_matches": float(text_count.mean()),
        "mean_visual_minus_text_matches": float(proposal_delta.mean()),
        "proposal_delta_ci95": proposal_ci, "layers": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--matched-run", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_matched(args.source_run, args.matched_run)
    lines = [
        f'Equal-budget visual/text deletion: {result["budget"]} keys per condition, '
        f'{result["n_eligible"]}/{result["n_source"]} eligible samples',
        "Teacher-token matches in the first block (means on eligible samples):",
        f'  Full={result["mean_full_matches"]:.2f}, full visual Cut={result["mean_full_visual_cut_matches"]:.2f}, '
        f'visual-{result["budget"]}={result["mean_visual_k_matches"]:.2f}, '
        f'text-{result["budget"]}={result["mean_text_k_matches"]:.2f}',
        f'  visual minus text={result["mean_visual_minus_text_matches"]:+.2f} '
        f'[95% CI {result["proposal_delta_ci95"][0]:+.2f}, '
        f'{result["proposal_delta_ci95"][1]:+.2f}]; negative means visual deletion loses more matches',
        "Training-reference distance: positive visual-minus-text means visual deletion shifts farther",
        "Layer  Site       Distance delta [95% CI]    OOD visual/text",
    ]
    for row in result["layers"]:
        lo, hi = row["distance_ci95"]
        lines.append(
            f'{row["layer"]:>5}  {row["site"]:<9} '
            f'{row["mean_visual_minus_text_distance"]:>+8.2f} [{lo:>+7.2f}, {hi:>+7.2f}]   '
            f'{row["visual_ood_rate"]:.1%}/{row["text_ood_rate"]:.1%}'
        )
    lines.append("Partial-budget cache proxy; not a full visual Cut or runtime video acceptance.")
    args.matched_run.mkdir(parents=True, exist_ok=True)
    (args.matched_run / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.matched_run / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
