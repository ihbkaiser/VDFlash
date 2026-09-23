"""Paired, checkpoint-specific representation test for DFlash H1.1.

The reference and evaluation NPZ files are produced by run_dflash_h11.py.
All thresholds are fitted on held-out Full training examples, never on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SITES = ("attn_out", "layer_out")


def fit_reference(train: np.ndarray, calibration: np.ndarray, *, components: int = 32):
    """Fit a regularized low-rank training reference and a held-out cutoff."""

    if train.ndim != 2 or calibration.ndim != 2 or train.shape[1] != calibration.shape[1]:
        raise ValueError("train/calibration must have matching [samples, width] shapes")
    if train.shape[0] < 20 or calibration.shape[0] < 10:
        raise ValueError("need at least 20 fit and 10 calibration examples")
    mean = train.mean(axis=0, dtype=np.float64)
    centered = train.astype(np.float64) - mean
    # With only a few hundred samples, full-width covariance is singular.
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(components, train.shape[0] // 4, train.shape[0] - 1)
    basis = vh[:rank].T
    variances = singular_values[:rank] ** 2 / (train.shape[0] - 1)
    ridge = max(float(np.median(variances)) * 0.1, 1e-8)

    def distance(values: np.ndarray) -> np.ndarray:
        projected = (values.astype(np.float64) - mean) @ basis
        return np.sum(projected**2 / (variances + ridge), axis=1)

    cutoff = float(np.quantile(distance(calibration), 0.95))
    return distance, cutoff


def _paired_behavior(path: Path, sample_ids: list[str]) -> dict[str, dict]:
    paired: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cache_only = row.get("verification_mode") == "cached_teacher_tokens"
            if row.get("status") != "ok" or (not cache_only and not row.get("lossless")):
                raise ValueError("evaluation contains a failed or nonlossless row")
            sid, condition = str(row["sample_id"]), row["visual_condition"]
            if condition not in {"full", "cut"} or condition in paired.setdefault(sid, {}):
                raise ValueError("duplicate or unexpected evaluation condition")
            rounds = row.get("acceptance_rounds", [])
            if not rounds:
                raise ValueError("evaluation has no draft round")
            paired[sid][condition] = row
    if set(paired) != set(sample_ids) or any(set(pair) != {"full", "cut"} for pair in paired.values()):
        raise ValueError("evaluation reports are not paired with captured vectors")
    result = {}
    for sid, pair in paired.items():
        full, cut = pair["full"], pair["cut"]
        if full.get("verification_mode") != cut.get("verification_mode"):
            raise ValueError("Full/Cut use different verification modes")
        cache_only = full.get("verification_mode") == "cached_teacher_tokens"
        if (full["target_input_fingerprint"] != cut["target_input_fingerprint"]
                or full["target_output_hash"] != cut["target_output_hash"]):
            raise ValueError("Full/Cut do not share the target prompt and continuation")
        result[sid] = {
            "verification_mode": full.get("verification_mode", "target_video_decode"),
            "first_block_delta_accepted_proposals": (
                int(cut["acceptance_rounds"][0]["matched_proposals"])
                - int(full["acceptance_rounds"][0]["matched_proposals"])
            ),
        }
        if "h11_attention" in full and "h11_attention" in cut:
            if set(full["h11_attention"]) != set(cut["h11_attention"]):
                raise ValueError("Full/Cut attention layers do not match")
            result[sid]["attention"] = {
                layer: {
                    metric: {
                        "full": float(full["h11_attention"][layer][metric]),
                        "cut": float(cut["h11_attention"][layer][metric]),
                    }
                    for metric in ("text_mass", "top5_text_mass_conditional")
                }
                for layer in full["h11_attention"]
            }
        if not cache_only:
            result[sid]["tau_effective_delta"] = (
                float(cut["accepted_effective_tokens"])
                - float(full["accepted_effective_tokens"])
            )
    return result


def analyze(reference: Path, evaluation: Path, *, components: int = 32,
            reports: Path | None = None) -> dict:
    with np.load(reference, allow_pickle=False) as ref, np.load(evaluation, allow_pickle=False) as ev:
        train = ref["vectors"]
        full = ev["full"]
        cut = ev["cut"]
        ref_checkpoint = str(ref["checkpoint"].item())
        eval_checkpoint = str(ev["checkpoint"].item())
        sample_ids = ev["sample_ids"].tolist()
    if ref_checkpoint != eval_checkpoint:
        raise ValueError("reference and evaluation use different checkpoint paths")
    if train.ndim != 4 or full.ndim != 4 or train.shape[1:] != full.shape[1:] or full.shape != cut.shape:
        raise ValueError("expected matching [samples, layers, sites, width] arrays")
    if train.shape[2] != len(SITES) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("invalid site axis or duplicate evaluation sample IDs")
    if not (np.isfinite(train).all() and np.isfinite(full).all() and np.isfinite(cut).all()):
        raise ValueError("representation contains nonfinite values")
    behavior = _paired_behavior(reports, sample_ids) if reports is not None else None
    # Reference records are shuffled before capture. Calibration has no shared
    # sample with fitting data; 25% is reserved for the cutoff.
    fit_count = int(train.shape[0] * 0.75)
    per_sample = []
    results = []
    for layer in range(train.shape[1]):
        for site_index, site in enumerate(SITES):
            distance, cutoff = fit_reference(
                train[:fit_count, layer, site_index],
                train[fit_count:, layer, site_index],
                components=components,
            )
            full_d = distance(full[:, layer, site_index])
            cut_d = distance(cut[:, layer, site_index])
            delta = cut_d - full_d
            # Resample complete samples, preserving their Full/Cut pairing.
            rng = np.random.default_rng(42 + layer * len(SITES) + site_index)
            indices = rng.integers(0, len(delta), size=(2000, len(delta)))
            ci = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            summary = {
                "layer": layer,
                "site": site,
                "n_train_fit": fit_count,
                "n_train_calibration": len(train) - fit_count,
                "n_test": len(delta),
                "calibration_p95": cutoff,
                "mean_full_distance": float(full_d.mean()),
                "mean_cut_distance": float(cut_d.mean()),
                "mean_paired_delta": float(delta.mean()),
                "paired_delta_ci95": ci.tolist(),
                "full_ood_rate": float((full_d > cutoff).mean()),
                "cut_ood_rate": float((cut_d > cutoff).mean()),
            }
            if behavior is not None:
                modes = {behavior[sid]["verification_mode"] for sid in sample_ids}
                if len(modes) != 1:
                    raise ValueError("evaluation mixes cache and video verification")
                first_delta = np.asarray([
                    behavior[sid]["first_block_delta_accepted_proposals"] for sid in sample_ids
                ], dtype=np.float64)
                summary["mean_first_block_acceptance_delta"] = float(first_delta.mean())
                summary["verification_mode"] = modes.pop()
                if summary["verification_mode"] != "cached_teacher_tokens":
                    summary["mean_tau_effective_delta"] = float(np.mean([
                        behavior[sid]["tau_effective_delta"] for sid in sample_ids
                    ]))
                if site == "attn_out" and all(
                    str(layer) in behavior[sid].get("attention", {}) for sid in sample_ids
                ):
                    for metric in ("text_mass", "top5_text_mass_conditional"):
                        full_metric = np.asarray([
                            behavior[sid]["attention"][str(layer)][metric]["full"]
                            for sid in sample_ids
                        ])
                        cut_metric = np.asarray([
                            behavior[sid]["attention"][str(layer)][metric]["cut"]
                            for sid in sample_ids
                        ])
                        summary[f"mean_full_{metric}"] = float(full_metric.mean())
                        summary[f"mean_cut_{metric}"] = float(cut_metric.mean())
                        summary[f"mean_paired_{metric}_delta"] = float(
                            (cut_metric - full_metric).mean()
                        )
                # A correlation requires nonconstant distances and acceptance;
                # no correlation is reported for an underpowered constant pilot.
                if len(delta) >= 10 and delta.std() > 0 and first_delta.std() > 0:
                    summary["corr_distance_vs_first_block_acceptance"] = float(
                        np.corrcoef(delta, first_delta)[0, 1]
                    )
            results.append(summary)
            for sample_id, d_full, d_cut in zip(sample_ids, full_d, cut_d):
                row = {
                    "sample_id": sample_id, "layer": layer, "site": site,
                    "distance_full": float(d_full), "distance_cut": float(d_cut),
                    "delta": float(d_cut - d_full), "cutoff": cutoff,
                }
                if behavior is not None:
                    row.update(behavior[sample_id])
                per_sample.append(row)
    return {"checkpoint": ref_checkpoint, "summary": results, "per_sample": per_sample}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reports", type=Path, help="Paired Full/Cut JSONL from evaluate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components", type=int, default=32)
    args = parser.parse_args(argv)
    if args.components < 1:
        parser.error("--components must be positive")
    result = analyze(args.reference, args.evaluation, components=args.components,
                     reports=args.reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"checkpoint": result["checkpoint"], "layers": result["summary"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "paired_distances.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_sample"]:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"layers": len(result["summary"]), "samples": len(result["per_sample"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
