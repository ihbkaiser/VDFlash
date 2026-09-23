"""Check whether H1.1 PCA misses shifts and whether shifts track token loss.

Runs on completed H1.1 NPZ/JSONL files without loading target, draft, or CUDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_dflash_h11 import SITES, _paired_behavior


def fit_residual_reference(train: np.ndarray, calibration: np.ndarray, components: int):
    """Fit top training directions and calibrate distance outside their span."""
    if train.ndim != 2 or calibration.ndim != 2 or train.shape[1] != calibration.shape[1]:
        raise ValueError("reference arrays must be [samples, same_width]")
    if len(train) < 20 or len(calibration) < 10 or components < 1:
        raise ValueError("need >=20 fit, >=10 calibration samples, and positive components")
    mean = train.mean(axis=0, dtype=np.float64)
    centered = train.astype(np.float64) - mean
    # The Gram matrix is smaller than a full-width SVD for this 3B draft.
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    descending = np.argsort(eigenvalues)[::-1]
    selected = [index for index in descending[:components] if eigenvalues[index] > 1e-10]
    if not selected:
        raise ValueError("training representations have zero variance")
    basis = centered.T @ eigenvectors[:, selected]
    basis /= np.sqrt(eigenvalues[selected])[None, :]

    def distance(values: np.ndarray) -> np.ndarray:
        centered_values = values.astype(np.float64) - mean
        projected = centered_values @ basis
        squared = np.einsum("ij,ij->i", centered_values, centered_values)
        squared -= np.einsum("ij,ij->i", projected, projected)
        return np.sqrt(np.maximum(squared, 0.0))

    cutoff = float(np.quantile(distance(calibration), 0.95))
    return distance, cutoff


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for tied acceptance counts and visual-token counts."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def rank_correlation(values: np.ndarray, losses: np.ndarray,
                     controls: np.ndarray | None = None) -> float | None:
    """Spearman association; optional adjustment for visual count and anchor."""
    if len(values) < 10:
        return None
    x, y = _ranks(values), _ranks(losses)
    if controls is not None:
        covariates = np.column_stack([
            np.ones(len(x)), *(_ranks(controls[:, index]) for index in range(controls.shape[1]))
        ])
        x = x - covariates @ np.linalg.lstsq(covariates, x, rcond=None)[0]
        y = y - covariates @ np.linalg.lstsq(covariates, y, rcond=None)[0]
    if x.std() < 1e-10 or y.std() < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _cached_controls(report_file: Path, sample_ids: list[str]) -> np.ndarray | None:
    full_rows = {}
    with report_file.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("visual_condition") == "full":
                full_rows[str(row["sample_id"])] = row
    if set(full_rows) != set(sample_ids):
        raise ValueError("Full report rows do not match paired sample IDs")
    if not all("visual_count_before_cut" in row and "anchor" in row
               for row in full_rows.values()):
        return None
    return np.asarray([
        [full_rows[sid]["visual_count_before_cut"], full_rows[sid]["anchor"]]
        for sid in sample_ids
    ], dtype=np.float64)


def analyze_deep(run_dir: Path, *, components: int = 32, bootstrap: int = 1000) -> dict:
    with np.load(run_dir / "train_full.npz", allow_pickle=False) as ref, \
            np.load(run_dir / "test_full_cut.npz", allow_pickle=False) as ev:
        train = ref["vectors"]
        full, cut = ev["full"], ev["cut"]
        sample_ids = [str(item) for item in ev["sample_ids"].tolist()]
        ref_checkpoint = str(ref["checkpoint"].item())
        eval_checkpoint = str(ev["checkpoint"].item())
    if ref_checkpoint != eval_checkpoint:
        raise ValueError("reference and evaluation use different checkpoints")
    if train.ndim != 4 or full.shape != cut.shape or full.ndim != 4 or train.shape[1:] != full.shape[1:]:
        raise ValueError("reference and Full/Cut vectors have incompatible shapes")
    if train.shape[2] != len(SITES) or len(sample_ids) != len(set(sample_ids)) or len(full) != len(sample_ids):
        raise ValueError("invalid sites or sample IDs")
    if not all(np.isfinite(array).all() for array in (train, full, cut)):
        raise ValueError("nonfinite representations")
    report_file = run_dir / "test_full_cut.jsonl"
    behavior = _paired_behavior(report_file, sample_ids)
    controls = _cached_controls(report_file, sample_ids)
    original_path = run_dir / "analysis" / "summary.json"
    original_rows = {}
    if original_path.is_file():
        original = json.loads(original_path.read_text(encoding="utf-8"))
        if original["checkpoint"] != ref_checkpoint:
            raise ValueError("original PCA summary uses a different checkpoint")
        original_rows = {(row["layer"], row["site"]): row for row in original["layers"]}
    token_loss = np.asarray([
        -behavior[sid]["first_block_delta_accepted_proposals"] for sid in sample_ids
    ], dtype=np.float64)
    fit_count = int(len(train) * 0.75)
    if len(train) - fit_count < 10:
        raise ValueError("not enough calibration references")
    summary, per_sample = [], []
    for layer in range(train.shape[1]):
        for site_index, site in enumerate(SITES):
            print(f"[H1.1 deep] layer {layer} {site}", flush=True)
            distance, cutoff = fit_residual_reference(
                train[:fit_count, layer, site_index],
                train[fit_count:, layer, site_index], components,
            )
            full_distance = distance(full[:, layer, site_index])
            cut_distance = distance(cut[:, layer, site_index])
            delta = cut_distance - full_distance
            rng = np.random.default_rng(991 + layer * len(SITES) + site_index)
            indices = rng.integers(0, len(delta), size=(bootstrap, len(delta)))
            ci = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist()
            rho = rank_correlation(delta, token_loss)
            partial = rank_correlation(delta, token_loss, controls) if controls is not None else None
            associations = [rank_correlation(delta[idx], token_loss[idx]) for idx in indices]
            valid_associations = [value for value in associations if value is not None]
            rho_ci = (np.quantile(valid_associations, [0.025, 0.975]).tolist()
                      if len(valid_associations) >= bootstrap // 2 else None)
            row = {
                "layer": layer, "site": site, "n_fit": fit_count,
                "n_calibration": len(train) - fit_count, "n_pairs": len(delta),
                "pca_components": components, "residual_calibration_p95": cutoff,
                "mean_full_residual": float(full_distance.mean()),
                "mean_cut_residual": float(cut_distance.mean()),
                "mean_residual_delta": float(delta.mean()),
                "residual_delta_ci95": ci,
                "full_residual_ood_rate": float((full_distance > cutoff).mean()),
                "cut_residual_ood_rate": float((cut_distance > cutoff).mean()),
                "rho_residual_delta_vs_token_loss": rho,
                "rho_ci95": rho_ci,
                "rho_adjusted_for_visual_count_and_anchor": partial,
            }
            if (layer, site) in original_rows:
                row["original_pca_delta"] = original_rows[layer, site]["mean_paired_delta"]
                row["original_pca_full_ood_rate"] = original_rows[layer, site]["full_ood_rate"]
                row["original_pca_cut_ood_rate"] = original_rows[layer, site]["cut_ood_rate"]
            summary.append(row)
            for index, sample_id in enumerate(sample_ids):
                per_sample.append({
                    "sample_id": sample_id, "layer": layer, "site": site,
                    "residual_full": float(full_distance[index]),
                    "residual_cut": float(cut_distance[index]),
                    "residual_delta": float(delta[index]),
                    "token_loss": int(token_loss[index]),
                })
    return {"checkpoint": ref_checkpoint, "layers": summary, "per_sample": per_sample,
            "verification_mode": next(iter(behavior.values()))["verification_mode"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed results/h11_<timestamp> directory")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.components < 1 or args.bootstrap < 100:
        parser.error("components must be positive and bootstrap must be >=100")
    result = analyze_deep(args.run_dir, components=args.components,
                          bootstrap=args.bootstrap)
    output_dir = args.run_dir / "analysis_deep"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"checkpoint": result["checkpoint"], "verification_mode": result["verification_mode"],
                    "layers": result["layers"]}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "paired_residuals.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_sample"]:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    lines = [
        "Residual distance outside the top training PCA directions (Cut minus Full)",
        "Positive delta: Cut farther away in directions omitted by the original 32-D PCA score.",
        "Positive rho: greater residual shift is associated with more lost teacher matches.",
        "Layer  Site       N   PCA delta   Residual delta [95% CI]   OOD Full/Cut   rho [95% CI]   adjusted rho",
    ]
    for row in result["layers"]:
        lo, hi = row["residual_delta_ci95"]
        corr, interval, partial = (row["rho_residual_delta_vs_token_loss"],
                                   row["rho_ci95"], row["rho_adjusted_for_visual_count_and_anchor"])
        rho_text = "n/a" if corr is None else f"{corr:+.2f} [{interval[0]:+.2f},{interval[1]:+.2f}]" if interval else f"{corr:+.2f}"
        partial_text = "n/a" if partial is None else f"{partial:+.2f}"
        pca_text = (f'{row["original_pca_delta"]:+.2f}'
                    if "original_pca_delta" in row else "n/a")
        lines.append(
            f'{row["layer"]:>5}  {row["site"]:<9} {row["n_pairs"]:>3}  '
            f'{pca_text:>9}  '
            f'{row["mean_residual_delta"]:>8.2f} [{lo:>7.2f},{hi:>7.2f}]  '
            f'{row["full_residual_ood_rate"]:.1%}/{row["cut_residual_ood_rate"]:.1%}    '
            f'{rho_text:<20} {partial_text}'
        )
    lines.append("Adjusted rho controls for visual-token count and anchor position; associations are not causal.")
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    print(f"Saved {output_dir / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
