"""Verify that the follow-up detects shifts invisible to the original PCA score."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11 import fit_reference
from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11_deep import analyze_deep


class DeepH11AnalysisTests(unittest.TestCase):
    def test_off_subspace_shift_and_token_loss_association(self):
        rng = np.random.default_rng(47)
        width, n_train, n_pairs = 16, 80, 24
        train = rng.normal(size=(n_train, width))
        mean = train[:60].mean(axis=0)
        _, _, vh = np.linalg.svd(train[:60] - mean, full_matrices=False)
        off_subspace = vh[3]
        full = np.tile(mean, (n_pairs, 1)) + rng.normal(0, 0.03, (n_pairs, width))
        strengths = np.linspace(1.0, 8.0, n_pairs)
        cut = full + strengths[:, None] * off_subspace
        original_distance, _ = fit_reference(train[:60], train[60:], components=3)
        self.assertLess(np.abs(original_distance(cut) - original_distance(full)).max(), 1e-8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors = np.tile(train[:, None, None, :], (1, 2, 2, 1)).astype("float32")
            full_vectors = np.tile(full[:, None, None, :], (1, 2, 2, 1)).astype("float32")
            cut_vectors = np.tile(cut[:, None, None, :], (1, 2, 2, 1)).astype("float32")
            np.savez_compressed(root / "train_full.npz", vectors=vectors, checkpoint="checkpoint")
            np.savez_compressed(
                root / "test_full_cut.npz", full=full_vectors, cut=cut_vectors,
                sample_ids=np.asarray([f"pair{i}" for i in range(n_pairs)]),
                checkpoint="checkpoint",
            )
            (root / "analysis").mkdir()
            (root / "analysis" / "summary.json").write_text(json.dumps({
                "checkpoint": "checkpoint",
                "layers": [{"layer": 0, "site": "attn_out", "mean_paired_delta": 0.0,
                            "full_ood_rate": 0.05, "cut_ood_rate": 0.05}],
            }), encoding="utf-8")
            with (root / "test_full_cut.jsonl").open("w", encoding="utf-8") as handle:
                for index, strength in enumerate(strengths):
                    for condition in ("full", "cut"):
                        lost = round(strength) if condition == "cut" else 0
                        handle.write(json.dumps({
                            "sample_id": f"pair{index}", "visual_condition": condition,
                            "status": "ok", "verification_mode": "cached_teacher_tokens",
                            "target_input_fingerprint": f"prompt{index}",
                            "target_output_hash": f"teacher{index}",
                            "visual_count_before_cut": 6 + index % 3,
                            "anchor": 100 + index,
                            "acceptance_rounds": [{"matched_proposals": 12 - lost}],
                        }) + "\n")
            result = analyze_deep(root, components=3, bootstrap=100)
            row = result["layers"][0]
            self.assertEqual(row["original_pca_delta"], 0.0)
            self.assertGreater(row["residual_delta_ci95"][0], 0.0)
            self.assertGreater(row["cut_residual_ood_rate"], row["full_residual_ood_rate"])
            self.assertGreater(row["rho_residual_delta_vs_token_loss"], 0.8)


if __name__ == "__main__":
    unittest.main()
