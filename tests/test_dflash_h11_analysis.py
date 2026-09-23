"""Statistical gates for the paired DFlash H1.1 probe."""

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_h11 import analyze


class H11AnalysisTests(unittest.TestCase):
    def test_paired_shift_and_checkpoint_contract(self):
        rng = np.random.default_rng(7)
        training = rng.normal(size=(80, 2, 2, 12)).astype("float32")
        baseline = rng.normal(size=(20, 2, 2, 12)).astype("float32")
        intervention = baseline + 8.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref = root / "train.npz"
            test = root / "test.npz"
            np.savez_compressed(ref, vectors=training, checkpoint="/checkpoints/phase2")
            np.savez_compressed(
                test, full=baseline, cut=intervention,
                sample_ids=np.asarray([f"s{i}" for i in range(20)]),
                checkpoint="/checkpoints/phase2",
            )
            result = analyze(ref, test, components=3)
            self.assertEqual(len(result["summary"]), 4)
            self.assertTrue(all(row["mean_paired_delta"] > 0 for row in result["summary"]))
            self.assertTrue(all(row["cut_ood_rate"] > row["full_ood_rate"] for row in result["summary"]))
            reports = root / "reports.jsonl"
            with reports.open("w", encoding="utf-8") as handle:
                for i in range(20):
                    for condition, accepted in (("full", 2), ("cut", 1)):
                        handle.write(json.dumps({
                            "sample_id": f"s{i}", "visual_condition": condition,
                            "status": "ok", "lossless": True,
                            "target_input_fingerprint": f"prompt{i}",
                            "target_output_hash": f"output{i}",
                            "acceptance_rounds": [{"matched_proposals": accepted}],
                            "accepted_effective_tokens": float(accepted + 1),
                        }) + "\n")
            with_reports = analyze(ref, test, components=3, reports=reports)
            self.assertEqual(with_reports["summary"][0]["mean_first_block_acceptance_delta"], -1.0)
            self.assertEqual(with_reports["per_sample"][0]["tau_effective_delta"], -1.0)
            with reports.open("w", encoding="utf-8") as handle:
                for i in range(20):
                    for condition, accepted in (("full", 2), ("cut", 1)):
                        handle.write(json.dumps({
                            "sample_id": f"s{i}", "visual_condition": condition,
                            "status": "ok", "verification_mode": "cached_teacher_tokens",
                            "target_input_fingerprint": f"prompt{i}",
                            "target_output_hash": f"cached{i}",
                            "acceptance_rounds": [{"matched_proposals": accepted}],
                            "h11_attention": {
                                str(layer): {
                                    "text_mass": 0.7 if condition == "full" else 1.0,
                                    "top5_text_mass_conditional": 0.4 if condition == "full" else 0.5,
                                }
                                for layer in range(2)
                            },
                        }) + "\n")
            cached = analyze(ref, test, components=3, reports=reports)
            self.assertEqual(cached["summary"][0]["verification_mode"], "cached_teacher_tokens")
            self.assertNotIn("mean_tau_effective_delta", cached["summary"][0])
            self.assertAlmostEqual(cached["summary"][0]["mean_paired_text_mass_delta"], 0.3)
            self.assertAlmostEqual(
                cached["summary"][0]["mean_paired_top5_text_mass_conditional_delta"], 0.1
            )
            self.assertNotIn("mean_paired_text_mass_delta", cached["summary"][1])
            np.savez_compressed(ref, vectors=training, checkpoint="/checkpoints/other")
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                analyze(ref, test)


if __name__ == "__main__":
    unittest.main()
