"""Tests for the Qwen2.5-VL LLaVA training launcher."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT_DIR / "train_qwen25vl_dflash_llava_68k.sh"


class LlavaLauncherTest(unittest.TestCase):
    def _run_capture(self, skip_preflight: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory)
            (artifact_root / "manifest.jsonl").touch()
            env = {
                **os.environ,
                "ARTIFACT_ROOT": str(artifact_root),
                "IMAGE_ROOT": str(artifact_root / "images"),
                "PYTHON_BIN": "/bin/echo",
                "SKIP_PREFLIGHT": skip_preflight,
                "TARGET_MODEL_PATH": str(artifact_root / "target"),
            }
            return subprocess.run(
                ["bash", str(LAUNCHER), "--phase", "capture"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_skip_preflight_goes_directly_to_capture(self):
        result = self._run_capture("1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Skipping LLaVA preflight", result.stdout)
        self.assertNotIn("preflight_llava_caption.py", result.stdout)
        self.assertIn("prepare_llava_caption_hidden_states.py", result.stdout)
        self.assertIn("--sglang-mem-fraction-static 0.4", result.stdout)
        self.assertIn("--batch-size 16", result.stdout)
        self.assertIn("--num-preprocess-workers 8", result.stdout)
        self.assertIn("--num-io-threads 8", result.stdout)

    def test_preflight_runs_by_default(self):
        result = self._run_capture("0")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("preflight_llava_caption.py", result.stdout)
        self.assertIn("prepare_llava_caption_hidden_states.py", result.stdout)

    def test_invalid_skip_preflight_value_is_rejected(self):
        result = self._run_capture("yes")

        self.assertEqual(result.returncode, 2)
        self.assertIn("SKIP_PREFLIGHT must be 0 or 1", result.stderr)

    def test_two_gpu_training_profile_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hidden_states").mkdir()
            phase1 = root / "phase1"
            phase1.touch()
            env = {
                **os.environ,
                "ARTIFACT_ROOT": str(root),
                "OUTPUT_ROOT": str(root / "outputs"),
                "PHASE1_CHECKPOINT": str(phase1),
                "PYTHON_BIN": "/bin/echo",
                "SPECFORGE_ATTENTION_BACKEND": "flex_attention",
                "SPECFORGE_DATALOADER_WORKERS": "12",
                "SPECFORGE_FSDP_SHARDING": "NO_SHARD",
                "SPECFORGE_GLOBAL_BATCH_SIZE": "64",
                "SPECFORGE_MICRO_BATCH_SIZE": "16",
                "SPECFORGE_OBJECTIVE_CHUNK_BLOCKS": "256",
                "SPECFORGE_USE_LIGER": "0",
                "TARGET_MODEL_PATH": str(root / "target"),
            }

            result = subprocess.run(
                ["bash", str(LAUNCHER), "--phase", "train", "--gpus", "2"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("micro/rank=16 accumulation=2 global_batch=64", result.stdout)
        self.assertIn("model.use_liger_kernel=false", result.stdout)
        self.assertIn("data.dataloader_num_workers=12", result.stdout)
        self.assertIn("training.fsdp_sharding=NO_SHARD", result.stdout)
        self.assertIn("training.attention_backend=flex_attention", result.stdout)
        self.assertIn("training.objective_chunk_blocks=256", result.stdout)


if __name__ == "__main__":
    unittest.main()
