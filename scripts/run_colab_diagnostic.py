"""Bootstrap and run the diagnostic audit on Colab."""

import os
import subprocess
import sys
from pathlib import Path

# 1. Install required packages
print("[Bootstrap] Installing decord and qwen-vl-utils ...", flush=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "decord", "qwen-vl-utils", "accelerate"],
    check=True,
)

# 2. Set environment variables and sys.path
sys.path.insert(0, "/content/vdflash")
os.environ["PYTHONPATH"] = "/content/vdflash:" + os.environ.get("PYTHONPATH", "")
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

# 3. Create dummy externals directory so workspace root is recognized
(Path("/content/vdflash/externals")).mkdir(parents=True, exist_ok=True)

# 4. Run the diagnostic audit
from src.analyze.Validate_Sparrow_hypothesises.audit_dflash_textstate_matched_path import (
    run_diagnostic_audit,
)

print("[Bootstrap] Running diagnostic audit ...", flush=True)
results = run_diagnostic_audit(
    target_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    schedule_path=Path("/content/vdflash/results/dflash_textstate_intervention/pilot12_3k_20261106/design/block_schedule.json"),
    video_root=Path("/content/vdflash/dataset/VideoDetailCaption"),
    calibration_path=Path("/content/vdflash/results/dflash_distribution_replay/vdc50_balanced_distribution_replay_full_forced_trajectory_20261102/calibration/calibration.jsonl"),
    sample_ids=["v_1YU4MSK80cQ", "v_ej9TkJVjK9M", "v_zRqJZ0u6f88", "v_aLb79wtvxoQ"],
    device_str="cuda:0",
    output_dir=Path("/content/vdflash/results/audit_output"),
)
print("[Bootstrap] Completed diagnostic audit!", flush=True)
