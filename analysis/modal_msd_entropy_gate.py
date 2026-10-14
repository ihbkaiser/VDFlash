"""Run the MSD Qwen2-VL entropy/visual-necessity diagnostic on Modal L4."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import modal


MODULE_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in MODULE_PATH.parents if (parent / "externals" / "MSD").is_dir()),
    Path("/root/VDFlash"),
)
MSD_ROOT = REPO_ROOT / "externals" / "MSD"
ANALYSIS_ROOT = REPO_ROOT / "analysis"
APP_NAME = "vdflash-msd-qwen2vl-entropy-gate"
HF_VOLUME_NAME = os.environ.get("MODAL_HF_VOLUME", "qresafe-awq-hf-cache")
RESULTS_VOLUME_NAME = os.environ.get(
    "MODAL_RESULTS_VOLUME", "msd-qwen2vl-entropy-gate-results"
)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "datasets==2.16.1",
        "evaluate>=0.4.0",
        "jsonlines",
        "numexpr",
        "numpy==1.26.4",
        "openai",
        "peft>=0.2.0",
        "pytablewriter",
        "sqlitedict==2.1.0",
        "qwen-vl-utils==0.0.11",
        "decord",
        "av",
        "pillow",
        "loguru",
        "safetensors",
        "sentencepiece==0.1.99",
        "protobuf==3.20.0",
        "scikit-learn",
        "matplotlib",
        "sacrebleu>=1.5.0",
        "ftfy",
        "hf-transfer",
        "opencv-python-headless",
        "nltk",
        "yt-dlp",
        "tqdm-multiprocess",
        "transformers-stream-generator",
        "zstandard",
        "openpyxl",
        "tenacity==8.3.0",
        "tiktoken",
        "pydantic",
        "zss",
        "sentence-transformers",
        "timm",
        "einops",
        "pycocoevalcap",
        "sympy",
        "pyyaml",
    )
    .add_local_dir(MSD_ROOT, "/root/VDFlash/externals/MSD", copy=True)
    .add_local_dir(ANALYSIS_ROOT, "/root/VDFlash/analysis", copy=True)
    .env(
        {
            "PYTHONPATH": "/root/VDFlash/externals/MSD/EAGLE:/root/VDFlash/externals/MSD/lmms-eval:/root/VDFlash",
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "HF_DATASETS_CACHE": "/root/.cache/huggingface/datasets",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONHASHSEED": "42",
        }
    )
)

hf_volume = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    timeout=12 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/results": results_volume,
    },
)
def run_remote(
    limit: int = 2,
    output_dir: str = "/results/msd_entropy_gate_smoke",
    tasks: str = "chartqa,textvqa_val,coco2017_cap_val",
    anonymous_datasets: bool = False,
) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats_path = output / "draft_stats.jsonl"
    eval_path = output / "lmms_eval.json"
    if anonymous_datasets:
        # The selected public datasets are configured with token=True in the
        # vendored lmms-eval task files.  Remove only that explicit requirement
        # in the ephemeral container; do not modify the repository copy.
        task_files = {
            "chartqa": Path(
                "/root/VDFlash/externals/MSD/lmms-eval/lmms_eval/tasks/chartqa/chartqa.yaml"
            ),
            "textvqa_val": Path(
                "/root/VDFlash/externals/MSD/lmms-eval/lmms_eval/tasks/textvqa/"
                "_default_template_textvqa_yaml"
            ),
            "short_test": Path(
                "/root/VDFlash/externals/MSD/lmms-eval/lmms_eval/tasks/vdc/"
                "_default_template_yaml"
            ),
            "mvbench_action_sequence": Path(
                "/root/VDFlash/externals/MSD/lmms-eval/lmms_eval/tasks/mvbench/"
                "_default_template_yaml"
            ),
        }
        for task_name in tasks.split(","):
            task_file = task_files.get(task_name.strip())
            if task_file is None or not task_file.exists():
                continue
            contents = task_file.read_text(encoding="utf-8")
            contents = contents.replace("  token: True\n", "")
            task_file.write_text(contents, encoding="utf-8")

    if "short_test" in {task.strip() for task in tasks.split(",")}:
        # The VDC task's dataset cache can survive without the extracted video
        # directory.  For the one-example smoke path, materialize one cached
        # MP4 so the task can exercise the video model and draft instrumentation.
        vdc_video_dir = Path("/root/.cache/huggingface/vdc_test/Test_Videos")
        if not any(vdc_video_dir.glob("*")):
            snapshot_root = Path(
                "/root/.cache/huggingface/hub/datasets--wchai--lmms_VDC_test"
            )
            zip_paths = sorted(snapshot_root.rglob("videos_chunked_*.zip"))
            for zip_path in zip_paths:
                with zipfile.ZipFile(zip_path) as archive:
                    member = next(
                        (
                            name
                            for name in archive.namelist()
                            if name.lower().endswith((".mp4", ".mkv"))
                        ),
                        None,
                    )
                    if member is None:
                        continue
                    vdc_video_dir.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, (
                        vdc_video_dir / Path(member).name
                    ).open("wb") as target:
                        shutil.copyfileobj(source, target)
                    break
    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        "qwen2_vl_msd",
        "--model_args",
        "pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=200704,max_num_frames=1",
        "--msd_model_path",
        "lucylyn/MSD-Qwen2VL-7B-Instruct",
        "--tasks",
        tasks,
        "--limit",
        str(limit),
        "--batch_size",
        "1",
        "--gen_kwargs",
        "temperature=0",
        "--use_msd",
        "--msd_visual_mode",
        "full",
        "--msd_chain_mode",
        "--log_draft_stats",
        str(stats_path),
        "--seed",
        "42",
        "--predict_only",
        "--output_path",
        str(eval_path),
    ]
    with (output / "command.txt").open("w", encoding="utf-8") as handle:
        handle.write(" ".join(shlex.quote(part) for part in command) + "\n")
    try:
        with (output / "lmms_eval.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd="/root/VDFlash/externals/MSD/lmms-eval",
                env={**os.environ, "MSD_VDC_ALLOW_FALLBACK": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"lmms-eval failed with exit code {completed.returncode}")
        if not eval_path.exists():
            raise RuntimeError(
                "lmms-eval returned without producing its result file; "
                "see lmms_eval.log"
            )

        analysis = subprocess.run(
            [
                sys.executable,
                "/root/VDFlash/analysis/entropy_gate.py",
                "--input",
                str(stats_path),
                "--output-dir",
                str(output / "analysis"),
            ],
            cwd="/root/VDFlash",
            check=False,
        )
        if analysis.returncode:
            raise RuntimeError(f"entropy analysis failed with exit code {analysis.returncode}")
    finally:
        hf_volume.commit()
        results_volume.commit()
    print(f"MSD_ENTROPY_GATE_OUTPUT={output}")
    return 0


@app.local_entrypoint()
def main(args: str = "") -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--output-dir", default="/results/msd_entropy_gate_smoke"
    )
    parser.add_argument(
        "--tasks", default="chartqa,textvqa_val,coco2017_cap_val"
    )
    parser.add_argument("--anonymous-datasets", action="store_true")
    parsed = parser.parse_args(shlex.split(args))
    run_remote.remote(
        limit=parsed.limit,
        output_dir=parsed.output_dir,
        tasks=parsed.tasks,
        anonymous_datasets=parsed.anonymous_datasets,
    )

