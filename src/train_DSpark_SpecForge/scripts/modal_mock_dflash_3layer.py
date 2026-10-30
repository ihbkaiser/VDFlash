"""Low-cost Modal smoke test for the Qwen2.5-VL three-layer launch plan.

This test does not load Qwen weights or start training. It validates all four
layer presets against the real config resolver, runs the launcher regression
tests, then performs one tiny two-GPU DDP step on a Modal T4 pair.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import modal


APP_NAME = "vdflash-dflash-3layer-mock"
SPEC_DIR = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/VDFlash/src/train_Dflash_SpecForge"
LAYER_SETS = {
    "high_only": "26,28,30,32,34",
    "middle_high": "17,19,25,33,34",
    "middle_high_2": "17,28,30,32,34",
    "shallow_high": "1,28,30,32,34",
}


def _ignore_python_cache(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bash")
    .pip_install("pytest", "torch")
    .add_local_dir(
        SPEC_DIR / "specforge",
        remote_path=f"{REMOTE_ROOT}/specforge",
        copy=True,
        ignore=_ignore_python_cache,
    )
    .add_local_dir(
        SPEC_DIR / "scripts",
        remote_path=f"{REMOTE_ROOT}/scripts",
        copy=True,
        ignore=_ignore_python_cache,
    )
    .add_local_dir(
        SPEC_DIR / "configs",
        remote_path=f"{REMOTE_ROOT}/configs",
        copy=True,
        ignore=_ignore_python_cache,
    )
    .add_local_file(
        SPEC_DIR / "train_qwen25vl_dflash_3layer.sh",
        remote_path=f"{REMOTE_ROOT}/train_qwen25vl_dflash_3layer.sh",
        copy=True,
    )
    .add_local_file(
        SPEC_DIR / "train_qwen25vl_dflash_sharegpt_68k.sh",
        remote_path=f"{REMOTE_ROOT}/train_qwen25vl_dflash_sharegpt_68k.sh",
        copy=True,
    )
    .add_local_file(
        SPEC_DIR / "train_qwen25vl_dflash_llava_68k.sh",
        remote_path=f"{REMOTE_ROOT}/train_qwen25vl_dflash_llava_68k.sh",
        copy=True,
    )
    .add_local_file(
        SPEC_DIR / "train_qwen25vl_dflash_layers_25_27_29_31_33.sh",
        remote_path=f"{REMOTE_ROOT}/train_qwen25vl_dflash_layers_25_27_29_31_33.sh",
        copy=True,
    )
)

for relative_path in (
    "tests/test_scripts/test_dflash_3layer_launcher.py",
    "tests/test_scripts/test_two_phase_layer_launcher.py",
    "tests/test_hidden_state_layers.py",
    "tests/test_offline_capture/test_qwen25vl_text_capture.py",
):
    source_path = SPEC_DIR / relative_path
    image = image.add_local_file(
        source_path,
        remote_path=f"{REMOTE_ROOT}/{relative_path}",
        copy=True,
    )

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4:2",
    cpu=4,
    memory=8192,
    timeout=300,
)
def run_modal_mock() -> dict[str, object]:
    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(f"expected two visible GPUs, found {gpu_count}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    print(f"[modal-mock] GPUs={gpu_names}", flush=True)

    remote_root = Path(REMOTE_ROOT)
    root = str(remote_root)
    launcher = str(remote_root / "train_qwen25vl_dflash_3layer.sh")
    resolver = str(remote_root / "scripts/resolve_dflash_config.py")
    source_config = str(remote_root / "configs/qwen2.5-vl-3b-dflash.json")
    env = {**os.environ, "PYTHONPATH": root, "SPECFORGE_SKIP_GPU_CHECK": "1"}

    for name, target_ids in LAYER_SETS.items():
        config_result = subprocess.run(
            ["bash", launcher, "--set", name, "--gpu-ids", "0,1", "--print-config"],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        if f"TARGET_LAYER_IDS={target_ids}" not in config_result.stdout:
            raise AssertionError(f"launcher resolved the wrong layer IDs for {name}")
        if "SPECFORGE_DFLASH_DEPTH=3" not in config_result.stdout:
            raise AssertionError(f"launcher resolved the wrong draft depth for {name}")
        if "SPECFORGE_GPUS=2" not in config_result.stdout:
            raise AssertionError(f"launcher did not resolve two GPUs for {name}")

        with tempfile.TemporaryDirectory(prefix=f"dflash-{name}-") as temp_dir:
            for phase in ("phase1", "phase2"):
                output_path = Path(temp_dir) / f"draft_config_{phase}.json"
                subprocess.run(
                    [
                        sys.executable,
                        resolver,
                        "--input",
                        source_config,
                        "--output",
                        str(output_path),
                        "--phase",
                        phase,
                        "--target-layer-ids",
                        target_ids,
                        "--draft-num-hidden-layers",
                        "3",
                    ],
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                resolved = json.loads(output_path.read_text(encoding="utf-8"))
                actual_ids = resolved["dflash_config"]["target_layer_ids"]
                if actual_ids != [int(value) for value in target_ids.split(",")]:
                    raise AssertionError(
                        f"resolver changed {phase} layer IDs for {name}: {actual_ids}"
                    )
                if resolved["num_hidden_layers"] != 3:
                    raise AssertionError(
                        f"resolver did not make a 3-layer {phase} draft for {name}"
                    )
        print(
            f"[modal-mock] phase1/phase2 configs passed: {name} -> {target_ids}",
            flush=True,
        )

    test_paths = [
        str(remote_root / "tests/test_scripts/test_dflash_3layer_launcher.py"),
        str(remote_root / "tests/test_scripts/test_two_phase_layer_launcher.py"),
        str(remote_root / "tests/test_hidden_state_layers.py"),
        str(remote_root / "tests/test_offline_capture/test_qwen25vl_text_capture.py"),
    ]
    legacy_launcher = remote_root / "train_qwen25vl_dflash_layers_25_27_29_31_33.sh"
    legacy_text = legacy_launcher.read_text(encoding="utf-8")
    legacy_launcher.write_text(
        legacy_text.replace("\r\n", "\n"), encoding="utf-8", newline="\n"
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *test_paths],
        cwd=root,
        env=env,
        check=True,
    )

    worker_code = r'''from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(2718)
    model = DistributedDataParallel(
        torch.nn.Linear(4, 2).to(device), device_ids=[rank]
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features = torch.full((4, 4), float(rank + 1), device=device)
    loss = model(features).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("mock DDP loss is not finite")
    loss.backward()
    optimizer.step()
    parameters = [parameter.detach().clone() for parameter in model.module.parameters()]
    for parameter in parameters:
        reference = parameter.clone()
        dist.broadcast(reference, src=0)
        if not torch.allclose(parameter, reference, atol=1e-6, rtol=1e-6):
            raise RuntimeError("DDP parameters diverged between GPU ranks")
    print(f"[ddp-mock] rank={rank} loss={loss.item():.6f} device={device}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''
    with tempfile.TemporaryDirectory(prefix="dflash-ddp-") as temp_dir:
        worker = Path(temp_dir) / "two_gpu_ddp_mock.py"
        worker.write_text(worker_code, encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
            ],
            cwd=root,
            env=env,
            check=True,
            timeout=150,
        )

    print("[modal-mock] all four config checks and two-GPU DDP step passed", flush=True)
    return {
        "gpu_names": gpu_names,
        "layer_sets_checked": len(LAYER_SETS),
        "phase_configs_checked": len(LAYER_SETS) * 2,
        "ddp_step": "passed",
    }


@app.local_entrypoint()
def main() -> None:
    result = run_modal_mock.remote()
    print(json.dumps(result, sort_keys=True))
