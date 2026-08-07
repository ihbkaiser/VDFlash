from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
from typing import Any
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .config import DFlashTrainConfig
from .data import build_masked_blocks, make_anchor_generator, sample_anchor_positions
from .distributed import DistributedContext, initialize_distributed
from .losses import weighted_block_cross_entropy
from .model import DFLASH_IMPLEMENTATION_VERSION, DFlashVLMModel


class TeacherCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.index = json.loads((self.directory / "index.json").read_text())
        if self.metadata.get("format") != "video-dflash-teacher-cache-v2":
            raise ValueError(f"unsupported teacher cache format in {self.directory}")
        if len(self.index) != int(self.metadata["cached_count"]):
            raise ValueError("teacher cache index count does not match metadata")

    def load_example(self, index: int, device: torch.device) -> dict[str, Any]:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load teacher shards") from exc
        entry = self.index[index]
        prefix = entry["tensor_prefix"]
        values: dict[str, torch.Tensor] = {}
        with safe_open(str(self.directory / entry["shard"]), framework="pt", device="cpu") as reader:
            for name in (
                "input_ids",
                "position_ids",
                "context_hidden",
                "context_original_positions",
            ):
                values[name] = reader.get_tensor(f"{prefix}.{name}").to(device)
        return {**entry, **values}

    def load_token_io(self, device: torch.device, dtype: torch.dtype) -> "FrozenTargetTokenIO":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load cached target token I/O") from exc
        state = load_file(str(self.directory / self.metadata["static_token_io"]), device="cpu")
        embedding = state["token_embedding"].to(device=device, dtype=dtype)
        lm_head = state.get("lm_head")
        if lm_head is not None:
            lm_head = lm_head.to(device=device, dtype=dtype)
        del state
        return FrozenTargetTokenIO(embedding, lm_head)


class FrozenTargetTokenIO:
    """Only the frozen token embedding/LM-head rows, never the target transformer."""

    def __init__(self, embedding: torch.Tensor, lm_head: torch.Tensor | None):
        self.embedding_weight = embedding.detach().requires_grad_(False)
        self.lm_head_weight = (
            self.embedding_weight if lm_head is None else lm_head.detach().requires_grad_(False)
        )

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids.long(), self.embedding_weight)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.lm_head_weight)

    @property
    def requires_grad(self) -> bool:
        return self.embedding_weight.requires_grad or self.lm_head_weight.requires_grad


def _dtype(config: DFlashTrainConfig) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[
        config.mixed_precision
    ]


def _text_config(metadata: dict[str, Any]):
    try:
        from transformers import Qwen2_5_VLTextConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Transformers Qwen2.5-VL config is required for the draft") from exc
    return Qwen2_5_VLTextConfig.from_dict(metadata["target_text_config"])


def make_cached_draft_model(
    cache: TeacherCache,
    config: DFlashTrainConfig,
    *,
    device: torch.device,
) -> DFlashVLMModel:
    metadata = cache.metadata
    layer_ids = list(metadata["target_layer_ids"])
    requested_layers = list(config.selected_target_layers or layer_ids)
    if requested_layers != layer_ids:
        raise ValueError(f"teacher layer mismatch: cache={layer_ids}, config={requested_layers}")
    if config.num_target_features != len(layer_ids):
        raise ValueError("num_target_features does not match the teacher cache")
    model = DFlashVLMModel(
        _text_config(metadata),
        num_draft_layers=config.num_draft_layers,
        num_target_features=len(layer_ids),
        block_size=config.block_size,
        compile_flex_attention=config.compile_flex_attention,
    )
    model.target_layer_ids = layer_ids
    model.context_mode = config.context_mode
    model.mask_token_id = int(metadata["mask_token_id"])
    model.gradient_checkpointing = bool(config.gradient_checkpointing)
    return model.to(device=device, dtype=_dtype(config))


def _validate_cache_contract(cache: TeacherCache, config: DFlashTrainConfig) -> None:
    metadata = cache.metadata
    checks = {
        "target_model": config.target_model,
        "context_mode": config.context_mode,
        "num_target_features": config.num_target_features,
        "teacher_response_mode": config.teacher_response_mode,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"teacher cache mismatch for {key}: {metadata.get(key)!r} != {expected!r}")
    if int(metadata["max_seq_length"]) > config.max_seq_length:
        raise ValueError("training max_seq_length is shorter than cached teacher sequences")


def _autocast(device: torch.device, config: DFlashTrainConfig):
    if device.type != "cuda" or config.mixed_precision == "no":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_dtype(config))


def _train_or_eval_example(
    example: dict[str, Any],
    draft: DFlashVLMModel,
    token_io: FrozenTargetTokenIO,
    config: DFlashTrainConfig,
    *,
    epoch: int,
    anchor_chunk_size: int,
    backward_scale: float,
    backward: bool,
) -> dict[str, float]:
    input_ids = example["input_ids"].long().view(1, -1)
    position_ids = example["position_ids"].long().unsqueeze(1)
    context_original = example["context_original_positions"].long()
    context_positions = position_ids[:, :, context_original]
    context_hidden = example["context_hidden"].unsqueeze(0).clone()
    generator = make_anchor_generator(
        config.seed,
        epoch,
        str(example["manifest_id"]),
        device=input_ids.device,
    )
    anchors = sample_anchor_positions(
        int(example["response_start"]),
        int(example["response_end"]),
        config.block_size,
        config.num_anchors,
        generator=generator,
        device=input_ids.device,
    )
    total_anchors = int(anchors.numel())
    totals = {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}
    for start in range(0, total_anchors, anchor_chunk_size):
        chunk = anchors[start : start + anchor_chunk_size]
        blocks = build_masked_blocks(
            input_ids,
            chunk,
            block_size=config.block_size,
            mask_token_id=int(draft.mask_token_id),
            position_ids=position_ids,
        )
        with torch.no_grad():
            noise_embeddings = token_io.embed(blocks.block_input_ids.reshape(1, -1))
        with _autocast(input_ids.device, config):
            hidden = draft(
                noise_embeddings=noise_embeddings,
                target_context=context_hidden,
                context_position_ids=context_positions,
                block_position_ids=blocks.block_position_ids,
                anchors=blocks.anchors,
                context_original_positions=context_original,
                use_flex_attention=config.use_flex_attention,
            )
            logits = token_io.logits(hidden).reshape(
                1,
                chunk.numel(),
                config.block_size,
                -1,
            )[:, :, 1:, :]
            labels = blocks.labels[:, 1:].unsqueeze(0).long()
            loss, metrics = weighted_block_cross_entropy(
                logits,
                labels,
                decay=float(config.loss_decay),
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite DFlash loss for {example['manifest_id']}")
        anchor_scale = float(chunk.numel()) / float(total_anchors)
        if backward:
            (loss * anchor_scale * backward_scale).backward()
        for key in totals:
            totals[key] += metrics[key] * anchor_scale
    return totals


def _chunk_candidates(config: DFlashTrainConfig) -> list[int]:
    candidates: list[int] = []
    value = min(config.anchor_chunk_size, config.num_anchors)
    while True:
        candidates.append(value)
        if value <= config.min_anchor_chunk_size:
            return candidates
        value = max(config.min_anchor_chunk_size, value // 2)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _run_group(
    cache: TeacherCache,
    indices: list[int],
    draft: DFlashVLMModel,
    token_io: FrozenTargetTokenIO,
    config: DFlashTrainConfig,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    if not indices:
        optimizer.zero_grad(set_to_none=True)
        return {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}, min(
            config.anchor_chunk_size, config.num_anchors
        )
    last_oom: BaseException | None = None
    for chunk_size in _chunk_candidates(config):
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}
        try:
            scale = 1.0 / len(indices)
            for index in indices:
                example = cache.load_example(index, device)
                metrics = _train_or_eval_example(
                    example,
                    draft,
                    token_io,
                    config,
                    epoch=epoch,
                    anchor_chunk_size=chunk_size,
                    backward_scale=scale,
                    backward=True,
                )
                for key in totals:
                    totals[key] += metrics[key] * scale
                del example
            return totals, chunk_size
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not _is_oom(exc):
                raise
            last_oom = exc
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[oom] retry group with smaller anchor_chunk_size after {chunk_size}")
    raise RuntimeError("cache-only draft training OOM at minimum anchor chunk size") from last_oom


@torch.no_grad()
def evaluate_cached_loss(
    cache: TeacherCache,
    draft: DFlashVLMModel,
    token_io: FrozenTargetTokenIO,
    config: DFlashTrainConfig,
    *,
    device: torch.device,
) -> float:
    was_training = draft.training
    draft.eval()
    count = min(config.eval_cache_samples, len(cache.index))
    total = 0.0
    for index in range(count):
        example = cache.load_example(index, device)
        metrics = _train_or_eval_example(
            example,
            draft,
            token_io,
            config,
            epoch=0,
            anchor_chunk_size=min(config.anchor_chunk_size, config.num_anchors),
            backward_scale=1.0,
            backward=False,
        )
        total += metrics["loss"]
    if was_training:
        draft.train()
    return total / count


def _checkpoint_metadata(
    cache: TeacherCache,
    draft: DFlashVLMModel,
    config: DFlashTrainConfig,
    *,
    step: int,
) -> dict[str, Any]:
    teacher = cache.metadata
    metadata = config.to_dict()
    metadata.update(
        {
            "target_layer_ids": list(draft.target_layer_ids),
            "mask_token_id": int(draft.mask_token_id),
            "target_vocab_size": int(teacher["target_vocab_size"]),
            "target_hidden_size": int(teacher["target_hidden_size"]),
            "tokenizer_fingerprint": teacher["tokenizer_fingerprint"],
            "processor_fingerprint": teacher["processor_fingerprint"],
            "target_provenance": teacher["target_provenance"],
            "teacher_cache_format": teacher["format"],
            "implementation_version": DFLASH_IMPLEMENTATION_VERSION,
            "teacher_source_manifest_sha256": teacher["source_manifest_sha256"],
            "step": step,
        }
    )
    return metadata


def save_cached_checkpoint(
    directory: str | Path,
    cache: TeacherCache,
    draft: DFlashVLMModel,
    config: DFlashTrainConfig,
    *,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    progress: dict[str, Any] | None = None,
    rng_states: list[dict[str, Any]] | None = None,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to save draft checkpoints") from exc
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in draft.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    (directory / "dflash_config.json").write_text(
        json.dumps(_checkpoint_metadata(cache, draft, config, step=step), indent=2, sort_keys=True)
    )
    if optimizer is not None and progress is not None:
        trainer_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng_states": rng_states,
            **progress,
        }
        torch.save(trainer_state, directory / "trainer_state.pt")


def _validate_checkpoint_metadata(
    metadata: dict[str, Any],
    cache: TeacherCache,
    draft: DFlashVLMModel,
    config: DFlashTrainConfig,
) -> None:
    teacher = cache.metadata
    checks = {
        "implementation_version": DFLASH_IMPLEMENTATION_VERSION,
        "target_model": config.target_model,
        "target_hidden_size": teacher["target_hidden_size"],
        "target_vocab_size": teacher["target_vocab_size"],
        "context_mode": config.context_mode,
        "block_size": config.block_size,
        "num_draft_layers": config.num_draft_layers,
        "tokenizer_fingerprint": teacher["tokenizer_fingerprint"],
        "processor_fingerprint": teacher["processor_fingerprint"],
        "mask_token_id": int(draft.mask_token_id),
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint mismatch for {key}: {metadata.get(key)!r} != {expected!r}")
    if list(metadata.get("target_layer_ids", [])) != list(draft.target_layer_ids):
        raise ValueError("checkpoint selected target layers do not match the teacher cache")


def load_cached_checkpoint(
    directory: str | Path,
    cache: TeacherCache,
    draft: DFlashVLMModel,
    config: DFlashTrainConfig,
) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to load draft checkpoints") from exc
    directory = Path(directory)
    metadata = json.loads((directory / "dflash_config.json").read_text())
    _validate_checkpoint_metadata(metadata, cache, draft, config)
    state = load_file(str(directory / "model.safetensors"), device=str(next(draft.parameters()).device))
    draft.load_state_dict(state, strict=True)
    return metadata


def _scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rotate_snapshots(root: Path, keep: int) -> None:
    snapshots = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(("epoch-", "step-"))
        ),
        key=lambda path: path.stat().st_mtime,
    )
    for path in snapshots[:-keep]:
        shutil.rmtree(path)


def _local_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _gather_rng_states(
    distributed: DistributedContext,
) -> list[dict[str, Any]] | None:
    local = _local_rng_state(distributed.device)
    if not distributed.enabled:
        return [local]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * distributed.world_size if distributed.is_main else None
    )
    dist.gather_object(local, gathered, dst=0)
    if not distributed.is_main:
        return None
    assert gathered is not None and all(state is not None for state in gathered)
    return [state for state in gathered if state is not None]


def _restore_rng_state(
    states: list[dict[str, Any]],
    distributed: DistributedContext,
) -> None:
    if len(states) != distributed.world_size:
        raise ValueError(
            "resume world size differs from the checkpoint RNG state: "
            f"{distributed.world_size} != {len(states)}"
        )
    state = states[distributed.rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if distributed.device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], distributed.device)


def _broadcast_model(draft: DFlashVLMModel, distributed: DistributedContext) -> None:
    if not distributed.enabled:
        return
    for value in draft.state_dict().values():
        dist.broadcast(value, src=0)


def _synchronize_step(
    draft: DFlashVLMModel,
    metrics: dict[str, float],
    local_records: int,
    global_records: int,
    chunk_size: int,
    distributed: DistributedContext,
) -> tuple[dict[str, float], int]:
    """Average gradients/metrics over a possibly uneven last global batch."""

    if global_records < 1:
        raise RuntimeError("an optimizer step cannot contain zero records")
    local_weight = float(local_records) / float(global_records)
    for parameter in draft.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        else:
            parameter.grad.mul_(local_weight)
    if distributed.enabled:
        bucket: list[torch.Tensor] = []
        bucket_bytes = 0

        def reduce_bucket() -> None:
            if not bucket:
                return
            flat = torch.cat([gradient.reshape(-1) for gradient in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for gradient in bucket:
                count = gradient.numel()
                gradient.copy_(flat[offset : offset + count].view_as(gradient))
                offset += count

        for parameter in draft.parameters():
            assert parameter.grad is not None
            gradient_bytes = parameter.grad.numel() * parameter.grad.element_size()
            if bucket and bucket_bytes + gradient_bytes > 64 * 1024 * 1024:
                reduce_bucket()
                bucket = []
                bucket_bytes = 0
            bucket.append(parameter.grad)
            bucket_bytes += gradient_bytes
        reduce_bucket()

    packed = torch.tensor(
        [
            metrics["loss"] * local_records,
            metrics["token_accuracy"] * local_records,
            metrics["valid_tokens"] * local_records,
            float(local_records),
            float(chunk_size),
        ],
        dtype=torch.float64,
        device=distributed.device,
    )
    if distributed.enabled:
        dist.all_reduce(packed[:4], op=dist.ReduceOp.SUM)
        dist.all_reduce(packed[4:], op=dist.ReduceOp.MIN)
    count = float(packed[3].item())
    synchronized = {
        "loss": float(packed[0].item() / count),
        "token_accuracy": float(packed[1].item() / count),
        "valid_tokens": float(packed[2].item() / count),
    }
    return synchronized, int(packed[4].item())


def _distributed_eval_loss(
    cache: TeacherCache,
    draft: DFlashVLMModel,
    token_io: FrozenTargetTokenIO,
    config: DFlashTrainConfig,
    distributed: DistributedContext,
) -> float:
    value = 0.0
    if distributed.is_main:
        value = evaluate_cached_loss(
            cache, draft, token_io, config, device=distributed.device
        )
    packed = torch.tensor(value, dtype=torch.float64, device=distributed.device)
    if distributed.enabled:
        dist.broadcast(packed, src=0)
    return float(packed.item())


def _training_contract(
    cache: TeacherCache,
    config: DFlashTrainConfig,
    distributed: DistributedContext,
    *,
    local_records_per_step: int,
    total_steps: int,
) -> dict[str, Any]:
    """Fields which must remain fixed for mathematically exact continuation."""

    return {
        "teacher_source_manifest_sha256": cache.metadata["source_manifest_sha256"],
        "cached_count": len(cache.index),
        "world_size": distributed.world_size,
        "local_records_per_step": local_records_per_step,
        "global_records_per_step": local_records_per_step * distributed.world_size,
        "epochs": config.epochs,
        "max_train_steps": config.max_train_steps,
        "total_steps": total_steps,
        "seed": config.seed,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "adam_beta1": config.adam_beta1,
        "adam_beta2": config.adam_beta2,
        "adam_eps": config.adam_eps,
        "warmup_ratio": config.warmup_ratio,
        "gradient_clip_norm": config.gradient_clip_norm,
        "target_model": config.target_model,
        "context_mode": config.context_mode,
        "max_seq_length": config.max_seq_length,
        "block_size": config.block_size,
        "num_anchors": config.num_anchors,
        "anchor_chunk_size": config.anchor_chunk_size,
        "min_anchor_chunk_size": config.min_anchor_chunk_size,
        "loss_decay": config.loss_decay,
        "num_draft_layers": config.num_draft_layers,
        "num_target_features": config.num_target_features,
        "target_layer_ids": list(cache.metadata["target_layer_ids"]),
        "mixed_precision": config.mixed_precision,
        "use_flex_attention": config.use_flex_attention,
        "compile_flex_attention": config.compile_flex_attention,
        "gradient_checkpointing": config.gradient_checkpointing,
    }


def _validate_training_contract(saved: Any, current: dict[str, Any]) -> None:
    if not isinstance(saved, dict):
        raise ValueError(
            "checkpoint predates the exact-resume training contract and cannot be resumed safely"
        )
    differences = {
        key: {"checkpoint": saved.get(key), "current": value}
        for key, value in current.items()
        if saved.get(key) != value
    }
    if differences:
        raise ValueError(
            "exact resume rejected changed training parameters: "
            + json.dumps(differences, sort_keys=True)
        )


def _atomic_checkpoint(
    destination: Path,
    cache: TeacherCache,
    draft: DFlashVLMModel,
    config: DFlashTrainConfig,
    *,
    step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: dict[str, Any],
    rng_states: list[dict[str, Any]],
) -> None:
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    save_cached_checkpoint(
        temporary,
        cache,
        draft,
        config,
        step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=progress,
        rng_states=rng_states,
    )
    _fsync_tree(temporary)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    _fsync_directory(directory)


def _copy_checkpoint_atomic(source: Path, destination: Path) -> None:
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, copy_function=os.link)
    except OSError:
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def _update_latest(checkpoint_root: Path, recovery: Path) -> None:
    latest = checkpoint_root / "latest"
    if latest.exists() and not latest.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink latest checkpoint: {latest}")
    temporary = checkpoint_root / f".latest.tmp-{uuid.uuid4().hex}"
    os.symlink(recovery.name, temporary, target_is_directory=True)
    os.replace(temporary, latest)
    _fsync_directory(checkpoint_root)


def _commit_recovery_checkpoint(
    checkpoint_root: Path,
    cache: TeacherCache,
    draft: DFlashVLMModel,
    config: DFlashTrainConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    progress: dict[str, Any],
    distributed: DistributedContext,
    *,
    step: int,
    snapshot_name: str | None,
) -> None:
    """Commit every completed optimizer step and atomically advance ``latest``."""

    rng_states = _gather_rng_states(distributed)
    if distributed.is_main:
        assert rng_states is not None
        recovery = checkpoint_root / f"recovery-step-{step:08d}"
        _atomic_checkpoint(
            recovery,
            cache,
            draft,
            config,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            rng_states=rng_states,
        )
        _update_latest(checkpoint_root, recovery)
        if snapshot_name is not None:
            _copy_checkpoint_atomic(recovery, checkpoint_root / snapshot_name)
            _rotate_snapshots(checkpoint_root, config.keep_last_checkpoints)
        for path in checkpoint_root.glob("recovery-step-*"):
            if path != recovery and path.is_dir():
                shutil.rmtree(path)
    distributed.barrier()


def _resolve_resume_checkpoint(config: DFlashTrainConfig, output_dir: Path) -> Path | None:
    requested = config.resume_from_checkpoint.strip()
    latest = output_dir / "checkpoints" / "latest"
    if requested in {"auto", "latest"}:
        if not latest.exists():
            raise FileNotFoundError(f"no latest checkpoint exists at {latest}")
        return latest.resolve(strict=True)
    if requested:
        return Path(requested).expanduser().resolve(strict=True)
    if config.auto_resume and not config.overwrite and latest.exists():
        return latest.resolve(strict=True)
    return None


def _snapshot_name(
    config: DFlashTrainConfig,
    *,
    epoch: int,
    previous_offset: int,
    next_offset: int,
    epoch_size: int,
    step: int,
) -> str | None:
    boundary: float | None = None
    if config.save_every_epochs:
        interval = config.save_every_epochs
        before = epoch + previous_offset / epoch_size
        after = epoch + next_offset / epoch_size
        before_bucket = math.floor((before + 1e-12) / interval)
        after_bucket = math.floor((after + 1e-12) / interval)
        if after_bucket > before_bucket:
            boundary = after_bucket * interval
    if boundary is not None:
        return f"epoch-{boundary:08.4f}-step-{step:08d}"
    if config.save_every_steps and step % config.save_every_steps == 0:
        return f"step-{step:08d}"
    return None


_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received {signal.Signals(signum).name}; stopping after atomic recovery")


def _stop_requested(distributed: DistributedContext) -> bool:
    packed = torch.tensor(
        int(_STOP_REQUESTED), dtype=torch.int32, device=distributed.device
    )
    if distributed.enabled:
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return bool(packed.item())


def _validate_gradients(draft: DFlashVLMModel, token_io: FrozenTargetTokenIO) -> None:
    if token_io.requires_grad:
        raise RuntimeError("cached target embedding/LM head unexpectedly requires gradients")
    gradients = [parameter.grad for parameter in draft.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("no draft/projection gradients were produced")
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("draft/projection gradients contain NaN or Inf")


def _reload_checkpoint_on_cpu(
    output_dir: Path,
    cache: TeacherCache,
    config: DFlashTrainConfig,
    reference: DFlashVLMModel,
) -> None:
    reloaded = make_cached_draft_model(cache, config, device=torch.device("cpu"))
    load_cached_checkpoint(output_dir, cache, reloaded, config)
    reference_state = reference.state_dict()
    for name, value in reloaded.state_dict().items():
        if not torch.equal(value, reference_state[name].detach().cpu()):
            raise RuntimeError(f"save/reload mismatch for draft tensor {name}")
    print("[reload] checkpoint weights are bit-exact")


def train_cached_draft(config: DFlashTrainConfig) -> list[dict[str, Any]]:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    distributed = initialize_distributed(config.device, config.distributed_backend)
    device = distributed.device
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.manual_seed(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    cache = TeacherCache(config.teacher_cache_dir)
    _validate_cache_contract(cache, config)
    token_io = cache.load_token_io(device, _dtype(config))
    draft = make_cached_draft_model(cache, config, device=device)
    _broadcast_model(draft, distributed)
    draft.train()
    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    local_records_per_step = config.micro_batch_size * config.gradient_accumulation_steps
    global_records_per_step = local_records_per_step * distributed.world_size
    steps_per_epoch = math.ceil(len(cache.index) / global_records_per_step)
    natural_steps = steps_per_epoch * config.epochs
    total_steps = min(natural_steps, config.max_train_steps) if config.max_train_steps else natural_steps
    scheduler = _scheduler(optimizer, max(1, total_steps), config.warmup_ratio)
    contract = _training_contract(
        cache,
        config,
        distributed,
        local_records_per_step=local_records_per_step,
        total_steps=total_steps,
    )
    output_dir = Path(config.output_dir).expanduser().resolve()
    if config.checkpoint and config.resume_from_checkpoint:
        raise ValueError("use either --checkpoint or --resume, not both")
    resume_checkpoint = _resolve_resume_checkpoint(config, output_dir)
    if distributed.is_main:
        if output_dir.exists() and any(output_dir.iterdir()) and resume_checkpoint is None:
            if not config.overwrite:
                raise FileExistsError(
                    f"training output already exists at {output_dir}; pass --overwrite or --resume"
                )
            if output_dir == output_dir.parent:
                raise RuntimeError("refusing to overwrite a filesystem root")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    checkpoint_root = output_dir / "checkpoints"
    if distributed.is_main:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    distributed.barrier()

    if config.checkpoint and resume_checkpoint is None:
        load_cached_checkpoint(config.checkpoint, cache, draft, config)
        _broadcast_model(draft, distributed)
        if distributed.is_main:
            print(f"[init] loaded Stage 1 weights from {config.checkpoint}")

    # Fresh ranks intentionally get distinct runtime RNG streams. Model weights
    # were initialized/broadcast using the common seed above.
    rank_seed = config.seed + distributed.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(rank_seed)

    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 0
    next_sample_offset = 0
    initial_eval_loss: float | None = None
    samples_seen = 0
    saved_sampler_state: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        load_cached_checkpoint(resume_checkpoint, cache, draft, config)
        state = torch.load(
            resume_checkpoint / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        _validate_training_contract(state.get("training_contract"), contract)
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in list(values.items()):
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        next_sample_offset = int(state["next_sample_offset"])
        samples_seen = int(state.get("samples_seen", 0))
        history = list(state.get("history", []))
        initial_eval_loss = state.get("initial_eval_loss")
        saved_sampler_state = state.get("sampler_state")
        if not isinstance(saved_sampler_state, dict):
            raise ValueError("resume checkpoint is missing sampler_state")
        if int(saved_sampler_state.get("epoch", -1)) != start_epoch:
            raise ValueError("sampler epoch does not match trainer progress")
        if int(saved_sampler_state.get("next_sample_offset", -1)) != next_sample_offset:
            raise ValueError("sampler offset does not match trainer progress")
        expected_order = list(range(len(cache.index)))
        random.Random(config.seed + start_epoch).shuffle(expected_order)
        if list(saved_sampler_state.get("order", [])) != expected_order:
            raise ValueError("saved sampler permutation does not match seed/cache contract")
        rng_states = state.get("rng_states")
        if not isinstance(rng_states, list):
            raise ValueError("resume checkpoint is missing per-rank RNG states")
        _restore_rng_state(rng_states, distributed)
        if distributed.is_main:
            print(
                f"[resume] checkpoint={resume_checkpoint} epoch={start_epoch} "
                f"next_sample={next_sample_offset} step={global_step} "
                f"samples_seen={samples_seen}"
            )

    if initial_eval_loss is None:
        initial_eval_loss = _distributed_eval_loss(
            cache, draft, token_io, config, distributed
        )
    trainable_parameters = sum(parameter.numel() for parameter in draft.parameters())
    if distributed.is_main:
        print(
            f"[train-setup] stage={config.stage} records={len(cache.index)} "
            f"world_size={distributed.world_size} dtype={config.mixed_precision} "
            f"micro_batch_per_rank={config.micro_batch_size} "
            f"grad_accum={config.gradient_accumulation_steps} "
            f"global_records_per_step={global_records_per_step} params={trainable_parameters} "
            f"target_transformer_loaded=False initial_loss={initial_eval_loss:.6f}"
        )

    stop = bool(config.max_train_steps and global_step >= config.max_train_steps)
    resume_epoch = start_epoch
    resume_next_sample = next_sample_offset
    interrupted = False
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, _request_stop)
    try:
        for epoch in range(start_epoch, config.epochs):
            if stop:
                break
            order = list(range(len(cache.index)))
            random.Random(config.seed + epoch).shuffle(order)
            first = next_sample_offset if epoch == start_epoch else 0
            if epoch == start_epoch and saved_sampler_state is not None:
                order = list(saved_sampler_state["order"])
            for group_start in range(first, len(order), global_records_per_step):
                if _stop_requested(distributed):
                    interrupted = True
                    break
                global_indices = order[
                    group_start : group_start + global_records_per_step
                ]
                local_start = distributed.rank * local_records_per_step
                indices = global_indices[
                    local_start : local_start + local_records_per_step
                ]
                local_metrics, local_chunk_size = _run_group(
                    cache,
                    indices,
                    draft,
                    token_io,
                    config,
                    optimizer,
                    epoch=epoch,
                    device=device,
                )
                metrics, chunk_size = _synchronize_step(
                    draft,
                    local_metrics,
                    len(indices),
                    len(global_indices),
                    local_chunk_size,
                    distributed,
                )
                if global_step == 0:
                    _validate_gradients(draft, token_io)
                    if distributed.is_main:
                        print(
                            "[gradients] target_token_io=False "
                            "draft_projection=True distributed_average=True finite=True"
                        )
                grad_norm = clip_grad_norm_(draft.parameters(), config.gradient_clip_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError("gradient norm is NaN or Inf")
                optimizer.step()
                scheduler.step()
                global_step += 1
                samples_seen += len(global_indices)
                next_offset = min(group_start + len(global_indices), len(order))
                resume_epoch = epoch
                resume_next_sample = next_offset
                row = {
                    "step": global_step,
                    "epoch": epoch + next_offset / len(order),
                    "loss": metrics["loss"],
                    "token_accuracy": metrics["token_accuracy"],
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "anchor_chunk_size": chunk_size,
                    "records": len(global_indices),
                    "samples_seen": samples_seen,
                }
                history.append(row)
                sampler_state = {
                    "epoch": epoch,
                    "next_sample_offset": next_offset,
                    "order": order,
                }
                progress = {
                    "global_step": global_step,
                    "epoch": resume_epoch,
                    "next_sample_offset": resume_next_sample,
                    "samples_seen": samples_seen,
                    "sampler_state": sampler_state,
                    "training_contract": contract,
                    "history": history,
                    "initial_eval_loss": initial_eval_loss,
                }
                snapshot = _snapshot_name(
                    config,
                    epoch=epoch,
                    previous_offset=group_start,
                    next_offset=next_offset,
                    epoch_size=len(order),
                    step=global_step,
                )
                _commit_recovery_checkpoint(
                    checkpoint_root,
                    cache,
                    draft,
                    config,
                    optimizer,
                    scheduler,
                    progress,
                    distributed,
                    step=global_step,
                    snapshot_name=snapshot,
                )
                if distributed.is_main:
                    print("[step] " + json.dumps(row, sort_keys=True))
                    if snapshot is not None:
                        print(f"[checkpoint] snapshot={checkpoint_root / snapshot}")
                if config.max_train_steps and global_step >= config.max_train_steps:
                    stop = True
                    break
                if _stop_requested(distributed):
                    interrupted = True
                    break
            next_sample_offset = 0
            saved_sampler_state = None
            if stop or interrupted:
                break
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if interrupted:
        if distributed.is_main:
            print(
                f"[interrupted] exact recovery committed at step={global_step}; "
                f"resume={checkpoint_root / 'latest'}"
            )
        distributed.barrier()
        distributed.close()
        return history

    final_eval_loss = _distributed_eval_loss(cache, draft, token_io, config, distributed)
    if resume_next_sample == len(cache.index) and resume_epoch + 1 < config.epochs:
        # This can only occur when max_train_steps ends exactly at an epoch
        # boundary; retain the completed epoch state rather than inventing a
        # permutation which was never consumed.
        pass
    final_order = list(range(len(cache.index)))
    random.Random(config.seed + resume_epoch).shuffle(final_order)
    final_progress = {
        "global_step": global_step,
        "epoch": resume_epoch,
        "next_sample_offset": resume_next_sample,
        "samples_seen": samples_seen,
        "sampler_state": {
            "epoch": resume_epoch,
            "next_sample_offset": resume_next_sample,
            "order": final_order,
        },
        "training_contract": contract,
        "history": history,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
    }
    final_rng_states = _gather_rng_states(distributed)
    if distributed.is_main:
        assert final_rng_states is not None
        save_cached_checkpoint(
            output_dir,
            cache,
            draft,
            config,
            step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=final_progress,
            rng_states=final_rng_states,
        )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True)
        )
        (output_dir / "COMPLETED").write_text(
            json.dumps({"global_step": global_step, "final_eval_loss": final_eval_loss})
            + "\n"
        )
        _reload_checkpoint_on_cpu(output_dir, cache, config, draft)
    distributed.barrier()
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    if distributed.enabled:
        peak_tensor = torch.tensor(peak, device=device)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        peak = float(peak_tensor.item())
    decreased = final_eval_loss < initial_eval_loss
    if distributed.is_main:
        print(
            f"[train-result] steps={global_step} initial_loss={initial_eval_loss:.6f} "
            f"final_loss={final_eval_loss:.6f} decreased={decreased} "
            f"max_rank_peak_vram={peak:.2f}GiB checkpoint={output_dir}"
        )
    if config.require_loss_decrease and not decreased:
        raise AssertionError(
            f"fixed-subset cached loss did not decrease: {initial_eval_loss} -> {final_eval_loss}"
        )
    distributed.close()
    return history
