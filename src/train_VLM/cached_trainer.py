from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .config import DFlashTrainConfig
from .data import build_masked_blocks, make_anchor_generator, sample_anchor_positions
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
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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


def _rotate_checkpoints(root: Path, keep: int) -> None:
    if not root.exists():
        return
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for path in directories[:-keep]:
        shutil.rmtree(path)


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
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    cache = TeacherCache(config.teacher_cache_dir)
    _validate_cache_contract(cache, config)
    token_io = cache.load_token_io(device, _dtype(config))
    draft = make_cached_draft_model(cache, config, device=device)
    draft.train()
    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    effective_group = config.micro_batch_size * config.gradient_accumulation_steps
    natural_steps = math.ceil(len(cache.index) / effective_group) * config.epochs
    total_steps = min(natural_steps, config.max_train_steps) if config.max_train_steps else natural_steps
    scheduler = _scheduler(optimizer, max(1, total_steps), config.warmup_ratio)
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not config.resume_from_checkpoint:
        if not config.overwrite:
            raise FileExistsError(
                f"training output already exists at {output_dir}; pass --overwrite or --resume"
            )
        if output_dir == output_dir.parent:
            raise RuntimeError("refusing to overwrite a filesystem root")
        shutil.rmtree(output_dir)
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if config.checkpoint and config.resume_from_checkpoint:
        raise ValueError("use either --checkpoint or --resume, not both")
    if config.checkpoint:
        load_cached_checkpoint(config.checkpoint, cache, draft, config)
        print(f"[init] loaded weights from {config.checkpoint}")

    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 0
    next_group_start = 0
    initial_eval_loss: float | None = None
    if config.resume_from_checkpoint:
        load_cached_checkpoint(config.resume_from_checkpoint, cache, draft, config)
        state = torch.load(
            Path(config.resume_from_checkpoint) / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        for values in optimizer.state.values():
            for key, value in list(values.items()):
                if torch.is_tensor(value):
                    values[key] = value.to(device)
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        next_group_start = int(state["next_group_start"])
        history = list(state.get("history", []))
        initial_eval_loss = state.get("initial_eval_loss")
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(
            f"[resume] checkpoint={config.resume_from_checkpoint} epoch={start_epoch} "
            f"next_group={next_group_start} step={global_step}"
        )

    if initial_eval_loss is None:
        initial_eval_loss = evaluate_cached_loss(cache, draft, token_io, config, device=device)
    trainable_parameters = sum(parameter.numel() for parameter in draft.parameters())
    print(
        f"[train-setup] stage={config.stage} records={len(cache.index)} device={device} "
        f"dtype={config.mixed_precision} micro_batch={config.micro_batch_size} "
        f"grad_accum={config.gradient_accumulation_steps} params={trainable_parameters} "
        f"target_transformer_loaded=False initial_loss={initial_eval_loss:.6f}"
    )

    stop = bool(config.max_train_steps and global_step >= config.max_train_steps)
    resume_epoch = start_epoch
    resume_next_group = next_group_start
    for epoch in range(start_epoch, config.epochs):
        if stop:
            break
        order = list(range(len(cache.index)))
        random.Random(config.seed + epoch).shuffle(order)
        first = next_group_start if epoch == start_epoch else 0
        for group_start in range(first, len(order), effective_group):
            indices = order[group_start : group_start + effective_group]
            metrics, chunk_size = _run_group(
                cache,
                indices,
                draft,
                token_io,
                config,
                optimizer,
                epoch=epoch,
                device=device,
            )
            if global_step == 0:
                _validate_gradients(draft, token_io)
                print("[gradients] target_token_io=False draft_projection=True finite=True")
            grad_norm = clip_grad_norm_(draft.parameters(), config.gradient_clip_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError("gradient norm is NaN or Inf")
            optimizer.step()
            scheduler.step()
            global_step += 1
            row = {
                "step": global_step,
                "epoch": epoch + 1,
                "loss": metrics["loss"],
                "token_accuracy": metrics["token_accuracy"],
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "anchor_chunk_size": chunk_size,
                "records": len(indices),
            }
            history.append(row)
            print("[step] " + json.dumps(row, sort_keys=True))
            next_offset = group_start + effective_group
            if next_offset >= len(order):
                resume_epoch = epoch + 1
                resume_next_group = 0
            else:
                resume_epoch = epoch
                resume_next_group = next_offset
            progress = {
                "global_step": global_step,
                "epoch": resume_epoch,
                "next_group_start": resume_next_group,
                "history": history,
                "initial_eval_loss": initial_eval_loss,
            }
            if config.save_every_steps and global_step % config.save_every_steps == 0:
                checkpoint = checkpoint_root / f"step-{global_step:08d}"
                save_cached_checkpoint(
                    checkpoint,
                    cache,
                    draft,
                    config,
                    step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    progress=progress,
                )
                _rotate_checkpoints(checkpoint_root, config.keep_last_checkpoints)
            if config.max_train_steps and global_step >= config.max_train_steps:
                stop = True
                break
        next_group_start = 0
        if stop:
            break

    final_eval_loss = evaluate_cached_loss(cache, draft, token_io, config, device=device)
    final_progress = {
        "global_step": global_step,
        "epoch": resume_epoch,
        "next_group_start": resume_next_group,
        "history": history,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
    }
    save_cached_checkpoint(
        output_dir,
        cache,
        draft,
        config,
        step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        progress=final_progress,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True))
    _reload_checkpoint_on_cpu(output_dir, cache, config, draft)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    decreased = final_eval_loss < initial_eval_loss
    print(
        f"[train-result] steps={global_step} initial_loss={initial_eval_loss:.6f} "
        f"final_loss={final_eval_loss:.6f} decreased={decreased} peak_vram={peak:.2f}GiB "
        f"checkpoint={output_dir}"
    )
    if config.require_loss_decrease and not decreased:
        raise AssertionError(
            f"fixed-subset cached loss did not decrease: {initial_eval_loss} -> {final_eval_loss}"
        )
    return history
