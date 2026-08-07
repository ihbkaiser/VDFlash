from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.utils import clip_grad_norm_

from .config import DFlashTrainConfig
from .data import (
    build_masked_blocks,
    make_anchor_generator,
    sample_anchor_positions,
    select_context_positions,
)
from .losses import weighted_block_cross_entropy
from .model import DFLASH_IMPLEMENTATION_VERSION, DFlashVLMModel, build_target_layer_ids
from .target import PreparedExample, Qwen25VLTargetAdapter, load_jsonl


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device, config: DFlashTrainConfig):
    enabled = device.type == "cuda" and config.mixed_precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def make_draft_model(adapter: Qwen25VLTargetAdapter, config: DFlashTrainConfig) -> DFlashVLMModel:
    layer_ids = list(config.selected_target_layers or build_target_layer_ids(
        int(adapter.text_config.num_hidden_layers), config.num_target_features
    ))
    if max(layer_ids) >= int(adapter.text_config.num_hidden_layers):
        raise ValueError("selected_target_layers contains an index outside the target model")
    model = DFlashVLMModel(
        adapter.text_config,
        num_draft_layers=config.num_draft_layers,
        num_target_features=len(layer_ids),
        block_size=config.block_size,
        compile_flex_attention=config.compile_flex_attention,
    )
    model.target_layer_ids = layer_ids
    model.context_mode = config.context_mode
    model.mask_token_id = adapter.resolve_mask_token_id()
    model.gradient_checkpointing = bool(config.gradient_checkpointing)
    target_dtype = adapter.input_embeddings.weight.dtype
    return model.to(device=adapter.device, dtype=target_dtype)


def extract_training_context(
    adapter: Qwen25VLTargetAdapter,
    example: PreparedExample,
    config: DFlashTrainConfig,
    layer_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = adapter.forward_clean(example.inputs)
    selected = adapter.selected_hidden_features(outputs, layer_ids)[0]
    image_ids, video_ids = adapter.visual_token_ids
    original_positions = select_context_positions(
        example.input_ids[0],
        context_mode=config.context_mode,
        image_token_ids=image_ids,
        video_token_ids=video_ids,
    )
    # ``forward_clean`` returns inference tensors so the frozen target keeps no
    # autograd state. Clone selected features outside inference_mode before the
    # trainable projection consumes them; Linear backward is otherwise forbidden
    # from saving an inference tensor.
    context_hidden = selected[original_positions].unsqueeze(0).clone()
    context_positions = example.position_ids[:, :, original_positions]
    return context_hidden, context_positions, original_positions


def train_example(
    adapter: Qwen25VLTargetAdapter,
    draft_model: DFlashVLMModel,
    example: PreparedExample,
    config: DFlashTrainConfig,
    *,
    generator: torch.Generator,
    anchor_chunk_size: int | None = None,
    backward: bool = True,
    backward_scale: float = 1.0,
    backward_fn: Callable[[torch.Tensor], None] | None = None,
) -> dict[str, float]:
    """Run one clean sequence and accumulate its gradient over anchor chunks.

    Every chunk is normalized by the full number of sampled anchors, so retrying
    the entire accumulation group with a smaller chunk has exactly the same
    objective and anchors as the original attempt.
    """

    layer_ids = list(
        getattr(
            draft_model,
            "target_layer_ids",
            build_target_layer_ids(adapter.text_config.num_hidden_layers, config.num_target_features),
        )
    )
    context_hidden, context_positions, context_original_positions = extract_training_context(
        adapter, example, config, layer_ids
    )
    anchors = sample_anchor_positions(
        example.response_start,
        example.response_end,
        config.block_size,
        config.num_anchors,
        generator=generator,
        device=example.input_ids.device,
    )
    total_blocks = int(anchors.numel())
    chunk_size = int(anchor_chunk_size or config.anchor_chunk_size)
    total_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0.0
    for start in range(0, total_blocks, chunk_size):
        chunk_anchors = anchors[start : start + chunk_size]
        blocks = build_masked_blocks(
            example.input_ids,
            chunk_anchors,
            block_size=config.block_size,
            mask_token_id=int(draft_model.mask_token_id),
            position_ids=example.position_ids,
        )
        with torch.no_grad():
            noise_ids = blocks.block_input_ids.reshape(1, -1)
            noise_embeddings = adapter.input_embeddings(noise_ids)
        with _autocast_context(adapter.device, config):
            hidden = draft_model(
                noise_embeddings=noise_embeddings,
                target_context=context_hidden,
                context_position_ids=context_positions,
                block_position_ids=blocks.block_position_ids,
                anchors=blocks.anchors,
                context_original_positions=context_original_positions,
                use_flex_attention=config.use_flex_attention,
            )
            logits = adapter.lm_head(hidden).reshape(
                1, chunk_anchors.numel(), config.block_size, -1
            )[:, :, 1:, :]
            labels = blocks.labels[:, 1:].unsqueeze(0)
            loss, metrics = weighted_block_cross_entropy(
                logits, labels, decay=float(config.loss_decay)
            )
        scale = float(chunk_anchors.numel()) / float(total_blocks)
        if backward:
            scaled_loss = loss * scale * backward_scale
            (backward_fn or torch.Tensor.backward)(scaled_loss)
        total_loss += metrics["loss"] * scale
        total_accuracy += metrics["token_accuracy"] * scale
        total_tokens += metrics["valid_tokens"]
    return {
        "loss": total_loss,
        "token_accuracy": total_accuracy,
        "valid_tokens": total_tokens,
        "anchors": float(total_blocks),
    }


def _checkpoint_metadata(
    draft_model: DFlashVLMModel,
    config: DFlashTrainConfig,
    adapter: Qwen25VLTargetAdapter,
    *,
    step: int | None,
) -> dict[str, Any]:
    metadata = config.to_dict()
    metadata.update(
        {
            "implementation_version": DFLASH_IMPLEMENTATION_VERSION,
            "target_layer_ids": list(getattr(draft_model, "target_layer_ids", [])),
            "mask_token_id": int(getattr(draft_model, "mask_token_id", -1)),
            "target_vocab_size": adapter.vocab_size,
            "target_hidden_size": adapter.hidden_size,
            "tokenizer_fingerprint": adapter.tokenizer_fingerprint(),
            "processor_fingerprint": adapter.processor_fingerprint(),
            "target_provenance": adapter.target_provenance(),
            "step": step,
        }
    )
    return metadata


def save_draft_checkpoint(
    output_dir: str | Path,
    draft_model: DFlashVLMModel,
    config: DFlashTrainConfig,
    adapter: Qwen25VLTargetAdapter,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int | None = None,
    trainer_progress: dict[str, Any] | None = None,
) -> None:
    """Save a self-validating draft checkpoint, optionally with resume state."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to save a DFlash checkpoint") from exc
    state = {key: value.detach().cpu().contiguous() for key, value in draft_model.state_dict().items()}
    save_file(state, str(output_dir / "model.safetensors"))
    (output_dir / "dflash_config.json").write_text(
        json.dumps(_checkpoint_metadata(draft_model, config, adapter, step=step), indent=2, sort_keys=True)
    )
    if optimizer is not None and trainer_progress is not None:
        trainer_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            **trainer_progress,
        }
        torch.save(trainer_state, output_dir / "trainer_state.pt")


def load_draft_checkpoint(
    checkpoint_dir: str | Path,
    adapter: Qwen25VLTargetAdapter,
    config: DFlashTrainConfig,
) -> DFlashVLMModel:
    """Load a draft-only checkpoint and validate its target/processor contract."""

    checkpoint_dir = Path(checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "dflash_config.json").read_text())
    for key, expected in (
        ("implementation_version", DFLASH_IMPLEMENTATION_VERSION),
        ("target_model", config.target_model),
        ("target_revision", config.target_revision),
        ("target_hidden_size", adapter.hidden_size),
        ("target_vocab_size", adapter.vocab_size),
        ("context_mode", config.context_mode),
        ("block_size", config.block_size),
        ("tokenizer_fingerprint", adapter.tokenizer_fingerprint()),
        ("processor_fingerprint", adapter.processor_fingerprint()),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"Checkpoint mismatch for {key}: {metadata.get(key)!r} != {expected!r}")
    model = make_draft_model(adapter, config)
    if list(metadata.get("target_layer_ids", [])) != list(getattr(model, "target_layer_ids", [])):
        raise ValueError("Checkpoint target_layer_ids do not match the current target config")
    if int(metadata.get("mask_token_id", -1)) != int(model.mask_token_id):
        raise ValueError("Checkpoint mask_token_id does not match the current tokenizer")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("safetensors is required to load a DFlash checkpoint") from exc
    state = load_file(str(checkpoint_dir / "model.safetensors"), device=str(adapter.device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Invalid DFlash checkpoint; missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _load_training_checkpoint(
    checkpoint_dir: str | Path,
    draft_model: DFlashVLMModel,
    adapter: Qwen25VLTargetAdapter,
    config: DFlashTrainConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    restored = load_draft_checkpoint(checkpoint_dir, adapter, config)
    draft_model.load_state_dict(restored.state_dict(), strict=True)
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        raise ValueError(f"{checkpoint_dir} is export-only and cannot be resumed (trainer_state.pt is missing)")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    # ``torch.load(..., map_location='cpu')`` avoids a transient GPU spike, but
    # Adam's moments must live beside their parameters before the next step.
    for parameter_state in optimizer.state.values():
        for key, value in list(parameter_state.items()):
            if torch.is_tensor(value):
                parameter_state[key] = value.to(adapter.device)
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, config: DFlashTrainConfig):
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _is_oom(exc: RuntimeError) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _chunk_candidates(config: DFlashTrainConfig) -> list[int]:
    candidates = []
    chunk = min(config.anchor_chunk_size, config.num_anchors)
    while True:
        candidates.append(chunk)
        if chunk <= config.min_anchor_chunk_size:
            break
        chunk = max(config.min_anchor_chunk_size, chunk // 2)
    return candidates


def _checkpoint_progress(
    *,
    epoch: int,
    next_group_start: int,
    step: int,
    history: list[dict[str, float]],
    sums: dict[str, float],
    usable: int,
    skipped: int,
    best_acceptance: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "next_group_start": next_group_start,
        "step": step,
        "history": history,
        "epoch_sums": sums,
        "epoch_usable": usable,
        "epoch_skipped": skipped,
        "best_acceptance": best_acceptance,
    }


def _rotate_checkpoints(checkpoint_root: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in checkpoint_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for path in checkpoints[:-keep]:
        shutil.rmtree(path)


def _prepare_group(
    adapter: Qwen25VLTargetAdapter,
    group: list[dict[str, Any]],
    config: DFlashTrainConfig,
    *,
    group_start: int,
) -> tuple[list[tuple[dict[str, Any], PreparedExample]], int]:
    prepared: list[tuple[dict[str, Any], PreparedExample]] = []
    skipped = 0
    for offset, record in enumerate(group):
        try:
            example = adapter.prepare_record(record, max_seq_length=config.max_seq_length)
            if example.response_end - example.response_start < config.block_size:
                raise ValueError(f"response has fewer than block_size={config.block_size} tokens")
            prepared.append((record, example))
        except ValueError as exc:
            skipped += 1
            print(f"[skip] {record.get('id', group_start + offset)}: {exc}")
    return prepared, skipped


def _run_group_with_oom_retry(
    *,
    adapter: Qwen25VLTargetAdapter,
    draft_model: DFlashVLMModel,
    prepared: list[tuple[dict[str, Any], PreparedExample]],
    config: DFlashTrainConfig,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    backward_fn: Callable[[torch.Tensor], None],
) -> tuple[dict[str, float], int]:
    """Retry the *whole* accumulation group after OOM to preserve its gradient."""

    last_oom: RuntimeError | None = None
    for chunk_size in _chunk_candidates(config):
        optimizer.zero_grad(set_to_none=True)
        group_sums = {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}
        try:
            group_scale = 1.0 / len(prepared)
            for record, example in prepared:
                generator = make_anchor_generator(
                    config.seed, epoch, example.sample_id, device=adapter.device
                )
                metrics = train_example(
                    adapter,
                    draft_model,
                    example,
                    config,
                    generator=generator,
                    anchor_chunk_size=chunk_size,
                    backward=True,
                    backward_scale=group_scale,
                    backward_fn=backward_fn,
                )
                for key in group_sums:
                    group_sums[key] += metrics[key]
            return group_sums, chunk_size
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            last_oom = exc
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[oom] retrying accumulation group with anchor_chunk_size={max(config.min_anchor_chunk_size, chunk_size // 2)}")
    assert last_oom is not None
    raise RuntimeError(
        f"DFlash training still OOMs at min_anchor_chunk_size={config.min_anchor_chunk_size}"
    ) from last_oom


def train_records(
    adapter: Qwen25VLTargetAdapter,
    draft_model: DFlashVLMModel,
    records: list[dict[str, Any]],
    config: DFlashTrainConfig,
    *,
    validation_records: list[dict[str, Any]] | None = None,
    accelerator: Any | None = None,
) -> list[dict[str, float]]:
    """Train with deterministic anchors, resumable checkpoints, and validation-best export."""

    if not records:
        raise ValueError("training manifest is empty")
    adapter.freeze()
    draft_model.train()
    optimizer = torch.optim.AdamW(
        draft_model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    optimizer_steps = max(1, math.ceil(len(records) / config.gradient_accumulation_steps) * config.epochs)
    scheduler = _make_scheduler(optimizer, optimizer_steps, config)

    if accelerator is not None:
        draft_model, optimizer, scheduler = accelerator.prepare(draft_model, optimizer, scheduler)
        backward_fn = accelerator.backward
        clip_fn = accelerator.clip_grad_norm_
        is_main = accelerator.is_main_process
        unwrap = accelerator.unwrap_model
    else:
        backward_fn = torch.Tensor.backward
        clip_fn = clip_grad_norm_
        is_main = True
        unwrap = lambda model: model

    output_dir = Path(config.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    if is_main:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    history: list[dict[str, float]] = []
    step = 0
    start_epoch = 0
    resume_group_start = 0
    resume_sums = {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}
    resume_usable = 0
    resume_skipped = 0
    best_acceptance = float("-inf")
    if config.resume_from_checkpoint:
        resume = _load_training_checkpoint(
            config.resume_from_checkpoint,
            unwrap(draft_model),
            adapter,
            config,
            optimizer,
            scheduler,
        )
        step = int(resume["step"])
        start_epoch = int(resume["epoch"])
        resume_group_start = int(resume["next_group_start"])
        history = list(resume.get("history", []))
        resume_sums = dict(resume.get("epoch_sums", resume_sums))
        resume_usable = int(resume.get("epoch_usable", 0))
        resume_skipped = int(resume.get("epoch_skipped", 0))
        best_acceptance = float(resume.get("best_acceptance", best_acceptance))
        print(f"[resume] epoch={start_epoch + 1}, next_group_start={resume_group_start}, step={step}")

    for epoch in range(start_epoch, config.epochs):
        sums = dict(resume_sums) if epoch == start_epoch else {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}
        usable = resume_usable if epoch == start_epoch else 0
        skipped = resume_skipped if epoch == start_epoch else 0
        group_first = resume_group_start if epoch == start_epoch else 0
        for group_start in range(group_first, len(records), config.gradient_accumulation_steps):
            group = records[group_start : group_start + config.gradient_accumulation_steps]
            prepared, group_skipped = _prepare_group(adapter, group, config, group_start=group_start)
            skipped += group_skipped
            if not prepared:
                continue
            group_metrics, used_chunk = _run_group_with_oom_retry(
                adapter=adapter,
                draft_model=draft_model,
                prepared=prepared,
                config=config,
                epoch=epoch,
                optimizer=optimizer,
                backward_fn=backward_fn,
            )
            clip_fn(draft_model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            usable += len(prepared)
            for key in sums:
                sums[key] += group_metrics[key]
            if used_chunk != config.anchor_chunk_size:
                print(f"[oom-recovery] optimizer_step={step}, anchor_chunk_size={used_chunk}")
            if step % config.save_every_steps == 0 and is_main:
                checkpoint_dir = checkpoint_root / f"step-{step:08d}"
                save_draft_checkpoint(
                    checkpoint_dir,
                    unwrap(draft_model),
                    config,
                    adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    trainer_progress=_checkpoint_progress(
                        epoch=epoch,
                        next_group_start=group_start + config.gradient_accumulation_steps,
                        step=step,
                        history=history,
                        sums=sums,
                        usable=usable,
                        skipped=skipped,
                        best_acceptance=best_acceptance,
                    ),
                )
                _rotate_checkpoints(checkpoint_root, config.keep_last_checkpoints)
        if usable == 0:
            raise RuntimeError("No usable records remained after manifest/anchor/length validation")
        epoch_metrics: dict[str, float] = {
            "epoch": float(epoch + 1),
            "loss": sums["loss"] / usable,
            "token_accuracy": sums["token_accuracy"] / usable,
            "valid_tokens": sums["valid_tokens"] / usable,
            "usable_records": float(usable),
            "skipped_records": float(skipped),
            "optimizer_step": float(step),
        }
        if validation_records:
            from .evaluate import evaluate_records

            draft_model.eval()
            validation = evaluate_records(adapter, unwrap(draft_model), validation_records, config, epoch=epoch)
            draft_model.train()
            epoch_metrics.update({f"validation_{key}": value for key, value in validation.items()})
            acceptance = validation["accepted_prefix"]
            if acceptance > best_acceptance:
                best_acceptance = acceptance
                if is_main:
                    save_draft_checkpoint(output_dir / "best", unwrap(draft_model), config, adapter, step=step)
        history.append(epoch_metrics)
        if is_main:
            print(json.dumps(epoch_metrics, sort_keys=True))
            checkpoint_dir = checkpoint_root / f"epoch-{epoch + 1:03d}-step-{step:08d}"
            save_draft_checkpoint(
                checkpoint_dir,
                unwrap(draft_model),
                config,
                adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                trainer_progress=_checkpoint_progress(
                    epoch=epoch + 1,
                    next_group_start=0,
                    step=step,
                    history=history,
                    sums={"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0},
                    usable=0,
                    skipped=0,
                    best_acceptance=best_acceptance,
                ),
            )
            _rotate_checkpoints(checkpoint_root, config.keep_last_checkpoints)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True))
            save_draft_checkpoint(output_dir, unwrap(draft_model), config, adapter, step=step)
        resume_group_start = 0
        resume_sums = {"loss": 0.0, "token_accuracy": 0.0, "valid_tokens": 0.0}
        resume_usable = 0
        resume_skipped = 0
    return history


def main() -> None:  # pragma: no cover - exercised by CLI users
    import argparse

    parser = argparse.ArgumentParser(description="Train a DFlash Qwen2.5-VL draft adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    config = DFlashTrainConfig.from_file(args.config)
    if args.resume_from_checkpoint is not None:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    try:
        from accelerate import Accelerator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("accelerate is required for the training CLI") from exc
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    seed_everything(config.seed)
    adapter = Qwen25VLTargetAdapter.from_pretrained(config, device=accelerator.device)
    draft = make_draft_model(adapter, config)
    records = load_jsonl(config.train_manifest)
    validation_records = load_jsonl(config.validation_manifest) if config.validation_manifest else None
    train_records(
        adapter,
        draft,
        records,
        config,
        validation_records=validation_records,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
