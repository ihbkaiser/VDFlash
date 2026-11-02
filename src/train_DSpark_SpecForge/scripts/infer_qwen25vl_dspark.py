#!/usr/bin/env python3
"""Greedy Qwen2.5-VL inference with the repository's DSpark drafter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_DSPARK_ROOT = _WORKSPACE_ROOT / "src" / "train_DSpark_SpecForge"
for _import_root in (_WORKSPACE_ROOT, _DSPARK_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from src.workspace import resolve_workspace_path
from specforge.export.checkpoint_io import materialize_draft, resolve_training_state
from specforge.qwen25vl import prepare_inference_prompt


def _load_target(path: str, revision: str | None, device_map: str):
    try:
        from transformers import AutoModelForImageTextToText as AutoTargetModel
    except ImportError:  # pragma: no cover - compatibility with older Transformers
        try:
            from transformers import AutoModelForVision2Seq as AutoTargetModel
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoTargetModel

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "attn_implementation": "sdpa",
    }
    if revision:
        kwargs["revision"] = revision
    try:
        return AutoTargetModel.from_pretrained(path, **kwargs)
    except (AttributeError, ImportError, ValueError):
        if AutoTargetModel.__name__ != "AutoModelForImageTextToText":
            raise
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(path, **kwargs)


def _load_config_and_validate_checkpoint(state: dict[str, Any], config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != ["DSparkDraftModel"]:
        raise ValueError(
            "draft config must declare architectures=['DSparkDraftModel']; "
            f"got {config.get('architectures')!r}"
        )
    if state.get("strategy") != "dspark":
        raise ValueError(
            "refusing to run a non-DSpark training state: "
            f"strategy={state.get('strategy')!r}"
        )

    dspark = config.get("dflash_config") or {}
    comparisons = {
        "dspark_draft_num_hidden_layers": int(config["num_hidden_layers"]),
        "dspark_target_layer_ids": tuple(
            int(value) for value in dspark.get("target_layer_ids", ())
        ),
        "dspark_block_size": int(config["block_size"]),
        "dspark_mask_token_id": int(dspark["mask_token_id"]),
    }
    mismatches = {
        key: {"checkpoint": state.get(key), "config": value}
        for key, value in comparisons.items()
        if state.get(key) != value
    }
    if dspark.get("projector_type") != "dspark":
        mismatches["dflash_config.projector_type"] = {
            "checkpoint": state.get("strategy"),
            "config": dspark.get("projector_type"),
        }
    if mismatches:
        raise ValueError(f"checkpoint/config mismatch: {json.dumps(mismatches)}")

    state_dict = state.get("draft_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("checkpoint is missing a non-empty draft_state_dict")
    markov_weight = state_dict.get("markov_head.markov_w1.weight")
    confidence_weight = state_dict.get("confidence_head.proj.weight")
    expected_rank = int(dspark.get("markov_rank", 0))
    expected_vocab = int(config["vocab_size"])
    if markov_weight is None or tuple(markov_weight.shape) != (
        expected_vocab,
        expected_rank,
    ):
        raise ValueError(
            "checkpoint Markov embedding does not match draft config: "
            f"shape={getattr(markov_weight, 'shape', None)}, "
            f"expected={(expected_vocab, expected_rank)}"
        )
    if bool(dspark.get("enable_confidence_head")) and confidence_weight is None:
        raise ValueError("DSpark config enables confidence head but checkpoint lacks it")
    if confidence_weight is not None:
        confidence_input = int(config["hidden_size"])
        if dspark.get("confidence_head_with_markov"):
            confidence_input += expected_rank
        if tuple(confidence_weight.shape) != (1, confidence_input):
            raise ValueError(
                "checkpoint confidence head does not match draft config: "
                f"shape={tuple(confidence_weight.shape)}, "
                f"expected={(1, confidence_input)}"
            )
    return config


def _checkpoint_config_digest(state: dict[str, Any]) -> tuple[str | None, int | None]:
    try:
        provenance = dict(state.get("specforge_model_provenance") or ())
        fingerprint = provenance["draft_config"]
        content = fingerprint[2]
        if content[0] == "sha256":
            return str(content[3]), int(content[2])
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return None, None


def _target_fingerprints(state: dict[str, Any]):
    try:
        provenance = dict(state.get("specforge_model_provenance") or ())
        model = provenance["target_model"]
        return model[2]
    except (KeyError, IndexError, TypeError):
        return None


def _target_revision(state: dict[str, Any]) -> str | None:
    try:
        provenance = dict(state.get("specforge_model_provenance") or ())
        revision = provenance.get("target_revision")
        return str(revision) if revision else None
    except (TypeError, ValueError):
        return None


def _verify_target_snapshot(
    state: dict[str, Any], snapshot_path: Path, *, allow_mismatch: bool
) -> list[str]:
    fingerprints = _target_fingerprints(state)
    if not fingerprints:
        message = "checkpoint has no target-model file provenance"
        if not allow_mismatch:
            raise ValueError(
                f"{message}; pass --allow-target-mismatch only if intentional"
            )
        return [message]
    mismatches = []
    for item in fingerprints:
        if not isinstance(item, (tuple, list)) or len(item) < 3:
            mismatches.append(f"malformed target provenance record: {item!r}")
            continue
        method, relative_name, expected_size = item[:3]
        if method not in {"sha256", "stat"}:
            mismatches.append(
                f"unsupported target provenance method {method!r} for {relative_name}"
            )
            continue
        local_file = snapshot_path / str(relative_name)
        if not local_file.is_file():
            mismatches.append(f"missing target file {relative_name}")
            continue
        actual_size = local_file.stat().st_size
        if actual_size != int(expected_size):
            mismatches.append(
                f"{relative_name} size {actual_size} != checkpoint {expected_size}"
            )
            continue
        if method == "sha256":
            if len(item) < 4:
                mismatches.append(f"missing target SHA-256 for {relative_name}")
                continue
            expected_hash = str(item[3])
            digest = hashlib.sha256()
            with local_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                mismatches.append(
                    f"{relative_name} SHA-256 {actual_hash} != checkpoint {expected_hash}"
                )
    if mismatches and not allow_mismatch:
        raise ValueError(
            "target model does not match checkpoint provenance:\n- "
            + "\n- ".join(mismatches)
            + "\nUse --allow-target-mismatch only if this difference is intentional."
        )
    return mismatches


def _resolve_target_snapshot(path: str, revision: str | None) -> Path:
    local_path = Path(path).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": path}
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs)).resolve()


def _materialize_draft(state: dict[str, Any], config_path: Path):
    from specforge.hidden_state import resolve_dflash_config

    depth = int(state["dspark_draft_num_hidden_layers"])
    target_layers = tuple(int(value) for value in state["dspark_target_layer_ids"])
    block_size = int(state["dspark_block_size"])
    expected_hash, expected_size = _checkpoint_config_digest(state)
    with tempfile.TemporaryDirectory(prefix="vdflash-dspark-config-") as temporary:
        resolved_path = Path(temporary) / "draft_config.json"
        resolved = resolve_dflash_config(
            config_path,
            str(resolved_path),
            target_layer_ids=target_layers,
            draft_num_hidden_layers=depth,
            phase="inference",
        )
        resolved["block_size"] = block_size
        resolved_path.write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        actual_size = resolved_path.stat().st_size
        actual_hash = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
        if expected_size is not None and actual_size != expected_size:
            raise ValueError(
                "resolved draft config size differs from checkpoint provenance: "
                f"{actual_size} != {expected_size}"
            )
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(
                "resolved draft config SHA-256 differs from checkpoint provenance: "
                f"{actual_hash} != {expected_hash}; pass the matching --draft-config"
            )
        draft = materialize_draft(state, str(resolved_path)).eval()
    return draft, resolved


def _stop_token_ids(target, processor) -> list[int]:
    value = getattr(target.generation_config, "eos_token_id", None)
    if value is None:
        value = getattr(processor.tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("target model and tokenizer have no EOS token ID")
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({int(token_id) for token_id in values if token_id is not None})


def _move(value: Any, device: torch.device):
    return value.to(device) if torch.is_tensor(value) else value


def _records(args: argparse.Namespace):
    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        record = {
            "id": args.sample_id or image_path.stem,
            "image": image_path.name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path.name},
                        {"type": "text", "text": args.prompt or "Describe this image."},
                    ],
                }
            ],
        }
        return [(0, record, image_path.parent)]

    manifest = resolve_workspace_path(args.manifest)
    image_root = resolve_workspace_path(args.image_root)
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    start = args.start_index
    if start < 0 or start >= len(records):
        raise IndexError(f"start-index {start} is outside [0, {len(records)})")
    end = len(records) if args.num_samples is None else min(
        len(records), start + args.num_samples
    )
    if end <= start:
        raise ValueError("num-samples must be positive")
    return [(index, records[index], image_root) for index in range(start, end)]


def _load_profile(path: str | None) -> dict[int, float] | None:
    if path is None:
        return None
    profile_path = resolve_workspace_path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    rates = payload.get("step_rate_by_batch", payload)
    if not isinstance(rates, dict):
        raise ValueError("throughput profile must be a JSON object")
    return {int(batch): float(rate) for batch, rate in rates.items()}


def _resolve_checkpoint_path(checkpoint: str | None) -> Path:
    if checkpoint:
        return resolve_workspace_path(checkpoint)
    artifact_root = Path(
        os.environ.get(
            "DSPARK_ARTIFACT_ROOT",
            _WORKSPACE_ROOT / "artifacts" / "ihbkaiser_qwen25vl-3b-msd-checkpoints",
        )
    ).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = resolve_workspace_path(str(artifact_root))
    extract_root = Path(
        os.environ.get("DSPARK_EXTRACT_ROOT", artifact_root / "extracted_dspark")
    ).expanduser()
    if not extract_root.is_absolute():
        extract_root = resolve_workspace_path(str(extract_root))
    selection_path = extract_root / "selected_checkpoint.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"no --checkpoint supplied and no selection file exists at {selection_path}; "
            "run scripts/prepare_dspark_inference.sh first"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    relative = Path(selection["checkpoint_dir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe checkpoint path in {selection_path}")
    return extract_root / relative


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="one local image for direct inference")
    source.add_argument("--manifest", help="Qwen2.5-VL JSONL image manifest")
    parser.add_argument("--prompt", help="prompt for --image")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--image-root", help="root for relative manifest image paths")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--target-model-path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="local model directory or Hugging Face repository ID",
    )
    parser.add_argument("--target-revision", default=None)
    parser.add_argument(
        "--allow-target-mismatch",
        action="store_true",
        help="continue when target files do not match checkpoint provenance",
    )
    parser.add_argument(
        "--draft-config",
        default=str(_DSPARK_ROOT / "configs" / "qwen2.5-vl-3b-dspark.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DSPARK_CHECKPOINT"),
        help="checkpoint directory or training_state.pt; defaults to the selected archive checkpoint",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument(
        "--scheduler",
        choices=("fixed", "threshold", "profile"),
        default="fixed",
        help="fixed block, cumulative-confidence threshold, or profiled SPS schedule",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--throughput-profile",
        help=(
            "JSON object with step_rate_by_batch for verification batches "
            "1..block_size+1"
        ),
    )
    parser.add_argument(
        "--skip-target-verification",
        action="store_true",
        help="skip the greedy target-only reference run",
    )
    parser.add_argument("--output-jsonl", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.manifest and not args.image_root:
        raise ValueError("--image-root is required with --manifest")
    if args.image_root and not args.manifest:
        raise ValueError("--image-root is only valid with --manifest")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.image_min_pixels <= 0 or args.image_max_pixels < args.image_min_pixels:
        raise ValueError("image pixel limits must be positive and max >= min")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DSpark Qwen2.5-VL inference requires a CUDA GPU; run this command "
            "inside a GPU allocation"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the checkpoint uses BF16; select a CUDA GPU with BF16 support")

    draft_config_path = resolve_workspace_path(args.draft_config)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    state = resolve_training_state(str(checkpoint_path))
    checkpoint_step = int(state.get("global_step", -1))
    draft_config = _load_config_and_validate_checkpoint(state, draft_config_path)

    checkpoint_target_revision = _target_revision(state)
    if (
        args.target_revision
        and checkpoint_target_revision
        and args.target_revision != checkpoint_target_revision
        and not args.allow_target_mismatch
    ):
        raise ValueError(
            "requested target revision differs from checkpoint provenance: "
            f"{args.target_revision} != {checkpoint_target_revision}"
        )
    target_snapshot = _resolve_target_snapshot(
        args.target_model_path,
        args.target_revision or checkpoint_target_revision,
    )
    target_mismatches = _verify_target_snapshot(
        state,
        target_snapshot,
        allow_mismatch=args.allow_target_mismatch,
    )
    for mismatch in target_mismatches:
        print(f"Target provenance warning: {mismatch}", file=sys.stderr)

    from transformers import AutoConfig, AutoProcessor

    target_config = AutoConfig.from_pretrained(target_snapshot)
    if getattr(target_config, "model_type", None) != "qwen2_5_vl":
        raise ValueError(
            "the checkpoint requires a Qwen2.5-VL target, got model_type="
            f"{getattr(target_config, 'model_type', None)!r}"
        )
    resolved_target_revision = (
        getattr(target_config, "_commit_hash", None)
        or args.target_revision
        or checkpoint_target_revision
    )
    if (
        checkpoint_target_revision
        and resolved_target_revision
        and checkpoint_target_revision != resolved_target_revision
        and not args.allow_target_mismatch
    ):
        raise ValueError(
            "loaded target revision differs from checkpoint provenance: "
            f"{resolved_target_revision} != {checkpoint_target_revision}"
        )
    text_config = getattr(target_config, "text_config", None) or target_config
    expected_hidden = int(draft_config["hidden_size"])
    actual_hidden = int(getattr(text_config, "hidden_size"))
    target_layers = int(getattr(text_config, "num_hidden_layers"))
    target_vocab = int(getattr(text_config, "vocab_size"))
    target_layer_ids = tuple(
        int(value) for value in draft_config["dflash_config"]["target_layer_ids"]
    )
    expected_target_layers = int(draft_config.get("num_target_layers", target_layers))
    if actual_hidden != expected_hidden or target_vocab != int(draft_config["vocab_size"]):
        raise ValueError(
            "target/draft dimensions do not match: "
            f"target hidden={actual_hidden}, vocab={target_vocab}; "
            f"draft hidden={expected_hidden}, vocab={draft_config['vocab_size']}"
        )
    if target_layers != expected_target_layers:
        raise ValueError(
            "target depth does not match the resolved draft config: "
            f"target layers={target_layers}, expected={expected_target_layers}"
        )
    if max(target_layer_ids) >= target_layers:
        raise ValueError(
            f"target layer tap {max(target_layer_ids)} is outside target depth {target_layers}"
        )

    print(
        f"Loading DSpark step={state.get('global_step')} "
        f"layers={target_layer_ids} from {checkpoint_path}",
        file=sys.stderr,
        flush=True,
    )
    draft, draft_config = _materialize_draft(state, draft_config_path)
    del state
    processor = AutoProcessor.from_pretrained(target_snapshot)
    target = _load_target(
        str(target_snapshot),
        None,
        args.device_map,
    ).eval()
    device_map = getattr(target, "hf_device_map", None)
    if device_map:
        mapped_devices = {str(value) for value in device_map.values()}
        if len(mapped_devices) > 1 or mapped_devices & {"cpu", "disk"}:
            raise RuntimeError(
                "the local DSpark verifier requires the target and draft to share "
                "one CUDA device; this target was sharded/offloaded across "
                f"{sorted(mapped_devices)}. Load the target on one sufficiently "
                "large GPU or use a serving integration that supports sharding."
            )
    device = next(target.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(
            f"target model loaded on {device}, but DSpark inference requires CUDA"
        )
    draft.to(device=device, dtype=torch.bfloat16).eval()
    stop_ids = _stop_token_ids(target, processor)
    records = _records(args)
    profile = _load_profile(args.throughput_profile)
    if args.scheduler == "profile" and profile is None:
        raise ValueError("--throughput-profile is required for --scheduler profile")
    if args.scheduler != "profile" and profile is not None:
        raise ValueError("--throughput-profile is only valid for --scheduler profile")

    output_handle = None
    try:
        if args.output_jsonl:
            output_path = resolve_workspace_path(args.output_jsonl)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")
        else:
            output_handle = sys.stdout

        any_mismatch = False
        for index, record, image_root in records:
            prepared = prepare_inference_prompt(
                processor,
                target_config,
                record,
                image_root=image_root,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
            )
            input_ids = prepared["input_ids"].to(device)
            attention_mask = prepared["attention_mask"].to(device)
            position_ids = prepared["position_ids"].to(device)
            media = {
                key: _move(value, device)
                for key, value in prepared["multimodal_inputs"].items()
            }

            with torch.inference_mode():
                reference = None
                if not args.skip_target_verification:
                    reference = target.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **media,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=stop_ids,
                    )
                speculative = draft.spec_generate(
                    target,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=stop_ids,
                    temperature=0.0,
                    position_ids=position_ids,
                    target_kwargs={**media, "attention_mask": attention_mask},
                    confidence_scheduler=args.scheduler,
                    confidence_threshold=args.confidence_threshold,
                    step_rate_by_batch=profile,
                )

            generated = speculative[0, input_ids.shape[1] :]
            reference_tokens = (
                reference[0, input_ids.shape[1] :] if reference is not None else None
            )
            matches = (
                bool(torch.equal(generated, reference_tokens))
                if reference_tokens is not None
                else None
            )
            any_mismatch = any_mismatch or matches is False
            sample_id = (
                record.get("id")
                or record.get("sample_id")
                or record.get("video_name")
                or f"manifest-{index}"
            )
            result = {
                "sample_id": sample_id,
                "checkpoint_step": checkpoint_step,
                "target_model": args.target_model_path,
                "target_revision": resolved_target_revision,
                "scheduler": args.scheduler,
                "greedy_match": matches,
                "generated_token_ids": generated.tolist(),
                "text": processor.tokenizer.decode(
                    generated, skip_special_tokens=True
                ),
            }
            if reference_tokens is not None:
                result["target_token_ids"] = reference_tokens.tolist()
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()

        if any_mismatch:
            print(
                "DSpark greedy tokens differed from target-only generation for "
                "one or more samples",
                file=sys.stderr,
            )
            return 2
        print(
            f"Completed {len(records)} sample(s); "
            f"greedy verification={'skipped' if args.skip_target_verification else 'passed'}",
            file=sys.stderr,
        )
        return 0
    finally:
        if output_handle is not None and output_handle is not sys.stdout:
            output_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
