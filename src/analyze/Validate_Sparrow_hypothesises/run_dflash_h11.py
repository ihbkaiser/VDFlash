"""Capture DFlash H1.1 representations from Phase 2 features or video runs.

Run `reference` and either `evaluate-cache` or `evaluate` with the same checkpoint.
`reference` reads existing SpecForge .ckpt/.ckpt.gz teacher features; neither
command trains weights or changes the frozen Qwen target.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.infer.qwen25vl_dflash_compare import (
    _ensure_specforge_importable,
    _load_draft,
    _load_target,
    _resolve_dtype,
)
from src.analyze.Validate_Sparrow_hypothesises.run_dflash_h21_h31 import (
    DEFAULT_CALIBRATION,
    DEFAULT_DRAFT_CONFIG,
    DEFAULT_MANIFEST,
    DEFAULT_TARGET_MODEL,
    DEFAULT_VIDEO_ROOT,
    _load_records,
    _run_one,
)
from src.workspace import resolve_workspace_path


@contextmanager
def capture_first_block(draft: Any):
    """Capture mask query 1 after attention and after each decoder layer."""

    vectors: dict[tuple[int, str], np.ndarray] = {}
    weights: dict[int, np.ndarray] = {}
    seen: set[tuple[int, str]] = set()
    handles = []

    def hook(layer_index: int, site: str):
        def record(_module, _inputs, output):
            key = layer_index, site
            if key in seen:
                return
            seen.add(key)
            value = output[0] if site == "attn_out" else output
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 2:
                raise ValueError("H1.1 needs one batch with a block of at least two queries")
            vectors[key] = value[0, 1].detach().float().cpu().numpy().copy()
            if site == "attn_out" and len(output) > 1 and torch.is_tensor(output[1]):
                weights[layer_index] = output[1][0, :, 1, :].detach().float().cpu().numpy().copy()
        return record

    for i, layer in enumerate(draft.layers):
        handles.append(layer.self_attn.register_forward_hook(hook(i, "attn_out")))
        handles.append(layer.register_forward_hook(hook(i, "layer_out")))
    try:
        yield vectors, weights
    finally:
        for handle in handles:
            handle.remove()


def stack_vectors(records: dict[tuple[int, str], np.ndarray], layer_count: int) -> np.ndarray:
    from .analyze_dflash_h11 import SITES

    expected = {(layer, site) for layer in range(layer_count) for site in SITES}
    if set(records) != expected:
        raise ValueError(f"missing/extra draft hooks: {sorted(expected ^ set(records))}")
    result = np.stack([
        np.stack([records[layer, site] for site in SITES])
        for layer in range(layer_count)
    ]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite captured draft representation")
    return result


def first_valid_anchor(loss_mask: torch.Tensor) -> int:
    valid = ((loss_mask[:-1] > 0.5) & (loss_mask[1:] > 0.5)).nonzero().flatten()
    if valid.numel() == 0 or int(valid[0]) == 0:
        raise ValueError("feature has no usable two-token supervised anchor")
    return int(valid[0])


def _cached_forward(path: Path, *, draft: Any, target: Any, device: torch.device,
                    max_length: int, cut: bool = False,
                    visual_token_ids: set[int] | None = None,
                    score_proposals: bool = True):
    from specforge.runtime.data_plane.feature_store import load_feature_file
    from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask

    raw = load_feature_file(str(path))
    input_ids = raw["input_ids"].reshape(-1)[:max_length].to(dtype=torch.long)
    loss_mask = raw["loss_mask"].reshape(-1)[:max_length]
    hidden = raw["hidden_states"]
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        hidden = hidden[0]
    hidden = hidden[:max_length]
    pos = raw["position_ids"]
    if pos.ndim == 3 and pos.shape[1] == 1:
        pos = pos[:, 0, :]
    pos = pos[..., :max_length]
    if pos.ndim != 2 or pos.shape[0] != 3:
        raise ValueError(f"expected three-axis Qwen2.5-VL positions: {path}")
    if len(input_ids) != len(loss_mask) or hidden.shape[0] != len(input_ids) or pos.shape[1] != len(input_ids):
        raise ValueError(f"inconsistent feature lengths: {path}")
    anchor = first_valid_anchor(loss_mask)
    visual_positions = [
        i for i, token in enumerate(input_ids[:anchor].tolist())
        if int(token) in (visual_token_ids or set())
    ]
    if cut and not visual_positions:
        raise ValueError(f"no visual tokens before supervised anchor: {path}")
    if cut and draft.sliding_window:
        raise NotImplementedError("cache-only Cut for sliding-window drafts needs position-aware masking")
    keep = torch.ones(anchor, dtype=torch.bool)
    if cut:
        keep[visual_positions] = False
    context_length = int(keep.sum())
    if context_length == 0:
        raise ValueError(f"cut removed the complete draft context: {path}")
    block_size = int(draft.block_size)
    noise_ids = torch.full((1, block_size), int(draft.mask_token_id), dtype=torch.long)
    noise_ids[0, 0] = input_ids[anchor]
    embed = target.get_input_embeddings()
    embed_device = next(embed.parameters()).device
    noise = embed(noise_ids.to(embed_device)).to(device)
    offsets = torch.arange(block_size).view(1, 1, -1)
    draft_positions = pos[:, anchor].view(3, 1, 1) + offsets
    all_positions = torch.cat([pos[:, :anchor][:, keep].unsqueeze(1), draft_positions], dim=-1).to(device)
    attention_mask = create_dflash_sdpa_mask(
        torch.tensor([[context_length]], device=device),
        torch.tensor([[True]], device=device),
        context_length, block_size, device,
        sliding_window=draft.sliding_window if draft.sliding_window else None,
    )
    if draft.sliding_window:
        full_mask = create_dflash_sdpa_mask(
            torch.tensor([[context_length]], device=device), torch.tensor([[True]], device=device),
            context_length, block_size, device,
        )
        attention_mask = {"full_attention": full_mask, "sliding_attention": attention_mask}
    with capture_first_block(draft) as (vectors, weights), torch.inference_mode():
        draft_hidden = draft(
            position_ids=all_positions,
            attention_mask=attention_mask,
            noise_embedding=noise,
            target_hidden=hidden[:anchor][keep].unsqueeze(0).to(device=device, dtype=noise.dtype),
        )
        # Cache-only acceptance is against stored teacher tokens; there is no
        # fresh target verification or video decoding in this path.
        if score_proposals:
            lm_head = target.get_output_embeddings()
            head_device = next(lm_head.parameters()).device
            proposals = lm_head(draft_hidden[:, 1:].to(head_device)).argmax(dim=-1)[0].cpu()
    labels = input_ids[anchor + 1 : anchor + block_size]
    supervised = loss_mask[anchor + 1 : anchor + block_size] > 0.5
    comparable = min(block_size - 1, len(labels))
    accepted = None
    if score_proposals:
        accepted = 0
        for index in range(comparable):
            if not bool(supervised[index]) or int(proposals[index]) != int(labels[index]):
                break
            accepted += 1
    prompt_hash = hashlib.sha256(input_ids[: anchor + 1].numpy().tobytes()).hexdigest()
    continuation_hash = hashlib.sha256(labels.numpy().tobytes()).hexdigest()
    return {
        "vectors": stack_vectors(vectors, len(draft.layers)),
        "weights": weights, "anchor": anchor,
        "visual_positions": visual_positions if not cut else [],
        "visual_count": len(visual_positions),
        "accepted": accepted, "proposal_count": comparable,
        "target_input_fingerprint": prompt_hash,
        "target_output_hash": continuation_hash,
    }


def _visual_token_ids(target: Any) -> set[int]:
    config = getattr(target, "config", None)
    values = set()
    for name in ("image_token_id", "image_token_index", "video_token_id", "video_token_index"):
        token = getattr(config, name, None)
        if token is not None:
            if isinstance(token, (list, tuple)):
                values.update(int(item) for item in token)
            else:
                values.add(int(token))
    if not values:
        raise ValueError("target config does not identify image/video token IDs")
    return values


def attention_summary(weights: dict[int, np.ndarray], visual_positions: list[int], block_size: int):
    result = {}
    for layer, value in weights.items():
        context_length = value.shape[-1] - block_size
        if context_length < 0:
            raise ValueError("attention key length shorter than draft block")
        visual = set(int(p) for p in visual_positions)
        if any(p < 0 or p >= context_length for p in visual):
            raise ValueError("visual positions do not match compacted context")
        text = [p for p in range(context_length) if p not in visual]
        head_mean = value.mean(axis=0)
        text_weights = head_mean[text]
        mass = float(text_weights.sum())
        conditional = text_weights / mass if mass > 0 else text_weights
        top5 = float(np.sort(conditional)[-5:].sum()) if len(conditional) else 0.0
        result[str(layer)] = {
            "text_mass": mass,
            "top5_text_mass_conditional": top5,
            "context_length": context_length,
            "text_keys": len(text),
            "visual_keys": len(visual),
        }
    return result


def _paths(args):
    args.checkpoint = resolve_workspace_path(args.checkpoint)
    args.draft_config = resolve_workspace_path(args.draft_config)
    args.output_dir = resolve_workspace_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"reference", "evaluate-cache"}:
        args.feature_root = resolve_workspace_path(args.feature_root)
    else:
        args.manifest = resolve_workspace_path(args.manifest)
        args.video_root = resolve_workspace_path(args.video_root)
        if args.calibration:
            args.calibration = resolve_workspace_path(args.calibration)


def run_reference(args):
    _ensure_specforge_importable()
    from specforge.runtime.data_plane.offline_reader import list_feature_files
    from specforge.hidden_state import validate_hidden_state_metadata

    paths = [Path(p) for p in list_feature_files(str(args.feature_root))]
    if len(paths) < args.reference_samples:
        raise ValueError(f"only {len(paths)} feature files, requested {args.reference_samples}")
    with Path(args.draft_config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_hidden_state_metadata(
        str(args.feature_root),
        target_layer_ids=config["dflash_config"]["target_layer_ids"],
        hidden_size=int(config["hidden_size"]), expected_phase="phase2",
    )
    rng = np.random.default_rng(args.seed)
    chosen = [paths[int(i)] for i in rng.permutation(len(paths))]
    dtype = _resolve_dtype(args.dtype, torch.device(args.device))
    # Inspect the actual weight keys before model construction. Changing a
    # three-layer JSON to five layers cannot manufacture missing trained layers.
    _ensure_specforge_importable()
    from specforge.export.checkpoint_io import materialize_draft, resolve_training_state

    state = resolve_training_state(str(args.checkpoint))
    expected_layers = int(config["num_hidden_layers"])
    trained_layers = {
        int(match.group(1))
        for key in state["draft_state_dict"]
        if (match := re.match(r"^layers\.(\d+)\.", key))
    }
    if trained_layers and trained_layers != set(range(expected_layers)):
        raise ValueError(
            f"checkpoint has draft layers {sorted(trained_layers)}, but the config "
            f"requires layers 0..{expected_layers - 1}; use a matching checkpoint"
        )
    draft = materialize_draft(state, str(args.draft_config))
    del state
    draft.to(device=torch.device(args.draft_device), dtype=dtype).eval()
    print(f"[H1.1] checkpoint/config loaded {expected_layers} draft layers", flush=True)
    _processor, target, _ = _load_target(
        args.target_model, device=torch.device(args.device), dtype=dtype,
        attention=args.target_attention, device_map=args.device_map, max_memory=args.max_memory,
    )
    draft.config._attn_implementation = "eager"
    vectors, metadata = [], []
    skipped = 0
    for path in chosen:
        try:
            captured = _cached_forward(
                path, draft=draft, target=target, device=torch.device(args.draft_device),
                max_length=args.train_max_length, score_proposals=False,
            )
        except ValueError as exc:
            if "no usable two-token supervised anchor" not in str(exc):
                raise
            skipped += 1
            continue
        vectors.append(captured["vectors"])
        metadata.append({"feature_file": str(path), "anchor": captured["anchor"]})
        if len(vectors) % 25 == 0:
            print(f"[H1.1 reference] {len(vectors)}/{args.reference_samples}", flush=True)
        if len(vectors) == args.reference_samples:
            break
    if len(vectors) != args.reference_samples:
        raise ValueError(f"only {len(vectors)} usable training features; skipped {skipped}")
    np.savez_compressed(args.output_dir / "train_full.npz", vectors=np.stack(vectors), checkpoint=str(args.checkpoint))
    (args.output_dir / "train_full_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_evaluate_cache(args):
    _ensure_specforge_importable()
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    metadata_path = args.output_dir / "train_full_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("run reference first in this output directory")
    training_files = {
        str(Path(row["feature_file"]).resolve())
        for row in json.loads(metadata_path.read_text(encoding="utf-8"))
    }
    candidates = [Path(path) for path in list_feature_files(str(args.feature_root))
                  if str(Path(path).resolve()) not in training_files]
    if len(candidates) < args.limit:
        raise ValueError("not enough held-out feature files after reference selection")
    rng = np.random.default_rng(args.seed + 1)
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    dtype = _resolve_dtype(args.dtype, torch.device(args.device))
    _processor, target, _ = _load_target(
        args.target_model, device=torch.device(args.device), dtype=dtype,
        attention=args.target_attention, device_map=args.device_map, max_memory=args.max_memory,
    )
    draft, _ = _load_draft(
        str(args.checkpoint), str(args.draft_config), device=torch.device(args.draft_device), dtype=dtype,
    )
    draft.config._attn_implementation = "eager"
    visual_ids = _visual_token_ids(target)
    full_vectors, cut_vectors, sample_ids, reports = [], [], [], []
    skipped = 0
    for path in candidates:
        try:
            full = _cached_forward(
                path, draft=draft, target=target, device=torch.device(args.draft_device),
                max_length=args.train_max_length, visual_token_ids=visual_ids,
            )
            if not full["visual_count"]:
                skipped += 1
                continue
            cut = _cached_forward(
                path, draft=draft, target=target, device=torch.device(args.draft_device),
                max_length=args.train_max_length, cut=True, visual_token_ids=visual_ids,
            )
        except ValueError as exc:
            if "no usable two-token supervised anchor" not in str(exc):
                raise
            skipped += 1
            continue
        sample_id = str(path.relative_to(args.feature_root))
        for condition, captured in (("full", full), ("cut", cut)):
            if len(captured["weights"]) != len(draft.layers):
                raise RuntimeError("eager attention did not return weights for every draft layer")
            reports.append({
                "sample_id": sample_id, "visual_condition": condition,
                "status": "ok", "verification_mode": "cached_teacher_tokens",
                "target_input_fingerprint": captured["target_input_fingerprint"],
                "target_output_hash": captured["target_output_hash"],
                "visual_positions_before_cut": full["visual_positions"],
                "visual_count_before_cut": full["visual_count"],
                "anchor": captured["anchor"],
                "acceptance_rounds": [{
                    "matched_proposals": captured["accepted"],
                    "proposal_count": captured["proposal_count"],
                }],
                "h11_attention": attention_summary(
                    captured["weights"], captured["visual_positions"], int(draft.block_size)
                ),
            })
        full_vectors.append(full["vectors"])
        cut_vectors.append(cut["vectors"])
        sample_ids.append(sample_id)
        print(f"[H1.1 cache evaluation] {len(sample_ids)}/{args.limit} {sample_id}", flush=True)
        if len(sample_ids) == args.limit:
            break
    if len(sample_ids) != args.limit:
        raise ValueError(f"only {len(sample_ids)} usable held-out visual features; skipped {skipped}")
    np.savez_compressed(
        args.output_dir / "test_full_cut.npz",
        full=np.stack(full_vectors), cut=np.stack(cut_vectors),
        sample_ids=np.asarray(sample_ids), checkpoint=str(args.checkpoint),
    )
    with (args.output_dir / "test_full_cut.jsonl").open("w", encoding="utf-8") as handle:
        for row in reports:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_evaluate(args):
    records = _load_records(args.manifest, args.video_root, args.calibration, "vdc")
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("no VDC test records selected")
    dtype = _resolve_dtype(args.dtype, torch.device(args.device))
    processor, target, _ = _load_target(
        args.target_model, device=torch.device(args.device), dtype=dtype,
        attention=args.target_attention, device_map=args.device_map, max_memory=args.max_memory,
    )
    draft, _ = _load_draft(
        str(args.checkpoint), str(args.draft_config), device=torch.device(args.draft_device), dtype=dtype,
    )
    draft.config._attn_implementation = "eager"
    args.experiment = "h3_1"
    args.dataset = "vdc"
    args.training_corpus = "llava68k"
    args.draft_depth = len(draft.layers)
    args.capture_attention = False
    # _run_one requires only VDC-specific settings with dataset=vdc.
    args.mvbench_num_frames = 8
    args.mvbench_min_pixels = 256 * 28 * 28
    args.mvbench_max_pixels = 256 * 28 * 28
    paired, sample_ids, report_rows = [], [], []
    for record in records:
        target_cache = {}
        pair = {}
        fingerprints = set()
        target_hashes = set()
        for condition in ("full", "cut"):
            with capture_first_block(draft) as (vectors, weights):
                row = _run_one(
                    args=args, sample_record=record, prompt_variant="natural",
                    condition={"visual_condition": condition, "retention_percentage": 100.0 if condition == "full" else 0.0},
                    target=target, draft=draft, processor=processor,
                    device=torch.device(args.device), draft_device=torch.device(args.draft_device),
                    target_cache=target_cache,
                )
            if row["status"] != "ok" or not row["lossless"]:
                raise RuntimeError(f"invalid H1.1 run: {record['sample_id']} {condition}")
            if len(weights) != len(draft.layers):
                raise RuntimeError("eager attention did not return weights for every draft layer")
            fingerprints.add(row["target_input_fingerprint"])
            target_hashes.add(row["target_output_hash"])
            row["h11_attention"] = attention_summary(
                weights, row["visual_positions"], int(draft.block_size)
            )
            report_rows.append(row)
            pair[condition] = stack_vectors(vectors, len(draft.layers))
        if len(fingerprints) != 1 or len(target_hashes) != 1:
            raise RuntimeError(f"unpaired target for {record['sample_id']}")
        sample_ids.append(str(record["sample_id"]))
        paired.append(pair)
        print(f"[H1.1 evaluation] {len(paired)}/{len(records)} {record['sample_id']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    np.savez_compressed(
        args.output_dir / "test_full_cut.npz",
        full=np.stack([p["full"] for p in paired]), cut=np.stack([p["cut"] for p in paired]),
        sample_ids=np.asarray(sample_ids), checkpoint=str(args.checkpoint),
    )
    with (args.output_dir / "test_full_cut.jsonl").open("w", encoding="utf-8") as handle:
        for row in report_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "evaluate-cache", "evaluate"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", default=str(DEFAULT_DRAFT_CONFIG))
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--feature-root")
    parser.add_argument("--reference-samples", type=int, default=200)
    parser.add_argument("--train-max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--target-visual-tokens", type=int, default=3000)
    parser.add_argument("--allow-out-of-tolerance", action="store_true")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--device-map", choices=("cuda", "auto", "model_parallel"), default="model_parallel")
    parser.add_argument("--max-memory", default="0:22GiB,1:14GiB")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode in {"reference", "evaluate-cache"} and not args.feature_root:
        raise ValueError("--feature-root is required for reference/evaluate-cache")
    if args.mode == "reference" and args.reference_samples < 40:
        raise ValueError("--reference-samples must be at least 40")
    if args.mode == "reference" and args.train_max_length < 2:
        raise ValueError("--train-max-length must be at least 2")
    if args.mode in {"evaluate", "evaluate-cache"} and (args.limit is not None and args.limit < 1):
        raise ValueError("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("H1.1 capture requires CUDA and the matching target/checkpoint")
    _paths(args)
    if args.mode == "reference":
        run_reference(args)
    elif args.mode == "evaluate-cache":
        run_evaluate_cache(args)
    else:
        run_evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
