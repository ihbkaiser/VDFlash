"""All-layer visual-attention report for Qwen2.5-VL-3B.

This runner measures the attention emitted by the final instruction token to
the video-token columns during one native multimodal prefill.  Every decoder
layer is recorded with the model's native, zero-based index; no layer cutoff
or ablation is applied.  Raw per-head values are kept in JSONL and the report
plot uses a separately documented global normalization for display.

The GPU/model imports are lazy so the schema, aggregation, and plotting
helpers remain testable in a CPU-only environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from .figure3a_qwen25vl3b import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_TASKS,
    build_mvbench_prompt,
    read_mvbench_records,
)


DEFAULT_OUTPUT_DIR = "results/figure3b_qwen25vl3b_visual_attention_20260822"
DEFAULT_OUTPUT = f"{DEFAULT_OUTPUT_DIR}/visual_attention.jsonl"


def resolve_model_source(model_id: str) -> str:
    """Resolve a cached Hugging Face snapshot when the configured cache differs.

    It is common for this repository's Qwen2-VL experiments to use the project
    cache while a separately downloaded Qwen2.5-VL checkpoint lives in the
    user's default Hugging Face cache.  In offline mode Transformers cannot
    discover that second cache automatically.  Prefer an explicit local path,
    then inspect configured/default cache roots for a complete snapshot; if no
    local snapshot exists, return the original model id so online resolution
    retains its normal behavior.
    """

    explicit = Path(model_id).expanduser()
    if explicit.is_dir() and (explicit / "config.json").is_file():
        return str(explicit)
    normalized = model_id.replace("/", "--")
    roots: list[Path] = []
    for value in (os.environ.get("HF_HUB_CACHE"),):
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        roots.append(Path(xdg_cache).expanduser() / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    snapshots: list[Path] = []
    seen_roots: set[Path] = set()
    for root in roots:
        root = root.resolve() if root.exists() else root
        if root in seen_roots:
            continue
        seen_roots.add(root)
        snapshot_root = root / f"models--{normalized}" / "snapshots"
        if not snapshot_root.is_dir():
            continue
        snapshots.extend(
            snapshot
            for snapshot in snapshot_root.iterdir()
            if snapshot.is_dir() and (snapshot / "config.json").is_file()
        )
    if snapshots:
        return str(max(snapshots, key=lambda path: path.stat().st_mtime))
    return model_id


def normalize_attention_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Normalize a heatmap by one global maximum, preserving its dimensions."""

    values = [[float(value) for value in row] for row in matrix]
    if not values:
        return []
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("attention matrix must be rectangular")
    if any(not math.isfinite(value) or value < 0 for row in values for value in row):
        raise ValueError("attention matrix must contain finite non-negative values")
    scale = max((value for row in values for value in row), default=0.0)
    if scale <= 0.0:
        return [[0.0 for _ in row] for row in values]
    return [[value / scale for value in row] for row in values]


def smooth_profile(values: Sequence[float], window: int = 3) -> list[float]:
    """Return a centered moving average used only for the report overlay."""

    data = [float(value) for value in values]
    if window <= 0 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    if not data:
        return []
    radius = window // 2
    return [
        sum(data[max(0, index - radius) : min(len(data), index + radius + 1)])
        / len(data[max(0, index - radius) : min(len(data), index + radius + 1)])
        for index in range(len(data))
    ]


def _success_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("condition", "")) == "visual_attention"]


def validate_attention_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_samples: int | None = None,
    expected_layer_count: int | None = None,
) -> dict[str, Any]:
    """Validate complete native layer coverage and return a compact audit."""

    success = _success_rows(rows)
    if not success:
        raise ValueError("no successful visual-attention rows")
    layer_values = sorted({int(row["layer"]) for row in success})
    layer_count = int(expected_layer_count or max(int(row["layer_count"]) for row in success))
    expected_layers = list(range(layer_count))
    if layer_values != expected_layers:
        if layer_values and layer_values[0] != 0:
            raise ValueError(
                "layer indices must use zero-based native decoder indexing; "
                f"observed {layer_values}"
            )
        raise ValueError(f"incomplete layer coverage: expected {expected_layers}, observed {layer_values}")
    sample_ids = sorted({str(row["sample_id"]) for row in success})
    if expected_samples is not None and len(sample_ids) != int(expected_samples):
        raise ValueError(
            f"sample coverage mismatch: expected {expected_samples}, observed {len(sample_ids)}"
        )
    expected_rows = len(sample_ids) * layer_count
    if len(success) != expected_rows:
        raise ValueError(f"row coverage mismatch: expected {expected_rows}, observed {len(success)}")
    keys = {(str(row["sample_id"]), int(row["layer"])) for row in success}
    if len(keys) != len(success):
        raise ValueError("duplicate sample/layer attention rows")
    return {
        "num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "num_success_rows": len(success),
        "layer_count": layer_count,
        "layer_indices": layer_values,
        "expected_rows": expected_rows,
    }


def _percentile_ci(values: Sequence[float]) -> tuple[float, float]:
    if len(values) <= 1:
        value = float(values[0]) if values else 0.0
        return value, value
    standard_error = pstdev(values) / math.sqrt(len(values))
    margin = 1.96 * standard_error
    average = mean(values)
    return average - margin, average + margin


def aggregate_visual_attention_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate raw rows into per-layer statistics and a head-sorted matrix."""

    success = _success_rows(rows)
    if not success:
        raise ValueError("no successful visual-attention rows")
    layer_indices = sorted({int(row["layer"]) for row in success})
    layer_count = max(int(row.get("layer_count", 0)) for row in success)
    if layer_indices != list(range(layer_count)):
        raise ValueError("attention rows do not cover a complete zero-based layer range")
    head_count = len(success[0]["per_head_visual_mass"])
    if head_count <= 0:
        raise ValueError("per-head attention arrays must not be empty")
    by_layer: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in success:
        heads = [float(value) for value in row["per_head_visual_mass"]]
        if len(heads) != head_count:
            raise ValueError("all attention rows must have the same number of heads")
        if any(not math.isfinite(value) or value < 0 for value in heads):
            raise ValueError("per-head attention values must be finite and non-negative")
        by_layer[int(row["layer"])].append(row)

    layer_stats: list[dict[str, Any]] = []
    matrix_original_order: list[list[float]] = []
    for layer in layer_indices:
        group = by_layer[layer]
        masses = [float(row.get("visual_mass", sum(row["per_head_visual_mass"]))) for row in group]
        per_head = [
            mean(float(row["per_head_visual_mass"][head]) for row in group)
            for head in range(head_count)
        ]
        ci_low, ci_high = _percentile_ci(masses)
        layer_stats.append(
            {
                "layer": layer,
                "num_samples": len(group),
                "mean_visual_mass": mean(masses),
                "median_visual_mass": median(masses),
                "std_visual_mass": pstdev(masses) if len(masses) > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_per_head_visual_mass": per_head,
            }
        )
        matrix_original_order.append(per_head)

    head_means = [
        mean(matrix_original_order[layer_index][head] for layer_index in range(len(layer_indices)))
        for head in range(head_count)
    ]
    head_order = sorted(range(head_count), key=lambda head: (-head_means[head], head))
    matrix_raw = [[row[head] for head in head_order] for row in matrix_original_order]
    raw_layer_sum = [sum(row) for row in matrix_raw]
    return {
        "layer_indices": layer_indices,
        "layer_count": layer_count,
        "head_count": head_count,
        "head_order": head_order,
        "matrix_raw": matrix_raw,
        "matrix_normalized": normalize_attention_matrix(matrix_raw),
        "layer_stats": layer_stats,
        "raw_per_layer_sum": raw_layer_sum,
        "smoothed_per_layer_sum": smooth_profile(raw_layer_sum),
        "normalization": "global maximum of mean per-layer/per-head raw visual mass",
        "smoothing": "centered moving average, window=3, for red dashed overlay only",
    }


def build_attention_row(
    *,
    sample_id: str,
    task: str,
    layer: int,
    layer_count: int,
    query_index: int,
    visual_positions: Sequence[int],
    per_head_visual_mass: Sequence[float],
    model: str,
    visual_token_count: int | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build one auditable raw attention row with an explicit layer schema."""

    if layer < 0 or layer >= layer_count:
        raise ValueError(f"layer must be in [0, {layer_count - 1}], got {layer}")
    heads = [float(value) for value in per_head_visual_mass]
    visual = [int(value) for value in visual_positions]
    row = {
        "row_id": f"{sample_id}:layer-{layer}",
        "paper_figure": "Figure 3(b)",
        "condition": "visual_attention",
        "sample_id": str(sample_id),
        "task": str(task),
        "model": model,
        "target_model": model,
        "layer": int(layer),
        "layer_count": int(layer_count),
        "layer_index_convention": "zero_based_native_decoder_layer",
        "attention_query": "last_instruction",
        "query_index": int(query_index),
        "visual_positions": visual,
        "visual_token_count": len(visual) if visual_token_count is None else int(visual_token_count),
        "per_head_visual_mass": heads,
        "visual_mass": sum(heads),
    }
    row.update(metadata)
    return row


def plot_visual_attention_report(
    aggregate: Mapping[str, Any],
    output: str | Path,
    *,
    model: str = DEFAULT_MODEL,
) -> None:
    """Render a paper-style head-sorted heatmap with a smoothed layer profile."""

    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(aggregate["matrix_normalized"], dtype=float)
    layers = list(aggregate["layer_indices"])
    if matrix.ndim != 2 or not layers:
        raise ValueError("cannot plot an empty attention matrix")
    head_count = matrix.shape[1]
    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    image = axis.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xlabel("Sorted Heads")
    axis.set_ylabel("LLM Layers")
    model_label = model.rsplit("/", 1)[-1].replace("-Instruct", "")
    axis.set_title(f"Sparrow Figure 3(b): {model_label}")
    axis.set_xticks(np.arange(head_count))
    # The x-axis is the sorted rank used by the paper; the original head
    # indices are retained separately in ``head_order`` for auditability.
    axis.set_xticklabels([str(index) for index in range(head_count)], fontsize=7)
    axis.set_yticks(np.arange(len(layers)))
    axis.set_yticklabels([str(layer) for layer in layers], fontsize=7)
    profile = np.asarray(aggregate["smoothed_per_layer_sum"], dtype=float)
    if profile.size and float(profile.max()) > 0:
        profile = profile / float(profile.max()) * max(1, head_count - 1)
        axis.plot(profile, np.arange(len(layers)), color="#a40000", linestyle="--", linewidth=1.8, label="Smoothed Sum")
        axis.legend(loc="upper right", fontsize="small", framealpha=0.9)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Sum of Visual Attention (Normalized)")
    figure.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _batch_value(batch: Any, key: str) -> Any:
    return batch.get(key) if isinstance(batch, Mapping) else getattr(batch, key, None)


def _layer_count(model: Any) -> int:
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return len(candidate.layers)
    value = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if value is not None:
        return int(value)
    raise RuntimeError("could not locate Qwen decoder layers")


def _attention_position_embeddings(module: Any, hidden_states: Any, position_ids: Any, position_embeddings: Any):
    if position_embeddings is not None:
        return position_embeddings
    if position_ids is None or not hasattr(module, "rotary_emb"):
        return None
    return module.rotary_emb(hidden_states, position_ids)


def _mrope_section(module: Any) -> list[int]:
    scaling = getattr(module, "rope_scaling", None) or {}
    section = scaling.get("mrope_section") if hasattr(scaling, "get") else None
    if section is not None:
        return [int(value) for value in section]
    head_dim = int(module.head_dim)
    first = head_dim // 8
    remainder = head_dim // 2 - first
    return [first, remainder // 2, remainder - remainder // 2]


def _query_attention_qwen25(
    module: Any,
    hidden_states: Any,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    query_index: int,
) -> Any:
    """Compute one attention row without materializing a sequence square."""

    import torch

    if hidden_states.shape[0] != 1:
        raise ValueError("visual-attention probes require batch size one")
    q_len = int(hidden_states.shape[1])
    if query_index < 0 or query_index >= q_len:
        raise IndexError(f"query index {query_index} outside sequence length {q_len}")
    query = module.q_proj(hidden_states)
    key = module.k_proj(hidden_states)
    query = query.view(1, q_len, -1, module.head_dim).transpose(1, 2)
    key = key.view(1, q_len, -1, module.head_dim).transpose(1, 2)
    rotary = _attention_position_embeddings(module, hidden_states, position_ids, position_embeddings)
    if rotary is not None:
        module_name = type(module).__module__
        if "qwen2_5_vl" in module_name:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
        query, key = apply_multimodal_rotary_pos_emb(
            query, key, rotary[0], rotary[1], _mrope_section(module)
        )
    groups = int(getattr(module, "num_key_value_groups", 1))
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(query[:, :, query_index : query_index + 1, :], key.transpose(-1, -2))
    scores = scores / (float(module.head_dim) ** 0.5)
    key_len = scores.shape[-1]
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            mask = attention_mask[:, :, query_index : query_index + 1, :key_len]
        elif attention_mask.ndim == 3:
            mask = attention_mask[:, query_index : query_index + 1, :key_len].unsqueeze(1)
        elif attention_mask.ndim == 2:
            mask = attention_mask[:, None, None, :key_len]
            if mask.dtype == torch.bool or not torch.is_floating_point(mask):
                mask = torch.where(mask > 0, torch.zeros_like(mask, dtype=scores.dtype), torch.finfo(scores.dtype).min)
            else:
                mask = (1.0 - mask.to(scores.dtype)) * torch.finfo(scores.dtype).min
        else:
            raise ValueError(f"unsupported attention mask rank: {attention_mask.ndim}")
        scores = scores + mask.to(scores.dtype)
    else:
        causal = torch.arange(key_len, device=scores.device)[None, :] > query_index
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
    finite_rows = torch.isfinite(scores).any(dim=-1, keepdim=True)
    safe_scores = torch.where(finite_rows, scores, torch.zeros_like(scores))
    weights = torch.softmax(safe_scores.float(), dim=-1)
    weights = torch.where(finite_rows, weights, torch.zeros_like(weights))
    return weights[0, :, 0, :].detach().to("cpu")


@contextmanager
def _capture_qwen25_query_attention(model: Any, query_index: int):
    """Capture a final-instruction attention row from every decoder layer."""

    layers = []
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            layers = list(candidate.layers)
            break
    if not layers:
        raise RuntimeError("could not locate Qwen decoder layers")
    captured: dict[int, Any] = {}
    originals: list[tuple[Any, Any]] = []
    for layer_index, layer in enumerate(layers):
        module = layer.self_attn
        original = module.forward
        signature = inspect.signature(original)

        def wrapped(*args: Any, _module=module, _index=layer_index, _original=original, _signature=signature, **kwargs: Any):
            output = _original(*args, **kwargs)
            bound = _signature.bind_partial(*args, **kwargs)
            hidden_states = bound.arguments.get("hidden_states", args[0] if args else None)
            if hidden_states is not None:
                captured[_index] = _query_attention_qwen25(
                    _module,
                    hidden_states,
                    bound.arguments.get("attention_mask"),
                    bound.arguments.get("position_ids"),
                    bound.arguments.get("position_embeddings"),
                    query_index,
                )
            return output

        originals.append((module, original))
        module.forward = wrapped
    try:
        yield captured
    finally:
        for module, original in originals:
            module.forward = original


def _run_record(model: Any, processor: Any, record: Mapping[str, Any], args: argparse.Namespace, layer_count: int) -> list[dict[str, Any]]:
    import torch

    from .Validate_Sparrow_hypothesises.model_analysis import find_instruction_masks
    from .Validate_Sparrow_hypothesises.runtime import model_device, move_batch_to_device, process_video

    path = Path(str(record["video_path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    batch = process_video(
        processor,
        path,
        build_mvbench_prompt(record),
        fps=args.fps,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
    )
    batch = move_batch_to_device(batch, model_device(model))
    input_ids_tensor = _batch_value(batch, "input_ids")
    if input_ids_tensor is None:
        raise RuntimeError("processor batch has no input_ids")
    input_ids = input_ids_tensor[0].detach().to("cpu").tolist()
    video_token_id = getattr(getattr(model, "config", None), "video_token_id", None)
    if video_token_id is None:
        raise RuntimeError("Qwen2.5 model config has no video_token_id")
    visual_positions = (input_ids_tensor[0] == int(video_token_id)).nonzero(as_tuple=False).flatten()
    visual_positions_list = visual_positions.detach().to("cpu").tolist()
    if not visual_positions_list:
        raise RuntimeError(f"no video-token positions found for {record['sample_id']}")
    masks = find_instruction_masks(input_ids_tensor, processor, visual_positions_list)
    query_index = int(masks["query_index"])
    with _capture_qwen25_query_attention(model, query_index) as captured:
        with torch.inference_mode():
            model(**batch, use_cache=False, output_attentions=False, return_dict=True)
    if sorted(captured) != list(range(layer_count)):
        raise RuntimeError(
            f"attention capture incomplete for {record['sample_id']}: "
            f"expected {layer_count} layers, observed {sorted(captured)}"
        )
    input_fingerprint = _sha256_ids(input_ids)
    video_sha256 = _sha256_file(path)
    rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        weights = captured[layer]
        per_head = weights[:, visual_positions_list].sum(dim=-1).tolist()
        # Store the potentially large absolute token-position list once per
        # sample (layer 0); every other layer retains the count and an explicit
        # reference so raw per-layer rows stay compact.
        stored_positions = visual_positions_list if layer == 0 else []
        rows.append(
            build_attention_row(
                sample_id=str(record["sample_id"]),
                task=str(record["task"]),
                layer=layer,
                layer_count=layer_count,
                query_index=query_index,
                visual_positions=stored_positions,
                per_head_visual_mass=per_head,
                model=args.model,
                visual_token_count=len(visual_positions_list),
                video=record.get("video"),
                video_path=str(record["video_path"]),
                video_sha256=video_sha256,
                prompt=build_mvbench_prompt(record),
                instruction_positions=masks["instruction_positions"],
                assistant_marker_start=masks["assistant_marker_start"],
                text_positions=masks["text_positions"],
                input_fingerprint=input_fingerprint,
                fps=args.fps,
                max_frames=args.max_frames,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                dtype=args.dtype,
                device_map=args.device_map,
                quantized=args.quantized,
                visual_positions_reference_layer=0,
                visual_positions_stored=layer == 0,
            )
        )
    return rows


def _error_row(record: Mapping[str, Any], args: argparse.Namespace, layer_count: int, exc: Exception) -> dict[str, Any]:
    path = Path(str(record["video_path"]))
    return {
        "row_id": f"{record['sample_id']}:error",
        "paper_figure": "Figure 3(b)",
        "condition": "error",
        "sample_id": str(record["sample_id"]),
        "task": str(record["task"]),
        "model": args.model,
        "target_model": args.model,
        "layer_count": layer_count,
        "layer_index_convention": "zero_based_native_decoder_layer",
        "video_path": str(record["video_path"]),
        "video_sha256": _sha256_file(path) if path.is_file() else None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _write_summary(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    layer_count: int,
    output_path: Path,
    summary_path: Path,
    plot_path: Path,
    model_source: str,
) -> dict[str, Any]:
    errors = [row for row in rows if row.get("condition") == "error"]
    success = _success_rows(rows)
    aggregate: dict[str, Any] | None = None
    coverage_audit: dict[str, Any] | None = None
    if success:
        aggregate = aggregate_visual_attention_rows(success)
        coverage_audit = validate_attention_rows(
            success,
            expected_samples=len(records) if not errors else None,
            expected_layer_count=layer_count,
        )
        plot_visual_attention_report(aggregate, plot_path, model=args.model)
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    summary = {
        "paper_figure": "Figure 3(b)",
        "experiment": "all-layer visual attention",
        "model": args.model,
        "model_source": model_source,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256_file(Path(args.manifest)),
        "tasks": list(args.tasks),
        "num_records": len(records),
        "num_success_rows": len(success),
        "num_error_records": len(errors),
        "errors_by_task": dict(sorted({task: sum(1 for row in errors if row.get("task") == task) for task in {str(row.get("task")) for row in errors}}.items())),
        "expected_rows_if_complete": len(records) * layer_count,
        "coverage": len(success) / max(1, len(records) * layer_count),
        "layer_count": layer_count,
        "layer_indices": list(range(layer_count)),
        "layer_index_convention": "zero_based_native_decoder_layer",
        "attention_query": "final instruction token before assistant generation marker",
        "metric": "sum attention weights from the query to all video-token positions, retained per head",
        "visual_token_count_summary": (
            {
                "num_samples": len({str(row["sample_id"]) for row in success if int(row["layer"]) == 0}),
                "min": min(int(row["visual_token_count"]) for row in success if int(row["layer"]) == 0),
                "max": max(int(row["visual_token_count"]) for row in success if int(row["layer"]) == 0),
                "mean": mean(int(row["visual_token_count"]) for row in success if int(row["layer"]) == 0),
                "median": median(int(row["visual_token_count"]) for row in success if int(row["layer"]) == 0),
            }
            if any(int(row["layer"]) == 0 for row in success)
            else None
        ),
        "normalization": "raw values are preserved; heatmap uses one global maximum of mean layer/head mass",
        "smoothing": "centered moving average window=3 for red dashed overlay only",
        "fps": args.fps,
        "max_frames": args.max_frames,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "quantized": args.quantized,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "raw_output": str(output_path),
        "plot_output": str(plot_path),
        "coverage_audit": coverage_audit,
        "aggregate": aggregate,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=360 * 420)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--quantized", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    import torch

    from .Validate_Sparrow_hypothesises.model_analysis import load_qwen_model
    from .Validate_Sparrow_hypothesises.runtime import build_qwen2vl_video_processor, require_cuda

    require_cuda()
    records = read_mvbench_records(args.manifest, tasks=args.tasks, limit_per_task=args.limit_per_task)
    model_source = resolve_model_source(args.model)
    if model_source != args.model:
        print(f"using local model snapshot: {model_source}", flush=True)
    model = load_qwen_model(
        model_source,
        device_map=args.device_map,
        dtype=args.dtype,
        quantized=args.quantized,
        attn_implementation="eager",
    )
    processor = build_qwen2vl_video_processor(model_source, args.min_pixels, args.max_pixels)
    layer_count = _layer_count(model)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    plot_path = Path(args.plot_output) if args.plot_output else output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output_handle:
        for index, record in enumerate(records, start=1):
            try:
                record_rows = _run_record(model, processor, record, args, layer_count)
                rows.extend(record_rows)
                for row in record_rows:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                print(f"[{index}/{len(records)}] {record['sample_id']} captured {layer_count} layers", flush=True)
            except Exception as exc:  # keep long runs auditable by sample
                row = _error_row(record, args, layer_count, exc)
                rows.append(row)
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
                print(f"[{index}/{len(records)}] {record['sample_id']} ERROR {exc}", flush=True)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    summary = _write_summary(
        rows,
        records,
        args,
        layer_count,
        output_path,
        summary_path,
        plot_path,
        model_source,
    )
    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"wrote summary to {summary_path}")
    print(f"wrote plot to {plot_path}" if summary.get("aggregate") else "no plot: no successful rows")
    return 2 if summary["num_error_records"] else 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
