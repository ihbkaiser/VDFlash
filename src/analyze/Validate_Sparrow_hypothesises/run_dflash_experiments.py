"""Orchestrate the Qwen2.5-VL DFlash Sparrow/MSD validation stages."""

from __future__ import annotations

import argparse
import gc
import json
from contextlib import contextmanager
from datetime import date
from dataclasses import asdict
from pathlib import Path
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import torch

from .dflash_contract import (
    DFLASH_LENGTH_TARGETS,
    DFLASH_RETENTION_PERCENTAGES,
    DFlashExperiment,
    DFlashSemanticStatus,
    make_dflash_metadata,
)
from .dflash_report import write_dflash_report
from .dflash_runtime import (
    apply_hidden_context_mask,
    find_visual_positions,
    input_fingerprint,
)
from .run_dflash_attention import run_dflash_context_attention
from .run_dflash_layers import eager_target_attention, run_qwen25vl_layer_diagnostics
from .run_dflash_length import run_hidden_visual_retention, run_length_sweep

DFLASH_STAGE_ORDER = ("length", "retention", "attention", "layers", "report")
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CHECKPOINT = "dataset/qwen25vl-3b-dflash-llava68k-latest/training_state.pt"
DEFAULT_DRAFT_CONFIG = str(
    Path(__file__).resolve().parents[2]
    / "train_Dflash_SpecForge"
    / "configs"
    / "qwen2.5-vl-3b-dflash.json"
)
DEFAULT_MANIFEST = "dataset/VideoDetailCaption/test.jsonl"
DEFAULT_VIDEO_ROOT = "dataset/VideoDetailCaption"


def default_output_dir() -> str:
    return f"results/sparrow_validation_dflash_qwen25vl3b_{date.today().isoformat()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", *DFLASH_STAGE_ORDER, "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--draft-config", default=DEFAULT_DRAFT_CONFIG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--calibration-input", default="dataset/VideoDetailCaption/calibration.jsonl")
    parser.add_argument("--output-dir", default=default_output_dir())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "no"), default="auto")
    parser.add_argument("--target-attention", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--layer-visual-targets", type=int, nargs="+", default=[3000])
    parser.add_argument("--layer-cut-points", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_dflash_experiments(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected DFlash stages.

    Full model wiring is loaded lazily by the stage implementation.  Dry-run
    validates paths/options and emits a plan without importing Transformers or
    the MSD command path.
    """

    output_dir = Path(args.output_dir)
    selected = DFLASH_STAGE_ORDER if args.stage == "all" else (args.stage,)
    plan = {
        "backend": "dflash",
        "stages": list(selected),
        "target_model": args.target_model,
        "checkpoint": str(args.checkpoint),
        "draft_config": str(args.draft_config),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        return {**plan, "preflight": _preflight(args)}
    if args.stage == "preflight":
        return {**plan, "preflight": _preflight(args)}
    return execute_dflash_stages(args)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "checkpoint": Path(args.checkpoint).expanduser(),
        "draft_config": Path(args.draft_config).expanduser(),
        "manifest": Path(args.manifest).expanduser(),
        "video_root": Path(args.video_root).expanduser(),
        "calibration_input": Path(args.calibration_input).expanduser(),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    checkpoint = paths["checkpoint"]
    calibration_coverage: dict[str, dict[str, int]] = {}
    if paths["calibration_input"].is_file():
        for row in _read_jsonl(paths["calibration_input"]):
            target = row.get("target_visual_tokens")
            if target is None:
                continue
            key = str(int(target))
            summary = calibration_coverage.setdefault(key, {"total": 0, "ok": 0})
            summary["total"] += 1
            if row.get("calibration_status", row.get("status")) == "ok":
                summary["ok"] += 1
    calibration_gaps = [
        int(target)
        for target in DFLASH_LENGTH_TARGETS
        if calibration_coverage.get(str(target), {}).get("ok", 0) == 0
    ]
    calibration_incomplete_targets = [
        int(target)
        for target in DFLASH_LENGTH_TARGETS
        if calibration_coverage.get(str(target), {}).get("ok", 0)
        < calibration_coverage.get(str(target), {}).get("total", 0)
    ]
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "missing_paths": missing,
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "cuda_available": bool(torch.cuda.is_available()),
        "target_model": args.target_model,
        "calibration_coverage": calibration_coverage,
        "calibration_gaps": calibration_gaps,
        "calibration_incomplete_targets": calibration_incomplete_targets,
        "ready_for_model_run": not missing and bool(torch.cuda.is_available()),
        "ready_for_full_grid": (
            not missing
            and bool(torch.cuda.is_available())
            and not calibration_incomplete_targets
        ),
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object in {path}")
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _stage_path(output_dir: Path, stage: str) -> Path:
    return output_dir / {
        "length": "figure1a_length_sweep.jsonl",
        "retention": "figure1b_target_hidden_visual_retention.jsonl",
        "attention": "figure2_dflash_context_attention.jsonl",
        "layers": "figure3_3b_6_target_diagnostics.jsonl",
    }[stage]


def _stage_complete_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f".{stage}.complete"


def _stage_row_key(stage: str, row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Identify one resumable sample/condition unit in a stage file."""

    condition_key = {
        "length": "length_target",
        "retention": "retention_percentage",
        "attention": "target_visual_tokens",
        "layers": "target_visual_tokens",
    }.get(stage)
    if condition_key is None:
        raise ValueError(f"stage {stage!r} has no resumable row key")
    return (
        str(row.get("sample_id", "")),
        stage,
        str(row.get(condition_key, "")),
    )


def _read_stage_rows(path: Path) -> list[dict[str, Any]]:
    """Read a stage journal while dropping a crash-truncated final line."""

    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines) - 1:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object in stage journal {path}")
        rows.append(value)
    return rows


@contextmanager
def _stage_row_sink(path: Path, *, append: bool) -> Iterator[Callable[[Mapping[str, Any]], None]]:
    """Flush each completed row so an interrupted GPU run remains resumable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", encoding="utf-8")

    def sink(row: Mapping[str, Any]) -> None:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
        handle.flush()

    try:
        yield sink
    finally:
        handle.close()


def _release_cuda_condition(device: torch.device) -> None:
    """Release temporary prompt/cache allocations between heterogeneous jobs."""

    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_record(sample: Any, calibration_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in calibration_rows:
        try:
            grouped[int(row["target_visual_tokens"])] = dict(row)
        except (KeyError, TypeError, ValueError):
            continue
    record = asdict(sample)
    record.update(
        {
            "id": sample.sample_id,
            "sample_id": sample.sample_id,
            "calibration_by_target": grouped,
            "input_fingerprint": sample.fingerprint(),
        }
    )
    return record


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    return make_dflash_metadata(
        target_model=args.target_model,
        draft_checkpoint=str(args.checkpoint),
        draft_config=str(args.draft_config),
        experiment=DFlashExperiment.LENGTH_SWEEP,
        semantic_status=DFlashSemanticStatus.DIRECT,
    )


def _unsupported(error: str, *, input_value: str = "unavailable") -> dict[str, Any]:
    return {"status": "unsupported", "error": error, "input_fingerprint": input_value, "metrics": {}}


def _calibration_settings(sample: dict[str, Any], target: int) -> tuple[dict[str, Any] | None, str | None]:
    point = sample.get("calibration_by_target", {}).get(int(target))
    if point is None:
        return None, f"no calibration point for target_visual_tokens={target}"
    if point.get("calibration_status", point.get("status")) != "ok":
        return None, (
            f"calibration point target_visual_tokens={target} is not usable: "
            f"{point.get('calibration_status', point.get('status'))}"
        )
    settings = point.get("candidate_settings")
    if not isinstance(settings, dict):
        return None, f"calibration point target_visual_tokens={target} has no candidate_settings"
    return settings, None


def _prepare_prompt(
    *,
    processor: Any,
    target: Any,
    sample: dict[str, Any],
    video_root: str,
    device: torch.device,
    target_visual_tokens: int,
):
    from src.infer.qwen25vl_dflash_compare import prepare_video_prompt

    settings, error = _calibration_settings(sample, target_visual_tokens)
    if error:
        raise RuntimeError(error)
    prompt = prepare_video_prompt(
        processor,
        target,
        sample,
        video_root=video_root,
        device=device,
        num_frames=int(settings["frames"]),
        video_min_pixels=int(settings["max_pixels"]),
        video_max_pixels=int(settings["max_pixels"]),
        video_reader="torchvision",
    )
    positions = find_visual_positions(prompt.inputs["input_ids"], target=target, processor=processor)
    return prompt, positions, input_fingerprint(prompt.inputs), settings


def _decode_prompt(
    *,
    target: Any,
    draft: Any,
    processor: Any,
    prompt: Any,
    device: torch.device,
    max_new_tokens: int,
    prefill_transform=None,
    capture_attention: bool = False,
) -> dict[str, Any]:
    from src.infer.qwen25vl_dflash_compare import (
        InstrumentedDFlashDecoder,
        _eos_token_ids,
        _target_greedy,
        capture_dflash_attention,
    )

    target_output, target_timing = _target_greedy(
        target,
        prompt,
        max_new_tokens=max_new_tokens,
        stop_token_ids=_eos_token_ids(processor, target),
        device=device,
    )
    decoder = InstrumentedDFlashDecoder(target, draft, device=device)
    capture_context = capture_dflash_attention(draft) if capture_attention else None
    if capture_context is None:
        context_manager = _null_capture()
    else:
        context_manager = capture_context
    with context_manager as attention_records:
        speculative = decoder.decode(
            input_ids=prompt.inputs["input_ids"],
            position_ids=prompt.position_ids,
            target_kwargs=prompt.target_kwargs,
            max_new_tokens=max_new_tokens,
            stop_token_ids=_eos_token_ids(processor, target),
            prefill_target_hidden_transform=prefill_transform,
        )
    prompt_length = int(prompt.inputs["input_ids"].shape[1])
    target_ids = target_output[0, prompt_length:].detach().cpu().tolist()
    speculative_ids = speculative.output_ids[0, prompt_length:].detach().cpu().tolist()
    result = {
        "status": "ok" if target_ids == speculative_ids else "mismatch",
        "target_output_ids": target_ids,
        "speculative_output_ids": speculative_ids,
        "target_input_fingerprint": input_fingerprint(prompt.inputs),
        "metrics": {
            "target_output_tokens": len(target_ids),
            "speculative_output_tokens": len(speculative_ids),
            "tau": speculative.tau_effective,
            "tau_proposal": speculative.tau_proposal,
            "tau_effective": speculative.tau_effective,
        },
        "timing": {
            "target": target_timing,
            "speculative": speculative.as_dict().get("timing", {}),
        },
        "acceptance": speculative.as_dict().get("acceptance_rounds", []),
        "attention_records": attention_records,
    }
    result["speedup"] = {
        "end_to_end": target_timing["end_to_end_s"]
        / max(speculative.end_to_end_s, 1e-12)
    }
    del decoder, speculative, target_output
    gc.collect()
    return result


class _null_capture:
    def __enter__(self):
        return []

    def __exit__(self, exc_type, exc, tb):
        return False


def _target_side_probe(
    *,
    target: Any,
    processor: Any,
    prompt: Any,
    visual_positions: list[int],
    device: torch.device,
    max_new_tokens: int,
    layer_cut_points: list[int],
) -> list[dict[str, Any]]:
    """Run target-side Qwen2.5-VL analogues using generic layer probes."""

    from src.analyze.Validate_Sparrow_hypothesises.model_analysis import (
        capture_query_attention,
        find_instruction_masks,
        layerwise_input_cosine,
        mask_visual_keys,
        prepare_qwen2vl_prefill,
    )

    batch = dict(prompt.inputs)
    prepared = prepare_qwen2vl_prefill(target, batch, device)
    masks = find_instruction_masks(batch["input_ids"], processor, visual_positions)
    rows: list[dict[str, Any]] = []

    for cut in layer_cut_points:
        with mask_visual_keys(target, visual_positions, int(cut)):
            with torch.inference_mode():
                output = target.generate(
                    **batch,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
        input_length = int(batch["input_ids"].shape[1])
        candidate = output[0, input_length:].detach().cpu().tolist()
        rows.append(
            {
                "experiment": DFlashExperiment.TARGET_VISUAL_KV.value,
                "layer_index": int(cut),
                "visual_positions": visual_positions,
                "metrics": {
                    "diagnostic_output_length": len(candidate),
                    "visual_kv_masked_from": int(cut),
                },
            }
        )

    with capture_query_attention(target, int(masks["query_index"])) as captured:
        with torch.inference_mode():
            target(**batch, use_cache=False, output_attentions=False, return_dict=True)
    visual = torch.as_tensor(visual_positions, dtype=torch.long)
    for layer, weights in sorted(captured.items()):
        if visual.numel():
            visual_mass = float(weights[:, visual].sum().item())
        else:
            visual_mass = 0.0
        rows.append(
            {
                "experiment": DFlashExperiment.TARGET_ATTENTION.value,
                "layer_index": int(layer) + 1,
                "query_position": int(masks["query_index"]),
                "visual_positions": visual_positions,
                "metrics": {
                    "visual_attention_mass": visual_mass,
                    "attention_heads": int(weights.shape[0]),
                    "attention_key_length": int(weights.shape[-1]),
                },
            }
        )

    text_positions = list(masks["instruction_positions"]) + list(masks["text_positions"])
    for curve in layerwise_input_cosine(target, batch, prepared, visual_positions, text_positions):
        rows.append(
            {
                "experiment": DFlashExperiment.TARGET_HIDDEN_COSINE.value,
                "layer_index": int(curve["layer"]),
                "metrics": {
                    "visual_cosine": float(curve["visual_cosine"]),
                    "text_cosine": float(curve["text_cosine"]),
                },
            }
        )
    return rows


def execute_dflash_stages(args: argparse.Namespace) -> dict[str, Any]:
    """Execute selected stages and write independent resumable evidence files."""

    from src.analyze.Validate_Sparrow_hypothesises.dataset import load_vdc_manifest
    from src.infer.qwen25vl_dflash_compare import _load_draft, _load_target, _resolve_dtype

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run DFlash validation on a GPU host")
    samples = load_vdc_manifest(args.manifest, args.video_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    calibration = _read_jsonl(args.calibration_input)
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in calibration:
        by_sample.setdefault(str(row.get("sample_id")), []).append(row)
    records = [_sample_record(sample, by_sample.get(sample.sample_id, [])) for sample in samples]
    metadata = _metadata(args)
    all_rows: list[dict[str, Any]] = []

    stage_names = DFLASH_STAGE_ORDER if args.stage == "all" else (args.stage,)
    existing: dict[str, list[dict[str, Any]]] = {}
    completed_stages: set[str] = set()
    for stage in ("length", "retention", "attention", "layers"):
        path = _stage_path(output_dir, stage)
        if (args.resume or args.stage == "report") and path.is_file():
            existing[stage] = _read_stage_rows(path)
            all_rows.extend(existing[stage])
        if (args.resume or args.stage == "report") and _stage_complete_path(output_dir, stage).is_file():
            completed_stages.add(stage)
    needed_model = any(stage in stage_names for stage in ("length", "retention", "attention", "layers"))
    models: dict[str, Any | None] = {"target": None, "processor": None, "draft": None}
    if needed_model:
        dtype = _resolve_dtype(args.dtype, device)
        models["processor"], models["target"], _ = _load_target(
            args.target_model,
            device=device,
            dtype=dtype,
            attention=args.target_attention,
        )
        models["draft"], _ = _load_draft(args.checkpoint, args.draft_config, device=device, dtype=dtype)
    try:
        if "length" in stage_names and "length" not in completed_stages:
            def length_decode(sample, condition):
                target_tokens = int(condition["length_target"])
                try:
                    prompt, _visual, fingerprint, settings = _prepare_prompt(
                        processor=models["processor"], target=models["target"], sample=sample,
                        video_root=args.video_root, device=device,
                        target_visual_tokens=target_tokens,
                    )
                except Exception as exc:
                    return _unsupported(f"{type(exc).__name__}: {exc}")
                result = _decode_prompt(
                    target=models["target"], draft=models["draft"], processor=models["processor"], prompt=prompt,
                    device=device, max_new_tokens=args.max_new_tokens,
                )
                result["input_fingerprint"] = fingerprint
                result["actual_visual_tokens"] = len(_visual)
                result["calibration_settings"] = settings
                return result

            stage_path = _stage_path(output_dir, "length")
            completed_keys = {
                _stage_row_key("length", row) for row in existing.get("length", [])
            }
            with _stage_row_sink(
                stage_path,
                append=args.resume and stage_path.is_file(),
            ) as sink:
                rows = run_length_sweep(
                    records, length_decode, metadata=metadata,
                    length_targets=DFLASH_LENGTH_TARGETS, limit=None,
                    row_sink=sink,
                    skip=lambda sample, condition: _stage_row_key(
                        "length", {**sample, **condition}
                    ) in completed_keys,
                    cleanup=lambda: _release_cuda_condition(device),
                )
            all_rows.extend(rows)
            _stage_complete_path(output_dir, "length").touch()

        if "retention" in stage_names and "retention" not in completed_stages:
            def retention_decode(sample, condition):
                if sample.get("_retention_error"):
                    return _unsupported(sample["_retention_error"])
                prompt = sample["_retention_prompt"]
                mask = torch.as_tensor(condition["hidden_context_mask"], dtype=torch.bool, device=device)
                result = _decode_prompt(
                    target=models["target"], draft=models["draft"], processor=models["processor"], prompt=prompt,
                    device=device, max_new_tokens=args.max_new_tokens,
                    prefill_transform=lambda hidden: apply_hidden_context_mask(hidden, mask),
                )
                result["input_fingerprint"] = f"{sample['full_target_input_fingerprint']}:retention-{condition['retention_percentage']}"
                result["actual_visual_tokens"] = len(sample["visual_positions"])
                result["calibration_settings"] = sample.get("_retention_settings")
                return result

            stage_path = _stage_path(output_dir, "retention")
            completed_keys = {
                _stage_row_key("retention", row) for row in existing.get("retention", [])
            }
            with _stage_row_sink(
                stage_path,
                append=args.resume and stage_path.is_file(),
            ) as sink:
                for record in records:
                    sample = dict(record)
                    expected_keys = {
                        _stage_row_key(
                            "retention",
                            {
                                "sample_id": sample["sample_id"],
                                "retention_percentage": percentage,
                            },
                        )
                        for percentage in DFLASH_RETENTION_PERCENTAGES
                    }
                    if expected_keys.issubset(completed_keys):
                        continue
                    try:
                        prompt, positions, fingerprint, settings = _prepare_prompt(
                            processor=models["processor"], target=models["target"], sample=sample,
                            video_root=args.video_root, device=device,
                            target_visual_tokens=3000,
                        )
                        sample["context_length"] = int(prompt.inputs["input_ids"].shape[1])
                        sample["visual_positions"] = positions
                        sample["full_target_input_fingerprint"] = fingerprint
                        sample["_retention_prompt"] = prompt
                        sample["_retention_settings"] = settings
                    except Exception as exc:
                        sample["context_length"] = 1
                        sample["visual_positions"] = []
                        sample["full_target_input_fingerprint"] = "unavailable"
                        sample["_retention_error"] = f"{type(exc).__name__}: {exc}"
                    sample_rows = run_hidden_visual_retention(
                        [sample], retention_decode, metadata=metadata,
                        retention_percentages=DFLASH_RETENTION_PERCENTAGES, limit=None,
                        row_sink=sink,
                        skip=lambda current, condition: _stage_row_key(
                            "retention", {**current, **condition}
                        ) in completed_keys,
                        cleanup=lambda: _release_cuda_condition(device),
                    )
                    all_rows.extend(sample_rows)
                    sample.pop("_retention_prompt", None)
                    _release_cuda_condition(device)
            _stage_complete_path(output_dir, "retention").touch()

        if "attention" in stage_names and "attention" not in completed_stages:
            attention_records: list[dict[str, Any]] = []
            for target_tokens in (400, 3000):
                for record in records:
                    sample = dict(record)
                    sample["attention_target"] = target_tokens
                    attention_records.append(sample)

            def attention_probe(sample):
                try:
                    prompt, visual, fingerprint, _settings = _prepare_prompt(
                        processor=models["processor"], target=models["target"], sample=sample,
                        video_root=args.video_root, device=device,
                        target_visual_tokens=int(sample["attention_target"]),
                    )
                    config = getattr(models["draft"], "config", None)
                    previous = getattr(config, "_attn_implementation", None)
                    if config is not None:
                        config._attn_implementation = "eager"
                    try:
                        result = _decode_prompt(
                            target=models["target"], draft=models["draft"], processor=models["processor"], prompt=prompt,
                            device=device, max_new_tokens=args.max_new_tokens,
                            capture_attention=True,
                        )
                    finally:
                        if config is not None:
                            config._attn_implementation = previous
                    summaries = []
                    from src.infer.qwen25vl_dflash_compare import summarize_dflash_attention
                    for record in result["attention_records"]:
                        summary = summarize_dflash_attention(
                            record["weights"], context_length=int(record["context_length"])
                        )
                        summary.update({
                            "layer_index": record["layer_index"],
                            "target_visual_tokens": int(sample["attention_target"]),
                            "input_fingerprint": fingerprint,
                            "metrics": summary.copy(),
                        })
                        summaries.append(summary)
                    return summaries
                except Exception as exc:
                    return [{"status": "unsupported", "error": f"{type(exc).__name__}: {exc}"}]

            stage_path = _stage_path(output_dir, "attention")
            completed_keys = {
                _stage_row_key("attention", row) for row in existing.get("attention", [])
            }
            with _stage_row_sink(
                stage_path,
                append=args.resume and stage_path.is_file(),
            ) as sink:
                rows = run_dflash_context_attention(
                    attention_records, attention_probe, metadata=metadata, limit=None,
                    row_sink=sink,
                    skip=lambda sample: _stage_row_key(
                        "attention",
                        {**sample, "target_visual_tokens": sample["attention_target"]},
                    ) in completed_keys,
                    cleanup=lambda: _release_cuda_condition(device),
                )
            all_rows.extend(rows)
            _stage_complete_path(output_dir, "attention").touch()

        if "layers" in stage_names and "layers" not in completed_stages:
            layer_records: list[dict[str, Any]] = []
            for target_tokens in args.layer_visual_targets:
                for record in records:
                    sample = dict(record)
                    sample["layer_target"] = int(target_tokens)
                    layer_records.append(sample)

            def layer_probe(sample):
                try:
                    prompt, visual, fingerprint, _settings = _prepare_prompt(
                        processor=models["processor"], target=models["target"], sample=sample,
                        video_root=args.video_root, device=device,
                        target_visual_tokens=int(sample["layer_target"]),
                    )
                    with eager_target_attention(models["target"]):
                        diagnostics = _target_side_probe(
                            target=models["target"], processor=models["processor"], prompt=prompt,
                            visual_positions=visual, device=device,
                            max_new_tokens=args.max_new_tokens,
                            layer_cut_points=args.layer_cut_points,
                        )
                    for diagnostic in diagnostics:
                        diagnostic["target_visual_tokens"] = int(sample["layer_target"])
                        diagnostic["input_fingerprint"] = fingerprint
                    return diagnostics
                except Exception as exc:
                    return [{
                        "experiment": DFlashExperiment.TARGET_HIDDEN_COSINE.value,
                        "status": "unsupported",
                        "error": f"{type(exc).__name__}: {exc}",
                    }]

            stage_path = _stage_path(output_dir, "layers")
            completed_keys = {
                _stage_row_key("layers", row) for row in existing.get("layers", [])
            }
            with _stage_row_sink(
                stage_path,
                append=args.resume and stage_path.is_file(),
            ) as sink:
                rows = run_qwen25vl_layer_diagnostics(
                    layer_records, layer_probe, metadata=metadata, limit=None,
                    row_sink=sink,
                    skip=lambda sample: _stage_row_key(
                        "layers",
                        {**sample, "target_visual_tokens": sample["layer_target"]},
                    ) in completed_keys,
                    cleanup=lambda: _release_cuda_condition(device),
                )
            all_rows.extend(rows)
            _stage_complete_path(output_dir, "layers").touch()
    finally:
        models["draft"] = None
        models["target"] = None
        models["processor"] = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report_path = None
    if "report" in stage_names or args.stage == "all":
        report_path = write_dflash_report(
            all_rows,
            output_dir,
            metadata={
                "backend": "dflash",
                "target_model": args.target_model,
                "draft_checkpoint": str(args.checkpoint),
                "manifest": str(args.manifest),
                "device": str(device),
                "dtype": args.dtype,
            },
        )
    return {
        "backend": "dflash",
        "output_dir": str(output_dir),
        "rows": len(all_rows),
        "report": str(report_path) if report_path else None,
        "stage_files": {stage: str(_stage_path(output_dir, stage)) for stage in existing},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_dflash_experiments(args)
    print(result)
    if result.get("preflight", {}).get("missing_paths"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
