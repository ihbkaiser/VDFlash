from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import DFlashTrainConfig
from .video import VideoProcessorMetadata, prepare_qwen_messages


def _tensor_dict_to_device(values: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        result[key] = value.to(device) if torch.is_tensor(value) else value
    return result


@dataclass
class PreparedExample:
    inputs: dict[str, Any]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    response_start: int
    response_end: int
    sample_id: str


def validate_manifest_record(record: dict[str, Any], *, require_target_response: bool) -> None:
    """Validate the manifest contract before expensive processor/model work."""

    if not isinstance(record.get("id"), (str, int)) or record.get("id") == "":
        raise ValueError("manifest record requires a non-empty 'id'")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"record {record.get('id')} requires a non-empty messages list")
    if require_target_response:
        response = record.get("target_response")
        if not isinstance(response, dict) or not isinstance(response.get("token_ids"), list):
            raise ValueError(
                f"record {record.get('id')} requires target_response.token_ids from prepare_responses"
            )
        if not all(isinstance(token_id, int) for token_id in response["token_ids"]):
            raise ValueError(f"record {record.get('id')} target_response.token_ids must be integers")
        if not isinstance(response.get("text"), str):
            raise ValueError(f"record {record.get('id')} target_response.text must be a string")
        if not isinstance(response.get("generation"), dict):
            raise ValueError(
                f"record {record.get('id')} target_response.generation is required for reproducibility"
            )


class Qwen25VLTargetAdapter:
    """Compatibility layer around the changing Transformers Qwen2.5-VL API."""

    def __init__(self, model: Any, processor: Any, *, device: torch.device):
        self.model = model
        self.processor = processor
        self.device = device
        self.config = model.config
        self.text_config = getattr(self.config, "text_config", self.config)
        self.input_embeddings = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        if self.lm_head is None and hasattr(model, "lm_head"):
            self.lm_head = model.lm_head
        if self.lm_head is None:
            raise RuntimeError("Target model does not expose an output embedding/LM head")

    @classmethod
    def from_pretrained(
        cls,
        config: DFlashTrainConfig,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen25VLTargetAdapter":
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Transformers is required to load Qwen2.5-VL") from exc
        processor = AutoProcessor.from_pretrained(config.target_model, revision=config.target_revision)
        model_cls = AutoModelForImageTextToText
        kwargs: dict[str, Any] = {
            "revision": config.target_revision,
            "low_cpu_mem_usage": True,
            "attn_implementation": config.target_attn_implementation,
        }
        if dtype is None:
            dtype = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "no": torch.float32,
            }[config.mixed_precision]
        kwargs["dtype"] = dtype
        try:
            model = model_cls.from_pretrained(config.target_model, **kwargs)
        except (AttributeError, ValueError):  # older Transformers releases
            try:
                from transformers import AutoModelForVision2Seq
            except ImportError:
                from transformers import AutoModelForCausalLM as AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(config.target_model, **kwargs)
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(target_device)
        adapter = cls(model, processor, device=target_device)
        adapter.requested_model = config.target_model
        adapter.requested_revision = config.target_revision
        adapter.processor.dflash_processor_kwargs = dict(config.processor_kwargs)
        adapter.image_min_pixels = config.image_min_pixels
        adapter.image_max_pixels = config.image_max_pixels
        adapter.video_reader = config.video_reader
        adapter.video_num_frames = config.video_num_frames
        adapter.video_min_pixels = config.video_min_pixels
        adapter.video_max_pixels = config.video_max_pixels
        return adapter

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def vocab_size(self) -> int:
        return int(getattr(self.text_config, "vocab_size", self.lm_head.weight.shape[0]))

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    @property
    def visual_token_ids(self) -> tuple[set[int], set[int]]:
        image_ids: set[int] = set()
        video_ids: set[int] = set()
        for name in ("image_token_id", "image_token_index"):
            value = getattr(self.config, name, None)
            if value is not None:
                image_ids.add(int(value))
        for name in ("video_token_id", "video_token_index"):
            value = getattr(self.config, name, None)
            if value is not None:
                video_ids.add(int(value))
        return image_ids, video_ids

    def resolve_mask_token_id(self) -> int:
        """Use an existing padded vocabulary row; never resize a frozen target."""

        vocab = self.processor.tokenizer.get_vocab()
        used = set(int(value) for value in vocab.values())
        used.update(int(value) for value in getattr(self.processor.tokenizer, "all_special_ids", []))
        for name in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            value = getattr(self.config, name, None)
            if value is not None:
                used.add(int(value))
        embedding_rows = int(self.input_embeddings.weight.shape[0])
        head_rows = int(self.lm_head.weight.shape[0])
        padded_vocab_size = min(self.vocab_size, embedding_rows, head_rows)
        for token_id in range(len(self.processor.tokenizer), padded_vocab_size):
            if token_id not in used:
                return token_id
        raise RuntimeError(
            "The target tokenizer has no unused padded vocabulary row for MASK; "
            "provide a target checkpoint with a reserved row instead of resizing it."
        )

    def tokenizer_fingerprint(self) -> str:
        payload = {
            "name": getattr(self.processor.tokenizer, "name_or_path", None),
            "vocab_size": self.vocab_size,
            "special_ids": sorted(int(x) for x in self.processor.tokenizer.all_special_ids),
            "vocab": sorted((str(key), int(value)) for key, value in self.processor.tokenizer.get_vocab().items()),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def processor_fingerprint(self) -> str:
        """Fingerprint the processor configuration in addition to tokenizer IDs."""

        serialized = None
        to_json_string = getattr(self.processor, "to_json_string", None)
        if callable(to_json_string):
            try:
                serialized = to_json_string()
            except (TypeError, ValueError, RuntimeError):
                serialized = None
        payload = {
            "class": f"{type(self.processor).__module__}.{type(self.processor).__qualname__}",
            "serialized": serialized,
            "processor_kwargs": getattr(self.processor, "dflash_processor_kwargs", {}),
            "video_preprocessing": {
                "image_min_pixels": getattr(self, "image_min_pixels", None),
                "image_max_pixels": getattr(self, "image_max_pixels", None),
                "num_frames": getattr(self, "video_num_frames", None),
                "min_pixels": getattr(self, "video_min_pixels", None),
                "max_pixels": getattr(self, "video_max_pixels", None),
                "reader": getattr(self, "video_reader", None),
            },
            "tokenizer_fingerprint": self.tokenizer_fingerprint(),
        }
        return hashlib.sha256(
            json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def target_provenance(self) -> dict[str, Any]:
        resolved_commit = (
            getattr(self.config, "_commit_hash", None)
            or getattr(self.text_config, "_commit_hash", None)
        )
        return {
            "target_model": getattr(self, "requested_model", None)
            or getattr(self.config, "_name_or_path", None),
            "target_revision": getattr(self, "requested_revision", None),
            "target_commit": resolved_commit,
            "tokenizer_fingerprint": self.tokenizer_fingerprint(),
            "processor_fingerprint": self.processor_fingerprint(),
        }

    def validate_record_provenance(self, record: dict[str, Any]) -> None:
        provenance = record.get("provenance")
        if provenance is None:
            raise ValueError(
                f"record {record.get('id')} has no provenance; regenerate it with prepare_responses"
            )
        if not isinstance(provenance, dict):
            raise ValueError(f"record {record.get('id')} provenance must be an object")
        expected = self.target_provenance()
        for key in (
            "target_model",
            "target_revision",
            "tokenizer_fingerprint",
            "processor_fingerprint",
        ):
            if provenance.get(key) != expected[key]:
                raise ValueError(
                    f"record {record.get('id')} provenance mismatch for {key}: "
                    f"{provenance.get(key)!r} != {expected[key]!r}"
                )
        recorded_commit = provenance.get("target_commit")
        if recorded_commit is not None and expected["target_commit"] is not None:
            if recorded_commit != expected["target_commit"]:
                raise ValueError(
                    f"record {record.get('id')} provenance mismatch for target_commit: "
                    f"{recorded_commit!r} != {expected['target_commit']!r}"
                )

    def _compute_position_ids(self, inputs: dict[str, Any]) -> torch.Tensor:
        if "position_ids" in inputs and torch.is_tensor(inputs["position_ids"]):
            positions = inputs["position_ids"]
            if positions.ndim == 2:
                return positions.unsqueeze(0).expand(3, -1, -1)
            if positions.ndim == 3 and positions.shape[0] == 3:
                return positions

        candidates = ["compute_3d_position_ids", "get_rope_index"]
        kwargs = {
            key: inputs[key]
            for key in (
                "input_ids",
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "attention_mask",
                "mm_token_type_ids",
                "past_key_values",
                "inputs_embeds",
            )
            if key in inputs
        }
        failures: list[str] = []
        model_objects = [self.model]
        for attribute in ("model", "language_model"):
            candidate = getattr(self.model, attribute, None)
            if candidate is not None and candidate not in model_objects:
                model_objects.append(candidate)
        nested = getattr(getattr(self.model, "model", None), "language_model", None)
        if nested is not None and nested not in model_objects:
            model_objects.append(nested)
        for model_object in model_objects:
            for method_name in candidates:
                method = getattr(model_object, method_name, None)
                if method is None:
                    continue
                try:
                    positions = method(**kwargs)
                    if isinstance(positions, tuple):
                        positions = positions[0]
                    if positions.ndim == 2:
                        positions = positions.unsqueeze(0).expand(3, -1, -1)
                    if positions.ndim == 3 and positions.shape[0] == 3:
                        return positions
                except (TypeError, ValueError, RuntimeError) as exc:
                    failures.append(
                        f"{type(model_object).__name__}.{method_name}: {exc}"
                    )
                    continue
        visual_keys = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        if any(key in inputs for key in visual_keys):
            detail = "; ".join(failures) or "no compatible Qwen2.5-VL RoPE API was found"
            raise RuntimeError(
                "Unable to compute Qwen2.5-VL 3-axis M-RoPE positions for a multimodal record; "
                f"refusing an unsafe 1D fallback ({detail})"
            )
        length = inputs["input_ids"].shape[-1]
        position = torch.arange(length, device=self.device).view(1, 1, -1)
        return position.expand(3, 1, -1)

    def prepare_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], VideoProcessorMetadata]:
        """Create one batch-size-one processor input on the target device."""

        return prepare_qwen_messages(
            self.processor,
            messages,
            device=self.device,
            processor_kwargs=getattr(self.processor, "dflash_processor_kwargs", {}),
            video_reader=getattr(self, "video_reader", "torchvision"),
            image_min_pixels=getattr(self, "image_min_pixels", None),
            image_max_pixels=getattr(self, "image_max_pixels", None),
            video_num_frames=getattr(self, "video_num_frames", None),
            video_min_pixels=getattr(self, "video_min_pixels", None),
            video_max_pixels=getattr(self, "video_max_pixels", None),
        )

    def prepare_record(self, record: dict[str, Any], *, max_seq_length: int) -> PreparedExample:
        validate_manifest_record(record, require_target_response=True)
        self.validate_record_provenance(record)
        response = record["target_response"]
        response_ids = response.get("token_ids") if isinstance(response, dict) else None
        if response_ids is None:
            raise ValueError("target_response.token_ids is required; run prepare_responses first")
        messages = record["messages"]
        inputs, _ = self.prepare_messages(messages)
        prompt_ids = inputs["input_ids"]
        response_tensor = torch.tensor(response_ids, dtype=prompt_ids.dtype).view(1, -1)
        response_start = int(prompt_ids.shape[1])
        if response_start >= max_seq_length:
            raise ValueError(
                f"prompt consumes {response_start} tokens, leaving no response room under {max_seq_length}"
            )
        full_ids = torch.cat([prompt_ids, response_tensor], dim=1)
        if full_ids.shape[1] > max_seq_length:
            raise ValueError(
                f"clean sequence has {full_ids.shape[1]} tokens, exceeding max_seq_length={max_seq_length}; "
                "the response is not truncated because DFlash requires exact target token IDs"
            )
        if response_tensor.shape[1] < 2:
            raise ValueError(f"record {record.get('id')} has an empty/too-short response")
        inputs["input_ids"] = full_ids
        inputs["attention_mask"] = torch.ones_like(full_ids)
        inputs.pop("position_ids", None)
        inputs = _tensor_dict_to_device(inputs, self.device)
        position_ids = self._compute_position_ids(inputs)
        inputs["position_ids"] = position_ids
        response_end = int(full_ids.shape[1])
        return PreparedExample(
            inputs=inputs,
            input_ids=full_ids.to(self.device),
            position_ids=position_ids.to(self.device),
            response_start=response_start,
            response_end=response_end,
            sample_id=str(record.get("id", "")),
        )

    @torch.inference_mode()
    def forward_clean(self, inputs: dict[str, Any]) -> Any:
        self.model.eval()
        return self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )

    def selected_hidden_features(self, outputs: Any, layer_ids: Iterable[int]) -> torch.Tensor:
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Target did not return hidden_states")
        selected = [hidden_states[int(layer_id) + 1] for layer_id in layer_ids]
        return torch.cat(selected, dim=-1)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Manifest record must be an object at line {line_number}")
        records.append(value)
    return records
