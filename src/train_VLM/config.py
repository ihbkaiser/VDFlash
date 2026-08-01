from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class DFlashTrainConfig:
    """Configuration shared by preparation, training and evaluation.

    Values which are not specified by the DFlash paper are explicit defaults so
    that a run can be reproduced.  In particular, the weighted CE reduction is
    normalized by the sum of token weights, and AdamW uses PyTorch defaults.
    """

    target_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    target_revision: str | None = None
    context_mode: Literal["full", "text_only"] = "full"
    max_seq_length: int = 3072
    block_size: int = 16
    num_anchors: int = 512
    anchor_chunk_size: int = 64
    min_anchor_chunk_size: int = 8
    num_draft_layers: int = 5
    num_target_features: int = 5
    loss_decay: float | None = None

    epochs: int = 6
    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.04
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 8
    seed: int = 42
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    use_flex_attention: bool = True
    compile_flex_attention: bool = True
    gradient_checkpointing: bool = True

    response_max_new_tokens: int = 1024
    temperature: float = 0.0
    save_every_steps: int = 1000
    keep_last_checkpoints: int = 3
    resume_from_checkpoint: str = ""
    output_dir: str = "checkpoints/dflash_qwen25vl"
    train_manifest: str = ""
    validation_manifest: str = ""

    processor_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_mode not in {"full", "text_only"}:
            raise ValueError("context_mode must be 'full' or 'text_only'")
        if self.mixed_precision not in {"no", "bf16", "fp16"}:
            raise ValueError("mixed_precision must be one of: no, bf16, fp16")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.num_anchors < 1 or self.anchor_chunk_size < 1 or self.min_anchor_chunk_size < 1:
            raise ValueError("num_anchors and anchor chunk sizes must be positive")
        if self.min_anchor_chunk_size > self.anchor_chunk_size:
            raise ValueError("min_anchor_chunk_size must be <= anchor_chunk_size")
        if self.anchor_chunk_size > self.num_anchors:
            self.anchor_chunk_size = self.num_anchors
        if self.keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be positive")
        if self.loss_decay is None:
            self.loss_decay = {16: 7.0, 10: 5.0, 8: 4.0}.get(
                self.block_size, max(1.0, self.block_size / 2)
            )
        if self.max_seq_length < self.block_size:
            raise ValueError("max_seq_length must be >= block_size")

    @property
    def predicted_tokens_per_block(self) -> int:
        return self.block_size - 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_file(cls, path: str | Path) -> "DFlashTrainConfig":
        path = Path(path)
        raw = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - optional CLI dependency
                raise RuntimeError("PyYAML is required for YAML configs") from exc
            values = yaml.safe_load(raw) or {}
        else:
            values = json.loads(raw)
        if not isinstance(values, dict):
            raise ValueError(f"Config must contain an object: {path}")
        return cls(**values)


def config_from_target(target_config: Any, **overrides: Any) -> DFlashTrainConfig:
    """Build a config while preserving the target model name if available."""

    name = getattr(target_config, "_name_or_path", None)
    values: dict[str, Any] = {}
    if name:
        values["target_model"] = name
    values.update(overrides)
    return DFlashTrainConfig(**values)
