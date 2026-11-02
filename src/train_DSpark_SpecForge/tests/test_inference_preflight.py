from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
DSPARK_ROOT = REPO_ROOT / "src" / "train_DSpark_SpecForge"
if str(DSPARK_ROOT) not in sys.path:
    sys.path.insert(0, str(DSPARK_ROOT))

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INFER = _load_script_module(
    "_dspark_infer_test_module",
    DSPARK_ROOT / "scripts" / "infer_qwen25vl_dspark.py",
)
PREPARE = _load_script_module(
    "_dspark_prepare_test_module",
    DSPARK_ROOT / "scripts" / "prepare_dspark_checkpoint.py",
)


@pytest.fixture
def published_config():
    path = DSPARK_ROOT / "configs" / "qwen2.5-vl-3b-dspark.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _state_for_config(config: dict) -> dict:
    dspark = config["dflash_config"]
    rank = int(dspark.get("markov_rank", 0))
    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    state_dict = {}
    if rank:
        state_dict["markov_head.markov_w1.weight"] = torch.empty(
            vocab, rank, device="meta"
        )
        state_dict["markov_head.markov_w2.weight"] = torch.empty(
            vocab, rank, device="meta"
        )
        head_type = dspark.get("markov_head_type", "vanilla")
        if head_type == "gated":
            state_dict["markov_head.gate_proj.weight"] = torch.empty(
                rank, hidden + rank, device="meta"
            )
        elif head_type == "rnn":
            state_dict["markov_head.joint_proj.weight"] = torch.empty(
                3 * rank, hidden + 2 * rank, device="meta"
            )
    if dspark.get("enable_confidence_head", False):
        confidence_input = hidden + (
            rank if dspark.get("confidence_head_with_markov", False) else 0
        )
        state_dict["confidence_head.proj.weight"] = torch.empty(
            1, confidence_input, device="meta"
        )
    return {
        "strategy": "dspark",
        "dspark_draft_num_hidden_layers": int(config["num_hidden_layers"]),
        "dspark_target_layer_ids": tuple(dspark["target_layer_ids"]),
        "dspark_block_size": int(config["block_size"]),
        "dspark_mask_token_id": int(dspark["mask_token_id"]),
        "draft_state_dict": state_dict,
    }


class _CaptureLMHead:
    def __init__(self):
        self.inputs = None

    def __call__(self, hidden):
        self.inputs = hidden.detach().clone()
        token_ids = hidden[..., 0].long()
        logits = torch.full((*token_ids.shape, 16), -100.0)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 100.0)
        return logits


def test_dspark_proposals_use_first_trained_hidden_positions():
    lm_head = _CaptureLMHead()
    target = SimpleNamespace(lm_head=lm_head)
    draft = SimpleNamespace(markov_head=None)
    hidden = torch.arange(8.0).view(1, 8, 1)
    block_ids = torch.tensor([[9, 0, 0, 0, 0, 0, 0, 0]])

    proposals = DSparkDraftModel._sample_draft_tokens(draft, target, hidden, block_ids)

    assert proposals.tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    torch.testing.assert_close(lm_head.inputs, hidden[:, :7])


def test_dspark_markov_sampler_receives_anchor_and_aligned_hidden_states():
    class CaptureMarkov:
        def sample_block_tokens(self, logits, **kwargs):
            self.logits = logits
            self.kwargs = kwargs
            return torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), logits

    markov = CaptureMarkov()
    target = SimpleNamespace(lm_head=lambda hidden: torch.zeros(1, hidden.shape[1], 8))
    draft = SimpleNamespace(markov_head=markov)
    hidden = torch.arange(8.0).view(1, 8, 1)
    block_ids = torch.tensor([[42, 0, 0, 0, 0, 0, 0, 0]])

    result = DSparkDraftModel._sample_draft_tokens(draft, target, hidden, block_ids)

    assert result.shape == (1, 7)
    assert markov.kwargs["first_prev_token_ids"].tolist() == [42]
    torch.testing.assert_close(markov.kwargs["hidden_states"], hidden[:, :7])


def test_dspark_proposal_generation_rejects_short_hidden_sequence():
    with pytest.raises(ValueError, match="fewer trained positions"):
        DSparkDraftModel._sample_draft_tokens(
            SimpleNamespace(markov_head=None),
            SimpleNamespace(lm_head=lambda hidden: torch.zeros(1, 7, 8)),
            torch.zeros(1, 6, 2),
            torch.zeros(1, 8, dtype=torch.long),
        )


def test_dspark_generation_uses_seven_draft_positions_and_eight_verifier_positions():
    draft = object.__new__(DSparkDraftModel)
    nn.Module.__init__(draft)
    draft.block_size = 7
    draft.confidence_head = None

    with mock.patch.object(
        DFlashDraftModel,
        "spec_generate",
        autospec=True,
        return_value="generated",
    ) as base_generate:
        result = draft.spec_generate(
            object(), torch.tensor([[1, 2]]), 4, [], 0.0
        )

    assert result == "generated"
    assert base_generate.call_args.kwargs["draft_block_size"] == 7
    assert base_generate.call_args.kwargs["verification_block_size"] == 8


class _ConfidenceStub:
    block_size = 3
    confidence_head = object()

    def __init__(self, logits):
        self.logits = logits
        self.seen = None

    def predict_confidence(self, hidden, *, prev_token_ids):
        self.seen = (hidden.detach().clone(), prev_token_ids.detach().clone())
        return self.logits[:, : hidden.shape[1]]


def test_threshold_scheduler_selects_a_cumulative_survival_prefix():
    model = _ConfidenceStub(torch.zeros(1, 3))
    selector = DSparkDraftModel._make_verification_length_selector(
        model,
        "threshold",
        confidence_threshold=0.5,
        step_rate_by_batch=None,
    )

    selected = selector(
        torch.zeros(1, 3, 4), torch.tensor([[11, 12, 13, 14]]), 3
    )

    assert selected == 1
    assert model.seen[1].tolist() == [[11, 12, 13]]


def test_profile_scheduler_uses_verifier_batches_through_block_size_plus_one():
    high_confidence = math.log(0.9 / 0.1)
    model = _ConfidenceStub(torch.full((1, 3), high_confidence))
    selector = DSparkDraftModel._make_verification_length_selector(
        model,
        "profile",
        confidence_threshold=0.0,
        step_rate_by_batch={1: 1.0, 2: 2.0, 3: 2.0, 4: 1.0},
    )

    assert selector(torch.zeros(1, 3, 4), torch.zeros(1, 4, dtype=torch.long), 3) == 2


@pytest.mark.parametrize(
    "rates",
    [
        {1: 1.0, 2: 1.0, 3: 1.0},
        {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0},
        {1: 1.0, 2: float("nan"), 3: 1.0, 4: 1.0},
    ],
)
def test_profile_scheduler_rejects_incomplete_or_invalid_rates(rates):
    with pytest.raises(ValueError, match="invalid throughput profile"):
        DSparkDraftModel._make_verification_length_selector(
            _ConfidenceStub(torch.zeros(1, 3)),
            "profile",
            confidence_threshold=0.0,
            step_rate_by_batch=rates,
        )


def test_profile_scheduler_rejects_missing_confidence_head():
    with pytest.raises(ValueError, match="confidence head"):
        DSparkDraftModel._make_verification_length_selector(
            SimpleNamespace(block_size=3, confidence_head=None),
            "threshold",
            confidence_threshold=0.5,
            step_rate_by_batch=None,
        )


def test_checkpoint_metadata_validates_published_dspark_config(published_config):
    config_path, config = published_config
    validated = INFER._load_config_and_validate_checkpoint(
        _state_for_config(config), config_path
    )
    assert validated["block_size"] == 7
    assert validated["dflash_config"]["target_layer_ids"] == [26, 28, 30, 32, 34]


def test_checkpoint_metadata_accepts_dspark_without_markov_head(
    published_config, tmp_path
):
    _config_path, config = published_config
    config = copy.deepcopy(config)
    config["dflash_config"]["markov_rank"] = 0
    config["dflash_config"]["confidence_head_with_markov"] = False
    state = _state_for_config(config)
    config_path = tmp_path / "no_markov_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert (
        INFER._load_config_and_validate_checkpoint(state, config_path)["dflash_config"][
            "markov_rank"
        ]
        == 0
    )


def test_checkpoint_metadata_rejects_missing_markov_projection(published_config):
    config_path, config = published_config
    state = _state_for_config(config)
    state["draft_state_dict"].pop("markov_head.markov_w2.weight")

    with pytest.raises(ValueError, match="markov_head.markov_w2.weight"):
        INFER._load_config_and_validate_checkpoint(state, config_path)


def test_target_snapshot_provenance_checks_config_hash_and_weight_size(tmp_path):
    config_bytes = b'{"model_type":"qwen2_5_vl"}\n'
    (tmp_path / "config.json").write_bytes(config_bytes)
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights")
    fingerprints = (
        ("sha256", "config.json", len(config_bytes), hashlib.sha256(config_bytes).hexdigest()),
        ("stat", "model-00001.safetensors", len(b"weights"), 123),
    )
    state = {
        "specforge_model_provenance": {
            "target_model": ("directory-stat-v2", str(tmp_path), fingerprints)
        }
    }

    assert INFER._verify_target_snapshot(
        state, tmp_path, allow_mismatch=False
    ) == []
    (tmp_path / "config.json").write_bytes(config_bytes.replace(b"vl", b"xl"))
    with pytest.raises(ValueError, match="SHA-256"):
        INFER._verify_target_snapshot(state, tmp_path, allow_mismatch=False)


def test_training_state_can_be_loaded_with_memory_mapping(tmp_path):
    checkpoint = tmp_path / "training_state.pt"
    torch.save({"draft_state_dict": {"weight": torch.arange(8)}}, checkpoint)

    state = INFER.resolve_training_state(str(checkpoint), mmap=True)

    torch.testing.assert_close(
        state["draft_state_dict"]["weight"], torch.arange(8)
    )


def test_checkpoint_selector_prefers_latest_phase_two_and_filters_run():
    candidates = {
        "phase1-old": {
            "phase": "phase1",
            "run_id": "targets_high",
            "step": 6126,
            "files": {"training_state.pt": {}},
        },
        "phase2-spread": {
            "phase": "phase2",
            "run_id": "targets_spread",
            "step": 6372,
            "files": {"training_state.pt": {}},
        },
        "phase2-high-old": {
            "phase": "phase2",
            "run_id": "targets_high",
            "step": 6200,
            "files": {"training_state.pt": {}},
        },
        "phase2-high-latest": {
            "phase": "phase2",
            "run_id": "targets_high",
            "step": 6372,
            "files": {"training_state.pt": {}},
        },
    }

    selected_path, selected = PREPARE._select_checkpoint(
        candidates, "auto", "targets_high", "latest"
    )

    assert selected_path == "phase2-high-latest"
    assert selected["phase"] == "phase2"
    assert selected["step"] == 6372


def test_checkpoint_selector_requires_run_when_phase_has_multiple_runs():
    candidates = {
        run_id: {
            "phase": "phase2",
            "run_id": run_id,
            "step": 6372,
            "files": {"training_state.pt": {}},
        }
        for run_id in ("targets_high", "targets_spread")
    }

    with pytest.raises(ValueError, match="multiple runs"):
        PREPARE._select_checkpoint(candidates, "phase2", None, "latest")


def test_tiny_dspark_speculative_generation_matches_target_greedy_on_cpu(request):
    from transformers import Qwen3Config
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    previous_threads = torch.get_num_threads()
    previous_rng_state = torch.random.get_rng_state()
    request.addfinalizer(lambda: torch.set_num_threads(previous_threads))
    request.addfinalizer(lambda: torch.random.set_rng_state(previous_rng_state))
    torch.set_num_threads(1)
    torch.manual_seed(73)

    text_config = Qwen2_5_VLTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    vision_config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=32,
        fullatt_block_indexes=[],
    )
    target_config = Qwen2_5_VLConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=123,
        video_token_id=124,
        vision_start_token_id=125,
        vision_end_token_id=126,
        eos_token_id=127,
        attn_implementation="eager",
    )
    target = Qwen2_5_VLForConditionalGeneration(target_config).to(
        dtype=torch.bfloat16
    ).eval()

    draft_config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_target_layers=2,
        max_position_embeddings=1024,
        layer_types=["full_attention"],
    )
    draft_config.block_size = 3
    draft_config._attn_implementation = "eager"
    draft_config.dflash_config = {
        "projector_type": "dspark",
        "target_layer_ids": [0],
        "position_encoding": "qwen2_5_vl_mrope",
        "mrope_section": [1, 1, 2],
        "mask_token_id": 127,
        "markov_rank": 8,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
    }
    draft = DSparkDraftModel(draft_config).to(dtype=torch.bfloat16).eval()

    input_ids = torch.tensor([[1, 9, 17, 21]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    reference = target.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=12,
        do_sample=False,
        use_cache=True,
        eos_token_id=[127],
    )
    speculative = draft.spec_generate(
        target,
        input_ids,
        max_new_tokens=12,
        stop_token_ids=[127],
        temperature=0.0,
        position_ids=position_ids,
        target_kwargs={"attention_mask": attention_mask},
    )

    assert torch.equal(speculative, reference)
