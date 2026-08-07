import json
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
import pytest

from src.train_VLM.data import (
    build_masked_blocks,
    make_anchor_generator,
    make_dense_attention_mask,
    sample_anchor_positions,
    select_context_positions,
)
from src.train_VLM.decode import speculative_decode
from src.train_VLM.config import DFlashTrainConfig
from src.train_VLM.cached_trainer import FrozenTargetTokenIO
from src.train_VLM.losses import weighted_block_cross_entropy
from src.train_VLM.model import DFlashVLMModel, MultiModalRotaryEmbedding, RMSNorm
from src.train_VLM.teacher_cache import (
    _PreparedTeacherExample,
    _batch_text_inputs,
    _fit_clean_sequence,
    _generate_target_response_ids,
)
from src.train_VLM.target import Qwen25VLTargetAdapter, load_jsonl
from src.train_VLM.trainer import (
    load_draft_checkpoint,
    make_draft_model,
    save_draft_checkpoint,
    train_records,
)
from src.train_VLM.vlm_decode import Qwen25VLDFlashDecoder
from src.train_VLM.video import _apply_media_defaults
from src.train_VLM.real_data import prepare_real_manifest, select_source_records


def tiny_text_config():
    return SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_bias=True,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        initializer_range=0.02,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [2, 1, 1]},
    )


def test_anchor_blocks_and_causal_context():
    ids = torch.arange(30)
    position_ids = torch.arange(30).view(1, 1, 30).expand(3, 1, 30)
    anchors = sample_anchor_positions(
        5, 25, 4, 3, generator=torch.Generator().manual_seed(7)
    )
    blocks = build_masked_blocks(
        ids, anchors, block_size=4, mask_token_id=99, position_ids=position_ids
    )
    assert torch.equal(blocks.block_input_ids[:, 0], ids[anchors])
    assert torch.all(blocks.block_input_ids[:, 1:] == 99)
    assert torch.all(blocks.labels[:, 0] == -100)
    assert blocks.block_position_ids.shape == (3, 1, 12)
    mask = make_dense_attention_mask(anchors, torch.arange(30), block_size=4)[0, 0]
    for block, anchor in enumerate(anchors.tolist()):
        rows = mask[block * 4 : block * 4 + 4]
        assert torch.all(rows[:, :anchor])
        assert not torch.any(rows[:, anchor:30])
        own = rows[:, 30 + block * 4 : 30 + (block + 1) * 4]
        assert torch.all(own)
        other = torch.cat([rows[:, 30 : 30 + block * 4], rows[:, 30 + (block + 1) * 4 :]], dim=1)
        assert not torch.any(other)


def test_weighted_loss_prefers_first_prediction():
    labels = torch.tensor([[[1, 2, 3]]])
    logits = torch.zeros(1, 1, 3, 5)
    logits[0, 0, 0, 1] = 10
    loss, metrics = weighted_block_cross_entropy(logits, labels, decay=7)
    assert metrics["valid_tokens"] == 3
    assert torch.isfinite(loss)


def test_video_defaults_are_parameterized_without_mutating_messages():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "clip.mp4"},
                {"type": "video", "video": "fixed.mp4", "nframes": 6, "max_pixels": 20000},
            ],
        }
    ]
    materialized = _apply_media_defaults(
        messages,
        video_num_frames=8,
        video_min_pixels=12544,
        video_max_pixels=12544,
    )
    assert "nframes" not in messages[0]["content"][0]
    assert materialized[0]["content"][0] == {
        "type": "video",
        "video": "clip.mp4",
        "nframes": 8,
        "min_pixels": 12544,
        "max_pixels": 12544,
    }
    assert materialized[0]["content"][1]["nframes"] == 6
    assert materialized[0]["content"][1]["max_pixels"] == 20000


def test_real_dataset_selection_is_deterministic_and_records_source_indices(tmp_path):
    records = [
        {
            "id": f"real-{index}",
            "conversations": [
                {"from": "human", "value": f"question {index}"},
                {"from": "gpt", "value": f"answer {index}"},
            ],
        }
        for index in range(8)
    ]
    annotation = tmp_path / "sharegpt.json"
    annotation.write_text(json.dumps(records))
    first, valid = select_source_records(
        annotation,
        stage="text",
        seed=42,
        max_samples=4,
    )
    second, _ = select_source_records(
        annotation,
        stage="text",
        seed=42,
        max_samples=4,
    )
    assert first == second
    assert valid == 8
    assert len(first) == 4
    assert all(records[source_index]["id"] == source_id for _, source_index, source_id in first)


def test_concurrent_manifest_writers_never_publish_partial_jsonl(tmp_path):
    records = [
        {
            "id": f"real-{index}",
            "conversations": [
                {"from": "human", "value": f"question {index}"},
                {"from": "gpt", "value": (f"answer {index} " * 100).strip()},
            ],
        }
        for index in range(32)
    ]
    annotation = tmp_path / "sharegpt.json"
    annotation.write_text(json.dumps(records))
    manifest = tmp_path / "shared" / "manifest.jsonl"
    config = DFlashTrainConfig(
        stage="text",
        data_path=str(annotation),
        prepared_manifest=str(manifest),
        max_samples=32,
        overwrite=True,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: prepare_real_manifest(config), range(2)))

    assert all(result[0] == manifest for result in results)
    parsed = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(parsed) == 32
    assert not list(manifest.parent.glob(f".{manifest.name}.*.tmp"))


def test_load_jsonl_preserves_unicode_line_and_paragraph_separators(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    expected = [
        {"id": "line-separator", "text": "before\u2028after"},
        {"id": "paragraph-separator", "text": "before\u2029after"},
    ]
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in expected),
        encoding="utf-8",
    )

    assert load_jsonl(manifest) == expected


def test_cached_target_token_io_is_frozen_but_backpropagates_to_draft_hidden():
    token_io = FrozenTargetTokenIO(torch.randn(7, 4), None)
    hidden = torch.randn(2, 4, requires_grad=True)
    token_io.logits(hidden).sum().backward()
    assert hidden.grad is not None
    assert token_io.embedding_weight.grad is None
    assert not token_io.requires_grad


def test_pipeline_stage_defaults_and_selected_layers_validation():
    assert DFlashTrainConfig().target_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert DFlashTrainConfig(stage="text").dataset_repo == (
        "anon8231489123/ShareGPT_Vicuna_unfiltered"
    )
    assert DFlashTrainConfig(stage="multimodal").dataset_repo == "liuhaotian/LLaVA-Pretrain"
    with pytest.raises(ValueError, match="exactly num_target_features"):
        DFlashTrainConfig(num_target_features=2, selected_target_layers=[1])


def test_dataset_teacher_mode_disables_generation_budget():
    config = DFlashTrainConfig(
        teacher_response_mode="dataset",
        response_max_new_tokens=0,
        max_seq_length=2048,
    )
    assert config.response_max_new_tokens == 0
    with pytest.raises(ValueError, match="target_generate"):
        DFlashTrainConfig(
            teacher_response_mode="target_generate",
            response_max_new_tokens=0,
        )
    batched = DFlashTrainConfig(teacher_batch_size=8)
    assert batched.teacher_length_bucket_size == 128
    with pytest.raises(ValueError, match="bucket_size"):
        DFlashTrainConfig(teacher_batch_size=8, teacher_length_bucket_size=4)


def test_sharegpt_manifest_keeps_multi_turn_context_and_final_response(tmp_path):
    annotation = tmp_path / "sharegpt.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "id": "multi-turn",
                    "conversations": [
                        {"from": "human", "value": "first question"},
                        {"from": "gpt", "value": "first answer"},
                        {"from": "human", "value": "follow-up"},
                        {"from": "gpt", "value": "dataset final answer"},
                    ],
                }
            ]
        )
    )
    manifest = tmp_path / "manifest.jsonl"
    config = DFlashTrainConfig(
        stage="text",
        data_path=str(annotation),
        prepared_manifest=str(manifest),
        max_samples=1,
    )

    prepare_real_manifest(config)
    record = load_jsonl(manifest)[0]

    assert [message["role"] for message in record["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert record["messages"][-1]["content"][0]["text"] == "follow-up"
    assert record["target_text"] == "dataset final answer"


def test_dflash_tiny_forward_and_backward():
    config = tiny_text_config()
    model = DFlashVLMModel(config, num_draft_layers=2, num_target_features=2, block_size=4)
    anchors = torch.tensor([3, 5])
    context_pos = torch.arange(12).view(1, 1, 12).expand(3, 1, 12)
    block_pos = torch.arange(8).view(1, 1, 8).expand(3, 1, 8)
    output = model(
        noise_embeddings=torch.randn(1, 8, 32),
        target_context=torch.randn(1, 12, 64),
        context_position_ids=context_pos,
        block_position_ids=block_pos,
        anchors=anchors,
        context_original_positions=torch.arange(12),
        use_flex_attention=False,
    )
    assert output.shape == (1, 8, 32)
    output.square().mean().backward()
    assert model.fc.weight.grad is not None


def test_draft_context_kv_cache_only_grows_with_new_target_context():
    config = tiny_text_config()
    model = DFlashVLMModel(config, num_draft_layers=2, num_target_features=2, block_size=4).eval()
    first_context = torch.randn(1, 6, 64)
    first_positions = torch.arange(6).view(1, 1, 6).expand(3, 1, 6)
    first_hidden, cache = model(
        noise_embeddings=torch.randn(1, 4, 32),
        target_context=first_context,
        context_position_ids=first_positions,
        block_position_ids=torch.arange(4).view(1, 1, 4).expand(3, 1, 4),
        anchors=torch.tensor([5]),
        context_original_positions=torch.arange(6),
        use_flex_attention=False,
        return_draft_context_cache=True,
    )
    assert first_hidden.shape == (1, 4, 32)
    assert [item[0].shape[2] for item in cache] == [6, 6]
    second_context = torch.randn(1, 2, 64)
    second_positions = torch.arange(8).view(1, 1, 8).expand(3, 1, 8)
    second_hidden, grown_cache = model(
        noise_embeddings=torch.randn(1, 4, 32),
        target_context=second_context,
        context_position_ids=second_positions[:, :, -2:],
        block_position_ids=torch.arange(4, 8).view(1, 1, 4).expand(3, 1, 4),
        anchors=torch.tensor([7]),
        context_original_positions=torch.arange(8),
        use_flex_attention=False,
        draft_context_cache=cache,
        return_draft_context_cache=True,
    )
    assert second_hidden.shape == (1, 4, 32)
    assert [item[0].shape[2] for item in grown_cache] == [8, 8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlexAttention integration test")
def test_cuda_flex_attention_gqa_forward_and_backward():
    config = SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_bias=True,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        initializer_range=0.02,
        rope_theta=10000.0,
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [4, 2, 2]},
    )
    model = DFlashVLMModel(
        config,
        num_draft_layers=2,
        num_target_features=2,
        block_size=16,
        compile_flex_attention=True,
    ).cuda().train()
    output = model(
        noise_embeddings=torch.randn(1, 32, 64, device="cuda"),
        target_context=torch.randn(1, 32, 128, device="cuda"),
        context_position_ids=torch.arange(32, device="cuda").view(1, 1, 32).expand(3, 1, 32),
        block_position_ids=torch.arange(32, device="cuda").view(1, 1, 32).expand(3, 1, 32),
        anchors=torch.tensor([3, 20], device="cuda"),
        context_original_positions=torch.arange(32, device="cuda"),
        use_flex_attention=True,
    )
    output.square().mean().backward()
    assert output.shape == (1, 32, 64)
    assert model.fc.weight.grad is not None


def test_callback_decoder_keeps_target_posterior():
    def draft(prefix, count):
        return torch.ones(count, dtype=torch.long)

    def verify(prefix, proposals):
        # Target agrees with every proposal and emits token 2 as the bonus.
        logits = torch.full((proposals.numel() + 1, 4), -10.0)
        logits[:-1, 1] = 10.0
        logits[-1, 2] = 10.0
        return logits

    output, stats = speculative_decode(
        torch.tensor([0]),
        draft_propose=draft,
        target_verify=verify,
        max_new_tokens=3,
        block_size=3,
    )
    assert output.tolist() == [0, 1, 1, 2]
    assert stats.mean_acceptance_length == 3


def test_callback_decoder_never_emits_after_eos():
    def draft(prefix, count):
        return torch.tensor([1, 2][:count])

    def verify(prefix, proposals):
        logits = torch.full((proposals.numel() + 1, 5), -10.0)
        for index, token_id in enumerate(proposals.tolist()):
            logits[index, token_id] = 10.0
        logits[-1, 3] = 10.0  # Would be a bonus after EOS without truncation.
        return logits

    output, stats = speculative_decode(
        torch.tensor([0]),
        draft_propose=draft,
        target_verify=verify,
        max_new_tokens=3,
        block_size=3,
        stop_token_ids=[2],
    )
    assert output.tolist() == [0, 1, 2]
    assert stats.acceptance_lengths == [2]


def test_vlm_decoder_is_lossless_for_perfect_and_always_wrong_drafts():
    vocab_size = 32

    class Cache:
        def __init__(self):
            self.key_cache = [torch.zeros(1, 1, 0, 1)]

        def get_seq_length(self):
            return self.key_cache[0].shape[-2]

        def append(self, count):
            self.key_cache[0] = torch.zeros(1, 1, self.get_seq_length() + count, 1)

        def crop(self, length):
            self.key_cache[0] = self.key_cache[0][..., :length, :]

    class Target:
        def eval(self):
            return self

        def __call__(self, input_ids, past_key_values=None, **_kwargs):
            cache = past_key_values or Cache()
            cache.append(input_ids.shape[1])
            next_ids = (input_ids + 1) % vocab_size
            logits = torch.full((*input_ids.shape, vocab_size), -1_000.0)
            logits.scatter_(-1, next_ids.unsqueeze(-1), 1_000.0)
            return SimpleNamespace(logits=logits, past_key_values=cache)

    class Adapter:
        device = torch.device("cpu")
        visual_token_ids = (set(), set())
        _set_input_sequence = Qwen25VLTargetAdapter._set_input_sequence

        def __init__(self):
            self.model = Target()
            self.input_embeddings = lambda token_ids: F.one_hot(
                token_ids, num_classes=vocab_size
            ).float()
            self.lm_head = lambda hidden: hidden

        def _compute_position_ids(self, inputs):
            positions = torch.arange(inputs["input_ids"].shape[1]).view(1, 1, -1)
            return positions.expand(3, 1, -1)

        def selected_hidden_features(self, outputs, layer_ids):
            return torch.zeros(1, outputs.logits.shape[1], 4 * len(layer_ids))

    class Draft:
        mask_token_id = vocab_size - 1
        target_layer_ids = [0]

        def __init__(self, perfect):
            self.perfect = perfect

        def eval(self):
            return self

        def __call__(self, noise_embeddings, return_draft_context_cache=False, **_kwargs):
            hidden = torch.zeros_like(noise_embeddings)
            anchor = noise_embeddings[:, 0].argmax(dim=-1)
            for offset in range(hidden.shape[1]):
                token = (anchor + offset) % vocab_size if self.perfect else anchor * 0
                hidden[:, offset] = F.one_hot(token, num_classes=vocab_size).float()
            return (hidden, []) if return_draft_context_cache else hidden

    config = SimpleNamespace(block_size=4, context_mode="full", use_flex_attention=False)
    prompt = torch.tensor([[1]])
    inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "position_ids": torch.zeros(3, 1, 1, dtype=torch.long),
    }
    expected = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    perfect = Qwen25VLDFlashDecoder(Adapter(), Draft(True), config).generate(
        inputs, max_new_tokens=8
    )
    wrong = Qwen25VLDFlashDecoder(Adapter(), Draft(False), config).generate(
        inputs, max_new_tokens=8
    )
    assert torch.equal(perfect.output_ids, expected)
    assert [step.accepted_proposals for step in perfect.steps] == [3, 2]
    assert torch.equal(wrong.output_ids, expected)
    assert all(step.accepted_proposals == 0 for step in wrong.steps)


def test_anchor_sampling_is_order_independent_per_epoch_and_id():
    def anchors_for(sample_id):
        return sample_anchor_positions(
            10,
            100,
            16,
            8,
            generator=make_anchor_generator(42, 3, sample_id),
        )

    a_first = anchors_for("sample-a")
    _ = anchors_for("sample-b")
    a_second = anchors_for("sample-a")
    assert torch.equal(a_first, a_second)
    assert not torch.equal(a_first, sample_anchor_positions(
        10, 100, 16, 8, generator=make_anchor_generator(42, 4, "sample-a")
    ))


def test_text_only_context_keeps_boundaries_and_drops_visual_placeholders():
    tokens = torch.tensor([11, 98, 96, 99, 97, 12])
    retained = select_context_positions(
        tokens,
        context_mode="text_only",
        image_token_ids={98},
        video_token_ids={99},
    )
    assert retained.tolist() == [0, 2, 4, 5]


def test_mrope_supports_different_query_and_key_lengths():
    rotary = MultiModalRotaryEmbedding(tiny_text_config())
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 5, 8)
    query_positions = torch.arange(3).view(1, 1, 3).expand(3, 1, 3)
    key_positions = torch.arange(5).view(1, 1, 5).expand(3, 1, 5)
    rotated_query, rotated_key = rotary.apply(query, key, query_positions, key_positions)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert torch.isfinite(rotated_query).all() and torch.isfinite(rotated_key).all()


def test_mrope_numerically_matches_qwen25vl_for_visual_positions():
    from transformers import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLRotaryEmbedding,
        apply_multimodal_rotary_pos_emb,
    )

    config = Qwen2_5_VLTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_theta=10_000.0,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
    )
    query = torch.randn(1, 4, 7, 8)
    key = torch.randn(1, 2, 7, 8)
    positions = torch.stack(
        [torch.arange(7), torch.arange(7) % 3, torch.arange(7) % 5]
    ).unsqueeze(1)
    official_rotary = Qwen2_5_VLRotaryEmbedding(config)
    cos, sin = official_rotary(query, positions)
    expected_query, expected_key = apply_multimodal_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        config.rope_scaling["mrope_section"],
    )
    actual_query, actual_key = MultiModalRotaryEmbedding(config).apply(
        query, key, positions, positions
    )
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


def test_rmsnorm_numerically_matches_qwen_in_bf16():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    hidden = torch.randn(4, 32).bfloat16()
    expected_norm = Qwen2RMSNorm(32, eps=1e-6).to(dtype=torch.bfloat16)
    actual_norm = RMSNorm(32, eps=1e-6).to(dtype=torch.bfloat16)
    with torch.no_grad():
        actual_norm.weight.copy_(expected_norm.weight)
    assert torch.equal(actual_norm(hidden), expected_norm(hidden))


def test_target_generated_teacher_response_uses_raw_greedy_settings():
    calls = {}

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, **kwargs):
            calls.update(kwargs)
            return torch.tensor([[1, 2, 3, 4, 9]])

    adapter = SimpleNamespace(
        model=Model(),
        processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=9)),
    )
    config = DFlashTrainConfig(
        max_seq_length=16,
        block_size=2,
        response_max_new_tokens=8,
        teacher_response_mode="target_generate",
    )
    response, reached_limit = _generate_target_response_ids(
        adapter,
        {"input_ids": torch.tensor([[1, 2, 3]])},
        prompt_length=3,
        config=config,
    )
    assert response == [4, 9]
    assert not reached_limit
    assert calls["do_sample"] is False
    assert calls["repetition_penalty"] == 1.0
    assert calls["temperature"] is None


def test_teacher_response_extends_multimodal_token_types_as_text():
    class Model:
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, input_ids, **_kwargs):
            return torch.cat([input_ids, torch.tensor([[4, 9]])], dim=-1)

        def get_rope_index(self, input_ids, attention_mask, mm_token_type_ids, **_kwargs):
            assert input_ids.shape == attention_mask.shape == mm_token_type_ids.shape
            positions = torch.arange(input_ids.shape[-1]).view(1, 1, -1)
            return positions.expand(3, input_ids.shape[0], -1), None

    adapter = object.__new__(Qwen25VLTargetAdapter)
    adapter.model = Model()
    adapter.device = torch.device("cpu")
    adapter.processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=9))
    adapter.prepare_messages = MethodType(
        lambda self, _messages: (
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[1, 1, 0]]),
            },
            None,
        ),
        adapter,
    )
    config = DFlashTrainConfig(
        max_seq_length=16,
        block_size=2,
        response_max_new_tokens=8,
        teacher_response_mode="target_generate",
    )

    inputs, position_ids, response_start, response_end, _, _ = _fit_clean_sequence(
        adapter,
        {"messages": [{"role": "user", "content": "x"}]},
        config,
    )

    assert inputs["mm_token_type_ids"].tolist() == [[1, 1, 0, 0, 0]]
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape == (1, 5)
    assert position_ids.shape == (3, 1, 5)
    assert (response_start, response_end) == (3, 5)


def test_batched_dataset_teacher_preparation_stays_on_cpu_until_forward():
    class Tokenizer:
        def __call__(self, _text, **_kwargs):
            return SimpleNamespace(input_ids=[4, 5])

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, _messages, *, add_generation_prompt, **_kwargs):
            return "prompt" if add_generation_prompt else "prompt response"

    adapter = object.__new__(Qwen25VLTargetAdapter)
    adapter.device = torch.device("meta")
    adapter.processor = Processor()
    adapter.prepare_messages = MethodType(
        lambda self, _messages, *, device=None: (
            {
                "input_ids": torch.tensor([[1, 2, 3]], device=device),
                "attention_mask": torch.ones(1, 3, dtype=torch.long, device=device),
            },
            None,
        ),
        adapter,
    )
    config = DFlashTrainConfig(
        stage="text",
        max_seq_length=16,
        block_size=2,
        teacher_batch_size=2,
        teacher_response_mode="dataset",
        response_max_new_tokens=0,
    )

    inputs, position_ids, response_start, response_end, _, _ = _fit_clean_sequence(
        adapter,
        {
            "messages": [{"role": "user", "content": "x"}],
            "target_text": "response",
        },
        config,
    )

    assert inputs["input_ids"].device.type == "cpu"
    assert inputs["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
    assert position_ids.device.type == "cpu"
    assert (response_start, response_end) == (3, 5)


def test_multimodal_position_ids_fail_closed_without_qwen_rope_api():
    adapter = object.__new__(Qwen25VLTargetAdapter)
    adapter.model = SimpleNamespace()
    adapter.device = torch.device("cpu")
    with pytest.raises(RuntimeError, match="3-axis M-RoPE"):
        adapter._compute_position_ids(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.randn(1, 3, 2, 2),
            }
        )


def test_prepared_sequence_is_not_silently_truncated():
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    adapter = object.__new__(Qwen25VLTargetAdapter)
    adapter.processor = Processor()
    adapter.device = torch.device("cpu")
    adapter.validate_record_provenance = MethodType(lambda self, record: None, adapter)
    record = {
        "id": "long",
        "messages": [{"role": "user", "content": "x"}],
        "target_response": {
            "token_ids": [4, 5, 6],
            "text": "x",
            "generation": {"do_sample": False},
        },
    }
    with pytest.raises(ValueError, match="not truncated"):
        adapter.prepare_record(record, max_seq_length=5)


def test_tiny_qwen25vl_decoder_uses_target_cache():
    transformers = __import__("transformers")
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        import pytest

        pytest.skip("Qwen2.5-VL is unavailable in this Transformers build")
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    target_config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 100,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "rope_theta": 1_000_000,
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 0]},
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "spatial_merge_size": 1,
            "patch_size": 2,
            "temporal_patch_size": 1,
        },
        image_token_id=98,
        video_token_id=99,
        vision_start_token_id=96,
        vision_end_token_id=97,
    )
    target = Qwen2_5_VLForConditionalGeneration(target_config)

    class Tokenizer:
        all_special_ids = [0, 96, 97]
        name_or_path = "tiny-qwen25vl"

        def __len__(self):
            return 90

        def get_vocab(self):
            return {str(index): index for index in range(90)}

    class Processor:
        tokenizer = Tokenizer()

    adapter = Qwen25VLTargetAdapter(target, Processor(), device=torch.device("cpu"))
    config = DFlashTrainConfig(
        target_model="tiny-qwen25vl",
        block_size=4,
        num_draft_layers=2,
        num_target_features=5,
        mixed_precision="no",
        use_flex_attention=False,
    )
    adapter.freeze()
    draft = make_draft_model(adapter, config)
    decoder = Qwen25VLDFlashDecoder(adapter, draft, config)
    prompt = torch.tensor([[3, 4, 5]])
    inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "position_ids": torch.arange(3).view(1, 1, -1).expand(3, 1, -1),
    }
    longer_ids = torch.tensor([[6, 7, 8, 9, 10]])
    longer_inputs = {
        "input_ids": longer_ids,
        "attention_mask": torch.ones_like(longer_ids),
        "position_ids": torch.arange(5).view(1, 1, -1).expand(3, 1, -1),
    }
    layer_ids = [1, 4]
    expected_prompt = adapter.selected_hidden_features(
        adapter.forward_clean(inputs), layer_ids
    )
    expected_longer = adapter.selected_hidden_features(
        adapter.forward_clean(longer_inputs), layer_ids
    )
    teacher_examples = [
        _PreparedTeacherExample(
            source_offset=index,
            record={"id": str(index)},
            inputs=values,
            position_ids=values["position_ids"],
            response_start=1,
            response_end=int(values["input_ids"].shape[1]),
            response_truncated=False,
            dropped_turns=0,
        )
        for index, values in enumerate((inputs, longer_inputs))
    ]
    batched_inputs = _batch_text_inputs(teacher_examples, pad_token_id=0)
    selected = adapter.forward_selected_hidden(batched_inputs, layer_ids)
    torch.testing.assert_close(selected[0, :3], expected_prompt[0])
    torch.testing.assert_close(selected[1, :5], expected_longer[0])

    result = decoder.generate(inputs, max_new_tokens=5)
    assert result.output_ids.shape == (1, 8)
    assert sum(result.acceptance_lengths) >= 3
    assert result.target_forward_calls == len(result.acceptance_lengths) + 1
    assert result.num_output_tokens == 5
    assert result.end_to_end_latency_s >= result.prefill_latency_s
    assert result.final_cache_length == result.output_ids.shape[1] - 1
    assert all(step.target_cache_length == step.target_cache_key_shape[-2] for step in result.steps)


def test_draft_checkpoint_round_trip_validates_processor_contract(tmp_path):
    transformers = __import__("transformers")
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        pytest.skip("Qwen2.5-VL is unavailable in this Transformers build")
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    target = Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(
            text_config={
                "vocab_size": 100,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "rope_theta": 1_000_000,
                "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 0]},
                "pad_token_id": 0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 16,
                "spatial_merge_size": 1,
                "patch_size": 2,
                "temporal_patch_size": 1,
            },
            image_token_id=98,
            video_token_id=99,
        )
    )

    class Tokenizer:
        all_special_ids = [0]
        name_or_path = "tiny-qwen25vl"

        def __len__(self):
            return 90

        def get_vocab(self):
            return {str(index): index for index in range(90)}

    class Processor:
        tokenizer = Tokenizer()

    adapter = Qwen25VLTargetAdapter(target, Processor(), device=torch.device("cpu"))
    adapter.requested_model = "tiny-qwen25vl"
    adapter.requested_revision = None
    config = DFlashTrainConfig(
        target_model="tiny-qwen25vl",
        block_size=4,
        num_draft_layers=2,
        num_target_features=5,
        mixed_precision="no",
        use_flex_attention=False,
    )
    draft = make_draft_model(adapter, config)
    save_draft_checkpoint(tmp_path, draft, config, adapter, step=7)
    restored = load_draft_checkpoint(tmp_path, adapter, config)
    for key, value in draft.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    mismatched = DFlashTrainConfig(**(config.to_dict() | {"context_mode": "text_only"}))
    with pytest.raises(ValueError, match="context_mode"):
        load_draft_checkpoint(tmp_path, adapter, mismatched)
    metadata_path = tmp_path / "dflash_config.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("implementation_version")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="implementation_version"):
        load_draft_checkpoint(tmp_path, adapter, config)


def test_tiny_optimizer_step_exports_best_and_resumable_checkpoint(tmp_path):
    transformers = __import__("transformers")
    if not hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
        pytest.skip("Qwen2.5-VL is unavailable in this Transformers build")
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    target = Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(
            text_config={
                "vocab_size": 100,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "rope_theta": 1_000_000,
                "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 0]},
                "pad_token_id": 0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 16,
                "spatial_merge_size": 1,
                "patch_size": 2,
                "temporal_patch_size": 1,
            },
            image_token_id=98,
            video_token_id=99,
        )
    )

    class Tokenizer:
        all_special_ids = [0]
        name_or_path = "tiny-qwen25vl"

        def __len__(self):
            return 90

        def get_vocab(self):
            return {str(index): index for index in range(90)}

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[3, 4, 5]])}

    adapter = Qwen25VLTargetAdapter(target, Processor(), device=torch.device("cpu"))
    adapter.requested_model = "tiny-qwen25vl"
    adapter.requested_revision = None
    config = DFlashTrainConfig(
        target_model="tiny-qwen25vl",
        block_size=4,
        num_anchors=2,
        anchor_chunk_size=2,
        min_anchor_chunk_size=1,
        num_draft_layers=2,
        num_target_features=5,
        mixed_precision="no",
        use_flex_attention=False,
        epochs=1,
        gradient_accumulation_steps=1,
        output_dir=str(tmp_path),
        validation_manifest="provided-directly",
    )
    record = {
        "id": "train-one",
        "messages": [{"role": "user", "content": "x"}],
        "target_response": {
            "token_ids": [6, 7, 8, 9],
            "text": "x",
            "generation": {"do_sample": False},
        },
        "provenance": adapter.target_provenance(),
    }
    draft = make_draft_model(adapter, config)
    history = train_records(
        adapter,
        draft,
        [record],
        config,
        validation_records=[record],
    )
    assert history[0]["usable_records"] == 1
    assert "validation_accepted_prefix" in history[0]
    assert all(not parameter.requires_grad for parameter in adapter.model.parameters())
    assert any(parameter.grad is not None for parameter in draft.parameters())
    assert (tmp_path / "best" / "model.safetensors").exists()
    checkpoint_dirs = list((tmp_path / "checkpoints").iterdir())
    assert len(checkpoint_dirs) == 1
    assert (checkpoint_dirs[0] / "trainer_state.pt").exists()
    resumed_config = DFlashTrainConfig(
        **(config.to_dict() | {"epochs": 2, "resume_from_checkpoint": str(checkpoint_dirs[0])})
    )
    resumed_history = train_records(
        adapter,
        make_draft_model(adapter, resumed_config),
        [record],
        resumed_config,
        validation_records=[record],
    )
    assert [item["epoch"] for item in resumed_history] == [1.0, 2.0]
