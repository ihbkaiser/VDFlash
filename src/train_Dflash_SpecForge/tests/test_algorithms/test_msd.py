from __future__ import annotations

import os
import tempfile

import pytest
import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.msd.curriculum import (
    choose_visual_sample,
    visual_ratio_for_epoch,
)
from specforge.algorithms.msd.data import (
    build_offline_collator,
    build_paired_offline_reader,
    normalize_offline_sample,
)
from specforge.algorithms.msd.model import (
    add_reference_uniform_noise,
    decouple_msd_inputs,
    msd_loss,
)
from specforge.config.schema import TrainingConfig
from specforge.modeling.draft.msd import MSDDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import MSDTrainStrategy


@pytest.mark.parametrize(
    ("epoch_now", "expected"),
    [
        (1, 0.0),
        (20, 0.0),
        (21, 0.05),
        (22, 0.10),
        (39, 0.95),
        (40, 1.0),
    ],
)
def test_msd_40_epoch_ratio_matches_reference_boundaries(
    epoch_now: int,
    expected: float,
) -> None:
    assert visual_ratio_for_epoch(epoch_now, 40) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("epoch_now", "total_epoch"),
    [(0, 40), (41, 40), (1, 0), (1, 39)],
)
def test_msd_ratio_rejects_invalid_or_noncanonical_epochs(
    epoch_now: int,
    total_epoch: int,
) -> None:
    with pytest.raises(ValueError):
        visual_ratio_for_epoch(epoch_now, total_epoch)


def test_msd_choice_is_stateless_and_resume_stable() -> None:
    first = [choose_visual_sample(0, 27, index, 40) for index in range(100)]
    resumed = [
        choose_visual_sample(0, 27, index, 40) for index in range(50, 100)
    ]

    assert first[50:] == resumed
    assert any(first)
    assert not all(first)


def test_msd_choice_obeys_pure_text_and_pure_visual_epochs() -> None:
    assert not any(choose_visual_sample(0, 20, index, 40) for index in range(64))
    assert all(choose_visual_sample(0, 40, index, 40) for index in range(64))


def test_msd_normalizer_reproduces_next_token_shift() -> None:
    raw = {
        "input_ids": torch.tensor([10, 11, 12, 13]),
        "loss_mask": torch.tensor([0, 1, 1, 1]),
        "target_hidden_state": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        ),
        "input_embeddings": torch.tensor(
            [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]]]
        ),
        "visual_token_mask": torch.tensor([False, True, True, False]),
        "position_ids": torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 4, 5, 3],
                [0, 6, 7, 3],
            ]
        ),
    }

    normalized = normalize_offline_sample(raw, max_len=4)

    assert normalized["input_ids"].tolist() == [[11, 12, 13, 0]]
    assert normalized["loss_mask"].tolist() == [[0, 1, 1, 0]]
    assert normalized["target_hidden_state"].tolist() == [
        [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [0.0, 0.0]]
    ]
    assert normalized["next_token_embeddings"].tolist() == [
        [[30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [0.0, 0.0]]
    ]
    assert torch.equal(
        normalized["conditioning_hidden_state"], raw["target_hidden_state"]
    )
    assert normalized["visual_embeddings"].tolist() == [
        [[30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [0.0, 0.0]]
    ]
    assert normalized["visual_token_mask"].tolist() == [[True, True, False, False]]
    assert tuple(normalized["position_ids"].shape) == (3, 1, 4)


def test_msd_normalizer_rejects_mismatched_sequence_lengths() -> None:
    raw = {
        "input_ids": torch.tensor([1, 2, 3]),
        "loss_mask": torch.ones(3),
        "target_hidden_state": torch.ones(1, 3, 2),
        "input_embeddings": torch.ones(1, 2, 2),
        "visual_token_mask": torch.zeros(3, dtype=torch.bool),
        "position_ids": torch.arange(3).repeat(3, 1),
    }

    with pytest.raises(ValueError, match="sequence lengths"):
        normalize_offline_sample(raw, max_len=3)


def test_decouple_uses_projection_for_text_and_raw_embedding_for_visual() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    next_embeddings = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
    )
    visual_embeddings = torch.tensor([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    visual_mask = torch.tensor([[False, True, False]])
    projection = torch.nn.Linear(4, 2, bias=False)
    projection.weight.data.copy_(
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    )

    actual = decouple_msd_inputs(
        hidden,
        next_embeddings,
        visual_embeddings,
        visual_mask,
        projection,
    )

    assert torch.equal(actual[:, 0], torch.tensor([[11.0, 22.0]]))
    assert torch.equal(actual[:, 1], visual_embeddings[:, 1])
    assert torch.equal(actual[:, 2], torch.tensor([[55.0, 66.0]]))


def test_reference_uniform_noise_is_seeded_and_scaled_by_sequence_length() -> None:
    hidden = torch.zeros(1, 4, 2)
    first = add_reference_uniform_noise(
        hidden,
        width=0.2,
        generator=torch.Generator().manual_seed(7),
    )
    second = add_reference_uniform_noise(
        hidden,
        width=0.2,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(first, second)
    assert first.abs().max().item() <= 12.8 + 1e-5


def test_msd_loss_matches_literal_reference_and_masks_padding() -> None:
    predicted = torch.tensor([[[1.0, 2.0], [2.0, 3.0]]], requires_grad=True)
    target = torch.tensor([[[1.5, 1.5], [99.0, 99.0]]])
    mask = torch.tensor([[1.0, 0.0]])
    head = torch.nn.Linear(2, 3, bias=False)
    head.weight.data.copy_(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    )
    for parameter in head.parameters():
        parameter.requires_grad = False

    result = msd_loss(predicted, target, mask, head, 1.0, 0.1)

    denominator = mask.sum() + 1e-5
    feature_per_token = torch.nn.functional.smooth_l1_loss(
        predicted, target, reduction="none"
    ).mean(-1)
    expected_feature = (feature_per_token * mask).sum() / denominator
    target_probs = head(target[:, :1]).softmax(-1)
    soft_per_token = -(
        target_probs * head(predicted[:, :1]).log_softmax(-1)
    ).sum(-1)
    expected_soft = soft_per_token.sum() / denominator
    assert torch.allclose(result.feature_loss, expected_feature)
    assert torch.allclose(result.soft_target_loss, expected_soft)
    assert torch.allclose(result.loss, expected_feature + 0.1 * expected_soft)

    result.loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert head.weight.grad is None


def test_msd_loss_rejects_an_empty_loss_mask() -> None:
    head = torch.nn.Linear(2, 3, bias=False)
    with pytest.raises(ValueError, match="at least one valid token"):
        msd_loss(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2),
            head,
        )


def test_msd_is_a_first_class_multimodal_specforge_registration() -> None:
    registration = builtin_algorithm_registry().resolve("msd")

    assert registration.spec.draft.default_architecture == "MSDDraftModel"
    assert registration.spec.capabilities.attention_backends == {"sdpa"}
    assert registration.spec.modalities == {"multimodal"}
    assert registration.spec.draft.fixed_override_values == ()
    assert registration.providers.model.draft_config.architecture == "MSDDraftModel"


def test_msd_training_fields_enforce_reference_schedule_and_nonnegative_terms() -> None:
    parsed = TrainingConfig(
        strategy="msd",
        num_epochs=40,
        msd_feature_loss_weight=1.0,
        msd_soft_loss_weight=0.1,
        msd_noise_width=0.2,
        msd_total_epochs=40,
    )
    assert parsed.msd_total_epochs == 40

    for field, invalid in (
        ("msd_feature_loss_weight", -1.0),
        ("msd_soft_loss_weight", -0.1),
        ("msd_noise_width", -0.01),
        ("msd_total_epochs", 39),
    ):
        values = {"strategy": "msd", "num_epochs": 40, field: invalid}
        with pytest.raises(ValueError):
            TrainingConfig(**values)


def test_msd_train_strategy_returns_finite_named_objective_terms() -> None:
    from specforge.modeling.draft.msd import MSDConfig

    draft = MSDDraftModel(
        MSDConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            mrope_section=[1, 1, 0],
        )
    )
    head = torch.nn.Linear(8, 16, bias=False)
    head.requires_grad_(False)
    strategy = MSDTrainStrategy(
        draft,
        target_head=head,
        feature_loss_weight=1.0,
        soft_loss_weight=0.1,
        noise_width=0.0,
    )
    batch, length, width = 1, 4, 8
    tensors = {
        "input_ids": torch.arange(length).unsqueeze(0),
        "loss_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "target_hidden_state": torch.randn(batch, length, width),
        "conditioning_hidden_state": torch.randn(batch, length, width),
        "next_token_embeddings": torch.randn(batch, length, width),
        "visual_embeddings": torch.randn(batch, length, width),
        "visual_token_mask": torch.tensor([[False, True, True, False]]),
        "position_ids": torch.arange(length).repeat(3, batch, 1),
        "attention_mask": torch.ones(batch, length, dtype=torch.bool),
    }

    output = strategy.forward_loss(
        TrainBatch(sample_ids=["tiny"], strategy="msd", tensors=tensors)
    )

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert {"feature_loss", "soft_target_loss", "accuracy"} <= output.metrics.keys()
    output.loss.backward()
    assert draft.layers[0].self_attn.q_proj.weight.grad is not None
    assert head.weight.grad is None
    assert draft.embed_tokens.weight.grad is None


def _write_msd_record(root: str, index: int, length: int = 3) -> None:
    torch.save(
        {
            "input_ids": torch.arange(length),
            "loss_mask": torch.ones(length),
            "target_hidden_state": torch.ones(1, length, 2),
            "input_embeddings": torch.ones(1, length, 2),
            "visual_token_mask": torch.zeros(length, dtype=torch.bool),
            "position_ids": torch.arange(length).repeat(3, 1),
        },
        os.path.join(root, f"{index:04d}.ckpt"),
    )


def test_msd_paired_reader_applies_epoch_curriculum_without_mutable_rng() -> None:
    with tempfile.TemporaryDirectory() as text_root, tempfile.TemporaryDirectory() as visual_root:
        for index in range(8):
            _write_msd_record(text_root, index)
            _write_msd_record(visual_root, index)

        text_refs = build_paired_offline_reader(
            text_root,
            visual_root,
            run_id="msd-reader",
            ttt_length=1,
            max_len=8,
            epoch_now=20,
            curriculum_seed=0,
        ).read()
        visual_refs = build_paired_offline_reader(
            text_root,
            visual_root,
            run_id="msd-reader",
            ttt_length=1,
            max_len=8,
            epoch_now=40,
            curriculum_seed=0,
        ).read()
        first = build_paired_offline_reader(
            text_root,
            visual_root,
            run_id="msd-reader",
            ttt_length=1,
            max_len=8,
            epoch_now=27,
            curriculum_seed=11,
        ).read()
        resumed = build_paired_offline_reader(
            text_root,
            visual_root,
            run_id="msd-reader",
            ttt_length=1,
            max_len=8,
            epoch_now=27,
            curriculum_seed=11,
        ).read()[4:]

    assert {ref.metadata["msd_corpus"] for ref in text_refs} == {"text"}
    assert {ref.metadata["msd_corpus"] for ref in visual_refs} == {"visual"}
    assert [ref.feature_store_uri for ref in first[4:]] == [
        ref.feature_store_uri for ref in resumed
    ]
    assert all(set(ref.feature_keys) == {
        "input_ids",
        "loss_mask",
        "target_hidden_state",
        "input_embeddings",
        "visual_token_mask",
        "position_ids",
    } for ref in first)


def test_msd_paired_reader_requires_matched_nonempty_feature_cohorts() -> None:
    with tempfile.TemporaryDirectory() as text_root, tempfile.TemporaryDirectory() as visual_root:
        _write_msd_record(text_root, 0)
        with pytest.raises(ValueError, match="same non-zero number"):
            build_paired_offline_reader(
                text_root,
                visual_root,
                run_id="msd-reader",
                ttt_length=1,
                max_len=8,
                epoch_now=21,
            )


def test_msd_collator_pads_hidden_and_three_axis_position_tensors() -> None:
    first = normalize_offline_sample(
        {
            "input_ids": torch.arange(2),
            "loss_mask": torch.ones(2),
            "target_hidden_state": torch.ones(1, 2, 3),
            "input_embeddings": torch.ones(1, 2, 3),
            "visual_token_mask": torch.zeros(2, dtype=torch.bool),
            "position_ids": torch.arange(2).repeat(3, 1),
        },
        4,
    )
    second = normalize_offline_sample(
        {
            "input_ids": torch.arange(4),
            "loss_mask": torch.ones(4),
            "target_hidden_state": torch.ones(1, 4, 3),
            "input_embeddings": torch.ones(1, 4, 3),
            "visual_token_mask": torch.zeros(4, dtype=torch.bool),
            "position_ids": torch.arange(4).repeat(3, 1),
        },
        4,
    )

    batch = build_offline_collator()([first, second])

    assert batch["target_hidden_state"].shape == (2, 4, 3)
    assert batch["position_ids"].shape == (3, 2, 4)
    assert batch["loss_mask"][0].tolist() == [1.0, 0.0, 0.0, 0.0]
