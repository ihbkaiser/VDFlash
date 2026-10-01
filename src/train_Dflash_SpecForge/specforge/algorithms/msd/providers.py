"""First-class SpecForge registration for original multimodal MSD."""

from __future__ import annotations

from specforge.algorithms.common.defaults import (
    no_missing_checkpoint_keys,
    one_loss_token,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    StepProvider,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.algorithms.msd.curriculum import MSD_CURRICULUM_VERSION
from specforge.algorithms.msd.data import (
    NORMALIZER_ID,
    RAW_FEATURE_KEYS,
    build_offline_collator,
    build_offline_normalizer,
    build_offline_reader,
)

ALGORITHM_NAME = "msd"
DRAFT_ARCHITECTURE = "MSDDraftModel"


def build_step(wrapped_model, *, target_head=None, **options):
    from specforge.training.strategies.base import MSDTrainStrategy

    return MSDTrainStrategy(wrapped_model, target_head=target_head, **options)


def step_options(config):
    return {
        "feature_loss_weight": config.training.msd_feature_loss_weight,
        "soft_loss_weight": config.training.msd_soft_loss_weight,
        "noise_width": config.training.msd_noise_width,
    }


def resume_contract(config, draft_model, _training_model):
    return {
        "msd_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "msd_feature_loss_weight": float(config.training.msd_feature_loss_weight),
        "msd_soft_loss_weight": float(config.training.msd_soft_loss_weight),
        "msd_noise_width": float(config.training.msd_noise_width),
        "msd_total_epochs": int(config.training.msd_total_epochs),
        "msd_curriculum_seed": int(config.training.msd_curriculum_seed),
        "msd_curriculum_version": MSD_CURRICULUM_VERSION,
        "msd_target_hidden_size": int(draft_model.config.hidden_size),
        "msd_target_vocab_size": int(draft_model.config.vocab_size),
    }


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_msd_model

    return build_msd_model(
        config, draft_model, draft_config, target_config, tokenizer
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_msd_capture_layers

    return resolve_msd_capture_layers(config, draft_config, target_config)


def apply_draft_overrides(config, draft_config):
    from specforge.algorithms.model_providers import apply_msd_overrides

    return apply_msd_overrides(config, draft_config)


def needs_input_tools(_config, _draft_model):
    return False


def algorithm_spec() -> AlgorithmSpec:
    normalized = {
        "input_ids",
        "loss_mask",
        "target_hidden_state",
        "conditioning_hidden_state",
        "next_token_embeddings",
        "visual_embeddings",
        "visual_token_mask",
        "position_ids",
        "attention_mask",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"num_hidden_layers"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="multimodal",
                required_tensors=normalized,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_msd_qwen25vl_v1",
                    required_tensors=set(RAW_FEATURE_KEYS),
                    normalizer=NORMALIZER_ID,
                ),
            ),
        ),
        capabilities=AlgorithmCapabilities(attention_backends={"sdpa"}),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=step_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=True,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                apply_overrides=apply_draft_overrides,
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=one_loss_token,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=4,
            allow_missing_warm_start_embedding=True,
        ),
        offline=(
            OfflineDataProvider(
                modality="multimodal",
                normalizer_id=NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="msd",
                    aux_feature=None,
                    last_hidden_feature="target_hidden_state",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                        ("input_embeddings", "input_embeddings"),
                        ("visual_token_mask", "visual_token_mask"),
                        ("position_ids", "position_ids"),
                    ),
                ),
                build_reader=build_offline_reader,
                build_normalizer=build_offline_normalizer,
                build_collator=build_offline_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
