import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "train_EAGLE3_SpecForge"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def test_phase1_copy_contains_the_specforge_eagle3_runtime():
    package = ROOT / "src" / "train_EAGLE3_SpecForge"
    required = (
        package / "specforge" / "cli.py",
        package / "specforge" / "algorithms" / "eagle3" / "model.py",
        package / "specforge" / "algorithms" / "eagle3" / "providers.py",
        package / "specforge" / "training" / "controller.py",
        package / "specforge" / "training" / "backend.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    assert not missing, f"missing copied SpecForge runtime files: {missing}"


def test_phase1_runtime_registers_only_eagle3_algorithm_and_draft():
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.modeling.draft.registry import available_drafts

    assert builtin_algorithm_registry().names == ("eagle3",)
    assert available_drafts() == ["LlamaForCausalLMEagle3"]
