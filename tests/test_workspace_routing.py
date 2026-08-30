from pathlib import Path

from src.workspace import (
    find_workspace_root,
    resolve_workspace_path,
    workspace_path,
    workspace_python,
)


def test_workspace_root_is_derived_from_repository_filesystem():
    root = find_workspace_root(Path(__file__))

    assert root == Path(__file__).resolve().parents[1]
    assert workspace_path("src", "infer").is_dir()


def test_relative_paths_and_python_are_rooted_at_the_workspace():
    root = Path(__file__).resolve().parents[1]

    assert resolve_workspace_path("results/infer") == root / "results/infer"
    assert resolve_workspace_path(root / "results/infer") == root / "results/infer"
    assert workspace_python(root) == root / ".venv/bin/python"


def test_baseline_shell_launchers_use_one_workspace_resolver():
    root = Path(__file__).resolve().parents[1]
    launchers = (
        root / "scripts/run_mvbench100_ablation.sh",
        root / "src/infer/run_qwen25vl_3b_dflash_vdc_compare.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_dflash_validation_gpu.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_msd_2gpu.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_msd_memory_aware_2gpu.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_msd_model_parallel_2gpu.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_dflash_model_parallel_2gpu.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_sparrow_stages_t4.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_sparrow_validation_gpu.sh",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_tmux_full_validation.sh",
        root / "src/analyze/Whether_they_are_appliable_for_dDrafter/run_qwen35_dflash_t4.sh",
        root / "externals/Sparrow/run_videodetailcaption_baseline.sh",
        root / "externals/ViSpec/baseline.sh",
    )

    for launcher in launchers:
        contents = launcher.read_text(encoding="utf-8")
        assert ".venv-msd" not in contents
        assert "resolve_workspace.sh" in contents


def test_inference_cli_defaults_are_absolute_workspace_paths():
    from src.infer.qwen25vl_dflash_compare import build_parser

    root = Path(__file__).resolve().parents[1]
    args = build_parser().parse_args([])

    assert args.manifest == root / "dataset/VideoDetailCaption/test.jsonl"
    assert args.video_root == root / "dataset/VideoDetailCaption"
    assert args.output == root / "results/infer/qwen25vl_3b_dflash_vdc_sample0.json"
    assert args.output_dir == root / "results/infer/qwen25vl_3b_dflash_vdc50"


def test_sparrow_python_runners_normalize_paths_before_loading_data():
    root = Path(__file__).resolve().parents[1]
    runners = (
        root / "src/analyze/Validate_Sparrow_hypothesises/run_msd.py",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_attention.py",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_draft_attention.py",
        root / "src/analyze/Validate_Sparrow_hypothesises/run_layer_analysis.py",
    )

    for runner in runners:
        contents = runner.read_text(encoding="utf-8")
        assert "resolve_namespace_paths" in contents


def test_report_defaults_are_absolute_workspace_paths():
    root = Path(__file__).resolve().parents[1]

    from src.infer.build_dflash_infer_report import default_run_specs
    from src.infer.build_dflash_epoch_comparison_report import default_epoch_specs

    assert default_run_specs()[0].path == root / "results/infer/mvbench100_full_20260823"
    assert default_epoch_specs()[1].vdc50_dir == (
        root / "results/infer/dflash20e_20260825/vdc50_exp_full_dflash_20260823"
    )
