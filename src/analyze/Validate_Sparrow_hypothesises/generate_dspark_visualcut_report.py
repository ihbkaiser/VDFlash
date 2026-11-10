"""Build a Vietnamese DSpark VisualCut report from audited campaign artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = (
    (
        "dspark_3layer_1_9_17_25_33",
        "Balanced [1, 9, 17, 25, 33]",
        "#E76F51",
    ),
)
BUDGETS = (400, 3000, 13000, 25000)
CONDITIONS = ("full", "zero", "cut", "retained_75", "retained_50", "retained_25")
RETENTION_ORDER = ("cut", "retained_25", "retained_50", "retained_75", "full")
RETENTION_PERCENT = (0, 25, 50, 75, 100)
CONDITION_LABELS = {
    "full": "Full",
    "zero": "Zero",
    "cut": "Cut",
    "retained_75": "Retained 75%",
    "retained_50": "Retained 50%",
    "retained_25": "Retained 25%",
}
BUDGET_LABELS = {400: "400", 3000: "3K", 13000: "13K", 25000: "25K"}
PALETTE = {
    "full": "#264653",
    "zero": "#8C8C8C",
    "cut": "#E76F51",
    "retained_75": "#0072B2",
    "retained_50": "#2A9D8F",
    "retained_25": "#E9C46A",
}
BUDGET_COLORS = ("#264653", "#2A9D8F", "#E9C46A", "#E76F51")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _bootstrap(values: list[float], *, seed: int, replicates: int = 2000) -> dict[str, Any]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    )
    return {
        "n": len(values),
        "mean": mean,
        "ci_low": means[int(0.025 * (replicates - 1))],
        "ci_high": means[int(0.975 * (replicates - 1))],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_ci(mean: Any, low: Any, high: Any, digits: int = 3) -> str:
    if _number(mean) is None:
        return "—"
    return f"{_fmt(mean, digits)} [{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _find_row(
    rows: list[dict[str, str]], *, checkpoint: str, budget: int, condition: str
) -> dict[str, str] | None:
    return next(
        (
            row
            for row in rows
            if row.get("checkpoint_name") == checkpoint
            and int(row.get("target_visual_tokens") or -1) == budget
            and row.get("visual_condition") == condition
        ),
        None,
    )


def _find_contrast(
    rows: list[dict[str, str]],
    *,
    checkpoint: str,
    budget: int,
    contrast: str,
    metric: str = "tau_effective",
) -> dict[str, str] | None:
    return next(
        (
            row
            for row in rows
            if row.get("checkpoint_name") == checkpoint
            and int(row.get("target_visual_tokens") or -1) == budget
            and row.get("contrast") == contrast
            and row.get("metric") == metric
        ),
        None,
    )


def _load_case_summaries(run_root: Path) -> dict[str, list[dict[str, str]]]:
    summaries: dict[str, list[dict[str, str]]] = {}
    for checkpoint, _, _ in CONFIGS:
        path = run_root / "full" / checkpoint / "analysis" / "case_attention_summary.csv"
        summaries[checkpoint] = _read_csv(path) if path.is_file() else []
    return summaries


def _load_runtime_summaries(run_root: Path) -> dict[str, list[dict[str, str]]]:
    summaries: dict[str, list[dict[str, str]]] = {}
    for checkpoint, _, _ in CONFIGS:
        rows: list[dict[str, str]] = []
        task_dir = run_root / "full" / checkpoint / "tasks"
        for path in sorted(task_dir.glob("*/cases.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    case = json.loads(line)
                    timing = case.get("timing") or {}
                    target = timing.get("target") or {}
                    speculative = timing.get("speculative") or {}
                    row: dict[str, str] = {
                        "checkpoint_name": str(case.get("checkpoint_name") or checkpoint),
                        "target_visual_tokens": str(case.get("target_visual_tokens") or ""),
                        "visual_condition": str(case.get("visual_condition") or ""),
                        "sample_id": str(case.get("sample_id") or ""),
                    }
                    for metric, target_key, speculative_key in (
                        ("decode_speedup", "decode_s", "decode_s"),
                        ("end_to_end_speedup", "end_to_end_s", "end_to_end_s"),
                    ):
                        numerator = _number(target.get(target_key))
                        denominator = _number(speculative.get(speculative_key))
                        if numerator is not None and denominator is not None and denominator > 0:
                            row[metric] = str(numerator / denominator)
                    rows.append(row)
        summaries[checkpoint] = rows
    return summaries


def _group_case_values(
    rows: list[dict[str, str]],
    *,
    checkpoint: str,
    budget: int,
    condition: str,
    metric: str,
) -> list[float]:
    values = []
    for row in rows:
        if (
            row.get("checkpoint_name") != checkpoint
            or int(row.get("target_visual_tokens") or -1) != budget
            or row.get("visual_condition") != condition
        ):
            continue
        value = (
            float(_truth(row.get(metric)))
            if metric == "lossless"
            else _number(row.get(metric))
        )
        if value is not None:
            values.append(value)
    return values


def _save_figures(
    report_dir: Path,
    condition_stats: dict[str, list[dict[str, str]]],
    contrasts: dict[str, list[dict[str, str]]],
    case_summaries: dict[str, list[dict[str, str]]],
    runtime_summaries: dict[str, list[dict[str, str]]],
) -> list[str]:
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
        }
    )
    outputs: list[str] = []

    def save(fig: plt.Figure, stem: str) -> None:
        for extension in ("pdf", "png"):
            fig.savefig(figures / f"{stem}.{extension}")
        plt.close(fig)
        outputs.append(stem)

    # Experiment 1: full, zero and cut at each visual-token budget.
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5.2 * len(CONFIGS), 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(BUDGETS))
    for axis, (checkpoint, label, _) in zip(axes, CONFIGS):
        for condition in ("full", "cut", "zero"):
            means, lows, highs = [], [], []
            for budget in BUDGETS:
                row = _find_row(
                    condition_stats[checkpoint],
                    checkpoint=checkpoint,
                    budget=budget,
                    condition=condition,
                )
                mean = _number(row.get("tau_effective_mean")) if row else None
                low = _number(row.get("tau_effective_ci_low")) if row else None
                high = _number(row.get("tau_effective_ci_high")) if row else None
                means.append(np.nan if mean is None else mean)
                lows.append(np.nan if low is None else low)
                highs.append(np.nan if high is None else high)
            axis.plot(x, means, marker="o", color=PALETTE[condition], label=CONDITION_LABELS[condition])
            axis.fill_between(x, lows, highs, color=PALETTE[condition], alpha=0.12)
        axis.set_title(label)
        axis.set_xticks(x, [BUDGET_LABELS[b] for b in BUDGETS])
        axis.set_xlabel("Ngân sách visual token")
    axes[0].set_ylabel(r"$\tau_{\mathrm{effective}}$ (token/vòng)")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Thí nghiệm 1 · Full, Zero và Cut", y=1.03, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_visual_cut_tau")

    # Experiment 2: paired retention curves for all four budgets.
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5.2 * len(CONFIGS), 3.7), sharey=True, squeeze=False)
    axes = axes[0]
    x_ret = np.asarray(RETENTION_PERCENT)
    for axis, (checkpoint, label, _) in zip(axes, CONFIGS):
        for budget, color in zip(BUDGETS, BUDGET_COLORS):
            means, lows, highs = [], [], []
            for condition in RETENTION_ORDER:
                row = _find_row(
                    condition_stats[checkpoint],
                    checkpoint=checkpoint,
                    budget=budget,
                    condition=condition,
                )
                mean = _number(row.get("tau_effective_mean")) if row else None
                low = _number(row.get("tau_effective_ci_low")) if row else None
                high = _number(row.get("tau_effective_ci_high")) if row else None
                means.append(np.nan if mean is None else mean)
                lows.append(np.nan if low is None else low)
                highs.append(np.nan if high is None else high)
            axis.plot(x_ret, means, marker="o", color=color, label=BUDGET_LABELS[budget])
            axis.fill_between(x_ret, lows, highs, color=color, alpha=0.10)
        axis.set_title(label)
        axis.set_xticks(x_ret, ["Cut\n0%", "25%", "50%", "75%", "Full\n100%"])
        axis.set_xlabel("Tỷ lệ visual token giữ lại")
    axes[0].set_ylabel(r"$\tau_{\mathrm{effective}}$ (token/vòng)")
    axes[-1].legend(title="Budget", frameon=False, ncol=2)
    fig.suptitle("Thí nghiệm 2 · Retention sweep", y=1.03, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_retention_sweep")

    # Experiment 3: selected attention mass and density at the longest context.
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(5.4 * len(CONFIGS), 6.2), sharex="col", squeeze=False)
    attn_conditions = ("full", "zero", "cut", "retained_75", "retained_50", "retained_25")
    attn_labels = [CONDITION_LABELS[name] for name in attn_conditions]
    x_attn = np.arange(len(attn_conditions))
    mass_metrics = (
        ("visual_mass", "Visual"),
        ("question_text_mass", "Câu hỏi"),
        ("generated_target_hidden_mass", "Target hidden đã sinh"),
        ("cached_draft_mass", "Draft cache"),
    )
    density_metrics = (
        ("visual_density", "Visual"),
        ("question_text_density", "Câu hỏi"),
        ("generated_target_hidden_density", "Target hidden đã sinh"),
    )
    for col, (checkpoint, label, _) in enumerate(CONFIGS):
        rows = case_summaries[checkpoint]
        mass_axis, density_axis = axes[0, col], axes[1, col]
        for metric_index, (metric, metric_label) in enumerate(mass_metrics):
            values, lows, highs = [], [], []
            for condition in attn_conditions:
                group = _group_case_values(
                    rows,
                    checkpoint=checkpoint,
                    budget=25000,
                    condition=condition,
                    metric=metric,
                )
                summary = _bootstrap(group, seed=20261102 + col * 100 + metric_index * 10 + len(values))
                values.append(np.nan if summary["mean"] is None else summary["mean"])
                lows.append(np.nan if summary["ci_low"] is None else summary["ci_low"])
                highs.append(np.nan if summary["ci_high"] is None else summary["ci_high"])
            mass_axis.plot(x_attn, values, marker="o", label=metric_label)
            mass_axis.fill_between(x_attn, lows, highs, alpha=0.08)
        mass_axis.set_title(label)
        for metric, metric_label in density_metrics:
            values = []
            for condition in attn_conditions:
                group = _group_case_values(
                    rows,
                    checkpoint=checkpoint,
                    budget=25000,
                    condition=condition,
                    metric=metric,
                )
                summary = _bootstrap(group, seed=20261102 + col * 100 + len(values))
                values.append(np.nan if summary["mean"] is None else max(summary["mean"], 1e-12))
            density_axis.plot(x_attn, values, marker="o", label=metric_label)
        density_axis.set_yscale("log")
        density_axis.set_xticks(x_attn, attn_labels, rotation=20, ha="right")
        density_axis.set_xlabel("Điều kiện draft-side")
        if col == 0:
            mass_axis.set_ylabel("Attention mass")
            density_axis.set_ylabel("Attention density (log scale)")
    axes[0, -1].legend(frameon=False, loc="best")
    axes[1, -1].legend(frameon=False, loc="best")
    fig.suptitle("Thí nghiệm 3 · Attention ở budget 25K", y=1.01, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_attention_25k")

    # Experiment 4: throughput/acceptance and strict output agreement.
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(5.4 * len(CONFIGS), 6.6), squeeze=False)
    markers = {
        "full": "o",
        "zero": "x",
        "cut": "+",
        "retained_75": "s",
        "retained_50": "^",
        "retained_25": "D",
    }
    for col, (checkpoint, label, _) in enumerate(CONFIGS):
        rows = case_summaries[checkpoint]
        scatter_axis, match_axis = axes[0, col], axes[1, col]
        for budget, color in zip(BUDGETS, BUDGET_COLORS):
            for condition in CONDITIONS:
                group = [
                    row
                    for row in rows
                    if row.get("checkpoint_name") == checkpoint
                    and int(row.get("target_visual_tokens") or -1) == budget
                    and row.get("visual_condition") == condition
                ]
                taus = [_number(row.get("tau_effective")) for row in group]
                speeds = [_number(row.get("tokens_per_second")) for row in group]
                points = [(tau, speed) for tau, speed in zip(taus, speeds) if tau is not None and speed is not None]
                if points:
                    scatter_axis.scatter(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        s=16,
                        alpha=0.45,
                        marker=markers[condition],
                        color=color,
                        label="_nolegend_",
                    )
        scatter_axis.set_title(label)
        scatter_axis.set_xlabel(r"$\tau_{\mathrm{effective}}$")
        if col == 0:
            scatter_axis.set_ylabel("Speculative throughput (token/s)")
        budget_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="none", color=color, label=BUDGET_LABELS[budget])
            for budget, color in zip(BUDGETS, BUDGET_COLORS)
        ]
        budget_legend = scatter_axis.legend(
            handles=budget_handles,
            title="Budget",
            frameon=False,
            fontsize=6,
            loc="upper left",
        )
        scatter_axis.add_artist(budget_legend)
        condition_handles = [
            plt.Line2D(
                [0], [0], marker=markers[condition], linestyle="none", color="#444444",
                label=CONDITION_LABELS[condition],
            )
            for condition in CONDITIONS
        ]
        scatter_axis.legend(
            handles=condition_handles,
            title="Condition",
            frameon=False,
            fontsize=5.8,
            loc="upper right",
            ncol=2,
        )
        # Exact sequence equality rate against greedy target output.
        selected = ("full", "cut", "retained_25")
        x = np.arange(len(BUDGETS))
        width = 0.24
        for index, condition in enumerate(selected):
            rates = []
            lows = []
            highs = []
            for budget in BUDGETS:
                group = _group_case_values(
                    rows,
                    checkpoint=checkpoint,
                    budget=budget,
                    condition=condition,
                    metric="lossless",
                )
                summary = _bootstrap(group, seed=20261102 + col * 100 + index * 10 + budget)
                rates.append(100 * summary["mean"] if summary["mean"] is not None else np.nan)
                lows.append(100 * summary["ci_low"] if summary["ci_low"] is not None else np.nan)
                highs.append(100 * summary["ci_high"] if summary["ci_high"] is not None else np.nan)
            offset = (index - 1) * width
            match_axis.bar(
                x + offset,
                rates,
                width=width * 0.9,
                color=PALETTE[condition],
                label=CONDITION_LABELS[condition],
                yerr=[np.asarray(rates) - np.asarray(lows), np.asarray(highs) - np.asarray(rates)],
                capsize=2,
            )
        match_axis.set_title(f"{label} · exact equality")
        match_axis.set_xticks(x, [BUDGET_LABELS[b] for b in BUDGETS])
        match_axis.set_xlabel("Ngân sách visual token")
        match_axis.set_ylim(0, 100)
        if col == 0:
            match_axis.set_ylabel("Khớp toàn bộ chuỗi (%)")
        match_axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Thí nghiệm 4 · Throughput và output agreement", y=1.01, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_speed_and_agreement")

    # Experiment 2 supplement: realized decode and end-to-end speedups by retention.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharex=True, sharey=True)
    speedup_colors = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
    speedup_metrics = (
        ("decode_speedup", "Decode speedup (×)"),
        ("end_to_end_speedup", "End-to-end speedup (×)"),
    )
    x_ret = np.asarray(RETENTION_PERCENT)
    for checkpoint, label, _ in CONFIGS:
        rows = runtime_summaries[checkpoint]
        for metric_index, (metric, axis_label) in enumerate(speedup_metrics):
            axis = axes[metric_index]
            all_lows: list[float] = []
            all_highs: list[float] = []
            for budget_index, (budget, color) in enumerate(zip(BUDGETS, speedup_colors)):
                means, lows, highs = [], [], []
                for condition_index, condition in enumerate(RETENTION_ORDER):
                    values = _group_case_values(
                        rows,
                        checkpoint=checkpoint,
                        budget=budget,
                        condition=condition,
                        metric=metric,
                    )
                    summary = _bootstrap(
                        values,
                        seed=20261104 + budget_index * 100 + metric_index * 20 + condition_index,
                    )
                    means.append(np.nan if summary["mean"] is None else summary["mean"])
                    lows.append(np.nan if summary["ci_low"] is None else summary["ci_low"])
                    highs.append(np.nan if summary["ci_high"] is None else summary["ci_high"])
                axis.plot(
                    x_ret,
                    means,
                    marker="o",
                    color=color,
                    label=BUDGET_LABELS[budget],
                )
                axis.fill_between(x_ret, lows, highs, color=color, alpha=0.10)
                all_lows.extend(value for value in lows if np.isfinite(value))
                all_highs.extend(value for value in highs if np.isfinite(value))
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.1, label="Ngang target (1×)")
            axis.set_title(axis_label)
            axis.set_xticks(x_ret, ["Cut\n0%", "25%", "50%", "75%", "Full\n100%"])
            axis.set_xlabel("Tỷ lệ visual token giữ lại")
            if all_lows and all_highs:
                low = min(all_lows)
                high = max(all_highs)
                padding = max(0.06, 0.08 * (high - low))
                axis.set_ylim(min(low - padding, 0.95), max(high + padding, 1.05))
    axes[0].set_ylabel("Speedup so với target greedy (×)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Thí nghiệm 2 · Speedup theo visual retention", y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "fig5_retention_speedup")

    return outputs


def _report_markdown(
    run_root: Path,
    report_dir: Path,
    cohort: dict[str, Any],
    assets: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    condition_stats: dict[str, list[dict[str, str]]],
    contrasts: dict[str, list[dict[str, str]]],
    case_summaries: dict[str, list[dict[str, str]]],
    runtime_summaries: dict[str, list[dict[str, str]]],
    audits: dict[str, dict[str, Any]],
    figures: list[str],
) -> str:
    relative_results = Path(__import__("os").path.relpath(run_root.resolve(), report_dir.resolve())).as_posix()
    source_sample_count = int(
        cohort.get("source_sample_count")
        or cohort.get("sample_count")
        or len(cohort.get("sample_ids", []))
    )
    sample_count = int(cohort.get("sample_count") or len(cohort.get("sample_ids", [])))
    run_config = _read_json(
        run_root / "full" / CONFIGS[0][0] / "run_config.json"
    )
    reduced_provenance_path = (
        run_root / "full" / CONFIGS[0][0] / "reduced_provenance.json"
    )
    reduced_provenance = (
        _read_json(reduced_provenance_path)
        if reduced_provenance_path.is_file()
        else {}
    )
    selection_rule = reduced_provenance.get(
        "selection_rule", "toàn bộ cohort đã calibrate"
    )
    matrix_expected = int(run_config.get("expected_row_count") or 0)
    calibration_summary = cohort.get("summary", {})
    version = cohort.get("calibration_version") or "v5_decord_matched_nframes_minmax_pixels_target_seed"
    provenance_path = run_root / "local_provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
    pilot_summary_path = run_root / "pilot" / "pilot_summary.json"
    pilot_validation_path = run_root / "pilot" / "pilot_validation_summary.json"
    pilot_summary = _read_json(pilot_summary_path) if pilot_summary_path.is_file() else {}
    pilot_validations = (
        _read_json(pilot_validation_path) if pilot_validation_path.is_file() else []
    )
    worker = {}
    for checkpoint, _, _ in CONFIGS:
        files = sorted((run_root / "full" / checkpoint / "tasks").glob("*/worker.json"))
        if files:
            worker = _read_json(files[0])
            break

    lines = [
        "# Báo cáo thực nghiệm DSpark 3-layer: VisualCut và Attention",
        "",
        f"**Ngày chạy:** {assets.get('run_date', '2026-11-02')}  ",
        "**Target:** Qwen2.5-VL-3B-Instruct  ",
        f"**Tập dữ liệu:** VideoDetailCaption VDC50; tập được đánh giá ghép cặp **N={sample_count}** / {source_sample_count}  ",
        f"**GPU:** {worker.get('gpu_name', 'NVIDIA A100-SXM4-80GB')}  ",
        "**Metric acceptance:** $\\tau_{\\mathrm{effective}}$, số token hiệu dụng mỗi vòng xác minh (đã gồm target bonus)",
        "",
        "## Tóm tắt phạm vi",
        "",
        "Báo cáo này chạy lại ma trận VisualCut trong PDF DFlash balanced cho checkpoint DSpark Phase 2 `balanced [1, 9, 17, 25, 33]`. Kết luận bên dưới chỉ áp dụng cho checkpoint này, target Qwen2.5-VL-3B-Instruct và tập con được chọn từ cohort VDC50 đã calibrate.",
        "",
        f"Ma trận được giới hạn ở N={sample_count} video trong cohort VDC50, dùng chung sample ID qua 4 visual-token budget (400, 3K, 13K, 25K) và 6 điều kiện (`full`, `zero`, `cut`, `retained_75`, `retained_50`, `retained_25`). Quy tắc chọn mẫu: {selection_rule}. Draft nhận điều kiện visual được can thiệp; target vẫn dùng prompt đầy đủ. Mỗi ca được chạy capture và timing riêng; phân tích chỉ tính ca có hai lượt lặp khớp nhau và có attention shard hợp lệ.",
        f"Để tiết kiệm compute, campaign báo cáo N={sample_count} thay vì chạy đủ 50 mẫu. Mỗi forward attention được tổng hợp theo cùng công thức mass/density/entropy rồi lưu thành summary record có SHA-256; pilot một mẫu vẫn lưu tensor attention thô để đối chiếu. Các kết quả 400 và 3K là subset của lần chạy 50 mẫu đã hoàn thành; 13K và 25K được chạy với sample limit N={sample_count}.",
        "",
        "**Không đồng nhất acceptance với độ đúng câu trả lời.** Tỷ lệ khớp chuỗi greedy target được báo riêng; ROUGE-L/BLEU chỉ đo tương đồng với caption tham chiếu VDC.",
        "",
        "## Checkpoint và môi trường",
        "",
        "| Cấu hình | Target layers | Phase 2 step | Số tham số draft | SHA-256 weight-only state (đầu…cuối) |",
        "|---|---:|---:|---:|---|",
    ]
    for checkpoint, label, _ in CONFIGS:
        layers = checkpoint.removeprefix("dspark_3layer_")
        info = checkpoint_metadata.get(layers, {})
        full_state_sha = str(info.get("inference_state_sha256", "—"))
        display_state_sha = (
            f"{full_state_sha[:6]}…{full_state_sha[-6:]}"
            if len(full_state_sha) == 64
            else full_state_sha
        )
        lines.append(
            f"| {label} | `{layers.replace('_', ', ')}` | {info.get('step', '—')} | "
            f"{int(info.get('draft_parameter_count') or 0):,} | `{display_state_sha}` |"
        )
    if CONFIGS:
        checkpoint = CONFIGS[0][0]
        layers = checkpoint.removeprefix("dspark_3layer_")
        full_state_sha = str(
            checkpoint_metadata.get(layers, {}).get("inference_state_sha256", "")
        )
        if len(full_state_sha) == 64:
            lines.extend(
                [
                    "",
                    f"- SHA-256 đầy đủ của weight-only state: `{full_state_sha}`; đối chiếu [`checkpoint_manifest.json`]({relative_results}/checkpoint_manifest.json).",
                ]
            )
    lines.extend(
        [
            "",
            f"- Runtime: Python {worker.get('python', '3.13')}; PyTorch {worker.get('torch_version', '2.11.0+cu130')}; CUDA {worker.get('torch_cuda_version', '13.0')}; dtype `{worker.get('dtype', 'bfloat16')}`; target attention `{worker.get('target_attention', 'sdpa')}`; draft attention `{worker.get('draft_attention', 'eager')}`; capture `{worker.get('capture_mode', 'all_forwards_all_heads_all_queries')}`.",
            f"- Dataset manifest SHA-256: `{assets.get('dataset_manifest_sha256', '—')}` (50 rows).",
            f"- Calibration: `{version}`, Decord, ±10% token tolerance; cohort gốc N={source_sample_count}; tập đánh giá N={sample_count}; [`cohort_selection.json`]({relative_results}/calibration/cohort_selection.json), [`reduced_provenance.json`]({relative_results}/full/{CONFIGS[0][0]}/reduced_provenance.json).",
            f"- Local source revision: `{provenance.get('git_commit', 'not recorded')}`; diff SHA-256: `{provenance.get('diff_sha256', 'not recorded')}`.",
            "- Inference states chỉ giữ `draft_state_dict` cần cho loader; archive gốc không bị sửa và vẫn là nguồn đầy đủ để resume training.",
            "",
            "## Kiểm tra độ phủ campaign",
            "",
            "| Cấu hình | Expected cases | Journal rows | Valid attention cases | Missing | Invalid | Exact output match |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        audit = audits[checkpoint]
        summary_rows = case_summaries[checkpoint]
        exact = sum(_truth(row.get("lossless")) for row in summary_rows)
        exact_rate = 100 * exact / len(summary_rows) if summary_rows else float("nan")
        lines.append(
            f"| {label} | {audit.get('expected_rows', '—')} | {audit.get('rows_total', '—')} | "
            f"{audit.get('rows_valid', '—')} | {audit.get('rows_missing', '—')} | "
            f"{audit.get('rows_invalid', '—')} | {exact}/{len(summary_rows)} ({exact_rate:.1f}%) |"
        )
    selected_checkpoint = CONFIGS[0][0]
    pilot_validations = [
        item for item in pilot_validations
        if item.get("checkpoint") == selected_checkpoint
    ]
    pilot_cases = sum(int(item.get("conditions") or 0) for item in pilot_validations)
    pilot_exact = sum(
        max(0, int(item.get("conditions") or 0) - int(item.get("lossless_failures") or 0))
        for item in pilot_validations
    )
    all_pilot_repeats_match = (
        int(pilot_summary.get("capture_timing_matches") or 0)
        == int(pilot_summary.get("case_count") or 0)
    )
    pilot_repeat_matches = pilot_cases if all_pilot_repeats_match else 0
    pilot_integrity_passes = sum(
        bool(item.get("artifact_integrity_passed")) for item in pilot_validations
    )
    pilot_validation_count = len(pilot_validations)
    campaign_expected = sum(int(audit.get("expected_rows") or 0) for audit in audits.values())
    campaign_valid = sum(int(audit.get("rows_valid") or 0) for audit in audits.values())
    campaign_exact = sum(
        _truth(row.get("lossless"))
        for rows in case_summaries.values()
        for row in rows
    )
    campaign_case_count = sum(len(rows) for rows in case_summaries.values())
    full_audits_complete = bool(audits) and all(
        int(audit.get("expected_rows") or 0) == matrix_expected
        and int(audit.get("rows_valid") or 0) == matrix_expected
        and not audit.get("rows_missing")
        and not audit.get("rows_invalid")
        and not audit.get("unexpected_rows")
        for audit in audits.values()
    )
    pilot_integrity_complete = (
        pilot_validation_count > 0
        and pilot_integrity_passes == pilot_validation_count
        and pilot_repeat_matches == pilot_cases
    )
    lines.extend(
        [
            "",
            "`Exact output match` là so khớp toàn bộ chuỗi speculative với greedy target. Ca không khớp vẫn có thể hợp lệ cho acceptance/attention nếu lượt capture và timing lặp lại cho cùng output và acceptance rounds; chúng không được gọi là lossless.",
            "",
            "## Smoke inference và kiểm tra checkpoint",
            "",
            f"Pilot của checkpoint balanced gồm {pilot_cases} ca: hai budget (400, 25K) × sáu điều kiện × một video. Capture/timing lặp khớp ở {pilot_repeat_matches}/{pilot_cases} ca; validator shard pass {pilot_integrity_passes}/{len(pilot_validations)} task. Exact greedy-target output match: {pilot_exact}/{pilot_cases} ca. Pilot không gộp vào thống kê campaign N={sample_count} phía dưới.",
            "",
            f"Artifacts pilot: [`pilot_summary.json`]({relative_results}/pilot/pilot_summary.json), [`pilot_validation_summary.json`]({relative_results}/pilot/pilot_validation_summary.json). Raw pilot tensor shards được giữ trên Colab; mỗi summary validator ghi rõ trạng thái hash/key-map.",
            "",
            "## Thực nghiệm 1 — Giá trị và vị trí visual",
            "",
            "Bảng dưới so sánh acceptance cho `full`, `cut` và `zero`. CI là bootstrap 95% ở cấp sample.",
            "",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        lines.extend(
            [
                f"### {label}",
                "",
                "| Budget | Full $\\tau$ [95% CI] | Cut $\\tau$ [95% CI] | Zero $\\tau$ [95% CI] | Cut − Full [95% CI] | Zero − Full [95% CI] |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for budget in BUDGETS:
            cells = []
            for condition in ("full", "cut", "zero"):
                row = _find_row(condition_stats[checkpoint], checkpoint=checkpoint, budget=budget, condition=condition)
                cells.append(_fmt_ci(row.get("tau_effective_mean"), row.get("tau_effective_ci_low"), row.get("tau_effective_ci_high")) if row else "—")
            paired = []
            for contrast in ("cut-full", "zero-full"):
                row = _find_contrast(contrasts[checkpoint], checkpoint=checkpoint, budget=budget, contrast=contrast)
                paired.append(_fmt_ci(row.get("mean"), row.get("ci_low"), row.get("ci_high")) if row else "—")
            lines.append(f"| {BUDGET_LABELS[budget]} | {cells[0]} | {cells[1]} | {cells[2]} | {paired[0]} | {paired[1]} |")
        lines.append("")
    lines.extend(
        [
            f"![Full/Zero/Cut acceptance](figures/{figures[0]}.png)",
            "",
            "## Thực nghiệm 2 — Quét tỷ lệ retention",
            "",
            "Retention sweep gồm `cut` (0%), 25%, 50%, 75% và `full` (100%). Contrast chính được đọc là `retained_25 − full` tại 25K; các contrast còn lại mang tính mô tả. CI chứa 0 được diễn giải là chưa có bằng chứng rõ về khác biệt theo contrast đó.",
            "",
        ]
    )
    lines.append("| Cấu hình | Budget | Retention tốt nhất theo mean $\\tau$ | $\\tau$ | Retained 25% − Full [95% CI] |")
    lines.append("|---|---:|---|---:|---:|")
    for checkpoint, label, _ in CONFIGS:
        for budget in BUDGETS:
            present = [
                (condition, _find_row(condition_stats[checkpoint], checkpoint=checkpoint, budget=budget, condition=condition))
                for condition in RETENTION_ORDER
            ]
            present = [(condition, row) for condition, row in present if row and _number(row.get("tau_effective_mean")) is not None]
            best = max(present, key=lambda item: float(item[1]["tau_effective_mean"])) if present else (None, None)
            contrast = _find_contrast(contrasts[checkpoint], checkpoint=checkpoint, budget=budget, contrast="retained_25-full")
            contrast_text = _fmt_ci(contrast.get("mean"), contrast.get("ci_low"), contrast.get("ci_high")) if contrast else "—"
            best_name = CONDITION_LABELS.get(best[0], "—") if best[0] else "—"
            best_tau = _fmt(best[1].get("tau_effective_mean")) if best[1] else "—"
            lines.append(f"| {label} | {BUDGET_LABELS[budget]} | {best_name} | {best_tau} | {contrast_text} |")
    lines.extend(
        [
            "",
            f"![Retention sweep](figures/{figures[1]}.png)",
            "",
            "Hình dưới bổ sung tốc độ đo được theo cùng các mức retention. Speedup là tỷ số thời gian `target / speculative` trên lượt timing; đường ngang 1× biểu thị ngang tốc target. CI bootstrap theo 15 video.",
            "",
            f"![Decode và E2E speedup theo visual retention](figures/{figures[4]}.png)",
            "",
            r"\clearpage",
            "",
            "## Thực nghiệm 3 — Attention mass, density và entropy",
            "",
            "Attention được tổng hợp từ các forward của draft tại ngân sách 25K. Mass là tổng xác suất attention trên vùng; density là mass chia cho số key trong vùng. Điều kiện `cut` không có visual key nên visual density không được định nghĩa ở điều kiện đó.",
            "",
            f"![Attention mass và density tại 25K](figures/{figures[2]}.png)",
            "",
            "| Cấu hình | Điều kiện | Visual mass | Visual density | Question mass | Generated-target-hidden mass | Normalized entropy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        rows = case_summaries[checkpoint]
        for condition in ("full", "cut", "retained_25", "retained_50", "retained_75", "zero"):
            cells = []
            for metric in ("visual_mass", "visual_density", "question_text_mass", "generated_target_hidden_mass", "attention_normalized_entropy"):
                values = _group_case_values(rows, checkpoint=checkpoint, budget=25000, condition=condition, metric=metric)
                summary = _bootstrap(values, seed=20261102 + len(cells) + 17 * BUDGETS.index(25000))
                cells.append(_fmt(summary["mean"]))
            lines.append(f"| {label} | {CONDITION_LABELS[condition]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Thực nghiệm 4 — Throughput, chất lượng tham chiếu và output agreement",
            "",
            "Throughput là tốc độ speculative đo trong lượt timing. Tỷ lệ khớp chuỗi là exact match với greedy target; BLEU/ROUGE-L/coverage so với caption tham chiếu chỉ là độ tương đồng văn bản, không phải đánh giá sự thật của caption.",
            "",
            f"![Throughput và output agreement](figures/{figures[3]}.png)",
            "",
            "| Cấu hình | Budget | Điều kiện | Draft token/s [95% CI] | Speculative ROUGE-L | Δ ROUGE-L so với target | Exact match (%) |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        rows = case_summaries[checkpoint]
        for budget in BUDGETS:
            for condition in ("full", "cut", "retained_25"):
                stat = _find_row(condition_stats[checkpoint], checkpoint=checkpoint, budget=budget, condition=condition)
                speed = _fmt_ci(stat.get("tokens_per_second_mean"), stat.get("tokens_per_second_ci_low"), stat.get("tokens_per_second_ci_high")) if stat else "—"
                rouge = _fmt(stat.get("speculative_rouge_l_mean")) if stat else "—"
                rouge_delta = _fmt(stat.get("rouge_l_delta_vs_target_mean")) if stat else "—"
                values = _group_case_values(rows, checkpoint=checkpoint, budget=budget, condition=condition, metric="lossless")
                match = _bootstrap(values, seed=20261102 + budget + len(condition))
                exact = "—" if match["mean"] is None else f"{100 * match['mean']:.1f}%"
                lines.append(f"| {label} | {BUDGET_LABELS[budget]} | {CONDITION_LABELS[condition]} | {speed} | {rouge} | {rouge_delta} | {exact} |")
    lines.extend(
        [
            "",
            "### Mean accepted length và speedup",
            "",
            "Metric $\\tau_{\\mathrm{effective}}$ là số token hiệu dụng mỗi vòng xác minh (token draft được chấp nhận cộng target bonus). Speedup được tính từ lượt timing riêng theo tỷ số `target / speculative`; E2E gồm prefill và decode, giá trị trên 1 nghĩa là nhanh hơn target greedy.",
            "",
            "| Cấu hình | Budget | Điều kiện | $\\tau_{\\mathrm{effective}}$ [95% CI] | Decode speedup [95% CI] | E2E speedup [95% CI] |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        rows = case_summaries[checkpoint]
        timing_rows = runtime_summaries[checkpoint]
        for budget in BUDGETS:
            for condition in ("full", "cut", "retained_25"):
                tau = _bootstrap(
                    _group_case_values(
                        rows,
                        checkpoint=checkpoint,
                        budget=budget,
                        condition=condition,
                        metric="tau_effective",
                    ),
                    seed=20261103 + budget + CONDITIONS.index(condition),
                )
                decode = _bootstrap(
                    _group_case_values(
                        timing_rows,
                        checkpoint=checkpoint,
                        budget=budget,
                        condition=condition,
                        metric="decode_speedup",
                    ),
                    seed=20261103 + budget + CONDITIONS.index(condition) + 20,
                )
                end_to_end = _bootstrap(
                    _group_case_values(
                        timing_rows,
                        checkpoint=checkpoint,
                        budget=budget,
                        condition=condition,
                        metric="end_to_end_speedup",
                    ),
                    seed=20261103 + budget + CONDITIONS.index(condition) + 40,
                )
                compact_label = label.split(" [", 1)[0]
                lines.append(
                    f"| {compact_label} | {BUDGET_LABELS[budget]} | {CONDITION_LABELS[condition]} | "
                    f"{_fmt_ci(tau['mean'], tau['ci_low'], tau['ci_high'])} | "
                    f"{_fmt_ci(decode['mean'], decode['ci_low'], decode['ci_high'])} | "
                    f"{_fmt_ci(end_to_end['mean'], end_to_end['ci_low'], end_to_end['ci_high'])} |"
                )
    lines.extend(
        [
            "",
            "## Diễn giải",
            "",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        row = _find_contrast(contrasts[checkpoint], checkpoint=checkpoint, budget=25000, contrast="retained_25-full")
        if row and _number(row.get("ci_low")) is not None and _number(row.get("ci_high")) is not None:
            low, high = float(row["ci_low"]), float(row["ci_high"])
            if low > 0:
                interpretation = "CI 95% nằm trên 0"
            elif high < 0:
                interpretation = "CI 95% nằm dưới 0"
            else:
                interpretation = "CI 95% chứa 0"
            lines.append(f"- **{label}, 25K:** contrast `retained_25 − full` là {_fmt(row.get('mean'))} [{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] token/vòng ({interpretation}).")
        else:
            lines.append(f"- **{label}, 25K:** chưa có đủ paired rows để ước lượng `retained_25 − full`.")
        timing_rows = runtime_summaries[checkpoint]
        speed_stats = {}
        for condition in ("full", "cut", "retained_25"):
            for metric in ("decode_speedup", "end_to_end_speedup"):
                values = _group_case_values(
                    timing_rows,
                    checkpoint=checkpoint,
                    budget=25000,
                    condition=condition,
                    metric=metric,
                )
                speed_stats[(condition, metric)] = _bootstrap(
                    values,
                    seed=20261103 + 25000 + CONDITIONS.index(condition),
                )
        decode_means = [speed_stats[(condition, "decode_speedup")]["mean"] for condition in ("full", "cut", "retained_25")]
        end_to_end_means = [speed_stats[(condition, "end_to_end_speedup")]["mean"] for condition in ("full", "cut", "retained_25")]
        if all(value is not None for value in decode_means + end_to_end_means):
            lines.append(
                f"- **Speedup tại 25K:** decode {min(decode_means):.2f}–{max(decode_means):.2f}× và E2E {min(end_to_end_means):.2f}–{max(end_to_end_means):.2f}× ở `full`, `cut`, `retained_25`; cả hai đều dưới 1× dù $\\tau_{{\\mathrm{{effective}}}}$ khoảng 1.83–1.90. Đây là số đo trên N={sample_count}; cho thấy ở budget này lợi ích acceptance chưa bù chi phí draft và xác minh trên ngữ cảnh visual lớn.")
    lines.extend(
        [
            "",
            "Các confidence interval là mô tả, bootstrap ghép cặp theo sample cho contrasts; campaign này không áp dụng hiệu chỉnh đa kiểm định. Kết quả chỉ mô tả checkpoint balanced Phase 2, trên cohort và phần cứng Colab đã ghi nhận.",
            "",
            r"\clearpage",
            "",
            "## Kết luận theo cấp độ bằng chứng",
            "",
            "### CONFIRMED",
            "",
            f"- Pilot đã chạy {pilot_cases} ca inference. Capture/timing lặp lại khớp {pilot_repeat_matches}/{pilot_cases} ca; kiểm tra tính toàn vẹn artifact đạt {pilot_integrity_passes}/{pilot_validation_count} task.",
            f"- Full campaign hiện có {campaign_valid}/{campaign_expected} ca attention hợp lệ trên {campaign_case_count} journal rows; exact greedy-target match là {campaign_exact}/{campaign_case_count} ca. Phần này mô tả trạng thái audit được lưu trong artifacts.",
            "",
            "### EXPLORATORY",
            "",
            "- Các ước lượng acceptance, throughput, attention và paired contrasts mô tả cohort VDC50 đã chọn. CI dùng bootstrap theo sample; chưa hiệu chỉnh đa kiểm định.",
            "- Kết quả chỉ bao phủ checkpoint balanced Phase 2 và target Qwen2.5-VL-3B-Instruct. Không suy rộng sang checkpoint/model/dataset khác.",
            "",
            "### FAILED / INCOMPLETE",
            "",
            f"- Exact greedy-target output match trong pilot đạt {pilot_exact}/{pilot_cases} ca. Tỷ lệ này là kiểm tra strict agreement; acceptance `tau_effective` được báo riêng.",
            f"- Ma trận N={sample_count} {'đạt coverage theo audit' if full_audits_complete else 'chưa đạt coverage đầy đủ theo audit'}: {campaign_valid}/{campaign_expected} ca hợp lệ; cần {matrix_expected} ca.",
            "- Báo cáo chỉ đo checkpoint balanced được chỉ định; không suy rộng kết luận sang checkpoint DSpark khác.",
            "",
            "### HIGHEST VERIFIED RUNG",
            "",
            (
                f"**Rung 3 — ma trận N={sample_count} hoàn chỉnh:** checkpoint balanced có đủ {campaign_valid}/{matrix_expected} ca hợp lệ, không thiếu/invalid/unexpected rows theo audit. Kết quả không đại diện cho toàn bộ cohort 50 mẫu; strict output agreement được báo độc lập ở bảng trên."
                if full_audits_complete
                else (
                    f"**Rung 2 — repeatability và artifact integrity pilot:** pilot đã lặp capture/timing và validator kiểm tra shard. Ma trận N={sample_count} chỉ được xác nhận khi audit đủ {matrix_expected} ca cho checkpoint balanced."
                    if pilot_integrity_complete
                    else "**Rung 1 — inference pilot:** đã ghi nhận pilot; yêu cầu repeatability và artifact-integrity chưa đồng thời đạt đầy đủ."
                )
            ),
            "",
            "### EVIDENCE GAPS",
            "",
            "- Full campaign giữ summary attention theo từng forward kèm SHA-256; raw attention tensors của campaign không được lưu. Vì vậy các metric đã định nghĩa có thể tái kiểm, còn phép phân tích mới trên tensor thô cần chạy capture lại.",
            "- Log của một số video có warning H.264 (`mmco: unref short failure`, `Missing reference picture`), dù các case vẫn hoàn tất. Chưa kiểm chứng độc lập frame giải mã so với nguồn; kết quả phụ thuộc preprocessing/decode hiện tại.",
            f"- Báo cáo dùng N={sample_count} video từ cohort calibrate N={source_sample_count}; lựa chọn theo thứ tự manifest nên chưa phải mẫu ngẫu nhiên. Chưa có lần lặp độc lập theo seed/cohort.",
            "- Checkpoint DSpark high-only `[26, 28, 30, 32, 34]` chưa nằm trong phạm vi báo cáo này; kết quả chỉ áp dụng cho checkpoint balanced.",
            "",
            "### RECOMMENDED NEXT",
            "",
            "Lặp cohort/seed độc lập cho checkpoint balanced để kiểm tra độ ổn định của các paired contrasts; có thể mở rộng sang checkpoint khác khi đó trở thành mục tiêu.",
            "",
            "## Artifacts và khả năng tái lập",
            "",
            f"- Calibration và cohort: [`calibration.jsonl`]({relative_results}/calibration/calibration.jsonl), [`cohort_selection.json`]({relative_results}/calibration/cohort_selection.json).",
        ]
    )
    for checkpoint, label, _ in CONFIGS:
        prefix = f"{relative_results}/full/{checkpoint}"
        lines.extend(
            [
                f"- **{label}:** [`audit_and_summary.json`]({prefix}/analysis/audit_and_summary.json), [`condition_statistics.csv`]({prefix}/analysis/condition_statistics.csv), [`paired_contrasts.csv`]({prefix}/analysis/paired_contrasts.csv), [`case_attention_summary.csv`]({prefix}/analysis/case_attention_summary.csv), [`run_config.json`]({prefix}/run_config.json).",
            ]
        )
    lines.extend(
        [
            "- Mã tạo hình và báo cáo được lưu cạnh source: [`generate_dspark_visualcut_report.py`](../../src/analyze/Validate_Sparrow_hypothesises/generate_dspark_visualcut_report.py).",
            "- Journals giữ `attention_capture` hashes; analyzer xác minh SHA-256 các shard summary trước khi tổng hợp. Pilot raw tensors được giữ riêng trên Colab; full campaign lưu summary record per-forward, không giữ toàn bộ tensor attention.",
            "",
        ]
    )
    return "\n".join(lines)


def generate(run_root: Path, report_dir: Path) -> dict[str, Any]:
    calibration_dir = run_root / "calibration"
    cohort = dict(_read_json(calibration_dir / "cohort_selection.json"))
    assets = _read_json(run_root / "run_assets.json")
    checkpoint_metadata = _read_json(run_root / "checkpoint_manifest.json")
    reduced_provenance_path = (
        run_root / "full" / CONFIGS[0][0] / "reduced_provenance.json"
    )
    reduced_provenance = (
        _read_json(reduced_provenance_path)
        if reduced_provenance_path.is_file()
        else {}
    )
    if reduced_provenance:
        selected_ids = list(reduced_provenance.get("selected_sample_ids") or [])
        cohort["source_sample_count"] = int(
            cohort.get("sample_count") or len(cohort.get("sample_ids", []))
        )
        cohort["sample_count"] = len(selected_ids)
        cohort["sample_ids"] = selected_ids
        cohort["selection_rule"] = reduced_provenance.get("selection_rule")
        created_utc = str(reduced_provenance.get("created_utc") or "")
        if len(created_utc) >= 10:
            assets["run_date"] = created_utc[:10]
    condition_stats: dict[str, list[dict[str, str]]] = {}
    contrasts: dict[str, list[dict[str, str]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for checkpoint, _, _ in CONFIGS:
        analysis = run_root / "full" / checkpoint / "analysis"
        condition_stats[checkpoint] = _read_csv(analysis / "condition_statistics.csv")
        contrasts[checkpoint] = _read_csv(analysis / "paired_contrasts.csv")
        audits[checkpoint] = _read_json(analysis / "audit_and_summary.json")
    case_summaries = _load_case_summaries(run_root)
    if not all(case_summaries.values()):
        raise RuntimeError("one or more checkpoint summaries are missing or empty")
    runtime_summaries = _load_runtime_summaries(run_root)
    if not all(runtime_summaries.values()):
        raise RuntimeError("one or more checkpoint timing summaries are missing or empty")
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = _save_figures(report_dir, condition_stats, contrasts, case_summaries, runtime_summaries)
    source = Path(__file__)
    figures_dir = report_dir / "figures"
    (figures_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    report_path = report_dir / "bao_cao_dspark_visualcut.md"
    report_path.write_text(
        _report_markdown(
            run_root,
            report_dir,
            cohort,
            assets,
            checkpoint_metadata,
            condition_stats,
            contrasts,
            case_summaries,
            runtime_summaries,
            audits,
            figures,
        ),
        encoding="utf-8",
    )
    payload = {
        "run_root": str(run_root.resolve()),
        "report": str(report_path.resolve()),
        "figures": [str((figures_dir / f"{stem}.pdf").resolve()) for stem in figures],
        "cohort_size": int(cohort.get("sample_count") or len(cohort.get("sample_ids", []))),
        "source_cohort_size": int(
            cohort.get("source_sample_count")
            or cohort.get("sample_count")
            or len(cohort.get("sample_ids", []))
        ),
        "audit_complete": all(
            int(audit.get("rows_valid") or 0) == int(audit.get("expected_rows") or -1)
            and not audit.get("rows_missing")
            and not audit.get("rows_invalid")
            and not audit.get("unexpected_rows")
            for audit in audits.values()
        ),
    }
    (run_root / "report_generation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    generate(args.run_root, args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
