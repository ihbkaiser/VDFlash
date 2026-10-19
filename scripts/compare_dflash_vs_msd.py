#!/usr/bin/env python3
"""Compare DFlash vs. MSD speculative decoding benchmarks on Qwen2.5-VL-3B.

Generates comprehensive comparison statistics, tables, and hallucination metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "video", "clip", "scene", "someone", "something",
}


def extract_objects(text: str) -> list[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def analyze_vidhalluc(jsonl_path: Path, manifest_path: Path = Path("dataset/VidHalluc/test.jsonl")) -> dict[int, dict[str, Any]]:
    gt_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                gt_data[d.get("video_name")] = d

    rows = load_jsonl(jsonl_path)
    by_ret = defaultdict(list)
    for r in rows:
        ret = int(round(float(r.get("retention_percentage", 100.0))))
        by_ret[ret].append(r)

    results = {}
    for ret in sorted(by_ret.keys(), reverse=True):
        group = by_ret[ret]
        total = len(group)
        lossless_count = sum(1 for x in group if x.get("lossless") is True)
        mean_speedup = sum(float(x.get("decode_speedup", 1.0)) for x in group) / max(total, 1)
        mean_e2e_speedup = sum(float(x.get("end_to_end_speedup", 1.0)) for x in group) / max(total, 1)
        # In DFlash, tau_effective is explicitly stored. In MSD logs, accepted_prefix_tokens is stored (or mean_accepted_tokens).
        # Standard speculative decoding tau = 1 + accepted_draft_tokens (or effective tokens per step).
        def get_tau(row):
            if "tau_effective" in row and row["tau_effective"] is not None:
                return float(row["tau_effective"])
            if "accepted_prefix_tokens" in row and row["accepted_prefix_tokens"] is not None:
                return float(row["accepted_prefix_tokens"]) + 1.0
            if "mean_accepted_tokens" in row and row["mean_accepted_tokens"] is not None:
                return float(row["mean_accepted_tokens"])
            return 1.0
        mean_tau = sum(get_tau(x) for x in group) / max(total, 1)

        # Accuracy & Hallucination
        correct = 0
        bin_correct = 0
        bin_total = 0
        for x in group:
            vid = x.get("video_name") or x.get("sample_id")
            gt = gt_data.get(vid)
            if not gt:
                continue
            pred_text = x.get("speculative_output_text", "").strip()
            gt_ans = gt.get("answer", "")
            task_type = gt.get("task_type")

            if task_type == "BinaryQA":
                bin_total += 1
                p_lower = pred_text.lower()
                pred = "Yes" if "yes" in p_lower else ("No" if "no" in p_lower else "Unknown")
                if pred == gt_ans:
                    bin_correct += 1
                    correct += 1
            else:
                # MCQ or other
                p_char = pred_text[:1].upper() if pred_text else ""
                gt_char = gt_ans[:1].upper() if gt_ans else ""
                if p_char and p_char == gt_char:
                    correct += 1

        results[ret] = {
            "total_samples": total,
            "decode_speedup": mean_speedup,
            "end_to_end_speedup": mean_e2e_speedup,
            "tau_effective": mean_tau,
            "lossless_rate": (lossless_count / total * 100.0) if total else 0.0,
            "overall_accuracy": (correct / total * 100.0) if total else 0.0,
            "binary_accuracy": (bin_correct / bin_total * 100.0) if bin_total else 0.0,
        }
    return results


def analyze_videochair(jsonl_path: Path, manifest_path: Path = Path("dataset/VideoChair/test.jsonl")) -> dict[int, dict[str, Any]]:
    gt_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                gt_data[d.get("video_name")] = d

    rows = load_jsonl(jsonl_path)
    by_ret = defaultdict(list)
    for r in rows:
        ret = int(round(float(r.get("retention_percentage", 100.0))))
        by_ret[ret].append(r)

    results = {}
    for ret in sorted(by_ret.keys(), reverse=True):
        group = by_ret[ret]
        total = len(group)
        lossless_count = sum(1 for x in group if x.get("lossless") is True)
        mean_speedup = sum(float(x.get("decode_speedup", 1.0)) for x in group) / max(total, 1)
        mean_e2e_speedup = sum(float(x.get("end_to_end_speedup", 1.0)) for x in group) / max(total, 1)
        def get_tau(row):
            if "tau_effective" in row and row["tau_effective"] is not None:
                return float(row["tau_effective"])
            if "accepted_prefix_tokens" in row and row["accepted_prefix_tokens"] is not None:
                return float(row["accepted_prefix_tokens"]) + 1.0
            if "mean_accepted_tokens" in row and row["mean_accepted_tokens"] is not None:
                return float(row["mean_accepted_tokens"])
            return 1.0
        mean_tau = sum(get_tau(x) for x in group) / max(total, 1)

        # CHAIR metrics
        total_gen_objs = 0
        total_halluc_objs = 0
        total_recalled_objs = 0
        total_gt_objs = 0

        for x in group:
            vid = x.get("video_name") or x.get("sample_id")
            gt = gt_data.get(vid)
            if not gt:
                continue
            pred_text = x.get("speculative_output_text", "")
            gen_objs = extract_objects(pred_text)
            gt_objs = set(gt.get("ground_truth_objects", []))
            total_gt_objs += len(gt_objs)

            for obj in gen_objs:
                total_gen_objs += 1
                if obj in gt_objs:
                    total_recalled_objs += 1
                else:
                    total_halluc_objs += 1

        chair_i = (total_halluc_objs / total_gen_objs) if total_gen_objs else 0.0
        obj_prec = 1.0 - chair_i
        obj_recall = (total_recalled_objs / total_gt_objs) if total_gt_objs else 0.0

        results[ret] = {
            "total_samples": total,
            "decode_speedup": mean_speedup,
            "end_to_end_speedup": mean_e2e_speedup,
            "tau_effective": mean_tau,
            "lossless_rate": (lossless_count / total * 100.0) if total else 0.0,
            "chair_i": chair_i * 100.0,
            "object_precision": obj_prec * 100.0,
            "object_recall": obj_recall * 100.0,
        }
    return results


def print_comparison_table():
    msd_vh = Path("results/msd_eval_20261007/vidhalluc/msd_retention_vidhalluc.jsonl")
    dflash_vh = Path("results/dflash_eval_20261013/vidhalluc/dflash_retention_vidhalluc.jsonl")

    msd_vc = Path("results/msd_eval_20261007/videochair/msd_retention_videochair.jsonl")
    dflash_vc = Path("results/dflash_eval_20261013/videochair/dflash_retention_videochair.jsonl")

    print("\n" + "=" * 90)
    print(" HEAD-TO-HEAD COMPARISON: DFLASH VS. MSD ON QWEN2.5-VL-3B (2-GPU EVALUATION)")
    print("=" * 90)

    # 1. VidHalluc
    if dflash_vh.exists():
        df_vh_res = analyze_vidhalluc(dflash_vh)
        msd_vh_res = analyze_vidhalluc(msd_vh) if msd_vh.exists() else {}

        print("\n### 1. VidHalluc Benchmark (100 Samples)")
        print(f"| Retention | Model | Samples | Decode Speedup | E2E Speedup | tau_eff | Lossless Rate | Task Accuracy |")
        print(f"|---|---|---|---|---|---|---|---|")
        for ret in [100, 50, 25, 0]:
            if ret in msd_vh_res:
                m = msd_vh_res[ret]
                print(f"| {ret}% | **MSD (Depth 3)** | {m['total_samples']} | {m['decode_speedup']:.2f}x | {m['end_to_end_speedup']:.2f}x | {m['tau_effective']:.2f} | {m['lossless_rate']:.1f}% | {m['overall_accuracy']:.1f}% |")
            if ret in df_vh_res:
                d = df_vh_res[ret]
                print(f"| {ret}% | **DFlash (Depth 3)** | {d['total_samples']} | **{d['decode_speedup']:.2f}x** | **{d['end_to_end_speedup']:.2f}x** | **{d['tau_effective']:.2f}** | **{d['lossless_rate']:.1f}%** | {d['overall_accuracy']:.1f}% |")

    # 2. VideoChair
    if dflash_vc.exists():
        df_vc_res = analyze_videochair(dflash_vc)
        msd_vc_res = analyze_videochair(msd_vc) if msd_vc.exists() else {}

        print("\n### 2. VideoChair Benchmark (100 Samples)")
        print(f"| Retention | Model | Samples | Decode Speedup | E2E Speedup | tau_eff | CHAIR_i (lower is better) | Obj Precision |")
        print(f"|---|---|---|---|---|---|---|---|")
        for ret in [100, 50, 25, 0]:
            if ret in msd_vc_res:
                m = msd_vc_res[ret]
                print(f"| {ret}% | **MSD (Depth 3)** | {m['total_samples']} | {m['decode_speedup']:.2f}x | {m['end_to_end_speedup']:.2f}x | {m['tau_effective']:.2f} | {m['chair_i']:.2f}% | {m['object_precision']:.2f}% |")
            if ret in df_vc_res:
                d = df_vc_res[ret]
                print(f"| {ret}% | **DFlash (Depth 3)** | {d['total_samples']} | **{d['decode_speedup']:.2f}x** | **{d['end_to_end_speedup']:.2f}x** | **{d['tau_effective']:.2f}** | **{d['chair_i']:.2f}%** | {d['object_precision']:.2f}% |")


if __name__ == "__main__":
    print_comparison_table()
