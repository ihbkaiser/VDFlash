#!/usr/bin/env python3
"""Đánh giá chi tiết các chỉ số Performance & Hallucination trên VidHalluc và VideoChair

Tính toán các chỉ số:
1. VidHalluc:
   - BinaryQA: Accuracy, Precision, Recall, F1, Hallucination Rate
   - MCQ: Accuracy, Hallucination Rate
   - Scene Transition: Scene Change Detection Accuracy
   - Sequence Ranking: Temporal Sequence Accuracy
   - Overall Benchmark Accuracy qua 4 mức cut level (100%, 50%, 25%, 0%)
2. VideoChair (CHAIR):
   - CHAIR_i (Instance-level Object Hallucination)
   - CHAIR_s (Sentence-level Object Hallucination)
   - Object Precision (1 - CHAIR_i)
   - Object Recall (Tỷ lệ thực thể GT được nhận diện)
   - Số lượng thực thể sinh ra trung bình / video
   - So sánh qua 4 mức cut level (100%, 50%, 25%, 0%)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

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

def evaluate_vidhalluc():
    print("=" * 60)
    print("ĐÁNH GIÁ CHỈ SỐ VIDHALLUC")
    print("=" * 60)

    gt_data = {}
    with open("dataset/VidHalluc/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gt_data[d["video_name"]] = d

    results_by_ret = defaultdict(lambda: {
        "binary": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0, "correct": 0},
        "mcq": {"total": 0, "correct": 0},
        "transition": {"total": 0, "correct": 0, "change_detected_correct": 0},
        "ranking": {"total": 0, "correct": 0},
        "all": {"total": 0, "correct": 0}
    })

    with open("results/msd_eval_20261007/vidhalluc/msd_retention_vidhalluc.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "error" in d or "retention_percentage" not in d:
                continue
            vid = d["sample_id"]
            ret = int(round(d["retention_percentage"]))
            gt = gt_data.get(vid)
            if not gt:
                continue

            pred_text = d["speculative_output_text"]
            task_type = gt.get("task_type")
            gt_ans = gt.get("answer")
            res = results_by_ret[ret]
            res["all"]["total"] += 1

            if task_type == "BinaryQA":
                res["binary"]["total"] += 1
                p_lower = pred_text.strip().lower()
                if p_lower.startswith("yes"):
                    pred = "Yes"
                elif p_lower.startswith("no"):
                    pred = "No"
                else:
                    pred = "Unknown"

                is_correct = (pred == gt_ans)
                if is_correct:
                    res["binary"]["correct"] += 1
                    res["all"]["correct"] += 1
                
                # confusion matrix for Yes (Positive) vs No (Negative)
                if gt_ans == "Yes":
                    if pred == "Yes":
                        res["binary"]["tp"] += 1
                    else:
                        res["binary"]["fn"] += 1
                else:
                    if pred == "No":
                        res["binary"]["tn"] += 1
                    else:
                        res["binary"]["fp"] += 1

            elif task_type == "MCQ":
                res["mcq"]["total"] += 1
                choices = gt.get("choices", {})
                gt_choice_text = choices.get(gt_ans, "").lower()
                p_lower = pred_text.strip().lower()
                
                # Check option letter or choice text match
                matched = False
                if f"({gt_ans.lower()})" in p_lower or f"option {gt_ans.lower()}" in p_lower:
                    matched = True
                elif gt_choice_text and gt_choice_text in p_lower:
                    matched = True
                elif p_lower.startswith(gt_ans.lower()) or f" {gt_ans.lower()}." in p_lower:
                    matched = True

                if matched:
                    res["mcq"]["correct"] += 1
                    res["all"]["correct"] += 1

            elif task_type == "Scene_Transition":
                res["transition"]["total"] += 1
                p_lower = pred_text.strip().lower()
                # Check binary scene change
                gt_has_change = ("scene change: yes" in gt_ans.lower() or "yes" in gt_ans.lower())
                pred_has_change = (p_lower.startswith("yes") or "scene changes" in p_lower or "scene change: yes" in p_lower)
                if gt_has_change == pred_has_change:
                    res["transition"]["change_detected_correct"] += 1

            elif task_type == "Sequence_Ranking":
                res["ranking"]["total"] += 1
                p_lower = pred_text.strip().lower()
                if gt_ans.lower() in p_lower:
                    res["ranking"]["correct"] += 1
                    res["all"]["correct"] += 1

    summary_vidhalluc = {}
    for ret in [100, 50, 25, 0]:
        r = results_by_ret[ret]
        b = r["binary"]
        m = r["mcq"]
        t = r["transition"]
        rk = r["ranking"]
        a = r["all"]

        acc_bin = b["correct"] / b["total"] if b["total"] else 0.0
        acc_mcq = m["correct"] / m["total"] if m["total"] else 0.0
        acc_trans = t["change_detected_correct"] / t["total"] if t["total"] else 0.0
        acc_rank = rk["correct"] / rk["total"] if rk["total"] else 0.0
        acc_all = a["correct"] / a["total"] if a["total"] else 0.0

        prec_bin = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 0.0
        rec_bin = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0.0
        f1_bin = 2 * prec_bin * rec_bin / (prec_bin + rec_bin) if (prec_bin + rec_bin) else 0.0

        summary_vidhalluc[ret] = {
            "all_acc": acc_all,
            "all_halluc_rate": 1.0 - acc_all,
            "mcq_acc": acc_mcq,
            "mcq_correct": m["correct"],
            "mcq_total": m["total"],
            "bin_acc": acc_bin,
            "bin_correct": b["correct"],
            "bin_total": b["total"],
            "bin_precision": prec_bin,
            "bin_recall": rec_bin,
            "bin_f1": f1_bin,
            "trans_change_acc": acc_trans,
            "trans_correct": t["change_detected_correct"],
            "trans_total": t["total"],
            "rank_acc": acc_rank,
            "rank_correct": rk["correct"],
            "rank_total": rk["total"],
        }

        print(f"\n--- Retention {ret}% ---")
        print(f"Toàn bộ VidHalluc: {a['correct']}/{a['total']} ({acc_all*100:.2f}%) | Tỷ lệ Ảo giác: {(1-acc_all)*100:.2f}%")
        print(f"MCQ (35 mẫu): {m['correct']}/{m['total']} ({acc_mcq*100:.2f}%)")
        print(f"BinaryQA (25 mẫu): {b['correct']}/{b['total']} ({acc_bin*100:.2f}%) | Precision: {prec_bin*100:.2f}%, Recall: {rec_bin*100:.2f}%, F1: {f1_bin*100:.2f}%")
        print(f"Scene Transition (20 mẫu - Phát hiện đổi cảnh): {t['change_detected_correct']}/{t['total']} ({acc_trans*100:.2f}%)")
        print(f"Sequence Ranking (20 mẫu): {rk['correct']}/{rk['total']} ({acc_rank*100:.2f}%)")

    return summary_vidhalluc


def evaluate_videochair():
    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ CHỈ SỐ VIDEOCHAIR (CHAIR METRICS)")
    print("=" * 60)

    gt_data = {}
    with open("dataset/VideoChair/test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gt_data[d["video_name"]] = {
                "objects": set(obj.lower() for obj in d.get("ground_truth_objects", [])),
                "captions": d.get("ground_truth_captions", [])
            }

    results_by_ret = defaultdict(lambda: {
        "total_videos": 0,
        "total_gen_objects": 0,
        "total_halluc_objects": 0,
        "total_gt_objects": 0,
        "total_recalled_objects": 0,
        "sentences_with_halluc": 0,
        "chair_i_list": [],
        "chair_s_list": [],
    })

    with open("results/msd_eval_20261007/videochair/msd_retention_videochair.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "error" in d or "retention_percentage" not in d:
                continue
            vid = d["sample_id"]
            ret = int(round(d["retention_percentage"]))
            gt = gt_data.get(vid)
            if not gt:
                continue

            pred_text = d["speculative_output_text"]
            gen_objects = extract_objects(pred_text)
            gt_objects = gt["objects"]

            halluc_count = 0
            recalled_count = 0
            for obj in gen_objects:
                if obj in gt_objects:
                    recalled_count += 1
                else:
                    halluc_count += 1

            r = results_by_ret[ret]
            r["total_videos"] += 1
            r["total_gen_objects"] += len(gen_objects)
            r["total_halluc_objects"] += halluc_count
            r["total_gt_objects"] += len(gt_objects)
            r["total_recalled_objects"] += recalled_count

            # Sentence/caption level
            is_sent_halluc = (halluc_count > 0)
            if is_sent_halluc:
                r["sentences_with_halluc"] += 1

            # Per-caption CHAIR_i
            c_i = halluc_count / len(gen_objects) if gen_objects else 0.0
            r["chair_i_list"].append(c_i)
            r["chair_s_list"].append(1.0 if is_sent_halluc else 0.0)

    summary_chair = {}
    for ret in [100, 50, 25, 0]:
        r = results_by_ret[ret]
        n_vid = r["total_videos"]
        tot_gen = r["total_gen_objects"]
        tot_hal = r["total_halluc_objects"]
        tot_rec = r["total_recalled_objects"]
        tot_gt = r["total_gt_objects"]

        # Dataset-level CHAIR_i
        chair_i = tot_hal / tot_gen if tot_gen else 0.0
        chair_s = r["sentences_with_halluc"] / n_vid if n_vid else 0.0
        obj_precision = tot_rec / tot_gen if tot_gen else 0.0
        obj_recall = tot_rec / tot_gt if tot_gt else 0.0
        mean_obj_count = tot_gen / n_vid if n_vid else 0.0

        summary_chair[ret] = {
            "chair_i": chair_i,
            "chair_s": chair_s,
            "object_precision": obj_precision,
            "object_recall": obj_recall,
            "mean_objects_per_caption": mean_obj_count,
            "total_gen_objects": tot_gen,
            "total_halluc_objects": tot_hal,
            "total_recalled_objects": tot_rec,
        }

        print(f"\n--- Retention {ret}% ---")
        print(f"CHAIR_i (Instance Hallucination): {chair_i*100:.2f}% (Tỷ lệ object bị ảo giác)")
        print(f"CHAIR_s (Sentence Hallucination): {chair_s*100:.2f}% (Tỷ lệ câu chứa ảo giác)")
        print(f"Object Precision: {obj_precision*100:.2f}% | Object Recall: {obj_recall*100:.2f}%")
        print(f"Số object sinh ra TB/video: {mean_obj_count:.1f} objects")

    return summary_chair

if __name__ == "__main__":
    vid_res = evaluate_vidhalluc()
    chair_res = evaluate_videochair()

    # Lưu kết quả ra file JSON
    out_file = Path("results/6_10_26/hallucination_performance_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"vidhalluc": vid_res, "videochair": chair_res}, f, indent=2)
    print(f"\nĐã lưu toàn bộ số liệu vào {out_file}!")
