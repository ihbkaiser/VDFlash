"""Trích xuất và chuẩn hóa chi tiết từng Draft Iteration từ kết quả MSD.

Script này đọc file kết quả MSD (.jsonl) và bóc tách cấu trúc acceptance_by_position
thành bảng dữ liệu chi tiết từng vòng suy đoán (iteration/round), bao gồm:
- sample_id, retention_percentage
- round_index (thứ tự vòng suy đoán)
- t0_token, t0_token_id, t0_accepted (kết quả cửa ải token đầu tiên)
- proposed_tokens (toàn bộ chuỗi draft đề xuất trong round)
- accepted_tokens (chuỗi token được target chấp thuận)
- accepted_length (số lượng token draft được accept)
- fallback_token (token do target sinh ra để sửa sai hoặc tiếp tục)
- is_zero_acc (vòng suy đoán có bị hủy hoàn toàn không)
- difficulty_category (phân loại DỄ, KHÓ, hoặc NỘI DUNG theo chuẩn định nghĩa)

Usage:
    python scripts/extract_draft_iterations.py \
      --input results/msd_eval_20261007/vidhalluc/msd_retention_vidhalluc.jsonl \
      --output results/msd_eval_20261007/vidhalluc/draft_iterations_detailed.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

EASY_PREFIXES = {
    " a", " an", " the", " this", " that", " these", " those",
    " of", " in", " on", " at", " with", " by", " from", " to", " for",
    " is", " are", " was", " were", " be", " been", " being",
    " variety", " objects", " scenes", " including", " per", "ched",
    " flowers", " birds", " actions", " followed", " background",
}

HARD_PREFIXES = {
    "The", "In", "A", "On", "At", "There", "This", "It", "After", "During",
    " then", " and", " but", " while", " where", " as", " because",
    " video", " scene", " clip", " camera", " view", " shot",
    " begins", " starts", " features", " shows", " captures", " presents",
}


def classify_t0(token_text: str) -> str:
    cleaned = token_text.strip()
    if token_text in EASY_PREFIXES or cleaned.lower() in {
        "a", "an", "the", "this", "that", "these", "those", "of", "in", "on", "at", "with", "by", "from", "to", "for"
    }:
        return "EASY"
    if token_text in HARD_PREFIXES or cleaned in {
        "The", "In", "A", "On", "At", "There", "This", "It", "After", "During", "then", "and", "but", "while"
    }:
        return "HARD"
    return "CONTENT"


def extract_iterations(input_path: str | Path) -> list[dict[str, Any]]:
    iterations = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id", "")
            retention = float(row.get("retention_percentage", 100.0))
            lossless = row.get("lossless", True)
            acc_pos = row.get("acceptance_by_position", [])

            # Nhóm các bản ghi theo round_index
            rounds_map: dict[int, list[dict[str, Any]]] = {}
            for rec in acc_pos:
                rnd = rec.get("round_index")
                if rnd is not None:
                    rounds_map.setdefault(rnd, []).append(rec)

            for rnd in sorted(rounds_map.keys()):
                records = rounds_map[rnd]
                # Lọc các token do draft đề xuất (role == 'accepted_proposal' hoặc rejected token)
                # Trong format của MSD: các token có proposed == 1 là từ draft proposal;
                # token có role == 'fallback' là token target chốt hạ ở cuối round.
                proposals = [r for r in records if r.get("role") == "accepted_proposal"]
                fallbacks = [r for r in records if r.get("role") == "fallback"]

                fallback_token = fallbacks[0].get("target_token_text", "") if fallbacks else ""
                fallback_id = fallbacks[0].get("target_token_id") if fallbacks else None

                # Token đầu tiên t_0 của round
                if proposals:
                    # Nếu có ít nhất 1 proposal được accept
                    t0_rec = proposals[0]
                    t0_text = t0_rec.get("speculative_token_text", "")
                    t0_id = t0_rec.get("speculative_token_id")
                    t0_accepted = 1
                elif fallbacks and fallbacks[0].get("speculative_token_text") is not None:
                    # Nếu không có proposal nào được accept (Zero-acc round), t0 là token draft bị trượt
                    t0_rec = fallbacks[0]
                    t0_text = t0_rec.get("speculative_token_text", "")
                    t0_id = t0_rec.get("speculative_token_id")
                    t0_accepted = 0
                else:
                    continue

                accepted_tokens_list = [r.get("speculative_token_text", "") for r in proposals]
                accepted_length = len(accepted_tokens_list)
                is_zero_acc = (accepted_length == 0)
                category = classify_t0(t0_text)

                iterations.append({
                    "sample_id": sample_id,
                    "retention_percentage": retention,
                    "round_index": rnd,
                    "t0_token": t0_text,
                    "t0_token_id": t0_id,
                    "t0_accepted": t0_accepted,
                    "accepted_length": accepted_length,
                    "is_zero_acc": 1 if is_zero_acc else 0,
                    "difficulty_category": category,
                    "accepted_tokens_text": "".join(accepted_tokens_list),
                    "fallback_token": fallback_token,
                    "fallback_token_id": fallback_id,
                    "lossless": 1 if lossless else 0,
                })

    return iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract draft iteration details from MSD results")
    parser.add_argument("--input", required=True, help="Input JSONL file with MSD traces")
    parser.add_argument("--output", required=True, help="Output CSV file for detailed iterations")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading MSD traces from {input_path}...")
    iterations = extract_iterations(input_path)
    print(f"Extracted {len(iterations)} draft iterations.")

    if iterations:
        fieldnames = list(iterations[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(iterations)
        print(f"Saved detailed iteration table to {output_path}")
    else:
        print("No iterations found in input file.")


if __name__ == "__main__":
    main()
