"""Extract attention distribution profiles (Mass, Density, Layer breakdown) for balanced config."""

import json
from pathlib import Path
import numpy as np

from src.analyze.Validate_Sparrow_hypothesises.attention_artifacts import load_attention_shard
from src.analyze.Validate_Sparrow_hypothesises.analyze_dflash_visualcut_attention import (
    _summarize_attention_record, _valid_row
)
from src.analyze.Validate_Sparrow_hypothesises.visualcut_report_utils import sample_evenly

CONFIG_NAME = "dflash_3layer_1_9_17_25_33"
BUDGETS = (400, 3000, 13000, 25000)

def main():
    root = Path("results/visualcut_attention/vdc50_visualcut_balanced_20261029/tasks")
    cases_by_id = {}
    for p in sorted(root.glob("*/cases.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if _valid_row(row):
                    cases_by_id[str(row["row_id"])] = {
                        "row_id": row["row_id"],
                        "checkpoint_name": row["checkpoint_name"],
                        "sample_id": row["sample_id"],
                        "target_visual_tokens": row["target_visual_tokens"],
                        "visual_condition": row["visual_condition"],
                        "prompt_regions_original": row.get("prompt_regions_original"),
                    }

    print(f"Loaded {len(cases_by_id)} valid cases for attention mapping.")

    results = {CONFIG_NAME: {}}

    for budget in BUDGETS:
        print(f"\nProcessing attention profiles for budget {budget}...")
        budget_dirs = [
            root / f"{CONFIG_NAME}_v{budget}_chunk0",
            root / f"{CONFIG_NAME}_v{budget}_chunk1",
        ]

        all_shards = []
        for bdir in budget_dirs:
            shards = sorted((bdir / "attention").glob("*.pt.zst"))
            all_shards.extend(sample_evenly(shards, 4))

        print(f"  Sampled {len(all_shards)} shards across chunks.")

        records_by_cond = {}
        for s_path in all_shards:
            for record in load_attention_shard(s_path):
                case_id = str(record.get("case_id", ""))
                row = cases_by_id.get(case_id)
                if row is None:
                    continue
                cond = row.get("visual_condition")
                try:
                    summary, _ = _summarize_attention_record(row, record)
                except Exception:
                    continue

                mean_mass = np.mean(summary["region_mass_by_head"], axis=0)
                mean_dens = np.mean(summary["region_density_by_head"], axis=0)
                norm_ent = np.mean(summary["normalized_entropy_by_head"])
                raw_ent = np.mean(summary["entropy_by_head"])
                layer = summary["layer_index"]

                vis_mass = float(mean_mass[0])
                vis_dens = float(mean_dens[0])
                prompt_text_mass = float(mean_mass[1] + mean_mass[2] + mean_mass[3] + mean_mass[4])
                q_text_dens = float(mean_dens[1])
                gen_text_mass = float(mean_mass[5] + mean_mass[6])
                curr_query_mass = float(mean_mass[7])

                gen_keys = summary["region_key_counts"].get("generated_target_hidden", 0) + summary["region_key_counts"].get("cached_draft", 0)
                gen_text_dens = (gen_text_mass / gen_keys) if gen_keys > 0 else 0.0

                records_by_cond.setdefault(cond, []).append({
                    "layer": layer,
                    "vis_mass": vis_mass,
                    "vis_dens": vis_dens,
                    "prompt_text_mass": prompt_text_mass,
                    "q_text_dens": q_text_dens,
                    "gen_text_mass": gen_text_mass,
                    "gen_text_dens": gen_text_dens,
                    "curr_query_mass": curr_query_mass,
                    "norm_entropy": float(norm_ent),
                    "entropy": float(raw_ent),
                })

        budget_summary = {}
        for cond, recs in records_by_cond.items():
            vis_masses = [r["vis_mass"] for r in recs]
            vis_dens = [r["vis_dens"] for r in recs if r["vis_dens"] > 0]
            prompt_text_masses = [r["prompt_text_mass"] for r in recs]
            q_dens = [r["q_text_dens"] for r in recs if r["q_text_dens"] > 0]
            gen_text_masses = [r["gen_text_mass"] for r in recs]
            gen_dens = [r["gen_text_dens"] for r in recs if r["gen_text_dens"] > 0]
            curr_query_masses = [r["curr_query_mass"] for r in recs]
            ents = [r["norm_entropy"] for r in recs]

            layer_vis_mass = {}
            for l in (0, 1, 2):
                l_recs = [r["vis_mass"] for r in recs if r["layer"] == l]
                if l_recs:
                    layer_vis_mass[f"layer_{l}"] = float(np.mean(l_recs))

            m_vis = float(np.mean(vis_masses)) if vis_masses else 0.0
            m_ptext = float(np.mean(prompt_text_masses)) if prompt_text_masses else 0.0
            m_gtext = float(np.mean(gen_text_masses)) if gen_text_masses else 0.0
            m_query = float(np.mean(curr_query_masses)) if curr_query_masses else 0.0

            d_vis = float(np.mean(vis_dens)) if vis_dens else 0.0
            d_q = float(np.mean(q_dens)) if q_dens else 0.0
            d_gen = float(np.mean(gen_dens)) if gen_dens else 0.0

            budget_summary[cond] = {
                "count": len(recs),
                "vis_mass_mean": m_vis,
                "prompt_text_mass_mean": m_ptext,
                "gen_text_mass_mean": m_gtext,
                "curr_query_mass_mean": m_query,
                "total_mass_check": m_vis + m_ptext + m_gtext + m_query,
                "vis_dens_mean": d_vis,
                "q_dens_mean": d_q,
                "gen_dens_mean": d_gen,
                "dens_ratio_q_vs_vis": (d_q / d_vis) if d_vis > 0 else 0.0,
                "dens_ratio_gen_vs_vis": (d_gen / d_vis) if d_vis > 0 else 0.0,
                "dens_ratio_gen_vs_q": (d_gen / d_q) if d_q > 0 else 0.0,
                "norm_entropy_mean": float(np.mean(ents)) if ents else 0.0,
                "layer_vis_mass": layer_vis_mass,
            }
        results[CONFIG_NAME][str(budget)] = budget_summary
        print(f"  Budget {budget}: summarized {sum(len(r) for r in records_by_cond.values())} attention records.")

    out_file = Path("results/visualcut_attention/vdc50_visualcut_balanced_20261029/summary_reports_data/attention_profiles.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully saved attention profiles to {out_file}")

if __name__ == "__main__":
    main()
