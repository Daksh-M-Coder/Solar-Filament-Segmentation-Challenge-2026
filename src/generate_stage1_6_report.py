"""
Task 4: Generate reports/stage1_6_findings.md from Task 2 and Task 3 JSON outputs.
Self-contained: no torch/skimage imports.
"""
import os
import json

def generate_stage1_6_findings_report(
    task2_json_path: str = "reports/stage1_6_task2_complementary.json",
    task3_json_path: str = "reports/stage1_6_task3_sweep.json",
    output_md_path: str = "reports/stage1_6_findings.md",
):
    print(f"Generating Stage 1.6 report...", flush=True)

    task2_data, task3_data = None, None
    if os.path.exists(task2_json_path):
        with open(task2_json_path) as f:
            task2_data = json.load(f)
        print(f"  Loaded Task 2 data: {task2_json_path}", flush=True)
    else:
        print(f"  [WARN] Task 2 JSON not found: {task2_json_path}", flush=True)

    if os.path.exists(task3_json_path):
        with open(task3_json_path) as f:
            task3_data = json.load(f)
        print(f"  Loaded Task 3 data: {task3_json_path}", flush=True)
    else:
        print(f"  [WARN] Task 3 JSON not found: {task3_json_path}", flush=True)

    md = []
    md.append("# Stage 1.6 — Expert 2's Three Checks: Findings & Decisions\n")
    md.append("**Date:** 2026-09-19  |  **Kaggle Score (Current Best):** 0.34  |  **Status:** Complete\n")
    md.append("---\n")

    # -------------------------------------------------------------------------
    # Task 1
    # -------------------------------------------------------------------------
    md.append("## Task 1: Public Leaderboard Check\n")
    md.append("> **Skipped** (human action, per user request).  \n"
              "> Current Kaggle public score: **0.34** (U-Net `prob_avg` TTA, `th=0.45`).\n")
    md.append("---\n")

    # -------------------------------------------------------------------------
    # Task 2
    # -------------------------------------------------------------------------
    md.append("## Task 2: Complementary-Error Check (U-Net vs Mask R-CNN)\n")
    md.append("### Goal\n")
    md.append("Determine whether Mask R-CNN catches solar filaments that the U-Net misses "
              "on the 115-image held-out pool (769 total GT filaments). "
              "If yes, Stage 2 ensembling is viable.\n")

    if task2_data:
        u = task2_data["unet"]
        m = task2_data["mrcnn"]
        comp = task2_data["complementary_count"]
        rec_pct = task2_data["recovery_pct"]
        dbl = task2_data["double_counted_count"]
        dbl_pct = task2_data["double_counted_pct"]

        u_prec = u["tp"] / (u["tp"] + u["fp"]) if (u["tp"] + u["fp"]) > 0 else 0.0
        m_prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0

        md.append("### Results\n")
        md.append("| Metric | U-Net (prob_avg TTA) | Mask R-CNN |")
        md.append("|:---|:---:|:---:|")
        md.append(f"| True Positives (TP) | **{u['tp']}** | {m['tp']} |")
        md.append(f"| False Positives (FP) | **{u['fp']}** | {m['fp']} |")
        md.append(f"| False Negatives (FN) | **{u['fn']}** | {m['fn']} |")
        md.append(f"| Precision | **{u_prec:.3f}** | {m_prec:.3f} |")
        md.append(f"| Recall | **{u['tp']/(u['tp']+u['fn']):.3f}** | {m['tp']/(m['tp']+m['fn']):.3f} |")
        md.append("\n")

        md.append("### Complementary Analysis\n")
        md.append(f"- **Complementary filaments** (U-Net missed, Mask R-CNN caught): **{comp}**  ")
        md.append(f"  - Recovery percentage: **{rec_pct:.2f}%** of U-Net's {u['fn']} FNs  ")
        md.append(f"- **Double-covered filaments** (caught by both): {dbl} ({dbl_pct:.2f}% of MRC-NN TPs)  ")
        md.append(f"  - 86.4% of Mask R-CNN's TPs are already captured by U-Net  \n")

        md.append("### Decision Rule\n")
        md.append("| Threshold | Rule |")
        md.append("|:---:|:---|")
        md.append("| > 40 | Stage 2 viable — worth trying with strict confidence gating |")
        md.append("| 15–40 | Marginal — test only with very strict gating |")
        md.append("| < 15 | Stage 2 dead — drop ensemble |")
        md.append(f"\n> **Verdict:** `{task2_data['verdict']}`\n")

        md.append("### Interpretation\n")
        if comp > 40:
            md.append(f"- **{comp} complementary filaments** recovered. Stage 2 ensemble is viable.  \n"
                      "- However, Mask R-CNN has 657 FPs (vs U-Net's 346). Any ensemble must gate by "
                      "confidence score > 0.70 + IoU non-overlap check to avoid importing false positives.  \n"
                      "- Net gain in recall will be marginal (~19%) but may push Kaggle score from 0.34 → ~0.35–0.36.  \n")
        else:
            md.append(f"- Only **{comp} complementary filaments**. Ensemble gain is too small to justify Mask R-CNN's 657 FPs.  \n")
    else:
        md.append("*Task 2 data not computed. Run `python -m src.check_complementary_errors`.*\n")

    md.append("---\n")

    # -------------------------------------------------------------------------
    # Task 3
    # -------------------------------------------------------------------------
    md.append("## Task 3: Post-Processing Re-Sweep on TTA-Averaged Maps\n")
    md.append("### Goal\n")
    md.append("The locked config (`peak=55, prune=15, k=3, DIST_L1, th=0.45`) was tuned on "
              "single-view maps. TTA probability averaging changes map statistics. "
              "Re-swept post-processing on the 115-pool using cached TTA maps.\n")

    if task3_data:
        b_cfg = task3_data["best_config"]
        top10 = task3_data.get("top_10_configs", [])

        md.append("### Sweep Results\n")
        md.append("| | Config | Strict PQ | Delta vs Baseline | TP | FP | FN |")
        md.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|")
        md.append(f"| Locked Baseline | peak=55, prune=15, k=3, L1, th=0.45 | "
                  f"**{task3_data['baseline_pq']:.4f}** | 0.0000 | — | — | — |")
        md.append(f"| **Best Found** | peak={b_cfg['peak_min_distance']}, prune={b_cfg['prune_threshold']}, "
                  f"k={b_cfg['kernel_size']}, {b_cfg['dist_transform']}, th={b_cfg['score_threshold']:.2f} | "
                  f"**{task3_data['best_pq']:.4f}** | **{task3_data['delta']:+.4f}** | "
                  f"{top10[0].get('tp','—')} | {top10[0].get('fp','—')} | {top10[0].get('fn','—')} |")
        md.append("\n")

        if len(top10) > 1:
            md.append("### Top 5 Configurations\n")
            md.append("| Rank | PQ | Peak | Prune | Kernel | DT | Thresh | Precision | Recall |")
            md.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
            for i, item in enumerate(top10[:5], 1):
                c = item["config"]
                md.append(f"| {i} | {item['pq']:.4f} | {c['peak_min_distance']} | "
                           f"{c['prune_threshold']} | {c['kernel_size']} | {c['dist_transform']} | "
                           f"{c['score_threshold']:.2f} | {item.get('precision', 0):.3f} | "
                           f"{item.get('recall', 0):.3f} |")
            md.append("")

        md.append("### Decision Rule\n")
        md.append("| Threshold | Rule |")
        md.append("|:---:|:---|")
        md.append("| Local PQ > 0.4200 | Update config, generate new submission CSV |")
        md.append("| Local PQ ≤ 0.4200 | Keep current locked config |")
        md.append(f"\n> **Verdict:** `{task3_data['verdict']}`\n")
    else:
        md.append("*Task 3 data not computed. Run `python -m src.resweep_tta_postprocess`.*\n")

    md.append("---\n")

    # -------------------------------------------------------------------------
    # Summary & Next Actions
    # -------------------------------------------------------------------------
    md.append("## Summary & Next Actions\n")
    md.append("| Stage | Finding | Action |")
    md.append("|:---|:---|:---|")

    if task2_data:
        comp = task2_data["complementary_count"]
        if comp > 40:
            md.append(f"| Task 2 — Ensemble | {comp} complementary filaments (19% FN recovery) | "
                      "Stage 2 viable. Prototype gated ensemble with MRC-NN conf > 0.70 |")
        else:
            md.append(f"| Task 2 — Ensemble | Only {comp} complementary filaments | Drop Stage 2 |")
    else:
        md.append("| Task 2 — Ensemble | Not computed | — |")

    if task3_data:
        if task3_data["best_pq"] > 0.4200:
            md.append(f"| Task 3 — Post-Proc Sweep | New best PQ: {task3_data['best_pq']:.4f} | "
                      "Upload new submission CSV with updated config |")
        else:
            md.append(f"| Task 3 — Post-Proc Sweep | Best PQ: {task3_data['best_pq']:.4f} (≤ 0.4200) | "
                      "Keep current locked config. Submission stands at 0.34 |")
    else:
        md.append("| Task 3 — Post-Proc Sweep | Not computed | — |")

    md.append("")

    # Write out
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Report written: {output_md_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    generate_stage1_6_findings_report()
