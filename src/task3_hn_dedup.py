"""
Task 3 – Dedup of hard_negatives_manifest.json
===============================================
Group entries in hard_negatives_manifest.json by (source_image_filename,
rounded bbox or (area, confidence)). Keep one entry per group (highest
confidence). Save deduped manifest as hard_negatives_v1_dedup.json.

Report:
  (a) original count
  (b) unique count
  (c) unique source images spanned
  (d) tile area distribution after dedup (min/median/max)
"""

import json
import os
import sys
import numpy as np

MANIFEST_IN  = "data/hard_negatives_v1/hard_negatives_manifest.json"
MANIFEST_OUT = "data/hard_negatives_v1/hard_negatives_v1_dedup.json"


def round_to_nearest(x: float, step: int = 50) -> int:
    """Round float area to nearest step (bucket-based dedup key)."""
    return int(round(x / step) * step)


def main():
    with open(MANIFEST_IN) as f:
        data = json.load(f)

    original_entries = data["manifest"]
    original_count   = data.get("total_hard_negatives_found", len(original_entries))

    print(f"Original entries in manifest (saved tiles): {len(original_entries)}")
    print(f"Total HN found (before top-500 filter):     {original_count}")

    # ── Dedup strategy ───────────────────────────────────────────────────────
    # Key = (source_image, area_bucket, confidence_bucket)
    # area is rounded to nearest 50px; confidence to nearest 0.05
    # If two tiles share the same source_image AND area_bucket AND conf_bucket
    # they are considered duplicates. Keep the one with highest confidence.
    groups: dict = {}

    for entry in original_entries:
        src        = entry["source_image"]
        area_key   = round_to_nearest(entry["area"], step=50)
        conf_key   = round(entry["confidence"] / 0.05) * 0.05  # 0.05 buckets
        group_key  = (src, area_key, round(conf_key, 2))

        if group_key not in groups:
            groups[group_key] = entry
        else:
            # Keep highest confidence
            if entry["confidence"] > groups[group_key]["confidence"]:
                groups[group_key] = entry

    deduped = list(groups.values())
    # Re-sort by original rank / score
    deduped.sort(key=lambda e: e.get("score", e.get("confidence", 0)), reverse=True)

    unique_count   = len(deduped)
    unique_images  = len(set(e["source_image"] for e in deduped))
    areas          = [e["area"] for e in deduped]
    area_min       = int(np.min(areas))
    area_median    = float(np.median(areas))
    area_max       = int(np.max(areas))

    # ── Save ─────────────────────────────────────────────────────────────────
    out_data = {
        "original_saved_count":     len(original_entries),
        "total_hn_found_before_top500": original_count,
        "unique_count":             unique_count,
        "unique_source_images":     unique_images,
        "area_distribution": {
            "min":    area_min,
            "median": area_median,
            "max":    area_max,
        },
        "dedup_strategy": "group by (source_image, area±50px, confidence±0.05); keep highest confidence",
        "manifest": deduped,
    }
    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
    with open(MANIFEST_OUT, "w") as f:
        json.dump(out_data, f, indent=2)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("DEDUP SUMMARY")
    print("="*55)
    print(f"  (a) Original count (saved tiles)  : {len(original_entries)}")
    print(f"      Total HN found (all 1,539)    : {original_count}")
    print(f"  (b) Unique count after dedup       : {unique_count}")
    print(f"      Removed as duplicates          : {len(original_entries) - unique_count}")
    print(f"  (c) Unique source images spanned   : {unique_images}")
    print(f"  (d) Area distribution (px)")
    print(f"      Min    : {area_min}")
    print(f"      Median : {area_median:.0f}")
    print(f"      Max    : {area_max}")
    print(f"\n  Saved deduped manifest: {MANIFEST_OUT}")
    print("="*55)


if __name__ == "__main__":
    main()
