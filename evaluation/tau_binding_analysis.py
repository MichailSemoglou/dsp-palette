"""
Quantify how often the hard τ_dist constraint binds during greedy
selection (DSP method, τ_dist = 10.0, N = 75 test images).

For each test image and each greedy-expansion step (slots 2–5), replays the
candidate-loop and counts how many candidates were rejected because
min_ΔE2000(candidate, current_palette) < τ_dist.

Also asserts, image-by-image, that A1 (τ=0) and A2 (τ=10) palettes are
identical, confirming the hard floor never alters the selected palette on
this corpus.

Outputs
-------
results/ablation/tau_binding.csv   Per-image counts of τ rejections.
results/ablation/tau_binding_summary.json  Aggregate statistics.

Usage
-----
    python -m evaluation.tau_binding_analysis \
        --manifest   corpus/manifest.json \
        --corpus-root corpus/ \
        --ablation-dir results/ablation/raw/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from dsp.metrics import delta_e2000, srgb_to_lab
from dsp.selector import _min_de_to_palette, _quantize_to_candidates


TAU_DIST = 10.0
N_PALETTE = 5
MAX_CANDIDATES = 256


def _count_tau_rejections_for_image(
    img_path: Path,
    tau_dist: float = TAU_DIST,
    n: int = N_PALETTE,
) -> dict:
    """Replay the greedy loop for one image; return rejection counts per slot."""
    img = Image.open(img_path).convert("RGB")
    candidates_rgb, freqs = _quantize_to_candidates(img, MAX_CANDIDATES)
    candidates_lab = srgb_to_lab(candidates_rgb.astype(np.float64))
    K = len(candidates_rgb)

    # Initialise with highest-frequency candidate (slot 1 — no constraint check)
    palette_indices: list[int] = [int(np.argmax(freqs))]
    in_palette = set(palette_indices)

    # Per-slot counts
    rejections_per_slot: list[int] = [0]  # slot 1 has 0 rejections (no constraint)
    total_candidates_evaluated_per_slot: list[int] = [1]

    for slot in range(2, n + 1):
        palette_lab_list = [candidates_lab[i] for i in palette_indices]
        rejected = 0
        best_score = -math.inf
        best_idx: Optional[int] = None

        for idx in range(K):
            if idx in in_palette:
                continue
            min_de = _min_de_to_palette(candidates_lab[idx], palette_lab_list)
            if min_de < tau_dist:
                rejected += 1
                continue
            score = math.log(freqs[idx] + 1e-12) + min_de  # alpha=beta=1
            if score > best_score:
                best_score = score
                best_idx = idx

        rejections_per_slot.append(rejected)
        evaluated = sum(1 for i in range(K) if i not in in_palette)
        total_candidates_evaluated_per_slot.append(evaluated)

        if best_idx is None:
            # Relaxation path (rare) — still record zero tau-binding
            # since the constraint was not satisfied by any candidate
            remaining = [i for i in range(K) if i not in in_palette]
            if remaining:
                best_idx = max(
                    remaining,
                    key=lambda i: _min_de_to_palette(candidates_lab[i], palette_lab_list),
                )
            else:
                break

        palette_indices.append(best_idx)
        in_palette.add(best_idx)

    return {
        "rejections_per_slot": rejections_per_slot,
        "candidates_per_slot": total_candidates_evaluated_per_slot,
        "total_rejections": sum(rejections_per_slot),
        "n_slots": len(palette_indices),
    }


def _assert_a1_equals_a2(ablation_dir: Path) -> None:
    """Assert palette identity for A1 vs A2 on all 75 test images."""
    mismatches = []
    total = 0
    for jpath in sorted(ablation_dir.glob("coco_[0-9]*.json")):
        with open(jpath) as f:
            d = json.load(f)
        a1 = sorted(d["methods"]["ablation_score_only"]["palette_hex"])
        a2 = sorted(d["methods"]["ablation_score_constraint"]["palette_hex"])
        total += 1
        if a1 != a2:
            mismatches.append(d["image_id"])
    if mismatches:
        raise AssertionError(
            f"A1 != A2 on {len(mismatches)} images: {mismatches[:5]}"
        )
    print(f"PASS  A1 == A2 palettes on all {total} test images.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Count τ_dist constraint rejections during DSP greedy selection.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--ablation-dir", required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    corpus_root   = Path(args.corpus_root)
    ablation_dir  = Path(args.ablation_dir)

    # Load test-set image list (val2017, dev=False)
    with open(manifest_path) as f:
        manifest = json.load(f)
    test_entries = [
        e for e in manifest
        if e.get("subset") == "photographs" and not e.get("dev")
    ]
    print(f"Test set: {len(test_entries)} images.")

    # 1. Assert A1 == A2 image-by-image
    _assert_a1_equals_a2(ablation_dir)

    # 2. Count rejections per image
    rows = []
    n_images_with_zero_rejections = 0
    n_images_with_any_rejection   = 0

    for i, entry in enumerate(test_entries, 1):
        img_path = corpus_root / entry.get("subset", "photographs") / entry["filename"]
        if not img_path.exists():
            print(f"  SKIP {entry['id']}: image not found at {img_path}")
            continue

        result = _count_tau_rejections_for_image(img_path, tau_dist=TAU_DIST)

        if result["total_rejections"] == 0:
            n_images_with_zero_rejections += 1
        else:
            n_images_with_any_rejection += 1

        row = {"image_id": entry["id"]}
        for slot_idx, (rej, cand) in enumerate(
            zip(result["rejections_per_slot"], result["candidates_per_slot"]), 1
        ):
            row[f"slot{slot_idx}_rejections"] = rej
            row[f"slot{slot_idx}_candidates"] = cand
        row["total_rejections"] = result["total_rejections"]
        rows.append(row)

        if i % 15 == 0:
            print(f"  [{i:3d}/{len(test_entries)}]  "
                  f"rejections so far: {sum(r['total_rejections'] for r in rows)}")

    df = pd.DataFrame(rows)

    # 3. Output CSV
    out_dir = ablation_dir.parent
    csv_path = out_dir / "tau_binding.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # 4. Summary
    total_rejections_all   = int(df["total_rejections"].sum())
    images_with_rejection  = int((df["total_rejections"] > 0).sum())
    total_images           = len(df)

    # Per-slot aggregate rejection counts (slots 2–5)
    slot_totals = {}
    for slot in range(1, N_PALETTE + 1):
        col = f"slot{slot}_rejections"
        if col in df.columns:
            slot_totals[slot] = int(df[col].sum())

    summary = {
        "tau_dist": TAU_DIST,
        "n_palette": N_PALETTE,
        "n_images": total_images,
        "total_rejections_across_all_images_and_slots": total_rejections_all,
        "images_with_at_least_one_rejection": images_with_rejection,
        "images_with_zero_rejections": total_images - images_with_rejection,
        "rejections_per_slot_totals": slot_totals,
        "a1_equals_a2_all_images": True,
    }
    summary_path = out_dir / "tau_binding_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {summary_path}")

    # 5. Console report
    sep = "=" * 68
    print(f"\n{sep}")
    print("TAU CONSTRAINT BINDING ANALYSIS")
    print(f"τ_dist = {TAU_DIST},  n = {N_PALETTE},  N = {total_images} test images")
    print(sep)
    print(f"A1 == A2 palettes          : CONFIRMED on all {total_images} images")
    print(f"Total τ rejections          : {total_rejections_all}")
    print(f"Images with ≥1 rejection   : {images_with_rejection} of {total_images}")
    print(f"Images with 0 rejections   : {total_images - images_with_rejection} of {total_images}")
    print()
    print("Rejections per greedy slot (summed across all images):")
    for slot, total in slot_totals.items():
        print(f"  Slot {slot}: {total} rejections"
              + ("  [no constraint at slot 1]" if slot == 1 else ""))
    print()
    if total_rejections_all == 0:
        print("Interpretation: τ_dist never rejected any candidate on this corpus.")
        print("  The scoring term β·min_ΔE2000 implicitly drives separation;")
        print("  the hard floor at τ=10 is never the binding constraint.")
    else:
        pct = 100 * images_with_rejection / total_images
        print(f"Interpretation: τ_dist bound on {images_with_rejection} of "
              f"{total_images} images ({pct:.1f}%).")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
