"""
Descriptive statistics — WCAG fire count and Surface/On-Surface
AA pass rate per method across N=75 test images.

(a) WCAG replacement fire count for the stored DSP results.
    (These counts are also printed by ablation_report.py but this script
    provides a standalone record and checks the per-method baselines.)

(b) Surface / On-Surface WCAG AA pass rate per method.
    For every result JSON, we assign semantic roles (surface + on-surface)
    via ``assign_roles`` and check whether that pair achieves WCAG AA
    contrast ≥ 4.5:1.  Results are reported per method.

Output
------
results/ablation/descriptive_aa.csv  — per-image per-method AA flag
results/ablation/descriptive_aa_summary.json — aggregate pass rates

Usage
-----
    python -m evaluation.descriptive_stats \
        --results-dir results/raw/ \
        --manifest corpus/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dsp.metrics import srgb_to_lab, wcag_contrast
from dsp.roles import assign_roles


WCAG_AA_THRESHOLD = 4.5
METHODS = ["dsp", "kmeans_lab", "kmeans_rgb", "median_cut"]


def _load_test_ids(manifest_path: Path) -> set[str]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {e["id"] for e in manifest if e.get("subset") == "photographs" and not e.get("dev")}


def _aa_pass_for_method(method_data: dict) -> bool | None:
    """Return True if surface/on-surface pair is WCAG AA, False if not, None on error."""
    if "error" in method_data or not method_data:
        return None
    palette_rgb_raw = method_data.get("palette_rgb")
    freqs_raw       = method_data.get("frequencies")
    if palette_rgb_raw is None or freqs_raw is None:
        return None
    try:
        palette_rgb = np.array(palette_rgb_raw, dtype=np.uint8)
        palette_lab = srgb_to_lab(palette_rgb)
        frequencies = list(freqs_raw)
        roles = assign_roles(palette_rgb, palette_lab, frequencies, mode="auto")
        if roles.surface is None or roles.on_surface is None:
            return None
        contrast = wcag_contrast(
            palette_rgb[roles.on_surface],
            palette_rgb[roles.surface],
        )
        return contrast >= WCAG_AA_THRESHOLD
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute Surface/On-Surface WCAG AA pass rate per method.",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)

    results_dir   = Path(args.results_dir)
    manifest_path = Path(args.manifest)

    test_ids = _load_test_ids(manifest_path)
    print(f"Test set: {len(test_ids)} image IDs.")

    rows = []
    wcag_fires = {m: {"replacement_applied": 0, "distinctness_compromised": 0, "n": 0}
                  for m in METHODS}

    for jpath in sorted(results_dir.glob("coco_[0-9]*.json")):
        with open(jpath) as f:
            d = json.load(f)
        image_id = d.get("image_id")
        if image_id not in test_ids:
            continue

        methods_data = d.get("methods", {})
        row: dict = {"image_id": image_id}

        for method in METHODS:
            mdata = methods_data.get(method, {})
            if not mdata or "error" in mdata:
                row[f"{method}_aa_pass"] = None
                continue

            # WCAG fire counts (DSP only)
            if method == "dsp":
                wcag_fires["dsp"]["replacement_applied"] += int(bool(mdata.get("wcag_replacement_applied")))
                wcag_fires["dsp"]["distinctness_compromised"] += int(bool(mdata.get("wcag_distinctness_compromised")))
                wcag_fires["dsp"]["n"] += 1

            aa_pass = _aa_pass_for_method(mdata)
            row[f"{method}_aa_pass"] = aa_pass

        rows.append(row)

    df = pd.DataFrame(rows)

    out_dir = results_dir.parent / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "descriptive_aa.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # Aggregate AA pass rates
    aa_summary: dict = {}
    for method in METHODS:
        col = f"{method}_aa_pass"
        if col in df.columns:
            valid = df[col].dropna()
            n_pass = int(valid.sum())
            n_total = len(valid)
            aa_summary[method] = {
                "aa_pass": n_pass,
                "n_images": n_total,
                "aa_pass_rate": round(n_pass / n_total, 4) if n_total > 0 else None,
            }

    summary = {
        "n_test_images": len(df),
        "wcag_aa_threshold": WCAG_AA_THRESHOLD,
        "surface_on_surface_aa_pass_rates": aa_summary,
        "dsp_wcag_replacement_fires": wcag_fires["dsp"],
    }
    summary_path = out_dir / "descriptive_aa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved → {summary_path}")

    # Console report
    sep = "=" * 68
    print(f"\n{sep}")
    print("DESCRIPTIVE STATISTICS — WCAG AA")
    print(f"N = {len(df)} test images")
    print(sep)

    print(f"\n(a) DSP WCAG post-selection fires (N={wcag_fires['dsp']['n']} images):")
    ra = wcag_fires['dsp']['replacement_applied']
    dc = wcag_fires['dsp']['distinctness_compromised']
    print(f"  wcag_replacement_applied    : {ra} of {wcag_fires['dsp']['n']}")
    print(f"  wcag_distinctness_compromised: {dc} of {wcag_fires['dsp']['n']}")

    print(f"\n(b) Surface/On-Surface WCAG AA pass rate per method:")
    for method in METHODS:
        if method not in aa_summary:
            continue
        d_m = aa_summary[method]
        pct = 100.0 * d_m["aa_pass_rate"] if d_m["aa_pass_rate"] is not None else float("nan")
        print(f"  {method:18s}: {d_m['aa_pass']:3d} / {d_m['n_images']:3d}  ({pct:.1f}%)")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
