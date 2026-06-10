"""
wcag_recheck.py — Re-verify ALL WCAG-derived quantities after the
luminance boundary-value bug fix.

Bug summary
-----------
Both ``relative_luminance`` and ``srgb_to_lab`` in ``dsp/metrics.py``
previously detected input range with ``if arr.max() > 1.0``.  A palette colour
whose maximum channel value is exactly 1 (e.g. uint8 ``[1, 0, 0]`` = near-black
#010000) failed the guard and was misread as a normalised pure-red float,
returning L ≈ 0.2126 instead of the correct L ≈ 0.000065.  Stored
``wcag_aa_coverage`` values computed by the runners are therefore wrong for
any image whose palette contains such a colour.

This script:
  1. Loads ALL stored result JSONs (results/raw/ for 4 main methods,
     results/ablation/raw/ for 6 ablation methods).
  2. Filters to the 75-image test set (subset="photographs", dev=False)
     using the corpus manifest for results/raw/, and uses all 75 ablation
     JSONs for results/ablation/raw/.
  3. Also processes the 25-image train set (subset="photographs_train").
  4. Recomputes ``wcag_aa_coverage`` and ``wcag_aaa_coverage`` from stored
     ``palette_rgb`` using the FIXED functions.
  5. Diffs every recomputed value against the stored value.
  6. Identifies false-pass (stored PASS, recomputed FAIL) and
     false-fail (stored FAIL, recomputed PASS) direction.
  7. Saves full diff to ``results/ablation/wcag_recheck.csv``.
  8. Prints a summary and explicitly reports whether Table I WCAG AA
     column values change for any of the four main methods.

Usage
-----
    python3 -m research.evaluation.wcag_recheck
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# Project root → sys.path
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from dsp.metrics import wcag_pair_coverage  # uses fixed functions

WCAG_AA  = 4.5
WCAG_AAA = 7.0

_MANIFEST_PATH   = _ROOT / "corpus/manifest.json"
_RESULTS_RAW     = _ROOT / "results/raw"
_ABLATION_RAW    = _ROOT / "results/ablation/raw"
_OUTPUT_DIR      = _ROOT / "results/ablation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest() -> dict[str, dict]:
    """Return {image_id: manifest_entry}.  Manifest uses key 'id'."""
    manifest = json.loads(_MANIFEST_PATH.read_text())
    return {entry["id"]: entry for entry in manifest}


def _recompute_wcag(palette_rgb_raw: list[list[int]]) -> tuple[float, float]:
    """Return (wcag_aa_coverage, wcag_aaa_coverage) using the fixed functions."""
    palette = np.array(palette_rgb_raw, dtype=np.uint8)
    aa  = wcag_pair_coverage(palette, threshold=WCAG_AA)
    aaa = wcag_pair_coverage(palette, threshold=WCAG_AAA)
    return aa, aaa


def _process_json_file(
    path: pathlib.Path,
    include_image_ids: set[str] | None,
    source_label: str,
) -> list[dict[str, Any]]:
    """Load one result JSON and return a list of diff rows (one per method)."""
    data = json.loads(path.read_text())
    image_id = str(data.get("image_id", ""))

    if include_image_ids is not None and image_id not in include_image_ids:
        return []

    subset   = data.get("subset", "")
    methods  = data.get("methods", {})
    rows: list[dict[str, Any]] = []

    for method_name, entry in methods.items():
        palette_rgb_raw = entry.get("palette_rgb")
        if palette_rgb_raw is None:
            continue

        metrics_stored = entry.get("metrics", {})
        stored_aa  = metrics_stored.get("wcag_aa_coverage")
        stored_aaa = metrics_stored.get("wcag_aaa_coverage")

        recomp_aa, recomp_aaa = _recompute_wcag(palette_rgb_raw)

        # --- Determine if any colour has max == 1 (the trigger condition) ---
        palette_u8 = np.array(palette_rgb_raw, dtype=np.uint8)
        has_max1_colour = bool(any(palette_u8[i].max() == 1 for i in range(len(palette_u8))))

        # AA coverage diff
        aa_changed  = stored_aa  is not None and not np.isclose(stored_aa,  recomp_aa,  atol=1e-9)
        aaa_changed = stored_aaa is not None and not np.isclose(stored_aaa, recomp_aaa, atol=1e-9)

        # Direction of AA change
        if aa_changed:
            stored_pass  = (stored_aa  >= 1/10)  # any pair passed
            recomp_pass  = (recomp_aa  >= 1/10)
            if stored_pass and not recomp_pass:
                aa_direction = "false-pass→true-fail"
            elif not stored_pass and recomp_pass:
                aa_direction = "false-fail→true-pass"
            else:
                aa_direction = "magnitude-only"
        else:
            aa_direction = "unchanged"

        rows.append({
            "source":          source_label,
            "subset":          subset,
            "image_id":        image_id,
            "method":          method_name,
            "has_max1_colour": has_max1_colour,
            "stored_aa":       stored_aa,
            "recomp_aa":       round(recomp_aa, 6),
            "aa_delta":        round(recomp_aa - (stored_aa or 0), 6),
            "aa_changed":      aa_changed,
            "aa_direction":    aa_direction,
            "stored_aaa":      stored_aaa,
            "recomp_aaa":      round(recomp_aaa, 6),
            "aaa_delta":       round(recomp_aaa - (stored_aaa or 0), 6),
            "aaa_changed":     aaa_changed,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",  default=str(_RESULTS_RAW),  type=str)
    parser.add_argument("--ablation-dir", default=str(_ABLATION_RAW), type=str)
    parser.add_argument("--manifest",     default=str(_MANIFEST_PATH), type=str)
    parser.add_argument("--output-dir",   default=str(_OUTPUT_DIR),   type=str)
    args = parser.parse_args(argv)

    results_dir  = pathlib.Path(args.results_dir)
    ablation_dir = pathlib.Path(args.ablation_dir)
    output_dir   = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest     = _load_manifest()

    # Build test-set and train-set image_id sets
    test_ids  = {
        iid for iid, e in manifest.items()
        if e.get("subset") == "photographs" and not e.get("dev", False)
    }
    train_ids = {
        iid for iid, e in manifest.items()
        if e.get("subset") == "photographs_train"
    }
    dev_ids   = {
        iid for iid, e in manifest.items()
        if e.get("subset") == "photographs" and e.get("dev", False)
    }

    print(f"Test set:  {len(test_ids)} images")
    print(f"Train set: {len(train_ids)} images")
    print(f"Dev set:   {len(dev_ids)} images")

    all_rows: list[dict[str, Any]] = []

    # ── results/raw/ ─────────────────────────────────────────────────────────
    raw_jsons = sorted(results_dir.glob("*.json"))
    print(f"\nProcessing {len(raw_jsons)} JSONs in {results_dir} …")
    include_raw = test_ids | train_ids  # process both; label in 'subset' column
    for path in raw_jsons:
        rows = _process_json_file(path, include_raw, source_label="raw")
        all_rows.extend(rows)

    # ── results/ablation/raw/ ────────────────────────────────────────────────
    abl_jsons = sorted(ablation_dir.glob("*.json"))
    print(f"Processing {len(abl_jsons)} JSONs in {ablation_dir} …")
    for path in abl_jsons:
        rows = _process_json_file(path, include_image_ids=None, source_label="ablation")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # ── Save full diff ────────────────────────────────────────────────────────
    out_path = output_dir / "wcag_recheck.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull diff saved to {out_path}  ({len(df)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────────
    changed   = df[df["aa_changed"]]
    unchanged = df[~df["aa_changed"]]

    print(f"\n{'='*60}")
    print("WCAG AA coverage change summary")
    print(f"{'='*60}")
    print(f"Total (image, method) pairs checked: {len(df)}")
    print(f"  AA coverage CHANGED:   {len(changed)}")
    print(f"  AA coverage UNCHANGED: {len(unchanged)}")
    print(f"  Images with max1 colour present: "
          f"{df.groupby('image_id')['has_max1_colour'].any().sum()}")

    if len(changed) > 0:
        print("\nChanged rows:")
        print(changed[["source","subset","image_id","method",
                        "stored_aa","recomp_aa","aa_delta",
                        "aa_direction"]].to_string(index=False))

        # Direction breakdown
        dirs = changed["aa_direction"].value_counts()
        print(f"\nDirection breakdown:\n{dirs.to_string()}")
    else:
        print("\nNo changes detected. Bug had no effect on stored WCAG AA coverage.")

    # ── False-pass→true-fail (most dangerous direction) ──────────────────────
    false_passes = changed[changed["aa_direction"] == "false-pass→true-fail"]
    if len(false_passes) > 0:
        print(f"\nWARNING: {len(false_passes)} FALSE-PASS cases (stored PASS, correct FAIL):")
        print(false_passes[["source","image_id","method","stored_aa","recomp_aa"]].to_string(index=False))
    else:
        print("\nNo false-pass → true-fail direction changes found.")

    # ── Table I impact (main raw results, 4 methods, test set) ──────────────
    print(f"\n{'='*60}")
    print("Table I WCAG AA column — per-method mean over test set (75 images)")
    print(f"{'='*60}")
    raw_test = df[(df["source"] == "raw") & (df["subset"] == "photographs")]
    # dev images are also in raw/ with subset="photographs"; filter dev out
    raw_test = raw_test[raw_test["image_id"].isin(test_ids)]

    if len(raw_test) > 0:
        for method, g in raw_test.groupby("method"):
            old_mean = g["stored_aa"].mean()
            new_mean = g["recomp_aa"].mean()
            n_changed = g["aa_changed"].sum()
            marker = " ← CHANGED" if n_changed > 0 else ""
            print(f"  {method:<30s}  old={old_mean:.4f}  new={new_mean:.4f}  "
                  f"({n_changed} images changed){marker}")
    else:
        print("  No raw test-set rows found.")

    # ── Ablation method means ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Table III / Ablation WCAG AA column — per-method mean over ablation set (75 images)")
    print(f"{'='*60}")
    abl = df[df["source"] == "ablation"]
    if len(abl) > 0:
        for method, g in abl.groupby("method"):
            old_mean = g["stored_aa"].mean()
            new_mean = g["recomp_aa"].mean()
            n_changed = g["aa_changed"].sum()
            marker = " ← CHANGED" if n_changed > 0 else ""
            print(f"  {method:<30s}  old={old_mean:.4f}  new={new_mean:.4f}  "
                  f"({n_changed} images changed){marker}")
    else:
        print("  No ablation rows found.")

    # ── AAA summary ──────────────────────────────────────────────────────────
    aaa_changed = df[df["aaa_changed"]]
    print(f"\nWCAG AAA coverage changed in {len(aaa_changed)} (image, method) pairs.")
    if len(aaa_changed) > 0:
        print(aaa_changed[["source","image_id","method",
                            "stored_aaa","recomp_aaa","aaa_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
