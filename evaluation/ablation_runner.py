"""
Ablation evaluation runner.

Evaluates five DSP ablation conditions (A1–A5) and the constraint-matched
k-Means Lab + constraints baseline on the 75-image test set (COCO val2017,
dev=False).

Before any evaluation the runner asserts that condition A3 (Full DSP
reproducer) matches the stored per-image DSP metrics within tolerance=1e-4.
If the assertion fails the script stops and reports the mismatch; it does
NOT attempt to correct the stored results.

Per-image output files are written to ``results/ablation/raw/{image_id}.json``
in a format consistent with the main runner's ``results/raw/`` JSON schema.

Usage
-----
    python -m research.evaluation.ablation_runner \\
        --manifest  corpus/manifest.json \\
        --corpus-root corpus/ \\
        --results-dir results/raw/ \\
        --ablation-dir results/ablation/raw/ \\
        --n 5

Condition keys written to output JSON
--------------------------------------
    ablation_score_only         (A1)
    ablation_score_constraint   (A2)
    ablation_full_dsp           (A3 — saved for cross-check only)
    ablation_freq_only          (A4)
    ablation_dist_only          (A5)
    kmeans_lab_constrained      (k-Means Lab + constraints baseline)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from dsp.ablation import ABLATION_CONDITIONS, assert_a3_reproduces_dsp, run_condition
from dsp.metrics import (
    WCAG_AA_NORMAL,
    assert_de2000_available,
    harmony_alignment,
    min_pairwise_delta_e,
    reconstruction_error_de2000,
    wcag_pair_coverage,
)

# Fail loudly at import time if the real CIEDE2000 path is not active.
assert_de2000_available()
import baselines.kmeans_lab_constrained as _kmlc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_metrics(palette_rgb: np.ndarray, palette_lab: np.ndarray,
                     image_arr: np.ndarray) -> dict:
    """Return the four standard evaluation metrics for a palette."""
    return {
        "min_pairwise_de2000": float(min_pairwise_delta_e(palette_lab)),
        "wcag_aa_coverage": float(wcag_pair_coverage(palette_rgb, threshold=WCAG_AA_NORMAL)),
        "reconstruction_error_de2000": float(
            reconstruction_error_de2000(image_arr, palette_rgb, seed=42)
        ),
        "harmony_alignment": float(harmony_alignment(palette_lab)),
    }


def _run_ablation_conditions(image: Image.Image, image_arr: np.ndarray,
                              n: int) -> dict:
    """Run all ablation conditions on one image; return a results dict."""
    results: dict[str, dict] = {}

    for key in ABLATION_CONDITIONS:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = run_condition(key, image, n=n)
        elapsed = time.perf_counter() - t0

        results[key] = {
            "palette_rgb": sel.palette_rgb.tolist(),
            "palette_hex": [
                "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
                for r, g, b in sel.palette_rgb
            ],
            "frequencies": sel.frequencies,
            "wcag_guaranteed": sel.wcag_guaranteed,
            "wcag_replacement_applied": sel.wcag_replacement_applied,
            "wcag_distinctness_compromised": sel.wcag_distinctness_compromised,
            "metrics": _compute_metrics(sel.palette_rgb, sel.palette_lab, image_arr),
            "runtime_seconds": elapsed,
        }

    # k-Means Lab + constraints baseline
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmlc_sel = _kmlc.extract_palette(image, n=n)
    elapsed = time.perf_counter() - t0

    results["kmeans_lab_constrained"] = {
        "palette_rgb": kmlc_sel.palette_rgb.tolist(),
        "palette_hex": [
            "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
            for r, g, b in kmlc_sel.palette_rgb
        ],
        "frequencies": kmlc_sel.frequencies,
        "wcag_guaranteed": kmlc_sel.wcag_guaranteed,
        "wcag_replacement_applied": kmlc_sel.wcag_replacement_applied,
        "wcag_distinctness_compromised": kmlc_sel.wcag_distinctness_compromised,
        "metrics": _compute_metrics(kmlc_sel.palette_rgb, kmlc_sel.palette_lab, image_arr),
        "runtime_seconds": elapsed,
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run DSP ablation evaluation on the 75-image test set.",
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to corpus/manifest.json",
    )
    parser.add_argument(
        "--corpus-root", required=True,
        help="Root directory containing the image sub-folders.",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to existing results/raw/ (used for DSP assertion check).",
    )
    parser.add_argument(
        "--ablation-dir", required=True,
        help="Output directory for ablation per-image JSON files.",
    )
    parser.add_argument("--n", type=int, default=5, help="Target palette size.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    corpus_root = Path(args.corpus_root)
    results_dir = Path(args.results_dir)
    ablation_dir = Path(args.ablation_dir)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build test set: val2017 images with dev=False
    test_entries = [
        e for e in manifest
        if e.get("subset") == "photographs" and not e.get("dev", False)
    ]
    print(f"Test set: {len(test_entries)} images (val2017, dev=False).")

    # ------------------------------------------------------------------
    # Assert A3 reproduces DSP before proceeding
    # ------------------------------------------------------------------
    print("\nAsserting A3 (full DSP reproducer) matches stored DSP metrics...")
    try:
        assert_a3_reproduces_dsp(results_dir, tolerance=1e-4, n_check=len(test_entries))
    except AssertionError as exc:
        print(f"\nFAIL  A3 assertion failed:\n  {exc}", file=sys.stderr)
        print(
            "\nAborting.  The ablation runner will NOT overwrite existing results.\n"
            "Do not modify the DSP pipeline to force this assertion to pass.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run ablation conditions on every test image
    # ------------------------------------------------------------------
    print("\nRunning ablation conditions on test images...\n")
    total_wall = time.perf_counter()

    for i, entry in enumerate(test_entries, 1):
        image_id = entry["id"]
        subset = entry["subset"]
        filename = entry["filename"]
        img_path = corpus_root / subset / filename

        if not img_path.exists():
            print(f"  [{i:3d}/{len(test_entries)}]  SKIP  {image_id}  (file missing)")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = Image.open(img_path).convert("RGB")
        image_arr = np.array(image, dtype=np.uint8)

        condition_results = _run_ablation_conditions(image, image_arr, n=args.n)

        out = {
            "image_id": image_id,
            "n": args.n,
            "subset": subset,
            "filename": filename,
            "methods": condition_results,
        }

        out_path = ablation_dir / f"{image_id}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        # Progress heartbeat every 10 images
        if i % 10 == 0 or i == len(test_entries):
            elapsed_so_far = time.perf_counter() - total_wall
            print(f"  [{i:3d}/{len(test_entries)}]  {elapsed_so_far:.1f}s elapsed")

    total_elapsed = time.perf_counter() - total_wall
    print(
        f"\nDone.  {len(test_entries)} images processed in {total_elapsed:.1f}s.\n"
        f"Results written to: {ablation_dir}"
    )


if __name__ == "__main__":
    main()
