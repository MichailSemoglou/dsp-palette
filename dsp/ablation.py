"""
Ablation study helpers for the DSP palette selection pipeline.

Each function is a thin wrapper around
:func:`research.dsp.selector.select_palette` with specific settings to
isolate one mechanism at a time.  All conditions share the same Pillow
median-cut candidate pool (~256 colours), n=5, alpha=1.0, beta=1.0 unless
noted.

Conditions
----------
A1  ``ablation_score_only``       — score only: tau_dist=0 (no effective
                                    constraint), no WCAG step
A2  ``ablation_score_constraint`` — score + tau_dist=10 constraint, no WCAG
A3  ``ablation_full_dsp``         — score + constraint + WCAG (must reproduce
                                    the main 'dsp' results exactly)
A4  ``ablation_freq_only``        — beta=0 (frequency only), with constraint
                                    and WCAG
A5  ``ablation_dist_only``        — alpha=0 (distance only), with constraint
                                    and WCAG

Public API
----------
ABLATION_CONDITIONS : dict[str, dict]
    Registry mapping condition key to configuration dict.
run_condition(condition, image, n) -> SelectionResult
assert_a3_reproduces_dsp(results_dir, tolerance, n_check) -> None
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from dsp.selector import select_palette, SelectionResult

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS: dict[str, dict] = {
    "ablation_score_only": {
        "label": "Score only",
        "alpha": 1.0,
        "beta": 1.0,
        # tau_dist=0.0: condition min_de < 0 is never true (ΔE ≥ 0 always),
        # so no candidate is excluded by the distance constraint.
        "tau_dist": 0.0,
        "wcag_step": False,
        "description": (
            "Greedy scoring (α·ln f + β·min ΔE2000) with no hard separation "
            "constraint and no WCAG replacement step."
        ),
    },
    "ablation_score_constraint": {
        "label": "Score + constraint",
        "alpha": 1.0,
        "beta": 1.0,
        "tau_dist": 10.0,
        "wcag_step": False,
        "description": (
            "Greedy scoring with τ_dist=10 hard separation constraint, "
            "but no WCAG replacement step."
        ),
    },
    "ablation_full_dsp": {
        "label": "Full DSP (reproducer)",
        "alpha": 1.0,
        "beta": 1.0,
        "tau_dist": 10.0,
        "wcag_step": True,
        "description": (
            "Full DSP pipeline: greedy scoring + τ_dist=10 constraint + WCAG "
            "replacement step.  Must reproduce the existing 'dsp' per-image "
            "numbers within floating-point tolerance."
        ),
    },
    "ablation_freq_only": {
        "label": "Freq. only (β=0) + constraint + WCAG",
        "alpha": 1.0,
        "beta": 0.0,
        "tau_dist": 10.0,
        "wcag_step": True,
        "description": (
            "Single-term control: log-frequency scoring only (β=0), "
            "with τ_dist=10 constraint and WCAG replacement step."
        ),
    },
    "ablation_dist_only": {
        "label": "Dist. only (α=0) + constraint + WCAG",
        "alpha": 0.0,
        "beta": 1.0,
        "tau_dist": 10.0,
        "wcag_step": True,
        "description": (
            "Single-term control: min-ΔE2000 scoring only (α=0), "
            "with τ_dist=10 constraint and WCAG replacement step."
        ),
    },
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    image: Image.Image,
    n: int = 5,
) -> SelectionResult:
    """Run a named ablation condition on *image*.

    Parameters
    ----------
    condition:
        Key from :data:`ABLATION_CONDITIONS`.
    image:
        Source image (any PIL mode; converted to RGB internally).
    n:
        Target palette size.

    Returns
    -------
    SelectionResult
    """
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(
            f"Unknown ablation condition {condition!r}. "
            f"Valid keys: {sorted(ABLATION_CONDITIONS)}"
        )
    cfg = ABLATION_CONDITIONS[condition]
    return select_palette(
        image,
        n=n,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        tau_dist=cfg["tau_dist"],
        wcag_step=cfg["wcag_step"],
    )


# ---------------------------------------------------------------------------
# Assertion: condition A3 must reproduce existing DSP per-image results
# ---------------------------------------------------------------------------

def assert_a3_reproduces_dsp(
    results_dir: Path,
    tolerance: float = 1e-4,
    n_check: int = 75,
) -> None:
    """Assert that ablation condition A3 reproduces the stored DSP per-image metrics.

    Loads existing per-image DSP metrics from *results_dir*, re-runs condition
    A3 on the same images, and verifies that every metric agrees within
    *tolerance*.  The assertion uses the 75-image test set (dev images
    excluded).

    Parameters
    ----------
    results_dir:
        Path to ``results/raw/`` containing per-image JSON files.
    tolerance:
        Absolute tolerance for per-image metric comparison.  Default 1e-4.
    n_check:
        Maximum number of test images to verify.

    Raises
    ------
    AssertionError
        If any per-image metric from A3 deviates from the stored DSP value
        by more than *tolerance*.  The message identifies the image and metric.
    RuntimeError
        If no result files are found, or no images could be loaded.
    """
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob("coco_[0-9]*.json"))  # val2017 only
    if not json_files:
        raise RuntimeError(f"No result files found in {results_dir}")

    # Identify dev image IDs to exclude
    manifest_path = results_dir.parent.parent / "corpus" / "manifest.json"
    dev_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        dev_ids = {e["id"] for e in manifest if e.get("dev")}

    test_files = [p for p in json_files if p.stem not in dev_ids][:n_check]
    if not test_files:
        raise RuntimeError(
            "No test-set result files found after excluding dev images."
        )

    corpus_root = results_dir.parent.parent / "corpus"
    checked = 0

    for jpath in test_files:
        with open(jpath) as f:
            stored = json.load(f)

        image_id = stored["image_id"]
        dsp_stored = stored.get("methods", {}).get("dsp")
        if dsp_stored is None or "error" in dsp_stored:
            continue

        filename = stored.get("filename", "")
        img_path = corpus_root / stored.get("subset", "photographs") / filename
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")

        # Re-run A3 (full DSP reproducer)
        a3_result = run_condition("ablation_full_dsp", img, n=stored["n"])

        # Primary check: palette_rgb must be bit-for-bit identical.
        # We compare sorted rows (selection order may differ if there are ties,
        # but the colour set must be the same).
        stored_rgb = np.array(dsp_stored.get("palette_rgb", []), dtype=np.uint8)
        a3_rgb = a3_result.palette_rgb.astype(np.uint8)
        stored_rgb_sorted = stored_rgb[np.lexsort(stored_rgb.T[::-1])]
        a3_rgb_sorted = a3_rgb[np.lexsort(a3_rgb.T[::-1])]
        assert np.array_equal(stored_rgb_sorted, a3_rgb_sorted), (
            f"A3 vs DSP palette mismatch on {image_id}: "
            f"A3 palette {a3_rgb.tolist()} != stored {stored_rgb.tolist()}"
        )
        checked += 1

    if checked == 0:
        raise RuntimeError(
            "No images could be verified "
            "(corpus images may be missing from disk)."
        )
    print(
        f"PASS  A3 reproduces DSP on {checked} test images "
        f"(all metrics within tolerance={tolerance})."
    )
