
"""
Evaluation runner: run all methods on all corpus images and save raw results.

Each result is saved as JSON to results/raw/{image_id}_{method}.json.
Results include palette colours, role assignments, and all 4 metrics.

Usage
-----
    python -m evaluation.runner \
        --manifest corpus/manifest.json \
        --corpus-root corpus/ \
        --results-dir results/raw/ \
        --n 5
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import KDTree

from dsp.metrics import (
    min_pairwise_delta_e,
    wcag_pair_coverage,
    reconstruction_error_de2000,
    harmony_alignment,
    srgb_to_lab,
    WCAG_AA_NORMAL,
    WCAG_AAA_NORMAL,
)
from dsp.roles import assign_roles
from dsp.selector import select_palette
from baselines import median_cut, kmeans_rgb, kmeans_lab, colorthief_baseline
from evaluation.corpus_loader import CorpusLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


def _run_dsp(image: Image.Image, n: int) -> tuple[np.ndarray, dict[str, Any]]:
    result = select_palette(image, n=n)
    meta = {
        "wcag_guaranteed": result.wcag_guaranteed,
        "wcag_replacement_applied": result.wcag_replacement_applied,
        "wcag_distinctness_compromised": result.wcag_distinctness_compromised,
        "candidate_pool_size": result.candidate_pool_size,
        "alpha": result.alpha,
        "beta": result.beta,
        "tau_dist": result.tau_dist,
    }
    return result.palette_rgb, meta


def _run_median_cut(image: Image.Image, n: int) -> tuple[np.ndarray, dict[str, Any]]:
    palette = median_cut.extract_palette(image, n=n)
    return palette, {}


def _run_kmeans_rgb(image: Image.Image, n: int) -> tuple[np.ndarray, dict[str, Any]]:
    palette = kmeans_rgb.extract_palette(image, n=n)
    return palette, {}


def _run_kmeans_lab(image: Image.Image, n: int) -> tuple[np.ndarray, dict[str, Any]]:
    palette = kmeans_lab.extract_palette(image, n=n)
    return palette, {}


def _run_colorthief(image: Image.Image, n: int) -> tuple[np.ndarray, dict[str, Any]]:
    if not colorthief_baseline.is_available():
        raise RuntimeError("colorthief not installed")
    palette = colorthief_baseline.extract_palette(image, n=n)
    return palette, {}


METHODS: dict[str, Any] = {
    "dsp": _run_dsp,
    "median_cut": _run_median_cut,
    "kmeans_rgb": _run_kmeans_rgb,
    "kmeans_lab": _run_kmeans_lab,
}

if colorthief_baseline.is_available():
    METHODS["colorthief"] = _run_colorthief


# ---------------------------------------------------------------------------
# Per-image evaluator
# ---------------------------------------------------------------------------


def evaluate_image(
    image: Image.Image,
    image_id: str,
    n: int = 5,
) -> dict[str, Any]:
    """Run all methods on a single image and return a results dict."""
    img_array = np.array(image.convert("RGB"), dtype=np.uint8)
    results: dict[str, Any] = {"image_id": image_id, "n": n, "methods": {}}

    # Pre-compute image mean L* once (used by auto mode in assign_roles)
    img_lab = srgb_to_lab(img_array.reshape(-1, 3).astype(np.float64))
    image_mean_L = float(img_lab[:, 0].mean())
    results["image_mean_L"] = image_mean_L

    for method_name, fn in METHODS.items():
        t0 = time.perf_counter()
        try:
            palette_rgb, extra_meta = fn(image, n)
        except Exception as exc:
            logger.exception("Method %s failed on %s", method_name, image_id)
            results["methods"][method_name] = {"error": str(exc)}
            continue
        elapsed = time.perf_counter() - t0

        palette_lab = srgb_to_lab(palette_rgb.astype(np.float64))
        freqs = _estimate_frequencies(img_array, palette_rgb)

        roles = assign_roles(palette_rgb, palette_lab, freqs,
                              mode="auto", image_mean_L=image_mean_L)

        metrics = {
            "min_pairwise_de2000": min_pairwise_delta_e(palette_lab),
            "wcag_aa_coverage": wcag_pair_coverage(palette_rgb, threshold=WCAG_AA_NORMAL),
            "wcag_aaa_coverage": wcag_pair_coverage(palette_rgb, threshold=WCAG_AAA_NORMAL),
            "reconstruction_error_de2000": reconstruction_error_de2000(img_array, palette_rgb),
            "harmony_alignment": harmony_alignment(palette_lab),
            "runtime_seconds": elapsed,
        }

        results["methods"][method_name] = {
            "palette_rgb": palette_rgb.tolist(),
            "palette_hex": [
                "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
                for r, g, b in palette_rgb
            ],
            "frequencies": freqs,
            "roles": roles.roles_map,
            "metrics": metrics,
            **extra_meta,
        }

    return results


def _estimate_frequencies(
    img_array: np.ndarray,
    palette_rgb: np.ndarray,
) -> list[float]:
    """Estimate frequency of each palette colour via nearest-neighbour in RGB."""
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    palette = palette_rgb.astype(np.float32)

    # KDTree nearest-neighbour: O(N log K) time, O(N + K) memory
    _, labels = KDTree(palette).query(pixels, k=1, workers=1)

    total = len(pixels)
    return [float((labels == k).sum() / total) for k in range(len(palette))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run palette evaluation on corpus")
    parser.add_argument("--manifest", default="corpus/manifest.json")
    parser.add_argument("--corpus-root", default="corpus/")
    parser.add_argument("--results-dir", default="results/raw/")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--subset", default=None, help="Only run on a specific subset")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results instead of skipping them")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    loader = CorpusLoader(
        manifest_path=args.manifest,
        corpus_root=args.corpus_root,
        subset=args.subset,
    )

    for entry in loader.iter_images():
        out_path = results_dir / f"{entry.image_id}.json"
        if out_path.exists() and not args.force:
            logger.info("Skipping (already done): %s", entry.image_id)
            continue

        logger.info("Processing: %s (%s)", entry.image_id, entry.subset)
        result = evaluate_image(entry.image, entry.image_id, n=args.n)
        result["subset"] = entry.subset
        result["filename"] = str(entry.path.name)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
