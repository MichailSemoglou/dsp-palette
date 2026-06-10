"""
Constraint-matched baseline: k-Means Lab over-clustering + DSP selection.

This baseline tests whether DSP's performance gains come from its candidate
pool composition (Pillow median-cut, ~256 colours) or from its constraint
machinery (τ_dist hard constraint + WCAG replacement step), by replacing the
Pillow pool with a k-Means Lab over-clustered pool while keeping all
selection logic identical.

Construction
------------
1. Run k-Means Lab with k=k_over (default 20) on all image pixels.
2. For each cluster, the representative is the pixel closest to the centroid
   in CIELAB space (matching the convention in ``kmeans_lab.py`` to avoid
   out-of-gamut artefacts from centroid inversion).
3. Cluster normalised frequency = cluster pixel count / total pixels.
4. Pass the resulting (k_over, 3) uint8 candidates + (k_over,) float64
   frequencies to :func:`research.dsp.selector.select_from_candidates` with
   default DSP parameters: alpha=beta=1.0, tau_dist=10.0, wcag_step=True.

Interpretation
--------------
DSP and this baseline share identical greedy scoring, τ_dist constraint, and
WCAG replacement step.  They differ only in the candidate pool source.
- If DSP wins:  the Pillow median-cut pool (with its ~256 distinct colours)
  yields better spread than 20 k-Means Lab centroids, and/or the richer pool
  allows the greedy step to find more distinct selections.
- If results are equivalent:  the constraint machinery drives DSP's gains
  regardless of how the candidate pool was constructed.

Public API
----------
extract_palette(image, n, k_over, random_state, alpha, beta, tau_dist)
    -> SelectionResult
"""

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import]

from dsp.metrics import srgb_to_lab
from dsp.selector import select_from_candidates, SelectionResult


def extract_palette(
    image: Image.Image,
    n: int = 5,
    k_over: int = 20,
    random_state: int = 42,
    alpha: float = 1.0,
    beta: float = 1.0,
    tau_dist: float = 10.0,
) -> SelectionResult:
    """Extract a palette via k-Means Lab over-clustering + DSP selection.

    Parameters
    ----------
    image:
        PIL Image (any mode; converted to RGB internally).
    n:
        Target palette size.
    k_over:
        Number of k-Means clusters for over-clustering (candidate pool size).
        Must be > n.  Default 20.
    random_state:
        Seed for k-Means reproducibility.  Matches the seed used by the
        standard ``kmeans_lab`` baseline (42).
    alpha, beta, tau_dist:
        Passed through to :func:`~research.dsp.selector.select_from_candidates`
        unchanged; defaults match full DSP settings.

    Returns
    -------
    SelectionResult
        Same dataclass type as :func:`~research.dsp.selector.select_palette`.
    """
    if k_over <= n:
        raise ValueError(
            f"k_over ({k_over}) must be greater than n ({n}) "
            "to allow greedy selection from the over-clustered pool."
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels_rgb = np.array(image, dtype=np.float64).reshape(-1, 3)
    pixels_lab = srgb_to_lab(pixels_rgb)  # (N, 3)

    km = KMeans(n_clusters=k_over, random_state=random_state, n_init="auto")
    labels = km.fit_predict(pixels_lab)

    counts = np.bincount(labels, minlength=k_over)
    freqs = counts.astype(np.float64) / counts.sum()

    # Representative = nearest actual pixel to centroid in Lab space
    # (avoids out-of-gamut artefacts from centroid inversion).
    candidates_rgb = np.zeros((k_over, 3), dtype=np.uint8)
    for k in range(k_over):
        mask = labels == k
        if mask.any():
            cluster_lab = pixels_lab[mask]
            centre_lab = km.cluster_centers_[k]
            diffs = cluster_lab - centre_lab
            nearest_in_cluster = int(np.argmin((diffs ** 2).sum(axis=1)))
            candidates_rgb[k] = pixels_rgb[mask][nearest_in_cluster].astype(np.uint8)
        else:
            # Empty cluster (very rare at k_over=20): fall back to first pixel.
            candidates_rgb[k] = pixels_rgb[0].astype(np.uint8)

    return select_from_candidates(
        candidates_rgb,
        freqs,
        n=n,
        alpha=alpha,
        beta=beta,
        tau_dist=tau_dist,
        wcag_step=True,
    )
