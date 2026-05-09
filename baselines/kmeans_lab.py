
"""Baseline: k-means clustering in CIELAB colour space."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import]

from research.dsp.metrics import srgb_to_lab


def extract_palette(
    image: Image.Image,
    n: int = 5,
    random_state: int = 42,
) -> NDArray[np.uint8]:
    """Return n cluster centres from k-means fit on CIELAB pixels.

    The returned palette is in sRGB (0-255) for consistency with other methods.

    Parameters
    ----------
    image:
        PIL Image (any mode).
    n:
        Number of clusters / palette colours.
    random_state:
        Seed for KMeans reproducibility.

    Returns
    -------
    NDArray shape (n, 3) uint8 sRGB — sorted by descending cluster size.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels_rgb = np.array(image, dtype=np.float64).reshape(-1, 3)
    pixels_lab = srgb_to_lab(pixels_rgb)  # (N, 3)

    km = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
    labels = km.fit_predict(pixels_lab)

    # Map each cluster centre back to the nearest actual pixel colour in sRGB
    # (avoids out-of-gamut artefacts from inverting Lab→sRGB)
    counts = np.bincount(labels, minlength=n)
    representatives = np.zeros((n, 3), dtype=np.uint8)
    for k in range(n):
        mask = labels == k
        if mask.any():
            # Use the pixel closest to the cluster centre in Lab space
            cluster_pixels_lab = pixels_lab[mask]
            centre_lab = km.cluster_centers_[k]
            diffs = cluster_pixels_lab - centre_lab
            nearest_in_cluster = np.argmin((diffs**2).sum(axis=1))
            representatives[k] = pixels_rgb[mask][nearest_in_cluster].astype(np.uint8)

    order = np.argsort(-counts)
    return representatives[order]
