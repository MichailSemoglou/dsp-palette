
"""Baseline: k-means clustering in RGB colour space."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.cluster import KMeans  # type: ignore[import]


def extract_palette(
    image: Image.Image,
    n: int = 5,
    random_state: int = 42,
) -> NDArray[np.uint8]:
    """Return n cluster centres from k-means fit on RGB pixels.

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
    NDArray shape (n, 3) uint8 — centres sorted by descending cluster size.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixels = np.array(image, dtype=np.float32).reshape(-1, 3)

    km = KMeans(n_clusters=n, random_state=random_state, n_init="auto")
    labels = km.fit_predict(pixels)

    centres = km.cluster_centers_  # (n, 3) float32
    counts = np.bincount(labels, minlength=n)
    order = np.argsort(-counts)
    return np.clip(centres[order], 0, 255).astype(np.uint8)
