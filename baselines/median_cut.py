
"""Baseline: Pillow median-cut (mirrors the current package behaviour)."""

import warnings

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def extract_palette(image: Image.Image, n: int = 5) -> NDArray[np.uint8]:
    """Return top-n colours from Pillow median-cut quantization.

    Parameters
    ----------
    image:
        PIL Image (any mode).
    n:
        Number of palette colours.

    Returns
    -------
    NDArray shape (n, 3) uint8 — sorted by descending frequency.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    quantized = image.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    quantized_rgb = quantized.convert("RGB")

    pixels = np.array(quantized_rgb, dtype=np.uint8).reshape(-1, 3)
    unique, counts = np.unique(pixels, axis=0, return_counts=True)

    # Sort descending by frequency, take top-n
    order = np.argsort(-counts)
    result = unique[order[:n]]
    if len(result) < n:
        warnings.warn(
            f"median_cut: Pillow returned only {len(result)} unique colour(s) "
            f"for n={n}; padding palette by repeating the most-frequent colour.",
            UserWarning,
            stacklevel=2,
        )
        pad = np.tile(result[0], (n - len(result), 1))
        result = np.vstack([result, pad])
    return result
