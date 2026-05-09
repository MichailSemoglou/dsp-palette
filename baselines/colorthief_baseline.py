
"""Baseline: ColorThief palette extraction (optional dependency)."""

import numpy as np
from numpy.typing import NDArray
from PIL import Image

try:
    from colorthief import ColorThief  # type: ignore[import]

    _COLORTHIEF_AVAILABLE = True
except ImportError:
    _COLORTHIEF_AVAILABLE = False


def is_available() -> bool:
    """Return True if the colorthief package is installed."""
    return _COLORTHIEF_AVAILABLE


def extract_palette(
    image: Image.Image,
    n: int = 5,
) -> NDArray[np.uint8]:
    """Return n colours from ColorThief.

    Parameters
    ----------
    image:
        PIL Image (any mode).  Written to a temporary buffer for ColorThief.
    n:
        Number of palette colours.

    Returns
    -------
    NDArray shape (n, 3) uint8.

    Raises
    ------
    ImportError
        If colorthief is not installed.
    """
    if not _COLORTHIEF_AVAILABLE:
        raise ImportError(
            "colorthief is not installed.  Run: pip install colorthief"
        )

    import io

    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    ct = ColorThief(buf)
    palette = ct.get_palette(color_count=n, quality=1)

    # ColorThief may return fewer colours than requested for small/simple images
    result = np.array(palette[:n], dtype=np.uint8)
    if len(result) == 0:
        return np.zeros((n, 3), dtype=np.uint8)
    if len(result) < n:
        # Pad with last colour to keep consistent shape
        pad = np.tile(result[-1], (n - len(result), 1))
        result = np.vstack([result, pad])
    return result
