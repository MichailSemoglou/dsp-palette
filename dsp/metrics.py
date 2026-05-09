
"""
Perceptual and design-system metrics for palette evaluation.

All colour-science computations use the ``colour`` library (colour-science on
PyPI) which implements ITU/CIE standards.  Where colour is unavailable the
module falls back to a pure-NumPy ΔE2000 implementation so that unit tests
can run in a lean environment.

Public API
----------
delta_e2000(lab1, lab2) -> float
    CIE ΔE 2000 between two CIELAB colours.

wcag_contrast(rgb1, rgb2) -> float
    WCAG 2.1 contrast ratio between two sRGB colours.

wcag_level(contrast_ratio) -> str
    "AAA" / "AA" / "A" / "fail" based on WCAG 2.1 thresholds.

harmony_alignment(palette_lab) -> float
    Mean harmonic alignment score for a palette.

min_pairwise_delta_e(palette_lab) -> float
    Minimum ΔE2000 over all palette pairs.

wcag_pair_coverage(palette_rgb, threshold) -> float
    Fraction of C(n,2) pairs satisfying a given contrast threshold.

reconstruction_error(image_rgb, palette_rgb) -> float
    Mean ΔE2000 from each pixel to its nearest palette colour.

reconstruction_error_de2000(image_rgb, palette_rgb, max_pixels, seed) -> float
    Vectorised ΔE2000 reconstruction error with pixel subsampling.
    Preferred metric for all reported results.

reconstruction_error_fast(image_rgb, palette_rgb) -> float
    ΔE76 (L2-in-Lab) approximation.  Retained for backwards compatibility.
"""

import logging
import math
import warnings
from itertools import combinations
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Attempt to import colour-science; fall back to pure-NumPy implementation.
# ---------------------------------------------------------------------------
try:
    import colour  # type: ignore[import]

    _COLOUR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _COLOUR_AVAILABLE = False


# ---------------------------------------------------------------------------
# sRGB ↔ CIELAB conversion helpers
# ---------------------------------------------------------------------------

# D65 reference white (XYZ, 2° observer)
_D65_XYZ = np.array([95.0489, 100.0, 108.8840])


def srgb_to_lab(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an array of sRGB colours (0–255 uint8 or 0–1 float) to CIELAB.

    Parameters
    ----------
    rgb:
        Shape (..., 3), values either in [0, 255] (uint8) or [0, 1] (float).

    Returns
    -------
    NDArray of shape (..., 3) with L*, a*, b* values.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    if _COLOUR_AVAILABLE:
        return colour.XYZ_to_Lab(
            colour.sRGB_to_XYZ(rgb),
            illuminant=colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                "D65"
            ],
        )

    # --- Pure-NumPy fallback (IEC 61966-2-1, D65 2° observer) ---------------
    # 1. Linearise sRGB (gamma removal)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # 2. Linear sRGB → XYZ (D65, IEC 61966-2-1 matrix)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ M.T * 100.0  # scale to match _D65_XYZ

    # 3. XYZ → CIELAB
    xyz_norm = xyz / _D65_XYZ
    epsilon = 0.008856
    kappa = 903.3
    f = np.where(
        xyz_norm > epsilon,
        np.cbrt(xyz_norm),
        (kappa * xyz_norm + 16.0) / 116.0,
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ---------------------------------------------------------------------------
# ΔE2000
# ---------------------------------------------------------------------------


def delta_e2000(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIE ΔE 2000 between two CIELAB colours.

    Parameters
    ----------
    lab1, lab2:
        CIELAB triples (L*, a*, b*).

    Returns
    -------
    float  — ΔE2000 distance (≥ 0).

    Reference
    ---------
    Sharma G., Wu W., Dalal E. N. (2005).  "The CIEDE2000 Color-Difference
    Formula: Implementation Notes, Supplementary Test Data, and Mathematical
    Observations."  Color Research & Application, 30(1), 21–30.
    """
    if _COLOUR_AVAILABLE:
        return float(
            colour.delta_E(
                np.asarray(lab1, dtype=np.float64),
                np.asarray(lab2, dtype=np.float64),
                method="CIE 2000",
            )
        )

    # Pure-NumPy implementation of CIEDE2000
    L1, a1, b1 = (float(x) for x in lab1)
    L2, a2, b2 = (float(x) for x in lab2)

    # Step 1 – a' adjustment
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - math.sqrt(C_avg7 / (C_avg7 + 25**7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    # Step 2 – C', h'
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    # Step 3 – ΔL', ΔC', ΔH'
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    # Step 4 – CIEDE2000
    Lp_avg = (L1 + L2) / 2
    Cp_avg = (C1p + C2p) / 2
    Cp_avg7 = Cp_avg**7

    if C1p * C2p == 0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_avg = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hp_avg = (h1p + h2p + 360) / 2
    else:
        hp_avg = (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(hp_avg - 30))
        + 0.24 * math.cos(math.radians(2 * hp_avg))
        + 0.32 * math.cos(math.radians(3 * hp_avg + 6))
        - 0.20 * math.cos(math.radians(4 * hp_avg - 63))
    )

    SL = 1 + 0.015 * (Lp_avg - 50) ** 2 / math.sqrt(20 + (Lp_avg - 50) ** 2)
    SC = 1 + 0.045 * Cp_avg
    SH = 1 + 0.015 * Cp_avg * T

    d_theta = 30 * math.exp(-(((hp_avg - 275) / 25) ** 2))
    RC = 2 * math.sqrt(Cp_avg7 / (Cp_avg7 + 25**7))
    RT = -math.sin(math.radians(2 * d_theta)) * RC

    return math.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def delta_e2000_batch(
    lab_ref: NDArray[np.float64],
    lab_targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """\u0394E2000 from one reference Lab to many target Labs.

    Vectorised via ``colour.delta_E`` when colour-science is available;
    falls back to a Python loop over the pure-NumPy implementation.

    Parameters
    ----------
    lab_ref:
        Shape (3,) reference colour.
    lab_targets:
        Shape (N, 3) array of target colours.

    Returns
    -------
    NDArray of shape (N,) with \u0394E2000 values.
    """
    targets = np.asarray(lab_targets, dtype=np.float64)
    if _COLOUR_AVAILABLE:
        ref = np.broadcast_to(
            np.asarray(lab_ref, dtype=np.float64), targets.shape
        ).copy()
        return colour.delta_E(ref, targets, method="CIE 2000")
    return np.array([delta_e2000(lab_ref, t) for t in targets])


# ---------------------------------------------------------------------------
# WCAG contrast
# ---------------------------------------------------------------------------

WCAG_AA_NORMAL = 4.5
WCAG_AAA_NORMAL = 7.0


def relative_luminance(rgb: Sequence[float]) -> float:
    """WCAG 2.1 relative luminance for sRGB (values 0–255 or 0–1)."""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    lin = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def wcag_contrast(rgb1: Sequence[float], rgb2: Sequence[float]) -> float:
    """WCAG 2.1 contrast ratio between two sRGB colours.

    Parameters
    ----------
    rgb1, rgb2:
        RGB triples, either in [0, 255] or [0, 1].

    Returns
    -------
    float — contrast ratio in [1, 21].

    Reference
    ---------
    WCAG 2.1 §1.4.3: https://www.w3.org/TR/WCAG21/#contrast-minimum
    """
    L1 = relative_luminance(rgb1)
    L2 = relative_luminance(rgb2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)


def wcag_level(contrast_ratio: float) -> str:
    """Map a contrast ratio to a WCAG 2.1 conformance level string.

    Returns one of: ``"AAA"``, ``"AA"``, ``"A"`` (large text only AA), or
    ``"fail"``.
    """
    if contrast_ratio >= WCAG_AAA_NORMAL:
        return "AAA"
    if contrast_ratio >= WCAG_AA_NORMAL:
        return "AA"
    if contrast_ratio >= 3.0:
        return "A"
    return "fail"


# ---------------------------------------------------------------------------
# Palette-level metrics
# ---------------------------------------------------------------------------


def min_pairwise_delta_e(palette_lab: NDArray[np.float64]) -> float:
    """Minimum ΔE2000 over all pairs in a palette.

    Parameters
    ----------
    palette_lab:
        Shape (n, 3) array of CIELAB colours.

    Returns
    -------
    float — minimum ΔE2000 (0 if fewer than 2 colours).
    """
    n = len(palette_lab)
    if n < 2:
        return 0.0
    return min(delta_e2000(palette_lab[i], palette_lab[j]) for i, j in combinations(range(n), 2))


def wcag_pair_coverage(
    palette_rgb: NDArray[np.float64],
    threshold: float = WCAG_AA_NORMAL,
) -> float:
    """Fraction of C(n, 2) pairs meeting the given contrast threshold.

    Parameters
    ----------
    palette_rgb:
        Shape (n, 3) array of sRGB colours (0–255).
    threshold:
        Contrast ratio threshold (default 4.5 for WCAG AA normal text).

    Returns
    -------
    float in [0, 1].
    """
    n = len(palette_rgb)
    if n < 2:
        return 0.0
    pairs = list(combinations(range(n), 2))
    hits = sum(1 for i, j in pairs if wcag_contrast(palette_rgb[i], palette_rgb[j]) >= threshold)
    return hits / len(pairs)


# ---------------------------------------------------------------------------
# Harmonic alignment
# ---------------------------------------------------------------------------

# Canonical harmony angles (degrees) — complementary, analogous, triadic,
# split-complementary reference points.
_HARMONY_ANGLES = np.array([30.0, 60.0, 90.0, 120.0, 180.0])
_HARMONY_SIGMA = 15.0  # degrees


def _alignment_score(hue_diff: float) -> float:
    """exp(-((θ - c) / σ)²) maximised over canonical angles."""
    diffs = np.abs(hue_diff - _HARMONY_ANGLES)
    # Wrap at 180° (hue difference is symmetric)
    diffs = np.minimum(diffs, 360.0 - diffs)
    return float(np.max(np.exp(-((diffs / _HARMONY_SIGMA) ** 2))))


def harmony_alignment(palette_lab: NDArray[np.float64]) -> float:
    """Mean harmonic alignment score for all pairs in a palette.

    Hue angle is computed from a* and b* in CIELAB.  Reported descriptively;
    not a primary win-claim.

    Parameters
    ----------
    palette_lab:
        Shape (n, 3) CIELAB array.

    Returns
    -------
    float in [0, 1] — higher means hue differences are closer to canonical
    harmony angles.  Returns 0.0 for palettes with fewer than 2 colours.
    """
    n = len(palette_lab)
    if n < 2:
        return 0.0

    hues = np.degrees(np.arctan2(palette_lab[:, 2], palette_lab[:, 1])) % 360.0
    scores = [
        _alignment_score(abs(hues[i] - hues[j]) % 360.0)
        for i, j in combinations(range(n), 2)
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


def reconstruction_error(
    image_rgb: NDArray[np.float64],
    palette_rgb: NDArray[np.float64],
) -> float:
    """Mean ΔE2000 from each image pixel to its nearest palette colour.

    Parameters
    ----------
    image_rgb:
        Shape (H, W, 3) or (N, 3) sRGB array (values 0–255).
    palette_rgb:
        Shape (k, 3) sRGB palette (values 0–255).

    Returns
    -------
    float — mean reconstruction ΔE2000 (lower = better fidelity).
    """
    warnings.warn(
        "reconstruction_error() uses a per-pixel Python loop and is slow. "
        "Use reconstruction_error_de2000() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    pixels = image_rgb.reshape(-1, 3).astype(np.float64)
    palette = np.asarray(palette_rgb, dtype=np.float64)

    pixels_lab = srgb_to_lab(pixels)
    palette_lab = srgb_to_lab(palette)

    # For each pixel find the nearest palette colour in CIELAB
    errors = []
    for plab in pixels_lab:
        nearest = min(delta_e2000(plab, c) for c in palette_lab)
        errors.append(nearest)

    return float(np.mean(errors))


def reconstruction_error_fast(
    image_rgb: NDArray[np.float64],
    palette_rgb: NDArray[np.float64],
) -> float:
    """Faster approximation of reconstruction error using L2 distance in Lab.

    Uses Euclidean distance in CIELAB (ΔE76) rather than the full ΔE2000
    formula.  Suitable for large images where exact ΔE2000 is expensive.
    Mark outputs from this function clearly in results.

    Parameters
    ----------
    image_rgb:
        Shape (H, W, 3) or (N, 3) sRGB array (values 0–255).
    palette_rgb:
        Shape (k, 3) sRGB palette (values 0–255).

    Returns
    -------
    float — mean ΔE76 reconstruction error.
    """
    pixels = image_rgb.reshape(-1, 3).astype(np.float64)
    palette = np.asarray(palette_rgb, dtype=np.float64)

    pixels_lab = srgb_to_lab(pixels)      # (N, 3)
    palette_lab = srgb_to_lab(palette)    # (k, 3)

    # Broadcast: (N, 1, 3) - (1, k, 3) → (N, k, 3)
    diffs = pixels_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
    dist = np.sqrt((diffs**2).sum(axis=-1))   # (N, k)
    return float(dist.min(axis=1).mean())


_recon_logger = logging.getLogger(__name__)


def reconstruction_error_de2000(
    image_rgb: NDArray[np.float64],
    palette_rgb: NDArray[np.float64],
    max_pixels: int = 50_000,
    seed: int = 42,
) -> float:
    """Mean per-pixel ΔE2000 from each image pixel to its nearest palette colour.

    Vectorised over pixels; loops only over the small palette (k ≤ ~8).
    Pixel count is capped at ``max_pixels`` via uniform random subsampling to
    keep runtimes acceptable on large images.

    Falls back to ΔE76 (``reconstruction_error_fast``) with a warning if
    colour-science is not installed.

    Parameters
    ----------
    image_rgb:
        Shape (H, W, 3) or (N, 3) sRGB array (values 0–255).
    palette_rgb:
        Shape (k, 3) sRGB palette (values 0–255).
    max_pixels:
        Maximum number of pixels to evaluate.  Pixels are sampled without
        replacement when the image exceeds this count.  Default 50 000.
    seed:
        Random seed for reproducible subsampling.  Default 42.

    Returns
    -------
    float — mean ΔE2000 reconstruction error (lower = better fidelity).
    """
    if not _COLOUR_AVAILABLE:
        _recon_logger.warning(
            "colour-science not available; falling back to ΔE76 for reconstruction error."
        )
        return reconstruction_error_fast(image_rgb, palette_rgb)

    pixels = image_rgb.reshape(-1, 3).astype(np.float64)
    n_pixels = len(pixels)
    if n_pixels > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_pixels, size=max_pixels, replace=False)
        pixels = pixels[idx]

    palette = np.asarray(palette_rgb, dtype=np.float64)
    k = len(palette)

    pixels_lab = srgb_to_lab(pixels)    # (N, 3)
    palette_lab = srgb_to_lab(palette)  # (k, 3)

    # For each palette entry, compute ΔE2000 from every pixel to that entry.
    # colour.delta_E is element-wise, so broadcast palette[j] to (N, 3).
    n = len(pixels_lab)
    de_matrix = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        p_repeated = np.broadcast_to(palette_lab[j], (n, 3)).copy()
        de_matrix[:, j] = colour.delta_E(pixels_lab, p_repeated, method="CIE 2000")

    return float(de_matrix.min(axis=1).mean())
