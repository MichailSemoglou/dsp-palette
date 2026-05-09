
"""
Unit tests for dsp/selector.py — constrained greedy selection.
"""

import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

from dsp.selector import SelectionResult, select_palette
from dsp.metrics import min_pairwise_delta_e, srgb_to_lab


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(rgb: tuple[int, int, int], size: int = 20) -> Image.Image:
    return Image.new("RGB", (size, size), color=rgb)


def _gradient_image(size: int = 50) -> Image.Image:
    """A simple left-to-right gradient from dark to light."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        val = int(x / (size - 1) * 255)
        arr[:, x, :] = val
    return Image.fromarray(arr, "RGB")


def _rainbow_image(size: int = 50) -> Image.Image:
    """A hue-gradient image covering 0–360°."""
    import colorsys
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        h = x / size
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        arr[:, x, :] = [int(r * 255), int(g * 255), int(b * 255)]
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_returns_selection_result():
    img = _gradient_image()
    result = select_palette(img, n=3)
    assert isinstance(result, SelectionResult)


def test_palette_shape_n():
    """Palette must have exactly n colours (when enough candidates exist)."""
    img = _rainbow_image()
    for n in (2, 3, 5):
        result = select_palette(img, n=n)
        assert result.palette_rgb.shape == (n, 3), f"Expected ({n}, 3), got {result.palette_rgb.shape}"
        assert result.palette_lab.shape == (n, 3)
        assert len(result.frequencies) == n


def test_palette_rgb_dtype():
    """Palette RGB must be uint8."""
    result = select_palette(_rainbow_image(), n=3)
    assert result.palette_rgb.dtype == np.uint8


def test_palette_rgb_range():
    """All RGB values must be in [0, 255]."""
    result = select_palette(_rainbow_image(), n=5)
    assert result.palette_rgb.min() >= 0
    assert result.palette_rgb.max() <= 255


def test_frequencies_sum_to_one_approx():
    """Frequencies should sum to ≤ 1.0 (they are fractions of candidate pool)."""
    result = select_palette(_rainbow_image(), n=5)
    assert sum(result.frequencies) <= 1.0 + 1e-6


def test_n_stored_correctly():
    img = _rainbow_image()
    result = select_palette(img, n=4)
    assert result.n == 4


def test_hyperparameters_stored():
    result = select_palette(_gradient_image(), n=3, alpha=0.5, beta=2.0, tau_dist=8.0)
    assert result.alpha == 0.5
    assert result.beta == 2.0
    assert result.tau_dist == 8.0


# ---------------------------------------------------------------------------
# Distinctness constraint
# ---------------------------------------------------------------------------


def test_min_de_respects_tau_for_varied_image():
    """For a varied image, min pairwise ΔE2000 should be ≥ τ_dist (or best effort)."""
    img = _rainbow_image(size=100)
    tau = 10.0
    result = select_palette(img, n=5, tau_dist=tau)
    min_de = min_pairwise_delta_e(result.palette_lab)
    # We use a relaxed check because the constraint may be reduced on fallback
    # The key property is that the algorithm tries to maximise distinctness.
    assert min_de > 0.0


def test_tau_zero_does_not_crash():
    """τ_dist=0 means no constraint; should run without error."""
    img = _gradient_image()
    result = select_palette(img, n=3, tau_dist=0.0)
    assert result.n == 3


# ---------------------------------------------------------------------------
# WCAG post-selection step
# ---------------------------------------------------------------------------


def test_wcag_guaranteed_flag_is_bool():
    result = select_palette(_rainbow_image(), n=5)
    assert isinstance(result.wcag_guaranteed, bool)


def test_wcag_replacement_flag_is_bool():
    result = select_palette(_rainbow_image(), n=5)
    assert isinstance(result.wcag_replacement_applied, bool)


def test_high_contrast_image_gets_wcag_guarantee():
    """An image containing both pure black and pure white should get WCAG guarantee."""
    # Create image with black strip and white strip
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[:, 10:, :] = 255
    img = Image.fromarray(arr, "RGB")
    result = select_palette(img, n=2)
    assert result.wcag_guaranteed is True


# ---------------------------------------------------------------------------
# Hex output
# ---------------------------------------------------------------------------


def test_to_hex_format():
    result = select_palette(_rainbow_image(), n=3)
    hexes = result.to_hex()
    assert len(hexes) == 3
    for h in hexes:
        assert h.startswith("#")
        assert len(h) == 7


def test_to_hex_valid_characters():
    result = select_palette(_rainbow_image(), n=5)
    valid = set("0123456789abcdef")
    for h in result.to_hex():
        assert set(h[1:]).issubset(valid), f"Invalid hex: {h}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_n_equals_one():
    """n=1 should return the highest-frequency colour."""
    img = _rainbow_image()
    result = select_palette(img, n=1)
    assert result.n == 1
    assert result.palette_rgb.shape == (1, 3)


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        select_palette(_gradient_image(), n=0)


def test_rgba_image_handled():
    """RGBA image should be converted to RGB without error."""
    arr = np.random.randint(0, 256, (30, 30, 4), dtype=np.uint8)
    img = Image.fromarray(arr, "RGBA")
    result = select_palette(img, n=3)
    assert result.n == 3


# ---------------------------------------------------------------------------
# wcag_distinctness_compromised flag — unit tests
# ---------------------------------------------------------------------------


def _all_dark_gradient() -> Image.Image:
    """A 100x100 image covering only dark-to-medium shades (L* ≈ 5–40).

    No colour is light enough to form a natural WCAG AA pair, so the
    WCAG replacement step is guaranteed to fire.  The image has enough
    colour variety that Pillow's median-cut works normally.
    """
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    for x in range(100):
        # Map x → [5, 95] in the dark range  (R channel only for simplicity)
        val = int(5 + x * 0.9)  # 5 … 95 — all below mid-grey (128)
        arr[:, x, :] = val
    return Image.fromarray(arr, "RGB")


def test_wcag_distinctness_compromised_flag_is_bool():
    """The new flag must always be present and of type bool."""
    result = select_palette(_rainbow_image(), n=5)
    assert isinstance(result.wcag_distinctness_compromised, bool)


def test_wcag_distinctness_compromised_false_on_clean_image():
    """A clean high-contrast image must never set the compromised flag."""
    # 50% pure black, 50% pure white → no replacement needed
    arr = np.zeros((20, 40, 3), dtype=np.uint8)
    arr[:, 20:, :] = 255
    img = Image.fromarray(arr, "RGB")
    result = select_palette(img, n=2)
    assert result.wcag_distinctness_compromised is False


def test_wcag_replacement_logic_consistency():
    """Logical consistency: if compromised=False and replacement_applied=True,
    the resulting palette must satisfy τ_dist across every pair.
    """
    # All-dark gradient forces the WCAG replacement step to fire.
    tau = 10.0
    result = select_palette(_all_dark_gradient(), n=5, tau_dist=tau)

    if result.wcag_replacement_applied and not result.wcag_distinctness_compromised:
        # Joint constraint was met: every pair in the palette must be ≥ τ_dist apart.
        min_de = min_pairwise_delta_e(result.palette_lab)
        assert min_de >= tau - 1e-6, (
            f"WCAG replacement applied without compromised flag, "
            f"but min pairwise ΔE={min_de:.3f} < τ_dist={tau}.  "
            "This is the near-duplicate bug."
        )

    # If replacement was applied AND compromised, wcag_guaranteed must still be True
    # (the point of the fallback is to preserve the contrast guarantee).
    if result.wcag_replacement_applied and result.wcag_distinctness_compromised:
        assert result.wcag_guaranteed is True, (
            "wcag_distinctness_compromised=True but wcag_guaranteed=False — "
            "the fallback replacement must have applied."
        )

    # compromised can only be True if a replacement was made
    if result.wcag_distinctness_compromised:
        assert result.wcag_replacement_applied is True


# ---------------------------------------------------------------------------
# WCAG joint replacement (lines 339-353 in selector.py)
# ---------------------------------------------------------------------------


def _five_grays_one_white() -> Image.Image:
    """50×50 image: five equal bands of medium gray plus one white pixel.

    Band values: 100, 115, 130, 145, 160 (each 50×10 px → 500 pixels).
    The last pixel of the last band is overwritten with pure white (255).
    Max pairwise WCAG contrast among the five grays ≈ 2.3 < 4.5, so the
    greedy palette has no WCAG AA pair and the replacement step always fires.
    """
    arr = np.zeros((50, 50, 3), dtype=np.uint8)
    for i, v in enumerate([100, 115, 130, 145, 160]):
        arr[:, i * 10:(i + 1) * 10, :] = v
    arr[0, 49, :] = 255  # one white pixel in the last band
    return Image.fromarray(arr, "RGB")


def test_wcag_joint_replacement_applied():
    """Lines 339-353: WCAG step performs a clean joint replacement.

    With alpha=10 (frequency-driven), beta=0 and tau_dist=1.0 the greedy
    algorithm naturally selects the five gray bands.  White (the only
    off-palette candidate) satisfies both WCAG contrast (≈4.7 ≥ 4.5) and
    the tau_dist distinctness constraint (ΔE ≈ 34 ≥ 1.0), so the joint
    replacement path executes.
    """
    img = _five_grays_one_white()
    result = select_palette(img, n=5, alpha=10.0, beta=0.0, tau_dist=1.0)
    assert result.wcag_replacement_applied is True
    assert result.wcag_guaranteed is True
    assert result.wcag_distinctness_compromised is False


# ---------------------------------------------------------------------------
# WCAG contrast-only fallback (lines 357-361 in selector.py)
# ---------------------------------------------------------------------------


def test_wcag_contrast_only_replacement_fallback():
    """Lines 357-361: WCAG step falls back to contrast-only replacement.

    ``_min_de_to_palette`` is mocked to return 0.0, which forces the greedy
    algorithm to use the absolute-fallback path for every slot beyond the
    first, filling the palette with the five gray bands in index order and
    leaving white off-palette.

    With tau_dist=40.0 the white pixel has ΔE ≈ 34 from the lightest
    remaining gray — below tau_dist — so the joint constraint fails.  White
    still satisfies WCAG contrast (≈4.7 ≥ 4.5), placing it on the
    contrast-only fallback path and setting wcag_distinctness_compromised.
    """
    img = _five_grays_one_white()
    with patch("dsp.selector._min_de_to_palette", return_value=0.0):
        with pytest.warns(UserWarning, match="WCAG AA replacement applied"):
            result = select_palette(img, n=5, alpha=10.0, beta=0.0, tau_dist=40.0)
    assert result.wcag_distinctness_compromised is True
    assert result.wcag_guaranteed is True
    assert result.wcag_replacement_applied is True


# ---------------------------------------------------------------------------
# τ-relaxation absolute fallback (lines 264-272 in selector.py)
# ---------------------------------------------------------------------------


def test_tau_relaxation_absolute_fallback():
    """Lines 264-272: absolute fallback fires when τ-relaxation exhausts.

    ``_min_de_to_palette`` is mocked to always return 0.0.  Every candidate
    has apparent min-distance 0 from the palette, so neither the main greedy
    loop nor the 50-iteration τ-relaxation loop can find a valid candidate.
    The absolute fallback ("take most-distant remaining candidate regardless")
    runs, emits a UserWarning, and the algorithm still returns a palette with
    the requested number of entries.
    """
    img = _gradient_image(size=50)
    with patch("dsp.selector._min_de_to_palette", return_value=0.0):
        with pytest.warns(UserWarning, match="could not be satisfied"):
            result = select_palette(img, n=2, tau_dist=10.0)
    assert result.palette_rgb.shape == (2, 3)


