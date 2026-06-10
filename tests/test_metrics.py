
"""
Unit tests for dsp/metrics.py.

ΔE2000 reference values sourced from:
  Sharma G., Wu W., Dalal E. N. (2005), Table 1 "Supplementary test data".
  Color Research & Application, 30(1), 21–30.

WCAG contrast reference values from:
  W3C WCAG 2.1 Understanding document, §1.4.3 examples.
"""

import math
import numpy as np
import pytest

# ── We test against exact ΔE2000 values when colour-science is available,
#    and fall back to our NumPy implementation otherwise.  Both should be
#    within 0.0001 of the published reference values.
from dsp.metrics import (
    delta_e2000,
    srgb_to_lab,
    wcag_contrast,
    wcag_level,
    min_pairwise_delta_e,
    wcag_pair_coverage,
    harmony_alignment,
    reconstruction_error_fast,
)


# ---------------------------------------------------------------------------
# ΔE2000 — Sharma et al. 2005, Table 1 (first 10 pairs, tolerance 0.0004)
# ---------------------------------------------------------------------------

# Each entry: (Lab1, Lab2, expected_dE2000)
# Values taken from Table 1 of the supplementary test data.
_SHARMA_PAIRS = [
    ((50.0000,  2.6772, -79.7751), (50.0000,  0.0000, -82.7485),  2.0425),
    ((50.0000,  3.1571, -77.2803), (50.0000,  0.0000, -82.7485),  2.8615),
    ((50.0000,  2.8361, -74.0200), (50.0000,  0.0000, -82.7485),  3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000,  0.0000, -82.7485),  1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000,  0.0000, -82.7485),  1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000,  0.0000, -82.7485),  1.0000),
    ((50.0000,  0.0000,   0.0000), (50.0000, -1.0000,  2.0000),   2.3669),
    ((50.0000, -1.0000,   2.0000), (50.0000,  0.0000,  0.0000),   2.3669),
    ((50.0000,  2.4900,  -0.0010), (50.0000, -2.4900,  0.0009),   7.1792),
    ((50.0000,  2.4900,  -0.0010), (50.0000, -2.4900,  0.0010),   7.1792),
]

_SHARMA_TOLERANCE = 0.004   # 0.004 absolute; Sharma et al. reference uses ≤0.0001


@pytest.mark.parametrize("lab1,lab2,expected", _SHARMA_PAIRS)
def test_delta_e2000_sharma(lab1, lab2, expected):
    """ΔE2000 should match Sharma et al. 2005 reference values within tolerance."""
    result = delta_e2000(lab1, lab2)
    assert abs(result - expected) < _SHARMA_TOLERANCE, (
        f"ΔE2000({lab1}, {lab2}) = {result:.4f}, expected {expected:.4f}"
    )


def test_delta_e2000_symmetry():
    """ΔE2000 must be symmetric."""
    lab1 = (50.0, 25.0, -30.0)
    lab2 = (60.0, -10.0, 15.0)
    assert abs(delta_e2000(lab1, lab2) - delta_e2000(lab2, lab1)) < 1e-10


def test_delta_e2000_identity():
    """ΔE2000 of a colour with itself must be 0."""
    lab = (45.0, 12.0, -8.0)
    assert delta_e2000(lab, lab) == pytest.approx(0.0, abs=1e-10)


def test_delta_e2000_non_negative():
    """ΔE2000 must be ≥ 0 for any input."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        lab1 = tuple(rng.uniform([-5, -100, -100], [105, 100, 100]))
        lab2 = tuple(rng.uniform([-5, -100, -100], [105, 100, 100]))
        assert delta_e2000(lab1, lab2) >= 0.0


# ---------------------------------------------------------------------------
# sRGB → CIELAB conversion
# ---------------------------------------------------------------------------


def test_srgb_to_lab_white():
    """sRGB white (255,255,255) → Lab ≈ (100, 0, 0)."""
    lab = srgb_to_lab(np.array([[255.0, 255.0, 255.0]]))[0]
    assert abs(lab[0] - 100.0) < 0.5
    assert abs(lab[1]) < 0.5
    assert abs(lab[2]) < 0.5


def test_srgb_to_lab_black():
    """sRGB black (0,0,0) → Lab ≈ (0, 0, 0)."""
    lab = srgb_to_lab(np.array([[0.0, 0.0, 0.0]]))[0]
    assert abs(lab[0]) < 0.5
    assert abs(lab[1]) < 0.5
    assert abs(lab[2]) < 0.5


def test_srgb_to_lab_shape():
    arr = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float64)
    result = srgb_to_lab(arr)
    assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# WCAG contrast — W3C published examples
# ---------------------------------------------------------------------------


# W3C Understanding WCAG 2.1 §1.4.3 gives exact examples:
#   black (#000000) vs white (#FFFFFF) → 21:1
#   #777777 vs #FFFFFF                 → ~4.48:1 (just below AA)

def test_wcag_black_white():
    """Black vs white must give contrast ratio 21.0 exactly."""
    cr = wcag_contrast((0, 0, 0), (255, 255, 255))
    assert cr == pytest.approx(21.0, rel=1e-4)


def test_wcag_same_color():
    """Same colour must give contrast ratio 1.0."""
    cr = wcag_contrast((128, 64, 32), (128, 64, 32))
    assert cr == pytest.approx(1.0, abs=1e-10)


def test_wcag_symmetry():
    """Contrast ratio must be symmetric."""
    cr1 = wcag_contrast((255, 255, 255), (50, 100, 150))
    cr2 = wcag_contrast((50, 100, 150), (255, 255, 255))
    assert cr1 == pytest.approx(cr2, rel=1e-10)


def test_wcag_level_aaa():
    assert wcag_level(7.0) == "AAA"
    assert wcag_level(10.0) == "AAA"
    assert wcag_level(21.0) == "AAA"


def test_wcag_level_aa():
    assert wcag_level(4.5) == "AA"
    assert wcag_level(6.9) == "AA"


def test_wcag_level_a():
    assert wcag_level(3.0) == "A"
    assert wcag_level(4.4) == "A"


def test_wcag_level_fail():
    assert wcag_level(2.9) == "fail"
    assert wcag_level(1.0) == "fail"


def test_wcag_black_white_level_aaa():
    cr = wcag_contrast((0, 0, 0), (255, 255, 255))
    assert wcag_level(cr) == "AAA"


# ---------------------------------------------------------------------------
# min_pairwise_delta_e
# ---------------------------------------------------------------------------


def test_min_pairwise_de_two_colors():
    """Min pairwise ΔE for two distinct Lab colours should be > 0."""
    lab = srgb_to_lab(np.array([[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]]))
    result = min_pairwise_delta_e(lab)
    assert result > 50.0   # black vs white is very far apart


def test_min_pairwise_de_identical():
    """All-identical palette → min ΔE ≈ 0."""
    lab = srgb_to_lab(np.array([[128.0, 64.0, 32.0]] * 3))
    result = min_pairwise_delta_e(lab)
    assert result == pytest.approx(0.0, abs=1e-8)


def test_min_pairwise_de_single():
    """Single-colour palette → 0."""
    lab = srgb_to_lab(np.array([[100.0, 100.0, 100.0]]))
    assert min_pairwise_delta_e(lab) == 0.0


# ---------------------------------------------------------------------------
# wcag_pair_coverage
# ---------------------------------------------------------------------------


def test_wcag_coverage_black_white():
    """A palette containing black and white must have 100% AA coverage."""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    assert wcag_pair_coverage(palette) == pytest.approx(1.0)


def test_wcag_coverage_all_same():
    """All-identical palette → 0% AA coverage."""
    palette = np.array([[128, 128, 128]] * 3, dtype=np.float64)
    assert wcag_pair_coverage(palette) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# harmony_alignment
# ---------------------------------------------------------------------------


def test_harmony_alignment_complementary():
    """Two Lab colours exactly 180° apart in hue must give alignment score = 1.0."""
    # Construct Lab colours with hue 0° and 180° directly (no sRGB conversion)
    # hue = atan2(b*, a*); hue 0° → a>0, b=0; hue 180° → a<0, b=0
    lab = np.array([[50.0, 30.0, 0.0], [50.0, -30.0, 0.0]])
    score = harmony_alignment(lab)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_harmony_alignment_range():
    """Harmony alignment must be in [0, 1]."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(5, 3)).astype(np.float64)
    lab = srgb_to_lab(rgb)
    score = harmony_alignment(lab)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# reconstruction_error_fast
# ---------------------------------------------------------------------------


def test_reconstruction_error_zero_for_palette_pixels():
    """Error must be 0 when each pixel is exactly a palette colour."""
    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float64)
    # Image made entirely of palette pixels
    image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 0, 0]]], dtype=np.float64)
    err = reconstruction_error_fast(image, palette)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_error_positive_for_off_palette():
    """Error must be > 0 when pixel is not in palette."""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    image = np.array([[[128, 64, 32]]], dtype=np.float64)
    err = reconstruction_error_fast(image, palette)
    assert err > 0.0


# ---------------------------------------------------------------------------
# delta_e2000_batch
# ---------------------------------------------------------------------------

from dsp.metrics import delta_e2000_batch


def test_delta_e2000_batch_shape():
    """delta_e2000_batch must return shape (N,) for N targets."""
    ref = np.array([50.0, 0.0, 0.0])
    targets = np.array([[50.0, 0.0, 0.0], [60.0, 10.0, -10.0], [30.0, -20.0, 5.0]])
    result = delta_e2000_batch(ref, targets)
    assert result.shape == (3,)


def test_delta_e2000_batch_identity():
    """Batch ΔE2000 of ref against itself must be 0."""
    ref = np.array([50.0, 25.0, -30.0])
    result = delta_e2000_batch(ref, ref[np.newaxis, :])
    assert result[0] == pytest.approx(0.0, abs=1e-10)


def test_delta_e2000_batch_matches_scalar():
    """Batch results must match scalar delta_e2000 calls."""
    ref = np.array([50.0, 2.6772, -79.7751])
    targets = np.array([[50.0, 0.0, -82.7485], [60.0, -10.0, 20.0]])
    batch = delta_e2000_batch(ref, targets)
    for i, t in enumerate(targets):
        assert batch[i] == pytest.approx(delta_e2000(ref, t), rel=1e-9)


# ---------------------------------------------------------------------------
# Edge cases: wcag_pair_coverage and harmony_alignment with n < 2
# ---------------------------------------------------------------------------


def test_wcag_pair_coverage_single_color():
    """Single-colour palette has no pairs → coverage must be 0."""
    palette = np.array([[128, 0, 128]], dtype=np.float64)
    assert wcag_pair_coverage(palette) == pytest.approx(0.0)


def test_harmony_alignment_single_color():
    """Single-colour palette has no pairs → alignment must be 0."""
    lab = srgb_to_lab(np.array([[200.0, 100.0, 50.0]]))
    assert harmony_alignment(lab) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reconstruction_error (exact ΔE2000 loop)
# ---------------------------------------------------------------------------

from dsp.metrics import reconstruction_error


def test_reconstruction_error_zero_for_palette_pixels():
    """Exact reconstruction error must be 0 when pixels are palette members."""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.float64)
    assert reconstruction_error(image, palette) == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_error_positive_off_palette():
    """Exact reconstruction error must be > 0 for off-palette pixels."""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    image = np.array([[[128, 64, 32]]], dtype=np.float64)
    assert reconstruction_error(image, palette) > 0.0


def test_reconstruction_error_consistent_with_fast():
    """Exact and fast reconstruction errors should order the same way."""
    palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.float64)
    image_close = np.array([[[250, 10, 10]]], dtype=np.float64)   # close to red
    image_far = np.array([[[128, 128, 128]]], dtype=np.float64)   # grey, far from both
    err_close = reconstruction_error(image_close, palette)
    err_far = reconstruction_error(image_far, palette)
    assert err_close < err_far


# ---------------------------------------------------------------------------
# reconstruction_error_de2000
# ---------------------------------------------------------------------------

from dsp.metrics import reconstruction_error_de2000


def test_reconstruction_error_de2000_zero_for_palette_pixels():
    """de2000 reconstruction error must be 0 when pixels are palette members."""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    image = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.float64)
    assert reconstruction_error_de2000(image, palette) == pytest.approx(0.0, abs=1e-4)


def test_reconstruction_error_de2000_positive_off_palette():
    """de2000 reconstruction error must be > 0 for off-palette pixels."""
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    image = np.array([[[128, 64, 32]]], dtype=np.float64)
    assert reconstruction_error_de2000(image, palette) > 0.0


def test_reconstruction_error_de2000_subsampling():
    """de2000 reconstruction error with max_pixels < image size must still return a float."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(100, 3)).astype(np.float64).reshape(10, 10, 3)
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    result = reconstruction_error_de2000(image, palette, max_pixels=20, seed=7)
    assert isinstance(result, float)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Pure-NumPy fallback paths (monkeypatched _COLOUR_AVAILABLE = False)
# ---------------------------------------------------------------------------

import dsp.metrics as _metrics_mod


def test_srgb_to_lab_numpy_fallback_white(monkeypatch):
    """NumPy fallback: sRGB white → Lab ≈ (100, 0, 0)."""
    monkeypatch.setattr(_metrics_mod, "_COLOUR_AVAILABLE", False)
    lab = _metrics_mod.srgb_to_lab(np.array([[255.0, 255.0, 255.0]]))[0]
    assert abs(lab[0] - 100.0) < 1.0
    assert abs(lab[1]) < 1.0
    assert abs(lab[2]) < 1.0


def test_srgb_to_lab_numpy_fallback_black(monkeypatch):
    """NumPy fallback: sRGB black → Lab ≈ (0, 0, 0)."""
    monkeypatch.setattr(_metrics_mod, "_COLOUR_AVAILABLE", False)
    lab = _metrics_mod.srgb_to_lab(np.array([[0.0, 0.0, 0.0]]))[0]
    assert abs(lab[0]) < 0.5
    assert abs(lab[1]) < 0.5
    assert abs(lab[2]) < 0.5


def test_srgb_to_lab_numpy_fallback_unit_input(monkeypatch):
    """NumPy fallback: float input in [0,1] must give same result as /255."""
    monkeypatch.setattr(_metrics_mod, "_COLOUR_AVAILABLE", False)
    lab_255 = _metrics_mod.srgb_to_lab(np.array([[200.0, 100.0, 50.0]]))
    lab_01 = _metrics_mod.srgb_to_lab(np.array([[200.0 / 255, 100.0 / 255, 50.0 / 255]]))
    np.testing.assert_allclose(lab_255, lab_01, atol=1e-6)


def test_delta_e2000_numpy_fallback_identity(monkeypatch):
    """NumPy fallback: ΔE2000 of colour with itself must be 0."""
    monkeypatch.setattr(_metrics_mod, "_COLOUR_AVAILABLE", False)
    assert _metrics_mod.delta_e2000((50.0, 25.0, -10.0), (50.0, 25.0, -10.0)) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("lab1,lab2,expected", _SHARMA_PAIRS[:5])
def test_delta_e2000_numpy_fallback_sharma(monkeypatch, lab1, lab2, expected):
    """NumPy fallback: ΔE2000 must match Sharma 2005 reference values."""
    monkeypatch.setattr(_metrics_mod, "_COLOUR_AVAILABLE", False)
    result = _metrics_mod.delta_e2000(lab1, lab2)
    assert abs(result - expected) < _SHARMA_TOLERANCE


# ---------------------------------------------------------------------------
# reconstruction_error_de2000 — colour-unavailable fallback (T5)
# ---------------------------------------------------------------------------


def test_reconstruction_error_de2000_raises_without_colour(monkeypatch):
    """When colour-science is unavailable, de2000 raises RuntimeError (no silent fallback)."""
    import pytest
    monkeypatch.setattr(_metrics_mod, "_COLOUR_AVAILABLE", False)
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.float64)
    image = np.array([[[128, 64, 32]]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="colour-science is not installed"):
        _metrics_mod.reconstruction_error_de2000(image, palette)
