
"""
Unit tests for dsp/roles.py — semantic role assignment.
"""

import numpy as np
import pytest

from dsp.metrics import srgb_to_lab
from dsp.roles import RoleAssignment, assign_roles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_palette(rgb_list: list[tuple[int, int, int]]):
    """Convert list of RGB tuples to (palette_rgb, palette_lab, freqs)."""
    rgb = np.array(rgb_list, dtype=np.float64)
    lab = srgb_to_lab(rgb)
    freqs = [1.0 / len(rgb_list)] * len(rgb_list)
    return rgb.astype(np.uint8), lab, freqs


# ---------------------------------------------------------------------------
# surface = lightest (highest L*)
# ---------------------------------------------------------------------------


def test_surface_is_lightest():
    """surface must be the palette member with the highest L*."""
    rgb_list = [
        (20, 20, 20),   # very dark → low L*
        (200, 200, 200),  # light → high L*
        (100, 0, 200),   # mid-dark
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    assert result.surface is not None
    # Index 1 is (200,200,200) — should be selected as surface
    assert result.surface == 1


def test_surface_is_white_in_bw_palette():
    """With pure black and white, surface must be white (index 1 if second)."""
    rgb_list = [(0, 0, 0), (255, 255, 255)]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    assert result.surface == 1  # white is second entry


# ---------------------------------------------------------------------------
# on-surface = highest contrast against surface
# ---------------------------------------------------------------------------


def test_on_surface_max_contrast_against_surface():
    """on-surface must be the member with highest WCAG contrast vs surface."""
    from dsp.metrics import wcag_contrast

    rgb_list = [
        (255, 255, 255),  # surface (lightest)
        (10, 10, 10),     # near-black — should be on-surface
        (180, 180, 180),  # mid-grey — lower contrast
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)

    assert result.surface == 0  # white is index 0
    assert result.on_surface is not None
    # Near-black (index 1) should have highest contrast against white
    cr_1 = wcag_contrast(rgb_list[1], rgb_list[0])
    cr_2 = wcag_contrast(rgb_list[2], rgb_list[0])
    expected_on_surface = 1 if cr_1 >= cr_2 else 2
    assert result.on_surface == expected_on_surface


# ---------------------------------------------------------------------------
# primary = highest-frequency remaining (after surface + on-surface)
# ---------------------------------------------------------------------------


def test_primary_highest_frequency():
    """primary must be the highest-frequency remaining member."""
    rgb_list = [
        (255, 255, 255),  # index 0 → surface (lightest)
        (0, 0, 0),        # index 1 → on-surface (max contrast vs white)
        (200, 50, 50),    # index 2 → low frequency
        (50, 100, 200),   # index 3 → high frequency
    ]
    rgb = np.array(rgb_list, dtype=np.uint8)
    lab = srgb_to_lab(rgb.astype(np.float64))
    # Give index 3 a high frequency
    freqs = [0.05, 0.05, 0.10, 0.80]
    result = assign_roles(rgb, lab, freqs)
    assert result.primary == 3


# ---------------------------------------------------------------------------
# accent = highest chroma
# ---------------------------------------------------------------------------


def test_accent_highest_chroma():
    """accent must be the remaining member with highest C*ab."""
    import math

    rgb_list = [
        (240, 240, 240),  # near-white → surface
        (10, 10, 10),     # near-black → on-surface
        (180, 0, 0),      # high chroma red
        (100, 0, 200),    # high chroma purple
        (100, 100, 120),  # low chroma grey-blue
    ]
    rgb = np.array(rgb_list, dtype=np.uint8)
    lab = srgb_to_lab(rgb.astype(np.float64))
    freqs = [0.2, 0.2, 0.2, 0.2, 0.2]
    result = assign_roles(rgb, lab, freqs)

    assert result.accent is not None
    accent_chroma = math.sqrt(lab[result.accent, 1] ** 2 + lab[result.accent, 2] ** 2)

    # All non-assigned members must have chroma ≤ accent chroma
    assigned = {result.surface, result.on_surface, result.primary, result.secondary, result.accent}
    for i in range(len(rgb_list)):
        if i not in assigned:
            continue  # extra
        c = math.sqrt(lab[i, 1] ** 2 + lab[i, 2] ** 2)
        if i != result.accent:
            assert c <= accent_chroma + 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_colour_palette():
    """Single colour → only surface assigned, rest None."""
    rgb_list = [(128, 64, 32)]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    assert result.surface == 0
    assert result.on_surface is None
    assert result.primary is None
    assert result.secondary is None
    assert result.accent is None
    assert result.extras == []


def test_two_colour_palette():
    """Two colours → surface + on-surface assigned, rest None."""
    rgb_list = [(255, 255, 255), (0, 0, 0)]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    assert result.surface is not None
    assert result.on_surface is not None
    assert result.primary is None
    assert result.secondary is None
    assert result.accent is None


def test_five_colour_palette_all_roles_assigned():
    """n=5 palette must have all 5 named roles assigned (no extras, no None)."""
    rgb_list = [
        (240, 240, 240),
        (20, 20, 20),
        (200, 50, 50),
        (50, 100, 200),
        (50, 180, 50),
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    assert result.surface is not None
    assert result.on_surface is not None
    assert result.primary is not None
    assert result.secondary is not None
    assert result.accent is not None
    assert result.extras == []


def test_six_colour_palette_has_one_extra():
    """n=6 palette must produce one extra."""
    rgb_list = [
        (240, 240, 240),
        (20, 20, 20),
        (200, 50, 50),
        (50, 100, 200),
        (50, 180, 50),
        (255, 200, 0),
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    assert len(result.extras) == 1


def test_roles_map_covers_all_indices():
    """roles_map must contain every palette index exactly once."""
    rgb_list = [
        (240, 240, 240),
        (20, 20, 20),
        (200, 50, 50),
        (50, 100, 200),
        (50, 180, 50),
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    result = assign_roles(rgb, lab, freqs)
    roles_map = result.roles_map
    assert set(roles_map.keys()) == set(range(len(rgb_list)))


def test_empty_palette():
    """Empty palette returns all-None RoleAssignment."""
    rgb = np.zeros((0, 3), dtype=np.uint8)
    lab = np.zeros((0, 3), dtype=np.float64)
    result = assign_roles(rgb, lab, [])
    assert result.surface is None
    assert result.extras == []


# ---------------------------------------------------------------------------
# mode='auto' with explicit image_mean_L
# ---------------------------------------------------------------------------


def test_assign_roles_auto_dark_mode():
    """mode='auto' with image_mean_L<40 must select dark mode (surface=darkest)."""
    rgb_list = [
        (20, 20, 20),     # darkest → surface in dark mode
        (200, 200, 200),  # lightest
        (100, 50, 200),   # mid
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    # image_mean_L=30 < 40 → dark mode
    result = assign_roles(rgb, lab, freqs, mode="auto", image_mean_L=30.0)
    # In dark mode surface is the member with the lowest L*
    lowest_L_idx = int(np.argmin(lab[:, 0]))
    assert result.surface == lowest_L_idx


def test_assign_roles_auto_light_mode():
    """mode='auto' with image_mean_L≥40 must select light mode (surface=lightest)."""
    rgb_list = [
        (20, 20, 20),     # darkest
        (200, 200, 200),  # lightest → surface in light mode
        (100, 50, 200),   # mid
    ]
    rgb, lab, freqs = _make_palette(rgb_list)
    # image_mean_L=60 ≥ 40 → light mode
    result = assign_roles(rgb, lab, freqs, mode="auto", image_mean_L=60.0)
    # In light mode surface is the member with the highest L*
    highest_L_idx = int(np.argmax(lab[:, 0]))
    assert result.surface == highest_L_idx
