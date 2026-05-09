
"""
Semantic role assignment for extracted palettes.

Heuristic role assignment (documented as such).  Learning role assignment
from design-system corpora is identified as future work.

Roles (in fill-priority order for n < 5)
-----------------------------------------
surface      – lightest member (highest L*) in light mode;
               darkest member (lowest L*) in dark mode
on-surface   – highest WCAG contrast against surface
primary      – highest-frequency member among remaining, satisfying
               min ΔE ≥ τ_role from surface (default 10)
secondary    – remaining member maximally distant in hue from primary
accent       – remaining member with highest chroma (C*ab = sqrt(a*²+b*²))
extras       – any remaining members for n > 5, in order of frequency

mode parameter
--------------
``mode='light'`` (default)
    surface = highest L* member.  Suitable for white/cream-background
    design systems.

``mode='dark'``
    surface = lowest L* member.  Suitable for dark-theme design systems
    where the dominant background is near-black.

``mode='auto'``
    Selects light or dark automatically based on the image mean L*
    (passed via ``image_mean_L``).  Heuristic:

        mode = dark  if  mean_L* < 40  else  light

    If ``image_mean_L`` is None in auto mode, the function falls back to
    frequency-weighted mean L* of the palette itself as a proxy.

Public API
----------
assign_roles(palette_rgb, palette_lab, frequencies, tau_role,
             mode, image_mean_L) -> RoleAssignment
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .metrics import wcag_contrast, delta_e2000


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class RoleAssignment:
    """Semantic role assignment for a palette.

    Each entry is an index into the original palette arrays, or None if the
    palette is too small to fill that role.

    Attributes
    ----------
    surface : int | None
    on_surface : int | None
    primary : int | None
    secondary : int | None
    accent : int | None
    extras : list[int]
        Indices of palette members not assigned to a named role (n > 5).
    roles_map : dict[int, str]
        Palette-index → role-name mapping for convenient lookup.
    """

    surface: Optional[int]
    on_surface: Optional[int]
    primary: Optional[int]
    secondary: Optional[int]
    accent: Optional[int]
    extras: list[int] = field(default_factory=list)

    @property
    def roles_map(self) -> dict[int, str]:
        """Return {palette_index: role_name} dict."""
        out: dict[int, str] = {}
        for role, idx in [
            ("surface", self.surface),
            ("on-surface", self.on_surface),
            ("primary", self.primary),
            ("secondary", self.secondary),
            ("accent", self.accent),
        ]:
            if idx is not None:
                out[idx] = role
        for i, idx in enumerate(self.extras):
            out[idx] = f"extra-{i + 1}"
        return out


# ---------------------------------------------------------------------------
# Assignment logic
# ---------------------------------------------------------------------------


def assign_roles(
    palette_rgb: NDArray[np.uint8 | np.float64],
    palette_lab: NDArray[np.float64],
    frequencies: list[float],
    tau_role: float = 10.0,
    mode: Literal["light", "dark", "auto"] = "light",
    image_mean_L: Optional[float] = None,
) -> RoleAssignment:
    """Assign semantic design-system roles to a palette.

    Parameters
    ----------
    palette_rgb:
        Shape (n, 3) sRGB colours (0–255).
    palette_lab:
        Shape (n, 3) CIELAB colours corresponding to palette_rgb.
    frequencies:
        Length-n list of normalised pixel frequencies (0–1).
    tau_role:
        Minimum ΔE2000 required between primary candidate and surface.
        Candidates closer than this to surface are excluded from primary
        (they are too similar to surface to be useful as a primary token).
        Default 10.
    mode:
        ``'light'`` — surface = highest L* (default, matches most brand
        design systems).
        ``'dark'``  — surface = lowest L* (dark-theme schema).
        ``'auto'``  — choose light/dark based on ``image_mean_L``:
        dark if mean L* < 40, else light.  Falls back to
        frequency-weighted mean L* of the palette if ``image_mean_L``
        is None.
    image_mean_L:
        Pre-computed mean L* of the full image (used by ``mode='auto'``).
        If None and mode is 'auto', computed from palette frequencies.

    Returns
    -------
    RoleAssignment
    """
    n = len(palette_rgb)
    if n == 0:
        return RoleAssignment(
            surface=None,
            on_surface=None,
            primary=None,
            secondary=None,
            accent=None,
            extras=[],
        )

    available = list(range(n))

    # ------------------------------------------------------------------
    # Resolve effective mode
    # ------------------------------------------------------------------
    if mode == "auto":
        if image_mean_L is not None:
            mean_L = image_mean_L
        else:
            # Proxy: frequency-weighted mean L* of palette
            mean_L = float(
                sum(frequencies[i] * float(palette_lab[i, 0]) for i in range(n))
            )
        effective_mode: Literal["light", "dark"] = "dark" if mean_L < 40.0 else "light"
    else:
        effective_mode = mode

    # ------------------------------------------------------------------
    # surface — highest L* (light mode) or lowest L* (dark mode)
    # ------------------------------------------------------------------
    if effective_mode == "dark":
        surface_idx = min(available, key=lambda i: float(palette_lab[i, 0]))
    else:
        surface_idx = max(available, key=lambda i: float(palette_lab[i, 0]))
    available.remove(surface_idx)

    if not available:
        return RoleAssignment(
            surface=surface_idx,
            on_surface=None,
            primary=None,
            secondary=None,
            accent=None,
            extras=[],
        )

    # ------------------------------------------------------------------
    # on-surface — highest WCAG contrast against surface
    # ------------------------------------------------------------------
    on_surface_idx = max(
        available,
        key=lambda i: wcag_contrast(palette_rgb[i], palette_rgb[surface_idx]),
    )
    available.remove(on_surface_idx)

    if not available:
        return RoleAssignment(
            surface=surface_idx,
            on_surface=on_surface_idx,
            primary=None,
            secondary=None,
            accent=None,
            extras=[],
        )

    # ------------------------------------------------------------------
    # primary — highest-frequency among remaining, ΔE ≥ τ_role from surface
    # ------------------------------------------------------------------
    # Prefer candidates that are distinct from surface; fall back to all remaining
    eligible_primary = [
        i for i in available
        if delta_e2000(palette_lab[i], palette_lab[surface_idx]) >= tau_role
    ]
    if not eligible_primary:
        # Surface is very similar to everything — use all remaining
        eligible_primary = available[:]

    primary_idx = max(eligible_primary, key=lambda i: frequencies[i])
    available.remove(primary_idx)

    if not available:
        return RoleAssignment(
            surface=surface_idx,
            on_surface=on_surface_idx,
            primary=primary_idx,
            secondary=None,
            accent=None,
            extras=[],
        )

    # ------------------------------------------------------------------
    # secondary — most distant in hue from primary
    # ------------------------------------------------------------------
    primary_hue = math.degrees(
        math.atan2(float(palette_lab[primary_idx, 2]), float(palette_lab[primary_idx, 1]))
    ) % 360.0

    def _hue_distance(idx: int) -> float:
        h = math.degrees(
            math.atan2(float(palette_lab[idx, 2]), float(palette_lab[idx, 1]))
        ) % 360.0
        diff = abs(h - primary_hue)
        return min(diff, 360.0 - diff)

    secondary_idx = max(available, key=_hue_distance)
    available.remove(secondary_idx)

    if not available:
        return RoleAssignment(
            surface=surface_idx,
            on_surface=on_surface_idx,
            primary=primary_idx,
            secondary=secondary_idx,
            accent=None,
            extras=[],
        )

    # ------------------------------------------------------------------
    # accent — highest chroma C*ab = sqrt(a*² + b*²)
    # ------------------------------------------------------------------
    def _chroma(idx: int) -> float:
        a, b = float(palette_lab[idx, 1]), float(palette_lab[idx, 2])
        return math.sqrt(a**2 + b**2)

    accent_idx = max(available, key=_chroma)
    available.remove(accent_idx)

    # ------------------------------------------------------------------
    # extras — all remaining members
    # ------------------------------------------------------------------
    # Sort extras by descending frequency so downstream users get a
    # deterministic ordering.
    extras = sorted(available, key=lambda i: frequencies[i], reverse=True)

    return RoleAssignment(
        surface=surface_idx,
        on_surface=on_surface_idx,
        primary=primary_idx,
        secondary=secondary_idx,
        accent=accent_idx,
        extras=extras,
    )
