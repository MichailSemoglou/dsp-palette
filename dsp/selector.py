
"""
Constrained greedy palette selection in CIELAB (DSP method).

Algorithm
---------
1. Quantize image → candidate pool (~256 colours) with per-colour frequencies.
2. Convert candidates to CIELAB (D65).
3. Initialise palette with the highest-frequency candidate.
4. Greedy expand: pick candidate maximising
       score(c) = α·log(frequency(c)) + β·min_ΔE2000(c, palette)
   subject to  min_ΔE2000(c, palette) ≥ τ_dist.
5. Post-selection WCAG AA check: if no pair in the palette achieves contrast
   ≥ 4.5:1, replace the least distinct palette member with a candidate that
   (a) creates a qualifying contrast pair (≥ 4.5:1) AND
   (b) satisfies min_ΔE2000(candidate, remaining palette) ≥ τ_dist.
   If no candidate satisfies both constraints, fall back to contrast-only
   (``wcag_distinctness_compromised=True``) or report failure
   (``wcag_guaranteed=False``) for monochromatic images.
6. Return a ``SelectionResult`` dataframe-ready object with palette, Lab values,
   frequencies, and metadata flags.

Public API
----------
select_from_candidates(candidates_rgb, freqs, n, alpha, beta, tau_dist, wcag_step) -> SelectionResult
select_palette(image, n, alpha, beta, tau_dist, max_candidates, wcag_step) -> SelectionResult
"""

import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .metrics import (
    delta_e2000,
    srgb_to_lab,
    wcag_contrast,
    WCAG_AA_NORMAL,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SelectionResult:
    """Full output of the constrained greedy selector.

    Attributes
    ----------
    palette_rgb : NDArray shape (n, 3)
        Final palette in sRGB 0–255 uint8.
    palette_lab : NDArray shape (n, 3)
        Final palette in CIELAB.
    frequencies : list[float]
        Normalised frequency (0–1) of each palette colour in the source image.
    n : int
        Requested palette size.
    alpha : float
        Weight on log-frequency term.
    beta : float
        Weight on min-ΔE2000 term.
    tau_dist : float
        Minimum ΔE2000 threshold used during selection.
    wcag_guaranteed : bool
        True if at least one palette pair satisfies WCAG AA contrast (≥4.5:1)
        after the post-selection replacement step.
    wcag_replacement_applied : bool
        True if the post-selection WCAG step replaced a palette member.
    wcag_distinctness_compromised : bool
        True if the WCAG replacement step could only find a contrast-satisfying
        candidate that violates τ_dist (i.e. the joint contrast+distinctness
        constraint could not be met simultaneously).  The replacement is still
        applied so ``wcag_guaranteed`` stays True, but the palette may contain
        a near-duplicate pair.
    candidate_pool_size : int
        Actual number of distinct candidate colours considered.
    """

    palette_rgb: NDArray[np.uint8]
    palette_lab: NDArray[np.float64]
    frequencies: list[float]
    n: int
    alpha: float
    beta: float
    tau_dist: float
    wcag_guaranteed: bool
    wcag_replacement_applied: bool
    wcag_distinctness_compromised: bool
    candidate_pool_size: int

    def to_hex(self) -> list[str]:
        """Return palette as a list of hex strings (e.g. '#1a2b3c')."""
        return [
            "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
            for r, g, b in self.palette_rgb
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _quantize_to_candidates(
    image: Image.Image,
    max_candidates: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Median-cut quantization → (colours, normalised_frequencies).

    Returns
    -------
    colours : NDArray shape (K, 3) uint8
    freqs   : NDArray shape (K,)  float64, sum = 1.0
    """
    # Ensure RGB (drop alpha, handle palette-mode images)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Pillow median-cut quantize
    quantized = image.quantize(colors=max_candidates, method=Image.Quantize.MEDIANCUT)
    quantized_rgb = quantized.convert("RGB")

    # Count pixel frequencies
    pixels = np.array(quantized_rgb, dtype=np.uint8).reshape(-1, 3)
    # Build a dict keyed by tuple for fast counting
    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    total = counts.sum()
    return unique, counts.astype(np.float64) / total


def _min_de_to_palette(
    lab_candidate: NDArray[np.float64],
    palette_lab: list[NDArray[np.float64]],
) -> float:
    """Minimum ΔE2000 between a candidate Lab colour and all palette members."""
    return min(delta_e2000(lab_candidate, p) for p in palette_lab)


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------


def select_from_candidates(
    candidates_rgb: NDArray[np.uint8],
    freqs: NDArray[np.float64],
    n: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    tau_dist: float = 10.0,
    wcag_step: bool = True,
) -> SelectionResult:
    """Run the constrained greedy DSP selection on a pre-built candidate pool.

    Used by :func:`select_palette` and by constraint-matched baselines that
    supply their own quantisation pool (e.g.
    ``research.baselines.kmeans_lab_constrained``).

    Parameters
    ----------
    candidates_rgb : NDArray shape (K, 3) uint8
        Candidate colours in sRGB 0–255.
    freqs : NDArray shape (K,) float64
        Normalised pixel frequencies for each candidate (values sum ≈ 1.0).
    n, alpha, beta, tau_dist :
        Same semantics as :func:`select_palette`.
    wcag_step : bool
        If *True* (default), run the WCAG AA post-selection replacement check
        (Step 5 of the DSP pipeline).  Set to *False* to skip it entirely;
        ``wcag_guaranteed`` in the result then reflects whether the greedy
        palette happens to contain a qualifying contrast pair, with no
        correction applied.

    Returns
    -------
    SelectionResult
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")

    # ------------------------------------------------------------------
    # Step 2: convert candidates to Lab
    # ------------------------------------------------------------------
    candidates_lab = srgb_to_lab(candidates_rgb)  # (K, 3)
    K = len(candidates_rgb)

    # ------------------------------------------------------------------
    # Step 3: initialise with highest-frequency candidate
    # ------------------------------------------------------------------
    palette_indices: list[int] = [int(np.argmax(freqs))]

    # ------------------------------------------------------------------
    # Step 4: greedy expansion
    # ------------------------------------------------------------------
    # We keep track of which candidates are still eligible (not yet in palette)
    in_palette = set(palette_indices)

    while len(palette_indices) < n:
        best_score = -math.inf
        best_idx: Optional[int] = None

        palette_lab_list = [candidates_lab[i] for i in palette_indices]

        for idx in range(K):
            if idx in in_palette:
                continue

            min_de = _min_de_to_palette(candidates_lab[idx], palette_lab_list)

            # Hard constraint: must be clearly distinct from all palette members
            if min_de < tau_dist:
                continue

            score = alpha * math.log(freqs[idx] + 1e-12) + beta * min_de
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            # No candidate satisfies the hard constraint.  Relax τ progressively
            # and warn — this is a legitimate failure mode to surface in the paper.
            relaxed_tau = tau_dist
            _MAX_RELAX_ITERS = 50  # 0.75^50 ≈ 1.3e-4; prevents spin on monochromatic images
            _relax_iters = 0
            while best_idx is None and relaxed_tau > 1e-6 and _relax_iters < _MAX_RELAX_ITERS:
                relaxed_tau *= 0.75
                _relax_iters += 1
                palette_lab_list = [candidates_lab[i] for i in palette_indices]  # hoisted
                for idx in range(K):
                    if idx in in_palette:
                        continue
                    min_de = _min_de_to_palette(candidates_lab[idx], palette_lab_list)
                    if min_de < relaxed_tau:
                        continue
                    score = alpha * math.log(freqs[idx] + 1e-12) + beta * min_de
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is None:
                # Absolute fallback: take most distant remaining candidate regardless
                palette_lab_list = [candidates_lab[i] for i in palette_indices]
                remaining = [i for i in range(K) if i not in in_palette]
                if not remaining:
                    break  # pool exhausted
                best_idx = max(
                    remaining,
                    key=lambda i: _min_de_to_palette(candidates_lab[i], palette_lab_list),
                )
                warnings.warn(
                    f"DSP: τ_dist={tau_dist} could not be satisfied at palette size "
                    f"{len(palette_indices) + 1}; added most-distant remaining candidate "
                    f"(ΔE={_min_de_to_palette(candidates_lab[best_idx], palette_lab_list):.1f}). "
                    "This may indicate a low-variety image.",
                    stacklevel=2,
                )

        palette_indices.append(best_idx)
        in_palette.add(best_idx)

    # ------------------------------------------------------------------
    # Step 5: WCAG AA post-selection check (skipped when wcag_step=False)
    # ------------------------------------------------------------------
    wcag_replacement_applied = False
    wcag_distinctness_compromised = False

    def _palette_has_aa_pair(idx_list: list[int]) -> bool:
        for a, b in combinations(idx_list, 2):
            if wcag_contrast(candidates_rgb[a], candidates_rgb[b]) >= WCAG_AA_NORMAL:
                return True
        return False

    wcag_guaranteed = _palette_has_aa_pair(palette_indices)

    if wcag_step and not wcag_guaranteed:
        # Find the palette member with the lowest minimum ΔE to its neighbours
        # (the "least distinct" member) and try to replace it.
        palette_lab_list = [candidates_lab[i] for i in palette_indices]

        def _intra_min_de(pos: int) -> float:
            others = [candidates_lab[palette_indices[k]] for k in range(len(palette_indices)) if k != pos]
            if not others:
                return 0.0
            return min(delta_e2000(palette_lab_list[pos], o) for o in others)

        least_distinct_pos = min(range(len(palette_indices)), key=_intra_min_de)
        victim_idx = palette_indices[least_distinct_pos]

        # Remaining palette members after removing the victim
        remaining_indices = [idx for k, idx in enumerate(palette_indices) if k != least_distinct_pos]
        remaining_lab = [candidates_lab[idx] for idx in remaining_indices]

        # Find best replacement: prefer candidates satisfying BOTH contrast AND τ_dist.
        # Fall back to contrast-only if no joint-satisfying candidate exists.
        candidate_pool = [i for i in range(K) if i not in in_palette]
        best_replacement: Optional[int] = None
        best_contrast = 0.0
        best_replacement_fallback: Optional[int] = None
        best_contrast_fallback = 0.0

        for cand in candidate_pool:
            cand_lab = candidates_lab[cand]
            # Check contrast against all remaining palette members
            max_cr_with_remaining = max(
                (wcag_contrast(candidates_rgb[cand], candidates_rgb[pal_idx])
                 for pal_idx in remaining_indices),
                default=0.0,
            )
            if max_cr_with_remaining < WCAG_AA_NORMAL:
                continue  # does not create a qualifying contrast pair

            # Check τ_dist against remaining members (joint constraint)
            if remaining_lab:
                min_de_remaining = min(delta_e2000(cand_lab, rlab) for rlab in remaining_lab)
            else:
                min_de_remaining = float('inf')

            if min_de_remaining >= tau_dist:
                # Joint constraint satisfied — preferred candidate
                if max_cr_with_remaining > best_contrast:
                    best_contrast = max_cr_with_remaining
                    best_replacement = cand
            else:
                # Contrast-only fallback
                if max_cr_with_remaining > best_contrast_fallback:
                    best_contrast_fallback = max_cr_with_remaining
                    best_replacement_fallback = cand

        if best_replacement is not None:
            # Joint constraint satisfied — clean replacement
            palette_indices[least_distinct_pos] = best_replacement
            in_palette.discard(victim_idx)
            in_palette.add(best_replacement)
            wcag_replacement_applied = True
            wcag_guaranteed = True
        elif best_replacement_fallback is not None:
            # Contrast-only fallback: distinctness compromised
            palette_indices[least_distinct_pos] = best_replacement_fallback
            in_palette.discard(victim_idx)
            in_palette.add(best_replacement_fallback)
            wcag_replacement_applied = True
            wcag_guaranteed = True
            wcag_distinctness_compromised = True
            warnings.warn(
                "DSP: WCAG AA replacement applied but no candidate satisfies both "
                f"contrast (≥{WCAG_AA_NORMAL}) and τ_dist ({tau_dist} ΔE2000) "
                "simultaneously.  wcag_distinctness_compromised=True.",
                stacklevel=2,
            )
        else:
            # Honest reporting: the constraint cannot be satisfied from this pool
            warnings.warn(
                "DSP: WCAG AA contrast guarantee could not be satisfied from the "
                "candidate pool.  This is expected for monochromatic or low-contrast "
                "images.  The wcag_guaranteed flag is set to False.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    final_rgb = candidates_rgb[palette_indices]          # (n, 3)
    final_lab = candidates_lab[palette_indices]          # (n, 3)
    final_freqs = [float(freqs[i]) for i in palette_indices]

    return SelectionResult(
        palette_rgb=final_rgb,
        palette_lab=final_lab,
        frequencies=final_freqs,
        n=len(palette_indices),
        alpha=alpha,
        beta=beta,
        tau_dist=tau_dist,
        wcag_guaranteed=wcag_guaranteed,
        wcag_replacement_applied=wcag_replacement_applied,
        wcag_distinctness_compromised=wcag_distinctness_compromised,
        candidate_pool_size=K,
    )


def select_palette(
    image: Image.Image,
    n: int = 5,
    alpha: float = 1.0,
    beta: float = 1.0,
    tau_dist: float = 10.0,
    max_candidates: int = 256,
    wcag_step: bool = True,
) -> SelectionResult:
    """Run the constrained greedy DSP selection on *image*.

    Parameters
    ----------
    image:
        A ``PIL.Image.Image`` object (any mode; converted to RGB internally).
    n:
        Target palette size.
    alpha:
        Weight for log-frequency in the selection score.
    beta:
        Weight for min-ΔE2000 in the selection score.

        **Invariance note:** the selection is largely invariant to β/α ∈ [0.1, 10]
        because the τ_dist hard constraint already clamps minimum distinctness
        regardless of the weighting. The score function primarily orders candidates
        among those that already satisfy the constraint; shifting β/α redistributes
        emphasis between frequency and distance within that feasible set, but has
        negligible effect on which colours are ultimately selected
        (empirically: spread < 1 ΔE₂₀₀₀ across the full β/α range on N=30 images).
        The default α=β=1.0 is therefore representative of the family.
    tau_dist:
        Minimum ΔE2000 a candidate must have from all current palette members
        to be eligible for selection (default 10 — "clearly distinct").
    max_candidates:
        Number of colours in the median-cut quantization pool.
    wcag_step : bool
        If *True* (default), run the WCAG AA post-selection replacement check
        (Step 5 of the DSP pipeline).  Set to *False* to skip it entirely
        (useful for ablation studies).

    Notes
    -----
    **mode=auto threshold (L* < 40):** When ``assign_roles`` is subsequently
    called with ``mode='auto'``, it assigns dark-mode to images whose pixel-wise
    mean L* is below 40.  This threshold was selected qualitatively from a small
    development set; images with mean L* < 40 are visually dark enough that a
    dark-themed design system is the natural choice.  Learning this threshold
    from labelled design-system corpora is future work.

    Returns
    -------
    SelectionResult
    """
    candidates_rgb, freqs = _quantize_to_candidates(image, max_candidates)
    return select_from_candidates(
        candidates_rgb, freqs, n=n, alpha=alpha, beta=beta,
        tau_dist=tau_dist, wcag_step=wcag_step,
    )
