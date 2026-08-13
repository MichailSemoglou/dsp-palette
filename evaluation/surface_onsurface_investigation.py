"""
Investigate the Surface/On-Surface AA anomaly.

DSP's role-assigned Surface/On-Surface pair failed AA on 1 of 75 images,
while k-Means Lab and k-Means RGB passed on all 75.  This script finds that
image, explains the failure mode, and confirms whether it is a role-assignment
gap or a WCAG-step failure.

Key facts from the role heuristic (roles.py):
  Surface     = palette member with highest L* (light mode) / lowest L* (dark mode)
  On-Surface  = palette member with HIGHEST WCAG CONTRAST against Surface
                (selected from the remaining n-1 members)

Corollary: if Surface/On-Surface fails AA (< 4.5:1), then NO member in
the palette (other than Surface itself) achieves ≥ 4.5:1 against Surface.
A separate pair between two non-Surface members could still achieve AA —
that would be a role-assignment gap, not a WCAG-step failure.

The DSP WCAG post-selection step guarantees that SOME pair in the palette
achieves ≥ 4.5:1, but does not specifically guarantee the Surface role pair.

Output
------
results/ablation/surface_onsurface.csv  — per-image per-method detail
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dsp.metrics import srgb_to_lab, wcag_contrast
from dsp.roles import assign_roles


WCAG_AA_THRESHOLD = 4.5
METHODS = ["dsp", "kmeans_lab", "kmeans_rgb", "median_cut"]


def _load_test_ids(manifest_path: Path) -> set[str]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {e["id"] for e in manifest
            if e.get("subset") == "photographs" and not e.get("dev")}


def _analyse_method(method_data: dict) -> dict:
    """Detailed role-pair and all-pairs AA analysis for one method's result."""
    out: dict = {
        "surface_idx": None,
        "on_surface_idx": None,
        "surface_hex": None,
        "on_surface_hex": None,
        "surface_lab": None,
        "on_surface_lab": None,
        "role_pair_contrast": None,
        "role_pair_aa": None,
        "any_pair_aa": None,
        "best_any_pair_contrast": None,
        "best_any_pair_hex_a": None,
        "best_any_pair_hex_b": None,
        "wcag_replacement_applied": None,
        "wcag_guaranteed": None,
        "effective_mode": None,
    }

    if not method_data or "error" in method_data:
        return out

    palette_rgb_raw = method_data.get("palette_rgb")
    freqs_raw       = method_data.get("frequencies")
    if palette_rgb_raw is None or freqs_raw is None:
        return out

    palette_rgb  = np.array(palette_rgb_raw, dtype=np.uint8)
    palette_lab  = srgb_to_lab(palette_rgb)
    frequencies  = list(freqs_raw)
    n            = len(palette_rgb)

    roles = assign_roles(palette_rgb, palette_lab, frequencies, mode="auto")

    def to_hex(rgb: np.ndarray) -> str:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        return f"#{r:02x}{g:02x}{b:02x}"

    if roles.surface is not None:
        out["surface_idx"] = roles.surface
        out["surface_hex"] = to_hex(palette_rgb[roles.surface])
        out["surface_lab"] = [round(float(x), 3) for x in palette_lab[roles.surface]]

        # Effective mode: infer from which extreme L* was chosen
        surface_L = float(palette_lab[roles.surface, 0])
        all_L = [float(palette_lab[i, 0]) for i in range(n)]
        out["effective_mode"] = "dark" if surface_L == min(all_L) else "light"

    if roles.on_surface is not None:
        out["on_surface_idx"] = roles.on_surface
        out["on_surface_hex"] = to_hex(palette_rgb[roles.on_surface])
        out["on_surface_lab"] = [round(float(x), 3) for x in palette_lab[roles.on_surface]]

    if roles.surface is not None and roles.on_surface is not None:
        cr = wcag_contrast(
            palette_rgb[roles.on_surface],
            palette_rgb[roles.surface],
        )
        out["role_pair_contrast"] = round(cr, 4)
        out["role_pair_aa"] = cr >= WCAG_AA_THRESHOLD

    # Best contrast across ALL palette pairs (not just role pair)
    best_cr   = 1.0
    best_i    = None
    best_j    = None
    for i, j in combinations(range(n), 2):
        cr = wcag_contrast(
            palette_rgb[i],
            palette_rgb[j],
        )
        if cr > best_cr:
            best_cr = cr
            best_i  = i
            best_j  = j

    out["best_any_pair_contrast"] = round(best_cr, 4)
    out["any_pair_aa"] = best_cr >= WCAG_AA_THRESHOLD
    if best_i is not None:
        out["best_any_pair_hex_a"] = to_hex(palette_rgb[best_i])
        out["best_any_pair_hex_b"] = to_hex(palette_rgb[best_j])

    out["wcag_replacement_applied"] = bool(method_data.get("wcag_replacement_applied", False))
    out["wcag_guaranteed"]           = bool(method_data.get("wcag_guaranteed", True))

    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Investigate Surface/On-Surface WCAG AA anomaly.",
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--manifest",    required=True)
    args = parser.parse_args(argv)

    results_dir   = Path(args.results_dir)
    manifest_path = Path(args.manifest)

    test_ids = _load_test_ids(manifest_path)
    rows = []

    for jpath in sorted(results_dir.glob("coco_[0-9]*.json")):
        with open(jpath) as f:
            d = json.load(f)
        image_id = d.get("image_id")
        if image_id not in test_ids:
            continue

        methods_data = d.get("methods", {})
        for method in METHODS:
            mdata = methods_data.get(method, {})
            details = _analyse_method(mdata)
            row = {"image_id": image_id, "method": method}
            row.update(details)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save full per-image per-method table
    out_dir  = results_dir.parent / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "surface_onsurface.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ── Investigate DSP failures ───────────────────────────────────────────
    dsp_df  = df[df["method"] == "dsp"].copy()
    dsp_fail = dsp_df[dsp_df["role_pair_aa"] == False]

    sep = "=" * 72
    print(f"\n{sep}")
    print("SURFACE / ON-SURFACE WCAG AA INVESTIGATION")
    print(sep)

    print(f"\nDSP role-pair AA failures: {len(dsp_fail)} of {len(dsp_df)}")

    for _, row in dsp_fail.iterrows():
        print(f"\n{'─'*60}")
        print(f"Image ID              : {row['image_id']}")
        print(f"Effective mode        : {row['effective_mode']}")
        print(f"Surface colour        : {row['surface_hex']}  "
              f"(idx={row['surface_idx']}, L*={row['surface_lab'][0] if row['surface_lab'] else 'n/a'})")
        print(f"On-Surface colour     : {row['on_surface_hex']}  "
              f"(idx={row['on_surface_idx']}, L*={row['on_surface_lab'][0] if row['on_surface_lab'] else 'n/a'})")
        print(f"Role-pair contrast    : {row['role_pair_contrast']:.4f}:1  "
              f"{'PASS' if row['role_pair_aa'] else 'FAIL'} (threshold 4.5:1)")
        print(f"wcag_replacement_applied: {row['wcag_replacement_applied']}")
        print(f"wcag_guaranteed          : {row['wcag_guaranteed']}")
        print()
        print(f"Best contrast in ANY pair: {row['best_any_pair_contrast']:.4f}:1  "
              f"({'PASS' if row['any_pair_aa'] else 'FAIL'})")
        if row["any_pair_aa"]:
            print(f"  Best pair             : {row['best_any_pair_hex_a']}  vs  "
                  f"{row['best_any_pair_hex_b']}")
            print()
            print("  INTERPRETATION: The WCAG guarantee is satisfied by a")
            print("  non-role pair (two palette members that are not Surface or")
            print("  On-Surface in the heuristic sense). The role-assignment gap")
            print("  arises because On-Surface = max-contrast-against-Surface,")
            print("  not max-contrast globally. Surface itself has no member")
            print("  above 4.5:1, yet another pair in the palette does.")
        else:
            print()
            print("  INTERPRETATION: No pair in this palette achieves 4.5:1.")
            print("  This is a WCAG-step failure, not only a role-assignment gap.")

    # ── Confirm the role heuristic ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("ROLE HEURISTIC CONFIRMATION (from roles.py source)")
    print("  Surface     = palette member with extreme L*")
    print("                (highest in light mode; lowest in dark mode)")
    print("  On-Surface  = palette member with HIGHEST WCAG CONTRAST")
    print("                against Surface (from remaining n-1 members)")
    print()
    print("  Corollary: if Surface/On-Surface pair fails AA, then no member")
    print("  of the palette (other than Surface itself) achieves >= 4.5:1")
    print("  against Surface. A different non-Surface pair could still pass.")

    # ── Baseline context ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("BASELINE CONTEXT (surface/on-surface AA)")
    for method in METHODS:
        sub  = df[df["method"] == method]
        n_pass = int((sub["role_pair_aa"] == True).sum())
        n_fail = int((sub["role_pair_aa"] == False).sum())
        n_total = n_pass + n_fail
        print(f"\n  {method:16s}: {n_pass}/{n_total} pass")
        if n_fail > 0:
            fail_imgs = sub[sub["role_pair_aa"] == False]["image_id"].tolist()
            print(f"  Failing images: {fail_imgs}")

    # ── Clarify whether baselines have any WCAG guarantee ─────────────────
    print(f"\n{'─'*60}")
    print("BASELINE WCAG GUARANTEES")
    print("  k-Means Lab, k-Means RGB, and Median Cut have NO built-in WCAG")
    print("  post-selection step.  Their Surface/On-Surface pairs pass only")
    print("  because their palettes happen to contain a high-contrast pair")
    print("  that the same role heuristic selects.  This is incidental, not")
    print("  architectural.")
    print(f"\n{sep}\n")

    # ── Median Cut failure detail ──────────────────────────────────────────
    mc_fail = df[(df["method"] == "median_cut") & (df["role_pair_aa"] == False)]
    if not mc_fail.empty:
        print(f"Median Cut role-pair failures ({len(mc_fail)}):")
        for _, row in mc_fail.iterrows():
            print(f"  {row['image_id']}  contrast={row['role_pair_contrast']:.4f}  "
                  f"mode={row['effective_mode']}  any_pair_aa={row['any_pair_aa']}")

    # ── Root cause: relative_luminance boundary-value artefact ─────────────
    print(f"\n{'─'*60}")
    print("ROOT CAUSE: relative_luminance([1, 0, 0]) boundary-value artefact")
    print()
    print("  The image contains the colour #010000 = uint8 RGB (1, 0, 0).")
    print("  relative_luminance() auto-detects the input scale with the guard:")
    print("      if arr.max() > 1.0: arr = arr / 255.0")
    print("  For [1, 0, 0], max == 1.0 exactly, so the /255 path is NOT taken.")
    print("  The function interprets it as already-normalised [0,1] pure red")
    print("  and returns L = 0.2126 (pure-red luminance).")
    print()
    print("  srgb_to_lab() has the identical guard.  When called on the FULL")
    print("  palette array (max = 254), /255 IS applied and L* = 0.058 (correct).")
    print("  The role-assignment step uses the full-array path, so Surface is")
    print("  correctly identified as near-black.  But wcag_contrast() calls")
    print("  relative_luminance() on individual palette rows, where [1,0,0]")
    print("  has max == 1.0 and is again misidentified as pure red.")
    print()
    from dsp.metrics import relative_luminance
    import numpy as np
    L_wrong = relative_luminance(np.array([1, 0, 0], dtype=np.float64))
    L_right = 0.2126 * (1/255 / 12.92)   # correct linearisation of R=1/255
    L_near_white = relative_luminance(np.array([254, 253, 241], dtype=np.float64))
    cr_computed = (L_near_white + 0.05) / (L_wrong + 0.05)
    cr_true     = (L_near_white + 0.05) / (L_right + 0.05)
    print(f"  Computed L(#010000) : {L_wrong:.6f}  (treated as pure red)")
    print(f"  True     L(#010000) : {L_right:.8f}  (near-black, R=1/255)")
    print(f"  Computed contrast #fefdf1 vs #010000 : {cr_computed:.2f}:1  → FAIL")
    print(f"  True     contrast #fefdf1 vs #010000 : {cr_true:.2f}:1  → PASS")
    print()
    print("  Consequence: the DSP AA failure on coco_000000464522 is a")
    print("  MEASUREMENT ARTEFACT of the boundary case in relative_luminance,")
    print("  not a genuine design-system accessibility failure.  The actual")
    print("  Surface/On-Surface colours have a true contrast of ~20.5:1.")
    print()
    print("  The WCAG post-selection guarantee (wcag_guaranteed=True,")
    print("  wcag_replacement_applied=False) is also computed using the same")
    print("  buggy per-colour path.  The selector correctly found the pair")
    print("  #fefdf1 + #696836 (5.65:1, both with max>1 in their rows) and")
    print("  deemed the palette WCAG-compliant — but the Surface pair was")
    print("  never tested at its true luminance.")


if __name__ == "__main__":
    main()
