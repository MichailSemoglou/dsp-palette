"""
Generate mode_comparison_diagnostic.pdf and role-agreement count.
Run with: python3 -m evaluation.mode_diagnostic_fig
"""
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colour

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsp.selector import select_palette
from dsp.roles import assign_roles

RESULTS_DIR       = Path("results/raw")
CORPUS_VAL        = Path("corpus/photographs")
CORPUS_TRAIN      = Path("corpus/photographs_train")
FIGURES_DIR       = Path("figures")


def _image_path(rec: dict) -> Path:
    """Resolve the corpus photograph path for a result record."""
    fname = rec.get("filename", rec["image_id"].replace("coco_", "").replace("train_", "") + ".jpg")
    train = rec["image_id"].startswith("coco_train_")
    root  = CORPUS_TRAIN if train else CORPUS_VAL
    return root / fname

ROLE_ORDER = ["surface", "on-surface", "primary", "secondary", "accent", "extra"]

# Abbreviated labels displayed beneath each swatch
ROLE_LABEL = {
    "surface":    "Surf",
    "on-surface": "OnSf",
    "primary":    "Prim",
    "secondary":  "Sec",
    "accent":     "Acc",
    "extra":      "Xtr",
}

# Colour-coded mode header bands
MODE_COLOR = {
    "dark":  "#1a2744",   # deep navy
    "light": "#1d5f96",   # professional blue
}

BINS = [
    ( 9.0,  15.0),
    (15.0,  25.0),
    (25.0,  30.0),
    (30.0,  35.0),
    (35.0,  40.0),
    (40.0,  43.0),
    (43.0,  46.0),
    (46.0,  48.0),
    (48.0,  49.5),
    (49.5,  50.0),
]


def srgb_to_lab(rgb_arr):
    srgb = rgb_arr.astype(np.float64) / 255.0
    xyz  = colour.sRGB_to_XYZ(srgb)
    return colour.XYZ_to_Lab(xyz)


def min_de2000(lab_arr):
    n = len(lab_arr)
    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(colour.delta_E(lab_arr[i], lab_arr[j], method="CIE 2000"))
            if d < best:
                best = d
    return best


def draw_palette_strip(ax, pal, mode):
    """Draw a 5-swatch palette strip: header band on top, swatches in middle, role abbreviations below."""
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor("#f6f6f6")

    n_sw       = len(pal["rgb"])
    sw_w       = 5.0 / n_sw
    pad_x      = 0.055
    LABEL_H    = 0.175          # label strip height at the bottom
    y_swatch   = LABEL_H + 0.03 # bottom edge of swatch area
    y_top      = 0.795          # bottom edge of header band

    # ── coloured header band ─────────────────────────────────────────
    hdr = mpatches.Rectangle(
        (0, y_top), 5.0, 1.0 - y_top,
        facecolor=MODE_COLOR[mode], edgecolor="none", zorder=2,
    )
    ax.add_patch(hdr)
    ax.text(
        2.5, (y_top + 1.0) / 2,
        "mode = {}    \u0394E_min = {:.1f}".format(mode, pal["min_de"]),
        ha="center", va="center",
        fontsize=7.0, color="#ffffff", fontweight="bold", zorder=3,
    )

    # ── swatches + labels ────────────────────────────────────────────
    swatch_h = y_top - y_swatch - 0.020
    for si, (rgb, role) in enumerate(zip(pal["rgb"], pal["roles"])):
        x0    = si * sw_w
        hex_c = "#{:02X}{:02X}{:02X}".format(*rgb)
        rect  = mpatches.Rectangle(
            (x0 + pad_x, y_swatch),
            sw_w - 2 * pad_x, swatch_h,
            facecolor=hex_c, edgecolor="#c0c0c0", linewidth=0.5, zorder=2,
        )
        ax.add_patch(rect)
        # abbreviated role label below the swatch
        ax.text(
            x0 + sw_w / 2,
            LABEL_H / 2,
            ROLE_LABEL.get(role, role[:3]),
            ha="center", va="center",
            fontsize=6.5, color="#333333", fontweight="bold", zorder=3,
        )


def roles_identical(ra_dark, ra_light, n_palette):
    """
    Compare role assignments from two RoleAssignment objects.
    They use the SAME underlying palette (same palette_rgb indices),
    so we just compare the index assigned to each role slot.
    """
    for attr in ("surface", "on_surface", "primary", "secondary", "accent"):
        if getattr(ra_dark, attr) != getattr(ra_light, attr):
            return False
    return True


def main():
    # ── Global style ─────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         8.0,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load all dark images ─────────────────────────────────────────
    all_dark = []
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        rec = json.loads(fp.read_text())
        if rec["image_mean_L"] < 50.0:
            all_dark.append({
                "image_id": rec["image_id"],
                "mean_L":   rec["image_mean_L"],
                "filename": rec.get("filename",
                                    rec["image_id"].replace("coco_", "") + ".jpg"),
            })
    all_dark.sort(key=lambda r: r["mean_L"])
    print(f"Total dark images (mean L* < 50): {len(all_dark)}")

    # ── Pick one per bin ─────────────────────────────────────────────
    selected = []
    for lo, hi in BINS:
        cands = [r for r in all_dark if lo <= r["mean_L"] < hi]
        if cands:
            mid = (lo + hi) / 2.0
            cands.sort(key=lambda r: abs(r["mean_L"] - mid))
            selected.append(cands[0])
        else:
            print(f"  WARNING: no image for L* in [{lo}, {hi})")

    print(f"\nSelected {len(selected)} representative images:")
    for r in selected:
        print(f"  {r['image_id']}  L*={r['mean_L']:.2f}")

    # ── Build comparison figure ──────────────────────────────────────
    N = len(selected)
    fig, axes = plt.subplots(
        N, 3,
        figsize=(9.2, N * 1.65),
        gridspec_kw={"width_ratios": [2.2, 2.4, 2.4]},
        facecolor="white",
    )
    if N == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.28, wspace=0.08)

    for row_i, rec in enumerate(selected):
        fpath = _image_path(rec)
        img   = Image.open(fpath).convert("RGB")
        mean_L  = rec["mean_L"]

        # Run DSP once — same palette, different role assignment
        res = select_palette(img, n=5)

        pal_dark  = _assign_mode_palette(res, mean_L, "dark")
        pal_light = _assign_mode_palette(res, mean_L, "light")

        # thumbnail
        ax_t = axes[row_i][0]
        thumb = img.copy()
        thumb.thumbnail((160, 120), Image.LANCZOS)
        ax_t.imshow(np.array(thumb))
        ax_t.set_xticks([]); ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_edgecolor("#cccccc")
        ax_t.set_xlabel(
            "L* = {:.1f}".format(mean_L),
            fontsize=7.5, color="#444444", labelpad=3,
        )

        draw_palette_strip(axes[row_i][1], pal_dark,  "dark")
        draw_palette_strip(axes[row_i][2], pal_light, "light")

    # ── Column headers (row 0 only) ──────────────────────────────────
    axes[0][0].set_title(
        "Image  (mean L*)",
        fontsize=8.5, pad=5, color="#222222", fontweight="bold",
    )
    axes[0][1].set_title(
        "Dark Mode",
        fontsize=8.5, pad=5, color=MODE_COLOR["dark"], fontweight="bold",
    )
    axes[0][2].set_title(
        "Light Mode",
        fontsize=8.5, pad=5, color=MODE_COLOR["light"], fontweight="bold",
    )

    fig.suptitle(
        "DSP: mode = dark vs. mode = light — 10 borderline images sorted by mean L*\n"
        "Role order per strip:  Surface · On-Surface · Primary · Secondary · Accent",
        fontsize=8.5, y=1.010, color="#1a1a1a",
    )

    out = FIGURES_DIR / "mode_comparison_diagnostic.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ── Role-agreement count across all 59 dark images ───────────────
    print("\n── Role assignment agreement (mode=dark vs mode=light) ─────────")
    agree = 0
    disagree = 0
    for rec in all_dark:
        fpath = _image_path(rec)
        try:
            img_a = Image.open(fpath).convert("RGB")
        except Exception as e:
            print(f"  SKIP {rec['image_id']}: {e}")
            continue
        res_a = select_palette(img_a, n=5)
        ra_dark  = assign_roles(res_a.palette_rgb, res_a.palette_lab,
                                list(res_a.frequencies),
                                mode="dark",  image_mean_L=rec["mean_L"])
        ra_light = assign_roles(res_a.palette_rgb, res_a.palette_lab,
                                list(res_a.frequencies),
                                mode="light", image_mean_L=rec["mean_L"])
        if roles_identical(ra_dark, ra_light, len(res_a.palette_rgb)):
            agree += 1
        else:
            disagree += 1

    total = agree + disagree
    print(f"  Identical role assignments: {agree}/{total}  ({100*agree/total:.1f}%)")
    print(f"  Different role assignments: {disagree}/{total}  ({100*disagree/total:.1f}%)")
    print()
    print("Interpretation:")
    print(f"  {agree} images produce the SAME palette roles regardless of mode=dark/light.")
    print(f"  {disagree} images have at least one role index changed by the mode switch.")


def _assign_mode_palette(res, mean_L, mode):
    """Given a SelectionResult and mode, return a dict with sorted palette info."""
    ra = assign_roles(
        res.palette_rgb, res.palette_lab, list(res.frequencies),
        mode=mode, image_mean_L=mean_L,
    )
    idx_to_role = {}
    for role_name, idx in [
        ("surface",    ra.surface),
        ("on-surface", ra.on_surface),
        ("primary",    ra.primary),
        ("secondary",  ra.secondary),
        ("accent",     ra.accent),
    ]:
        if idx is not None:
            idx_to_role[idx] = role_name
    for idx in range(len(res.palette_rgb)):
        if idx not in idx_to_role:
            idx_to_role[idx] = "extra"

    sorted_indices = sorted(
        range(len(res.palette_rgb)),
        key=lambda i: ROLE_ORDER.index(idx_to_role.get(i, "extra")),
    )
    pal_rgb_sorted = [res.palette_rgb[i] for i in sorted_indices]
    roles_sorted   = [idx_to_role[i]     for i in sorted_indices]
    pal_lab_sorted = np.array([res.palette_lab[i] for i in sorted_indices])
    return {
        "rgb":    pal_rgb_sorted,
        "roles":  roles_sorted,
        "min_de": min_de2000(pal_lab_sorted),
        "ra":     ra,
    }


if __name__ == "__main__":
    main()
