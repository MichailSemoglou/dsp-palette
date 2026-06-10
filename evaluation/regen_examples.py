"""Regenerate fig3_examples.pdf from per-image result JSONs.

Row order: colorful, dark, high-variance, low-chroma.
Layout changes vs. the original report.py figure2_examples:
  - Row order reordered as above.
  - hspace increased from 0.30 to 0.45 → adds ~0.25 cm visible gap per row.
  - min-ΔE annotation moved from y=-0.10 to y=-0.15 so it clears the palette strip.
  - figsize height unchanged (4.80 in); the rendered height grows ~0.75 cm from
    the combined hspace increase and annotation shift.
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from dsp.metrics import relative_luminance

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/raw")
CORPUS_ROOT = Path("corpus")
OUTPUT_PATH = Path("figures/fig3_examples.pdf")

METHOD_ORDER = ["dsp", "kmeans_lab", "kmeans_rgb", "median_cut"]
METHOD_LABELS = {
    "dsp":        "DSP",
    "kmeans_lab": "k-Means Lab",
    "kmeans_rgb": "k-Means RGB",
    "median_cut": "Median Cut",
}
ROLE_ABBREV = {
    "surface":    "Srf",
    "on-surface": "OnS",
    "primary":    "Pri",
    "secondary":  "Sec",
    "accent":     "Acc",
    "extra":      "Ext",
}
ROLE_DISPLAY_ORDER = ["surface", "on-surface", "primary", "secondary", "accent", "extra"]

# Row order: colorful, dark, high-variance, low-chroma
FIG_IMAGES = [
    ("coco_000000288762", "Colorful image"),
    ("coco_000000438304", "Dark image (mode=auto)"),
    ("coco_000000021465", "High-variance image"),
    ("coco_000000117645", "Low-chroma image"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb01(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def _text_color_for_bg(hex_bg: str) -> str:
    r, g, b = _hex_to_rgb01(hex_bg)
    return "#000000" if relative_luminance([r, g, b]) > 0.179 else "#ffffff"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

n_rows = len(FIG_IMAGES)
n_cols = 1 + len(METHOD_ORDER)          # thumbnail + 4 method columns

# row_height increased from 1.20 to 1.285 so figsize_h grows from 4.80 to 5.14 in
# (+0.34 in ≈ +0.75 cm rendered). hspace=0.425 (was 0.30) keeps individual axes
# the same absolute height while routing the extra space entirely to inter-row gaps,
# yielding ~0.25 cm of visible whitespace between each pair of rows.
row_height = 1.285
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(2.4 * n_cols, row_height * n_rows),
    squeeze=False,
)
fig.subplots_adjust(wspace=0.05, hspace=0.425)

for row, (image_id, crit_label) in enumerate(FIG_IMAGES):
    result_path = RESULTS_DIR / f"{image_id}.json"
    if not result_path.exists():
        print(f"WARNING: missing result {result_path}")
        continue

    with open(result_path) as f:
        data = json.load(f)

    # ── thumbnail ────────────────────────────────────────────────────────
    ax_thumb = axes[row][0]
    filename = data.get("filename", image_id.replace("coco_000000", "") + ".jpg")
    img_path = CORPUS_ROOT / "photographs" / filename
    if img_path.exists():
        thumb = Image.open(img_path).convert("RGB")
        w, h = thumb.size
        target_ar = 1.85
        if w / h > target_ar:
            new_w = int(round(h * target_ar))
            left = (w - new_w) // 2
            thumb = thumb.crop((left, 0, left + new_w, h))
        else:
            new_h = int(round(w / target_ar))
            top_px = (h - new_h) // 2
            thumb = thumb.crop((0, top_px, w, top_px + new_h))
        thumb = thumb.resize((296, 160), Image.LANCZOS)
        ax_thumb.imshow(np.array(thumb), aspect="auto")
    ax_thumb.axis("off")
    mean_L = data.get("image_mean_L", 0.0)
    ax_thumb.set_title(
        f"{crit_label}\n" + rf"mean $L^*$={mean_L:.0f}",
        fontsize=6.5, pad=2,
    )

    # ── palette strips ────────────────────────────────────────────────────
    for col_offset, method in enumerate(METHOD_ORDER):
        ax = axes[row][1 + col_offset]
        mdata = data.get("methods", {}).get(method)
        if mdata is None or "error" in mdata:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
            ax.axis("off")
            continue

        palette_hex = mdata.get("palette_hex", [])
        roles_map   = {int(k): v for k, v in mdata.get("roles", {}).items()}
        n_sw = len(palette_hex)

        paired = [
            (palette_hex[i], roles_map.get(i, "extra"))
            for i in range(n_sw)
        ]
        paired.sort(
            key=lambda x: (
                ROLE_DISPLAY_ORDER.index(x[1])
                if x[1] in ROLE_DISPLAY_ORDER
                else len(ROLE_DISPLAY_ORDER)
            )
        )

        for sw_idx, (hex_c, role) in enumerate(paired):
            x    = sw_idx / n_sw
            w_sw = 1.0 / n_sw
            ax.add_patch(mpatches.Rectangle(
                (x, 0), w_sw, 1,
                color=hex_c,
                transform=ax.transAxes,
                clip_on=False,
            ))
            label   = ROLE_ABBREV.get(role, "?")
            txt_col = _text_color_for_bg(hex_c)
            ax.text(
                x + w_sw / 2, 0.5, label,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=5.5, color=txt_col,
                fontweight="bold",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        if row == 0:
            ax.set_title(METHOD_LABELS.get(method, method), fontsize=7.5, pad=3)

        # Annotation moved to y=-0.15 (was -0.10) so it clears the palette strip
        min_de = mdata.get("metrics", {}).get("min_pairwise_de2000", float("nan"))
        ax.text(
            0.5, -0.15, rf"$\Delta E_{{\min}}={min_de:.1f}$",
            transform=ax.transAxes,
            ha="center", fontsize=6, color="#444444",
        )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
