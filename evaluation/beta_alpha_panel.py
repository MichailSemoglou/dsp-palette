"""
Task 7: β/α invariance panel.

Sweeps the β/α ratio (with α = 1.0 fixed, varying β) over [0.25, 4] on a
30-image subsample of the 75 test images (seeded random.Random(2026)).

For each ratio value the script:
  • runs select_palette with (alpha=1.0, beta=ratio, tau_dist=10.0, n=5, wcag=True)
  • records mean minimum ΔE2000 and reconstruction error over the 30 images

Prints the measured spread in mean min ΔE2000 and saves a two-panel figure
to figures/beta_alpha_panel.pdf.  The figure is a NEW file; it does
not overwrite fig4_sensitivity.pdf.

Usage
-----
    python -m research.evaluation.beta_alpha_panel \\
        --manifest   corpus/manifest.json \\
        --corpus-root corpus/ \\
        --output-dir  figures/
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsp.metrics import (
    min_pairwise_delta_e,
    reconstruction_error_de2000,
    srgb_to_lab,
)
from dsp.selector import select_palette


ALPHA_FIXED = 1.0
BETA_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
N_SUBSET    = 30
SEED        = 2026
TAU_DIST    = 10.0
N_PALETTE   = 5
MAX_PIXELS  = 5_000


def _load_test_images(manifest_path: Path, corpus_root: Path) -> list[tuple[str, Path]]:
    """Return list of (image_id, path) for test images only."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = [
        e for e in manifest
        if e.get("subset") == "photographs" and not e.get("dev")
    ]
    result = []
    for e in entries:
        img_path = corpus_root / e["subset"] / e["filename"]
        if img_path.exists():
            result.append((e["id"], img_path))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="β/α invariance sweep and panel figure.",
    )
    parser.add_argument("--manifest",    required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--output-dir",  required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    corpus_root   = Path(args.corpus_root)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and subsample test images
    all_entries = _load_test_images(manifest_path, corpus_root)
    print(f"Test-set images found: {len(all_entries)}")

    rng = random.Random(SEED)
    shuffled = all_entries[:]
    rng.shuffle(shuffled)
    subset = shuffled[:N_SUBSET]
    print(f"Subsample: {len(subset)} images (seed={SEED})")

    # Sweep
    rows = []
    for i, (image_id, img_path) in enumerate(subset, 1):
        img_pil = Image.open(img_path).convert("RGB")
        img_arr = np.array(img_pil)

        for beta in BETA_VALUES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = select_palette(
                    img_pil,
                    n=N_PALETTE,
                    alpha=ALPHA_FIXED,
                    beta=beta,
                    tau_dist=TAU_DIST,
                    wcag_step=True,
                )
            min_de  = float(min_pairwise_delta_e(res.palette_lab))
            recon   = float(reconstruction_error_de2000(img_arr, res.palette_rgb,
                                                        max_pixels=MAX_PIXELS))
            rows.append({
                "image_id": image_id,
                "beta":     beta,
                "alpha":    ALPHA_FIXED,
                "ratio":    beta / ALPHA_FIXED,
                "min_pairwise_de2000":       min_de,
                "reconstruction_error_de2000": recon,
            })

        if i % 10 == 0:
            print(f"  [{i:3d}/{len(subset)}] done")

    df = pd.DataFrame(rows)

    # Save raw sweep data
    sweep_csv = output_dir.parent / "results" / "ablation" / "beta_alpha_sweep.csv"
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sweep_csv, index=False)
    print(f"Saved sweep data → {sweep_csv}")

    # Aggregate by ratio
    agg = (
        df.groupby("ratio")
        .agg(
            mean_min_de=("min_pairwise_de2000",          "mean"),
            std_min_de =("min_pairwise_de2000",           "std"),
            mean_recon =("reconstruction_error_de2000",  "mean"),
            std_recon  =("reconstruction_error_de2000",   "std"),
        )
        .reset_index()
    )

    # Measured spread
    spread_min_de  = float(agg["mean_min_de"].max() - agg["mean_min_de"].min())
    spread_recon   = float(agg["mean_recon"].max()  - agg["mean_recon"].min())

    # Print summary
    sep = "=" * 68
    print(f"\n{sep}")
    print("β/α INVARIANCE SWEEP (Task 7)")
    print(f"α = {ALPHA_FIXED} fixed;  β ∈ {BETA_VALUES}")
    print(f"N = {N_SUBSET} images (random.Random({SEED}) subsample of 75 test set)")
    print(sep)
    print(f"{'β/α ratio':>10}  {'mean min ΔE':>14}  {'std':>8}  "
          f"{'mean recon':>12}  {'std':>8}")
    for _, row_a in agg.iterrows():
        print(f"{row_a['ratio']:>10.2f}  {row_a['mean_min_de']:>14.3f}  "
              f"{row_a['std_min_de']:>8.3f}  {row_a['mean_recon']:>12.3f}  "
              f"{row_a['std_recon']:>8.3f}")
    print()
    print(f"Spread in mean min ΔE2000 : {spread_min_de:.3f}  ΔE2000 units")
    print(f"Spread in mean recon error: {spread_recon:.3f}  ΔE2000 units")

    if spread_min_de < 1.0:
        claim = f"< 1.0 (actual: {spread_min_de:.3f})  ← supports paper claim"
    else:
        claim = f"{spread_min_de:.3f}  ← EXCEEDS 1.0; review paper claim"
    print(f"Claim check (spread < 1.0): {claim}")
    print(f"{sep}\n")

    # Figure
    ratios = agg["ratio"].values

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.5, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    color_line = "#1A5276"

    # Top: min ΔE2000
    ax_top.errorbar(
        ratios, agg["mean_min_de"], yerr=agg["std_min_de"],
        color=color_line, marker="o", linewidth=2.0, markersize=6,
        capsize=4, label=r"Mean min $\Delta E_{2000}$",
    )
    ax_top.set_ylabel(r"Mean min $\Delta E_{2000}$", fontsize=10)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.annotate(
        f"spread = {spread_min_de:.2f}",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8, color="#555555",
    )

    # Bottom: reconstruction error
    ax_bot.errorbar(
        ratios, agg["mean_recon"], yerr=agg["std_recon"],
        color="#1E8449", marker="s", linewidth=2.0, markersize=6,
        capsize=4, label=r"Mean recon.\ $\Delta E_{2000}$",
    )
    ax_bot.set_ylabel(r"Mean recon.\ $\Delta E_{2000}$", fontsize=10)
    ax_bot.set_xlabel(r"$\beta / \alpha$ ratio", fontsize=10)
    ax_bot.set_xticks(BETA_VALUES)
    ax_bot.set_xticklabels([str(v) for v in BETA_VALUES])
    ax_bot.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        r"$\beta/\alpha$ sensitivity, $N={:d}$ images  (seed {:d})".format(
            N_SUBSET, SEED
        ),
        fontsize=10,
        y=0.995,
    )

    out_pdf = output_dir / "beta_alpha_panel.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved figure → {out_pdf}")


if __name__ == "__main__":
    main()
