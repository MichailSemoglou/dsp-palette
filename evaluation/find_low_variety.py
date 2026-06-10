#!/usr/bin/env python3
"""
find_low_variety.py
===================
Identify low-variety images from a COCO image folder that stress-test the DSP
hard distinctness constraint (min DeltaE2000 >= tau_dist).

The ranking signal is the DSP selection pipeline itself, not a colour proxy:
for each image we quantise (median-cut, K candidates), convert to CIELAB, and
run the constrained greedy selection. An image qualifies as a stress test when
the constraint cannot be satisfied at the full tau_dist and the fallback must
relax it -- i.e. five colours >= tau_dist apart do not exist. This is exactly
the regime that turns the "worst-case guarantee" into an empirically
characterised mechanism, and it hands you the activation statistics
(fallback rate, relaxation depth, achieved min DeltaE) as a by-product.

IMPORTANT (data hygiene): the low-variety corpus MUST be disjoint from the
90 images already drawn for the dev (15) + test (75) split, or you reintroduce
the very leakage concern raised in review. Pass the list of already-used
filenames via --exclude-list so they are skipped.

Dependencies: numpy, Pillow, scikit-image  (optional: tqdm)
    pip install numpy pillow scikit-image tqdm
"""

from __future__ import annotations
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000

# ---- DSP parameters (match the paper's defaults) --------------------------
K_QUANT = 256       # candidate pool size (median-cut)
N_COLORS = 5        # palette size
TAU_DIST = 10.0     # hard distinctness threshold
ALPHA = 1.0         # log-frequency weight
BETA = 1.0          # distinctness weight
RELAX = 0.75        # fallback: tau_eff *= RELAX per step
EPS = 1e-9
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---- Colour helpers --------------------------------------------------------
def quantize_to_lab(path: Path, k: int = K_QUANT):
    """Median-cut to <=k colours; return (lab[K,3], freq[K]) or None on failure."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    q = img.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    pal = np.asarray(q.getpalette()[: 3 * k], dtype=np.float64).reshape(-1, 3)
    counts = np.bincount(np.asarray(q, dtype=np.int64).ravel(), minlength=len(pal))
    used = counts > 0
    pal, counts = pal[used], counts[used]
    if len(pal) < N_COLORS:               # fewer distinct colours than a palette
        return pal, counts.astype(np.float64) / counts.sum()
    freq = counts.astype(np.float64) / counts.sum()
    lab = rgb2lab((pal / 255.0)[np.newaxis, :, :])[0]   # sRGB->Lab, D65
    return lab, freq


def dE_to_set(lab_all: np.ndarray, lab_set: np.ndarray) -> np.ndarray:
    """Min DeltaE2000 from every candidate (K,3) to a small set (m,3)."""
    dmin = np.full(len(lab_all), np.inf)
    for c in lab_set:
        d = deltaE_ciede2000(lab_all, np.broadcast_to(c, lab_all.shape))
        dmin = np.minimum(dmin, d)
    return dmin


def palette_min_dE(lab_sel: np.ndarray) -> float:
    """Smallest pairwise DeltaE2000 within the selected palette."""
    m = len(lab_sel)
    if m < 2:
        return 0.0
    best = np.inf
    for i in range(m):
        for j in range(i + 1, m):
            best = min(best, float(deltaE_ciede2000(lab_sel[i], lab_sel[j])))
    return best


# ---- Faithful constrained greedy selection (Eq. 1 & 2 + fallback) ----------
def dsp_select(lab, freq, n=N_COLORS, tau=TAU_DIST, alpha=ALPHA, beta=BETA, relax=RELAX):
    """Return dict of activation diagnostics for one image."""
    K = len(lab)
    chosen = [int(np.argmax(freq))]
    logf = np.log(freq + EPS)
    relax_steps = 0           # total fallback relaxations across all steps
    reject_count = 0          # candidate rejections by the constraint (any step)
    fallback = False          # did the constraint ever fail outright?

    if K < n:                 # degenerate: cannot even form a palette
        return dict(fallback=True, relax_steps=99, reject_count=0,
                    n_avail=K, min_dE=0.0)

    for _ in range(n - 1):
        dmin = dE_to_set(lab, lab[chosen])
        dmin[chosen] = -np.inf
        tau_eff = tau
        eligible = dmin >= tau_eff
        # count how many candidates the (full-tau) constraint rejects this step
        reject_count += int(np.sum((dmin < tau) & (dmin >= 0)))
        while not eligible.any():
            tau_eff *= relax
            relax_steps += 1
            fallback = True
            eligible = dmin >= tau_eff
            if tau_eff < 1e-6:
                break
        score = alpha * logf + beta * dmin
        score[~eligible] = -np.inf
        chosen.append(int(np.argmax(score)))

    return dict(fallback=fallback, relax_steps=relax_steps,
                reject_count=reject_count, n_avail=K,
                min_dE=palette_min_dE(lab[chosen]))


# ---- Driver ----------------------------------------------------------------
def load_exclude(path: str | None) -> set[str]:
    if not path:
        return set()
    names = set()
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s:
            names.add(Path(s).name)   # match on basename, ignore any directory
    return names


def main():
    ap = argparse.ArgumentParser(description="Find low-variety COCO images that stress the DSP constraint.")
    ap.add_argument("image_dir", help="Folder of COCO images (e.g. val2017/)")
    ap.add_argument("--out-dir", default="low_variety_corpus", help="Where to copy the selected images")
    ap.add_argument("--csv", default="low_variety_ranking.csv", help="Full ranking output")
    ap.add_argument("--exclude-list", default=None,
                    help="Text file of filenames already used in dev+test (one per line) to skip")
    ap.add_argument("--n-select", type=int, default=30, help="How many images to select")
    ap.add_argument("--max-scan", type=int, default=0, help="Cap images scanned (0 = all); useful for a quick trial")
    ap.add_argument("--seed", type=int, default=2026, help="Seed for deterministic tie-breaking of the scan order")
    args = ap.parse_args()

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):  # no-op fallback
            return x

    exclude = load_exclude(args.exclude_list)
    paths = sorted(p for p in Path(args.image_dir).iterdir()
                   if p.suffix.lower() in IMG_EXTS and p.name not in exclude)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)
    if args.max_scan:
        paths = paths[: args.max_scan]
    print(f"Scanning {len(paths)} images "
          f"({len(exclude)} excluded as already-used)...")

    rows = []
    for p in tqdm(paths):
        res = quantize_to_lab(p)
        if res is None:
            continue
        lab, freq = res
        diag = dsp_select(lab, freq)
        rows.append({"filename": p.name, **diag})

    if not rows:
        print("No images processed -- check the image directory path.")
        return

    # Rank: constraint that binds outright first; then deepest relaxation;
    # then lowest achieved palette min DeltaE; then fewest available candidates.
    rows.sort(key=lambda r: (not r["fallback"], -r["relax_steps"],
                             r["min_dE"], r["n_avail"]))

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "filename", "fallback",
                                          "relax_steps", "reject_count",
                                          "n_avail", "min_dE"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"rank": i, **r,
                        "min_dE": round(r["min_dE"], 3)})

    n_fb = sum(r["fallback"] for r in rows)
    print(f"\nScanned {len(rows)} images.")
    print(f"Constraint fallback triggered on {n_fb} "
          f"({100*n_fb/len(rows):.1f}%) -- these are genuine stress tests.")
    if n_fb < args.n_select:
        print(f"WARNING: only {n_fb} images force the fallback; the remaining "
              f"{args.n_select - n_fb} selected images are merely the tightest "
              f"available (constraint satisfied but barely). Consider scanning "
              f"more images or supplementing with constructed flat-colour inputs.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = rows[: args.n_select]
    for r in selected:
        shutil.copy2(Path(args.image_dir) / r["filename"], out / r["filename"])

    print(f"\nTop {len(selected)} written to '{out}/'; full ranking in '{args.csv}'.")
    print("\nrank  filename                  fallback  relax  min_dE")
    for i, r in enumerate(selected, 1):
        print(f"{i:>3}   {r['filename']:<24} {str(r['fallback']):>7}  "
              f"{r['relax_steps']:>4}   {r['min_dE']:>6.2f}")


if __name__ == "__main__":
    main()
