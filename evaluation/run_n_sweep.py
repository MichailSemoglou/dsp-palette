"""Evaluate DSP and baselines at n ∈ {3, 4, 6, 7} on the 75-image test set.

Results are saved to results/n_sweep/raw_n{k}/ so the n=5 results
in results/raw/ are untouched.

After all runs, Wilcoxon signed-rank tests (DSP vs each baseline) and
Cliff's delta are computed for min_pairwise_de2000 and wcag_aa_coverage,
and a summary table is printed.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluation.runner import evaluate_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH  = Path("corpus/manifest.json")
CORPUS_ROOT    = Path("corpus")
SWEEP_ROOT     = Path("results/n_sweep")
N_VALUES       = [3, 4, 6, 7]
METRICS        = ["min_pairwise_de2000", "wcag_aa_coverage"]
BASELINES      = ["kmeans_lab", "kmeans_rgb", "median_cut"]

# ---------------------------------------------------------------------------
# Load test-set image list (photographs, dev=False)
# ---------------------------------------------------------------------------
manifest = json.loads(MANIFEST_PATH.read_text())
test_entries = [
    e for e in manifest
    if e["subset"] == "photographs" and not e.get("dev", False)
]
print(f"Test images: {len(test_entries)}")

# ---------------------------------------------------------------------------
# Run evaluation at each n
# ---------------------------------------------------------------------------
for n in N_VALUES:
    out_dir = SWEEP_ROOT / f"raw_n{n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in test_entries:
        image_id = entry["id"]
        out_path = out_dir / f"{image_id}.json"
        if out_path.exists():
            logger.info("Skip (done): %s  n=%d", image_id, n)
            continue

        img_path = CORPUS_ROOT / entry["subset"] / entry["filename"]
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        with Image.open(img_path) as raw:
            img = raw.convert("RGB").copy()

        logger.info("n=%d  %s", n, image_id)
        result = evaluate_image(img, image_id, n=n)
        result["subset"] = entry["subset"]
        result["filename"] = entry["filename"]

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"n={n}: all done → {out_dir}")

# ---------------------------------------------------------------------------
# Aggregate and compute statistics
# ---------------------------------------------------------------------------

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: proportion of (x_i > y_j) - (x_i < y_j) pairs."""
    n, m = len(x), len(y)
    gt = sum(xi > yj for xi in x for yj in y)
    lt = sum(xi < yj for xi in x for yj in y)
    return (gt - lt) / (n * m)


print("\n" + "=" * 80)
print(f"{'n':>4}  {'Baseline':<14}  {'Metric':<30}  {'DSP mean':>10}  {'BL mean':>10}  "
      f"{'p-value':>10}  {'marker':>6}  {'delta':>8}")
print("=" * 80)

summary_rows = []

for n in N_VALUES:
    raw_dir = SWEEP_ROOT / f"raw_n{n}"
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text())
        image_id = data["image_id"]
        for method in ["dsp"] + BASELINES:
            mdata = data.get("methods", {}).get(method, {})
            if "error" in mdata:
                continue
            row = {"image_id": image_id, "method": method}
            for metric in METRICS:
                row[metric] = mdata.get("metrics", {}).get(metric, float("nan"))
            records.append(row)

    if not records:
        print(f"n={n}: no results found")
        continue

    df = pd.DataFrame(records)
    dsp_df = df[df["method"] == "dsp"].set_index("image_id")

    for baseline in BASELINES:
        bl_df = df[df["method"] == baseline].set_index("image_id")
        shared = dsp_df.index.intersection(bl_df.index)

        for metric in METRICS:
            d = dsp_df.loc[shared, metric].dropna()
            b = bl_df.loc[shared, metric].dropna()
            common = d.index.intersection(b.index)
            d, b = d[common], b[common]

            diff = d - b
            stat, p = stats.wilcoxon(diff)
            delta = _cliffs_delta(d.values, b.values)
            marker = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

            print(
                f"{n:>4}  {baseline:<14}  {metric:<30}  {d.mean():>10.3f}  "
                f"{b.mean():>10.3f}  {p:>10.4f}  {marker:>6}  {delta:>+8.3f}"
            )
            summary_rows.append({
                "n": n, "baseline": baseline, "metric": metric,
                "dsp_mean": d.mean(), "bl_mean": b.mean(),
                "p_value": p, "marker": marker, "cliffs_delta": delta,
            })

    print()

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
out_csv = SWEEP_ROOT / "n_sweep_stats.csv"
summary_df.to_csv(out_csv, index=False)
print(f"\nSaved summary → {out_csv}")
