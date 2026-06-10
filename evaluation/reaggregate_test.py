"""Re-aggregate on 75 test-set images only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

METRIC_COLS = [
    "min_pairwise_de2000",
    "wcag_aa_coverage",
    "wcag_aaa_coverage",
    "reconstruction_error_de2000",
    "harmony_alignment",
]

# Load manifest → get test IDs
with open("corpus/manifest.json") as f:
    manifest = json.load(f)
test_ids = {x["id"] for x in manifest if x.get("source") == "COCO val2017" and not x.get("dev", False)}
print(f"Test-set size: {len(test_ids)}")

# Load raw results filtered to test set
results_dir = Path("results/raw")
rows = []
for path in sorted(results_dir.glob("*.json")):
    with open(path) as f:
        data = json.load(f)
    image_id = data["image_id"]
    if image_id not in test_ids:
        continue
    for method, mdata in data.get("methods", {}).items():
        if "error" in mdata:
            continue
        row = {"image_id": image_id, "method": method}
        row.update(mdata.get("metrics", {}))
        rows.append(row)

df = pd.DataFrame(rows)
print(f"Rows loaded: {len(df)}, unique images: {df['image_id'].nunique()}, methods: {sorted(df['method'].unique())}")

# Aggregate
agg = df.groupby("method")[METRIC_COLS].agg(["mean", "median", "std"]).round(4)
agg.to_csv("results/aggregated/aggregate_metrics.csv")
print("Saved aggregate_metrics.csv")

# Wilcoxon + Cliff's delta
dsp_rows = df[df["method"] == "dsp"].set_index("image_id")
results = []
for method in sorted(df["method"].unique()):
    if method == "dsp":
        continue
    base_rows = df[df["method"] == method].set_index("image_id")
    common = dsp_rows.index.intersection(base_rows.index)
    print(f"  {method}: {len(common)} paired observations")
    for metric in METRIC_COLS:
        x = dsp_rows.loc[common, metric].values.astype(float)
        y = base_rows.loc[common, metric].values.astype(float)
        try:
            stat, p = stats.wilcoxon(x, y, alternative="two-sided")
        except ValueError:
            stat, p = float("nan"), float("nan")
        n1, n2 = len(x), len(y)
        dominance = sum(
            1 if xi > yj else (-1 if xi < yj else 0) for xi in x for yj in y
        )
        cliffs_d = dominance / (n1 * n2)
        results.append(
            {
                "method": method,
                "metric": metric,
                "wilcoxon_stat": round(stat, 4),
                "p_value": round(p, 6),
                "cliffs_delta": round(cliffs_d, 4),
                "significant_05": p < 0.05,
            }
        )

stats_df = pd.DataFrame(results)
stats_df.to_csv("results/aggregated/wilcoxon_tests.csv", index=False)
print("Saved wilcoxon_tests.csv")
df.to_csv("results/aggregated/tidy_results.csv", index=False)
print("Saved tidy_results.csv")

# Print key numbers for the paper
METHOD_ORDER = ["dsp", "kmeans_lab", "kmeans_rgb", "median_cut"]
print("\n=== KEY NUMBERS ===")
for col in ["min_pairwise_de2000", "wcag_aa_coverage", "reconstruction_error_de2000", "harmony_alignment"]:
    print(f"\n{col}:")
    for m in METHOD_ORDER:
        if m not in df["method"].values:
            continue
        sub = df[df["method"] == m][col]
        print(f"  {m}: mean={sub.mean():.4f} std={sub.std():.4f} median={sub.median():.4f}")

print("\n=== WILCOXON (key metrics) ===")
for _, row in stats_df[stats_df["metric"].isin(["min_pairwise_de2000", "wcag_aa_coverage", "reconstruction_error_de2000"])].sort_values(["metric", "method"]).iterrows():
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
    print(f"  {row['method']:12s} {row['metric']:35s} p={row['p_value']:.2e}  delta={row['cliffs_delta']:+.4f}  {sig}")

print("\n=== WIN COUNTS on min_pairwise_de2000 ===")
dsp_de = dsp_rows["min_pairwise_de2000"]
for m in ["kmeans_lab", "kmeans_rgb", "median_cut"]:
    if m not in df["method"].values:
        continue
    base_de = df[df["method"] == m].set_index("image_id")["min_pairwise_de2000"]
    common = dsp_de.index.intersection(base_de.index)
    wins = (dsp_de.loc[common] > base_de.loc[common]).sum()
    print(f"  DSP > {m}: {wins}/{len(common)}")

print("\n=== DARK/LIGHT MODE (L*<40) ===")
dark = light = 0
for path in sorted(results_dir.glob("*.json")):
    with open(path) as f:
        data = json.load(f)
    if data["image_id"] not in test_ids:
        continue
    ml = data.get("image_mean_L", 0)
    if ml < 40:
        dark += 1
    else:
        light += 1
print(f"  dark (L*<40): {dark}, light: {light}, total: {dark+light}")
print(f"  dark%: {dark / (dark + light) * 100:.1f}%")

print("\n=== WDC + REPLACEMENT COUNTS ===")
wdc_count = replacement_count = 0
for path in sorted(results_dir.glob("*.json")):
    with open(path) as f:
        data = json.load(f)
    if data["image_id"] not in test_ids:
        continue
    dsp = data.get("methods", {}).get("dsp", {})
    if dsp.get("wcag_distinctness_compromised", False):
        wdc_count += 1
    if dsp.get("wcag_replacement_applied", False):
        replacement_count += 1
print(f"  wcag_distinctness_compromised: {wdc_count}/75")
print(f"  wcag_replacement_applied: {replacement_count}/75")
