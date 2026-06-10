"""Regenerate fig2_violins.pdf from updated 75-image test-set data."""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

METHOD_ORDER = ["dsp", "kmeans_lab", "kmeans_rgb", "median_cut"]
METHOD_COLORS = {
    "dsp":        "#FC2233",
    "median_cut": "#859EAD",
    "kmeans_rgb": "#6AAEA6",
    "kmeans_lab": "#D7C599",
}
METHOD_LABELS = {
    "dsp":        "DSP",
    "median_cut": "Median Cut",
    "kmeans_rgb": "k-Means RGB",
    "kmeans_lab": "k-Means Lab",
}
METRIC_COLS = [
    "min_pairwise_de2000",
    "wcag_aa_coverage",
    "reconstruction_error_de2000",
    "harmony_alignment",
]
METRIC_LABELS = {
    "min_pairwise_de2000":             r"Min $\Delta E_{2000}$ $\uparrow$",
    "wcag_aa_coverage":                r"WCAG AA Coverage $\uparrow$",
    "reconstruction_error_de2000":     r"Recon.\ $\Delta E_{2000}$ $\downarrow$",
    "harmony_alignment":               r"Harmony Alignment $\uparrow$",
}


def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _add_significance_bracket(ax, x1, x2, y, h, marker, color="#333333", fontsize=8):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=color, clip_on=False)
    ax.text(
        (x1 + x2) / 2, y + h + 0.003 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        marker, ha="center", va="bottom", fontsize=fontsize, color=color,
    )


# Load updated data — filter to the 75-image held-out test set only
# (excludes the 15 dev-set images used for parameter selection)
_manifest = json.loads(
    (Path(__file__).parents[1] / "corpus" / "manifest.json").read_text()
)
_test_ids = {
    e["id"]
    for e in _manifest
    if e["subset"] == "photographs" and not e.get("dev", False)
}

tidy_df = pd.read_csv("results/aggregated/tidy_results.csv")
tidy_df = tidy_df[tidy_df["image_id"].isin(_test_ids)].copy()

# Recompute Wilcoxon tests on the correct N=75 test set
_dsp = tidy_df[tidy_df["method"] == "dsp"]
_records = []
for _bl in [m for m in METHOD_ORDER if m != "dsp"]:
    _bl_df = tidy_df[tidy_df["method"] == _bl]
    for _metric in METRIC_COLS:
        _d = _dsp.set_index("image_id")[_metric]
        _b = _bl_df.set_index("image_id")[_metric]
        _shared = _d.index.intersection(_b.index)
        _diff = (_d[_shared] - _b[_shared]).dropna()
        _stat, _p = stats.wilcoxon(_diff)
        _records.append({"method": _bl, "metric": _metric, "p_value": _p})
wtest_df = pd.DataFrame(_records)

print(f"tidy_df: {len(tidy_df)} rows, {tidy_df['image_id'].nunique()} images (test set only)")
print(f"wtest_df: {len(wtest_df)} rows (recomputed on N=75)")

methods = [m for m in METHOD_ORDER if m in tidy_df["method"].unique()]
n_metrics = len(METRIC_COLS)

fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
fig.subplots_adjust(wspace=0.40)

for ax, metric in zip(axes, METRIC_COLS):
    data_per_method = [
        tidy_df[tidy_df["method"] == m][metric].dropna().values for m in methods
    ]
    positions = list(range(len(methods)))
    colors = [METHOD_COLORS.get(m, "#888888") for m in methods]

    parts = ax.violinplot(
        data_per_method, positions=positions,
        showmedians=True, quantiles=[[0.25, 0.75]] * len(methods),
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor("#444444")
        pc.set_alpha(0.70)
    for part_name in ("cmedians", "cquantiles", "cmins", "cmaxes", "cbars"):
        if part_name in parts:
            parts[part_name].set_color("#333333")
            parts[part_name].set_linewidth(0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [METHOD_LABELS.get(m, m) for m in methods],
        rotation=30, ha="right", fontsize=7,
    )
    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=8, pad=4)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(axis="y", labelsize=7)

    if metric == "min_pairwise_de2000":
        ax.set_ylim(bottom=0)

    # Significance brackets: DSP (pos 0) vs each baseline
    if not wtest_df.empty:
        dsp_pos = methods.index("dsp") if "dsp" in methods else 0
        y_max = max(np.max(d) if len(d) else 0 for d in data_per_method)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        step = y_range * 0.09

        brackets_drawn = 0
        for k, method in enumerate(methods):
            if method == "dsp":
                continue
            wrow = wtest_df[
                (wtest_df["method"] == method) &
                (wtest_df["metric"] == metric)
            ]
            if wrow.empty:
                continue
            p_val = float(wrow.iloc[0]["p_value"])
            marker = _sig_marker(p_val)
            if marker == "ns":
                continue
            bracket_y = y_max + step * (k * 0.6 + 0.4)
            ax.set_ylim(ax.get_ylim()[0], bracket_y + step * 1.5)
            _add_significance_bracket(
                ax, dsp_pos, k, bracket_y, step * 0.3,
                marker, fontsize=8,
            )
            brackets_drawn += 1

        if brackets_drawn == 0:
            ax.text(
                0.5, 0.98, "all n.s.",
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7, color="#888888", style="italic",
            )

output = Path("figures/fig2_violins.pdf")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {output}")
