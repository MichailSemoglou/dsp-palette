"""
Ablation study report: statistics and LaTeX table (Table III).

Loads ablation per-image results, computes paired Wilcoxon tests against the
stored DSP results, and applies Bonferroni and Holm–Bonferroni corrections over
the ablation comparison family only.  The 12 original baseline comparisons
(3 baselines × 4 metrics, published in Table I) remain a separate Bonferroni
family at α = 0.05/12 ≈ 0.004 and are NOT re-corrected here.

Ablation Bonferroni family (separate from the original 12)
-----------------------------------------------------------
Conditions × metrics:
  A1–A5 (5 ablation conditions)  × 4 metrics = 20
  k-Means Lab + constraints      × 4 metrics =  4
  -------------------------------------------------------
  Total ablation comparisons                  = 24

  The k-Means Lab + constraints comparisons are included in the ablation
  family to assess a directly comparable, constraint-matched baseline.

α_corrected_ablation = 0.05 / 24 ≈ 0.00208
Holm–Bonferroni      = sort all 24 p-values ascending;
                       reject H_i if p_{(i)} < 0.05 / (24 − i + 1)

A3 is the DSP reproducer (expected: ns on all metrics — manipulation check).

Usage
-----
    python -m research.evaluation.ablation_report \\\
        --results-dir   results/raw/ \\
        --ablation-dir  results/ablation/raw/ \\
        --figures-dir   figures/ \\
        --output-dir    results/ablation/aggregated/ \\
        --wilcoxon-csv  results/aggregated/wilcoxon_tests.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All four evaluation metrics (same as METRIC_COLS in report.py)
METRIC_COLS = [
    "min_pairwise_de2000",
    "wcag_aa_coverage",
    "reconstruction_error_de2000",
    "harmony_alignment",
]

# Display order for ablation table rows
ABLATION_METHOD_ORDER = [
    "ablation_score_only",
    "ablation_score_constraint",
    "ablation_full_dsp",
    "ablation_freq_only",
    "ablation_dist_only",
    "kmeans_lab_constrained",
]

ABLATION_METHOD_LABELS = {
    "ablation_score_only":        r"A1: Score only (no $\tau$, no WCAG)",
    "ablation_score_constraint":  r"A2: Score + $\tau_{\text{dist}}$ constraint",
    "ablation_full_dsp":          r"A3: Full DSP (reproducer)",
    "ablation_freq_only":         r"A4: Freq.\ only ($\beta{=}0$) + constraint + WCAG",
    "ablation_dist_only":         r"A5: Dist.\ only ($\alpha{=}0$) + constraint + WCAG",
    "kmeans_lab_constrained":     r"k-Means Lab + constraints",
}

METRIC_HEADER = {
    "min_pairwise_de2000":          r"Min $\Delta E_{2000}$ $\uparrow$",
    "wcag_aa_coverage":             r"WCAG AA Cov.\ $\uparrow$",
    "reconstruction_error_de2000":  r"Recon.\ $\Delta E_{2000}$ $\downarrow$",
    "harmony_alignment":            r"Harmony $\uparrow$",
}

# Original 12-comparison family (3 baselines × 4 metrics) — published separately.
# NOT used for correction here; listed for documentation only.
N_ORIGINAL_COMPARISONS = 12

# Ablation comparison family: 6 methods × 4 metrics = 24.
# This is the family corrected in this report (separate from the original 12).
N_NEW_COMPARISONS = len(ABLATION_METHOD_ORDER) * len(METRIC_COLS)  # 24

# Grand total (documentation only; corrections are applied PER FAMILY).
N_TOTAL_COMPARISONS = N_ORIGINAL_COMPARISONS + N_NEW_COMPARISONS  # 36

ALPHA = 0.05
# Bonferroni threshold for the ABLATION family only (24 comparisons).
ALPHA_BONFERRONI = ALPHA / N_NEW_COMPARISONS  # 0.05 / 24 ≈ 0.00208


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_ablation_df(ablation_dir: Path) -> pd.DataFrame:
    """Load ablation per-image JSONs into a tidy DataFrame."""
    rows = []
    for path in sorted(ablation_dir.glob("coco_[0-9]*.json")):
        with open(path) as f:
            data = json.load(f)
        image_id = data["image_id"]
        for method, mdata in data.get("methods", {}).items():
            if "error" in mdata:
                continue
            row = {"image_id": image_id, "method": method}
            row.update(mdata.get("metrics", {}))
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No ablation result files found in {ablation_dir}")
    return pd.DataFrame(rows)


def _load_dsp_df(results_dir: Path) -> pd.DataFrame:
    """Load stored DSP per-image metrics (test set only, dev excluded)."""
    rows = []
    # Identify dev image IDs
    manifest_path = results_dir.parent.parent / "corpus" / "manifest.json"
    dev_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        dev_ids = {e["id"] for e in manifest if e.get("dev")}

    for path in sorted(results_dir.glob("coco_[0-9]*.json")):
        with open(path) as f:
            data = json.load(f)
        image_id = data["image_id"]
        if image_id in dev_ids:
            continue
        dsp_data = data.get("methods", {}).get("dsp")
        if dsp_data is None or "error" in dsp_data:
            continue
        row = {"image_id": image_id, "method": "dsp"}
        row.update(dsp_data.get("metrics", {}))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: (# pairs x>y  −  # pairs x<y) / n1·n2."""
    n1, n2 = len(x), len(y)
    dominance = (
        int(np.sum(x[:, None] > y[None, :]))
        - int(np.sum(x[:, None] < y[None, :]))
    )
    return dominance / (n1 * n2)


def _holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm–Bonferroni correction.  Returns list of rejection booleans."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    reject = [False] * n
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if p_values[idx] < threshold:
            reject[idx] = True
        else:
            # Once we fail to reject, all remaining are retained (step-down)
            break
    return reject


def compute_ablation_stats(
    ablation_df: pd.DataFrame,
    dsp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Wilcoxon + Cliff's delta for each ablation method × metric.

    DSP is the reference.  For A3 (the reproducer) a non-significant result
    is expected.

    Returns a DataFrame with columns:
        method, metric, n_pairs, mean_method, std_method, mean_dsp, std_dsp,
        wilcoxon_stat, p_value, cliffs_delta,
        sig_bonferroni, sig_holm
    (Holm correction is applied across the N_NEW_COMPARISONS ablation tests ONLY;
    Bonferroni is applied across the full N_TOTAL_COMPARISONS family.)
    """
    dsp_idx = dsp_df.set_index("image_id")
    rows = []

    for method in ABLATION_METHOD_ORDER:
        sub = ablation_df[ablation_df["method"] == method].set_index("image_id")
        common = dsp_idx.index.intersection(sub.index)
        if len(common) < 5:
            continue
        for metric in METRIC_COLS:
            if metric not in sub.columns or metric not in dsp_idx.columns:
                continue
            x_dsp = dsp_idx.loc[common, metric].values.astype(float)
            x_abl = sub.loc[common, metric].values.astype(float)

            try:
                stat, p = stats.wilcoxon(x_abl, x_dsp, alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), float("nan")

            cd = _cliffs_delta(x_abl, x_dsp)

            rows.append({
                "method": method,
                "metric": metric,
                "n_pairs": len(common),
                "mean_method": float(np.mean(x_abl)),
                "std_method": float(np.std(x_abl, ddof=1)),
                "mean_dsp": float(np.mean(x_dsp)),
                "std_dsp": float(np.std(x_dsp, ddof=1)),
                "wilcoxon_stat": round(stat, 4),
                "p_value": round(p, 6),
                "cliffs_delta": round(cd, 4),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Bonferroni significance over the full family (36 comparisons)
    df["sig_bonferroni"] = df["p_value"] < ALPHA_BONFERRONI

    # Holm–Bonferroni over the ablation comparisons only (24)
    holm_mask = _holm_bonferroni(df["p_value"].tolist())
    df["sig_holm"] = holm_mask

    return df


# ---------------------------------------------------------------------------
# Significance marker helper
# ---------------------------------------------------------------------------

def _sig_marker(p: float, bonferroni: bool, holm: bool) -> str:
    """Return a significance marker that reflects the Bonferroni-corrected threshold.

    Levels use α_corrected = 0.05/36 ≈ 0.00139 as the base threshold.
    """
    if math.isnan(p):
        return "ns"
    # Use standard presentation relative to α_corrected
    if p < ALPHA_BONFERRONI / 10:    # ~0.000139
        return "***"
    if p < ALPHA_BONFERRONI:          # ~0.00139
        return "**"
    if p < ALPHA:                     # 0.05
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def write_table3_latex(
    ablation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path,
    n_images: int,
) -> None:
    """Write LaTeX ablation table (booktabs, matching Table I/II style)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregated means + stds per method per metric
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for method in ABLATION_METHOD_ORDER:
        sub = ablation_df[ablation_df["method"] == method]
        agg[method] = {}
        for metric in METRIC_COLS:
            if metric in sub.columns:
                agg[method][metric] = (float(sub[metric].mean()), float(sub[metric].std()))
            else:
                agg[method][metric] = (float("nan"), float("nan"))

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Ablation study: mean $\pm$ std over $N="
        + str(n_images)
        + r"$ COCO test images. "
        r"Significance markers (Wilcoxon signed-rank, two-tailed) compare each "
        r"condition against full DSP; Bonferroni correction over $"
        + str(N_NEW_COMPARISONS)
        + r"$ ablation-family comparisons "
        r"($\alpha_{\text{corrected}} = 0.05/"
        + str(N_NEW_COMPARISONS)
        + r" \approx "
        + f"{ALPHA_BONFERRONI:.5f}"
        + r"$): $^{***}p < \alpha_c/10$, $^{**}p < \alpha_c$, $^*p < 0.05$, "
        r"$^{\mathrm{ns}}$not significant. "
        r"Condition A3 is the DSP reproducer (expected: ${\mathrm{ns}}$ on all metrics). "
        r"A1 = score only; A2 = score + $\tau_{\text{dist}}$ constraint; "
        r"A4 = freq.\ only ($\beta{=}0$); A5 = dist.\ only ($\alpha{=}0$).}"
    )
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    col_spec = "l" + "r" * len(METRIC_COLS)
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\hline\hline")
    header = "Condition & " + " & ".join(
        METRIC_HEADER[c] for c in METRIC_COLS
    ) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    # Separator position: ablation conditions above, constraint baseline below
    ablation_cond_keys = [k for k in ABLATION_METHOD_ORDER if k != "kmeans_lab_constrained"]
    constrained_keys = ["kmeans_lab_constrained"]

    def _row(method: str) -> str:
        cells = []
        for metric in METRIC_COLS:
            mean_v, std_v = agg.get(method, {}).get(metric, (float("nan"), float("nan")))
            if math.isnan(mean_v):
                cells.append("---")
                continue
            cell = rf"{mean_v:.3f}{{\scriptsize$\,\pm${std_v:.3f}}}"
            # Append significance marker (skip for A3 — it IS the reference)
            if method != "ablation_full_dsp" and not stats_df.empty:
                row = stats_df[
                    (stats_df["method"] == method) & (stats_df["metric"] == metric)
                ]
                if not row.empty:
                    p_val = float(row.iloc[0]["p_value"])
                    bonf = bool(row.iloc[0]["sig_bonferroni"])
                    holm = bool(row.iloc[0]["sig_holm"])
                    sig = _sig_marker(p_val, bonf, holm)
                    if sig != "ns":
                        cell += rf"$^{{\mathrm{{{sig}}}}}$"
                    else:
                        cell += r"$^{\mathrm{ns}}$"
            cells.append(cell)

        label = ABLATION_METHOD_LABELS.get(method, method)
        if method == "ablation_full_dsp":
            label = r"\textbf{" + label + r"}"
        return label + " & " + " & ".join(cells) + r" \\"

    for method in ablation_cond_keys:
        lines.append(_row(method))

    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(1 + len(METRIC_COLS)) + r"}{l}{"
                 r"\textit{Constraint-matched baseline}} \\")

    for method in constrained_keys:
        lines.append(_row(method))

    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX ablation table → {output_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _count_wcag_fires(ablation_dir: Path, results_dir: Path) -> dict:
    """Count WCAG replacement and distinctness-compromised events.

    Returns a dict with keys:
        dsp_replacement_applied   : int  (from stored results/raw DSP records)
        dsp_distinctness_compromised : int
        a3_replacement_applied    : int  (from ablation_full_dsp; should match DSP)
        n_images                  : int
    """
    counts: dict = {
        "dsp_replacement_applied": 0,
        "dsp_distinctness_compromised": 0,
        "a3_replacement_applied": 0,
        "n_images": 0,
    }
    # DSP from stored raw results (test set only; dev filtered by _load_dsp_df logic)
    manifest_path = results_dir.parent.parent / "corpus" / "manifest.json"
    dev_ids: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            dev_ids = {e["id"] for e in json.load(f) if e.get("dev")}
    for path in sorted(results_dir.glob("coco_[0-9]*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("image_id") in dev_ids:
            continue
        dsp = d.get("methods", {}).get("dsp", {})
        if not dsp or "error" in dsp:
            continue
        counts["dsp_replacement_applied"] += int(bool(dsp.get("wcag_replacement_applied")))
        counts["dsp_distinctness_compromised"] += int(bool(dsp.get("wcag_distinctness_compromised")))
        counts["n_images"] += 1
    # A3 from ablation raw results
    for path in sorted(ablation_dir.glob("coco_[0-9]*.json")):
        with open(path) as f:
            d = json.load(f)
        a3 = d.get("methods", {}).get("ablation_full_dsp", {})
        counts["a3_replacement_applied"] += int(bool(a3.get("wcag_replacement_applied")))
    return counts


def print_console_summary(
    stats_df: pd.DataFrame,
    dsp_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    wcag_counts: dict | None = None,
) -> None:
    """Print a formatted summary to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("ABLATION STUDY SUMMARY")
    print(f"Ablation Bonferroni family: {N_NEW_COMPARISONS} comparisons "
          f"(6 methods × 4 metrics; separate from the original 12-comparison baseline family)")
    print(f"α_corrected (ablation family) = 0.05 / {N_NEW_COMPARISONS} ≈ {ALPHA_BONFERRONI:.5f}")
    print("Original baseline family    = 12 comparisons at α = 0.05/12 ≈ 0.004 (published separately)")
    print(sep)

    dsp_agg = {
        metric: (float(dsp_df[metric].mean()), float(dsp_df[metric].std()))
        for metric in METRIC_COLS
        if metric in dsp_df.columns
    }
    print(f"\n{'DSP (reference)':40s}  ", end="")
    for metric in METRIC_COLS:
        m, s = dsp_agg.get(metric, (float("nan"), float("nan")))
        print(f"{m:7.3f}±{s:.3f}  ", end="")
    print()
    print(f"{'':40s}  " + "  ".join(f"{METRIC_HEADER[c]:>14s}" for c in METRIC_COLS))
    print()

    for method in ABLATION_METHOD_ORDER:
        sub = ablation_df[ablation_df["method"] == method]
        label = ABLATION_METHOD_LABELS.get(method, method)
        # Strip LaTeX markup for console display
        label_plain = (label
                       .replace(r"\textbf{", "").replace("}", "")
                       .replace(r"$\tau_{\text{dist}}$", "τ_dist")
                       .replace(r"$\beta{=}0$", "β=0")
                       .replace(r"$\alpha{=}0$", "α=0")
                       .replace(r"\ ", " ")
                       .replace("$", ""))
        print(f"\n{label_plain[:70]:70s}")
        vals_line = "  "
        stats_line = "  "
        for metric in METRIC_COLS:
            if metric not in sub.columns:
                vals_line += f"{'N/A':>16s}  "
                stats_line += f"{'':>16s}  "
                continue
            m = float(sub[metric].mean())
            s = float(sub[metric].std())
            vals_line += f"{m:7.3f}±{s:.3f}  "

            if method == "ablation_full_dsp":
                stats_line += f"{'(reference)':>14s}  "
                continue

            row = stats_df[
                (stats_df["method"] == method) & (stats_df["metric"] == metric)
            ]
            if row.empty:
                stats_line += f"{'':>14s}  "
            else:
                p = float(row.iloc[0]["p_value"])
                cd = float(row.iloc[0]["cliffs_delta"])
                bonf = bool(row.iloc[0]["sig_bonferroni"])
                holm = bool(row.iloc[0]["sig_holm"])
                sig = _sig_marker(p, bonf, holm)
                stats_line += f"p={p:.4f} d={cd:+.3f}{sig:>4s}  "

        print(vals_line)
        print(stats_line)

    # ── WCAG descriptive counts (plain; no significance claim) ─────────────
    if wcag_counts:
        n = wcag_counts["n_images"]
        ra  = wcag_counts["dsp_replacement_applied"]
        dc  = wcag_counts["dsp_distinctness_compromised"]
        a3r = wcag_counts["a3_replacement_applied"]
        print(f"\n--- WCAG post-selection descriptive counts (N={n} test images) ---")
        print(f"  DSP wcag_replacement_applied    : {ra} of {n}")
        print(f"  DSP wcag_distinctness_compromised: {dc} of {n}")
        print(f"  A3  wcag_replacement_applied    : {a3r} of {n}  "
              f"(should equal DSP: {'MATCH' if a3r == ra else 'MISMATCH'})")
        print("  Note: WCAG replacement count is reported as a plain descriptive")
        print("  number; no significance claim is made about its effect on")
        print("  reconstruction error.")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation report and LaTeX table (Table III).",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to existing results/raw/ (stored DSP per-image JSONs).",
    )
    parser.add_argument(
        "--ablation-dir", required=True,
        help="Path to results/ablation/raw/ (ablation per-image JSONs).",
    )
    parser.add_argument(
        "--figures-dir", required=True,
        help="Output directory for table3_ablation.tex.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for aggregated CSV files.",
    )
    parser.add_argument(
        "--wilcoxon-csv", default=None,
        help="Path to existing wilcoxon_tests.csv (for N_ORIGINAL_COMPARISONS "
             "documentation; not used computationally).",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    ablation_dir = Path(args.ablation_dir)
    figures_dir = Path(args.figures_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading ablation results from {ablation_dir} ...")
    ablation_df = _load_ablation_df(ablation_dir)
    print(f"  {len(ablation_df['image_id'].unique())} images, "
          f"{ablation_df['method'].unique().tolist()} methods found.")

    print(f"Loading DSP reference results from {results_dir} ...")
    dsp_df = _load_dsp_df(results_dir)
    print(f"  {len(dsp_df)} DSP test-set records loaded.")

    n_images = len(dsp_df["image_id"].unique())

    # Compute statistics
    print("\nComputing Wilcoxon statistics + Cliff's delta ...")
    stats_df = compute_ablation_stats(ablation_df, dsp_df)

    # Save CSVs
    stats_csv = output_dir / "ablation_wilcoxon.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved stats → {stats_csv}")

    agg_rows = []
    for method in ABLATION_METHOD_ORDER:
        sub = ablation_df[ablation_df["method"] == method]
        for metric in METRIC_COLS:
            if metric in sub.columns:
                agg_rows.append({
                    "method": method,
                    "metric": metric,
                    "mean": float(sub[metric].mean()),
                    "std": float(sub[metric].std()),
                    "median": float(sub[metric].median()),
                    "n": len(sub),
                })
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = output_dir / "ablation_aggregate.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregate → {agg_csv}")

    # Write LaTeX table
    table_path = figures_dir / "table3_ablation.tex"
    write_table3_latex(ablation_df, stats_df, table_path, n_images)

    # WCAG descriptive counts
    wcag_counts = _count_wcag_fires(ablation_dir, results_dir)

    # Console summary
    print_console_summary(stats_df, dsp_df, ablation_df, wcag_counts=wcag_counts)

    # Bonferroni documentation note
    print(
        f"Bonferroni correction documentation:\n"
        f"  Ablation family      : {N_NEW_COMPARISONS} comparisons "
        f"(6 methods \u00d7 {len(METRIC_COLS)} metrics; SEPARATE from original 12)\n"
        f"  \u03b1_Bonferroni (ablation) : 0.05 / {N_NEW_COMPARISONS} = {ALPHA_BONFERRONI:.6f}\n"
        f"  Original baseline family : {N_ORIGINAL_COMPARISONS} comparisons at "
        f"0.05/{N_ORIGINAL_COMPARISONS} \u2248 {0.05/N_ORIGINAL_COMPARISONS:.4f} (not re-corrected here)\n"
        f"  Holm\u2013Bonferroni      : applied across {N_NEW_COMPARISONS} ablation comparisons "
        f"(see ablation_wilcoxon.csv / sig_holm column)\n"
        f"\n"
        f"  Included in the ablation family: the 4 k-Means Lab + constraints comparisons.\n"
        f"  Full comparison list ({N_NEW_COMPARISONS}):\n"
    )
    for method in ABLATION_METHOD_ORDER:
        for metric in METRIC_COLS:
            print(f"    {method:40s} \u00d7 {metric}")



if __name__ == "__main__":
    main()
