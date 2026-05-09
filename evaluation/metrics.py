
"""
Aggregate raw per-image results and run statistical tests.

Usage
-----
    python -m research.evaluation.metrics \
        --results-dir results/raw/ \
        --output-dir results/aggregated/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

logger = logging.getLogger(__name__)

METRIC_COLS = [
    "min_pairwise_de2000",
    "wcag_aa_coverage",
    "wcag_aaa_coverage",
    "reconstruction_error_de2000",
    "harmony_alignment",
]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all per-image JSON results into a tidy DataFrame."""
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        image_id = data["image_id"]
        subset = data.get("subset", "unknown")
        for method, mdata in data.get("methods", {}).items():
            if "error" in mdata:
                continue
            row = {
                "image_id": image_id,
                "subset": subset,
                "method": method,
            }
            row.update(mdata.get("metrics", {}))
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-method means, medians, std on each metric."""
    return (
        df.groupby("method")[METRIC_COLS]
        .agg(["mean", "median", "std"])
        .round(4)
    )


def wilcoxon_vs_dsp(df: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test: DSP vs each baseline on each metric.

    Returns a DataFrame with columns: method, metric, statistic, p_value,
    cliffs_delta, significant (α=0.05).
    """
    if "dsp" not in df["method"].values:
        logger.warning("DSP results not found; skipping statistical tests")
        return pd.DataFrame()

    dsp_rows = df[df["method"] == "dsp"].set_index("image_id")
    results = []

    for method in df["method"].unique():
        if method == "dsp":
            continue
        baseline_rows = df[df["method"] == method].set_index("image_id")
        common_ids = dsp_rows.index.intersection(baseline_rows.index)
        if len(common_ids) < 5:
            logger.warning(
                "Fewer than 5 paired observations for %s vs DSP; skipping", method
            )
            continue

        for metric in METRIC_COLS:
            if metric not in dsp_rows.columns or metric not in baseline_rows.columns:
                continue
            x = dsp_rows.loc[common_ids, metric].values.astype(float)
            y = baseline_rows.loc[common_ids, metric].values.astype(float)

            try:
                stat, p = stats.wilcoxon(x, y, alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), float("nan")

            # Cliff's delta (non-parametric effect size) — vectorised
            n1, n2 = len(x), len(y)
            dominance = int(np.sum(x[:, None] > y[None, :])) - int(np.sum(x[:, None] < y[None, :]))
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

    return pd.DataFrame(results)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="research/results/raw/")
    parser.add_argument("--output-dir", default="research/results/aggregated/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_dir))
    if df.empty:
        logger.error("No results found in %s", args.results_dir)
        return

    agg = aggregate(df)
    agg.to_csv(output_dir / "aggregate_metrics.csv")
    logger.info("Saved aggregate_metrics.csv")

    stats_df = wilcoxon_vs_dsp(df)
    if not stats_df.empty:
        stats_df.to_csv(output_dir / "wilcoxon_tests.csv", index=False)
        logger.info("Saved wilcoxon_tests.csv")

    df.to_csv(output_dir / "tidy_results.csv", index=False)
    logger.info("Saved tidy_results.csv")


if __name__ == "__main__":
    main()
