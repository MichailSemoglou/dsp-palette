
"""Unit tests for evaluation/metrics.py."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import aggregate, wilcoxon_vs_dsp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(methods: list[str], n_images: int = 10, seed: int = 0) -> pd.DataFrame:
    """Create a minimal tidy DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for method in methods:
        for i in range(n_images):
            rows.append(
                {
                    "image_id": f"img_{i:03d}",
                    "subset": "photographs",
                    "method": method,
                    "min_pairwise_de2000": float(rng.uniform(5, 40)),
                    "wcag_aa_coverage": float(rng.uniform(0, 1)),
                    "wcag_aaa_coverage": float(rng.uniform(0, 1)),
                    "reconstruction_error_de2000": float(rng.uniform(5, 30)),
                    "harmony_alignment": float(rng.uniform(0, 1)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------------


def test_aggregate_returns_per_method_rows():
    df = _make_df(["dsp", "kmeans_rgb"], n_images=10)
    result = aggregate(df)
    assert set(result.index) == {"dsp", "kmeans_rgb"}


def test_aggregate_includes_mean_median_std():
    df = _make_df(["dsp"], n_images=10)
    result = aggregate(df)
    assert ("min_pairwise_de2000", "mean") in result.columns
    assert ("min_pairwise_de2000", "median") in result.columns
    assert ("min_pairwise_de2000", "std") in result.columns


# ---------------------------------------------------------------------------
# wilcoxon_vs_dsp()
# ---------------------------------------------------------------------------


def test_wilcoxon_vs_dsp_returns_dataframe_with_expected_columns():
    df = _make_df(["dsp", "kmeans_rgb"], n_images=20)
    result = wilcoxon_vs_dsp(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    for col in ("method", "metric", "wilcoxon_stat", "p_value", "cliffs_delta", "significant_05"):
        assert col in result.columns


def test_wilcoxon_cliffs_delta_in_minus_one_to_one():
    """Cliff's delta must always be in [-1, 1]."""
    df = _make_df(["dsp", "kmeans_rgb", "median_cut"], n_images=20)
    result = wilcoxon_vs_dsp(df)
    assert (result["cliffs_delta"].abs() <= 1.0 + 1e-9).all()


def test_wilcoxon_cliffs_delta_sign_dsp_dominates():
    """When DSP values are always higher than baseline, Cliff's delta > 0."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(20):
        rows.append(
            {
                "image_id": f"img_{i:03d}",
                "subset": "test",
                "method": "dsp",
                "min_pairwise_de2000": 30.0 + rng.uniform(0, 1),
                "wcag_aa_coverage": 0.0,
                "wcag_aaa_coverage": 0.0,
                "reconstruction_error_de2000": 0.0,
                "harmony_alignment": 0.0,
            }
        )
        rows.append(
            {
                "image_id": f"img_{i:03d}",
                "subset": "test",
                "method": "kmeans_rgb",
                "min_pairwise_de2000": 5.0 + rng.uniform(0, 1),
                "wcag_aa_coverage": 0.0,
                "wcag_aaa_coverage": 0.0,
                "reconstruction_error_de2000": 0.0,
                "harmony_alignment": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    result = wilcoxon_vs_dsp(df)
    row = result[
        (result["method"] == "kmeans_rgb") & (result["metric"] == "min_pairwise_de2000")
    ]
    assert not row.empty
    # DSP dominates: more dsp_values > baseline_values than baseline > dsp
    assert row.iloc[0]["cliffs_delta"] > 0


def test_wilcoxon_vs_dsp_warns_and_returns_empty_when_dsp_absent(caplog):
    """wilcoxon_vs_dsp must log a warning and return empty DataFrame when DSP is absent."""
    df = _make_df(["kmeans_rgb", "median_cut"], n_images=10)
    with caplog.at_level(logging.WARNING, logger="evaluation.metrics"):
        result = wilcoxon_vs_dsp(df)
    assert result.empty
    assert any("DSP" in msg or "dsp" in msg.lower() for msg in caplog.messages)


def test_wilcoxon_skips_zero_variance_pairs_without_runtime_warning():
    """Constant paired differences should be skipped, not passed to SciPy."""
    rows = []
    for i in range(10):
        value = float(i + 1)
        rows.append(
            {
                "image_id": f"img_{i:03d}",
                "subset": "test",
                "method": "dsp",
                "min_pairwise_de2000": value,
                "wcag_aa_coverage": 0.5,
                "wcag_aaa_coverage": 0.3,
                "reconstruction_error_de2000": value,
                "harmony_alignment": 0.5,
            }
        )
        rows.append(
            {
                "image_id": f"img_{i:03d}",
                "subset": "test",
                "method": "kmeans_rgb",
                "min_pairwise_de2000": value,
                "wcag_aa_coverage": 0.5,
                "wcag_aaa_coverage": 0.3,
                "reconstruction_error_de2000": value,
                "harmony_alignment": 0.5,
            }
        )
    df = pd.DataFrame(rows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wilcoxon_vs_dsp(df)

    assert result.empty
    assert not any("invalid value encountered" in str(w.message).lower() for w in caught)


def test_wilcoxon_skips_method_with_few_observations(caplog):
    """Fewer than 5 paired observations must be skipped with a warning."""
    rows = []
    for i in range(3):
        for method in ("dsp", "kmeans_rgb"):
            rows.append(
                {
                    "image_id": f"img_{i}",
                    "subset": "test",
                    "method": method,
                    "min_pairwise_de2000": float(i + 1),
                    "wcag_aa_coverage": 0.5,
                    "wcag_aaa_coverage": 0.3,
                    "reconstruction_error_de2000": float(i + 1),
                    "harmony_alignment": 0.5,
                }
            )
    df = pd.DataFrame(rows)
    with caplog.at_level(logging.WARNING, logger="evaluation.metrics"):
        result = wilcoxon_vs_dsp(df)
    # No rows for kmeans_rgb since < 5 pairs
    assert result.empty or (result["method"] != "kmeans_rgb").all()
