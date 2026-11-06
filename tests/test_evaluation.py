"""Tests for src.evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    bootstrap_gini,
    calibration_table,
    csi,
    gains_table,
    gini,
    ks_statistic,
    percentile_ci,
    performance_summary,
    pr_auc,
    psi,
)


def test_perfect_ranking_scores_one() -> None:
    y = pd.Series([0, 0, 0, 1, 1])
    score = pd.Series([0.1, 0.2, 0.3, 0.8, 0.9])
    assert gini(y, score) == pytest.approx(1.0)
    assert ks_statistic(y, score) == pytest.approx(1.0)


def test_reversed_ranking_has_negative_gini() -> None:
    y = pd.Series([0, 0, 1, 1])
    score = pd.Series([0.9, 0.8, 0.2, 0.1])
    assert gini(y, score) == pytest.approx(-1.0)


def test_performance_summary_is_consistent() -> None:
    rng = np.random.default_rng(0)
    score = pd.Series(rng.uniform(size=1000))
    y = pd.Series(rng.binomial(1, score))
    summary = performance_summary(y, score)
    assert list(summary.index) == ["auc", "gini", "ks"]
    assert summary["gini"] == pytest.approx(2 * summary["auc"] - 1)
    assert summary["ks"] == pytest.approx(ks_statistic(y, score))


def test_gains_table_bands_and_cumulatives() -> None:
    rng = np.random.default_rng(1)
    score = pd.Series(rng.uniform(size=1000))
    y = pd.Series(rng.binomial(1, score))
    table = gains_table(y, score, n_bins=10)
    assert list(table.index) == list(range(1, 11))
    assert (table["n"] == 100).all()
    assert table["n_bad"].sum() == y.sum()
    assert table.loc[1, "bad_rate"] > table.loc[10, "bad_rate"]
    assert table.loc[1, "min_score"] >= table.loc[2, "max_score"]
    assert table[["cum_pop_pct", "cum_bad_pct", "cum_good_pct"]].iloc[-1].tolist() == pytest.approx([1, 1, 1])


def test_gains_table_handles_tied_scores() -> None:
    y = pd.Series([1, 0] * 50)
    score = pd.Series([0.5] * 100)
    assert (gains_table(y, score, n_bins=4)["n"] == 25).all()


def test_calibration_table_ratio_near_one_when_calibrated() -> None:
    rng = np.random.default_rng(2)
    pd_ = pd.Series(rng.uniform(0.01, 0.3, size=20000))
    y = pd.Series(rng.binomial(1, pd_))
    table = calibration_table(y, pd_, n_bins=5)
    assert (table["n"] == 4000).all()
    assert table["ratio"].between(0.85, 1.15).all()


def test_psi_zero_for_same_and_large_for_shift() -> None:
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(size=10000))
    assert psi(base, base) == pytest.approx(0.0)
    assert psi(base, base + 1.0) > 0.25


def test_pr_auc_perfect_and_random() -> None:
    y = pd.Series([0, 0, 0, 1, 1])
    assert pr_auc(y, pd.Series([0.1, 0.2, 0.3, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(4)
    y_rand = pd.Series(rng.binomial(1, 0.1, size=20000))
    assert pr_auc(y_rand, pd.Series(rng.uniform(size=20000))) == pytest.approx(0.1, abs=0.01)


def test_gains_table_lift() -> None:
    rng = np.random.default_rng(5)
    score = pd.Series(rng.uniform(size=1000))
    y = pd.Series(rng.binomial(1, score))
    table = gains_table(y, score, n_bins=10)
    # Equal-size bands: average lift is 1 and cumulative lift ends at 1
    assert table["lift"].mean() == pytest.approx(1.0)
    assert table["cum_lift"].iloc[-1] == pytest.approx(1.0)
    assert table.loc[1, "lift"] > 1 > table.loc[10, "lift"]


def test_bootstrap_gini_shape_seed_and_centre() -> None:
    rng = np.random.default_rng(6)
    p = rng.uniform(size=2000)
    y = pd.Series(rng.binomial(1, p))
    scores = pd.DataFrame({"good": p, "noise": rng.uniform(size=2000)})
    samples = bootstrap_gini(y, scores, n_boot=50, seed=1)
    assert samples.shape == (50, 2)
    pd.testing.assert_frame_equal(samples, bootstrap_gini(y, scores, n_boot=50, seed=1))
    assert samples["good"].mean() == pytest.approx(gini(y, scores["good"]), abs=0.02)
    assert (samples["good"] > samples["noise"]).all()


def test_percentile_ci_bounds() -> None:
    samples = pd.DataFrame({"a": np.arange(101.0)})
    ci = percentile_ci(samples, level=0.9)
    assert list(ci.columns) == ["lower", "upper"]
    assert ci.loc["a"].tolist() == pytest.approx([5.0, 95.0])


def test_csi_zero_for_same_and_handles_unseen_level() -> None:
    base = pd.Series(["a"] * 50 + ["b"] * 50)
    assert csi(base, base) == pytest.approx(0.0)
    assert csi(base, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.25
    assert np.isfinite(csi(base, pd.Series(["a"] * 50 + ["c"] * 50)))
