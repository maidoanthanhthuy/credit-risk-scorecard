"""Tests for src.evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import calibration_table, gains_table, gini, ks_statistic, performance_summary, psi


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
