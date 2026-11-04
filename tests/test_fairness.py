"""Tests for src.fairness."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.fairness import age_bands, disparate_impact, group_metrics, proxy_auc  # noqa: E402


def test_age_bands_labels_and_edges() -> None:
    bands = age_bands(pd.Series([20, 25, 34.9, 35, 54, 55, 70]))
    assert bands.astype(str).tolist() == ["<25", "25-34", "25-34", "35-44", "45-54", "55+", "55+"]


def test_group_metrics_values() -> None:
    y = pd.Series([0, 1, 0, 0, 1, 1])
    pd_ = pd.Series([0.1, 0.5, 0.2, 0.1, 0.4, 0.6])
    score = pd.Series([600, 500, 560, 610, 540, 520])
    group = pd.Series(["A", "A", "A", "B", "B", None])
    m = group_metrics(y, pd_, score, group, cutoff=550)
    assert m.loc["A", "n"] == 3 and m.loc["B", "n"] == 2  # missing group dropped
    assert m.loc["A", "approval_rate"] == pytest.approx(2 / 3)
    assert m.loc["A", "good_approval_rate"] == 1.0
    assert m.loc["B", "calib_ratio"] == pytest.approx(0.5 / 0.25)
    assert m.loc["A", "auc"] == 1.0  # lower score = defaulter


def test_disparate_impact_reference() -> None:
    rates = pd.Series({"A": 0.8, "B": 0.6})
    assert disparate_impact(rates).tolist() == pytest.approx([1.0, 0.75])
    assert disparate_impact(rates, reference="B")["A"] == pytest.approx(0.8 / 0.6)


def test_proxy_auc_detects_informative_feature() -> None:
    rng = np.random.default_rng(0)
    g = pd.Series(rng.integers(0, 2, size=2000))
    X = pd.DataFrame({"proxy": g + rng.normal(scale=0.3, size=2000), "noise": rng.normal(size=2000)})
    auc, _ = proxy_auc(X[:1500], g[:1500], X[1500:], g[1500:], n_rounds=50)
    auc_noise, _ = proxy_auc(X[["noise"]][:1500], g[:1500], X[["noise"]][1500:], g[1500:], n_rounds=50)
    assert auc > 0.9 and auc_noise < 0.6
