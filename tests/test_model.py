"""Tests for src.model."""

import numpy as np
import pandas as pd

from src.model import (
    coefficient_table,
    compute_vif,
    fit_logit,
    next_violation,
    predict_pd,
    stepwise_drop,
)


def _make_data(n: int = 5000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic WoE-like data: ``a``, ``b`` reduce risk, ``pos`` increases it."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "pos"])
    logit = -2.0 - 1.0 * X["a"] - 0.8 * X["b"] + 0.8 * X["pos"]
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="TARGET")
    return X, y


def test_coefficient_table_excludes_const_and_recovers_signs() -> None:
    X, y = _make_data()
    coefs = coefficient_table(fit_logit(X, y))
    assert list(coefs.columns) == ["coef", "std_err", "p_value"]
    assert "const" not in coefs.index
    assert coefs.loc["a", "coef"] < 0 and coefs.loc["pos", "coef"] > 0


def test_compute_vif_flags_collinear_pair() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    X = pd.DataFrame(
        {"x": x, "dup": x + rng.normal(scale=0.1, size=1000), "indep": rng.normal(size=1000)}
    )
    vif = compute_vif(X)
    assert set(vif.index) == {"x", "dup", "indep"}
    assert vif["x"] > 10 and vif["dup"] > 10 and vif["indep"] < 2


def test_next_violation_prioritises_positive_coefficient() -> None:
    X, y = _make_data()
    assert next_violation(X, y)[:2] == ("pos", "positive_coef")
    assert next_violation(X[["a", "b"]], y) is None


def test_stepwise_drop_removes_positive_feature_and_logs() -> None:
    X, y = _make_data()
    kept, log = stepwise_drop(X, y)
    assert kept == ["a", "b"]
    assert log[["feature", "reason"]].values.tolist() == [["pos", "positive_coef"]]


def test_stepwise_drop_resolves_collinearity() -> None:
    X, y = _make_data()
    rng = np.random.default_rng(2)
    X = X[["a", "b"]].assign(a_dup=X["a"] + rng.normal(scale=0.05, size=len(X)))
    kept, log = stepwise_drop(X, y, max_p=1.0)
    assert "b" in kept
    assert len({"a", "a_dup"} & set(kept)) == 1
    assert len(log) == 1


def test_predict_pd_matches_fitted_values() -> None:
    X, y = _make_data()
    result = fit_logit(X, y)
    pd_hat = predict_pd(result, X)
    assert pd_hat.index.equals(X.index)
    assert np.allclose(pd_hat, result.predict())
