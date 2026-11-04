"""Tests for src.explain."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.explain import (  # noqa: E402
    base_score,
    center_points,
    lgbm_contributions,
    mean_abs_importance,
    numeric_view,
    pick_case,
    reason_codes,
    to_points,
)
from src.gbm import category_levels, prepare_features, refit_lgbm  # noqa: E402
from src.scorecard import pd_to_score, scaling_params  # noqa: E402

FAST = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "verbose": -1}


def _fitted() -> tuple[object, pd.DataFrame]:
    """Small LightGBM model with one numeric and one categorical feature."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=1000), "cat": rng.choice(["a", "b"], size=1000)})
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(2 - df["x"] - (df["cat"] == "b")))))
    X = prepare_features(df, ["x", "cat"], category_levels(df))
    return refit_lgbm(X, y, n_rounds=30, params=FAST), X


def test_contributions_sum_to_raw_score() -> None:
    model, X = _fitted()
    contrib, base = lgbm_contributions(model, X)
    assert list(contrib.columns) == ["x", "cat"]
    assert np.allclose(contrib.sum(axis=1) + base, model.predict(X, raw_score=True))


def test_points_sum_to_credit_score() -> None:
    model, X = _fitted()
    factor, offset = scaling_params()
    contrib, base = lgbm_contributions(model, X)
    score = base_score(base, factor, offset) + to_points(contrib, factor).sum(axis=1)
    pd_ = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    assert np.allclose(score, pd_to_score(pd_, factor, offset))


def test_center_points_uses_reference_mean() -> None:
    ref = pd.DataFrame({"a": [10.0, 20.0], "b": [5.0, 5.0]})
    out = center_points(pd.DataFrame({"a": [25.0], "b": [5.0]}), ref)
    assert out.iloc[0].tolist() == [10.0, 0.0]


def test_numeric_view_codes_categories() -> None:
    X = pd.DataFrame({"c": pd.Categorical(["b", None, "a"], categories=["a", "b"]), "x": [1, 2, 3]})
    out = numeric_view(X)
    assert out["c"].iloc[0] == 1 and np.isnan(out["c"].iloc[1]) and out["c"].iloc[2] == 0
    assert (out.dtypes == float).all()


def test_mean_abs_importance_sorted() -> None:
    pts = pd.DataFrame({"small": [1.0, -1.0], "big": [-5.0, 3.0]})
    assert mean_abs_importance(pts).index.tolist() == ["big", "small"]


def test_reason_codes_sign_and_order() -> None:
    row = pd.Series({"a": -3.0, "b": 2.0, "c": -8.0, "d": 0.5})
    assert reason_codes(row, k=4).index.tolist() == ["c", "a"]
    assert reason_codes(row, k=1, adverse=False).index.tolist() == ["b"]


def test_pick_case_closest_within_mask() -> None:
    score = pd.Series([500.0, 540.0, 560.0, 600.0], index=list("wxyz"))
    mask = pd.Series([True, True, False, True], index=list("wxyz"))
    assert pick_case(score, mask, 555.0) == "x"
    with pytest.raises(ValueError):
        pick_case(score, mask & False, 555.0)
