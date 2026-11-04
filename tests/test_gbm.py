"""Tests for src.gbm."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.gbm import (  # noqa: E402
    category_levels,
    ice_monotone_share,
    monotone_directions,
    prepare_features,
    refit_lgbm,
    train_lgbm,
)

FAST = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "verbose": -1}


def _make_data(n: int = 3000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """``up`` raises risk, ``down`` lowers it, ``cat`` level ``b`` is risky."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "up": rng.normal(size=n),
        "down": rng.normal(size=n),
        "cat": rng.choice(["a", "b", "c"], size=n),
    })
    logit = -2 + 1.0 * df["up"] - 1.0 * df["down"] + 1.0 * (df["cat"] == "b")
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))), name="TARGET")
    return df, y


def test_prepare_features_fixes_categories_and_keeps_input() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0], "cat": ["a", "b"]})
    test = pd.DataFrame({"x": [3.0, 4.0], "cat": ["b", "zzz"]})
    levels = category_levels(train)
    X = prepare_features(test, ["x", "cat"], levels)
    assert levels == {"cat": ["a", "b"]}
    assert list(X["cat"].cat.categories) == ["a", "b"]
    assert X["cat"].iloc[0] == "b" and pd.isna(X["cat"].iloc[1])
    assert test["cat"].tolist() == ["b", "zzz"]


def test_monotone_directions_signs() -> None:
    df, y = _make_data()
    X = prepare_features(df, list(df.columns), category_levels(df))
    assert monotone_directions(X, y) == {"up": 1, "down": -1, "cat": 0}


def test_monotone_model_respects_constraints() -> None:
    df, y = _make_data()
    X = prepare_features(df, list(df.columns), category_levels(df))
    mono = monotone_directions(X, y)
    model, best = train_lgbm(X[:2400], y[:2400], X[2400:], y[2400:], FAST, mono, max_rounds=300, patience=30)
    assert best >= 1
    sample = X[:300]
    assert ice_monotone_share(model, sample, "up", +1) == 1.0
    assert ice_monotone_share(model, sample, "down", -1) == 1.0


def test_refit_uses_fixed_rounds() -> None:
    df, y = _make_data(n=500)
    X = prepare_features(df, list(df.columns), category_levels(df))
    model = refit_lgbm(X, y, n_rounds=25, params=FAST)
    assert model.booster_.num_trees() == 25
