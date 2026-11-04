"""LightGBM challenger model: feature preparation, monotone constraints, training.

Raw (cleaned) features are used instead of WoE: LightGBM handles missing
values natively and categorical columns through the pandas ``category`` dtype.
Category levels are fit on train and reused unchanged on any other set.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "cat_smooth": 20.0,
    "random_state": 42,
    "verbose": -1,
}


def category_levels(df: pd.DataFrame) -> dict[str, list[str]]:
    """Collect the sorted levels of every non-numeric column.

    Args:
        df: Training data.

    Returns:
        Mapping of column name to its observed levels (missing excluded).
    """
    cols = df.select_dtypes(exclude="number").columns
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in cols}


def prepare_features(
    df: pd.DataFrame, features: list[str], levels: dict[str, list[str]]
) -> pd.DataFrame:
    """Select features and cast categorical columns to fixed categories.

    Args:
        df: Cleaned data (train, test or new applications).
        features: Columns to keep, in model order.
        levels: Output of :func:`category_levels` fit on train. Values not
            in the train levels become missing.

    Returns:
        Feature frame ready for LightGBM; input is not modified.
    """
    X = df[features].copy()
    for c in features:
        if c in levels:
            values = X[c].astype("object")
            X[c] = pd.Categorical(values.where(values.isin(levels[c])), categories=levels[c])
    return X


def monotone_directions(X: pd.DataFrame, y: pd.Series) -> dict[str, int]:
    """Derive a monotone constraint per feature from its trend with the target.

    Numeric features get ``+1`` if risk rises with the value (Spearman
    correlation with the target > 0), else ``-1``; categorical features get
    ``0`` (LightGBM cannot constrain them). Missing values are ignored, and
    LightGBM keeps them unconstrained.

    Args:
        X: Training features (output of :func:`prepare_features`).
        y: Binary target aligned with ``X`` (1 = bad/default).

    Returns:
        Mapping of feature to ``-1``, ``0`` or ``+1``, in ``X`` column order.
    """
    out: dict[str, int] = {}
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            out[c] = 0
        else:
            rho = X[c].corr(y, method="spearman")
            out[c] = 1 if rho > 0 else -1
    return out


def train_lgbm(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, object] | None = None,
    monotone: dict[str, int] | None = None,
    max_rounds: int = 5000,
    patience: int = 200,
) -> tuple[lgb.LGBMClassifier, int]:
    """Train LightGBM with early stopping on a validation set (AUC).

    Args:
        X_fit: Training features.
        y_fit: Training target.
        X_val: Validation features (carved from train, never the test set).
        y_val: Validation target.
        params: LightGBM parameters; defaults to :data:`DEFAULT_PARAMS`.
        monotone: Optional output of :func:`monotone_directions`.
        max_rounds: Maximum number of boosting rounds.
        patience: Stop after this many rounds without AUC improvement.

    Returns:
        Fitted model and its best number of rounds.
    """
    kwargs = {**(params or DEFAULT_PARAMS), "n_estimators": max_rounds}
    if monotone is not None:
        kwargs["monotone_constraints"] = [monotone[c] for c in X_fit.columns]
    model = lgb.LGBMClassifier(**kwargs)
    model.fit(
        X_fit,
        y_fit,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="auc",
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return model, int(model.best_iteration_)


def refit_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_rounds: int,
    params: dict[str, object] | None = None,
    monotone: dict[str, int] | None = None,
) -> lgb.LGBMClassifier:
    """Refit LightGBM on the full training set with a fixed number of rounds.

    Args:
        X: Full training features.
        y: Full training target.
        n_rounds: Number of boosting rounds (e.g. best rounds from
            :func:`train_lgbm`, scaled up for the larger sample).
        params: LightGBM parameters; defaults to :data:`DEFAULT_PARAMS`.
        monotone: Optional output of :func:`monotone_directions`.

    Returns:
        Fitted model.
    """
    kwargs = {**(params or DEFAULT_PARAMS), "n_estimators": n_rounds}
    if monotone is not None:
        kwargs["monotone_constraints"] = [monotone[c] for c in X.columns]
    return lgb.LGBMClassifier(**kwargs).fit(X, y)


def ice_monotone_share(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    feature: str,
    direction: int,
    n_grid: int = 20,
) -> float:
    """Share of rows whose individual PD curve follows the expected direction.

    For each row, ``feature`` is swept over a quantile grid of its observed
    values (other features fixed) and the predicted PD is checked to be
    non-decreasing (``direction = +1``) or non-increasing (``-1``).

    Args:
        model: Fitted LightGBM classifier.
        X: Sample of features (a few thousand rows is enough).
        feature: Numeric feature to sweep.
        direction: Expected direction, ``+1`` or ``-1``.
        n_grid: Number of grid points.

    Returns:
        Share of rows in ``[0, 1]`` with a monotone curve.
    """
    grid = np.unique(np.nanquantile(X[feature], np.linspace(0, 1, n_grid)))
    curves = np.column_stack(
        [model.predict_proba(X.assign(**{feature: v}))[:, 1] for v in grid]
    )
    steps = np.diff(curves, axis=1) * direction
    return float((steps >= -1e-12).all(axis=1).mean())
