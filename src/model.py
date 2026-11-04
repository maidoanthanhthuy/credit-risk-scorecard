"""Logistic regression on WoE features with sign, p-value and VIF checks.

WoE is defined as ``ln(%good / %bad)`` (see :func:`src.woe.woe_table`), so a
higher WoE means lower risk. With ``TARGET = 1`` for default, every
coefficient must therefore be negative.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor

CONST = "const"


def fit_logit(X: pd.DataFrame, y: pd.Series) -> BinaryResultsWrapper:
    """Fit a logistic regression with intercept using statsmodels.

    Args:
        X: WoE-encoded features.
        y: Binary target aligned with ``X`` (1 = bad/default).

    Returns:
        Fitted statsmodels Logit result (gives coefficients and p-values).
    """
    exog = sm.add_constant(X, has_constant="add")
    return sm.Logit(y, exog).fit(disp=0)


def coefficient_table(result: BinaryResultsWrapper) -> pd.DataFrame:
    """Summarise fitted coefficients, excluding the intercept.

    Args:
        result: Output of :func:`fit_logit`.

    Returns:
        DataFrame indexed by feature with columns ``coef``, ``std_err`` and
        ``p_value``.
    """
    table = pd.DataFrame(
        {"coef": result.params, "std_err": result.bse, "p_value": result.pvalues}
    )
    return table.drop(index=CONST)


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute the variance inflation factor of each feature.

    An intercept is added before computing VIF, as recommended by
    statsmodels; it is not included in the output.

    Args:
        X: WoE-encoded features.

    Returns:
        VIF per feature, indexed by feature name, sorted descending.
    """
    exog = sm.add_constant(X, has_constant="add")
    values = exog.to_numpy(dtype=float)
    vif = pd.Series(
        [variance_inflation_factor(values, i) for i in range(values.shape[1])],
        index=exog.columns,
        name="vif",
    )
    return vif.drop(index=CONST).sort_values(ascending=False)


def next_violation(
    X: pd.DataFrame, y: pd.Series, max_p: float = 0.05, max_vif: float = 5.0
) -> tuple[str, str, float] | None:
    """Find the single worst violation in a fitted model, by priority.

    Priority: (1) positive coefficient (largest one), (2) p-value above
    ``max_p`` (largest one), (3) VIF above ``max_vif`` (largest one).

    Args:
        X: WoE-encoded features.
        y: Binary target aligned with ``X``.
        max_p: Maximum accepted p-value.
        max_vif: Maximum accepted VIF.

    Returns:
        ``(feature, reason, value)`` with reason in ``{"positive_coef",
        "p_value", "vif"}``, or ``None`` if the model has no violation.
    """
    coefs = coefficient_table(fit_logit(X, y))
    positive = coefs["coef"][coefs["coef"] > 0]
    if not positive.empty:
        return positive.idxmax(), "positive_coef", float(positive.max())
    high_p = coefs["p_value"][coefs["p_value"] > max_p]
    if not high_p.empty:
        return high_p.idxmax(), "p_value", float(high_p.max())
    if X.shape[1] > 1:
        vif = compute_vif(X)
        if vif.iloc[0] > max_vif:
            return vif.index[0], "vif", float(vif.iloc[0])
    return None


def stepwise_drop(
    X: pd.DataFrame, y: pd.Series, max_p: float = 0.05, max_vif: float = 5.0
) -> tuple[list[str], pd.DataFrame]:
    """Iteratively drop the worst violating feature and refit until clean.

    Each step refits the model and removes one feature found by
    :func:`next_violation`, so earlier removals can resolve later issues
    (e.g. a sign flip caused by collinearity).

    Args:
        X: WoE-encoded candidate features.
        y: Binary target aligned with ``X``.
        max_p: Maximum accepted p-value.
        max_vif: Maximum accepted VIF.

    Returns:
        Kept features (original column order) and a drop log with columns
        ``step``, ``feature``, ``reason`` and ``value``.
    """
    kept = list(X.columns)
    log: list[dict[str, object]] = []
    while kept:
        violation = next_violation(X[kept], y, max_p=max_p, max_vif=max_vif)
        if violation is None:
            break
        feature, reason, value = violation
        kept.remove(feature)
        log.append({"step": len(log) + 1, "feature": feature, "reason": reason, "value": value})
    return kept, pd.DataFrame(log, columns=["step", "feature", "reason", "value"])


def predict_pd(result: BinaryResultsWrapper, X: pd.DataFrame) -> pd.Series:
    """Predict the probability of default with a fitted logit.

    Args:
        result: Output of :func:`fit_logit` (may have had ``remove_data()`` called).
        X: WoE-encoded features, same columns as used for fitting.

    Returns:
        Probability of default, indexed like ``X``.
    """
    exog = sm.add_constant(X, has_constant="add")
    return pd.Series(result.predict(exog), index=X.index, name="pd")
