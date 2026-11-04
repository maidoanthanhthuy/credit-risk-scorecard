"""Discrimination metrics for a credit scoring model: AUC, Gini, KS, gains table."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y: pd.Series, score: pd.Series) -> float:
    """Compute the Gini coefficient (accuracy ratio), ``2 * AUC - 1``.

    Args:
        y: Binary target (1 = bad/default).
        score: Predicted risk, higher = riskier (e.g. probability of default).

    Returns:
        Gini in ``[-1, 1]``; 1 = perfect ranking, 0 = random.
    """
    return float(2 * roc_auc_score(y, score) - 1)


def ks_statistic(y: pd.Series, score: pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic.

    KS is the maximum gap between the cumulative distributions of bads and
    goods over all score thresholds, i.e. ``max(TPR - FPR)``.

    Args:
        y: Binary target (1 = bad/default).
        score: Predicted risk, higher = riskier.

    Returns:
        KS in ``[0, 1]``.
    """
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def performance_summary(y: pd.Series, score: pd.Series) -> pd.Series:
    """Collect AUC, Gini and KS in one Series.

    Args:
        y: Binary target (1 = bad/default).
        score: Predicted risk, higher = riskier.

    Returns:
        Series with index ``auc``, ``gini``, ``ks``.
    """
    auc = float(roc_auc_score(y, score))
    return pd.Series({"auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y, score)})


def gains_table(y: pd.Series, score: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Split the population into equal-size risk bands and summarise each.

    Band 1 holds the riskiest ``1 / n_bins`` of the population. Ties are
    broken by order so every band has the same size (± 1).

    Args:
        y: Binary target (1 = bad/default).
        score: Predicted risk, higher = riskier.
        n_bins: Number of bands (10 = deciles).

    Returns:
        DataFrame indexed by ``band`` (1..n_bins) with columns ``n``,
        ``n_bad``, ``bad_rate``, ``min_score``, ``max_score``,
        ``cum_pop_pct``, ``cum_bad_pct`` (share of all bads captured so far),
        ``cum_good_pct`` and ``ks`` (``cum_bad_pct - cum_good_pct``).
    """
    rank = score.rank(method="first", ascending=False)
    band = pd.qcut(rank, n_bins, labels=range(1, n_bins + 1))
    table = (
        pd.DataFrame({"y": y, "score": score, "band": band})
        .groupby("band", observed=True)
        .agg(n=("y", "size"), n_bad=("y", "sum"), min_score=("score", "min"), max_score=("score", "max"))
    )
    n_good = table["n"] - table["n_bad"]
    table["bad_rate"] = table["n_bad"] / table["n"]
    table["cum_pop_pct"] = table["n"].cumsum() / table["n"].sum()
    table["cum_bad_pct"] = table["n_bad"].cumsum() / table["n_bad"].sum()
    table["cum_good_pct"] = n_good.cumsum() / n_good.sum()
    table["ks"] = table["cum_bad_pct"] - table["cum_good_pct"]
    return table[
        ["n", "n_bad", "bad_rate", "min_score", "max_score", "cum_pop_pct", "cum_bad_pct", "cum_good_pct", "ks"]
    ]


def calibration_table(y: pd.Series, pd_: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Compare mean predicted PD with the observed default rate per risk band.

    Args:
        y: Binary target (1 = bad/default).
        pd_: Predicted probability of default.
        n_bins: Number of equal-size bands (band 1 = riskiest).

    Returns:
        DataFrame indexed by ``band`` with ``n``, ``mean_pd``, ``bad_rate``
        and ``ratio`` (``bad_rate / mean_pd``; 1 = perfectly calibrated).
    """
    rank = pd_.rank(method="first", ascending=False)
    band = pd.qcut(rank, n_bins, labels=range(1, n_bins + 1))
    table = (
        pd.DataFrame({"y": y, "pd": pd_, "band": band})
        .groupby("band", observed=True)
        .agg(n=("y", "size"), mean_pd=("pd", "mean"), bad_rate=("y", "mean"))
    )
    table["ratio"] = table["bad_rate"] / table["mean_pd"]
    return table


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index of ``actual`` against ``expected``.

    Bins are deciles of ``expected``. Rule of thumb: < 0.1 stable,
    0.1-0.25 monitor, > 0.25 significant shift.

    Args:
        expected: Reference distribution (e.g. train scores).
        actual: New distribution (e.g. test scores).
        n_bins: Number of quantile bins taken from ``expected``.
        eps: Floor on bin shares to avoid ``log(0)``.

    Returns:
        PSI value (>= 0).
    """
    inner = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1))[1:-1])
    k = len(inner) + 1
    e = np.bincount(np.searchsorted(inner, expected, side="right"), minlength=k) / len(expected)
    a = np.bincount(np.searchsorted(inner, actual, side="right"), minlength=k) / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
