"""Validation metrics for a credit scoring model.

Discrimination (AUC, Gini, KS, PR-AUC, bootstrap CIs), rank ordering
(gains / lift), calibration and stability (PSI, CSI).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


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


def pr_auc(y: pd.Series, score: pd.Series) -> float:
    """Area under the precision-recall curve (average precision).

    Focuses on the bad class, so it is informative under class imbalance.
    A random model scores the bad rate, not 0.5.

    Args:
        y: Binary target (1 = bad/default).
        score: Predicted risk, higher = riskier.

    Returns:
        PR-AUC in ``[0, 1]``.
    """
    return float(average_precision_score(y, score))


def bootstrap_gini(y: pd.Series, scores: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Bootstrap distribution of the Gini of one or more models.

    All models are scored on the same resample in each round (paired
    bootstrap), so differences between columns give valid CIs for the gap
    between models.

    Args:
        y: Binary target (1 = bad/default).
        scores: One column of predicted risk per model, aligned with ``y``.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        DataFrame with ``n_boot`` rows and one Gini column per model.
    """
    rng = np.random.default_rng(seed)
    y_arr, s_arr = y.to_numpy(), scores.to_numpy()
    n = len(y_arr)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rows.append([2 * roc_auc_score(y_arr[idx], s_arr[idx, j]) - 1 for j in range(s_arr.shape[1])])
    return pd.DataFrame(rows, columns=scores.columns)


def percentile_ci(samples: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Percentile confidence interval of each column of bootstrap samples.

    Args:
        samples: Bootstrap draws, one column per statistic.
        level: Confidence level.

    Returns:
        DataFrame indexed by column name with ``lower`` and ``upper``.
    """
    alpha = (1 - level) / 2
    return samples.quantile([alpha, 1 - alpha]).T.set_axis(["lower", "upper"], axis=1)


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
        ``cum_good_pct``, ``ks`` (``cum_bad_pct - cum_good_pct``), ``lift``
        (band bad rate / overall bad rate) and ``cum_lift``
        (``cum_bad_pct / cum_pop_pct``).
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
    table["lift"] = table["bad_rate"] / (table["n_bad"].sum() / table["n"].sum())
    table["cum_lift"] = table["cum_bad_pct"] / table["cum_pop_pct"]
    return table[
        [
            "n", "n_bad", "bad_rate", "min_score", "max_score",
            "cum_pop_pct", "cum_bad_pct", "cum_good_pct", "ks", "lift", "cum_lift",
        ]
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


def csi(expected: pd.Series, actual: pd.Series, eps: float = 1e-6) -> float:
    """Characteristic Stability Index of a binned (categorical) feature.

    Same formula as :func:`psi`, but on the shares of existing bins
    (e.g. WoE bin labels) instead of quantiles. Levels missing from one side
    get share ``eps``.

    Args:
        expected: Reference bin labels (e.g. train).
        actual: New bin labels (e.g. test).
        eps: Floor on bin shares to avoid ``log(0)``.

    Returns:
        CSI value (>= 0); same thresholds as PSI.
    """
    e = expected.astype(object).value_counts(normalize=True, dropna=False)
    a = actual.astype(object).value_counts(normalize=True, dropna=False)
    e, a = e.align(a, fill_value=0.0)
    e, a = e.clip(lower=eps), a.clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())
