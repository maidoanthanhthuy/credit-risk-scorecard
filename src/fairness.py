"""Group fairness diagnostics for credit decisions.

Protected attributes are excluded from the models ("fairness through
unawareness"), so this module checks the outcome: approval rates,
calibration and ranking power per group, and whether the model features
can reconstruct a protected attribute (proxy check).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def age_bands(age: pd.Series, edges: list[float] | None = None) -> pd.Series:
    """Bucket ages into labelled bands.

    Args:
        age: Age in years.
        edges: Band edges; defaults to ``[-inf, 25, 35, 45, 55, inf]``.

    Returns:
        Ordered categorical with labels such as ``"<25"``, ``"25-34"``, ``"55+"``.
    """
    edges = edges or [-np.inf, 25, 35, 45, 55, np.inf]
    labels = [
        f"<{edges[1]:g}" if i == 0
        else f"{edges[i]:g}+" if i == len(edges) - 2
        else f"{edges[i]:g}-{edges[i + 1] - 1:g}"
        for i in range(len(edges) - 1)
    ]
    return pd.cut(age, bins=edges, labels=labels, right=False).rename("age_band")


def group_metrics(
    y: pd.Series, pd_: pd.Series, score: pd.Series, group: pd.Series, cutoff: float
) -> pd.DataFrame:
    """Decision and accuracy metrics per group at a score cut-off.

    Args:
        y: Binary target (1 = bad/default).
        pd_: Predicted probability of default.
        score: Credit score (approve if ``score >= cutoff``).
        group: Group label per applicant (missing labels are dropped).
        cutoff: Minimum score to approve.

    Returns:
        DataFrame indexed by group with ``n``, ``default_rate``, ``mean_pd``,
        ``calib_ratio`` (default_rate / mean_pd; < 1 = risk over-estimated),
        ``mean_score``, ``approval_rate``, ``good_approval_rate`` (approval
        among non-defaulters) and ``auc`` (within-group ranking power).
    """
    df = pd.DataFrame({"y": y, "pd": pd_, "score": score, "g": group}).dropna(subset=["g"])
    df["approved"] = df["score"] >= cutoff
    rows = {}
    for g, d in df.groupby("g", observed=True):
        rows[g] = {
            "n": len(d),
            "default_rate": d["y"].mean(),
            "mean_pd": d["pd"].mean(),
            "calib_ratio": d["y"].mean() / d["pd"].mean(),
            "mean_score": d["score"].mean(),
            "approval_rate": d["approved"].mean(),
            "good_approval_rate": d.loc[d["y"] == 0, "approved"].mean(),
            "auc": roc_auc_score(d["y"], -d["score"]) if d["y"].nunique() == 2 else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def disparate_impact(rates: pd.Series, reference: object | None = None) -> pd.Series:
    """Ratio of each group's rate to a reference group's rate.

    The "four-fifths rule" flags ratios below 0.8.

    Args:
        rates: Rate per group (e.g. ``approval_rate`` of :func:`group_metrics`).
        reference: Reference group; defaults to the group with the highest rate.

    Returns:
        Ratio per group (reference = 1).
    """
    ref = rates.idxmax() if reference is None else reference
    return rates / rates[ref]


def proxy_auc(
    X_train: pd.DataFrame,
    g_train: pd.Series,
    X_test: pd.DataFrame,
    g_test: pd.Series,
    n_rounds: int = 300,
) -> tuple[float, lgb.LGBMClassifier]:
    """How well model features predict a binary protected attribute.

    An AUC near 0.5 means the features carry little information about the
    attribute; well above 0.5 means proxies exist.

    Args:
        X_train: Features (prepared for LightGBM) of the training rows.
        g_train: Binary attribute (0/1) for the training rows.
        X_test: Features of the evaluation rows.
        g_test: Binary attribute for the evaluation rows.
        n_rounds: Boosting rounds.

    Returns:
        ``(auc, model)`` with the AUC on the evaluation rows.
    """
    model = lgb.LGBMClassifier(
        n_estimators=n_rounds, learning_rate=0.05, num_leaves=31, min_child_samples=100,
        random_state=42, verbose=-1,
    ).fit(X_train, g_train)
    return float(roc_auc_score(g_test, model.predict_proba(X_test)[:, 1])), model
