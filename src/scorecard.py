"""Score scaling (PDO) and points-based scorecard from a WoE logistic regression.

Scaling (Siddiqi): ``score = offset + factor * ln(odds)`` with
``odds = P(good) / P(bad)``, ``factor = PDO / ln(2)`` and
``offset = base_score - factor * ln(base_odds)``. A higher score means lower
risk; every ``PDO`` points the good:bad odds double.

With ``ln(odds) = -(b0 + sum_i b_i * woe_i)`` the score splits into points
per attribute: ``points_ij = -factor * b_i * woe_ij + (offset - factor * b0) / n``.
"""

import numpy as np
import pandas as pd


def scaling_params(
    base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0
) -> tuple[float, float]:
    """Compute the scaling factor and offset of a credit score.

    Args:
        base_score: Score assigned at ``base_odds``.
        base_odds: Good:bad odds at ``base_score`` (50 means 50:1).
        pdo: Points to double the odds.

    Returns:
        ``(factor, offset)``.
    """
    factor = pdo / np.log(2)
    return float(factor), float(base_score - factor * np.log(base_odds))


def pd_to_score(pd_: pd.Series, factor: float, offset: float) -> pd.Series:
    """Convert probability of default into a credit score.

    Args:
        pd_: Probability of default in ``(0, 1)``.
        factor: Scaling factor from :func:`scaling_params`.
        offset: Scaling offset from :func:`scaling_params`.

    Returns:
        Credit score (higher = safer), indexed like ``pd_``.
    """
    return (offset + factor * np.log((1 - pd_) / pd_)).rename("score")


def score_to_pd(score: pd.Series, factor: float, offset: float) -> pd.Series:
    """Convert a credit score back into probability of default.

    Args:
        score: Credit score.
        factor: Scaling factor from :func:`scaling_params`.
        offset: Scaling offset from :func:`scaling_params`.

    Returns:
        Probability of default, indexed like ``score``.
    """
    return (1 / (1 + np.exp((score - offset) / factor))).rename("pd")


def scorecard_table(
    coefs: pd.Series,
    intercept: float,
    tables: dict[str, pd.DataFrame],
    factor: float,
    offset: float,
) -> pd.DataFrame:
    """Build the points table: one row per (feature, bin).

    The intercept and offset are spread evenly across features, so a
    customer's score is the sum of the points of their bins.

    Args:
        coefs: Logit coefficients indexed by feature (no intercept).
        intercept: Logit intercept.
        tables: Mapping of feature to its WoE table (:func:`src.woe.woe_table`).
        factor: Scaling factor from :func:`scaling_params`.
        offset: Scaling offset from :func:`scaling_params`.

    Returns:
        DataFrame with columns ``feature``, ``bin``, ``share``, ``bad_rate``,
        ``woe`` and ``points``, features in ``coefs`` order.
    """
    base = neutral_points(coefs, intercept, factor, offset)
    parts = [
        tables[f][["share", "bad_rate", "woe"]]
        .reset_index()
        .rename(columns={tables[f].index.name or "index": "bin"})
        .assign(feature=f, points=lambda t, b=b: base - factor * b * t["woe"])
        for f, b in coefs.items()
    ]
    card = pd.concat(parts, ignore_index=True)
    card["bin"] = card["bin"].astype(str)
    return card[["feature", "bin", "share", "bad_rate", "woe", "points"]]


def neutral_points(coefs: pd.Series, intercept: float, factor: float, offset: float) -> float:
    """Points given for a bin unseen in training (WoE = 0, average risk).

    Args:
        coefs: Logit coefficients indexed by feature (no intercept).
        intercept: Logit intercept.
        factor: Scaling factor from :func:`scaling_params`.
        offset: Scaling offset from :func:`scaling_params`.

    Returns:
        Points per feature at WoE 0.
    """
    return (offset - factor * intercept) / len(coefs)


def points_frame(binned: pd.DataFrame, card: pd.DataFrame, default_points: float) -> pd.DataFrame:
    """Points earned by each customer on each feature.

    Args:
        binned: Bin labels per feature (output of :func:`src.woe.bin_frame`).
        card: Output of :func:`scorecard_table`.
        default_points: Points for a bin absent from ``card``
            (see :func:`neutral_points`).

    Returns:
        DataFrame indexed like ``binned`` with one column per card feature.
    """
    return pd.DataFrame(
        {
            feature: binned[feature]
            .astype(str)
            .map(dict(zip(rows["bin"], rows["points"])))
            .astype(float)
            .fillna(default_points)
            for feature, rows in card.groupby("feature", sort=False)
        },
        index=binned.index,
    )


def apply_scorecard(binned: pd.DataFrame, card: pd.DataFrame, default_points: float) -> pd.Series:
    """Score customers by summing the points of their bins.

    Args:
        binned: Bin labels per feature (output of :func:`src.woe.bin_frame`).
        card: Output of :func:`scorecard_table`.
        default_points: Points for a bin absent from ``card``
            (see :func:`neutral_points`).

    Returns:
        Total score per customer, indexed like ``binned``.
    """
    return points_frame(binned, card, default_points).sum(axis=1).rename("score")


def points_direction(card: pd.DataFrame, min_share: float = 0.005) -> pd.DataFrame:
    """Check that points fall as bad rate rises, feature by feature.

    Args:
        card: Output of :func:`scorecard_table`.
        min_share: Bins below this share are ignored (they get neutral WoE,
            see :func:`src.woe.woe_table`).

    Returns:
        DataFrame indexed by feature with ``min_points``, ``max_points``,
        ``range`` (weight of the feature in the score), ``spearman``
        (rank correlation of points vs bad rate over bins with enough data) and
        ``consistent`` (``spearman < 0``), sorted by ``range`` descending.
    """
    rows = {
        f: {
            "min_points": g["points"].min(),
            "max_points": g["points"].max(),
            "range": g["points"].max() - g["points"].min(),
            "spearman": g["points"].corr(g["bad_rate"], method="spearman"),
        }
        for f, g in card[card["share"] >= min_share].groupby("feature", sort=False)
    }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out["consistent"] = out["spearman"] < 0
    return out.sort_values("range", ascending=False)


def cutoff_table(y: pd.Series, score: pd.Series, cutoffs: list[float]) -> pd.DataFrame:
    """Approval trade-off when approving customers with ``score >= cutoff``.

    Args:
        y: Binary target (1 = bad/default).
        score: Credit score (higher = safer).
        cutoffs: Candidate minimum scores.

    Returns:
        DataFrame indexed by ``cutoff`` with ``approval_rate``,
        ``bad_rate_approved`` and ``bads_rejected_pct`` (share of all bads
        turned away).
    """
    rows = []
    for c in cutoffs:
        approved = score >= c
        rows.append(
            {
                "cutoff": c,
                "approval_rate": approved.mean(),
                "bad_rate_approved": y[approved].mean() if approved.any() else np.nan,
                "bads_rejected_pct": y[~approved].sum() / y.sum(),
            }
        )
    return pd.DataFrame(rows).set_index("cutoff")


DECISIONS: list[str] = ["Reject", "Review", "Approve"]


def assign_decision(score: pd.Series, reject_below: float, approve_from: float) -> pd.Series:
    """Map credit scores to a three-way decision.

    ``score < reject_below`` -> Reject, ``score >= approve_from`` -> Approve,
    anything in between -> Review (manual underwriting). Equal thresholds
    give a plain approve/reject rule.

    Args:
        score: Credit score (higher = safer).
        reject_below: Scores below this are rejected.
        approve_from: Scores at or above this are approved.

    Returns:
        Ordered categorical Series (``Reject < Review < Approve``), same
        index as ``score``.

    Raises:
        ValueError: If ``reject_below > approve_from``.
    """
    if reject_below > approve_from:
        raise ValueError("reject_below must not exceed approve_from")
    labels = np.select([score < reject_below, score >= approve_from], ["Reject", "Approve"], "Review")
    return pd.Series(pd.Categorical(labels, categories=DECISIONS, ordered=True), index=score.index, name="decision")


def decision_summary(y: pd.Series, decision: pd.Series) -> pd.DataFrame:
    """Volume and risk of each decision band.

    Args:
        y: Binary target (1 = bad/default).
        decision: Decision per applicant, e.g. from :func:`assign_decision`.

    Returns:
        DataFrame indexed by decision with ``n``, ``share`` (of applicants),
        ``n_bad``, ``bad_rate`` and ``bads_share`` (of all bads).
    """
    table = (
        pd.DataFrame({"y": y, "decision": decision})
        .groupby("decision", observed=False)
        .agg(n=("y", "size"), n_bad=("y", "sum"))
    )
    table["share"] = table["n"] / table["n"].sum()
    table["bad_rate"] = table["n_bad"] / table["n"].where(table["n"] > 0)
    table["bads_share"] = table["n_bad"] / table["n_bad"].sum()
    return table[["n", "share", "n_bad", "bad_rate", "bads_share"]]
