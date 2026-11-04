"""Explanations in score points: TreeSHAP for LightGBM, centred points for the scorecard.

Both models are put on the scale of the credit score (see :mod:`src.scorecard`):
``score = offset - factor * logit(PD)``. A SHAP value ``phi`` on the log-odds of
default therefore moves the score by ``-factor * phi`` points, so positive
points mean "safer" for both models.

For the scorecard (a logistic model on WoE), the SHAP value of a feature is
its points minus the average points of that feature over the reference
population, i.e. the centred points.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd


def lgbm_contributions(model: lgb.LGBMClassifier, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP values of a LightGBM classifier on the log-odds scale.

    Uses LightGBM's native ``pred_contrib`` (same algorithm as
    ``shap.TreeExplainer``) so categorical splits are handled exactly.

    Args:
        model: Fitted LightGBM classifier.
        X: Features prepared as for training.

    Returns:
        ``(contributions, base_value)``: one column per feature, indexed like
        ``X``; row sums plus ``base_value`` equal the raw log-odds of default.
    """
    raw = model.predict(X, pred_contrib=True)
    contrib = pd.DataFrame(raw[:, :-1], index=X.index, columns=X.columns)
    return contrib, float(raw[0, -1])


def to_points(contrib: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Convert log-odds-of-default contributions into score points.

    Args:
        contrib: Contributions on the log-odds of default.
        factor: Scaling factor from :func:`src.scorecard.scaling_params`.

    Returns:
        Points per feature (positive = raises the score = safer).
    """
    return -factor * contrib


def base_score(base_value: float, factor: float, offset: float) -> float:
    """Score of the average applicant (SHAP base value on the score scale).

    Args:
        base_value: SHAP base value on the log-odds of default.
        factor: Scaling factor from :func:`src.scorecard.scaling_params`.
        offset: Scaling offset from :func:`src.scorecard.scaling_params`.

    Returns:
        ``offset - factor * base_value``.
    """
    return offset - factor * base_value


def center_points(points: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Centre scorecard points on the reference population's average.

    Args:
        points: Points per feature (output of :func:`src.scorecard.points_frame`).
        reference: Points per feature for the reference population (train).

    Returns:
        ``points - reference.mean()``: the SHAP values of the scorecard.
    """
    return points - reference.mean()


def numeric_view(X: pd.DataFrame) -> pd.DataFrame:
    """Replace categorical columns by their integer codes (missing stays NaN).

    SHAP plots colour points by feature value and need numbers.

    Args:
        X: Features, possibly with ``category`` columns.

    Returns:
        All-float copy of ``X``.
    """
    out = X.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].cat.codes.replace(-1, np.nan)
    return out.astype(float)


def mean_abs_importance(points: pd.DataFrame) -> pd.Series:
    """Global importance: mean absolute contribution per feature.

    Args:
        points: Contributions per feature (any scale).

    Returns:
        Mean absolute contribution, sorted descending.
    """
    return points.abs().mean().sort_values(ascending=False)


def reason_codes(points: pd.Series, k: int = 4, adverse: bool = True) -> pd.Series:
    """Top features pushing one applicant's score down (or up).

    Args:
        points: Contributions of one applicant, in score points.
        k: Maximum number of reasons.
        adverse: ``True`` for reasons lowering the score (rejection),
            ``False`` for reasons raising it (approval).

    Returns:
        Up to ``k`` contributions with the requested sign, strongest first.
    """
    if adverse:
        return points[points < 0].sort_values().head(k)
    return points[points > 0].sort_values(ascending=False).head(k)


def pick_case(score: pd.Series, mask: pd.Series, target: float) -> object:
    """Pick the applicant within ``mask`` whose score is closest to ``target``.

    Args:
        score: Scores indexed by applicant.
        mask: Boolean filter aligned with ``score``.
        target: Score to aim for (e.g. a percentile of the filtered group).

    Returns:
        Index label of the chosen applicant.

    Raises:
        ValueError: If ``mask`` selects no applicant.
    """
    candidates = score[mask]
    if candidates.empty:
        raise ValueError("No applicant matches the mask")
    return (candidates - target).abs().idxmin()
