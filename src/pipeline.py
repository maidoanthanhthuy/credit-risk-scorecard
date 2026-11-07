"""End-to-end scoring of new applicants with the champion scorecard.

Input is one or more applicants in the *cleaned* format of ``data/train.pkl``
(output of :func:`src.cleaning.clean_application`: ages and durations in
years, categories as labels). Only the scorecard features are used; missing
features fall into the ``Missing`` bin, unknown categories into ``Rare``.

Flow: ``prepare_input`` -> ``bin_frame`` -> ``points_frame`` -> score, PD,
decision (:func:`src.scorecard.assign_decision`) and per-feature
contributions in points, centred on the training average (the exact SHAP
values of the scorecard, see :mod:`src.explain`).

All functions are pure except :func:`load_bundle`, which reads artifacts.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.explain import reason_codes
from src.scorecard import assign_decision, points_frame, score_to_pd
from src.woe import BinRule, bin_frame

# Decision thresholds chosen in 13_cutoff_strategy
REJECT_BELOW: float = 540.0
APPROVE_FROM: float = 560.0

# Valid [min, max] of numeric inputs in the cleaned format (durations in years).
# Out-of-range values would otherwise fall silently into an edge bin.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "EXT_SOURCE_1": (0.0, 1.0),
    "EXT_SOURCE_2": (0.0, 1.0),
    "EXT_SOURCE_3": (0.0, 1.0),
    "FLOORSMAX_AVG": (0.0, 1.0),
    "YEARS_EMPLOYED": (0.0, 100.0),
    "YEARS_REGISTRATION": (0.0, 100.0),
    "YEARS_ID_PUBLISH": (0.0, 100.0),
    "YEARS_LAST_PHONE_CHANGE": (0.0, 100.0),
    "OWN_CAR_AGE": (0.0, 100.0),
    "AMT_GOODS_PRICE": (0.0, np.inf),
}


@dataclass(frozen=True)
class ScorecardBundle:
    """Everything needed to score an applicant.

    Attributes:
        features: Scorecard features, in card order.
        spec: Binning rule per feature.
        card: Points table (:func:`src.scorecard.scorecard_table`).
        factor: Scaling factor.
        offset: Scaling offset.
        neutral: Points for a bin absent from the card.
        reject_below: Scores below this are rejected.
        approve_from: Scores at or above this are approved.
    """

    features: list[str]
    spec: dict[str, BinRule]
    card: pd.DataFrame
    factor: float
    offset: float
    neutral: float
    reject_below: float = REJECT_BELOW
    approve_from: float = APPROVE_FROM


@dataclass(frozen=True)
class ApplicantResult:
    """Score, decision and explanation of one applicant.

    Attributes:
        score: Credit score (sum of ``detail["points"]``).
        pd: Probability of default implied by the score.
        decision: ``"Reject"``, ``"Review"`` or ``"Approve"``.
        base_score: Score of the average training applicant; ``score =
            base_score + detail["contribution"].sum()``.
        detail: One row per feature with ``value``, ``bin``, ``points`` and
            ``contribution`` (points minus the training average; negative =
            pushes the score down), in card order.
        adverse: Up to ``k`` strongest negative contributions (reason codes).
        favourable: Up to ``k`` strongest positive contributions.
    """

    score: float
    pd: float
    decision: str
    base_score: float
    detail: pd.DataFrame
    adverse: pd.Series
    favourable: pd.Series


def load_bundle(output_dir: Path) -> ScorecardBundle:
    """Load the scorecard artifacts written by notebooks 05-08.

    Args:
        output_dir: Folder with ``model_features.csv``, ``woe_spec_final.pkl``,
            ``scorecard.csv`` and ``score_scaling.csv``.

    Returns:
        A :class:`ScorecardBundle` with the default decision thresholds.
    """
    features = pd.read_csv(output_dir / "model_features.csv")["feature"].tolist()
    with open(output_dir / "woe_spec_final.pkl", "rb") as f:
        spec = pickle.load(f)
    card = pd.read_csv(output_dir / "scorecard.csv", dtype={"bin": str})
    scaling = pd.read_csv(output_dir / "score_scaling.csv", index_col=0).iloc[:, 0]
    return ScorecardBundle(
        features=features,
        spec={f: spec[f] for f in features},
        card=card,
        factor=float(scaling["factor"]),
        offset=float(scaling["offset"]),
        neutral=float(scaling["neutral_points"]),
    )


def missing_features(applicants: pd.DataFrame, features: list[str]) -> list[str]:
    """Scorecard features absent from the input columns.

    Args:
        applicants: Input applicants.
        features: Scorecard features.

    Returns:
        Feature names not in ``applicants.columns``, in ``features`` order.
    """
    return [f for f in features if f not in applicants.columns]


def _as_label(value: object) -> object:
    """Turn integer-valued floats into int so ``1.0`` matches the level ``"1"``."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def range_errors(X: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> list[str]:
    """List numeric values outside their valid range.

    Missing values are allowed (they are scored as ``Missing``); infinite
    values are always errors.

    Args:
        X: Numeric features (float columns).
        ranges: Valid ``(min, max)`` per feature, bounds included; features
            not in ``X`` are skipped.

    Returns:
        One message per offending value, e.g. ``"EXT_SOURCE_3 = 5 (row 0) is
        outside [0, 1]"``; empty if all values are valid.
    """
    errors = []
    for f, (lo, hi) in ranges.items():
        if f not in X.columns:
            continue
        col = X[f]
        bad = col.notna() & (~np.isfinite(col) | (col < lo) | (col > hi))
        errors += [f"{f} = {v:g} (row {i}) is outside [{lo:g}, {hi:g}]" for i, v in col[bad].items()]
    return errors


def prepare_input(
    applicants: pd.DataFrame,
    spec: dict[str, BinRule],
    ranges: dict[str, tuple[float, float]] = VALID_RANGES,
) -> pd.DataFrame:
    """Select, type and validate the scorecard features.

    Absent features are added as missing. Numeric features are converted to
    float and checked against ``ranges``; categorical features keep their
    labels, with integer-valued floats turned into int (flags such as
    ``FLAG_DOCUMENT_3`` are stored as ``"1"``).

    Args:
        applicants: Applicants in the cleaned format.
        spec: Binning rule per scorecard feature.
        ranges: Valid ``(min, max)`` per numeric feature (see :data:`VALID_RANGES`).

    Returns:
        DataFrame with exactly the ``spec`` columns, indexed like ``applicants``.

    Raises:
        ValueError: If a numeric feature holds a non-numeric value or a value
            outside its valid range (all violations are listed).
    """
    out = {}
    for f, rule in spec.items():
        col = applicants[f] if f in applicants.columns else pd.Series(None, index=applicants.index, dtype=object)
        if rule["type"] == "numeric":
            try:
                out[f] = pd.to_numeric(col).astype(float)
            except (ValueError, TypeError) as err:
                raise ValueError(f"{f} must be numeric") from err
        else:
            # Explicit object dtype: Series.map would re-infer [1, NaN] as float and give back 1.0
            labels = [None if pd.isna(v) else _as_label(v) for v in col.astype(object)]
            out[f] = pd.Series(labels, index=applicants.index, dtype=object)
    X = pd.DataFrame(out, index=applicants.index)
    numeric = [f for f, rule in spec.items() if rule["type"] == "numeric"]
    errors = range_errors(X[numeric], ranges)
    if errors:
        raise ValueError("invalid input: " + "; ".join(errors))
    return X


def reference_points(card: pd.DataFrame) -> pd.Series:
    """Average points per feature over the training population.

    Uses the training share of each bin stored in the card, so no training
    data is needed.

    Args:
        card: Points table with ``feature``, ``share`` and ``points``.

    Returns:
        Mean points per feature, in card order.
    """
    weighted = card.assign(wp=card["share"] * card["points"]).groupby("feature", sort=False)
    return weighted["wp"].sum() / weighted["share"].sum()


def score_frame(applicants: pd.DataFrame, bundle: ScorecardBundle) -> pd.DataFrame:
    """Score many applicants at once.

    Args:
        applicants: Applicants in the cleaned format.
        bundle: Scorecard artifacts.

    Returns:
        DataFrame indexed like ``applicants`` with ``score``, ``pd`` and
        ``decision``.
    """
    X = prepare_input(applicants, bundle.spec)
    score = points_frame(bin_frame(X, bundle.spec), bundle.card, bundle.neutral).sum(axis=1)
    return pd.DataFrame(
        {
            "score": score,
            "pd": score_to_pd(score, bundle.factor, bundle.offset),
            "decision": assign_decision(score, bundle.reject_below, bundle.approve_from),
        }
    )


def score_applicant(applicant: dict | pd.Series, bundle: ScorecardBundle, k: int = 4) -> ApplicantResult:
    """Score and explain one applicant.

    Args:
        applicant: Feature values in the cleaned format (extra keys ignored).
        bundle: Scorecard artifacts.
        k: Number of adverse / favourable reasons to return.

    Returns:
        An :class:`ApplicantResult`.
    """
    row = pd.DataFrame([dict(applicant)])
    X = prepare_input(row, bundle.spec)
    binned = bin_frame(X, bundle.spec)
    points = points_frame(binned, bundle.card, bundle.neutral).iloc[0]
    ref = reference_points(bundle.card)
    contribution = points - ref
    score = float(points.sum())
    detail = pd.DataFrame(
        {
            "value": X.iloc[0].astype(object),
            "bin": binned.iloc[0].astype(str),
            "points": points,
            "contribution": contribution,
        }
    )
    return ApplicantResult(
        score=score,
        pd=float(score_to_pd(pd.Series([score]), bundle.factor, bundle.offset).iloc[0]),
        decision=str(assign_decision(pd.Series([score]), bundle.reject_below, bundle.approve_from).iloc[0]),
        base_score=float(ref.sum()),
        detail=detail,
        adverse=reason_codes(contribution, k=k, adverse=True),
        favourable=reason_codes(contribution, k=k, adverse=False),
    )
