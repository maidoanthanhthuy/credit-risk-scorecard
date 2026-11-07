"""Tests for src.pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    ScorecardBundle,
    load_bundle,
    missing_features,
    prepare_input,
    range_errors,
    reference_points,
    score_applicant,
    score_frame,
)
from src.scorecard import pd_to_score
from src.woe import apply_bins

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bundle() -> ScorecardBundle:
    """Two-feature scorecard: reference points x = 20, c = 25 (base 45)."""
    spec = {
        "x": {"type": "numeric", "edges": [-np.inf, 0.0, np.inf]},
        "c": {"type": "categorical", "levels": ["a", "b"]},
    }
    low, high = list(apply_bins(pd.Series([0.0]), spec["x"]).cat.categories)[:2]
    card = pd.DataFrame(
        [("x", low, 0.5, 10.0), ("x", high, 0.5, 30.0), ("c", "a", 0.75, 20.0), ("c", "b", 0.25, 40.0)],
        columns=["feature", "bin", "share", "points"],
    )
    return ScorecardBundle(
        features=["x", "c"], spec=spec, card=card, factor=20 / np.log(2), offset=500.0,
        neutral=25.0, reject_below=40.0, approve_from=60.0,
    )


def test_reference_points_is_share_weighted_mean() -> None:
    assert reference_points(_bundle().card).to_dict() == pytest.approx({"x": 20.0, "c": 25.0})


def test_score_applicant_points_and_explanation() -> None:
    result = score_applicant({"x": 1.0, "c": "b", "extra": 99}, _bundle())
    assert result.score == pytest.approx(70.0)
    assert result.decision == "Approve"
    assert result.base_score == pytest.approx(45.0)
    assert result.detail["contribution"].to_dict() == pytest.approx({"x": 10.0, "c": 15.0})
    assert result.base_score + result.detail["contribution"].sum() == pytest.approx(result.score)
    assert result.adverse.empty
    assert list(result.favourable.index) == ["c", "x"]
    assert 0 < result.pd < 1


def test_missing_and_unknown_values_get_neutral_points() -> None:
    bundle = _bundle()
    result = score_applicant({"x": None, "c": "zzz"}, bundle)
    assert result.detail["bin"].tolist() == ["Missing", "Rare"]
    assert result.score == pytest.approx(50.0)
    assert result.decision == "Review"
    assert missing_features(pd.DataFrame({"x": [1.0]}), bundle.features) == ["c"]
    assert score_applicant({}, bundle).score == pytest.approx(50.0)


def test_prepare_input_types() -> None:
    spec = {"flag": {"type": "categorical", "levels": ["0", "1"]}, "x": {"type": "numeric", "edges": [-np.inf, np.inf]}}
    X = prepare_input(pd.DataFrame({"flag": [1.0, np.nan], "x": ["2.5", None]}), spec)
    assert apply_bins(X["flag"], spec["flag"]).tolist() == ["1", "Missing"]
    assert X["x"].dtype == float
    with pytest.raises(ValueError, match="x must be numeric"):
        prepare_input(pd.DataFrame({"x": ["abc"]}), spec)


def test_score_frame_matches_score_applicant() -> None:
    bundle = _bundle()
    rows = pd.DataFrame({"x": [-1.0, 1.0, 1.0], "c": ["a", "a", "b"]}, index=[10, 11, 12])
    out = score_frame(rows, bundle)
    assert list(out.index) == [10, 11, 12]
    assert out["score"].tolist() == pytest.approx([30.0, 50.0, 70.0])
    assert out["decision"].astype(str).tolist() == ["Reject", "Review", "Approve"]
    assert out.loc[12, "score"] == pytest.approx(score_applicant(rows.loc[12], bundle).score)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "test.pkl").exists() or not (PROJECT_ROOT / "data" / "pd_test_all_models.pkl").exists(),
    reason="local data not available",
)
def test_pipeline_reproduces_scorecard_on_test_set() -> None:
    bundle = load_bundle(PROJECT_ROOT / "outputs")
    test = pd.read_pickle(PROJECT_ROOT / "data" / "test.pkl")
    expected = pd_to_score(pd.read_pickle(PROJECT_ROOT / "data" / "pd_test_all_models.pkl")["Scorecard"], bundle.factor, bundle.offset)
    scored = score_frame(test, bundle)
    assert (scored["score"] - expected.loc[scored.index]).abs().max() < 1e-6
    one = score_applicant(test.iloc[0], bundle)
    assert one.score == pytest.approx(scored["score"].iloc[0])
    assert one.base_score + one.detail["contribution"].sum() == pytest.approx(one.score)


def test_range_errors_lists_each_violation() -> None:
    X = pd.DataFrame({"p": [0.5, 5.0, np.nan], "y": [-10.0, 2.0, np.inf], "other": [1e9, 0.0, 0.0]})
    errors = range_errors(X, {"p": (0.0, 1.0), "y": (0.0, 100.0), "absent": (0.0, 1.0)})
    assert errors == [
        "p = 5 (row 1) is outside [0, 1]",
        "y = -10 (row 0) is outside [0, 100]",
        "y = inf (row 2) is outside [0, 100]",
    ]
    assert range_errors(X[["other"]], {"p": (0.0, 1.0)}) == []


def test_prepare_input_rejects_out_of_range() -> None:
    spec = {"x": {"type": "numeric", "edges": [-np.inf, np.inf]}, "c": {"type": "categorical", "levels": ["a"]}}
    with pytest.raises(ValueError, match=r"x = 2 \(row 0\) is outside \[0, 1\]"):
        prepare_input(pd.DataFrame({"x": [2.0], "c": ["a"]}), spec, ranges={"x": (0.0, 1.0)})
    assert prepare_input(pd.DataFrame({"x": [None]}), spec, ranges={"x": (0.0, 1.0)})["x"].isna().all()
