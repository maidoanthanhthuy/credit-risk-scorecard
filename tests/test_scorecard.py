"""Tests for src.scorecard."""

import numpy as np
import pandas as pd
import pytest

from src.scorecard import (
    apply_scorecard,
    assign_decision,
    cutoff_table,
    decision_summary,
    neutral_points,
    pd_to_score,
    points_direction,
    scaling_params,
    score_to_pd,
    scorecard_table,
)


def _woe_table(bins: list[str], bad_rate: list[float], woe: list[float]) -> pd.DataFrame:
    """Minimal WoE table with the columns used by the scorecard."""
    t = pd.DataFrame(
        {"share": 1 / len(bins), "bad_rate": bad_rate, "woe": woe},
        index=pd.Index(bins, name="bin"),
    )
    return t


TABLES = {
    "a": _woe_table(["low", "high"], [0.15, 0.05], [-0.7, 0.5]),
    "b": _woe_table(["x", "y", "Missing"], [0.12, 0.06, 0.09], [-0.4, 0.3, -0.1]),
}
COEFS = pd.Series({"a": -0.8, "b": -0.5})
INTERCEPT = -2.4


def test_scaling_matches_base_and_pdo() -> None:
    factor, offset = scaling_params(base_score=600, base_odds=50, pdo=20)
    pd_at_base = pd.Series([1 / 51])  # odds 50:1
    pd_double = pd.Series([1 / 101])  # odds 100:1
    assert pd_to_score(pd_at_base, factor, offset).iloc[0] == pytest.approx(600)
    assert pd_to_score(pd_double, factor, offset).iloc[0] == pytest.approx(620)


def test_score_to_pd_inverts_pd_to_score() -> None:
    factor, offset = scaling_params()
    p = pd.Series([0.01, 0.08, 0.5])
    assert np.allclose(score_to_pd(pd_to_score(p, factor, offset), factor, offset), p)


def test_scorecard_points_sum_to_model_score() -> None:
    factor, offset = scaling_params()
    card = scorecard_table(COEFS, INTERCEPT, TABLES, factor, offset)
    binned = pd.DataFrame({"a": ["low", "high"], "b": ["Missing", "y"]})
    woe = pd.DataFrame({"a": [-0.7, 0.5], "b": [-0.1, 0.3]})
    pd_ = 1 / (1 + np.exp(-(INTERCEPT + woe @ COEFS)))
    score = apply_scorecard(binned, card, neutral_points(COEFS, INTERCEPT, factor, offset))
    assert np.allclose(score, pd_to_score(pd_, factor, offset))


def test_apply_scorecard_gives_neutral_points_to_unseen_bin() -> None:
    factor, offset = scaling_params()
    card = scorecard_table(COEFS, INTERCEPT, TABLES, factor, offset)
    neutral = neutral_points(COEFS, INTERCEPT, factor, offset)
    score = apply_scorecard(pd.DataFrame({"a": ["low"], "b": ["unseen"]}), card, neutral)
    points_a_low = card.query("feature == 'a' and bin == 'low'")["points"].iloc[0]
    assert score.iloc[0] == pytest.approx(points_a_low + neutral)


def test_points_direction_flags_wrong_sign() -> None:
    factor, offset = scaling_params()
    card = scorecard_table(pd.Series({"a": -0.8, "b": 0.5}), INTERCEPT, TABLES, factor, offset)
    check = points_direction(card)
    assert bool(check.loc["a", "consistent"]) and not bool(check.loc["b", "consistent"])


def test_cutoff_table_extremes() -> None:
    y = pd.Series([1, 0, 1, 0])
    score = pd.Series([500, 550, 600, 650])
    table = cutoff_table(y, score, [0, 575, 1000])
    assert table.loc[0, "approval_rate"] == 1 and table.loc[0, "bads_rejected_pct"] == 0
    assert table.loc[575, "approval_rate"] == 0.5 and table.loc[575, "bad_rate_approved"] == 0.5
    assert table.loc[1000, "bads_rejected_pct"] == 1 and np.isnan(table.loc[1000, "bad_rate_approved"])


def test_points_direction_ignores_tiny_bins() -> None:
    tables = {"a": _woe_table(["low", "high", "tiny"], [0.15, 0.05, 0.0], [-0.7, 0.5, 0.0])}
    tables["a"]["share"] = [0.5, 0.499, 0.001]
    factor, offset = scaling_params()
    card = scorecard_table(pd.Series({"a": -0.8}), INTERCEPT, tables, factor, offset)
    assert points_direction(card, min_share=0.005).loc["a", "spearman"] == pytest.approx(-1.0)


def test_assign_decision_bands_and_boundaries() -> None:
    score = pd.Series([500.0, 539.9, 540.0, 559.9, 560.0, 600.0], index=list("abcdef"))
    decision = assign_decision(score, reject_below=540, approve_from=560)
    assert decision.tolist() == ["Reject", "Reject", "Review", "Review", "Approve", "Approve"]
    assert list(decision.index) == list("abcdef")
    assert decision.cat.ordered
    assert "Review" not in assign_decision(score, 550, 550).tolist()
    with pytest.raises(ValueError):
        assign_decision(score, 560, 540)


def test_decision_summary_totals() -> None:
    y = pd.Series([1, 0, 1, 0, 0, 0])
    decision = assign_decision(pd.Series([500.0, 520, 545, 555, 570, 600]), 540, 560)
    table = decision_summary(y, decision)
    assert list(table.index) == ["Reject", "Review", "Approve"]
    assert table["n"].tolist() == [2, 2, 2]
    assert table["share"].sum() == pytest.approx(1.0)
    assert table["bads_share"].sum() == pytest.approx(1.0)
    assert table.loc["Reject", "bad_rate"] == pytest.approx(0.5)
    assert table.loc["Approve", "bad_rate"] == pytest.approx(0.0)
