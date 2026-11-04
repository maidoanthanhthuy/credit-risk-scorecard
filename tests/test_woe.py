"""Tests for src.woe."""

import numpy as np
import pandas as pd
import pytest

from src.woe import (
    MISSING,
    RARE,
    apply_bins,
    categorical_levels,
    coarse_spec,
    interval_labels,
    is_monotonic,
    iv_strength,
    iv_summary,
    monotonic_edges,
    numeric_bin_edges,
    woe_frame,
    woe_table,
)

INF = np.inf


def _target(bad_rates: list[float], n: int = 100) -> pd.Series:
    """Build a target with ``n`` rows per group and the given bad rates."""
    parts = [np.r_[np.ones(round(r * n)), np.zeros(n - round(r * n))] for r in bad_rates]
    return pd.Series(np.concatenate(parts))


# --- numeric_bin_edges -------------------------------------------------------


def test_numeric_bin_edges_open_ended() -> None:
    edges = numeric_bin_edges(pd.Series(range(100)), n_bins=4)
    assert edges == [-INF, 24.75, 49.5, 74.25, INF]


def test_numeric_bin_edges_mass_at_zero_keeps_zero_split() -> None:
    edges = numeric_bin_edges(pd.Series([0] * 95 + [1, 2, 3, 4, 5]), n_bins=10)
    assert edges == [-INF, 0.0, INF]


# --- interval_labels ---------------------------------------------------------


def test_interval_labels_rounded() -> None:
    assert interval_labels([-INF, 0.12345678, INF]) == ["(-inf, 0.1235]", "(0.1235, inf]"]


def test_interval_labels_fallback_when_rounding_collides() -> None:
    labels = interval_labels([-INF, 0.100001, 0.100002, 0.100003, INF])
    assert len(set(labels)) == 4


# --- apply_bins --------------------------------------------------------------


def test_apply_bins_high_precision_floats_keep_all_rows() -> None:
    # Regression: matching on pd.cut's rounded labels once dropped rows silently
    s = pd.Series(np.random.default_rng(0).random(1000))
    bins = apply_bins(s, {"type": "numeric", "edges": numeric_bin_edges(s, 10)})
    assert bins.notna().all()
    assert bins.nunique() == 10


def test_apply_bins_numeric_missing_bin() -> None:
    bins = apply_bins(pd.Series([1.0, np.nan, 3.0]), {"type": "numeric", "edges": [-INF, 2.0, INF]})
    assert bins.tolist() == ["(-inf, 2]", MISSING, "(2, inf]"]


def test_apply_bins_categorical_rare_unseen_and_reserved_clash() -> None:
    rule = {"type": "categorical", "levels": ["A", "Other"]}
    bins = apply_bins(pd.Series(["A", "Other", "Z", None]), rule)
    assert bins.tolist() == ["A", "Other", RARE, MISSING]


def test_categorical_levels_integer_flags_match_apply() -> None:
    s = pd.Series([0] * 90 + [1] * 10)
    rule = {"type": "categorical", "levels": categorical_levels(s, min_share=0.05)}
    assert rule["levels"] == ["0", "1"]
    assert apply_bins(s, rule).tolist() == ["0"] * 90 + ["1"] * 10


# --- woe_table / iv ----------------------------------------------------------


def test_woe_table_counts_and_sign() -> None:
    bins = pd.Series(["low"] * 100 + ["high"] * 100)
    t = woe_table(bins, _target([0.1, 0.5]))
    assert t["n"].sum() == 200
    assert t.loc["high", "woe"] < 0 < t.loc["low", "woe"]


def test_woe_table_finite_when_bin_has_no_bads() -> None:
    bins = pd.Series(["a"] * 100 + ["b"] * 100)
    t = woe_table(bins, _target([0.0, 0.5]))
    assert np.isfinite(t["woe"]).all()


@pytest.mark.parametrize(
    ("iv", "label"),
    [(0.01, "useless"), (0.05, "weak"), (0.2, "medium"), (0.4, "strong"), (0.6, "suspicious")],
)
def test_iv_strength(iv: float, label: str) -> None:
    assert iv_strength(iv) == label


def test_iv_summary_sorted_desc() -> None:
    tables = {
        "a": pd.DataFrame({"iv": [0.01, 0.01]}),
        "b": pd.DataFrame({"iv": [0.2, 0.1]}),
    }
    assert iv_summary(tables).index.tolist() == ["b", "a"]


# --- coarse classing ---------------------------------------------------------


def test_monotonic_edges_merges_reversal() -> None:
    s = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 100))
    target = _target([0.1, 0.3, 0.2, 0.4])
    edges = monotonic_edges(s, target, [-INF, 1.0, 2.0, 3.0, INF])
    assert edges == [-INF, 1.0, 3.0, INF]


def test_monotonic_edges_keeps_monotonic_bins() -> None:
    s = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 100))
    edges = [-INF, 1.0, 2.0, 3.0, INF]
    assert monotonic_edges(s, _target([0.4, 0.3, 0.2, 0.1]), edges) == edges


def test_coarse_spec_leaves_categorical_unchanged() -> None:
    df = pd.DataFrame({"x": np.repeat([1.0, 2.0], 100), "c": ["a"] * 200})
    rule_c = {"type": "categorical", "levels": ["a"]}
    spec = {"x": {"type": "numeric", "edges": [-INF, 1.0, INF]}, "c": rule_c}
    assert coarse_spec(df, _target([0.1, 0.2]), spec)["c"] == rule_c


def test_is_monotonic_ignores_missing_bin() -> None:
    t = pd.DataFrame({"bad_rate": [0.1, 0.2, 0.05]}, index=["(-inf, 1]", "(1, inf]", MISSING])
    assert is_monotonic(t)


# --- woe_frame ---------------------------------------------------------------


def test_woe_frame_maps_woe_and_unseen_bin_to_zero() -> None:
    tables = {"x": pd.DataFrame({"woe": [0.5, -0.5]}, index=pd.Index(["a", "b"], name="bin"))}
    binned = pd.DataFrame({"x": ["a", "b", "c"]})
    assert woe_frame(binned, tables)["x"].tolist() == [0.5, -0.5, 0.0]


def test_woe_table_tiny_bin_gets_neutral_woe() -> None:
    # Regression: one good and no bads used to get a negative WoE (-1.33 on real data)
    bins = pd.Series(["big"] * 1000 + ["tiny"])
    target = pd.Series([1] * 100 + [0] * 900 + [0])
    t = woe_table(bins, target, min_share=0.005)
    assert t.loc["tiny", "woe"] == 0.0 and t.loc["tiny", "iv"] == 0.0
    assert woe_table(bins, target, min_share=0.0).loc["tiny", "woe"] < 0
