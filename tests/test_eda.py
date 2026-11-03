"""Tests for src.eda."""

import pandas as pd
import pytest

from src.eda import (
    classify_variables,
    imbalance_ratio,
    missing_summary,
    suggest_missing_strategy,
    summarize_structure,
    target_distribution,
)


def test_summarize_structure() -> None:
    s = summarize_structure(pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]}))
    assert (s["n_rows"], s["n_cols"]) == (2, 2)
    assert s["dtypes"] == {"a": "int64", "b": "float64"}
    assert s["dtype_counts"] == {"int64": 1, "float64": 1}


def test_target_distribution_and_imbalance() -> None:
    dist = target_distribution(pd.DataFrame({"TARGET": [0] * 9 + [1]}))
    assert dist["count"].tolist() == [9, 1]
    assert dist.loc[1, "ratio"] == pytest.approx(0.1)
    assert imbalance_ratio(dist) == pytest.approx(9.0)


def test_missing_summary_excludes_complete_columns_and_sorts() -> None:
    df = pd.DataFrame(
        {"full": [1, 2, 3, 4], "half": [1, None, 3, None], "most": [None, None, None, 1]}
    )
    m = missing_summary(df)
    assert m.index.tolist() == ["most", "half"]
    assert m.loc["most", "pct_missing"] == 75.0


@pytest.mark.parametrize(
    ("pct", "strategy"),
    [(70, "drop_candidate"), (60, "drop_candidate"), (30, "missing_bin"), (5, "missing_bin"), (4.9, "impute")],
)
def test_suggest_missing_strategy(pct: float, strategy: str) -> None:
    assert suggest_missing_strategy(pct) == strategy


def test_classify_variables_rules() -> None:
    df = pd.DataFrame(
        {
            "SK_ID_CURR": range(20),
            "TARGET": [0, 1] * 10,
            "flag": [0, 1] * 10,
            "amount": [float(i) for i in range(20)],
            "label": ["a", "b"] * 10,
            "count": [i % 5 for i in range(20)],
        }
    )
    vt = classify_variables(df, max_categories=10, force_numeric=("count",))
    assert vt["var_type"].to_dict() == {
        "flag": "categorical",
        "amount": "numeric",
        "label": "categorical",
        "count": "numeric",
    }
