"""Tests for src.split."""

import pandas as pd
import pytest

from src.split import split_summary, stratified_split


def _df() -> pd.DataFrame:
    """1,000 clients with a 10% default rate."""
    return pd.DataFrame({"SK_ID_CURR": range(1000), "TARGET": [1] * 100 + [0] * 900})


def test_stratified_split_preserves_rate_and_is_disjoint() -> None:
    train, test = stratified_split(_df(), test_size=0.2, random_state=0)
    assert (len(train), len(test)) == (800, 200)
    assert train["TARGET"].mean() == pytest.approx(0.1)
    assert test["TARGET"].mean() == pytest.approx(0.1)
    assert set(train["SK_ID_CURR"]).isdisjoint(test["SK_ID_CURR"])


def test_stratified_split_is_deterministic() -> None:
    a, _ = stratified_split(_df(), random_state=1)
    b, _ = stratified_split(_df(), random_state=1)
    assert a.index.equals(b.index)


def test_split_summary_share_uses_total() -> None:
    df = _df()
    train, test = stratified_split(df)
    s = split_summary({"full": df, "train": train, "test": test}, total=len(df))
    assert s["share"].tolist() == pytest.approx([1.0, 0.8, 0.2])
    assert s["default_rate"].tolist() == pytest.approx([0.1, 0.1, 0.1])
