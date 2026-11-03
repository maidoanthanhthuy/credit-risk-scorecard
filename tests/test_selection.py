"""Tests for src.selection."""

import numpy as np
import pandas as pd

from src.selection import drop_correlated, select_by_iv


def test_select_by_iv_threshold_and_order() -> None:
    iv = pd.DataFrame({"iv": [0.01, 0.3, 0.05]}, index=["a", "b", "c"])
    assert select_by_iv(iv, min_iv=0.02) == ["b", "c"]


def test_drop_correlated_keeps_higher_ranked_feature() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    woe = pd.DataFrame({"strong": x, "dup": 2 * x + 0.01, "indep": rng.normal(size=200)})
    assert drop_correlated(woe, ["strong", "dup", "indep"], threshold=0.7) == ["strong", "indep"]
