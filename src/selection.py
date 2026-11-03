"""Feature selection based on IV and WoE correlation."""

import pandas as pd


def select_by_iv(iv: pd.DataFrame, min_iv: float = 0.02) -> list[str]:
    """Keep features whose IV reaches a threshold.

    Args:
        iv: Output of :func:`src.woe.iv_summary` (indexed by feature, ``iv`` column).
        min_iv: Minimum IV to keep a feature.

    Returns:
        Feature names sorted by IV descending.
    """
    kept = iv[iv["iv"] >= min_iv].sort_values("iv", ascending=False)
    return kept.index.tolist()


def drop_correlated(
    woe: pd.DataFrame, ranked: list[str], threshold: float = 0.7
) -> list[str]:
    """Greedily drop features highly correlated with a better-ranked one.

    Features are visited in ``ranked`` order (best first); a feature is kept
    only if its absolute correlation with every already-kept feature is
    below ``threshold``.

    Args:
        woe: WoE-encoded features.
        ranked: Candidate features, best first (e.g. by IV).
        threshold: Absolute Pearson correlation above which a feature is dropped.

    Returns:
        Kept features, in ``ranked`` order.
    """
    corr = woe[ranked].corr().abs()
    kept: list[str] = []
    for col in ranked:
        if all(corr.loc[col, k] < threshold for k in kept):
            kept.append(col)
    return kept
