"""Binning, Weight of Evidence (WoE) and Information Value (IV).

Convention (Siddiqi): ``WoE = ln(%good / %bad)``, so a positive WoE means
lower risk than average and a negative WoE means higher risk.

Workflow (fit everything on train, apply unchanged to test):
    ``fit_bins`` (fine classing) -> ``coarse_spec`` (monotonic merge)
    -> ``bin_frame`` -> ``woe_table`` per column -> ``iv_summary``
    -> ``woe_frame`` (WoE-encoded features for the model).
"""

from typing import Any

import numpy as np
import pandas as pd

MISSING: str = "Missing"
# Reserved labels; "Rare" avoids clashing with real labels such as
# ORGANIZATION_TYPE = "Other"
RARE: str = "Rare"

BinRule = dict[str, Any]


def numeric_bin_edges(series: pd.Series, n_bins: int = 10) -> list[float]:
    """Compute quantile-based bin edges for a numeric column.

    Only interior quantiles are used as cut points, so a mass point such as
    "mostly zeros" still yields a ``(-inf, 0]`` vs ``(0, inf]`` split.
    Duplicate quantiles are collapsed, so there may be fewer than ``n_bins``.

    Args:
        series: Numeric column; missing values are ignored.
        n_bins: Target number of bins.

    Returns:
        Sorted edges starting with ``-inf`` and ending with ``inf``.
    """
    probs = np.linspace(0, 1, n_bins + 1)[1:-1]
    inner = sorted(float(q) for q in series.dropna().quantile(probs).unique())
    return [-np.inf, *inner, np.inf]


def categorical_levels(series: pd.Series, min_share: float = 0.01) -> list[str]:
    """Select category labels frequent enough to form their own bin.

    Args:
        series: Categorical column.
        min_share: Minimum share of all rows for a label to be kept;
            rarer labels are grouped into ``RARE`` when binning.

    Returns:
        Sorted list of kept labels, as strings.
    """
    shares = series.dropna().astype(str).value_counts() / len(series)
    return sorted(shares[shares >= min_share].index)


def fit_bins(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_bins: int = 10,
    min_share: float = 0.01,
) -> dict[str, BinRule]:
    """Learn a fine-classing rule for every feature (fit on train only).

    Args:
        df: Training DataFrame.
        numeric_cols: Columns binned by quantile intervals.
        categorical_cols: Columns binned by category label.
        n_bins: Target number of bins for numeric columns.
        min_share: Minimum share for a category to keep its own bin.

    Returns:
        Mapping of column name to rule: ``{"type": "numeric", "edges": [...]}``
        or ``{"type": "categorical", "levels": [...]}``.
    """
    numeric = {
        c: {"type": "numeric", "edges": numeric_bin_edges(df[c], n_bins)}
        for c in numeric_cols
    }
    categorical = {
        c: {"type": "categorical", "levels": categorical_levels(df[c], min_share)}
        for c in categorical_cols
    }
    return numeric | categorical


def _format_edge(x: float, decimals: int) -> str:
    """Format a bin edge compactly, e.g. ``0.2276``, ``135000``, ``inf``."""
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def interval_labels(edges: list[float], decimals: int = 4) -> list[str]:
    """Build readable, unique labels for right-closed intervals.

    Args:
        edges: Sorted bin edges.
        decimals: Decimal places used for rounding the edges.

    Returns:
        Labels such as ``"(0.2276, 0.3297]"``. Falls back to full precision
        if rounding would make two labels identical.
    """
    pairs = list(zip(edges[:-1], edges[1:]))
    labels = [f"({_format_edge(a, decimals)}, {_format_edge(b, decimals)}]" for a, b in pairs]
    if len(set(labels)) < len(labels):
        labels = [f"({a!r}, {b!r}]" for a, b in pairs]
    return labels


def apply_bins(series: pd.Series, rule: BinRule) -> pd.Series:
    """Map raw values to bin labels using a fitted rule.

    Missing values always go to the ``MISSING`` bin. For categorical rules,
    unseen or rare labels go to ``RARE``.

    Args:
        series: Raw column values.
        rule: Rule produced by :func:`fit_bins` or :func:`coarse_spec`.

    Returns:
        Ordered categorical Series of bin labels (intervals in ascending
        order, then ``RARE``/``MISSING``), same index as ``series``.
    """
    if rule["type"] == "numeric":
        edges = rule["edges"]
        intervals = interval_labels(edges)
        order = [*intervals, MISSING]
        # Map by bin position, not by label text: pd.cut rounds its own labels
        codes = pd.cut(series, bins=edges, labels=False)
        labels = codes.map(dict(enumerate(intervals)))
    else:
        order = [*rule["levels"], RARE, MISSING]
        as_str = series.astype(object).map(str, na_action="ignore")
        labels = as_str.where(as_str.isin(rule["levels"]), RARE)
    labels = labels.mask(series.isna(), MISSING)
    # Drop duplicates in case a real label equals a reserved one
    order = list(dict.fromkeys(order))
    return pd.Series(
        pd.Categorical(labels, categories=order, ordered=True),
        index=series.index,
        name=series.name,
    )


def bin_frame(df: pd.DataFrame, spec: dict[str, BinRule]) -> pd.DataFrame:
    """Apply a binning spec to every column it covers.

    Args:
        df: DataFrame with raw feature values.
        spec: Mapping of column name to binning rule.

    Returns:
        DataFrame of bin labels, one column per spec entry.
    """
    return pd.DataFrame({c: apply_bins(df[c], rule) for c, rule in spec.items()})


def woe_table(
    bins: pd.Series, target: pd.Series, smoothing: float = 0.5, min_share: float = 0.005
) -> pd.DataFrame:
    """Compute per-bin statistics, WoE and IV contribution.

    Args:
        bins: Bin labels of one feature (output of :func:`apply_bins`).
        target: Binary target aligned with ``bins`` (1 = bad/default).
        smoothing: Added to good and bad counts of each bin to avoid
            ``log(0)`` when a bin has no goods or no bads.
        min_share: Bins holding less than this share of the population get
            WoE 0 (average risk). Their WoE is unreliable, and additive
            smoothing pulls tiny bins towards 1:1 odds, which can even flip
            the sign (e.g. one good and no bads gives a negative WoE).

    Returns:
        DataFrame indexed by bin (empty bins dropped) with columns ``n``,
        ``share``, ``n_bad``, ``bad_rate``, ``woe`` and ``iv``.
    """
    stats = target.groupby(bins, observed=True).agg(n="count", n_bad="sum")
    good = stats["n"] - stats["n_bad"] + smoothing
    bad = stats["n_bad"] + smoothing
    dist_good = good / good.sum()
    dist_bad = bad / bad.sum()
    share = stats["n"] / stats["n"].sum()
    woe = np.log(dist_good / dist_bad).where(share >= min_share, 0.0)
    table = stats.assign(
        share=share,
        bad_rate=stats["n_bad"] / stats["n"],
        woe=woe,
        iv=(dist_good - dist_bad) * woe,
    )
    table.index.name = "bin"
    return table[["n", "share", "n_bad", "bad_rate", "woe", "iv"]]


def monotonic_edges(
    series: pd.Series, target: pd.Series, edges: list[float]
) -> list[float]:
    """Merge adjacent numeric bins until the bad rate is strictly monotonic.

    The trend direction follows the sign of the correlation between bin
    position and target. Missing values are ignored (they keep their own
    bin). Empty bins are absorbed by their right-hand neighbour.

    Args:
        series: Numeric column.
        target: Binary target aligned with ``series`` (1 = bad).
        edges: Fine-classing edges (from :func:`numeric_bin_edges`).

    Returns:
        Coarser edges, still starting with ``-inf`` and ending with ``inf``.
    """
    codes = pd.cut(series, bins=edges, labels=False)
    known = codes.notna()
    stats = (
        target[known]
        .groupby(codes[known].astype(int))
        .agg(["count", "sum"])
        .reindex(range(len(edges) - 1), fill_value=0)
    )
    increasing = bool(codes[known].corr(target[known]) >= 0)
    # Each block: [right edge, n, n_bad]
    blocks = [
        [edges[i + 1], int(row["count"]), float(row["sum"])]
        for i, row in stats.iterrows()
        if row["count"] > 0
    ]

    def violates(left: list, right: list) -> bool:
        r_left, r_right = left[2] / left[1], right[2] / right[1]
        return r_left >= r_right if increasing else r_left <= r_right

    i = 0
    while i < len(blocks) - 1:
        if violates(blocks[i], blocks[i + 1]):
            left, right = blocks[i], blocks[i + 1]
            blocks[i : i + 2] = [[right[0], left[1] + right[1], left[2] + right[2]]]
            i = max(i - 1, 0)  # a merge can create a new violation on the left
        else:
            i += 1
    return [-np.inf, *[b[0] for b in blocks[:-1]], np.inf]


def coarse_spec(
    df: pd.DataFrame, target: pd.Series, spec: dict[str, BinRule]
) -> dict[str, BinRule]:
    """Turn a fine-classing spec into a monotonic coarse-classing spec.

    Numeric rules get monotonic edges; categorical rules are unchanged
    (their rare labels are already merged into ``RARE``).

    Args:
        df: Training DataFrame.
        target: Binary target aligned with ``df``.
        spec: Fine-classing spec from :func:`fit_bins`.

    Returns:
        New spec with the same keys.
    """
    return {
        c: (
            {"type": "numeric", "edges": monotonic_edges(df[c], target, rule["edges"])}
            if rule["type"] == "numeric"
            else rule
        )
        for c, rule in spec.items()
    }


def is_monotonic(table: pd.DataFrame) -> bool:
    """Check whether bad rate is monotonic across ordinary bins.

    Args:
        table: Output of :func:`woe_table`.

    Returns:
        True if ``bad_rate`` is monotonic, ignoring ``MISSING``/``RARE`` bins.
    """
    rates = table["bad_rate"][~table.index.isin([MISSING, RARE])]
    return bool(rates.is_monotonic_increasing or rates.is_monotonic_decreasing)


def woe_frame(binned: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Replace bin labels with their WoE values.

    Args:
        binned: Bin labels (output of :func:`bin_frame`).
        tables: Mapping of feature name to its :func:`woe_table` (fit on train).

    Returns:
        Float DataFrame with one column per entry in ``tables``. Bins absent
        from the train table get WoE 0 (neutral, i.e. average risk).
    """
    return pd.DataFrame(
        {
            c: binned[c]
            .astype(str)
            .map(dict(zip(t.index.astype(str), t["woe"])))
            .astype(float)
            .fillna(0.0)
            for c, t in tables.items()
        },
        index=binned.index,
    )


def iv_strength(iv: float) -> str:
    """Label predictive power using the usual IV rule of thumb.

    Args:
        iv: Information Value of a feature.

    Returns:
        ``"useless"`` (< 0.02), ``"weak"`` (< 0.1), ``"medium"`` (< 0.3),
        ``"strong"`` (< 0.5) or ``"suspicious"`` (>= 0.5, check for leakage).
    """
    for threshold, label in ((0.02, "useless"), (0.1, "weak"), (0.3, "medium"), (0.5, "strong")):
        if iv < threshold:
            return label
    return "suspicious"


def iv_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize IV across features.

    Args:
        tables: Mapping of feature name to its :func:`woe_table`.

    Returns:
        DataFrame indexed by feature with columns ``iv``, ``n_bins`` and
        ``strength``, sorted by ``iv`` descending.
    """
    summary = pd.DataFrame(
        {
            "iv": {c: float(t["iv"].sum()) for c, t in tables.items()},
            "n_bins": {c: len(t) for c, t in tables.items()},
        }
    )
    summary["strength"] = summary["iv"].map(iv_strength)
    return summary.sort_values("iv", ascending=False)
