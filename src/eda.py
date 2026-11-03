"""Exploratory data analysis utilities."""

from typing import Any

import pandas as pd


def summarize_structure(df: pd.DataFrame) -> dict[str, Any]:
    """Return a structural summary of a DataFrame.

    Args:
        df: Input DataFrame (e.g. loaded from ``data/application_train.csv``).

    Returns:
        A dict with:
            - ``n_rows``: number of rows.
            - ``n_cols``: number of columns.
            - ``dtype_counts``: mapping of dtype name to column count.
            - ``dtypes``: mapping of column name to dtype name.
    """
    dtypes = df.dtypes.astype(str)
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtype_counts": dtypes.value_counts().to_dict(),
        "dtypes": dtypes.to_dict(),
    }


def target_distribution(df: pd.DataFrame, target_col: str = "TARGET") -> pd.DataFrame:
    """Return count and share of each class in the target column.

    Args:
        df: Input DataFrame containing the target column.
        target_col: Name of the binary target column (0 = repaid, 1 = default).

    Returns:
        DataFrame indexed by class label with columns ``count`` and ``ratio``
        (ratio sums to 1), sorted by class label.
    """
    counts = df[target_col].value_counts().sort_index()
    return pd.DataFrame({"count": counts, "ratio": counts / counts.sum()})


def imbalance_ratio(dist: pd.DataFrame) -> float:
    """Return the majority-to-minority class count ratio.

    Args:
        dist: Output of :func:`target_distribution`.

    Returns:
        Majority class count divided by minority class count
        (e.g. 11.4 means 11.4 repaid clients per defaulted client).
    """
    return float(dist["count"].max() / dist["count"].min())


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values for columns that have at least one missing value.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame indexed by column name with columns ``n_missing``,
        ``pct_missing`` (0-100) and ``dtype``, sorted by ``pct_missing``
        descending. Columns with no missing values are excluded.
    """
    n_missing = df.isna().sum()
    summary = pd.DataFrame(
        {
            "n_missing": n_missing,
            "pct_missing": n_missing / len(df) * 100,
            "dtype": df.dtypes.astype(str),
        }
    )
    return summary[summary["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def suggest_missing_strategy(
    pct_missing: float, high: float = 60.0, low: float = 5.0
) -> str:
    """Suggest a missing-value treatment for a scorecard feature.

    Args:
        pct_missing: Percentage of missing values in the column (0-100).
        high: Threshold (%) at or above which the column is a drop candidate.
        low: Threshold (%) below which simple imputation is sufficient.

    Returns:
        One of ``"drop_candidate"``, ``"missing_bin"`` (keep missing as a
        separate WoE bin) or ``"impute"`` (median/mode imputation).
    """
    if pct_missing >= high:
        return "drop_candidate"
    if pct_missing >= low:
        return "missing_bin"
    return "impute"


def classify_variables(
    df: pd.DataFrame,
    exclude: tuple[str, ...] = ("SK_ID_CURR", "TARGET"),
    max_categories: int = 10,
    force_numeric: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Classify each feature as numeric or categorical for WoE/IV binning.

    Rules:
        - Columns in ``force_numeric`` -> ``numeric``.
        - Non-numeric dtype -> ``categorical``.
        - Numeric dtype with at most ``max_categories`` distinct non-null
          values (e.g. binary ``FLAG_*`` columns) -> ``categorical``.
        - Otherwise -> ``numeric``.

    Args:
        df: Input DataFrame.
        exclude: Columns to skip (IDs, target).
        max_categories: Max distinct values for a numeric column to be
            treated as categorical.
        force_numeric: Numeric columns to always treat as numeric, e.g.
            low-cardinality count variables whose sparse tail values
            should be merged by interval binning.

    Returns:
        DataFrame indexed by column name with columns ``dtype``,
        ``n_unique`` and ``var_type``, sorted by ``var_type`` then ``n_unique``.
    """
    features = df.drop(columns=[c for c in exclude if c in df.columns])
    n_unique = features.nunique()
    is_numeric = features.dtypes.map(pd.api.types.is_numeric_dtype)
    is_forced = features.columns.isin(force_numeric) & is_numeric
    is_categorical = (~is_numeric | (n_unique <= max_categories)) & ~is_forced
    var_type = is_categorical.map({True: "categorical", False: "numeric"})
    result = pd.DataFrame(
        {"dtype": features.dtypes.astype(str), "n_unique": n_unique, "var_type": var_type}
    )
    return result.sort_values(["var_type", "n_unique"])
