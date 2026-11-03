"""Train/test splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "TARGET",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets preserving the target ratio.

    Deterministic for a fixed ``random_state``.

    Args:
        df: Input DataFrame containing the target column.
        target_col: Column used for stratification.
        test_size: Fraction of rows assigned to the test set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(train, test)`` of DataFrames.
    """
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    return train, test


def split_summary(
    splits: dict[str, pd.DataFrame],
    target_col: str = "TARGET",
    total: int | None = None,
) -> pd.DataFrame:
    """Compare size and default rate across data splits.

    Args:
        splits: Mapping of split name (e.g. ``"train"``) to DataFrame.
        target_col: Binary target column (1 = default).
        total: Row count used as denominator for ``share``. Defaults to the
            sum over all splits; pass the full dataset size when ``splits``
            also contains the full dataset.

    Returns:
        DataFrame indexed by split name with columns ``n_rows``,
        ``share`` (fraction of ``total``), ``n_default`` and ``default_rate``.
    """
    summary = pd.DataFrame(
        {
            "n_rows": {name: len(d) for name, d in splits.items()},
            "n_default": {name: int(d[target_col].sum()) for name, d in splits.items()},
        }
    )
    return summary.assign(
        share=summary["n_rows"] / (total or summary["n_rows"].sum()),
        default_rate=summary["n_default"] / summary["n_rows"],
    )[["n_rows", "share", "n_default", "default_rate"]]
