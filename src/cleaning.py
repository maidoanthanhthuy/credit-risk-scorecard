"""Minimal data cleaning before WoE/IV binning.

All functions return a new DataFrame and never modify their input.
"""

import numpy as np
import pandas as pd

DAYS_EMPLOYED_ANOMALY: int = 365243

INVALID_CATEGORIES: dict[str, tuple[str, ...]] = {
    "CODE_GENDER": ("XNA",),
    "NAME_FAMILY_STATUS": ("Unknown",),
}

DAYS_TO_YEARS: dict[str, str] = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "YEARS_EMPLOYED",
    "DAYS_REGISTRATION": "YEARS_REGISTRATION",
    "DAYS_ID_PUBLISH": "YEARS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE": "YEARS_LAST_PHONE_CHANGE",
}


def flag_days_employed_anomaly(
    df: pd.DataFrame,
    col: str = "DAYS_EMPLOYED",
    anomaly: int = DAYS_EMPLOYED_ANOMALY,
) -> pd.DataFrame:
    """Replace the placeholder employment value with NaN and add a flag column.

    The value 365243 (~1000 years) marks clients without current employment
    (mostly pensioners). It is kept as information via ``<col>_ANOM``.

    Args:
        df: Input DataFrame.
        col: Column holding the anomaly.
        anomaly: Placeholder value to replace.

    Returns:
        New DataFrame with ``col`` cleaned and a binary ``<col>_ANOM`` column.
    """
    is_anomaly = df[col] == anomaly
    return df.assign(
        **{
            col: df[col].mask(is_anomaly, np.nan),
            f"{col}_ANOM": is_anomaly.astype(int),
        }
    )


def replace_invalid_categories(
    df: pd.DataFrame,
    invalid: dict[str, tuple[str, ...]] = INVALID_CATEGORIES,
) -> pd.DataFrame:
    """Replace meaningless category labels with NaN.

    Args:
        df: Input DataFrame.
        invalid: Mapping of column name to labels treated as missing.

    Returns:
        New DataFrame where the listed labels are NaN.
    """
    return df.assign(
        **{
            col: df[col].mask(df[col].isin(labels))
            for col, labels in invalid.items()
            if col in df.columns
        }
    )


def convert_days_to_years(
    df: pd.DataFrame,
    mapping: dict[str, str] = DAYS_TO_YEARS,
) -> pd.DataFrame:
    """Convert negative day offsets (relative to application) to positive years.

    Args:
        df: Input DataFrame.
        mapping: Mapping of source ``DAYS_*`` column to new year column name.

    Returns:
        New DataFrame with year columns added and source columns dropped.
    """
    present = {src: dst for src, dst in mapping.items() if src in df.columns}
    # "+ 0.0" turns negative zero (-0.0, from -0 / 365.25) into 0.0
    years = {dst: (-df[src] / 365.25).round(2) + 0.0 for src, dst in present.items()}
    return df.assign(**years).drop(columns=list(present))


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full minimal cleaning pipeline to the application table.

    Order matters: the employment anomaly is flagged before day-to-year
    conversion so the placeholder never becomes a fake negative year value.

    Args:
        df: Raw application DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    return (
        df.pipe(flag_days_employed_anomaly)
        .pipe(replace_invalid_categories)
        .pipe(convert_days_to_years)
    )
