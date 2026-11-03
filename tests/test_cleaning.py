"""Tests for src.cleaning."""

import numpy as np
import pandas as pd

from src.cleaning import (
    DAYS_EMPLOYED_ANOMALY,
    clean_application,
    convert_days_to_years,
    flag_days_employed_anomaly,
    replace_invalid_categories,
)


def test_flag_days_employed_anomaly_does_not_modify_input() -> None:
    df = pd.DataFrame({"DAYS_EMPLOYED": [-100, DAYS_EMPLOYED_ANOMALY]})
    out = flag_days_employed_anomaly(df)
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1]
    assert out["DAYS_EMPLOYED"].iloc[0] == -100
    assert np.isnan(out["DAYS_EMPLOYED"].iloc[1])
    assert df["DAYS_EMPLOYED"].iloc[1] == DAYS_EMPLOYED_ANOMALY


def test_replace_invalid_categories() -> None:
    df = pd.DataFrame({"CODE_GENDER": ["F", "XNA"], "NAME_FAMILY_STATUS": ["Married", "Unknown"]})
    out = replace_invalid_categories(df)
    assert out["CODE_GENDER"].isna().tolist() == [False, True]
    assert out["NAME_FAMILY_STATUS"].isna().tolist() == [False, True]


def test_replace_invalid_categories_skips_absent_columns() -> None:
    out = replace_invalid_categories(pd.DataFrame({"CODE_GENDER": ["XNA"]}))
    assert list(out.columns) == ["CODE_GENDER"]


def test_convert_days_to_years_positive_and_no_negative_zero() -> None:
    out = convert_days_to_years(pd.DataFrame({"DAYS_BIRTH": [-3652.5, 0.0]}))
    assert "DAYS_BIRTH" not in out.columns
    assert out["AGE_YEARS"].tolist() == [10.0, 0.0]
    assert not np.signbit(out["AGE_YEARS"]).any()


def test_clean_application_flags_anomaly_before_conversion() -> None:
    df = pd.DataFrame(
        {"DAYS_EMPLOYED": [-365.25, DAYS_EMPLOYED_ANOMALY], "CODE_GENDER": ["M", "XNA"]}
    )
    out = clean_application(df)
    assert out["YEARS_EMPLOYED"].iloc[0] == 1.0
    assert np.isnan(out["YEARS_EMPLOYED"].iloc[1])
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1]
    assert out["CODE_GENDER"].isna().tolist() == [False, True]
