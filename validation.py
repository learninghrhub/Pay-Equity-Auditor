from __future__ import annotations

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "employee_id",
    "base_pay_annual",
    "total_cash_annual",
    "grade",
    "job_family",
    "country",
    "gender",
    "nationality",
    "expat_local",
]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: lower, strip, replace spaces with underscore.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def _to_numeric(df: pd.DataFrame, col: str, issues: list[str]) -> None:
    if col in df.columns:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna().sum()
        if after_na > before_na:
            issues.append(f"{col}: some values could not be converted to numeric (set to NaN).")


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Validates and cleans an uploaded compensation snapshot.

    Returns:
        df_clean: cleaned dataframe
        issues: list of warnings / data issues detected
    """
    issues: list[str] = []

    # 1) Standardize col names
    df = _standardize_columns(df)

    # 2) Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append("Missing required columns: " + ", ".join(missing))

    # 3) Ensure key columns exist even if missing (prevents crashes later)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    # 4) Type conversions
    _to_numeric(df, "base_pay_annual", issues)
    _to_numeric(df, "total_cash_annual", issues)

    # Optional numeric controls (if present)
    for c in ["tenure_years", "time_in_grade_years", "performance_rating", "fte"]:
        if c in df.columns:
            _to_numeric(df, c, issues)

    # 5) Basic cleaning
    # remove fully empty rows
    df = df.dropna(how="all").copy()

    # remove duplicate employee rows (keep latest occurrence)
    if "employee_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["employee_id"], keep="last")
        after = len(df)
        if after < before:
            issues.append(f"Removed {before - after} duplicate employee_id rows.")

    # 6) Pay sanity checks
    if "base_pay_annual" in df.columns:
        neg = (df["base_pay_annual"] < 0).sum()
        if neg > 0:
            issues.append(f"base_pay_annual has {neg} negative values (set to NaN).")
            df.loc[df["base_pay_annual"] < 0, "base_pay_annual"] = np.nan

    if "total_cash_annual" in df.columns:
        neg = (df["total_cash_annual"] < 0).sum()
        if neg > 0:
            issues.append(f"total_cash_annual has {neg} negative values (set to NaN).")
            df.loc[df["total_cash_annual"] < 0, "total_cash_annual"] = np.nan

    # total_cash should be >= base_pay (not always true, but usually)
    if "base_pay_annual" in df.columns and "total_cash_annual" in df.columns:
        bad = (df["total_cash_annual"] < df["base_pay_annual"]).sum()
        if bad > 0:
            issues.append(f"{bad} rows have total_cash_annual < base_pay_annual (check allowances/bonus fields).")

    # 7) Normalize categorical columns to string
    for c in ["grade", "job_family", "country", "business_unit", "gender", "nationality", "expat_local"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df, issues