from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _safe_one_hot(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    One-hot encode categorical columns safely.
    - Keeps numeric columns as-is
    - Encodes object/category columns
    - Drops first to avoid multicollinearity
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)

    cat_cols = [
        c for c in cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    num_cols = [c for c in cols if c not in cat_cols]

    parts = []
    if num_cols:
        parts.append(df[num_cols].copy())

    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols].fillna("NA"), drop_first=True))

    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    return X


def adjusted_gap_ols(
    df: pd.DataFrame,
    pay_col: str,
    protected_col: str,
    controls: List[str],
    min_n: int = 30,
) -> Dict[str, Any]:
    """
    Pay equity adjusted gap model using log(pay) OLS regression.

    Returns dict:
      {
        "n": int,
        "r2": float,
        "effects": list[dict],
        "model_summary_text": str
      }
    """

    d = df.copy()

    # Basic checks
    if pay_col not in d.columns:
        raise ValueError(f"'{pay_col}' not found in dataframe.")
    if protected_col not in d.columns:
        raise ValueError(f"'{protected_col}' not found in dataframe.")

    # Keep only relevant columns
    use_cols = [pay_col, protected_col] + [c for c in controls if c in d.columns]
    d = d[use_cols].copy()

    # Clean pay
    d[pay_col] = pd.to_numeric(d[pay_col], errors="coerce")
    d = d.dropna(subset=[pay_col, protected_col]).copy()
    d = d[d[pay_col] > 0].copy()

    if len(d) < min_n:
        raise ValueError(f"Not enough rows after cleaning (n={len(d)}). Need at least {min_n}.")

    # Log pay
    d["_log_pay"] = np.log(d[pay_col])

    # Protected treated as categorical (drop_first => reference group)
    protected_dummies = pd.get_dummies(
        d[protected_col].astype(str),
        prefix=protected_col,
        drop_first=True
    )

    # Controls (mixed numeric + categorical)
    X_controls = _safe_one_hot(d, controls)

    # Build X
    X = pd.concat([protected_dummies, X_controls], axis=1)

    # ---- CRITICAL FIX: ensure X and y are purely numeric floats ----
    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # Convert all columns to numeric (anything weird becomes NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN (can happen if a control is all missing)
    X = X.dropna(axis=1, how="all")

    # Add constant AFTER cleaning/conversion
    X = sm.add_constant(X, has_constant="add")

    # y numeric
    y = pd.to_numeric(d["_log_pay"], errors="coerce")

    # Keep only rows where BOTH y and all X values exist
    mask = y.notna() & X.notna().all(axis=1)
    X = X.loc[mask].astype(float)
    y = y.loc[mask].astype(float)

    if len(y) < min_n:
        raise ValueError(
            f"Not enough usable rows after numeric conversion (n={len(y)}). "
            f"Try fewer filters or remove problematic controls."
        )
    # ---------------------------------------------------------------

    # Fit OLS
    model = sm.OLS(y, X).fit()

    # Extract protected effects
    effects = []
    for term in protected_dummies.columns:
        coef = float(model.params.get(term, np.nan))
        pval = float(model.pvalues.get(term, np.nan))
        pct = (np.exp(coef) - 1.0) * 100.0  # exp(beta)-1 => % impact

        group_value = term.replace(f"{protected_col}_", "")

        effects.append(
            {
                "protected_attribute": protected_col,
                "group_vs_reference": group_value,
                "coef_log": round(coef, 6),
                "pct_impact": round(pct, 2),
                "p_value": round(pval, 6),
            }
        )

    return {
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "effects": effects,
        "model_summary_text": model.summary().as_text(),
    }