from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def score_outliers(
    df: pd.DataFrame,
    pay_col: str,
    group_cols: List[str],
    z_threshold: float = 2.5,
    iqr_multiplier: float = 1.5,
    min_group_n: int = 8,
) -> Dict[str, Any]:
    """
    Flags pay outliers within peer groups using robust rules:
      - z-score on log(pay) within group (if group size sufficient)
      - IQR rule within group
    Returns:
      {
        "flagged": DataFrame (original rows with flags + diagnostics),
        "summary": DataFrame (counts by group),
        "params": dict
      }
    """

    d = df.copy()

    if pay_col not in d.columns:
        raise ValueError(f"'{pay_col}' not found in dataframe.")

    # keep only groups that exist
    group_cols = [c for c in group_cols if c in d.columns]
    if not group_cols:
        # if no group columns, treat all as one group
        group_cols = ["__all__"]
        d["__all__"] = "ALL"

    d[pay_col] = pd.to_numeric(d[pay_col], errors="coerce")
    d = d.dropna(subset=[pay_col]).copy()
    d = d[d[pay_col] > 0].copy()

    # log for stability
    d["_log_pay"] = np.log(d[pay_col])

    def _iqr_bounds(s: pd.Series) -> Tuple[float, float]:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - iqr_multiplier * iqr
        hi = q3 + iqr_multiplier * iqr
        return lo, hi

    rows = []
    summaries = []

    for keys, g in d.groupby(group_cols, dropna=False):
        g = g.copy()
        n = len(g)

        # IQR on log pay
        lo_iqr, hi_iqr = _iqr_bounds(g["_log_pay"])

        g["iqr_low"] = lo_iqr
        g["iqr_high"] = hi_iqr
        g["flag_iqr"] = (g["_log_pay"] < lo_iqr) | (g["_log_pay"] > hi_iqr)

        # z-score on log pay (only if enough rows)
        if n >= min_group_n and g["_log_pay"].std(ddof=0) > 0:
            mu = g["_log_pay"].mean()
            sd = g["_log_pay"].std(ddof=0)
            g["z"] = (g["_log_pay"] - mu) / sd
            g["flag_z"] = g["z"].abs() >= z_threshold
        else:
            g["z"] = np.nan
            g["flag_z"] = False

        g["outlier_flag"] = g["flag_iqr"] | g["flag_z"]

        # attach group key columns for summary clarity
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                g[col] = val
            key_dict = dict(zip(group_cols, keys))
        else:
            g[group_cols[0]] = keys
            key_dict = {group_cols[0]: keys}

        summaries.append(
            {
                **key_dict,
                "n": n,
                "outliers": int(g["outlier_flag"].sum()),
                "outlier_rate": float(g["outlier_flag"].mean()) if n else 0.0,
            }
        )

        rows.append(g)

    flagged = pd.concat(rows, axis=0).reset_index(drop=True)
    summary = pd.DataFrame(summaries).sort_values(["outliers", "n"], ascending=[False, False])

    # Add a friendly label column (optional)
    flagged["outlier_reason"] = np.where(
        flagged["outlier_flag"],
        np.where(flagged["flag_z"] & flagged["flag_iqr"], "Z+IQR",
                 np.where(flagged["flag_z"], "Z", "IQR")),
        ""
    )

    # cleanup internal helper column
    flagged = flagged.drop(columns=["_log_pay"], errors="ignore")

    return {
        "flagged": flagged,
        "summary": summary,
        "params": {
            "pay_col": pay_col,
            "group_cols": group_cols,
            "z_threshold": z_threshold,
            "iqr_multiplier": iqr_multiplier,
            "min_group_n": min_group_n,
        },
    }