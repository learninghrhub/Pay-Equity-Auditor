from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def gap_summary_table(
    df: pd.DataFrame,
    pay_col: str,
    pred_col: str = "pred_pay",
    group_cols: Optional[List[str]] = None,
    protected_col: Optional[str] = None,
    min_n: int = 8,
) -> pd.DataFrame:
    """
    Creates a summary table showing unadjusted and model-adjusted pay gap.

    Assumes:
      - df[pay_col] exists (actual pay)
      - df[pred_col] exists (model predicted pay / fair pay)

    Optional:
      - group_cols for slicing (e.g., ["country","job_family","grade"])
      - protected_col for gap across protected attribute (e.g., "gender")

    Output columns:
      n, actual_mean, pred_mean, unadj_gap_pct, adj_gap_pct,
      underpaid_budget, overpaid_budget, meets_min_n (+ group columns)

    Notes:
      - unadj_gap_pct and adj_gap_pct are expressed as proportions (e.g., -0.12 = -12%)
      - underpaid_budget: sum of (pred - actual) where actual < pred
      - overpaid_budget: sum of (actual - pred) where actual > pred
    """

    d = df.copy()

    # --- Validate columns
    if pay_col not in d.columns:
        raise ValueError(f"'{pay_col}' not found in dataframe.")
    if pred_col not in d.columns:
        raise ValueError(f"'{pred_col}' not found in dataframe. (Model step must create it)")

    # --- Coerce numeric + clean rows
    d[pay_col] = pd.to_numeric(d[pay_col], errors="coerce")
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d = d.dropna(subset=[pay_col, pred_col]).copy()

    # --- Sanitize grouping columns
    use_group_cols: List[str] = []
    if group_cols:
        use_group_cols = [c for c in group_cols if c in d.columns]

    use_protected: Optional[str] = protected_col if (protected_col and protected_col in d.columns) else None

    by: List[str] = []
    if use_group_cols:
        by.extend(use_group_cols)
    if use_protected:
        by.append(use_protected)

    if not by:
        d["__all__"] = "ALL"
        by = ["__all__"]

    def _summ(g: pd.DataFrame) -> Dict[str, Any]:
        n = int(len(g))

        if n == 0:
            return {
                "n": 0,
                "actual_mean": np.nan,
                "pred_mean": np.nan,
                "unadj_gap_pct": np.nan,
                "adj_gap_pct": np.nan,
                "underpaid_budget": 0.0,
                "overpaid_budget": 0.0,
            }

        actual_mean = float(g[pay_col].mean())
        pred_mean = float(g[pred_col].mean())

        # Safe division
        if pred_mean > 0:
            unadj_gap_pct = float(actual_mean / pred_mean - 1.0)
        else:
            unadj_gap_pct = np.nan

        resid = g[pay_col] - g[pred_col]

        if pred_mean > 0:
            adj_gap_pct = float(resid.mean() / pred_mean)
        else:
            adj_gap_pct = np.nan

        underpaid_budget = float(np.clip(-resid, 0, None).sum())
        overpaid_budget = float(np.clip(resid, 0, None).sum())

        return {
            "n": n,
            "actual_mean": actual_mean,
            "pred_mean": pred_mean,
            "unadj_gap_pct": unadj_gap_pct,
            "adj_gap_pct": adj_gap_pct,
            "underpaid_budget": underpaid_budget,
            "overpaid_budget": overpaid_budget,
        }

    rows: List[Dict[str, Any]] = []

    # Use list() to avoid weird generator edge cases in some pandas versions
    grouped = list(d.groupby(by, dropna=False))

    for keys, g in grouped:
        if isinstance(keys, tuple):
            key_dict = dict(zip(by, keys))
            group_key = " | ".join([f"{k}={v}" for k, v in key_dict.items()])
        else:
            key_dict = {by[0]: keys}
            group_key = f"{by[0]}={keys}"

        res = _summ(g)
        row = {**key_dict, "group_key": group_key, **res}
        rows.append(row)

    res_df = pd.DataFrame(rows)

    # --- Absolute safety: ensure n exists always (prevents KeyError in downstream code)
    if "n" not in res_df.columns:
        res_df["n"] = 0

    res_df["meets_min_n"] = res_df["n"].astype(int) >= int(min_n)

    # Sort by biggest absolute adjusted gap then by n
    if "adj_gap_pct" in res_df.columns:
        res_df["_abs_adj"] = res_df["adj_gap_pct"].abs()
        res_df = res_df.sort_values(["_abs_adj", "n"], ascending=[False, False]).drop(columns=["_abs_adj"])

    return res_df


def outlier_budget(
    flagged: pd.DataFrame,
    pay_col: str,
    target_col: Optional[str] = None,
    flag_col: str = "outlier_flag",
) -> pd.DataFrame:
    """
    Summarizes budget impact for flagged outliers.

    Inputs:
      - flagged: dataframe from score_outliers(...) output
      - pay_col: actual pay column name
      - target_col: optional column representing "target/fair" pay (e.g., pred_pay)
      - flag_col: boolean column marking outliers

    Output:
      total_outliers, total_population, outlier_rate,
      spend_outliers, spend_total, spend_share_outliers,
      optional: adjustment_budget_to_target (if target_col provided)
    """

    d = flagged.copy()

    if pay_col not in d.columns:
        raise ValueError(f"'{pay_col}' not found in dataframe.")

    if flag_col not in d.columns:
        d[flag_col] = False

    d[pay_col] = pd.to_numeric(d[pay_col], errors="coerce")
    d = d.dropna(subset=[pay_col]).copy()

    total_population = int(len(d))
    total_outliers = int(d[flag_col].astype(bool).sum())
    outlier_rate = float(total_outliers / total_population) if total_population else 0.0

    spend_total = float(d[pay_col].sum())
    spend_outliers = float(d.loc[d[flag_col].astype(bool), pay_col].sum())
    spend_share_outliers = float(spend_outliers / spend_total) if spend_total else 0.0

    result: Dict[str, Any] = {
        "total_population": total_population,
        "total_outliers": total_outliers,
        "outlier_rate": outlier_rate,
        "spend_total": spend_total,
        "spend_outliers": spend_outliers,
        "spend_share_outliers": spend_share_outliers,
    }

    if target_col and target_col in d.columns:
        d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
        dd = d.dropna(subset=[target_col]).copy()
        diff = dd[target_col] - dd[pay_col]
        adj_budget = float(np.clip(diff, 0, None).sum())
        result["adjustment_budget_to_target"] = adj_budget

    return pd.DataFrame([result])