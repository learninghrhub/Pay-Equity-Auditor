import streamlit as st
import pandas as pd
import plotly.express as px

from core.validation import validate_and_clean
from core.peer_groups import assign_peer_groups
from core.modeling_gap import adjusted_gap_ols
from core.modeling_outliers import score_outliers
from core.reporting import gap_summary_table, outlier_budget


# ----------------------------
# Helpers
# ----------------------------
def read_file(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith(".xlsx"):
        return pd.read_excel(f)
    raise ValueError("Unsupported file type")


def normalize_validate_output(result):
    """
    validate_and_clean may return:
      - df_clean
      - (df_clean, report_dict) or (df_clean, messages)
    """
    if isinstance(result, tuple):
        df_clean = result[0]
        report = result[1] if len(result) > 1 else None
        return df_clean, report
    return result, None


def extract_df_from_outlier_dict(out_dict):
    """
    score_outliers returns Dict[str, Any].
    We try common keys to find the flagged dataframe.
    """
    if not isinstance(out_dict, dict):
        return None

    # common key names people use
    candidate_keys = [
        "flagged",
        "flagged_df",
        "df_flagged",
        "results",
        "data",
        "outliers",
        "df",
        "scored_df",
    ]

    for k in candidate_keys:
        if k in out_dict and isinstance(out_dict[k], pd.DataFrame):
            return out_dict[k]

    # fallback: any dataframe inside the dict
    for v in out_dict.values():
        if isinstance(v, pd.DataFrame):
            return v

    return None


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Pay Equity Auditor", layout="wide")
st.title("Pay Equity Auditor")
st.write("✅ App loaded successfully. Upload a file to begin.")


# ----------------------------
# Upload
# ----------------------------
file = st.file_uploader("Upload compensation snapshot (CSV / Excel)", type=["csv", "xlsx"])
if file is None:
    st.info("Please upload a CSV or XLSX file to proceed.")
    st.stop()

df_raw = read_file(file)

st.subheader("Data Preview (raw)")
st.dataframe(df_raw.head(20), use_container_width=True)


# ----------------------------
# Validate / Clean
# ----------------------------
st.subheader("Data Quality Checks")

try:
    result = validate_and_clean(df_raw)
    df, report = normalize_validate_output(result)

    st.success("No critical issues detected. Data cleaned successfully.")

    if report is not None:
        with st.expander("Validation / Cleaning Report"):
            # report might be dict or list or string — show safely
            if isinstance(report, dict):
                st.json(report)
            else:
                st.write(report)

except Exception as e:
    st.error(f"Validation / cleaning failed: {e}")
    st.stop()

st.subheader("Data Preview (cleaned)")
st.dataframe(df.head(20), use_container_width=True)


# ----------------------------
# Filters
# ----------------------------
st.subheader("Scope Filters")

filter_cols = []
col1, col2, col3 = st.columns(3)

if "country" in df.columns:
    with col1:
        countries = sorted(df["country"].dropna().astype(str).unique().tolist())
        sel_country = st.multiselect("Country", countries, default=[])
    if sel_country:
        df = df[df["country"].astype(str).isin(sel_country)]
        filter_cols.append("country")

if "grade" in df.columns:
    with col2:
        grades = sorted(df["grade"].dropna().astype(str).unique().tolist())
        sel_grade = st.multiselect("Grade", grades, default=[])
    if sel_grade:
        df = df[df["grade"].astype(str).isin(sel_grade)]
        filter_cols.append("grade")

if "job_family" in df.columns:
    with col3:
        fams = sorted(df["job_family"].dropna().astype(str).unique().tolist())
        sel_fam = st.multiselect("Job Family", fams, default=[])
    if sel_fam:
        df = df[df["job_family"].astype(str).isin(sel_fam)]
        filter_cols.append("job_family")

if len(df) < 30:
    st.warning("Your current filters leave fewer than 30 rows. Results may be unstable.")


# ----------------------------
# Analysis controls
# ----------------------------
st.subheader("Analysis Controls")

pay_candidates = [c for c in df.columns if "pay" in c.lower() or "salary" in c.lower()]
if not pay_candidates:
    st.error("No pay/salary column detected. Please ensure your dataset has a pay column like base_pay_annual.")
    st.stop()

protected_candidates = [c for c in df.columns if c.lower() in ["gender", "nationality", "ethnicity"]]
if not protected_candidates:
    protected_candidates = df.columns.tolist()

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    pay_col = st.selectbox("Pay measure", pay_candidates, index=0)
with c2:
    protected_col = st.selectbox("Protected attribute", protected_candidates, index=0)
with c3:
    underpaid_threshold = st.slider("Underpaid threshold (%) vs expected", min_value=-50, max_value=-1, value=-10, step=1)


# ----------------------------
# Unadjusted Summary
# ----------------------------
st.subheader("Unadjusted Pay (by group)")

try:
    unadj = gap_summary_table(
        df=df,
        pay_col=pay_col,
        pred_col="pred_pay" if "pred_pay" in df.columns else pay_col,
        group_cols=filter_cols,
        protected_col=protected_col,
        min_n=8,
    )
    st.dataframe(unadj, use_container_width=True)
except Exception as e:
    st.error(f"Unadjusted summary failed: {e}")


# ----------------------------
# Adjusted Gap (Regression)
# ----------------------------
st.subheader("Adjusted Pay Gap (Regression Audit)")

controls = [c for c in ["age", "tenure", "job_level", "grade", "job_family", "country"] if c in df.columns]

try:
    res = adjusted_gap_ols(
        df=df,
        pay_col=pay_col,
        protected_col=protected_col,
        controls=controls,
        min_n=30,
    )

    st.success(f"Model fitted. n={res['n']} | R²={res['r2']:.3f}")

    effects_df = pd.DataFrame(res["effects"])
    st.dataframe(effects_df, use_container_width=True)

    with st.expander("Model summary (statsmodels)"):
        st.text(res["model_summary_text"])

except Exception as e:
    st.error(f"Adjusted gap model failed: {e}")


# ----------------------------
# Outlier Detection
# ----------------------------
st.subheader("Outlier Detection")

try:
    df_pg = assign_peer_groups(df) if "peer_group" not in df.columns else df

    # group_cols is REQUIRED by your function signature
    if "peer_group" in df_pg.columns:
        group_cols = ["peer_group"]
    else:
        group_cols = [c for c in ["country", "grade", "job_family"] if c in df_pg.columns]

    if not group_cols:
        st.warning("No grouping columns available for outlier detection. Add country/grade/job_family or peer_group.")
        st.stop()

    st.write(f"Using outlier grouping: {group_cols}")

    out = score_outliers(df_pg, pay_col=pay_col, group_cols=group_cols)

    flagged_df = extract_df_from_outlier_dict(out)
    if flagged_df is None:
        st.error("score_outliers() did not return a dataframe inside the dict. Showing raw output:")
        st.json(out)
        st.stop()

    st.dataframe(flagged_df.head(50), use_container_width=True)

    bud = outlier_budget(
        flagged_df,
        pay_col=pay_col,
        target_col="pred_pay" if "pred_pay" in flagged_df.columns else None,
    )
    st.subheader("Outlier Budget Summary")
    st.dataframe(bud, use_container_width=True)

except Exception as e:
    st.error(f"Outlier analysis failed: {e}")