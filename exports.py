from __future__ import annotations

from typing import Dict, Any, Optional
from io import BytesIO
import pandas as pd


def export_excel_pack(
    cleaned_df: pd.DataFrame,
    outputs: Optional[Dict[str, Any]] = None,
    filename_prefix: str = "pay_equity_audit",
) -> bytes:
    """
    Creates an in-memory Excel file for Streamlit download.

    cleaned_df: main cleaned dataset (after validate_and_clean)
    outputs: optional dict of extra tables/dfs produced by the app
             e.g. {"unadjusted": df1, "adjusted": df2, "outliers": df3, "coef": df4}
    Returns: bytes (xlsx content)
    """

    bio = BytesIO()

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        # Main sheet
        cleaned_df.to_excel(writer, index=False, sheet_name="cleaned_data")

        # Extra sheets
        if outputs:
            for name, obj in outputs.items():
                if obj is None:
                    continue
                sheet = str(name)[:31]  # Excel sheet name limit
                if isinstance(obj, pd.DataFrame):
                    obj.to_excel(writer, index=False, sheet_name=sheet)
                else:
                    # If it's not a DF, try converting to DF
                    try:
                        pd.DataFrame(obj).to_excel(writer, index=False, sheet_name=sheet)
                    except Exception:
                        # last resort: write as text
                        pd.DataFrame({"value": [str(obj)]}).to_excel(writer, index=False, sheet_name=sheet)

    bio.seek(0)
    return bio.read()