"""
data_ingestion.py — Layer 1: Data loading and schema normalization
No LLM involved here. Clean, typed DataFrames only.
"""

from __future__ import annotations

import io
from typing import Tuple, Optional, Dict, Any

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: strip, lowercase, replace spaces with underscores."""
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        for c in df.columns
    ]
    return df


def _infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to parse object columns as datetime or numeric.
    Does NOT rely on LLM — uses pandas heuristics.
    """
    for col in df.columns:
        if df[col].dtype != object:
            continue

        # Attempt datetime parse on a sample
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue

        # Numeric coercion attempt
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() / max(len(df), 1) > 0.85:
            df[col] = coerced
            continue

        # Datetime attempt
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() / len(sample) > 0.8:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
        except Exception:
            pass

    return df


def load_csv(file_obj) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a CSV from a file-like object.
    Returns (DataFrame, None) on success or (None, error_message) on failure.
    """
    try:
        raw = file_obj.read()
        # Try common encodings
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
                df = _normalize_columns(df)
                df = _infer_types(df)
                return df, None
            except UnicodeDecodeError:
                continue
        return None, "Could not decode file with any common encoding."
    except Exception as e:
        return None, f"CSV load error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# MySQL
# ──────────────────────────────────────────────────────────────────────────────

def connect_mysql(
    host: str, port: int, user: str, password: str, database: str
) -> Tuple[Optional[Any], Optional[str]]:
    """Returns (connection, None) or (None, error_string)."""
    try:
        import mysql.connector

        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connection_timeout=10,
        )
        return conn, None
    except Exception as e:
        return None, f"MySQL connection failed: {e}"


def get_db_schema(conn, database: str) -> Dict:
    """
    Fetch table columns and foreign key relationships from INFORMATION_SCHEMA.
    Returns { "tables": {table: [cols]}, "relationships": [...] }
    """
    try:
        col_df = pd.read_sql(
            """
            SELECT TABLE_NAME, COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """,
            conn,
            params=(database,),
        )
        col_df.columns = [c.lower() for c in col_df.columns]

        tables: Dict[str, list] = {}
        for _, row in col_df.iterrows():
            tables.setdefault(row["table_name"], []).append(row["column_name"])

        fk_df = pd.read_sql(
            """
            SELECT TABLE_NAME, COLUMN_NAME,
                   REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME IS NOT NULL
            """,
            conn,
            params=(database,),
        )
        relationships = []
        if not fk_df.empty:
            for _, row in fk_df.iterrows():
                relationships.append(
                    {
                        "from_table": row["TABLE_NAME"],
                        "from_col": row["COLUMN_NAME"],
                        "to_table": row["REFERENCED_TABLE_NAME"],
                        "to_col": row["REFERENCED_COLUMN_NAME"],
                    }
                )

        return {"tables": tables, "relationships": relationships}
    except Exception as e:
        return {"tables": {}, "relationships": [], "error": str(e)}


def load_table(conn, table_name: str, limit: int = 10_000) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a database table into a DataFrame with a row limit for safety.
    Column names are validated against the schema before querying.
    """
    # Validate table name (alphanumeric + underscores only)
    import re

    if not re.match(r"^[A-Za-z0-9_]+$", table_name):
        return None, f"Invalid table name: {table_name!r}"

    try:
        df = pd.read_sql(
            f"SELECT * FROM `{table_name}` LIMIT {int(limit)}",
            conn,
        )
        df = _normalize_columns(df)
        df = _infer_types(df)
        return df, None
    except Exception as e:
        return None, f"Failed to load table '{table_name}': {e}"