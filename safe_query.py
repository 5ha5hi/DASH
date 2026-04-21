"""
safe_query.py — Layer 5: Constrained Query Executor
=====================================================
Replaces the dangerous exec() / LLM-generated code pattern with a
deterministic, whitelisted query DSL.

Supported query patterns:
  describe <col>
  top N by <col>
  bottom N by <col>
  group by <col> [sum|mean|count|max|min] <col>
  filter <col> [=|!=|>|<|>=|<=|contains] <value>
  value counts <col>
  correlate <col1> <col2>
  trend <col> by [day|month|year|week]

No exec(). No LLM-generated code. No SQL injection surface.
All column names are validated against the actual DataFrame before execution.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import pandas as pd


class SafeQueryExecutor:
    """
    Parses a restricted query DSL and executes it against a pandas DataFrame.
    Returns (result_df_or_scalar, error_string_or_None).
    """

    AGG_MAP = {
        "sum": "sum",
        "mean": "mean",
        "avg": "mean",
        "average": "mean",
        "count": "count",
        "max": "max",
        "maximum": "max",
        "min": "min",
        "minimum": "min",
        "median": "median",
        "std": "std",
        "stdev": "std",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Lowercase column index for fuzzy matching
        self._col_map = {c.lower(): c for c in df.columns}

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry
    # ──────────────────────────────────────────────────────────────────────────

    def execute(self, query: str) -> Tuple[Optional[Any], Optional[str]]:
        """Parse and execute a query string. Returns (result, error)."""
        q = query.strip().lower()

        if q.startswith("describe"):
            return self._describe(q)
        if q.startswith("top"):
            return self._top_n(q, ascending=False)
        if q.startswith("bottom"):
            return self._top_n(q, ascending=True)
        if q.startswith("group by"):
            return self._group_by(q)
        if q.startswith("filter"):
            return self._filter(q)
        if q.startswith("value counts"):
            return self._value_counts(q)
        if q.startswith("correlate"):
            return self._correlate(q)
        if q.startswith("trend"):
            return self._trend(q)

        return None, (
            "Unrecognised query. Supported: describe, top N by, bottom N by, "
            "group by, filter, value counts, correlate, trend."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Column resolution
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve(self, token: str) -> Optional[str]:
        """Resolve a token to an actual column name (case-insensitive)."""
        token = token.strip().lower().replace(" ", "_")
        return self._col_map.get(token)

    def _require_col(self, token: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (col, None) or (None, error_string)."""
        col = self._resolve(token)
        if col is None:
            available = ", ".join(self.df.columns[:20])
            return None, f"Column '{token}' not found. Available: {available}"
        return col, None

    # ──────────────────────────────────────────────────────────────────────────
    # Query handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _describe(self, q: str) -> Tuple[Any, Optional[str]]:
        # describe <col>   OR   describe (entire df)
        rest = q[len("describe"):].strip()
        if not rest:
            return self.df.describe(include="all").T.reset_index(), None
        col, err = self._require_col(rest)
        if err:
            return None, err

        series = self.df[col]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            result = pd.DataFrame({"Statistic": desc.index, "Value": desc.values})
        else:
            vc = series.value_counts()
            desc = pd.DataFrame(
                {"Value": vc.index, "Count": vc.values}
            )
            result = desc
        return result, None

    def _top_n(self, q: str, ascending: bool) -> Tuple[Any, Optional[str]]:
        # top N by <col>   |   bottom N by <col>
        keyword = "bottom" if ascending else "top"
        pattern = rf"^{keyword}\s+(\d+)\s+by\s+(.+)$"
        m = re.match(pattern, q)
        if not m:
            return None, f"Syntax: {keyword} N by <column>"
        n = int(m.group(1))
        if n > 1000:
            n = 1000  # safety cap
        col, err = self._require_col(m.group(2).strip())
        if err:
            return None, err
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return None, f"Column '{col}' is not numeric. Cannot rank."
        result = self.df.nlargest(n, col) if not ascending else self.df.nsmallest(n, col)
        return result.reset_index(drop=True), None

    def _group_by(self, q: str) -> Tuple[Any, Optional[str]]:
        # group by <col> [agg] <metric_col>
        # e.g. "group by department sum salary"
        rest = q[len("group by"):].strip()

        # Try: group by <dim> <agg> <metric>
        pattern = r"^(\S+)\s+(\S+)\s+(\S+)$"
        m = re.match(pattern, rest)

        if m:
            dim_token = m.group(1)
            agg_token = m.group(2).lower()
            metric_token = m.group(3)
        else:
            # Try: group by <dim> <metric>  (default agg = count)
            pattern2 = r"^(\S+)\s+(\S+)$"
            m2 = re.match(pattern2, rest)
            if m2:
                dim_token, metric_token = m2.group(1), m2.group(2)
                agg_token = "count"
            else:
                # Single column: group by <dim>
                dim_token = rest.strip()
                metric_token = None
                agg_token = "count"

        dim_col, err = self._require_col(dim_token)
        if err:
            return None, err

        agg_func = self.AGG_MAP.get(agg_token)
        if agg_func is None:
            return None, (
                f"Unknown aggregation '{agg_token}'. "
                f"Supported: {', '.join(self.AGG_MAP)}"
            )

        if metric_token:
            metric_col, err = self._require_col(metric_token)
            if err:
                return None, err
            if not pd.api.types.is_numeric_dtype(self.df[metric_col]):
                if agg_func not in ("count",):
                    return None, f"Column '{metric_col}' is not numeric. Try 'count' instead."
            try:
                if agg_func == "count":
                    result = (
                        self.df.groupby(dim_col)[metric_col]
                        .count()
                        .reset_index()
                        .rename(columns={metric_col: "count"})
                        .sort_values("count", ascending=False)
                    )
                else:
                    result = (
                        self.df.groupby(dim_col)[metric_col]
                        .agg(agg_func)
                        .round(2)
                        .reset_index()
                        .sort_values(metric_col, ascending=False)
                    )
            except Exception as e:
                return None, f"Group-by failed: {e}"
        else:
            try:
                result = (
                    self.df.groupby(dim_col)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
            except Exception as e:
                return None, f"Group-by failed: {e}"

        return result.reset_index(drop=True), None

    def _filter(self, q: str) -> Tuple[Any, Optional[str]]:
        # filter <col> <op> <value>
        rest = q[len("filter"):].strip()

        # Detect operator
        ops = [">=", "<=", "!=", ">", "<", "=", "contains", "startswith", "endswith"]
        op_found = None
        for op in ops:
            if f" {op} " in rest:
                op_found = op
                break

        if op_found is None:
            return None, (
                "Syntax: filter <column> <operator> <value>\n"
                "Operators: =  !=  >  <  >=  <=  contains  startswith  endswith"
            )

        parts = rest.split(f" {op_found} ", 1)
        col_token = parts[0].strip()
        value_str = parts[1].strip().strip("'\"")

        col, err = self._require_col(col_token)
        if err:
            return None, err

        series = self.df[col]

        try:
            if op_found in (">", "<", ">=", "<="):
                if not pd.api.types.is_numeric_dtype(series):
                    return None, f"Column '{col}' is not numeric. Cannot use {op_found}."
                value = float(value_str)
                if op_found == ">":
                    mask = series > value
                elif op_found == "<":
                    mask = series < value
                elif op_found == ">=":
                    mask = series >= value
                else:
                    mask = series <= value

            elif op_found == "=":
                # Try numeric match first
                try:
                    value = float(value_str)
                    mask = series == value
                except ValueError:
                    mask = series.astype(str).str.lower() == value_str.lower()

            elif op_found == "!=":
                try:
                    value = float(value_str)
                    mask = series != value
                except ValueError:
                    mask = series.astype(str).str.lower() != value_str.lower()

            elif op_found == "contains":
                mask = series.astype(str).str.lower().str.contains(value_str.lower(), na=False)

            elif op_found == "startswith":
                mask = series.astype(str).str.lower().str.startswith(value_str.lower())

            elif op_found == "endswith":
                mask = series.astype(str).str.lower().str.endswith(value_str.lower())

            else:
                return None, f"Unsupported operator: {op_found}"

            result = self.df[mask].reset_index(drop=True)
            if len(result) == 0:
                return pd.DataFrame(columns=self.df.columns), None
            return result, None

        except Exception as e:
            return None, f"Filter failed: {e}"

    def _value_counts(self, q: str) -> Tuple[Any, Optional[str]]:
        rest = q[len("value counts"):].strip()
        col, err = self._require_col(rest)
        if err:
            return None, err
        vc = self.df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        vc["percent"] = (vc["count"] / len(self.df) * 100).round(1)
        return vc, None

    def _correlate(self, q: str) -> Tuple[Any, Optional[str]]:
        rest = q[len("correlate"):].strip()
        tokens = rest.split()
        if len(tokens) < 2:
            return None, "Syntax: correlate <col1> <col2>"
        c1_tok, c2_tok = tokens[0], tokens[1]
        c1, err = self._require_col(c1_tok)
        if err:
            return None, err
        c2, err = self._require_col(c2_tok)
        if err:
            return None, err
        if not pd.api.types.is_numeric_dtype(self.df[c1]):
            return None, f"Column '{c1}' is not numeric."
        if not pd.api.types.is_numeric_dtype(self.df[c2]):
            return None, f"Column '{c2}' is not numeric."

        pair = self.df[[c1, c2]].dropna()
        r = pair[c1].corr(pair[c2])
        result = pd.DataFrame({
            "Column A": [c1],
            "Column B": [c2],
            "Pearson r": [round(r, 4)],
            "N (pairs)": [len(pair)],
            "Strength": ["strong" if abs(r) > 0.8 else "moderate" if abs(r) > 0.5 else "weak"],
            "Direction": ["positive" if r > 0 else "negative"],
        })
        return result, None

    def _trend(self, q: str) -> Tuple[Any, Optional[str]]:
        # trend <metric_col> by [day|month|year|week]
        rest = q[len("trend"):].strip()
        freq_map = {
            "day": "D",
            "daily": "D",
            "week": "W",
            "weekly": "W",
            "month": "ME",
            "monthly": "ME",
            "quarter": "QE",
            "quarterly": "QE",
            "year": "YE",
            "yearly": "YE",
            "annual": "YE",
        }

        # parse "trend <col> by <freq>"
        m = re.match(r"^(\S+)\s+by\s+(\S+)$", rest)
        if m:
            col_token, freq_token = m.group(1), m.group(2).lower()
        else:
            # try "trend <col>"  (default monthly)
            m2 = re.match(r"^(\S+)$", rest)
            if m2:
                col_token = m2.group(1)
                freq_token = "month"
            else:
                return None, "Syntax: trend <metric_col> by [day|week|month|quarter|year]"

        col, err = self._require_col(col_token)
        if err:
            return None, err

        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return None, f"Column '{col}' is not numeric."

        freq = freq_map.get(freq_token)
        if freq is None:
            return None, f"Unknown frequency '{freq_token}'. Options: {', '.join(freq_map)}"

        # Find a time column
        time_col = None
        for c in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[c]):
                time_col = c
                break
        if time_col is None:
            # Try parsing object columns
            for c in self.df.select_dtypes("object").columns:
                try:
                    parsed = pd.to_datetime(self.df[c], infer_datetime_format=True, errors="coerce")
                    if parsed.notna().sum() / max(len(self.df), 1) > 0.8:
                        time_col = c
                        break
                except Exception:
                    pass

        if time_col is None:
            return None, "No time/date column found in the dataset for trend analysis."

        temp = self.df[[time_col, col]].copy()
        temp[time_col] = pd.to_datetime(temp[time_col], infer_datetime_format=True, errors="coerce")
        temp = temp.dropna().set_index(time_col).sort_index()

        try:
            result = temp[col].resample(freq).sum().reset_index()
            result.columns = ["period", col]
            result["period"] = result["period"].astype(str)
            return result, None
        except Exception as e:
            return None, f"Trend computation failed: {e}"