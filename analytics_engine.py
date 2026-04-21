"""
analytics_engine.py — Layer 2: Deterministic Data Intelligence
==============================================================
This module is the heart of the data-first architecture.

It performs ALL numerical computation BEFORE the LLM sees any data:
  - Column classification (time / metric / dimension)
  - Per-column statistical profiling
  - KPI computation with real values
  - Trend detection on time series
  - Correlation analysis
  - Anomaly / outlier flagging

The LLM receives the OUTPUT of this module, not raw data.
Zero exec(), zero guesswork.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ──────────────────────────────────────────────────────────────────────────────
# Column Classification
# ──────────────────────────────────────────────────────────────────────────────

_TIME_KEYWORDS = {
    "date", "time", "year", "month", "week", "day", "created", "updated",
    "timestamp", "dt", "period", "at", "on",
}

_IDENTIFIER_KEYWORDS = {
    "id", "key", "uuid", "ref", "idx", "index",
    "postal", "zip", "postcode", "pincode", "pin",
    "phone", "fax", "ssn", "ein", "sku", "barcode",
    "order_id", "customer_id", "product_id", "row_id",
}

_IDENTIFIER_SUBSTRINGS = (
    "_id", "id_", " id", "id ",
    "postal", "zip", "postcode", "phone", "fax",
    "barcode", "sku", "isbn", "ein", "ssn",
)

_METRIC_KEYWORDS = {
    "amount", "price", "cost", "revenue", "sales", "salary", "wage", "fee",
    "score", "rate", "qty", "quantity", "units", "total", "sum",
    "avg", "mean", "profit", "margin", "discount", "tax",
    "duration", "size", "weight", "height", "length", "width", "volume",
    "spend", "budget", "forecast", "target", "actual", "count",
}

# ──────────────────────────────────────────────────────────────────────────────
# Business Metric Detection — Core KPIs
# ──────────────────────────────────────────────────────────────────────────────

_BUSINESS_METRIC_KEYWORDS = {
    "revenue": {"revenue", "sales", "income", "turnover", "earnings"},
    "cost": {"cost", "expense", "spend", "expenditure", "cogs", "overhead"},
    "profit": {"profit", "earnings", "net_income", "gross_profit", "ebitda"},
    "quantity": {"quantity", "qty", "units", "volume", "count"},
    "price": {"price", "unit_price", "rate", "fee"},
    "margin": {"margin", "margin_percent", "profit_margin"},
    "discount": {"discount", "rebate", "allowance"},
    "customers": {"customer", "client", "subscriber", "user"},
    "orders": {"order", "transaction", "purchase", "booking"},
}

_KPI_PRIORITY = [
    "profit", "revenue", "margin", "growth", "aov", "cac", "ltv",
    "conversion", "retention", "churn", "cost", "quantity", "price",
]

# ──────────────────────────────────────────────────────────────────────────────
# FIX: Segment-worthy dimension filter
# ──────────────────────────────────────────────────────────────────────────────

# Maximum cardinality for a dimension to be useful in segment analysis.
# Columns with more unique values than this are skipped (e.g. Order ID, City).
_MAX_SEGMENT_CARDINALITY = 50

# Columns that are structurally identifiers and should never be used for
# segment grouping, even if their cardinality is low.
_SEGMENT_BLOCKLIST_SUBSTRINGS = (
    "_id", "id_", " id", "row id", "order id", "customer id",
    "product id", "postal", "zip", "postcode", "phone",
)


def _segment_worthy_dims(df: pd.DataFrame, dim_cols: List[str]) -> List[str]:
    """
    Return dim_cols filtered to columns that are actually useful for
    segment-level business analysis:
      - cardinality <= _MAX_SEGMENT_CARDINALITY
      - name is not a structural identifier (order ID, row ID, etc.)

    Results are sorted so lower-cardinality (more meaningful) dimensions
    come first — e.g. Category (3) before Sub-Category (17) before State (49).
    This ensures the most interpretable groupings are always included.
    """
    worthy = []
    for col in dim_cols:
        col_lower = col.lower()
        # Block structural identifiers regardless of cardinality
        if any(sub in col_lower for sub in _SEGMENT_BLOCKLIST_SUBSTRINGS):
            continue
        n_unique = df[col].nunique()
        # Skip single-value columns (e.g. Country = "United States" only) —
        # they produce no meaningful segment split.
        if n_unique <= 1:
            continue
        if n_unique <= _MAX_SEGMENT_CARDINALITY:
            worthy.append((n_unique, col))

    # Sort by cardinality ascending so broadest groupings (e.g. Category=3)
    # come before finer ones (Sub-Category=17, Region=4, Segment=3).
    worthy.sort(key=lambda x: x[0])
    return [col for _, col in worthy]


def _is_identifier_column(name_lower: str, series: pd.Series) -> bool:
    tokens = set(name_lower.replace("_", " ").split())

    if tokens & _IDENTIFIER_KEYWORDS:
        return True

    if any(sub in name_lower for sub in _IDENTIFIER_SUBSTRINGS):
        return True

    if pd.api.types.is_integer_dtype(series) or (
        pd.api.types.is_float_dtype(series)
        and (series.dropna() % 1 == 0).all()
    ):
        clean = series.dropna()
        if len(clean) > 0:
            mn, mx = clean.min(), clean.max()
            if 1000 <= mn and mx <= 999999:
                unique_ratio = series.nunique() / max(len(series), 1)
                if unique_ratio > 0.05:
                    if not (tokens & _METRIC_KEYWORDS):
                        return True

    return False


def classify_column(series: pd.Series, col_name: str) -> str:
    name_lower = col_name.lower()
    tokens = set(name_lower.replace("_", " ").split())

    if pd.api.types.is_datetime64_any_dtype(series):
        return "time"

    if series.dtype == object:
        sample = series.dropna().head(50)
        if len(sample) > 5:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() / len(sample) > 0.8:
                    return "time"
            except Exception:
                pass

    if tokens & _TIME_KEYWORDS:
        return "time"

    if pd.api.types.is_numeric_dtype(series):
        if _is_identifier_column(name_lower, series):
            return "dimension"

        if tokens & _METRIC_KEYWORDS:
            return "metric"

        unique_vals = series.dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return "dimension"

        if series.nunique() <= 12 and pd.api.types.is_integer_dtype(series):
            return "dimension"

        return "metric"

    return "dimension"


# ──────────────────────────────────────────────────────────────────────────────
# Per-column profiling
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, float):
        return f"{val:,.{decimals}f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def profile_metric_column(series: pd.Series) -> Dict:
    clean = series.dropna()
    if len(clean) == 0:
        return {"stats": {}, "has_outliers": False, "skewness": None}

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((clean < lower_fence) | (clean > upper_fence)).sum())

    skew = float(clean.skew()) if len(clean) > 2 else 0.0

    stats = {
        "count": _fmt(len(clean)),
        "mean": _fmt(float(clean.mean())),
        "median": _fmt(float(clean.median())),
        "std": _fmt(float(clean.std())),
        "min": _fmt(float(clean.min())),
        "max": _fmt(float(clean.max())),
        "p25": _fmt(float(q1)),
        "p75": _fmt(float(q3)),
    }

    return {
        "stats": stats,
        "has_outliers": outlier_count > 0,
        "outlier_count": outlier_count,
        "skewness": round(skew, 3),
        "raw_mean": float(clean.mean()),
        "raw_sum": float(clean.sum()),
        "raw_min": float(clean.min()),
        "raw_max": float(clean.max()),
        "raw_median": float(clean.median()),
        "raw_std": float(clean.std()) if len(clean) > 1 else 0.0,
    }


def profile_dimension_column(series: pd.Series) -> Dict:
    clean = series.dropna()
    n_total = len(series)
    vc = clean.value_counts()

    top_values = []
    for val, cnt in vc.head(10).items():
        pct = round(cnt / n_total * 100, 1) if n_total > 0 else 0
        top_values.append([str(val), int(cnt), pct])

    stats = {
        "unique": _fmt(series.nunique()),
        "non_null": _fmt(len(clean)),
        "top_value": str(vc.index[0]) if len(vc) > 0 else "N/A",
        "top_freq": _fmt(int(vc.iloc[0])) if len(vc) > 0 else "N/A",
    }

    return {"stats": stats, "top_values": top_values}


def profile_time_column(series: pd.Series) -> Dict:
    parsed = pd.to_datetime(series, errors="coerce")
    clean = parsed.dropna()
    if len(clean) == 0:
        return {"stats": {}, "time_range_days": 0}

    min_dt, max_dt = clean.min(), clean.max()
    range_days = (max_dt - min_dt).days

    stats = {
        "min": str(min_dt.date()),
        "max": str(max_dt.date()),
        "range_days": _fmt(range_days),
        "non_null": _fmt(len(clean)),
    }
    return {"stats": stats, "time_range_days": range_days, "_parsed": parsed}


# ──────────────────────────────────────────────────────────────────────────────
# Trend Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_trend(values: List[float]) -> Tuple[str, str]:
    if len(values) < 3:
        return "Insufficient data", "neutral"

    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return "Insufficient data", "neutral"

    x, y = x[valid], y[valid]
    slope, _, r_value, p_value, _ = scipy_stats.linregress(x, y)

    mean_y = np.mean(y)
    if mean_y == 0:
        pct_change_per_period = 0.0
    else:
        pct_change_per_period = (slope / abs(mean_y)) * 100

    if p_value > 0.1 or abs(pct_change_per_period) < 0.5:
        direction = "neutral"
        label = f"Flat (slope={slope:+.3f}, p={p_value:.2f})"
    elif slope > 0:
        direction = "positive"
        label = f"↑ +{pct_change_per_period:.1f}%/period (r²={r_value**2:.2f})"
    else:
        direction = "negative"
        label = f"↓ {pct_change_per_period:.1f}%/period (r²={r_value**2:.2f})"

    return label, direction


# ──────────────────────────────────────────────────────────────────────────────
# Business Intelligence KPI Computation
# ──────────────────────────────────────────────────────────────────────────────

def _detect_business_metrics(df: pd.DataFrame, profiles: Dict) -> Dict[str, Optional[str]]:
    detected = {k: None for k in _BUSINESS_METRIC_KEYWORDS.keys()}

    for col in df.columns:
        col_lower = col.lower().replace("-", "_").replace(" ", "_")
        p = profiles.get(col, {})

        if p.get("role") != "metric":
            continue

        for canonical, keywords in _BUSINESS_METRIC_KEYWORDS.items():
            if detected[canonical] is not None:
                continue

            for kw in keywords:
                if kw in col_lower or col_lower in kw:
                    detected[canonical] = col
                    break

    return detected


def _compute_growth_rate(series: pd.Series, time_col: Optional[pd.Series] = None) -> Tuple[float, str]:
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0, "neutral"

    if len(clean) >= 2:
        mid = len(clean) // 2
        first_half = clean.iloc[:mid].mean()
        second_half = clean.iloc[mid:].mean()

        if first_half == 0:
            return 0.0, "neutral"

        growth = ((second_half - first_half) / abs(first_half)) * 100

        if growth > 5:
            return growth, "positive"
        elif growth < -5:
            return growth, "negative"
        return growth, "neutral"

    return 0.0, "neutral"


def _parse_user_kpi_question(question: str, df: pd.DataFrame, profiles: Dict, detected_metrics: Dict) -> Optional[Dict]:
    q = question.lower().strip()

    if "profit margin" in q or "margin" in q:
        rev_col = detected_metrics.get("revenue")
        cost_col = detected_metrics.get("cost")
        profit_col = detected_metrics.get("profit")

        if profit_col and rev_col:
            profit_sum = float(df[profit_col].sum())
            rev_sum = float(df[rev_col].sum())
            if rev_sum > 0:
                margin = (profit_sum / rev_sum) * 100
                return {
                    "name": "Profit Margin (Requested)",
                    "value_fmt": f"{margin:.1f}%",
                    "formula": f"SUM({profit_col}) / SUM({rev_col}) × 100",
                    "trend": "positive" if margin > 20 else "negative" if margin < 10 else "neutral",
                    "trend_label": "healthy" if margin > 20 else "low" if margin < 10 else "moderate",
                    "category": "user_requested",
                    "priority": 0,
                }
        elif rev_col and cost_col:
            rev_sum = float(df[rev_col].sum())
            cost_sum = float(df[cost_col].sum())
            if rev_sum > 0:
                margin = ((rev_sum - cost_sum) / rev_sum) * 100
                return {
                    "name": "Profit Margin (Requested)",
                    "value_fmt": f"{margin:.1f}%",
                    "formula": "(Revenue - Cost) / Revenue × 100",
                    "trend": "positive" if margin > 20 else "negative" if margin < 10 else "neutral",
                    "trend_label": "healthy" if margin > 20 else "low" if margin < 10 else "moderate",
                    "category": "user_requested",
                    "priority": 0,
                }

    if ("revenue" in q or "sales" in q) and ("growth" in q or "trend" in q or "change" in q):
        rev_col = detected_metrics.get("revenue") or detected_metrics.get("sales")
        if rev_col and rev_col in df.columns:
            growth, direction = _compute_growth_rate(df[rev_col].dropna())
            return {
                "name": "Revenue Growth (Requested)",
                "value_fmt": f"{growth:+.1f}%",
                "formula": "(Current Period - Prior Period) / Prior Period × 100",
                "trend": direction,
                "trend_label": f"{'Growing' if growth > 5 else 'Declining' if growth < -5 else 'Stable'}",
                "category": "user_requested",
                "priority": 0,
            }

    if "average order" in q or "aov" in q or "order value" in q:
        rev_col = detected_metrics.get("revenue")
        orders_col = detected_metrics.get("orders")
        qty_col = detected_metrics.get("quantity")

        if rev_col and orders_col:
            rev_sum = float(df[rev_col].sum())
            orders_sum = float(df[orders_col].sum())
            if orders_sum > 0:
                aov = rev_sum / orders_sum
                return {
                    "name": "Avg Order Value (Requested)",
                    "value_fmt": _fmt(aov),
                    "formula": "Revenue / Orders",
                    "trend": "neutral",
                    "trend_label": "",
                    "category": "user_requested",
                    "priority": 0,
                }
        elif rev_col and qty_col:
            rev_sum = float(df[rev_col].sum())
            qty_sum = float(df[qty_col].sum())
            if qty_sum > 0:
                aov = rev_sum / qty_sum
                return {
                    "name": "Avg Unit Value (Requested)",
                    "value_fmt": _fmt(aov),
                    "formula": "Revenue / Quantity",
                    "trend": "neutral",
                    "trend_label": "",
                    "category": "user_requested",
                    "priority": 0,
                }

    if "profit" in q and "margin" not in q:
        profit_col = detected_metrics.get("profit")
        rev_col = detected_metrics.get("revenue")
        cost_col = detected_metrics.get("cost")

        if profit_col and profit_col in df.columns:
            return {
                "name": "Total Profit (Requested)",
                "value_fmt": _fmt(float(df[profit_col].sum())),
                "formula": f"SUM({profit_col})",
                "trend": "neutral",
                "trend_label": "",
                "category": "user_requested",
                "priority": 0,
            }
        elif rev_col and cost_col:
            profit = float(df[rev_col].sum()) - float(df[cost_col].sum())
            return {
                "name": "Total Profit (Requested)",
                "value_fmt": _fmt(profit),
                "formula": f"SUM({rev_col}) - SUM({cost_col})",
                "trend": "neutral",
                "trend_label": "",
                "category": "user_requested",
                "priority": 0,
            }

    if ("total" in q or "sum" in q) and ("revenue" in q or "sales" in q):
        rev_col = detected_metrics.get("revenue") or detected_metrics.get("sales")
        if rev_col and rev_col in df.columns:
            return {
                "name": "Total Revenue (Requested)",
                "value_fmt": _fmt(float(df[rev_col].sum())),
                "formula": f"SUM({rev_col})",
                "trend": "neutral",
                "trend_label": "",
                "category": "user_requested",
                "priority": 0,
            }

    if "cost" in q or "expense" in q:
        cost_col = detected_metrics.get("cost")
        if cost_col and cost_col in df.columns:
            cost_sum = float(df[cost_col].sum())
            cost_growth, cost_dir = _compute_growth_rate(df[cost_col].dropna())
            return {
                "name": "Total Cost (Requested)",
                "value_fmt": _fmt(cost_sum),
                "formula": f"SUM({cost_col})",
                "trend": cost_dir,
                "trend_label": f"{cost_growth:+.1f}% period change",
                "category": "user_requested",
                "priority": 0,
            }

    if "show" in q or "what" in q:
        for col in df.columns:
            col_lower = col.lower()
            keywords = q.replace("show me", "").replace("what is", "").replace("the", "").split()
            for kw in keywords:
                if kw.strip() in col_lower or col_lower in kw.strip():
                    if profiles.get(col, {}).get("role") == "metric":
                        return {
                            "name": f"{col.replace('_', ' ').title()} (Requested)",
                            "value_fmt": _fmt(float(df[col].sum())),
                            "formula": f"SUM({col})",
                            "trend": "neutral",
                            "trend_label": "",
                            "category": "user_requested",
                            "priority": 0,
                        }

    return None


def _compute_business_kpis(
    df: pd.DataFrame,
    profiles: Dict,
    trend_series: Dict,
    detected_metrics: Dict[str, Optional[str]],
    user_kpi_questions: Optional[List[str]] = None,
) -> List[Dict]:
    kpis = []

    if user_kpi_questions:
        for question in user_kpi_questions:
            user_kpi = _parse_user_kpi_question(question, df, profiles, detected_metrics)
            if user_kpi:
                kpis.append(user_kpi)

    rev_col = detected_metrics.get("revenue")
    cost_col = detected_metrics.get("cost")
    profit_col = detected_metrics.get("profit")
    qty_col = detected_metrics.get("quantity")
    orders_col = detected_metrics.get("orders")
    customers_col = detected_metrics.get("customers")
    margin_col = detected_metrics.get("margin")

    profit_value = None
    if profit_col and profit_col in df.columns:
        profit_value = float(df[profit_col].sum())
        profit_trend_label, profit_trend_dir = _compute_growth_rate(df[profit_col].dropna())
        kpis.append({
            "name": "Total Profit",
            "value_fmt": _fmt(profit_value),
            "formula": f"SUM({profit_col})",
            "trend": profit_trend_dir,
            "trend_label": f"{profit_trend_label:+.1f}% period change",
            "category": "profitability",
            "priority": 1,
        })
    elif rev_col and cost_col:
        profit_series = df[rev_col].fillna(0) - df[cost_col].fillna(0)
        profit_value = float(profit_series.sum())
        profit_trend_label, profit_trend_dir = _compute_growth_rate(profit_series)
        kpis.append({
            "name": "Total Profit",
            "value_fmt": _fmt(profit_value),
            "formula": f"SUM({rev_col}) - SUM({cost_col})",
            "trend": profit_trend_dir,
            "trend_label": f"{profit_trend_label:+.1f}% period change",
            "category": "profitability",
            "priority": 1,
        })

    if margin_col and margin_col in df.columns and margin_col != profit_col:
        # Only use as a standalone margin column if it's actually a pre-computed
        # margin/percentage column, not the same column already used as profit.
        margin_value = float(df[margin_col].mean())
        kpis.append({
            "name": "Average Profit Margin",
            "value_fmt": f"{margin_value:.1f}%",
            "formula": f"AVG({margin_col})",
            "trend": "neutral",
            "trend_label": "",
            "category": "profitability",
            "priority": 2,
        })
    elif rev_col and profit_col and profit_col in df.columns:
        total_revenue = float(df[rev_col].sum())
        total_profit = float(df[profit_col].sum())
        if total_revenue != 0:
            margin_pct = (total_profit / total_revenue) * 100
            kpis.append({
                "name": "Profit Margin",
                "value_fmt": f"{margin_pct:.1f}%",
                "formula": "Profit / Revenue × 100",
                "trend": "positive" if margin_pct > 20 else "negative" if margin_pct < 10 else "neutral",
                "trend_label": "healthy" if margin_pct > 20 else "low margin" if margin_pct < 10 else "moderate",
                "category": "profitability",
                "priority": 2,
            })
    elif rev_col and cost_col:
        total_revenue = float(df[rev_col].sum())
        total_cost = float(df[cost_col].sum())
        if total_revenue != 0:
            margin_pct = ((total_revenue - total_cost) / total_revenue) * 100
            kpis.append({
                "name": "Profit Margin",
                "value_fmt": f"{margin_pct:.1f}%",
                "formula": "(Revenue - Cost) / Revenue × 100",
                "trend": "positive" if margin_pct > 20 else "negative" if margin_pct < 10 else "neutral",
                "trend_label": "healthy" if margin_pct > 20 else "low margin" if margin_pct < 10 else "moderate",
                "category": "profitability",
                "priority": 2,
            })

    if rev_col:
        total_revenue = float(df[rev_col].sum()) if rev_col in df.columns else 0
        rev_growth, rev_growth_dir = _compute_growth_rate(df[rev_col].dropna())

        kpis.append({
            "name": "Total Revenue",
            "value_fmt": _fmt(total_revenue),
            "formula": f"SUM({rev_col})",
            "trend": rev_growth_dir,
            "trend_label": f"{rev_growth:+.1f}% period change",
            "category": "revenue",
            "priority": 3,
        })

        kpis.append({
            "name": "Revenue Growth Rate",
            "value_fmt": f"{rev_growth:+.1f}%",
            "formula": "(Current Period - Prior Period) / Prior Period × 100",
            "trend": rev_growth_dir,
            "trend_label": f"{'Growing' if rev_growth > 0 else 'Declining' if rev_growth < 0 else 'Flat'}",
            "category": "growth",
            "priority": 4,
        })

    if rev_col and orders_col:
        total_rev = float(df[rev_col].sum()) if rev_col in df.columns else 0
        total_orders = float(df[orders_col].sum()) if orders_col in df.columns else 0
        if total_orders > 0:
            aov = total_rev / total_orders
            kpis.append({
                "name": "Avg Order Value (AOV)",
                "value_fmt": _fmt(aov),
                "formula": "Revenue / Orders",
                "trend": "neutral",
                "trend_label": "",
                "category": "efficiency",
                "priority": 5,
            })
    elif rev_col and qty_col:
        total_rev = float(df[rev_col].sum()) if rev_col in df.columns else 0
        total_qty = float(df[qty_col].sum()) if qty_col in df.columns else 0
        if total_qty > 0:
            aov = total_rev / total_qty
            kpis.append({
                "name": "Avg Unit Value",
                "value_fmt": _fmt(aov),
                "formula": "Revenue / Quantity",
                "trend": "neutral",
                "trend_label": "",
                "category": "efficiency",
                "priority": 5,
            })

    if cost_col and cost_col in df.columns:
        total_cost = float(df[cost_col].sum())
        cost_growth, cost_dir = _compute_growth_rate(df[cost_col].dropna())
        kpis.append({
            "name": "Total Cost",
            "value_fmt": _fmt(total_cost),
            "formula": f"SUM({cost_col})",
            "trend": cost_dir,
            "trend_label": f"{cost_growth:+.1f}% period change",
            "category": "cost",
            "priority": 6,
        })

    if customers_col and customers_col in df.columns:
        total_customers = float(df[customers_col].sum())
        cust_growth, cust_dir = _compute_growth_rate(df[customers_col].dropna())
        kpis.append({
            "name": "Total Customers",
            "value_fmt": _fmt(total_customers),
            "formula": f"SUM({customers_col})",
            "trend": cust_dir,
            "trend_label": f"{cust_growth:+.1f}% period change",
            "category": "customers",
            "priority": 7,
        })

        if rev_col:
            total_rev = float(df[rev_col].sum()) if rev_col in df.columns else 0
            if total_customers > 0:
                rev_per_cust = total_rev / total_customers
                kpis.append({
                    "name": "Revenue per Customer",
                    "value_fmt": _fmt(rev_per_cust),
                    "formula": "Revenue / Customers",
                    "trend": "neutral",
                    "trend_label": "",
                    "category": "efficiency",
                    "priority": 8,
                })

    metric_cols = [c for c, p in profiles.items() if p["role"] == "metric"]
    existing_kpi_cols = {kp.get("_col") for kp in kpis if kp.get("_col")}

    for col in metric_cols[:4]:
        if col in existing_kpi_cols:
            continue

        p = profiles[col]
        mp = p.get("metric_profile", {})
        if not mp:
            continue

        total = mp.get("raw_sum", 0)

        trend_label, trend_dir = "N/A", "neutral"
        for ts_name, ts_data in trend_series.items():
            if col in ts_name and ts_data:
                vals = [row.get(col, 0) for row in ts_data if isinstance(row.get(col), (int, float))]
                if len(vals) >= 3:
                    trend_label, trend_dir = detect_trend(vals)
                break

        kpis.append({
            "name": f"Total {col.replace('_', ' ').title()}",
            "value_fmt": _fmt(total),
            "formula": f"SUM({col})",
            "trend": trend_dir,
            "trend_label": trend_label if trend_label != "N/A" else "",
            "category": "other",
            "priority": 10,
            "_col": col,
        })

    null_cols = [
        (c, round(df[c].isna().mean() * 100, 1))
        for c in df.columns
        if df[c].isna().mean() > 0.05
    ]
    if null_cols:
        worst_col, worst_pct = max(null_cols, key=lambda x: x[1])
        kpis.append({
            "name": f"Data Completeness",
            "value_fmt": f"{100 - worst_pct:.1f}%",
            "formula": f"(1 - NULL_RATE({worst_col})) × 100",
            "trend": "negative" if worst_pct > 20 else "neutral",
            "trend_label": f"{worst_pct:.1f}% missing in {worst_col}",
            "category": "quality",
            "priority": 11,
        })

    kpis.append({
        "name": "Total Records",
        "value_fmt": f"{len(df):,}",
        "formula": "COUNT(*)",
        "trend": "neutral",
        "trend_label": "",
        "category": "summary",
        "priority": 12,
    })

    kpis.sort(key=lambda x: (x.get("priority", 99), x["name"]))

    return kpis[:12]


def _compute_segment_intelligence(
    df: pd.DataFrame,
    detected_metrics: Dict,
    dim_cols: List[str],
) -> Dict:
    """
    Detect business-critical segment patterns across ALL segment-worthy
    dimensions, not just the first three raw dim_cols.

    FIX: Uses _segment_worthy_dims() to filter out high-cardinality and
    identifier columns (Row ID, Order ID, Customer ID, etc.) before
    iterating. This ensures Category, Sub-Category, Region, Segment, and
    Ship Mode are all evaluated instead of being skipped because Row ID
    and Order ID consumed the first two slots in the raw dim_cols list.
    """
    rev_col = detected_metrics.get("revenue")
    profit_col = detected_metrics.get("profit")
    cost_col = detected_metrics.get("cost")

    # Derive profit column if not explicitly present
    if not profit_col and rev_col and cost_col:
        df = df.copy()
        df["_derived_profit"] = df[rev_col].fillna(0) - df[cost_col].fillna(0)
        profit_col = "_derived_profit"

    if not (rev_col and profit_col):
        return {}

    results: Dict[str, List] = {
        "loss_making_segments": [],
        "high_revenue_low_profit": [],
        "top_profit_segments": [],
    }

    # FIX: Filter to segment-worthy dims before slicing to top-N
    worthy_dims = _segment_worthy_dims(df, dim_cols)

    # Analyse up to 6 meaningful dimensions (was hardcoded dim_cols[:3])
    for dim in worthy_dims[:6]:
        agg = (
            df.groupby(dim)
            .agg({rev_col: "sum", profit_col: "sum"})
            .reset_index()
        )
        agg["margin"] = agg[profit_col] / agg[rev_col].replace(0, 1)

        # Loss-making segments (profit < 0)
        loss = agg[agg[profit_col] < 0].sort_values(profit_col)
        for _, row in loss.head(5).iterrows():
            results["loss_making_segments"].append({
                "dimension": dim,
                "segment": str(row[dim]),
                "revenue": round(float(row[rev_col]), 2),
                "profit": round(float(row[profit_col]), 2),
                "margin_pct": round(float(row["margin"]) * 100, 1),
            })

        # High revenue but low / negative margin
        revenue_threshold = agg[rev_col].quantile(0.5)   # FIX: lowered from 0.7 → 0.5 (median)
        risky = agg[
            (agg[rev_col] > revenue_threshold)
            & (agg["margin"] < 0.10)          # margin < 10%
        ]
        for _, row in risky.head(5).iterrows():
            results["high_revenue_low_profit"].append({
                "dimension": dim,
                "segment": str(row[dim]),
                "revenue": round(float(row[rev_col]), 2),
                "profit": round(float(row[profit_col]), 2),
                "margin_pct": round(float(row["margin"]) * 100, 1),
            })

        # Top profit drivers
        top = agg.sort_values(profit_col, ascending=False)
        for _, row in top.head(3).iterrows():
            results["top_profit_segments"].append({
                "dimension": dim,
                "segment": str(row[dim]),
                "profit": round(float(row[profit_col]), 2),
                "revenue": round(float(row[rev_col]), 2),
                "margin_pct": round(float(row["margin"]) * 100, 1),
            })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Correlation Matrix
# ──────────────────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame, metric_cols: List[str]) -> List[Dict]:
    if len(metric_cols) < 2:
        return []

    sub = df[metric_cols].dropna()
    if len(sub) < 10:
        return []

    try:
        corr_matrix = sub.corr()
    except Exception:
        return []

    results = []
    seen = set()
    for i, c1 in enumerate(metric_cols):
        for c2 in metric_cols[i + 1:]:
            pair = tuple(sorted([c1, c2]))
            if pair in seen:
                continue
            seen.add(pair)
            r = corr_matrix.loc[c1, c2]
            if abs(r) > 0.5:
                results.append({
                    "col_a": c1,
                    "col_b": c2,
                    "r": round(float(r), 3),
                    "strength": "strong" if abs(r) > 0.8 else "moderate",
                    "direction": "positive" if r > 0 else "negative",
                })

    return sorted(results, key=lambda x: -abs(x["r"]))


# ──────────────────────────────────────────────────────────────────────────────
# Trend Series Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_trend_series(
    df: pd.DataFrame,
    time_cols: List[str],
    metric_cols: List[str],
    max_series: int = 3,
) -> Dict[str, List[Dict]]:
    results = {}
    if not time_cols or not metric_cols:
        return results

    for tc in time_cols[:2]:
        parsed = pd.to_datetime(df[tc], errors="coerce")
        if parsed.notna().sum() < 10:
            continue

        temp = df.copy()
        temp["_dt"] = parsed
        range_days = (parsed.max() - parsed.min()).days

        if range_days <= 31:
            freq, label = "D", "Daily"
        elif range_days <= 365:
            freq, label = "ME", "Monthly"
        else:
            freq, label = "QE", "Quarterly"

        temp = temp.dropna(subset=["_dt"])
        temp = temp.set_index("_dt").sort_index()

        for mc in metric_cols[:max_series]:
            if mc not in temp.columns:
                continue
            try:
                grouped = temp[mc].resample(freq).sum().reset_index()
                grouped.columns = ["period", mc]
                grouped["period"] = grouped["period"].dt.strftime(
                    "%Y-%m-%d" if freq == "D" else "%Y-%m" if freq == "ME" else "%Y-Q%q"
                )
                key = f"{label} {mc.replace('_', ' ').title()} over {tc}"
                results[key] = grouped.to_dict(orient="records")
            except Exception:
                continue

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, profiles: Dict) -> List[Dict]:
    anomalies = []
    for col, p in profiles.items():
        null_pct = p.get("null_pct", 0)
        if null_pct > 20:
            anomalies.append({
                "column": col,
                "type": "high_null_rate",
                "description": f"{null_pct:.1f}% of values are null",
            })

        mp = p.get("metric_profile", {})
        if mp.get("has_outliers") and mp.get("outlier_count", 0) > 0:
            pct = round(mp["outlier_count"] / max(len(df), 1) * 100, 1)
            if pct > 1:
                anomalies.append({
                    "column": col,
                    "type": "outliers",
                    "description": f"{mp['outlier_count']:,} outliers ({pct}% of rows) detected via IQR fence",
                })

        skew = mp.get("skewness")
        if skew and abs(skew) > 2:
            anomalies.append({
                "column": col,
                "type": "high_skewness",
                "description": f"Skewness = {skew:.2f} — consider log transform before modeling",
            })

    return anomalies


# ──────────────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────────────

class AnalyticsEngine:
    """
    Deterministic analytics engine.
    All computations happen here. The LLM receives the output, not raw data.
    """

    def __init__(self, df: pd.DataFrame, user_kpi_questions: Optional[List[str]] = None):
        self.df = df.copy()
        self.user_kpi_questions = user_kpi_questions or []

    def run(self) -> Dict:
        df = self.df
        profiles: Dict[str, Dict] = {}

        # ── 1. Classify & profile each column ────────────────────────────────
        for col in df.columns:
            role = classify_column(df[col], col)
            null_pct = round(df[col].isna().mean() * 100, 2)

            p: Dict[str, Any] = {
                "role": role,
                "dtype": str(df[col].dtype),
                "null_pct": null_pct,
            }

            if role == "metric":
                mp = profile_metric_column(df[col])
                p.update(mp)
                p["metric_profile"] = mp

                clean_vals = df[col].dropna().tolist()
                if len(clean_vals) >= 10:
                    trend_label, trend_dir = detect_trend(clean_vals)
                    p["trend"] = trend_dir
                    p["trend_label"] = trend_label

            elif role == "dimension":
                dp = profile_dimension_column(df[col])
                p.update(dp)

            elif role == "time":
                tp = profile_time_column(df[col])
                p.update(tp)
                p.pop("_parsed", None)

            profiles[col] = p

        # ── 2. Classify columns by role ───────────────────────────────────────
        time_cols = [c for c, p in profiles.items() if p["role"] == "time"]
        metric_cols = [c for c, p in profiles.items() if p["role"] == "metric"]
        dim_cols = [c for c, p in profiles.items() if p["role"] == "dimension"]

        # ── 3. Trend series (time × metric) ───────────────────────────────────
        trend_series = compute_trend_series(df, time_cols, metric_cols)

        # ── 4. Business metric detection & KPIs ───────────────────────────────
        detected_metrics = _detect_business_metrics(df, profiles)

        kpis = _compute_business_kpis(
            df, profiles, trend_series, detected_metrics, self.user_kpi_questions
        )

        # ── 5. Segment intelligence ───────────────────────────────────────────
        # FIX: Pass full dim_cols list — _compute_segment_intelligence now
        # internally filters to segment-worthy dims via _segment_worthy_dims().
        segment_insights = _compute_segment_intelligence(df, detected_metrics, dim_cols)

        # ── 6. Correlations ────────────────────────────────────────────────────
        correlations = compute_correlations(df, metric_cols)

        # ── 7. Anomalies ───────────────────────────────────────────────────────
        anomalies = detect_anomalies(df, profiles)

        # ── 8. Group-by aggregations ───────────────────────────────────────────
        # FIX: Use _segment_worthy_dims() here too so groupby summaries cover
        # Category / Sub-Category / Region rather than Row ID / Order ID.
        groupby_summaries = {}
        worthy_dims = _segment_worthy_dims(df, dim_cols)
        if worthy_dims and metric_cols:
            for dc in worthy_dims[:6]:
                for mc in metric_cols[:2]:
                    key = f"{mc}_by_{dc}"
                    try:
                        agg = (
                            df.groupby(dc)[mc]
                            .agg(["sum", "mean", "count"])
                            .round(2)
                            .reset_index()
                            .sort_values("sum", ascending=False)
                            .head(20)
                        )
                        groupby_summaries[key] = agg.to_dict(orient="records")
                    except Exception:
                        pass

        return {
            "segment_insights": segment_insights,
            "column_profiles": profiles,
            "time_cols": time_cols,
            "metric_cols": metric_cols,
            "dim_cols": dim_cols,
            "trend_series": trend_series,
            "kpis": kpis,
            "correlations": correlations,
            "anomalies": anomalies,
            "groupby_summaries": groupby_summaries,
            "row_count": len(df),
            "col_count": len(df.columns),
            "detected_business_metrics": detected_metrics,
        }