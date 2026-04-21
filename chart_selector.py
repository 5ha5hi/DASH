"""
chart_selector.py — Layer 3: Data-Driven Chart Selection
=========================================================
Charts are chosen based on actual data characteristics:
  - cardinality
  - data type
  - distribution shape (skew, outliers)
  - whether a time dimension exists
  - correlation strength

NOT chosen by LLM guessing or column-name heuristics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def select_charts(analytics: Dict, df: pd.DataFrame) -> List[Dict]:
    """
    Return a list of chart spec dicts ready for Plotly rendering.
    Each spec has: type, title, rationale, and data-specific fields.
    """
    specs: List[Dict] = []

    profiles = analytics.get("column_profiles", {})
    metric_cols = analytics.get("metric_cols", [])
    dim_cols = analytics.get("dim_cols", [])
    time_cols = analytics.get("time_cols", [])
    groupby_summaries = analytics.get("groupby_summaries", {})
    correlations = analytics.get("correlations", [])
    trend_series = analytics.get("trend_series", {})

    # ── 1. Time series line charts ─────────────────────────────────────────────
    for ts_name, ts_data in trend_series.items():
        if ts_data and len(ts_data) > 2:
            ts_df = pd.DataFrame(ts_data)
            if ts_df.shape[1] >= 2:
                x_col = ts_df.columns[0]
                y_cols = ts_df.columns[1:].tolist()
                specs.append({
                    "type": "line",
                    "title": ts_name,
                    "data": ts_df,
                    "x": x_col,
                    "y": y_cols[0] if len(y_cols) == 1 else y_cols,
                    "rationale": "Time-bucketed trend of a metric column.",
                })

    # ── 2. Bar charts for group-by summaries ──────────────────────────────────
    for key, rows in groupby_summaries.items():
        if not rows:
            continue
        parts = key.split("_by_")
        if len(parts) != 2:
            continue
        mc, dc = parts[0], parts[1]
        plot_df = pd.DataFrame(rows)[[dc, "sum"]].rename(columns={"sum": f"Sum of {mc}"})
        specs.append({
            "type": "bar",
            "title": f"Sum of {mc.replace('_', ' ').title()} by {dc.replace('_', ' ').title()}",
            "data": plot_df,
            "x": dc,
            "y": f"Sum of {mc}",
            "rationale": f"Group-by aggregation shows how {mc} distributes across {dc} categories.",
        })

        # Pie if low cardinality
        n_cat = len(rows)
        if 2 <= n_cat <= 12:
            specs.append({
                "type": "pie",
                "title": f"Share of {mc.replace('_', ' ').title()} by {dc.replace('_', ' ').title()}",
                "data": plot_df,
                "names": dc,
                "values": f"Sum of {mc}",
                "rationale": f"Proportional breakdown of {mc} across {n_cat} categories.",
            })

    # ── 3. Histograms for metric columns ──────────────────────────────────────
    for mc in metric_cols[:4]:
        if mc not in df.columns:
            continue
        clean = df[mc].dropna()
        if len(clean) < 10:
            continue

        p = profiles.get(mc, {})
        skew = p.get("skewness", 0) or 0
        has_outliers = p.get("has_outliers", False)

        # Choose bin count by IQR rule (Freedman-Diaconis)
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            bin_width = 2 * iqr / (len(clean) ** (1 / 3))
            data_range = clean.max() - clean.min()
            bins = min(max(int(data_range / bin_width), 10), 80) if bin_width > 0 else 30
        else:
            bins = 30

        rationale = f"Distribution of {mc}."
        if abs(skew) > 1:
            rationale += f" Skew={skew:.2f} — right-skewed data."
        if has_outliers:
            rationale += f" {p.get('outlier_count', '')} outliers present."

        specs.append({
            "type": "histogram",
            "title": f"Distribution of {mc.replace('_', ' ').title()}",
            "col": mc,
            "bins": bins,
            "rationale": rationale,
        })

        # Box plot — especially useful when outliers detected
        if has_outliers or abs(skew) > 0.5:
            # Box by first dim col if available
            by_col: Optional[str] = None
            for dc in dim_cols:
                if dc in df.columns and df[dc].nunique() <= 15:
                    by_col = dc
                    break
            specs.append({
                "type": "box",
                "title": f"{mc.replace('_', ' ').title()} Distribution"
                + (f" by {by_col}" if by_col else ""),
                "col": mc,
                "by": by_col,
                "rationale": "Box plot surfaces median, IQR, and outliers more clearly than histogram.",
            })

    # ── 4. Scatter plots for correlated pairs ─────────────────────────────────
    for corr in correlations[:3]:
        c1, c2 = corr["col_a"], corr["col_b"]
        if c1 not in df.columns or c2 not in df.columns:
            continue
        specs.append({
            "type": "scatter",
            "title": f"{c1.replace('_',' ').title()} vs {c2.replace('_',' ').title()} (r={corr['r']})",
            "x": c1,
            "y": c2,
            "trendline": True,
            "rationale": f"{corr['strength'].title()} {corr['direction']} correlation (r={corr['r']}).",
        })

    # ── 5. Correlation heatmap if 4+ metric cols ──────────────────────────────
    if len(metric_cols) >= 4:
        sub = df[metric_cols[:10]].dropna()
        if len(sub) >= 10:
            try:
                corr_mat = sub.corr()
                specs.append({
                    "type": "heatmap",
                    "title": "Metric Correlation Matrix",
                    "z": corr_mat.values.tolist(),
                    "x": corr_mat.columns.tolist(),
                    "y": corr_mat.index.tolist(),
                    "rationale": "Heatmap of all pairwise Pearson correlations among numeric columns.",
                })
            except Exception:
                pass

    # ── 6. Value count bars for low-cardinality dimensions ────────────────────
    for dc in dim_cols[:3]:
        if dc not in df.columns:
            continue
        n_unique = df[dc].nunique()
        if not (2 <= n_unique <= 30):
            continue
        # Skip if we already generated a group-by chart for this dim
        already = any(s.get("x") == dc for s in specs if s["type"] == "bar")
        if already:
            continue

        vc = df[dc].value_counts().head(20).reset_index()
        vc.columns = [dc, "count"]
        specs.append({
            "type": "bar",
            "title": f"Record Count by {dc.replace('_', ' ').title()}",
            "data": vc,
            "x": dc,
            "y": "count",
            "rationale": f"Frequency distribution of {n_unique} categories in {dc}.",
        })

    return specs