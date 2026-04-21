"""
AI Analytics Platform — Data-First Architecture
================================================
Architecture layers:
  1. data_ingestion.py   — CSV / MySQL loading, schema normalization
  2. analytics_engine.py — Deterministic profiling, KPIs, trends, correlations
  3. chart_selector.py   — Data-driven chart recommendation
  4. ai_layer.py         — LLM used ONLY for explanation, not computation
  5. safe_query.py       — Constrained query executor (no exec())
  6. app.py              — Streamlit UI (this file)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List
import os

# ── Local modules ────────────────────────────────────────────────────────────
from data_ingestion import load_csv, connect_mysql, load_table, get_db_schema
from analytics_engine import AnalyticsEngine
from chart_selector import select_charts
from ai_layer import AIExplainer
from safe_query import SafeQueryExecutor

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics Platform",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2d3d;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #7dd3fc; letter-spacing: -0.02em; }
h4, h5 { color: #94a3b8; }

.metric-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 3px solid #38bdf8;
}
.metric-card.negative { border-left-color: #f87171; }
.metric-card.neutral  { border-left-color: #94a3b8; }

.insight-pill {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid #38bdf8;
    color: #7dd3fc;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}
.anomaly-pill {
    background: rgba(248,113,113,0.1);
    border-color: #f87171;
    color: #fca5a5;
}
.profile-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.profile-stat {
    background: #1e293b;
    border: 1px solid #2d3f55;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.profile-stat b { color: #38bdf8; }
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #1e2d3d;
    margin-bottom: 16px;
}
.badge {
    background: #0c4a6e;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
}
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
[data-testid="stDataFrame"] { border: 1px solid #1e2d3d; border-radius: 8px; }
hr { border-color: #1e2d3d !important; }

/* Hide Streamlit components for privacy */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stAppDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
DEFAULTS = {
    "df": None,
    "original_df": None,
    "analytics": None,       # AnalyticsEngine results (deterministic)
    "ai_explanations": None, # AI narration of computed analytics
    "data_source": "CSV",
    "db_conn": None,
    "db_schema": None,
    "active_table": None,
    "file_name": None,
    "query_result": None,
    "query_history": [],
    "last_query": "",
    "user_kpi_questions": [],  # User-defined KPI questions
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def run_analytics(df: pd.DataFrame, user_kpi_questions: Optional[List[str]] = None):
    """Runs the full deterministic analytics pipeline and caches results."""
    engine = AnalyticsEngine(df, user_kpi_questions)
    st.session_state.analytics = engine.run()
    st.session_state.ai_explanations = None  # invalidate stale AI text


def trend_icon(trend: str) -> str:
    return {"positive": "↑", "negative": "↓", "neutral": "→"}.get(trend, "→")


def trend_color(trend: str) -> str:
    return {"positive": "#4ade80", "negative": "#f87171", "neutral": "#94a3b8"}.get(trend, "#94a3b8")


def kpi_category_color(category: str) -> str:
    """Return border color for KPI category."""
    colors = {
        "profitability": "#fbbf24",  # amber
        "revenue": "#22c55e",  # green
        "growth": "#3b82f6",  # blue
        "efficiency": "#a855f7",  # purple
        "cost": "#f97316",  # orange
        "customers": "#ec4899",  # pink
        "user_requested": "#06b6d4",  # cyan
        "quality": "#64748b",  # slate
        "summary": "#94a3b8",  # gray
    }
    return colors.get(category, "#38bdf8")  # default sky blue


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## `📐 Analytics Platform`")
    st.markdown("---")

    MODEL = os.getenv("OPENROUTER_MODEL", "unknown")
    st.write(f"🔧 Model: {MODEL}")
    st.session_state.data_source = st.radio("Data Source", ["CSV", "Database"])

    if st.session_state.data_source == "CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded and uploaded.name != st.session_state.file_name:
            df, err = load_csv(uploaded)
            if err:
                st.error(err)
            else:
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.file_name = uploaded.name
                st.session_state.analytics = None
                st.session_state.ai_explanations = None
                st.session_state.query_result = None
                st.session_state.query_history = []
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
    else:
        st.markdown("#### MySQL Connection")
        host = st.text_input("Host", "localhost")
        port = st.number_input("Port", value=3306, step=1)
        user = st.text_input("User")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database")

        if st.button("Connect"):
            conn, err = connect_mysql(host, int(port), user, password, database)
            if err:
                st.error(err)
            else:
                st.session_state.db_conn = conn
                st.session_state.db_schema = get_db_schema(conn, database)
                st.success("Connected ✓")

        if st.session_state.db_schema:
            tables = sorted(st.session_state.db_schema.get("tables", {}).keys())
            sel = st.selectbox("Table", ["— select —"] + tables)
            if sel != "— select —" and sel != st.session_state.active_table:
                df, err = load_table(st.session_state.db_conn, sel)
                if err:
                    st.error(err)
                else:
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.active_table = sel
                    st.session_state.analytics = None
                    st.session_state.ai_explanations = None
                    st.session_state.query_result = None
                    st.session_state.query_history = []
                    st.success(f"Loaded {len(df):,} rows")

    st.markdown("---")
    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown(f"""
        <div class='profile-stat'>Rows: <b>{len(df_info):,}</b></div>
        <div class='profile-stat'>Columns: <b>{len(df_info.columns)}</b></div>
        """, unsafe_allow_html=True)

        if st.button("↺ Reset to Original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.analytics = None
            st.session_state.ai_explanations = None
            st.session_state.query_result = None
            st.rerun()

        csv_bytes = st.session_state.df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv_bytes, "data_export.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# GUARD: no data
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("## `📐 Analytics Platform`")
    st.info("Upload a CSV or connect to a database to begin.")
    st.stop()

df: pd.DataFrame = st.session_state.df

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
src = st.session_state.active_table or st.session_state.file_name or "Dataset"
st.markdown(f"## `📐 {src}`")
cols_h = st.columns(4)
cols_h[0].metric("Rows", f"{len(df):,}")
cols_h[1].metric("Columns", len(df.columns))
cols_h[2].metric("Numeric cols", len(df.select_dtypes("number").columns))
cols_h[3].metric(
    "Memory",
    f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    if df.memory_usage(deep=True).sum() > 1024 * 1024
    else f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_profile, tab_analytics, tab_dashboard, tab_query = st.tabs([
    "🔬 Data Profile", "📊 Analytics & KPIs", "📈 Dashboard", "🔍 Query"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown("### Column Profiles")
    st.caption("Computed deterministically from your data — no LLM guessing.")

    if st.session_state.analytics is None:
        if st.button("▶ Run Full Analysis", type="primary"):
            with st.spinner("Profiling data…"):
                run_analytics(df, st.session_state.user_kpi_questions)
            st.rerun()
        # Show basic preview even without full analysis
        st.dataframe(df.head(10), use_container_width=True)
    else:
        a = st.session_state.analytics
        profiles = a["column_profiles"]

        # Summary row
        time_cols   = [c for c, p in profiles.items() if p["role"] == "time"]
        metric_cols = [c for c, p in profiles.items() if p["role"] == "metric"]
        dim_cols    = [c for c, p in profiles.items() if p["role"] == "dimension"]

        r1, r2, r3 = st.columns(3)
        r1.markdown(f"**Time columns** ({len(time_cols)}): " + ", ".join(f"`{c}`" for c in time_cols) or "none")
        r2.markdown(f"**Metric columns** ({len(metric_cols)}): " + ", ".join(f"`{c}`" for c in metric_cols[:8]) or "none")
        r3.markdown(f"**Dimension columns** ({len(dim_cols)}): " + ", ".join(f"`{c}`" for c in dim_cols[:8]) or "none")

        st.divider()

        for col, p in profiles.items():
            role_badge = {"time": "🕐 TIME", "metric": "📏 METRIC", "dimension": "🏷 DIMENSION"}.get(p["role"], "❓")
            with st.expander(f"`{col}` — {role_badge}  ({p['dtype']})", expanded=False):
                pills_html = ""
                if p.get("null_pct", 0) > 5:
                    pills_html += f'<span class="insight-pill anomaly-pill">⚠ {p["null_pct"]:.1f}% missing</span>'
                if p.get("has_outliers"):
                    pills_html += f'<span class="insight-pill anomaly-pill">⚠ outliers detected</span>'
                if p.get("skewness") and abs(p.get("skewness", 0)) > 1:
                    pills_html += f'<span class="insight-pill">skew {p["skewness"]:.2f}</span>'
                if p.get("trend"):
                    pills_html += f'<span class="insight-pill">{trend_icon(p["trend"])} trend {p["trend"]}</span>'
                if pills_html:
                    st.markdown(pills_html, unsafe_allow_html=True)

                stats = p.get("stats", {})
                if stats:
                    stat_html = '<div class="profile-row">'
                    for k, v in stats.items():
                        stat_html += f'<div class="profile-stat">{k}: <b>{v}</b></div>'
                    stat_html += "</div>"
                    st.markdown(stat_html, unsafe_allow_html=True)

                if p.get("top_values"):
                    st.markdown("**Top values:**")
                    tv_df = pd.DataFrame(p["top_values"], columns=["Value", "Count", "Pct%"])
                    st.dataframe(tv_df, use_container_width=True, hide_index=True)

        # Correlations
        corr = a.get("correlations")
        if corr is not None and len(corr) > 0:
            st.divider()
            st.markdown("### 🔗 Strong Correlations (|r| > 0.5)")
            st.caption("Computed via Pearson correlation on complete cases.")
            corr_df = pd.DataFrame(corr)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS & KPIs
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    if st.session_state.analytics is None:
        st.info("Run the analysis first (Data Profile tab).")
        if st.button("▶ Run Analysis", type="primary", key="run_from_kpi"):
            with st.spinner("Profiling data…"):
                run_analytics(df, st.session_state.user_kpi_questions)
            st.rerun()
        st.stop()

    a = st.session_state.analytics
    kpis = a.get("kpis", [])
    segment_insights = a.get("segment_insights", {})
    detected_metrics = a.get("detected_business_metrics", {})

    # ── User KPI Questions ────────────────────────────────────────────────────
    st.markdown("### 🎯 Custom KPI Questions")
    st.caption("Ask specific business questions — the system will compute relevant KPIs.")

    # Initialize session state for KPI questions
    if "user_kpi_questions" not in st.session_state:
        st.session_state.user_kpi_questions = []

    # Input for new questions
    kpi_question_input = st.text_input(
        "Add a KPI question (e.g., 'What is the profit margin?', 'Show revenue growth')",
        key="kpi_question_input",
        placeholder="e.g., What is the profit margin? | Show me revenue growth | Calculate AOV"
    )

    col_add, col_clear = st.columns([1, 4])
    with col_add:
        if st.button("Add Question", type="primary", key="add_kpi_question"):
            if kpi_question_input.strip():
                st.session_state.user_kpi_questions.append(kpi_question_input.strip())
                st.session_state.analytics = None  # Trigger re-analysis
                st.rerun()
    with col_clear:
        if st.button("Clear Questions", key="clear_kpi_questions"):
            st.session_state.user_kpi_questions = []
            st.session_state.analytics = None
            st.rerun()

    # Display active questions
    if st.session_state.user_kpi_questions:
        st.markdown("**Active questions:**")
        for i, q in enumerate(st.session_state.user_kpi_questions):
            st.markdown(f'<span class="insight-pill">📊 {q}</span>', unsafe_allow_html=True)

    # ── Detected Business Metrics Summary ─────────────────────────────────────
    if detected_metrics and any(v for v in detected_metrics.values()):
        st.divider()
        st.markdown("### 📊 Detected Business Metrics")
        st.caption("Automatically identified from your data — used to compute derived KPIs.")

        detected_display = {k: v for k, v in detected_metrics.items() if v}
        if detected_display:
            cols_det = st.columns(len(detected_display))
            for idx, (canonical, col_name) in enumerate(detected_display.items()):
                with cols_det[idx % len(cols_det)]:
                    st.markdown(f"""
                    <div style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:8px 12px;margin-bottom:8px">
                        <div style="font-size:11px;color:#64748b;font-family:'IBM Plex Mono',monospace">{canonical.upper()}</div>
                        <div style="font-size:14px;color:#7dd3fc;font-family:'IBM Plex Mono',monospace">{col_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Computed KPI cards ────────────────────────────────────────────────────
    # ── Segment Intelligence (NEW) ────────────────────────────────────────────
    if segment_insights:
        st.divider()
        st.markdown("### 🚨 Segment Intelligence")
        st.caption("Automatically detected high-risk and high-opportunity segments.")

        # Loss-making segments
        loss_segments = segment_insights.get("loss_making_segments", [])
        if loss_segments:
            st.markdown("#### 🔴 Loss-Making Segments")
            for seg in loss_segments:
                st.markdown(
                    f"""
                    <div class="metric-card negative">
                        <b>{seg['dimension']}:</b> {seg['segment']}<br>
                        Revenue: ${seg['revenue']:,.0f} |
                        Profit: ${seg['profit']:,.0f} |
                        Margin: {seg['margin_pct']}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # High revenue low profit
        risky_segments = segment_insights.get("high_revenue_low_profit", [])
        if risky_segments:
            st.markdown("#### ⚠️ High Revenue, Low Profit")
            for seg in risky_segments:
                st.markdown(
                    f"""
                    <div class="metric-card neutral">
                        <b>{seg['dimension']}:</b> {seg['segment']}<br>
                        Revenue: ${seg['revenue']:,.0f} |
                        Profit: ${seg['profit']:,.0f} |
                        Margin: {seg['margin_pct']}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Top profit segments
        top_segments = segment_insights.get("top_profit_segments", [])
        if top_segments:
            st.markdown("#### 🟢 Top Profit Drivers")
            for seg in top_segments:
                st.markdown(
                    f"""
                    <div class="metric-card positive">
                        <b>{seg['dimension']}:</b> {seg['segment']}<br>
                        Profit: ${seg['profit']:,.0f} |
                        Revenue: ${seg['revenue']:,.0f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    st.divider()
    st.markdown("### 📌 Business KPIs")
    st.caption("All values calculated directly from your data. Prioritized by business impact.")

    if not kpis:
        st.info("No computable KPIs found for this dataset shape.")
    else:
        # Group KPIs by category
        kpi_by_category = {}
        for kpi in kpis:
            cat = kpi.get("category", "other")
            if cat not in kpi_by_category:
                kpi_by_category[cat] = []
            kpi_by_category[cat].append(kpi)

        # Category display order
        category_order = ["profitability", "revenue", "growth", "efficiency", "cost", "customers", "user_requested", "quality", "summary"]
        category_titles = {
            "profitability": "💰 Profitability",
            "revenue": "💵 Revenue",
            "growth": "📈 Growth",
            "efficiency": "⚡ Efficiency",
            "cost": "💸 Cost",
            "customers": "👥 Customers",
            "user_requested": "📋 Custom KPIs",
            "quality": "🔍 Data Quality",
            "summary": "📊 Summary",
        }

        for cat in category_order:
            if cat not in kpi_by_category:
                continue

            cat_kpis = kpi_by_category[cat]
            st.markdown(f"#### {category_titles.get(cat, cat.title())}")

            kpi_cols = st.columns(min(len(cat_kpis), 3))
            for i, kpi in enumerate(cat_kpis):
                trend = kpi.get("trend", "neutral")
                category = kpi.get("category", "other")
                border_color = kpi_category_color(category)

                card_class = "metric-card positive" if trend == "positive" else \
                             "metric-card negative" if trend == "negative" else "metric-card neutral"

                icon = trend_icon(trend)
                trend_val_color = trend_color(trend)

                with kpi_cols[i % min(len(cat_kpis), 3)]:
                    st.markdown(f"""
                    <div class="{card_class}" style="border-left-color: {border_color}">
                        <div style="font-size:11px;color:#64748b;font-family:'IBM Plex Mono',monospace;margin-bottom:4px">
                            {kpi['name']}
                        </div>
                        <div style="font-size:22px;font-weight:700;color:#f1f5f9">{kpi['value_fmt']}</div>
                        <div style="font-size:12px;color:{trend_val_color};margin-top:4px">{icon} {kpi.get('trend_label','')}</div>
                        <div style="font-size:10px;color:#475569;margin-top:6px;font-family:'IBM Plex Mono',monospace">{kpi.get('formula','')}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Trend series ─────────────────────────────────────────────────────────
    trend_series = a.get("trend_series", {})
    if trend_series:
        st.divider()
        st.markdown("### 📉 Trend Analysis")
        st.caption("Time-bucketed aggregations computed from your time columns.")
        for ts_name, ts_data in trend_series.items():
            if ts_data and len(ts_data) > 1:
                ts_df = pd.DataFrame(ts_data)
                if ts_df.shape[1] >= 2:
                    fig = px.line(
                        ts_df,
                        x=ts_df.columns[0],
                        y=ts_df.columns[1:].tolist(),
                        title=ts_name,
                        template="plotly_dark",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.8)",
                        height=320,
                        margin=dict(t=40, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Anomalies ─────────────────────────────────────────────────────────────
    anomalies = a.get("anomalies", [])
    if anomalies:
        st.divider()
        st.markdown("### ⚠️ Anomalies Detected")
        for anom in anomalies:
            st.markdown(f"""
            <span class="insight-pill anomaly-pill">⚠ {anom['column']}</span>
            <span style="color:#94a3b8;font-size:13px"> — {anom['description']}</span>
            """, unsafe_allow_html=True)

    # ── AI Narration ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🤖 AI Business Narrative")
    st.caption("The LLM receives pre-computed facts and explains their business implications only.")

    context_hint = st.text_input(
        "Optional: describe your business context (e.g. 'SaaS subscription data, monthly')",
        key="context_hint",
        placeholder="e.g. e-commerce order data, USD, India market"
    )

    if st.button("Generate AI Narrative", type="primary"):
        explainer = AIExplainer()
        with st.spinner("Sending computed analytics to LLM for narration…"):
            narration = explainer.narrate(a, context=context_hint)
        st.session_state.ai_explanations = narration
        st.rerun()

    if st.session_state.ai_explanations:
        exp = st.session_state.ai_explanations
        
        bc = exp.get("business_context", "")
        if bc:
            st.info(f"**Business Context:** {bc}")

        cols_ai = st.columns([1, 1])
        with cols_ai[0]:
            st.markdown("#### 🔍 Insights")
            for ins in exp.get("insights", []):
                st.markdown(f"- {ins}")
        with cols_ai[1]:
            st.markdown("#### 📋 Recommendations")
            for rec in exp.get("recommendations", []):
                st.markdown(f"- {rec}")

        watch = exp.get("watch_list", [])
        if watch:
            st.markdown("#### 📡 Watch List — Track These Weekly")
            for w in watch:
                st.markdown(
                    f'<span class="insight-pill">📡 {w}</span>',
                    unsafe_allow_html=True,
                )

        flags = exp.get("data_quality_flags", [])
        if flags:
            st.markdown("#### ⚠️ Data Quality Flags")
            for f in flags:
                st.markdown(
                    f'<span class="insight-pill anomaly-pill">⚠ {f}</span>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    if st.session_state.analytics is None:
        st.info("Run the analysis first (Data Profile tab).")
        st.stop()

    a = st.session_state.analytics
    st.markdown("### 📈 Recommended Charts")
    st.caption("Charts are selected by data characteristics (cardinality, dtype, distribution) — not guesswork.")

    chart_specs = select_charts(a, df)
    if not chart_specs:
        st.info("No chart recommendations for this dataset shape.")
    else:
        chart_cols = st.columns(2)
        for idx, spec in enumerate(chart_specs[:12]):
            with chart_cols[idx % 2]:
                try:
                    fig = None
                    kind = spec["type"]
                    title = spec["title"]

                    if kind == "bar":
                        plot_df = spec["data"]
                        fig = px.bar(plot_df, x=spec["x"], y=spec["y"], title=title,
                                     template="plotly_dark", text_auto=True)
                    elif kind == "pie":
                        plot_df = spec["data"]
                        fig = px.pie(plot_df, names=spec["names"], values=spec["values"],
                                     title=title, template="plotly_dark", hole=0.35)
                    elif kind == "histogram":
                        fig = px.histogram(df[spec["col"]].dropna(), title=title,
                                           template="plotly_dark", nbins=spec.get("bins", 30))
                    elif kind == "box":
                        if spec.get("by"):
                            fig = px.box(df, x=spec["by"], y=spec["col"],
                                         title=title, template="plotly_dark")
                        else:
                            fig = px.box(df, y=spec["col"], title=title, template="plotly_dark")
                    elif kind == "scatter":
                        fig = px.scatter(df, x=spec["x"], y=spec["y"],
                                         title=title, template="plotly_dark",
                                         trendline="ols" if spec.get("trendline") else None)
                    elif kind == "line":
                        plot_df = spec["data"]
                        fig = px.line(plot_df, x=spec["x"], y=spec["y"],
                                      title=title, template="plotly_dark", markers=True)
                    elif kind == "heatmap":
                        fig = go.Figure(data=go.Heatmap(
                            z=spec["z"], x=spec["x"], y=spec["y"],
                            colorscale="Blues",
                        ))
                        fig.update_layout(title=title, template="plotly_dark")

                    if fig:
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15,23,42,0.8)",
                            height=370,
                            margin=dict(t=45, b=20, l=20, r=20),
                            font=dict(family="IBM Plex Mono", color="#94a3b8"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        if spec.get("rationale"):
                            st.caption(f"💡 {spec['rationale']}")
                except Exception as e:
                    st.warning(f"Could not render chart: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUERY
# ══════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.markdown("### 🔍 Safe Data Query")
    st.caption(
        "Queries are executed via a **constrained, validated executor** — no arbitrary code or SQL injection. "
        "Supported: filter, group, sort, aggregate, describe."
    )

    if st.session_state.analytics:
        a = st.session_state.analytics
        profiles = a.get("column_profiles", {})
        dim_cols = [c for c, p in profiles.items() if p["role"] == "dimension"]
        metric_cols_q = [c for c, p in profiles.items() if p["role"] == "metric"]
        time_cols_q = [c for c, p in profiles.items() if p["role"] == "time"]

        st.markdown("#### Quick Queries")
        qcols = st.columns(3)
        quick_queries = []
        if metric_cols_q:
            quick_queries.append(f"describe {metric_cols_q[0]}")
        if dim_cols and metric_cols_q:
            quick_queries.append(f"group by {dim_cols[0]} sum {metric_cols_q[0]}")
            quick_queries.append(f"top 10 by {metric_cols_q[0]}")
        if time_cols_q and metric_cols_q:
            quick_queries.append(f"trend {metric_cols_q[0]} by month")
        if dim_cols:
            quick_queries.append(f"value counts {dim_cols[0]}")
        if len(metric_cols_q) >= 2:
            quick_queries.append(f"correlate {metric_cols_q[0]} {metric_cols_q[1]}")

        for i, qtext in enumerate(quick_queries[:6]):
            with qcols[i % 3]:
                if st.button(f"`{qtext}`", key=f"qq_{i}", use_container_width=True):
                    st.session_state.last_query = qtext
                    st.session_state.query_result = None
                    st.rerun()

    st.divider()

    query_text = st.text_input(
        "Enter query:",
        value=st.session_state.last_query,
        placeholder="e.g.  group by department sum salary  |  top 5 by revenue  |  filter status = Active",
        key="query_input"
    )

    col_run, col_help = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶ Execute", type="primary")
    with col_help:
        with st.expander("Query syntax reference"):
            st.markdown("""
| Pattern | Example |
|---|---|
| `describe <col>` | `describe salary` |
| `top N by <col>` | `top 10 by revenue` |
| `bottom N by <col>` | `bottom 5 by age` |
| `group by <col> [agg] <col>` | `group by dept sum salary` |
| `filter <col> = <val>` | `filter status = Active` |
| `filter <col> > <num>` | `filter age > 30` |
| `value counts <col>` | `value counts category` |
| `correlate <col1> <col2>` | `correlate price quantity` |
| `trend <col> by [day/month/year]` | `trend revenue by month` |
            """)

    if run_btn and query_text.strip():
        st.session_state.last_query = query_text
        executor = SafeQueryExecutor(df)
        result, error = executor.execute(query_text)
        if error:
            st.error(f"Query error: {error}")
            st.session_state.query_result = None
        else:
            st.session_state.query_result = result
            st.session_state.query_history.append(query_text)

    if st.session_state.query_result is not None:
        qr = st.session_state.query_result
        st.divider()
        st.markdown("#### Result")

        if isinstance(qr, pd.DataFrame):
            st.dataframe(qr, use_container_width=True)
            csv = qr.to_csv(index=False).encode()
            st.download_button("⬇ Download result", csv, "query_result.csv", "text/csv")

            # Auto-visualize if it looks like a group-by result
            if len(qr) <= 100 and len(qr.columns) == 2:
                x_c, y_c = qr.columns[0], qr.columns[1]
                if pd.api.types.is_numeric_dtype(qr[y_c]):
                    fig = px.bar(qr, x=x_c, y=y_c, template="plotly_dark", title=f"{y_c} by {x_c}")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15,23,42,0.8)",
                        height=360,
                        margin=dict(t=40, b=20, l=20, r=20),
                        font=dict(family="IBM Plex Mono", color="#94a3b8"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.metric("Result", str(qr))

    if st.session_state.query_history:
        with st.expander("Query history"):
            for h in reversed(st.session_state.query_history[-20:]):
                st.code(h, language="text")