"""
ai_layer.py — Layer 4: AI Explanation (not computation)
========================================================
The LLM here receives PRE-COMPUTED, STRUCTURED FACTS from the analytics engine
and is asked ONLY to:
  - narrate what the numbers mean in business language
  - surface non-obvious insights
  - give prioritised recommendations

The LLM NEVER:
  - computes KPI values
  - writes or executes code
  - guesses column meanings from names alone
  - generates SQL

This eliminates hallucinated numbers and generic, useless output.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_json(obj: Any, max_chars: int = 8000) -> str:
    """Convert analytics dict to a compact JSON string, truncated to max_chars."""
    from decimal import Decimal

    def default(o):
        if isinstance(o, Decimal):
            return float(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    raw = json.dumps(obj, default=default, indent=1)
    if len(raw) > max_chars:
        raw = raw[:max_chars] + "\n... [truncated]"
    return raw


# ──────────────────────────────────────────────────────────────────────────────
# Question Intent Detection
# ──────────────────────────────────────────────────────────────────────────────

# Maps intent keys to keyword sets that trigger them.
# More specific intents are listed first.
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "loss_making": [
        "loss", "loss-making", "losing", "losses", "unprofitable",
        "negative profit", "losing money", "in the red", "loss making",
    ],
    "high_revenue_low_profit": [
        "high sales low profit", "high revenue low profit", "high sales low margin",
        "high revenue low margin", "revenue but no profit", "selling but not profitable",
        "revenue without profit", "low margin despite high",
    ],
    "segment_profitability": [
        "category", "categories", "segment", "segments", "which categories",
        "which segments", "by category", "by segment", "product profitability",
        "region profitability", "most profitable", "least profitable",
        "sub-category", "subcategory",
    ],
    "top_performers": [
        "top", "best", "highest", "most revenue", "most profit",
        "best performing", "top performing", "leading",
    ],
    "anomalies": [
        "anomal", "outlier", "unusual", "spike", "drop", "abnormal",
        "unexpected", "weird", "strange",
    ],
    "growth": [
        "growth", "trend", "growing", "declining", "change over time",
        "period over period", "month over month", "year over year",
    ],
    "correlation": [
        "correlat", "relationship between", "related to", "affects", "drives",
    ],
}


def _detect_question_intent(question: str) -> List[str]:
    """
    Return a list of matched intent keys for the given question string.
    Multiple intents can fire simultaneously — e.g. "loss-making categories"
    matches both loss_making and segment_profitability.
    """
    q = question.lower().strip()
    matched = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            matched.append(intent)
    return matched


# ──────────────────────────────────────────────────────────────────────────────
# Payload Builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_narration_payload(analytics: Dict, context: str) -> str:
    """
    Build a compact, intent-aware payload for the LLM.

    When context contains a recognisable question, the payload is RESTRUCTURED
    so the most relevant data surfaces at the top. The LLM reads JSON
    top-to-bottom — key ordering directly influences what it prioritises.

    For loss / profitability questions:
      - segment_insights is the FIRST key (answer lives here)
      - groupby_summaries provides per-segment numbers for citation
      - metric_trends and correlations are dropped (noise for this question)
      - question_focus string injected so LLM knows exactly what was asked

    For generic overviews: original flat behaviour is preserved.
    """
    intents = _detect_question_intent(context) if context else []

    # ── Compact groupby summaries ─────────────────────────────────────────────
    groupby_compact = {}
    for key, rows in analytics.get("groupby_summaries", {}).items():
        if rows:
            groupby_compact[key] = rows[:5]

    # ── Per-column trend labels for metric cols ───────────────────────────────
    metric_trend_detail = {}
    for col, p in analytics.get("column_profiles", {}).items():
        if p.get("role") == "metric" and p.get("trend_label"):
            metric_trend_detail[col] = {
                "trend_direction": p.get("trend"),
                "trend_label": p.get("trend_label"),
                "mean": p.get("stats", {}).get("mean"),
                "min": p.get("stats", {}).get("min"),
                "max": p.get("stats", {}).get("max"),
            }

    # ── KPI payload ───────────────────────────────────────────────────────────
    kpis_payload = []
    for k in analytics.get("kpis", []):
        kpis_payload.append({
            "name": k["name"],
            "value": k["value_fmt"],
            "trend": k["trend"],
            "trend_label": k.get("trend_label", ""),
            "category": k.get("category", "other"),
            "formula": k.get("formula", ""),
        })

    segment_insights = analytics.get("segment_insights", {})

    # ── question_focus directive injected at top of payload ───────────────────
    if intents:
        question_focus = (
            f"The user asked: \"{context}\". "
            f"Detected intent(s): {', '.join(intents)}. "
            "Your response MUST directly answer this question using the data below. "
            "Do NOT give a generic business overview."
        )
    else:
        question_focus = None

    # ── Intent-aware payload assembly ─────────────────────────────────────────
    loss_or_profitability = bool(
        {"loss_making", "high_revenue_low_profit", "segment_profitability"} & set(intents)
    )

    payload: Dict[str, Any]

    if loss_or_profitability:
        # Lead with segment_insights — the answer lives here.
        # Drop metric_trends and correlations to reduce distraction.
        payload = {
            "question_focus": question_focus,
            # Primary answer — LLM reads this first
            "segment_insights": {
                "loss_making_segments": segment_insights.get("loss_making_segments", []),
                "high_revenue_low_profit": segment_insights.get("high_revenue_low_profit", []),
                "top_profit_segments": segment_insights.get("top_profit_segments", []),
            },
            # Segment-level breakdowns with exact numbers for citation
            "top_segments": groupby_compact,
            # Aggregate context (total revenue, profit, margin)
            "kpis": kpis_payload,
            "row_count": analytics.get("row_count"),
            "dim_cols": analytics.get("dim_cols", []),
            "metric_cols": analytics.get("metric_cols", []),
            "detected_business_metrics": analytics.get("detected_business_metrics", {}),
        }

    elif "anomalies" in intents:
        payload = {
            "question_focus": question_focus,
            "anomalies": analytics.get("anomalies", []),
            "metric_trends": metric_trend_detail,
            "kpis": kpis_payload,
            "segment_insights": segment_insights,
            "row_count": analytics.get("row_count"),
            "metric_cols": analytics.get("metric_cols", []),
        }

    elif "growth" in intents:
        payload = {
            "question_focus": question_focus,
            "metric_trends": metric_trend_detail,
            "kpis": [k for k in kpis_payload if k.get("category") in ("revenue", "growth", "profitability")],
            "correlations": analytics.get("correlations", [])[:4],
            "anomalies": analytics.get("anomalies", []),
            "time_cols": analytics.get("time_cols", []),
            "row_count": analytics.get("row_count"),
        }

    elif "correlation" in intents:
        payload = {
            "question_focus": question_focus,
            "correlations": analytics.get("correlations", []),
            "metric_trends": metric_trend_detail,
            "kpis": kpis_payload,
            "row_count": analytics.get("row_count"),
        }

    elif "top_performers" in intents:
        payload = {
            "question_focus": question_focus,
            "top_segments": groupby_compact,
            "segment_insights": {
                "top_profit_segments": segment_insights.get("top_profit_segments", []),
            },
            "kpis": kpis_payload,
            "row_count": analytics.get("row_count"),
            "dim_cols": analytics.get("dim_cols", []),
        }

    else:
        # Generic overview — original behaviour preserved
        payload = {
            "segment_insights": segment_insights,
            "row_count": analytics.get("row_count"),
            "time_cols": analytics.get("time_cols", []),
            "metric_cols": analytics.get("metric_cols", []),
            "dim_cols": analytics.get("dim_cols", []),
            "detected_business_metrics": analytics.get("detected_business_metrics", {}),
            "kpis": kpis_payload,
            "metric_trends": metric_trend_detail,
            "correlations": analytics.get("correlations", [])[:6],
            "anomalies": analytics.get("anomalies", []),
            "top_segments": groupby_compact,
        }
        if question_focus:
            payload["question_focus"] = question_focus

    return _safe_json(payload)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────

NARRATION_SYSTEM_PROMPT = """
You are a Chief Financial Officer and business strategist with 20 years of experience
turning financial and operational data into executable business decisions.

You will receive a JSON object of pre-computed BUSINESS INTELLIGENCE:
- KPIs with categories (profitability, revenue, growth, efficiency, cost, customers)
- Detected business metrics (revenue, cost, profit, quantity, orders, customers)
- Trends, correlations, anomalies — all calculated from real data

══════════════════════════════════════════════════════════════
CRITICAL — READ question_focus BEFORE ANYTHING ELSE
══════════════════════════════════════════════════════════════
If the JSON payload contains a "question_focus" key:

1. Read it FIRST, before any other field.
2. Your "business_context" field MUST open with a direct one-sentence answer
   to the user's question (e.g. "Three sub-categories are loss-making despite
   high sales: Tables, Bookcases, and Supplies.").
3. Every insight MUST be relevant to the question. Do NOT pad with generic
   KPI summaries unrelated to what was asked.
4. For loss / segment profitability questions:
   - Use "segment_insights.loss_making_segments" as your PRIMARY source for
     loss-making findings.
   - Use "segment_insights.high_revenue_low_profit" for high-sales-low-profit
     findings.
   - At least 3 insights must name specific segments with their exact revenue
     and profit figures from the data.
   - The remaining insights may provide supporting context (e.g. parent
     category trends, overall margin).
5. If segment_insights lists no entries, state: "No loss-making segments were
   detected in this dataset." Do NOT fabricate alternatives.
══════════════════════════════════════════════════════════════

YOUR MANDATE — produce output in four distinct sections:

1. BUSINESS CONTEXT (2-3 sentences)
   If question_focus is present: lead with a direct answer sentence.
   Otherwise: identify the business domain, what the dataset measures,
   and its apparent time scope. Lead with the single most important metric.

2. KEY INSIGHTS (5-7 items, each max 35 words)
   Focus on BUSINESS OUTCOMES:
   - PROFITABILITY: margins, profit trends, cost drivers
   - GROWTH: revenue changes, customer acquisition, expansion
   - EFFICIENCY: unit economics, AOV, revenue per customer
   - RISKS: declining trends, negative margins, anomalies

   Each insight MUST:
   - Reference a SPECIFIC value from the JSON (e.g. "-8.6% margin", "$17,725 loss")
   - Explain the BUSINESS IMPLICATION (what it means for profits, growth, or risk)

   GOOD examples:
     "Tables sub-category loses $17,725 on $206,966 revenue (-8.6% margin) —
      every unit sold destroys value; immediate pricing or discontinuation review needed."
     "Chairs generates $328K revenue but only 8.1% margin — high volume masking
      a structural cost problem; bundle pricing or supplier negotiation warranted."
   BAD examples (do NOT produce):
     "Standard Class shipping generates revenue." (irrelevant to category question)
     "Revenue shows a positive trend." (no value cited, no implication)

3. ACTIONABLE RECOMMENDATIONS (4-6 items, each max 50 words)
   Each must be tied to a specific metric value or segment from the data.
   PRIORITISED: [IMMEDIATE] / [THIS QUARTER] / [STRATEGIC]

   GOOD example:
     "[IMMEDIATE] Tables: -$17,725 profit on $207K revenue. Audit unit economics —
      raise prices, cut COGS, or discontinue low-margin SKUs. Every quarter of
      inaction compounds the loss."

4. WATCH LIST — KPIs TO TRACK (3-5 items)
   Current value + alert threshold for each.
   Example: "Tables profit — currently -$17,725; flag if losses exceed -$20K
   or revenue grows without margin recovery."

STRICT RULES:
- NEVER invent numbers — use only values present in the JSON payload.
- ALWAYS cite specific values when making claims.
- NEVER reference Ship Mode data to answer a category/segment question.
- NEVER say "I cannot determine" — work with what is provided.
- Output ONLY valid JSON. No markdown, no preamble, no trailing text.

Output schema (exact keys required):
{
  "business_context": "<string>",
  "insights": ["<string>", ...],
  "recommendations": ["<string>", ...],
  "watch_list": ["<string>", ...],
  "data_quality_flags": ["<string>", ...]
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# AIExplainer
# ──────────────────────────────────────────────────────────────────────────────

class AIExplainer:
    """
    Sends pre-computed analytics facts to the LLM and returns a structured
    narration. The LLM is strictly a language generator, not a calculator.
    """

    def __init__(self):
        self.client = _get_client()
        self.model = MODEL

    def narrate(self, analytics: Dict, context: str = "") -> Dict:
        payload_str = _build_narration_payload(analytics, context)

        # Detect intents to inject a focused directive into the user message.
        # This complements the question_focus field already in the payload —
        # belt-and-suspenders so the LLM can't miss what was asked.
        intents = _detect_question_intent(context) if context else []

        if intents:
            focused_directive = (
                f"\n\nIMPORTANT — The user asked a SPECIFIC question: \"{context}\"\n"
                f"Detected intent(s): {', '.join(intents)}\n"
                "Your response MUST directly answer this question. "
                "Begin business_context with a one-sentence direct answer. "
                "Do NOT produce a generic dataset overview."
            )
            # For loss/segment questions, add an explicit data pointer
            if {"loss_making", "high_revenue_low_profit", "segment_profitability"} & set(intents):
                focused_directive += (
                    "\n\nThe answer is in the payload:"
                    "\n  segment_insights.loss_making_segments  → segments/sub-categories losing money"
                    "\n  segment_insights.high_revenue_low_profit → high sales but thin/negative margins"
                    "\nCite each entry by name with its revenue and profit figures. "
                    "If those lists are empty, say so explicitly — do NOT substitute "
                    "Ship Mode or other unrelated groupings as the answer."
                )
        else:
            focused_directive = (
                f'\nBusiness context hint from user: "{context or "not provided"}"'
            )

        user_msg = (
            "Here are the pre-computed analytics for this dataset."
            + focused_directive
            + "\n\nAnalytics JSON:\n"
            + payload_str
            + "\n\nPlease provide your business narration."
        )
        print(f"[LLM] | Model: {MODEL}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=1200,
                timeout=45.0,
            )

            content = response.choices[0].message.content or ""
            content = content.strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = "\n".join(lines).strip()

            # Parse JSON
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # Try to extract JSON substring
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])

                # Fallback: wrap raw content
                return {
                    "business_context": "Could not parse structured response.",
                    "insights": [content[:500]] if content else [],
                    "recommendations": [],
                    "watch_list": [],
                    "data_quality_flags": [],
                }

        except Exception as e:
            return {
                "business_context": f"AI narration failed: {e}",
                "insights": [
                    "Run the analysis above — all KPIs and trends are pre-computed and accurate.",
                    "AI narration is a supplementary layer; the analytics panel contains all real numbers.",
                ],
                "recommendations": [
                    "Check your OPENROUTER_API_KEY environment variable.",
                    "Try a different model via OPENROUTER_MODEL env var.",
                ],
                "watch_list": [],
                "data_quality_flags": [],
            }