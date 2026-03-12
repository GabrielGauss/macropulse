"""
Claude AI macro commentary endpoint for MacroPulse.

GET /v1/regime/commentary

Calls Claude claude-sonnet-4-6 with the current regime signal and recent history
and returns a professional macro narrative suitable for institutional clients.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import json
import logging

import anthropic
from fastapi import APIRouter, HTTPException

from api.schemas.responses import CommentaryResponse
from config.settings import get_settings
from database import queries

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["MacroPulse"])

_SYSTEM_PROMPT = """\
You are a senior macro strategist at a quantitative hedge fund with deep expertise \
in monetary policy, credit markets, and cross-asset analysis.

You will receive the current macro regime signal from MacroPulse — a quantitative model \
that ingests Fed balance sheet data (WALCL), Treasury General Account (WTREGEN), \
Reverse Repo facility (RRPONTSYD), HY credit spreads, Treasury yield curves, \
equity volatility (VIX), and DXY to produce probabilistic macro regime classifications \
using PCA latent factors and a Hidden Markov Model.

Write a concise, professional macro commentary (2–3 tight paragraphs) that:
1. Interprets the current regime and its market implications.
2. Highlights the key signals driving the classification.
3. Flags what could trigger a regime transition.

Respond ONLY with a JSON object in exactly this format (no markdown, no code fences):
{
  "headline": "<one sentence capturing the dominant macro theme>",
  "narrative": "<2-3 paragraph analysis, separated by \\n\\n>",
  "key_signals": ["<signal 1>", "<signal 2>", "<signal 3>"],
  "watch_for": "<one sentence on what could shift the regime>"
}

Use precise, quantitative language. Be direct. No hedging language like \
"it appears" or "may potentially". Institutional clients only want actionable insight.\
"""


def _build_context(
    regime_row: dict,
    history: list[dict],
    liquidity: list[dict],
    factors: list[dict],
) -> str:
    """Assemble a structured context block for Claude."""
    probs = {
        "expansion":  regime_row.get("prob_expansion", 0),
        "tightening": regime_row.get("prob_tightening", 0),
        "risk_off":   regime_row.get("prob_risk_off", 0),
        "recovery":   regime_row.get("prob_recovery", 0),
    }
    dominant_prob = probs.get(regime_row["regime"], 0)

    # Recent regime transitions
    regime_history_str = "\n".join(
        f"  {r['time'].strftime('%Y-%m-%d')}  {r['regime']}  "
        f"(risk_score={r['risk_score']:.1f})"
        for r in history[:14]
    )

    # Liquidity trend direction
    liq_vals = [r["net_liquidity"] for r in liquidity if r.get("net_liquidity") is not None]
    if len(liq_vals) >= 2:
        liq_delta = liq_vals[0] - liq_vals[-1]
        liq_trend = f"{'expanding' if liq_delta > 0 else 'contracting'} ({liq_delta:+,.0f} M USD over {len(liq_vals)} days)"
    else:
        liq_trend = "insufficient data"

    # Latest PCA factor values
    if factors:
        f = factors[0]
        factors_str = (
            f"  Factor 1 (liquidity/rates): {f.get('factor_1', 0):.3f}\n"
            f"  Factor 2 (risk appetite):   {f.get('factor_2', 0):.3f}\n"
            f"  Factor 3 (credit stress):   {f.get('factor_3', 0):.3f}\n"
            f"  Factor 4 (momentum):        {f.get('factor_4', 0):.3f}"
        )
    else:
        factors_str = "  No factor data available."

    return f"""
MacroPulse Regime Signal — {regime_row['time'].strftime('%Y-%m-%d')}

Current Regime: {regime_row['regime'].upper()}  (confidence: {dominant_prob:.0%})
Risk Score: {regime_row['risk_score']:.1f}  (scale: -100 bearish → +100 bullish)
Volatility State: {regime_row.get('volatility_state', 'unknown')}
Model Version: {regime_row.get('model_version', 'v1')}

Regime Probabilities:
  Expansion:  {probs['expansion']:.1%}
  Recovery:   {probs['recovery']:.1%}
  Tightening: {probs['tightening']:.1%}
  Risk-Off:   {probs['risk_off']:.1%}

Recent Regime History (most recent first):
{regime_history_str}

Net Liquidity Proxy (FedAssets − TGA − RRP):
  Trend: {liq_trend}
  Latest: {liq_vals[0]:,.0f} M USD

PCA Latent Macro Factors (latest):
{factors_str}

Generate a professional macro commentary for institutional clients.
""".strip()


@router.get("/regime/commentary", response_model=CommentaryResponse)
def get_regime_commentary() -> CommentaryResponse:
    """
    Generate a Claude AI macro narrative for the current regime signal.

    Requires ANTHROPIC_API_KEY to be set in the environment.
    Cached for the session — call /v1/regime/current for raw data.
    """
    settings = get_settings()

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "ANTHROPIC_API_KEY is not configured. "
                "Add it to your .env file to enable AI commentary."
            ),
        )

    regime_row = queries.fetch_current_regime()
    if regime_row is None:
        raise HTTPException(status_code=404, detail="No regime data available. Run the pipeline first.")

    history = queries.fetch_regime_history(limit=14)
    liquidity = queries.fetch_latest_liquidity(limit=7)
    factors = queries.fetch_latest_factors(limit=1)

    context = _build_context(regime_row, history, liquidity, factors)
    logger.info("Requesting Claude commentary for regime=%s", regime_row["regime"])

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
    )

    raw = message.content[0].text.strip()

    try:
        parsed: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude returned non-JSON commentary; wrapping as narrative.")
        parsed = {
            "headline": f"MacroPulse: {regime_row['regime'].title()} regime confirmed",
            "narrative": raw,
            "key_signals": [],
            "watch_for": "",
        }

    return CommentaryResponse(
        timestamp=regime_row["time"],
        macro_regime=regime_row["regime"],
        risk_score=regime_row["risk_score"],
        headline=parsed.get("headline", ""),
        narrative=parsed.get("narrative", ""),
        key_signals=parsed.get("key_signals", []),
        watch_for=parsed.get("watch_for", ""),
        model_version=regime_row.get("model_version"),
    )
