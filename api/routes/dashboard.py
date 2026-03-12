"""
Plotly interactive dashboard endpoint for MacroPulse.

GET /dashboard  →  Returns a self-contained HTML page with 5 live charts:
  1. Macro regime probability area chart with FOMC date markers
  2. Risk score timeline
  3. Net liquidity proxy
  4. PCA factor lines
  5. 5-Day Regime Probability Forecast (bar/line)

All data is fetched from the database at request time (no client-side API calls
needed — the page is fully self-contained and can be saved or shared as a file).
"""

from __future__ import annotations

import datetime as dt
import logging

import plotly.graph_objects as go
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from database import queries
from services.fomc_calendar import get_fomc_dates
from services.forecaster import forecast_regime_probabilities
from services.performance import compute_regime_performance

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Dashboard"])

# ── Regime colour palette ────────────────────────────────────────────
_REGIME_COLORS = {
    "expansion":  "#22c55e",   # green
    "recovery":   "#3b82f6",   # blue
    "tightening": "#f97316",   # orange
    "risk_off":   "#ef4444",   # red
}


def _build_dashboard_html(
    history: list[dict],
    liquidity: list[dict],
    factors: list[dict],
    current: dict | None,
    forecast_rows: list[dict] | None = None,
    perf: dict | None = None,
) -> str:
    """Build and return a self-contained Plotly HTML dashboard."""

    history = list(reversed(history))         # oldest → newest for charts
    liquidity = list(reversed(liquidity))
    factors = list(reversed(factors))

    dates_regime  = [r["time"] for r in history]
    dates_liq     = [r["time"] for r in liquidity]
    dates_factors = [r["time"] for r in factors]

    # ── FOMC dates within the displayed range ────────────────────────
    fomc_dates: list[dt.date] = []
    if dates_regime:
        # dates_regime may contain datetime objects or date strings.
        def _to_date(v: object) -> dt.date:
            if isinstance(v, dt.datetime):
                return v.date()
            if isinstance(v, dt.date):
                return v
            return dt.datetime.fromisoformat(str(v)).date()

        range_start = _to_date(dates_regime[0])
        range_end   = _to_date(dates_regime[-1])
        fomc_dates  = get_fomc_dates(range_start, range_end)

    # ── Figure 1: Regime probability stacked area ────────────────────
    fig_regime = go.Figure()
    for label, color in _REGIME_COLORS.items():
        col = f"prob_{label}"
        y = [r.get(col, 0) for r in history]
        fig_regime.add_trace(go.Scatter(
            x=dates_regime, y=y,
            name=label.title(),
            mode="lines",
            stackgroup="one",
            fillcolor=color,
            line=dict(color=color, width=0.5),
        ))

    # Add vertical dashed lines for FOMC meeting dates.
    # Plotly date axes require x as millisecond epoch for add_vline.
    for fomc_dt in fomc_dates:
        x_ms = int(dt.datetime.combine(fomc_dt, dt.time()).timestamp() * 1000)
        fig_regime.add_vline(
            x=x_ms,
            line_dash="dash",
            line_color="rgba(251,191,36,0.6)",
            line_width=1,
            annotation_text="FOMC",
            annotation_position="top",
            annotation_font_size=8,
            annotation_font_color="rgba(251,191,36,0.8)",
        )

    fig_regime.update_layout(
        title="Macro Regime Probabilities  (dashed = FOMC meetings)",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )

    # ── Figure 2: Risk score ─────────────────────────────────────────
    risk_scores = [r.get("risk_score", 0) for r in history]
    regime_names = [r.get("regime", "") for r in history]
    bar_colors = [_REGIME_COLORS.get(r, "#94a3b8") for r in regime_names]

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(
        x=dates_regime, y=risk_scores,
        marker_color=bar_colors,
        name="Risk Score",
        showlegend=False,
    ))
    fig_risk.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
    fig_risk.update_layout(
        title="Daily Risk Score  (-100 bearish → +100 bullish)",
        yaxis=dict(range=[-110, 110]),
        height=300,
    )

    # ── Figure 3: Net liquidity ──────────────────────────────────────
    liq_vals = [r.get("net_liquidity") for r in liquidity]
    fig_liq = go.Figure()
    fig_liq.add_trace(go.Scatter(
        x=dates_liq, y=liq_vals,
        mode="lines",
        name="Net Liquidity",
        line=dict(color="#38bdf8", width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.15)",
    ))
    fig_liq.update_layout(
        title="Net Liquidity Proxy  (Fed Assets − TGA − RRP, M USD)",
        yaxis_tickformat=",.0f",
        height=300,
    )

    # ── Figure 4: PCA factors ────────────────────────────────────────
    fig_factors = go.Figure()
    factor_colors = ["#a855f7", "#ec4899", "#14b8a6", "#f59e0b"]
    factor_labels = [
        "Factor 1 (Liquidity/Rates)",
        "Factor 2 (Risk Appetite)",
        "Factor 3 (Credit Stress)",
        "Factor 4 (Momentum)",
    ]
    for i, (label, color) in enumerate(zip(factor_labels, factor_colors), start=1):
        col = f"factor_{i}"
        y = [r.get(col) for r in factors]
        fig_factors.add_trace(go.Scatter(
            x=dates_factors, y=y,
            mode="lines",
            name=label,
            line=dict(color=color, width=1.5),
        ))
    fig_factors.add_hline(y=0, line_dash="dash", line_color="#64748b", line_width=1)
    fig_factors.update_layout(
        title="PCA Latent Macro Factors",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )

    # ── Figure 5: 5-Day Regime Probability Forecast ──────────────────
    fig_forecast = go.Figure()
    if forecast_rows:
        fc_dates = [str(r.get("date", "")) for r in forecast_rows]
        for label, color in _REGIME_COLORS.items():
            col = f"prob_{label}"
            y_fc = [r.get(col, 0) for r in forecast_rows]
            fig_forecast.add_trace(go.Bar(
                x=fc_dates, y=y_fc,
                name=label.title(),
                marker_color=color,
                opacity=0.85,
            ))
        # Overlay confidence line on secondary axis.
        confidence_vals = [r.get("confidence", 0) for r in forecast_rows]
        fig_forecast.add_trace(go.Scatter(
            x=fc_dates, y=confidence_vals,
            name="Confidence",
            mode="lines+markers",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            yaxis="y2",
        ))
        fig_forecast.update_layout(
            barmode="stack",
            yaxis=dict(tickformat=".0%", range=[0, 1], title="Probability"),
            yaxis2=dict(
                overlaying="y",
                side="right",
                range=[0, 1],
                tickformat=".0%",
                title="Confidence",
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    else:
        fig_forecast.add_annotation(
            text="Forecast unavailable — run the pipeline first",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
    fig_forecast.update_layout(
        title="5-Day Regime Probability Forecast  (ARIMA)",
        height=300,
    )

    # ── Figure 6: Regime-following strategy vs Buy-and-Hold ──────────
    fig_perf = go.Figure()
    if perf and "equity_curve" in perf and not perf.get("error"):
        ec     = perf["equity_curve"]
        strat  = perf.get("strategy", {})
        bnh    = perf.get("buy_and_hold", {})

        # Convert cumulative return index → percentage gain
        strat_pct = [round((v - 1) * 100, 2) for v in ec["strategy"]]
        bnh_pct   = [round((v - 1) * 100, 2) for v in ec["buy_and_hold"]]

        fig_perf.add_trace(go.Scatter(
            x=ec["dates"], y=strat_pct,
            name="Regime Strategy",
            mode="lines",
            line=dict(color="#22c55e", width=2),
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.08)",
        ))
        fig_perf.add_trace(go.Scatter(
            x=ec["dates"], y=bnh_pct,
            name="Buy & Hold SPX",
            mode="lines",
            line=dict(color="#64748b", width=1.5, dash="dot"),
        ))
        fig_perf.add_hline(y=0, line_dash="dash", line_color="#334155", line_width=1)

        # Annotation box with key stats
        alpha = strat.get("alpha_vs_bnh_pct", 0)
        sharpe = strat.get("sharpe_ratio", 0)
        max_dd = strat.get("max_drawdown_pct", 0)
        fig_perf.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.97,
            text=(
                f"<b>Strategy</b>  Sharpe {sharpe:.2f} · "
                f"Alpha +{alpha:.1f}% · "
                f"MaxDD {max_dd:.1f}%"
            ),
            showarrow=False,
            font=dict(size=11, color="#94a3b8"),
            align="left",
            bgcolor="rgba(15,23,42,0.7)",
            borderpad=6,
        )
        lookback = perf.get("lookback_days", 252)
        fig_perf.update_layout(
            title=f"Regime Strategy vs Buy & Hold  ({lookback}d sample)",
            yaxis=dict(ticksuffix="%", title="Cumulative Return"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
        )
    else:
        fig_perf.add_annotation(
            text="Performance data unavailable — run the pipeline first",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        fig_perf.update_layout(title="Regime Strategy vs Buy & Hold", height=320)

    # ── Apply shared dark theme ──────────────────────────────────────
    dark_layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=12),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    for fig in (fig_regime, fig_risk, fig_liq, fig_factors, fig_forecast, fig_perf):
        fig.update_layout(**dark_layout)

    # ── Serialise figures to JSON ────────────────────────────────────
    def fig_json(fig: go.Figure) -> str:  # noqa: E301
        return fig.to_json()

    # ── Current regime badge ─────────────────────────────────────────
    if current:
        regime    = current.get("regime", "–")
        risk      = current.get("risk_score", 0)
        vol_state = current.get("volatility_state", "–")
        ts        = current["time"].strftime("%Y-%m-%d")
        badge_color = _REGIME_COLORS.get(regime, "#94a3b8")
        regime_badge = f"""
        <div class="regime-card" style="border-left: 4px solid {badge_color};">
          <div class="regime-label">CURRENT REGIME</div>
          <div class="regime-value" style="color:{badge_color};">{regime.upper()}</div>
          <div class="regime-meta">
            <span>Risk Score: <strong>{risk:.1f}</strong></span>
            <span>Volatility: <strong>{vol_state}</strong></span>
            <span>As of: <strong>{ts}</strong></span>
          </div>
        </div>"""
    else:
        regime_badge = "<p style='color:#94a3b8;'>No regime data — run the pipeline first.</p>"

    # ── Assemble HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MacroPulse Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" crossorigin="anonymous"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f172a;
    color: #e2e8f0;
    font-family: Inter, system-ui, sans-serif;
    padding: 24px;
  }}
  header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 16px;
  }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
  header h1 span {{ color: #38bdf8; }}
  .subtitle {{ color: #64748b; font-size: 0.85rem; margin-top: 2px; }}
  .regime-card {{
    background: #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 32px;
  }}
  .regime-label {{ color: #64748b; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; }}
  .regime-value {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; margin-top: 4px; }}
  .regime-meta {{ display: flex; gap: 24px; color: #94a3b8; font-size: 0.85rem; }}
  .regime-meta strong {{ color: #e2e8f0; }}
  .chart {{ background: #0f172a; border-radius: 10px; margin-bottom: 20px; overflow: hidden; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .refresh-note {{ color: #475569; font-size: 0.75rem; text-align: right; margin-top: 16px; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Macro<span>Pulse</span></h1>
    <div class="subtitle">Probabilistic macro regime intelligence · PCA + HMM · Powered by FRED &amp; yfinance</div>
  </div>
  <div id="last-updated" style="color:#64748b; font-size:0.8rem;"></div>
</header>

{regime_badge}

<div class="chart">
  <div id="chart-regime"></div>
</div>

<div class="chart-grid">
  <div class="chart"><div id="chart-risk"></div></div>
  <div class="chart"><div id="chart-liq"></div></div>
</div>

<div class="chart">
  <div id="chart-factors"></div>
</div>

<div class="chart">
  <div id="chart-forecast"></div>
</div>

<div class="chart">
  <div id="chart-perf"></div>
</div>

<div class="refresh-note">
  Dashboard auto-refreshes every 60s · Data source: FRED + Yahoo Finance · Models: PCA + HMM + GARCH + ARIMA
</div>

<script>
  const figs = {{
    "chart-regime":   {fig_json(fig_regime)},
    "chart-risk":     {fig_json(fig_risk)},
    "chart-liq":      {fig_json(fig_liq)},
    "chart-factors":  {fig_json(fig_factors)},
    "chart-forecast": {fig_json(fig_forecast)},
    "chart-perf":     {fig_json(fig_perf)}
  }};

  const cfg = {{ responsive: true, displayModeBar: false }};
  for (const [id, spec] of Object.entries(figs)) {{
    Plotly.newPlot(id, spec.data, spec.layout, cfg);
  }}

  document.getElementById("last-updated").textContent =
    "Last updated: " + new Date().toLocaleTimeString();

  setTimeout(() => location.reload(), 60_000);
</script>
</body>
</html>"""

    return html


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    """
    Serve the MacroPulse interactive Plotly dashboard.

    Returns a fully self-contained HTML page — no frontend build required.
    Refreshes automatically every 60 seconds.
    """
    import pandas as pd  # local import to keep module-level imports lean

    current   = queries.fetch_current_regime()
    history   = queries.fetch_regime_history(limit=252)   # ~1 year
    liquidity = queries.fetch_latest_liquidity(limit=252)
    factors   = queries.fetch_latest_factors(limit=252)

    # ── Generate 5-day forecast (best-effort; dashboard still loads on failure) ──
    forecast_rows: list[dict] | None = None
    try:
        recent_history = queries.fetch_regime_history(limit=60)
        if recent_history and len(recent_history) >= 10:
            hist_df = pd.DataFrame(list(reversed(recent_history)))
            hist_df = hist_df.set_index("time").sort_index()
            forecast_rows = forecast_regime_probabilities(hist_df, horizon=5)
    except Exception as exc:
        logger.warning("Dashboard forecast failed (non-fatal): %s", exc)

    # ── Performance attribution (best-effort) ────────────────────────
    perf: dict | None = None
    try:
        perf = compute_regime_performance(lookback_days=252)
    except Exception as exc:
        logger.warning("Dashboard performance failed (non-fatal): %s", exc)

    html = _build_dashboard_html(history, liquidity, factors, current, forecast_rows, perf)
    return HTMLResponse(content=html, status_code=200)
