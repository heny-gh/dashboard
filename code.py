from __future__ import annotations

from io import BytesIO
import base64
import html
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    ARTIFACT_FILENAMES,
    RAW_FILE_PATH,
    ROLLING_WINDOW,
    VOL_TARGET_WINDOW,
    get_expected_input_schema,
    reconstruct_payload_from_artifacts,
    result_to_payload,
    run_pipeline,
    validate_input_file,
)

# ============================================================
# VolSight - Macro-Event FX Volatility Engine
# Production Streamlit interface aligned with the final methodology:
# - Empirical volatility regime classification using realized-volatility thresholds
# - Empirical volatility regime classification using q25/q50/q75/q90
# - Macro-event diagnostics and event-specific model validation
# - Trader-oriented decision support for USD/TND risk management
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FORECAST_ARTIFACT_ROOT = BASE_DIR / "artifacts"
PAYLOAD_CACHE_FILENAME = "dashboard_payload.json"
STANDARD_CALIBRATION_WINDOWS = sorted({250, ROLLING_WINDOW, 1000})

LOGO_CANDIDATE_PATHS = [
    Path("logo.png"),
    Path("logo.svg"),
    Path("assets/logo.png"),
    Path("assets/logo.svg"),
    Path("logo_white.png"),
    Path("assets/logo_white.png"),
]

PAGE_NAMES = [
    "Executive FX Risk Cockpit",
    "Data & Market Inputs",
    "Volatility Regime Classification",
    "Macro-Event Regime Analysis",
    "Model Performance by Event",
    "USD/TND vs Global FX Risk Context",
    "Seasonality & Calendar Risk Patterns",
    "Forecast Engine & Benchmark Attribution",
    "Scenario & Calibration Laboratory",
    "Methodology & Audit Trail",
    "Downloads",
]

FINANCE_COLORS = {
    "page_bg": "#07101D",
    "panel_bg": "#101A2C",
    "panel_bg_2": "#121F34",
    "sidebar_bg": "#08111F",
    "text": "#EAF0FA",
    "muted": "#A8B3CC",
    "border": "#24314D",
    "primary": "#4F7BFF",
    "primary_dark": "#2D61FF",
    "teal": "#2FD8C9",
    "gold": "#F3B952",
    "red": "#FF5F68",
    "green": "#39D98A",
    "purple": "#9A7CFF",
    "slate": "#7F8EA7",
    "grid": "#1A2740",
    "input_bg": "#14243A",
    "hover_bg": "#1F2D48",
}

REGIME_COLORS = {
    "Low-volatility regime": FINANCE_COLORS["green"],
    "Normal-volatility regime": FINANCE_COLORS["primary"],
    "Elevated-volatility regime": FINANCE_COLORS["gold"],
    "High-volatility regime": "#FF9D5C",
    "Stress-volatility regime": FINANCE_COLORS["red"],
    "Unavailable": FINANCE_COLORS["slate"],
}

EVENT_COLORS = {
    "pre_covid_normal": FINANCE_COLORS["green"],
    "covid_shock": FINANCE_COLORS["purple"],
    "post_covid_recovery": FINANCE_COLORS["teal"],
    "ukraine_war_shock": FINANCE_COLORS["red"],
    "post_war_inflation_adjustment": FINANCE_COLORS["primary"],
    "us_tariff_shock": FINANCE_COLORS["gold"],
    "post_tariff_normalization": FINANCE_COLORS["green"],
    "iran_geopolitical_shock": "#FF9D5C",
    # Backward compatibility only for older saved artifacts.
    "post_conflict_normalization": FINANCE_COLORS["green"],
}

CHART_COLORWAY = [
    FINANCE_COLORS["teal"],
    FINANCE_COLORS["primary"],
    FINANCE_COLORS["gold"],
    FINANCE_COLORS["purple"],
    FINANCE_COLORS["green"],
    FINANCE_COLORS["red"],
    FINANCE_COLORS["slate"],
]


def calibration_window_options(custom_label: str = "Custom window") -> List[str]:
    return [f"{days} trading days" for days in STANDARD_CALIBRATION_WINDOWS] + [custom_label]


def default_calibration_window_index(options: List[str], default_days: int = ROLLING_WINDOW) -> int:
    label = f"{default_days} trading days"
    return options.index(label) if label in options else 0


def scenario_window_multiplier(window_days: int, baseline_days: int = ROLLING_WINDOW) -> float:
    # Visual-only sensitivity proxy centered on the active production window.
    if window_days == baseline_days:
        return 1.0
    if window_days < baseline_days:
        return min(1.18, 1.0 + ((baseline_days - window_days) / max(baseline_days, 1)) * 0.14)
    return max(0.88, 1.0 - ((window_days - baseline_days) / max(baseline_days, 1)) * 0.06)

MODEL_LABELS = {
    "naive": "FX Volatility Carry-Forward Benchmark",
    "garch": "Traditional Volatility Benchmark",
    "egarch": "Traditional Volatility Benchmark",
    "ridge": "Linear Macro-Risk Stabilizer",
    "xgboost": "Nonlinear Market-Risk Engine",
    "xgb": "Nonlinear Market-Risk Engine",
    "blend": "Final Macro-Event FX Volatility Engine",
    "regime": "Final Macro-Event FX Volatility Engine",
}

ARTIFACT_DISPLAY_NAMES = {
    "model_ready_dataset": "Model-Ready Dataset",
    "walkforward_results_development": "Development Walk-Forward Results",
    "walkforward_results_holdout": "Holdout Walk-Forward Results",
    "walkforward_summary_development": "Development Performance Summary",
    "walkforward_summary_holdout": "Holdout Performance Summary",
    "final_forecast": "Final Forecast Output",
    "high_vol_classifier_calibration_diagnostics": "High-Volatility Calibration Diagnostics",
    "event_descriptive_stats": "Macro-Event Volatility Analysis",
    "event_metrics_development": "Development Event Metrics",
    "event_metrics_holdout": "Holdout Event Metrics",
}

EVENT_DESCRIPTIVE_ROBUST_COLUMNS = [
    "reliability_label",
    "mean_target_vol_annualized",
    "median_target_vol_annualized",
    "q90_target_vol_annualized",
    "max_target_vol_annualized",
    "share_high_vol_days",
    "annualized_return_vol",
    "annualized_target_vol_ci_low",
    "annualized_target_vol_ci_high",
    "annualized_target_vol_vs_pre_covid_pct",
    "mean_target_vol_annualized_vs_broad_normal_pct",
    "q90_target_vol_annualized_vs_broad_normal_pct",
    "brown_forsythe_pvalue_vs_pre_covid",
    "variance_test_result",
    "source_dataset",
    "sample_size_note",
]

COMPLETED_EVENT_REGIMES = [
    "pre_covid_normal",
    "covid_shock",
    "post_covid_recovery",
    "ukraine_war_shock",
    "post_war_inflation_adjustment",
    "us_tariff_shock",
    "post_tariff_normalization",
]
ONGOING_EVENT_REGIME = "iran_geopolitical_shock"

# ============================================================
# Styling and layout helpers
# ============================================================


def inject_finance_theme() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --finance-primary: {FINANCE_COLORS['primary']};
            --finance-primary-dark: {FINANCE_COLORS['primary_dark']};
            --finance-teal: {FINANCE_COLORS['teal']};
            --finance-gold: {FINANCE_COLORS['gold']};
            --finance-red: {FINANCE_COLORS['red']};
            --finance-green: {FINANCE_COLORS['green']};
            --finance-purple: {FINANCE_COLORS['purple']};
            --finance-text: {FINANCE_COLORS['text']};
            --finance-muted: {FINANCE_COLORS['muted']};
            --finance-border: {FINANCE_COLORS['border']};
            --finance-bg: {FINANCE_COLORS['page_bg']};
            --finance-panel: {FINANCE_COLORS['panel_bg']};
            --finance-panel-2: {FINANCE_COLORS['panel_bg_2']};
            --finance-input-bg: {FINANCE_COLORS['input_bg']};
            --finance-hover-bg: {FINANCE_COLORS['hover_bg']};
        }}

        @keyframes fadeSlideUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .stApp {{
            background:
                radial-gradient(circle at 15% 5%, rgba(79, 123, 255, 0.17), transparent 360px),
                radial-gradient(circle at 85% 8%, rgba(47, 216, 201, 0.12), transparent 400px),
                linear-gradient(135deg, #07101D 0%, #0B1628 48%, #0F1C2D 100%);
            color: var(--finance-text);
        }}

        [data-testid="stAppViewContainer"] > .main .block-container {{
            max-width: 1500px;
            padding-top: 1.1rem;
            padding-left: 1.55rem;
            padding-right: 1.55rem;
            animation: fadeSlideUp 380ms ease-out both;
        }}

        [data-testid="stSidebar"] {{
            background: {FINANCE_COLORS['sidebar_bg']};
            border-right: 1px solid var(--finance-border);
            box-shadow: 18px 0 48px rgba(0,0,0,0.12);
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 13px;
            padding: 0.58rem 0.68rem;
            margin-bottom: 0.24rem;
            background: transparent;
            border: 1px solid transparent;
            transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: var(--finance-hover-bg);
            border-color: var(--finance-border);
            transform: translateX(3px);
        }}

        h1, h2, h3, h4, h5, h6 {{ color: var(--finance-text); }}
        p, li, label, .stMarkdown {{ color: var(--finance-text); }}
        [data-testid="stCaptionContainer"], .stCaption {{ color: var(--finance-muted); }}

        .app-brand {{
            display: flex;
            align-items: center;
            gap: 0.72rem;
            margin: 0.25rem 0 1rem 0;
            color: var(--finance-text);
            font-weight: 900;
            letter-spacing: 0.02em;
        }}
        .app-brand-mark {{
            width: 34px;
            height: 34px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #FFFFFF;
            background: linear-gradient(135deg, #4F7BFF, #2FD8C9);
            box-shadow: 0 12px 22px rgba(79,123,255,0.2);
        }}
        .app-tagline {{
            color: var(--finance-muted);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }}

        .topbar {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 1rem;
        }}
        .page-kicker {{
            color: var(--finance-teal);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }}
        .page-title {{
            font-size: clamp(1.55rem, 2vw, 2.25rem);
            color: var(--finance-text);
            line-height: 1.15;
            font-weight: 900;
        }}
        .topbar-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            justify-content: flex-end;
        }}
        .chip {{
            border: 1px solid var(--finance-border);
            background: rgba(20,36,58,0.85);
            border-radius: 999px;
            color: var(--finance-muted);
            padding: 0.48rem 0.75rem;
            font-size: 0.78rem;
            white-space: nowrap;
        }}
        .chip strong {{ color: var(--finance-text); }}

        .hero-panel {{
            border: 1px solid rgba(79, 123, 255, 0.26);
            background: linear-gradient(135deg, rgba(79,123,255,0.16), rgba(47,216,201,0.07) 45%, rgba(16,26,44,0.96));
            border-radius: 24px;
            padding: 1.4rem;
            box-shadow: 0 28px 70px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }}
        .panel {{
            border: 1px solid var(--finance-border);
            background: linear-gradient(180deg, rgba(18,31,52,0.94), rgba(13,23,38,0.96));
            border-radius: 18px;
            padding: 1.1rem;
            box-shadow: 0 18px 44px rgba(0,0,0,0.16);
            margin-bottom: 1rem;
        }}
        .panel-title {{
            color: var(--finance-text);
            font-weight: 850;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }}
        .panel-subtitle {{
            color: var(--finance-muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-bottom: 0.85rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.8rem 0 1rem 0;
        }}
        .kpi-card {{
            border: 1px solid var(--finance-border);
            background: rgba(255,255,255,0.035);
            border-radius: 16px;
            padding: 1rem;
            min-height: 124px;
            transition: transform 160ms ease, border-color 160ms ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(47,216,201,0.55);
        }}
        .kpi-label {{
            color: var(--finance-muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            color: var(--finance-text);
            font-size: 1.65rem;
            line-height: 1.1;
            font-weight: 900;
            margin-bottom: 0.35rem;
        }}
        .kpi-note {{
            color: var(--finance-muted);
            font-size: 0.82rem;
            line-height: 1.35;
        }}
        .decision-card {{
            border: 1px solid var(--finance-border);
            border-left: 4px solid var(--finance-teal);
            background: var(--finance-panel);
            border-radius: 16px;
            padding: 1rem;
            min-height: 135px;
            margin-bottom: 0.85rem;
        }}
        .decision-title {{
            color: var(--finance-teal);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 850;
            margin-bottom: 0.45rem;
        }}
        .decision-text {{
            color: var(--finance-text);
            font-size: 0.95rem;
            line-height: 1.55;
        }}
        .warning-box {{
            border: 1px solid rgba(243,185,82,0.35);
            border-left: 4px solid var(--finance-gold);
            background: rgba(243,185,82,0.08);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: var(--finance-text);
            line-height: 1.5;
            margin-bottom: 1rem;
        }}
        .method-box {{
            border: 1px solid rgba(47,216,201,0.25);
            border-left: 4px solid var(--finance-teal);
            background: rgba(47,216,201,0.06);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            color: var(--finance-text);
            line-height: 1.5;
            margin-bottom: 1rem;
        }}
        .small-muted {{
            color: var(--finance-muted);
            font-size: 0.82rem;
            line-height: 1.4;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px;
            border: 1px solid var(--finance-primary);
            color: var(--finance-text);
            background: var(--finance-input-bg);
            font-weight: 750;
            transition: transform 150ms ease, background 150ms ease, border-color 150ms ease;
        }}
        .stButton > button[kind="primary"] {{
            color: #FFFFFF;
            background: linear-gradient(135deg, #4F7BFF 0%, #2D61FF 100%);
            border-color: var(--finance-primary);
            box-shadow: 0 14px 30px rgba(79,123,255,0.26);
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            border-color: var(--finance-teal);
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid var(--finance-border);
            border-radius: 14px;
            overflow: hidden;
            background: var(--finance-panel);
        }}
        [data-testid="stPlotlyChart"] {{
            border: 1px solid var(--finance-border);
            border-radius: 16px;
            overflow: hidden;
            background: var(--finance-panel);
            box-shadow: 0 18px 42px rgba(0,0,0,0.12);
        }}
        .validation-card-interpretation {{
            color: #cbd5e1;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.65rem;
        }}
        hr {{ border-color: var(--finance-border); }}

        .validation-matrix-content {{
            display: grid;
            grid-template-columns: minmax(440px, 1.05fr) minmax(360px, 0.95fr);
            gap: 1.2rem;
            align-items: stretch;
            margin-top: 1.2rem;
        }}
        .validation-read-card {{
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(20, 36, 58, 0.78));
            border: 1px solid rgba(47, 216, 201, 0.28);
            border-left: 4px solid var(--finance-teal);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        }}
        .validation-read-title {{
            color: var(--finance-teal);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 850;
            margin-bottom: 0.55rem;
        }}
        .validation-read-text {{
            color: var(--finance-text);
            font-size: 0.94rem;
            line-height: 1.6;
        }}

        @media (max-width: 1000px) {{
            .validation-matrix-content {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
            .topbar {{ flex-direction: column; }}
            .topbar-actions {{ justify-content: flex-start; }}
        }}
        @media (max-width: 650px) {{
            .kpi-grid {{ grid-template-columns: 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_finance_layout(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        colorway=CHART_COLORWAY,
        paper_bgcolor=FINANCE_COLORS["panel_bg"],
        plot_bgcolor=FINANCE_COLORS["panel_bg"],
        font=dict(color=FINANCE_COLORS["text"], family="Arial, sans-serif"),
        title_font=dict(color=FINANCE_COLORS["text"], size=17),
        margin=dict(l=22, r=22, t=45, b=28),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=FINANCE_COLORS["input_bg"],
            bordercolor=FINANCE_COLORS["border"],
            font=dict(color=FINANCE_COLORS["text"]),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=FINANCE_COLORS["text"]),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            gridcolor=FINANCE_COLORS["grid"],
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=FINANCE_COLORS["muted"]),
        ),
        yaxis=dict(
            gridcolor=FINANCE_COLORS["grid"],
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color=FINANCE_COLORS["muted"]),
        ),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def apply_institutional_plotly_layout(
    fig: go.Figure,
    title: str,
    subtitle: Optional[str] = None,
    height: int = 460,
    yaxis_title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    hovermode: str = "closest",
) -> go.Figure:
    full_title = title
    if subtitle:
        full_title = (
            f"{html.escape(title)}"
            f"<br><span style='font-size:12px;color:{FINANCE_COLORS['muted']};font-weight:400;'>{html.escape(subtitle)}</span>"
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=18, r=26, t=78 if subtitle else 58, b=38),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=FINANCE_COLORS["text"], size=12),
        title=dict(text=full_title, x=0.0, xanchor="left", font=dict(size=17, color=FINANCE_COLORS["text"])),
        hovermode=hovermode,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=FINANCE_COLORS["muted"]),
        ),
        xaxis=dict(
            title=dict(text=xaxis_title or "", font=dict(color=FINANCE_COLORS["muted"], size=12)),
            gridcolor="rgba(168,179,204,0.12)",
            zerolinecolor="rgba(234,240,250,0.35)",
            linecolor="rgba(168,179,204,0.18)",
            tickfont=dict(color=FINANCE_COLORS["muted"], size=11),
            ticksuffix="%",
        ),
        yaxis=dict(
            title=dict(text=yaxis_title or "", font=dict(color=FINANCE_COLORS["muted"], size=12)),
            gridcolor="rgba(168,179,204,0.06)",
            zerolinecolor="rgba(234,240,250,0.18)",
            linecolor="rgba(168,179,204,0.18)",
            tickfont=dict(color=FINANCE_COLORS["text"], size=11),
            automargin=True,
        ),
    )
    return fig


def page_header(title: str, kicker: str, payload: Optional[dict]) -> None:
    base_date = get_display_base_date(payload)
    base_date_display = base_date.strftime("%d/%m/%Y") if base_date else "No run loaded"
    forecast_period = calculate_forecast_period(base_date)
    run_id = payload.get("run_id", "No run loaded") if payload else "No run loaded"
    current_event = get_current_event_regime(payload)
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="page-kicker">{html.escape(kicker)}</div>
                <div class="page-title">{html.escape(title)}</div>
            </div>
            <div class="topbar-actions">
                <div class="chip">Run: <strong>{html.escape(str(run_id))}</strong></div>
                <div class="chip">Base: <strong>{html.escape(base_date_display)}</strong></div>
                <div class="chip">Forecast: <strong>{html.escape(forecast_period)}</strong></div>
                <div class="chip">Macro regime: <strong>{html.escape(current_event)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", accent: str = "teal") -> str:
    color = FINANCE_COLORS.get(accent, FINANCE_COLORS["teal"])
    # Keep the HTML compact. Indented multi-line HTML can be interpreted by
    # Streamlit Markdown as a code block, causing raw <div> markup to appear.
    return (
        f'<div class="kpi-card" style="border-top: 3px solid {color};">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-note">{html.escape(str(note or ""))}</div>'
        '</div>'
    )


def render_kpi_grid(cards: List[str]) -> None:
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_market_watchlist(payload: Optional[dict], event_context: dict) -> None:
    dashboard = (payload.get("dashboard", {}) or {}) if payload else {}
    econ = dashboard.get("economic_event_analysis", {}) or {}
    headlines = econ.get("macro_event_headlines") or econ.get("headline") or econ.get("headlines") or "No headline commentary available."
    watchlist_items = [
        "DXY",
        "Brent",
        "MOVE",
        "VIX",
        "Global sentiment",
        "Macro-event headlines",
    ]
    watch_html = "".join(f"<li>{html.escape(item)}</li>" for item in watchlist_items)
    event_context_note = (
        f"Latest macro-event context: {html.escape(event_context['label'])}."
        if event_context["is_classified"]
        else "Latest macro-event context: No active event regime assigned."
    )
    source_note = ""
    if event_context["is_classified"] and event_context.get("source"):
        source_note = f" Source: {html.escape(event_context['source'])}."
    st.markdown(
        f"""
        <div class='decision-card'>
            <div class='decision-title'>Market Watchlist</div>
            <div class='decision-text'>Monitor the following risk signals and headlines for short-term USD/TND hedging decisions.</div>
            <ul style='margin: 0.35rem 0 0 1rem; color: {FINANCE_COLORS['text']};'>{watch_html}</ul>
            <div class='decision-text' style='margin-top:0.8rem;color:{FINANCE_COLORS['muted']}; font-size:0.92rem;'>{event_context_note}{source_note}</div>
            <div class='decision-text' style='margin-top:0.5rem;color:{FINANCE_COLORS['muted']}; font-size:0.92rem;'>Latest macro-event note: {html.escape(str(headlines))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_checklist() -> None:
    checklist_items = [
        "Review USD/TND exposure due over the next 3 trading days.",
        "Check DXY, Brent, MOVE, and VIX before hedge execution.",
        "Compare the official signal with the Scenario Laboratory if hedge timing is uncertain.",
        "Escalate if the high-volatility signal moves into alert zone.",
    ]
    notes_html = "".join(f"<li>{html.escape(item)}</li>" for item in checklist_items)
    st.markdown(
        f"""
        <div class='decision-card'>
            <div class='decision-title'>Action Checklist</div>
            <div class='decision-text'>Use this compact checklist to keep USD/TND risk actions aligned with the latest signal.</div>
            <ul style='margin: 0.35rem 0 0 1rem; color: {FINANCE_COLORS['text']};'>{notes_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_recent_volatility_pulse_df(payload: Optional[dict]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    records = payload.get("holdout_results", []) or (payload.get("dashboard", {}) or {}).get("history", []) or payload.get("latest_history", [])
    df = records_df(records)
    if df.empty:
        return pd.DataFrame()
    for col in ["actual", "actual_volatility", "pred_blend", "predicted_volatility"]:
        if col in df.columns:
            out = df[[col]].copy()
            if "Date" in df.columns:
                out["Date"] = df["Date"]
            out = out.dropna(subset=[col])
            if not out.empty:
                return out
    return pd.DataFrame()


def render_recent_volatility_pulse(payload: Optional[dict]) -> None:
    recent = get_recent_volatility_pulse_df(payload)
    if recent.empty:
        st.info("Recent volatility pulse data is unavailable.")
        return
    y_col = [col for col in recent.columns if col != "Date"][0]
    fig = go.Figure()
    if "Date" in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent["Date"],
                y=pd.to_numeric(recent[y_col], errors="coerce") * 100,
                mode="lines",
                line=dict(color=FINANCE_COLORS["gold"], width=2),
            )
        )
        xaxis = dict(title="Date", tickformat="%b %d", showgrid=False)
    else:
        fig.add_trace(
            go.Scatter(
                x=recent.index + 1,
                y=pd.to_numeric(recent[y_col], errors="coerce") * 100,
                mode="lines",
                line=dict(color=FINANCE_COLORS["gold"], width=2),
            )
        )
        xaxis = dict(title="Recent observations", showticklabels=False, showgrid=False)
    fig.update_layout(
        title="Recent USD/TND Volatility Pulse",
        yaxis_title="Volatility (%)",
        xaxis=xaxis,
        margin=dict(l=18, r=18, t=40, b=24),
        height=240,
        showlegend=False,
    )
    st.plotly_chart(apply_finance_layout(fig, height=240), use_container_width=True)
    st.caption("Recent realized-volatility path used as a quick market-temperature check; this is not a separate forecast.")


def render_navigation_cards() -> None:
    cols = st.columns(4, gap="large")
    nav_options = [
        ("Validate Regime Signal", "Volatility Regime Classification"),
        ("Review Macro-Event Context", "Macro-Event Regime Analysis"),
        ("Test Scenario Overlay", "Scenario & Calibration Laboratory"),
        ("Inspect Benchmark Attribution", "Forecast Engine & Benchmark Attribution"),
    ]
    for col, (label, target_page) in zip(cols, nav_options):
        with col:
            if st.button(label, use_container_width=True, key=f"nav_{target_page.replace(' ', '_')}"):
                st.session_state.selected_page = target_page
                st.rerun()


# ============================================================
# File and payload helpers
# ============================================================


def find_logo_path() -> Optional[Path]:
    for logo_path in LOGO_CANDIDATE_PATHS:
        candidate = BASE_DIR / logo_path
        if candidate.exists():
            return candidate
    return None


def get_logo_data_uri(logo_path: Path) -> str:
    suffix = logo_path.suffix.lower()
    mime_type = "image/svg+xml" if suffix == ".svg" else "image/png"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def local_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_input_path() -> Path:
    """Return the default Excel path relative to the deployed app directory.

    Streamlit Community Cloud runs the app on Linux from the GitHub repository.
    Using BASE_DIR avoids accidental dependence on the process working directory.
    """
    raw_path = Path(RAW_FILE_PATH)
    return raw_path if raw_path.is_absolute() else BASE_DIR / raw_path


def find_latest_run_dir() -> Optional[Path]:
    if not FORECAST_ARTIFACT_ROOT.exists():
        return None
    run_dirs = [path for path in FORECAST_ARTIFACT_ROOT.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def load_dashboard_payload_from_artifacts(run_dir: Path) -> Optional[dict]:
    payload_file = run_dir / PAYLOAD_CACHE_FILENAME
    if payload_file.exists():
        try:
            return json.loads(payload_file.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _artifact_candidates(payload: Optional[dict], key: str) -> List[Path]:
    candidates: List[Path] = []
    if payload:
        artifacts = payload.get("artifacts", {}) or {}
        if key in artifacts:
            candidates.append(Path(artifacts[key]))
        artifact_dir = payload.get("artifact_dir")
        if artifact_dir and key in ARTIFACT_FILENAMES:
            candidates.append(Path(artifact_dir) / ARTIFACT_FILENAMES[key])
    if key in ARTIFACT_FILENAMES:
        candidates.append(Path(ARTIFACT_FILENAMES[key]))
        candidates.append(BASE_DIR / ARTIFACT_FILENAMES[key])

    seen = set()
    existing = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            existing.append(candidate)
    return existing


def freshest_artifact_path(payload: Optional[dict], key: str) -> Optional[Path]:
    candidates = _artifact_candidates(payload, key)
    if not candidates:
        return None
    return max(candidates, key=_path_mtime)


@st.cache_data(show_spinner=False)
def _read_excel_cached(path_str: str, file_mtime: float, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if sheet_name is None:
        return pd.read_excel(path_str)
    return pd.read_excel(path_str, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def _read_json_cached(path_str: str, file_mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def infer_base_date_from_input_file(input_path: Path) -> Optional[str]:
    input_path = Path(input_path)
    if not input_path.exists():
        return None
    try:
        input_df = pd.read_excel(input_path, sheet_name="Sheet1")
        input_df["Date"] = pd.to_datetime(input_df["Date"], errors="coerce")
        dates = input_df["Date"].dropna()
        if dates.empty:
            return None
        return dates.max().date().isoformat()
    except Exception:
        return None


def infer_base_date_from_run_dir(run_dir: Path) -> Optional[str]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    artifact_names = set(ARTIFACT_FILENAMES.values())
    for input_file in sorted(run_dir.glob("*.xls*")):
        if input_file.name == PAYLOAD_CACHE_FILENAME or input_file.name in artifact_names:
            continue
        base_date = infer_base_date_from_input_file(input_file)
        if base_date:
            return base_date
    return None


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", ""))
    except Exception:
        try:
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                return parsed.to_pydatetime()
        except Exception:
            return None
    return None


def get_display_base_date(payload: Optional[dict]) -> Optional[datetime]:
    if not payload:
        return None
    candidates: List[Any] = []
    if payload.get("base_date"):
        candidates.append(payload["base_date"])
    forecast = payload.get("forecast", {}) or {}
    for key in ("base_date", "last_data_date", "forecast_date"):
        if forecast.get(key):
            candidates.append(forecast[key])
    if payload.get("artifact_dir"):
        inferred = infer_base_date_from_run_dir(Path(payload["artifact_dir"]))
        if inferred:
            candidates.append(inferred)
    for item in payload.get("holdout_results", []) or []:
        if isinstance(item, dict) and item.get("Date"):
            candidates.append(item["Date"])
    parsed = [dt for dt in (_parse_date(x) for x in candidates) if dt is not None]
    return max(parsed) if parsed else None


def calculate_forecast_period(base_date: Optional[datetime]) -> str:
    if not base_date:
        return "Unknown"
    start_date = base_date + timedelta(days=1)
    end_date = base_date + timedelta(days=3)
    return f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"


def save_dashboard_payload(payload: dict, run_dir: Path) -> None:
    (run_dir / PAYLOAD_CACHE_FILENAME).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


def load_latest_local() -> dict:
    latest_run = find_latest_run_dir()
    root_payload = BASE_DIR / PAYLOAD_CACHE_FILENAME
    latest_run_payload = latest_run / PAYLOAD_CACHE_FILENAME if latest_run is not None else None
    use_root_artifacts = root_payload.exists() and (
        latest_run_payload is None
        or not latest_run_payload.exists()
        or _path_mtime(root_payload) >= _path_mtime(latest_run_payload)
    )

    if use_root_artifacts:
        payload = load_dashboard_payload_from_artifacts(BASE_DIR)
        if payload is None:
            payload = reconstruct_payload_from_artifacts(BASE_DIR)
        payload["run_id"] = "root_artifacts"
        payload["artifact_dir"] = str(BASE_DIR)
        return payload

    if latest_run is None:
        raise FileNotFoundError("No saved forecast artifacts were found.")
    payload = load_dashboard_payload_from_artifacts(latest_run)
    if payload is None:
        payload = reconstruct_payload_from_artifacts(latest_run)
    payload["run_id"] = latest_run.name
    payload["artifact_dir"] = str(latest_run)
    inferred_base_date = infer_base_date_from_run_dir(latest_run)
    if inferred_base_date:
        payload["base_date"] = inferred_base_date
        payload.setdefault("forecast", {})["base_date"] = inferred_base_date
    return payload


def artifact_path(payload: Optional[dict], key: str, prefer_freshest: bool = False) -> Optional[Path]:
    if prefer_freshest:
        return freshest_artifact_path(payload, key)
    if payload:
        artifacts = payload.get("artifacts", {}) or {}
        if key in artifacts and Path(artifacts[key]).exists():
            return Path(artifacts[key])
        artifact_dir = payload.get("artifact_dir")
        if artifact_dir and key in ARTIFACT_FILENAMES:
            p = Path(artifact_dir) / ARTIFACT_FILENAMES[key]
            if p.exists():
                return p
    if key in ARTIFACT_FILENAMES:
        p = Path(ARTIFACT_FILENAMES[key])
        if p.exists():
            return p
    return None


def load_artifact_df(payload: Optional[dict], key: str) -> pd.DataFrame:
    p = artifact_path(payload, key)
    if p is None:
        return pd.DataFrame()
    try:
        return _read_excel_cached(str(p.resolve()), _path_mtime(p))
    except Exception:
        return pd.DataFrame()


def load_fresh_artifact_df(payload: Optional[dict], key: str) -> pd.DataFrame:
    p = artifact_path(payload, key, prefer_freshest=True)
    if p is None:
        return pd.DataFrame()
    try:
        df = _read_excel_cached(str(p.resolve()), _path_mtime(p)).copy()
        df.attrs["artifact_path"] = str(p)
        df.attrs["artifact_mtime"] = _path_mtime(p)
        return df
    except Exception:
        return pd.DataFrame()


def load_json_artifact(payload: Optional[dict], key: str) -> dict:
    p = artifact_path(payload, key, prefer_freshest=True)
    if p is None:
        return {}
    try:
        data = _read_json_cached(str(p.resolve()), _path_mtime(p))
        data["_artifact_path"] = str(p)
        data["_artifact_mtime"] = _path_mtime(p)
        return data
    except Exception:
        return {}


def get_high_vol_calibration_df(payload: Optional[dict], sheet_name: str) -> pd.DataFrame:
    path = artifact_path(payload, "high_vol_classifier_calibration_diagnostics")
    if path is None:
        return pd.DataFrame()
    try:
        return _read_excel_cached(str(path.resolve()), _path_mtime(path), sheet_name)
    except Exception:
        return pd.DataFrame()


def records_df(records: Optional[List[dict]]) -> pd.DataFrame:
    df = pd.DataFrame(records or [])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def get_holdout_df(payload: Optional[dict]) -> pd.DataFrame:
    if payload:
        df = records_df(payload.get("holdout_results", []))
        if not df.empty:
            return df
    return load_artifact_df(payload, "walkforward_results_holdout")


def get_development_df(payload: Optional[dict]) -> pd.DataFrame:
    if payload:
        df = records_df(payload.get("development_results", []))
        if not df.empty:
            return df
    return load_artifact_df(payload, "walkforward_results_development")


def get_model_ready_df(payload: Optional[dict]) -> pd.DataFrame:
    return load_artifact_df(payload, "model_ready_dataset")


def get_event_descriptive_df(payload: Optional[dict]) -> pd.DataFrame:
    artifact_df = load_fresh_artifact_df(payload, "event_descriptive_stats")
    if not artifact_df.empty and any(c in artifact_df.columns for c in EVENT_DESCRIPTIVE_ROBUST_COLUMNS):
        artifact_df.attrs["source"] = "freshest_artifact"
        return artifact_df

    if payload:
        econ = (payload.get("dashboard", {}) or {}).get("economic_event_analysis", {}) or {}
        df = pd.DataFrame(econ.get("descriptive_stats", []) or [])
        if not df.empty:
            df.attrs["source"] = "payload.dashboard.economic_event_analysis.descriptive_stats"
            return df
        metadata = payload.get("metadata", {}) or {}
        df = pd.DataFrame(metadata.get("event_descriptive_stats_records", []) or [])
        if not df.empty:
            df.attrs["source"] = "payload.metadata.event_descriptive_stats_records"
            return df

    if not artifact_df.empty:
        artifact_df.attrs["source"] = "freshest_artifact"
    return artifact_df


def get_run_metadata(payload: Optional[dict]) -> dict:
    metadata = {}
    if payload:
        metadata.update(payload.get("metadata", {}) or {})
    artifact_metadata = load_json_artifact(payload, "run_metadata")
    if artifact_metadata:
        artifact_metadata.pop("_artifact_mtime", None)
        metadata.update(artifact_metadata)
    elif payload:
        dashboard_payload = load_json_artifact(payload, "dashboard_payload")
        metadata.update((dashboard_payload.get("metadata", {}) or {}) if dashboard_payload else {})
    return metadata


def get_event_metrics_df(payload: Optional[dict], period: str = "holdout") -> pd.DataFrame:
    if payload:
        econ = (payload.get("dashboard", {}) or {}).get("economic_event_analysis", {}) or {}
        key = "holdout_event_metrics" if period == "holdout" else "development_event_metrics"
        df = pd.DataFrame(econ.get(key, []) or [])
        if not df.empty:
            return df
        metadata = payload.get("metadata", {}) or {}
        meta_key = "event_metrics_holdout_records" if period == "holdout" else "event_metrics_dev_records"
        df = pd.DataFrame(metadata.get(meta_key, []) or [])
        if not df.empty:
            return df
    artifact_key = "event_metrics_holdout" if period == "holdout" else "event_metrics_development"
    return load_artifact_df(payload, artifact_key)


def get_summary_df(payload: Optional[dict], period: str = "holdout") -> pd.DataFrame:
    if payload:
        records = payload.get("holdout_summary" if period == "holdout" else "development_summary", [])
        df = pd.DataFrame(records or [])
        if not df.empty:
            return df
    key = "walkforward_summary_holdout" if period == "holdout" else "walkforward_summary_development"
    return load_artifact_df(payload, key)


# ============================================================
# Formatting and finance-specific interpretation
# ============================================================


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def format_number(value: Any, digits: int = 4) -> str:
    x = safe_float(value)
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def format_pct(value: Any, digits: int = 2) -> str:
    x = safe_float(value)
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def format_ratio(value: Any, digits: int = 2) -> str:
    x = safe_float(value)
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}x"


def normalize_metric_ratio(value: Any) -> Optional[float]:
    """Normalize metric to 0-1 scale for consistent interpretation."""
    if value is None or pd.isna(value):
        return None
    try:
        val = float(value)
    except (ValueError, TypeError):
        return None
    if abs(val) > 1:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def format_metric_percent(value: Any, digits: int = 2) -> str:
    """Format metric as percentage with specified decimal places."""
    normalized = normalize_metric_ratio(value)
    if normalized is None:
        return "N/A"
    return f"{100.0 * normalized:.{digits}f}%"


def interpret_classification_accuracy(accuracy: Any) -> str:
    acc = normalize_metric_ratio(accuracy)
    if acc is None:
        return "Classification accuracy is unavailable for the current validation sample."
    val_str = format_metric_percent(accuracy)
    if acc >= 0.85:
        return f"With an accuracy of {val_str}, the model correctly classifies volatility regimes in a strong majority of observations."
    if acc >= 0.75:
        return f"With an accuracy of {val_str}, the model correctly classifies volatility regimes in roughly eight out of ten observations."
    if acc >= 0.65:
        return f"With an accuracy of {val_str}, the model shows acceptable classification power, but regime calls should be monitored with supporting market context."
    return f"With an accuracy of {val_str}, the model’s classification hit rate is limited; regime calls should be treated cautiously."


def interpret_recall(recall: Any) -> str:
    rec = normalize_metric_ratio(recall)
    if rec is None:
        return "Recall is unavailable for the current validation sample."
    val_str = format_metric_percent(recall)
    if rec >= 0.80:
        return f"With a recall of {val_str}, the model captures most realized high-volatility periods."
    if rec >= 0.65:
        return f"With a recall of {val_str}, the model captures a solid share of realized high-volatility periods."
    if rec >= 0.50:
        return f"With a recall of {val_str}, the model captures some high-volatility periods, but missed stress episodes remain material."
    return f"With a recall of {val_str}, the model misses many realized high-volatility periods; risk alerts should be treated cautiously."


def interpret_f1_score(f1_score: Any) -> str:
    f1 = normalize_metric_ratio(f1_score)
    if f1 is None:
        return "F1 score is unavailable for the current validation sample."
    val_str = format_metric_percent(f1_score)
    if f1 >= 0.75:
        return f"With an F1 score of {val_str}, the signal shows strong balance between capturing high-volatility periods and limiting false alerts."
    if f1 >= 0.60:
        return f"With an F1 score of {val_str}, the signal reasonably detects high-volatility periods, although some false alerts or missed episodes remain."
    if f1 >= 0.45:
        return f"With an F1 score of {val_str}, the signal provides partial classification value, but alert quality should be reviewed carefully."
    return f"With an F1 score of {val_str}, the signal has weak balance between false alerts and missed high-volatility periods."


def pretty_event_name(value: Any) -> str:
    mapping = {
        "pre_covid_normal": "Pre-COVID Normal",
        "covid_shock": "COVID Shock",
        "post_covid_recovery": "Post-COVID Recovery",
        "ukraine_war_shock": "Russia-Ukraine War Shock",
        "post_war_inflation_adjustment": "Post-War Inflation Adjustment",
        "us_tariff_shock": "U.S. Tariff Shock",
        "post_tariff_normalization": "Post-Tariff Normalization",
        "iran_geopolitical_shock": "Iran Geopolitical Shock",
        # Backward compatibility only for older saved artifacts.
        "post_conflict_normalization": "Post-Tariff Normalization",
    }
    text = str(value or "Unavailable")
    return mapping.get(text, text.replace("_", " ").title())


def is_valid_event_context_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    invalid = {
        "nan",
        "none",
        "null",
        "undefined",
        "unavailable",
        "not available",
        "not classified",
        "n/a",
    }
    return text.lower() not in invalid


def format_event_context_label(value: Any) -> Optional[str]:
    if not is_valid_event_context_text(value):
        return None
    text = str(value).strip()
    mapping = {
        "pre_covid_normal": "Pre-COVID Normal",
        "covid_shock": "COVID Shock",
        "post_covid_recovery": "Post-COVID Recovery",
        "ukraine_war_shock": "Russia-Ukraine War Shock",
        "post_war_inflation_adjustment": "Post-War Inflation Adjustment",
        "us_tariff_shock": "U.S. Tariff Shock",
        "post_tariff_normalization": "Post-Tariff Normalization",
        "post-tariff normalization regime": "Post-Tariff Normalization",
        "iran_geopolitical_shock": "Iran Geopolitical Shock",
        "iran / us-israel geopolitical escalation": "Iran Geopolitical Shock",
        # Backward compatibility only for older saved artifacts.
        "post_conflict_normalization": "Post-Tariff Normalization",
        "post-conflict normalization regime": "Post-Tariff Normalization",
    }
    lower_text = text.lower()
    if lower_text in mapping:
        return mapping[lower_text]
    if "_" in text:
        return text.replace("_", " ").title()
    if text.islower():
        return text.title()
    return text


def event_regime_code_label(value: Any) -> Optional[str]:
    try:
        code = int(value)
    except Exception:
        return None
    mapping = {
        0: "Pre-COVID Normal",
        1: "COVID Shock",
        2: "Post-COVID Recovery",
        3: "Russia-Ukraine War Shock",
        4: "Post-War Inflation Adjustment",
        5: "U.S. Tariff Shock",
        6: "Post-Tariff Normalization",
        7: "Iran Geopolitical Shock",
    }
    return mapping.get(code)


def extract_final_forecast_artifact_event_context(payload: Optional[dict]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not payload:
        return None, None, None
    artifacts = payload.get("artifacts", {}) or {}
    candidate_paths: List[Tuple[Path, str]] = []
    for key, value in artifacts.items():
        if not isinstance(value, str):
            continue
        if any(token in key.lower() for token in ["final_forecast", "final_forecast_blend", "forecast_blend"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
        elif any(token in value.lower() for token in ["final_forecast", "final_forecast_blend", "forecast_blend"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
    artifact_dir = payload.get("artifact_dir")
    if artifact_dir:
        artifact_dir_path = Path(artifact_dir)
        if artifact_dir_path.exists() and artifact_dir_path.is_dir():
            for file_path in sorted(artifact_dir_path.iterdir()):
                if file_path.is_file() and any(token in file_path.name.lower() for token in ["final_forecast", "final_forecast_blend", "forecast_blend"]):
                    candidate_paths.append((file_path, f"artifact_dir/{file_path.name}"))
    seen = set()
    unique_paths: List[Tuple[Path, str]] = []
    for path, source in candidate_paths:
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            unique_paths.append((path, source))
    columns = [
        "event_regime_description",
        "event_regime",
        "macro_event_regime",
        "current_macro_regime",
        "regime_description",
        "event_regime_code",
    ]
    for path, source in unique_paths:
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        for col in columns:
            if col in df.columns:
                series = df[col].dropna().astype(str).map(str.strip)
                series = series[series.apply(is_valid_event_context_text)]
                if not series.empty:
                    value = series.iloc[-1]
                    if col == "event_regime_code":
                        label = event_regime_code_label(value)
                    else:
                        label = format_event_context_label(value)
                    if label:
                        return label, f"{source}.{col}", "Macro-event context assigned from final forecast artifact"
    return None, None, None


def extract_event_descriptive_artifact_context(payload: Optional[dict], base_date: Optional[pd.Timestamp]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not payload or base_date is None or pd.isna(base_date):
        return None, None, None
    artifacts = payload.get("artifacts", {}) or {}
    candidate_paths: List[Tuple[Path, str]] = []
    for key, value in artifacts.items():
        if not isinstance(value, str):
            continue
        if any(token in key.lower() for token in ["event_descriptive", "event_regime_descriptive", "event_regime_descriptive_stats"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
        elif any(token in value.lower() for token in ["event_descriptive", "event_regime_descriptive", "event_regime_descriptive_stats"]):
            candidate_paths.append((Path(value), f"payload.artifacts[{key}]"))
    artifact_dir = payload.get("artifact_dir")
    if artifact_dir:
        artifact_dir_path = Path(artifact_dir)
        if artifact_dir_path.exists() and artifact_dir_path.is_dir():
            for file_path in sorted(artifact_dir_path.iterdir()):
                if file_path.is_file() and any(token in file_path.name.lower() for token in ["event_descriptive", "event_regime_descriptive", "event_regime_descriptive_stats"]):
                    candidate_paths.append((file_path, f"artifact_dir/{file_path.name}"))
    seen = set()
    unique_paths: List[Tuple[Path, str]] = []
    for path, source in candidate_paths:
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            unique_paths.append((path, source))
    for path, source in unique_paths:
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        if not {"start_date", "end_date"}.issubset(df.columns):
            continue
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        valid = df.dropna(subset=["start_date"]).copy()
        if valid.empty:
            continue
        valid["end_date_effective"] = valid["end_date"].fillna(pd.Timestamp.max)
        match = valid[(valid["start_date"] <= base_date) & (valid["end_date_effective"] >= base_date)]
        if match.empty:
            continue
        for col in ["event_regime_description", "event_regime"]:
            if col in match.columns:
                series = match[col].dropna().astype(str).map(str.strip)
                series = series[series.apply(is_valid_event_context_text)]
                if not series.empty:
                    label = format_event_context_label(series.iloc[-1])
                    if label:
                        return label, f"{source}.{col}", "Macro-event context assigned from event descriptive artifact"
    return None, None, None


def get_current_event_context(payload: Optional[dict]) -> dict:
    if not payload:
        return {
            "label": "Not classified",
            "description": "No active macro-event regime was assigned to the latest forecast.",
            "source": "No event context found in forecast payload or artifacts",
            "is_classified": False,
        }
    forecast = payload.get("forecast", {}) or {}
    final_forecast = payload.get("final_forecast", {}) or {}
    payload_checks = [
        ("event_regime_description", lambda x: format_event_context_label(x), "payload.forecast.event_regime_description"),
        ("event_regime", lambda x: format_event_context_label(x), "payload.forecast.event_regime"),
        ("macro_event_regime", lambda x: format_event_context_label(x), "payload.forecast.macro_event_regime"),
        ("current_macro_regime", lambda x: format_event_context_label(x), "payload.forecast.current_macro_regime"),
        ("event_regime_code", event_regime_code_label, "payload.forecast.event_regime_code"),
    ]
    for key, formatter, source in payload_checks:
        value = forecast.get(key)
        if is_valid_event_context_text(value):
            label = formatter(value)
            if label:
                return {
                    "label": label,
                    "description": "Macro-event context assigned from forecast payload.",
                    "source": source,
                    "is_classified": True,
                }
    for key in ("event_regime_description", "event_regime"):
        value = final_forecast.get(key)
        if is_valid_event_context_text(value):
            label = format_event_context_label(value)
            if label:
                return {
                    "label": label,
                    "description": "Macro-event context assigned from final forecast payload.",
                    "source": f"payload.final_forecast.{key}",
                    "is_classified": True,
                }
    artifact_label, artifact_source, artifact_desc = extract_final_forecast_artifact_event_context(payload)
    if artifact_label:
        return {
            "label": artifact_label,
            "description": artifact_desc or "Macro-event context assigned from final forecast artifact.",
            "source": artifact_source or "final forecast artifact",
            "is_classified": True,
        }
    base_date = forecast.get("base_date") or forecast.get("forecast_date") or forecast.get("date")
    if base_date is not None:
        base_date = pd.to_datetime(base_date, errors="coerce")
    desc_label, desc_source, desc_desc = extract_event_descriptive_artifact_context(payload, base_date)
    if desc_label:
        return {
            "label": desc_label,
            "description": desc_desc or "Macro-event context assigned from event descriptive artifact.",
            "source": desc_source or "event descriptive artifact",
            "is_classified": True,
        }
    return {
        "label": "Not classified",
        "description": "No active macro-event regime was assigned to the latest forecast.",
        "source": "No event context found in forecast payload or artifacts",
        "is_classified": False,
    }


def get_event_context_details(payload: Optional[dict]) -> Tuple[str, str]:
    event_context = get_current_event_context(payload)
    return event_context["label"], event_context["source"]


def get_current_event_regime(payload: Optional[dict]) -> str:
    return get_current_event_context(payload)["label"]


def financial_model_label(model_name: Any) -> str:
    text = str(model_name or "").lower()
    for key, label in MODEL_LABELS.items():
        if key in text:
            return label
    return str(model_name or "Model")


def get_forecast_value(payload: Optional[dict]) -> Optional[float]:
    if not payload:
        return None
    forecast = payload.get("forecast", {}) or {}
    for key in ("final_forecast_blend", "pred_blend", "forecast", "raw_forecast"):
        x = safe_float(forecast.get(key))
        if x is not None:
            return x
    final_df = load_artifact_df(payload, "final_forecast")
    if not final_df.empty:
        for col in ("final_forecast_blend", "pred_blend", "forecast"):
            if col in final_df.columns:
                return safe_float(final_df[col].dropna().iloc[-1])
    return None


def get_high_vol_probability(payload: Optional[dict]) -> Optional[float]:
    if not payload:
        return None
    forecast = payload.get("forecast", {}) or {}
    for key in ("regime_prob_high_vol", "high_vol_probability", "prob_high_vol"):
        x = safe_float(forecast.get(key))
        if x is not None:
            return x
    return None


def get_confidence_band(payload: Optional[dict]) -> Tuple[Optional[float], Optional[float]]:
    if not payload:
        return None, None
    forecast = payload.get("forecast", {}) or {}
    low = safe_float(forecast.get("forecast_p05"))
    high = safe_float(forecast.get("forecast_p95"))
    if low is None:
        low = safe_float(forecast.get("pred_blend_p05"))
    if high is None:
        high = safe_float(forecast.get("pred_blend_p95"))
    return low, high


def get_reference_volatility_series(payload: Optional[dict]) -> pd.Series:
    # Use model-ready target if available. It is the cleanest realized-volatility reference.
    model_ready = get_model_ready_df(payload)
    for col in ("y_target", "target", "realized_volatility"):
        if not model_ready.empty and col in model_ready.columns:
            return pd.to_numeric(model_ready[col], errors="coerce").dropna()
    holdout = get_holdout_df(payload)
    if not holdout.empty and "actual" in holdout.columns:
        return pd.to_numeric(holdout["actual"], errors="coerce").dropna()
    return pd.Series(dtype=float)


def classify_forecast_volatility_regime(payload: Optional[dict]) -> Dict[str, Any]:
    forecast_vol = get_forecast_value(payload)
    high_prob = get_high_vol_probability(payload)
    reference = get_reference_volatility_series(payload)

    if forecast_vol is None or reference.empty:
        return {
            "regime_label": "Unavailable",
            "forecast_volatility": forecast_vol,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "high_vol_probability": high_prob,
            "probability_signal": "Unavailable",
            "method": "Insufficient reference data",
            "recommended_interpretation": "Run a forecast or load saved results to classify the volatility regime.",
        }

    q25 = float(reference.quantile(0.25))
    q50 = float(reference.quantile(0.50))
    q75 = float(reference.quantile(0.75))
    q90 = float(reference.quantile(0.90))

    if forecast_vol < q25:
        label = "Low-volatility regime"
        note = "Forecast volatility is below the lower quartile of the historical realized-volatility distribution."
    elif forecast_vol < q50:
        label = "Normal-volatility regime"
        note = "Forecast volatility is inside the lower-middle part of the historical realized-volatility distribution."
    elif forecast_vol < q75:
        label = "Elevated-volatility regime"
        note = "Forecast volatility is above the historical median but below the high-volatility threshold."
    elif forecast_vol < q90:
        label = "High-volatility regime"
        note = "Forecast volatility is above the 75th percentile and signals elevated hedging sensitivity."
    else:
        label = "Stress-volatility regime"
        note = "Forecast volatility is in the upper tail of the historical distribution. Liquidity and hedge timing require close attention."

    if high_prob is None:
        prob_signal = "Probability unavailable"
    elif high_prob >= 0.60:
        prob_signal = "High-volatility alert"
    elif high_prob >= 0.40:
        prob_signal = "Watch zone"
    else:
        prob_signal = "Low high-volatility probability"

    return {
        "regime_label": label,
        "forecast_volatility": forecast_vol,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "high_vol_probability": high_prob,
        "probability_signal": prob_signal,
        "method": "Empirical realized-volatility quantile classification",
        "recommended_interpretation": note,
    }


def recommended_trading_action(classification: Dict[str, Any]) -> str:
    label = classification.get("regime_label", "Unavailable")
    high_prob = safe_float(classification.get("high_vol_probability"), 0.0) or 0.0
    if label in {"High-volatility regime", "Stress-volatility regime"} or high_prob >= 0.60:
        return "Defensive hedge stance. Reduce open exposure, prioritize hedge execution, and keep liquidity risk under close review."
    if label == "Elevated-volatility regime" or (high_prob >= 0.40 and high_prob < 0.60):
        return "Selective hedge stance. Tighten exposure limits and consider partial hedges while monitoring market drivers."
    if label in {"Normal-volatility regime", "Low-volatility regime"} and high_prob < 0.40:
        return "Maintain standard exposure limits. Monitor routine market drivers and keep risk alert triggers active."
    if label in {"Normal-volatility regime", "Low-volatility regime"}:
        return "Maintain standard exposure discipline while watching for any shift toward elevated risk."
    return "Load or run the model to generate a decision signal."


def risk_stance_label(classification: Dict[str, Any]) -> str:
    label = classification.get("regime_label", "Unavailable")
    high_prob = safe_float(classification.get("high_vol_probability"), 0.0) or 0.0
    if label in {"Stress-volatility regime", "High-volatility regime"} or high_prob >= 0.60:
        return "Defensive hedge stance"
    if label == "Elevated-volatility regime" or (high_prob >= 0.40 and high_prob < 0.60):
        return "Selective hedge stance"
    if label in {"Normal-volatility regime", "Low-volatility regime"}:
        return "Standard monitoring stance"
    return "No signal"


def build_trader_decision_summary(regime_label: str, high_vol_probability: Optional[float]) -> str:
    label = str(regime_label or "").lower()
    prob = safe_float(high_vol_probability)
    if prob is not None:
        if ("low" in label or "normal" in label) and prob < 0.40:
            return "Risk stance is stable. Maintain standard exposure limits and monitor routine USD/TND drivers."
        if "elevated" in label or (prob >= 0.40 and prob < 0.60):
            return "Risk stance is selective. Tighten exposure limits and consider partial hedges while monitoring global dollar and oil-risk signals."
        if "high" in label or "stress" in label or prob >= 0.60:
            return "Risk stance is defensive. Prioritize hedge execution, reduce open exposure, and monitor liquidity conditions closely."
    if "low" in label or "normal" in label:
        return "Risk stance is stable. Maintain standard exposure limits and monitor routine USD/TND drivers."
    if "elevated" in label:
        return "Risk stance is selective. Tighten exposure limits and consider partial hedges while monitoring global dollar and oil-risk signals."
    if "high" in label or "stress" in label:
        return "Risk stance is defensive. Prioritize hedge execution, reduce open exposure, and monitor liquidity conditions closely."
    return "Risk stance is not available. Review the latest model signal and macro-event context before taking action."


def get_trading_stance_payload(regime_label: str, high_vol_probability: Optional[float]) -> dict:
    label = str(regime_label or "").lower()
    prob = safe_float(high_vol_probability)
    if ("high" in label or "stress" in label) or (prob is not None and prob >= 0.60):
        return {
            "stance": "Defensive hedge stance",
            "guidance": "Prioritize hedge execution, reduce open exposure, and monitor liquidity conditions closely.",
            "severity": "defensive",
        }
    if "elevated" in label or (prob is not None and 0.40 <= prob < 0.60):
        return {
            "stance": "Selective hedge stance",
            "guidance": "Tighten exposure limits and consider partial hedges while monitoring market drivers.",
            "severity": "watch",
        }
    if "low" in label or "normal" in label or prob is None:
        return {
            "stance": "Standard monitoring stance",
            "guidance": "Maintain standard exposure limits and monitor routine USD/TND market drivers.",
            "severity": "stable",
        }
    return {
        "stance": "Standard monitoring stance",
        "guidance": "Maintain standard exposure limits and monitor routine USD/TND market drivers.",
        "severity": "stable",
    }


def build_risk_interpretation(regime_label: str, forecast_vol: Optional[float], thresholds: dict) -> str:
    regime = str(regime_label or "").lower()
    if "low" in regime:
        return "Forecast volatility is below the lower historical quartile. Market conditions are calm, but routine monitoring remains necessary."
    if "normal" in regime:
        return "Forecast volatility remains within the central historical range. Standard monitoring is appropriate."
    if "elevated" in regime:
        return "Forecast volatility is above the historical median but below the high-volatility threshold. Hedge timing should be monitored closely."
    if "high" in regime:
        return "Forecast volatility is above the high-volatility threshold. Hedge execution risk and exposure sensitivity are elevated."
    if "stress" in regime:
        return "Forecast volatility is in the historical upper tail. Defensive hedge execution and liquidity monitoring are required."
    return "Forecast regime is not available. Review model outputs and market conditions before taking hedging decisions."


def build_confidence_band_text(forecast_p05: Optional[float], forecast_p95: Optional[float], forecast_vol: Optional[float]) -> str:
    if forecast_p05 is None or forecast_p95 is None:
        return "Forecast dispersion cannot be assessed."
    band_width = forecast_p95 - forecast_p05
    relative_width = band_width / max(safe_float(forecast_vol) or 1e-8, 1e-8)
    if relative_width < 0.8:
        return "Forecast dispersion is contained."
    if relative_width < 1.5:
        return "Forecast dispersion is moderate."
    return "Forecast dispersion is wide; use the upper band for stress exposure planning."


def build_trading_desk_takeaway(
    forecast_vol: Optional[float],
    regime_label: str,
    high_vol_probability: Optional[float],
    forecast_p05: Optional[float],
    forecast_p95: Optional[float],
    event_context: dict,
    thresholds: dict,
) -> List[str]:
    prob_text = format_pct(high_vol_probability) if safe_float(high_vol_probability) is not None else "unavailable"
    p05_text = format_pct(forecast_p05)
    p95_text = format_pct(forecast_p95)
    paragraph_1 = (
        f"The official 3-day USD/TND volatility forecast stands at {format_pct(forecast_vol)}, placing the pair in a {regime_label}. "
        f"The high-volatility probability is {prob_text}, and the 90% forecast band ranges from {p05_text} to {p95_text}."
    )
    paragraph_2 = build_risk_interpretation(regime_label, forecast_vol, thresholds)
    regime = str(regime_label or "").lower()
    if ("low" in regime or "normal" in regime) and (safe_float(high_vol_probability) is None or safe_float(high_vol_probability) < 0.40):
        paragraph_3 = "Best practice in this regime is to maintain standard monitoring, avoid unnecessary hedge acceleration, and continue tracking DXY, Brent, MOVE, VIX, and local FX liquidity."
    elif "elevated" in regime or (safe_float(high_vol_probability) is not None and 0.40 <= safe_float(high_vol_probability) < 0.60):
        paragraph_3 = "Best practice in this regime is to tighten exposure monitoring, consider partial hedges for near-term USD/TND exposures, and review DXY, Brent, MOVE, VIX, global sentiment, and macro-event headlines before executing large FX transactions."
    else:
        paragraph_3 = "Best practice in this regime is to prioritize hedge execution, reduce uncovered short-term exposure, monitor liquidity conditions, and use the upper confidence band for stress exposure planning."
    if event_context.get("is_classified"):
        paragraph_3 = f"{paragraph_3} Current macro-event context: {event_context.get('label')}."
    else:
        paragraph_3 = f"{paragraph_3} No active macro-event regime is assigned to the latest forecast."
    return [paragraph_1, paragraph_2, paragraph_3]


# ============================================================
# Data upload and run helpers
# ============================================================


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schema() -> dict:
    return get_expected_input_schema()


def classify_scenario_against_thresholds(classification: Dict[str, Any]) -> Dict[str, Any]:
    q25 = safe_float(classification.get("q25"))
    q50 = safe_float(classification.get("q50"))
    q75 = safe_float(classification.get("q75"))
    q90 = safe_float(classification.get("q90"))
    forecast_vol = safe_float(classification.get("forecast_volatility"))
    if forecast_vol is None or any(v is None for v in [q25, q50, q75, q90]):
        classification["regime_label"] = "Unavailable"
        return classification
    if forecast_vol < q25:
        label = "Low-volatility regime"
        interp = "Scenario volatility is below the lower quartile of realized USD/TND volatility."
    elif forecast_vol < q50:
        label = "Normal-volatility regime"
        interp = "Scenario volatility remains within the lower-middle historical range."
    elif forecast_vol < q75:
        label = "Elevated-volatility regime"
        interp = "Scenario volatility is above the historical median but below the high-volatility threshold."
    elif forecast_vol < q90:
        label = "High-volatility regime"
        interp = "Scenario volatility is above the high-volatility threshold."
    else:
        label = "Stress-volatility regime"
        interp = "Scenario volatility is in the upper tail of the historical distribution."
    classification["regime_label"] = label
    classification["recommended_interpretation"] = interp
    prob = safe_float(classification.get("high_vol_probability"))
    if prob is not None:
        if prob >= 0.60:
            signal = "High-volatility alert"
        elif prob >= 0.40:
            signal = "Watch zone"
        else:
            signal = "Low high-volatility probability"
        classification["probability_signal"] = signal
    return classification


def normalize_schema_columns(schema: dict) -> Dict[str, str]:
    alias_to_canonical: Dict[str, str] = {}
    for col in schema.get("required_columns", []):
        alias_to_canonical[str(col).strip().lower()] = col
    for canonical, aliases in schema.get("accepted_aliases", {}).items():
        for alias in aliases:
            alias_to_canonical[str(alias).strip().lower()] = canonical
    return alias_to_canonical


def missing_uploaded_columns(df: pd.DataFrame, schema: dict) -> List[str]:
    alias_to_canonical = normalize_schema_columns(schema)
    present = {
        alias_to_canonical.get(str(col).strip().lower(), str(col).strip())
        for col in df.columns
    }
    return [col for col in schema.get("required_columns", []) if col not in present]


def validate_uploaded_file(uploaded_file: Any) -> None:
    if uploaded_file is None:
        return
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError("Upload must be an Excel file with extension .xlsx or .xls.")


def read_uploaded_excel(uploaded_file: Any) -> pd.DataFrame:
    schema = fetch_schema()
    return pd.read_excel(BytesIO(uploaded_file.getvalue()), sheet_name=schema.get("sheet_name", "Sheet1"))


def run_forecast(uploaded_file: Any, progress_callback: Optional[Any] = None) -> dict:
    run_id = local_run_id()
    run_dir = FORECAST_ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if uploaded_file is not None:
        validate_uploaded_file(uploaded_file)
        safe_upload_name = Path(uploaded_file.name).name
        input_path = run_dir / safe_upload_name
        input_path.write_bytes(uploaded_file.getvalue())
    else:
        input_path = default_input_path()
        if not input_path.exists():
            raise FileNotFoundError(
                "No Excel file was uploaded and the default input file "
                f"'{RAW_FILE_PATH}' was not found in the app repository. "
                "Upload the dataset from the Data & Market Inputs page, or add "
                "Final_data.xlsx to the GitHub repository root."
            )

    validate_input_file(input_path)
    result = run_pipeline(
        input_path=input_path,
        output_dir=run_dir,
        save_outputs=True,
        progress_callback=progress_callback,
    )
    payload = result_to_payload(result)
    payload["run_id"] = run_id
    payload["run_datetime"] = datetime.now(timezone.utc).isoformat()
    payload["artifacts"] = {key: str(path) for key, path in result.artifacts.items()}
    payload["artifact_dir"] = str(run_dir)
    base_date = infer_base_date_from_input_file(input_path)
    if base_date:
        payload["base_date"] = base_date
        payload.setdefault("forecast", {})["base_date"] = base_date
    save_dashboard_payload(payload, run_dir)
    return payload


def render_expected_schema(schema: dict) -> None:
    with st.expander("Expected Excel schema", expanded=False):
        st.caption(f"Expected sheet: {schema.get('sheet_name', 'Sheet1')}")
        st.write(", ".join(schema.get("required_columns", [])))
        aliases = schema.get("accepted_aliases", {})
        if aliases:
            alias_rows = [
                {"Canonical column": key, "Accepted names": ", ".join(value)}
                for key, value in aliases.items()
            ]
            st.dataframe(pd.DataFrame(alias_rows), hide_index=True, use_container_width=True)


def render_upload_preview(uploaded_file: Any, schema: dict) -> Dict[str, Any]:
    state: Dict[str, Any] = {"df": None, "missing": [], "error": None}
    if uploaded_file is None:
        default_path = default_input_path()
        if default_path.exists():
            st.info(f"No upload selected. The app will use the default `{RAW_FILE_PATH}` included in the repository.")
        else:
            st.warning(
                "No upload selected and no default `Final_data.xlsx` was found in the repository. "
                "Upload an Excel dataset before running the model."
            )
        return state
    try:
        df = read_uploaded_excel(uploaded_file)
    except Exception as exc:
        state["error"] = f"Could not read uploaded Excel file: {exc}"
        st.error(state["error"])
        return state

    missing = missing_uploaded_columns(df, schema)
    state["df"] = df
    state["missing"] = missing
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna() if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
    total_missing = int(df.isna().sum().sum())
    total_cells = max(1, int(df.shape[0] * df.shape[1]))
    missing_pct = 100.0 * total_missing / total_cells
    date_range = f"{dates.min().date()} to {dates.max().date()}" if not dates.empty else "Unavailable"
    schema_status = "Passed" if not missing else "Failed"
    data_quality = "Good" if missing_pct < 5 else "Review" if missing_pct < 15 else "Poor"

    render_kpi_grid([
        metric_card("Rows", f"{len(df):,}", "Input observations", "primary"),
        metric_card("Columns", f"{df.shape[1]:,}", "Market variables", "teal"),
        metric_card("Date Range", date_range, "Input coverage", "gold"),
        metric_card("Data Quality", data_quality, f"Missing values: {missing_pct:.1f}%", "green" if data_quality == "Good" else "gold"),
    ])
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        st.success(f"Schema validation {schema_status}. All required columns are available.")

    with st.expander("Dataset preview and diagnostics", expanded=False):
        st.dataframe(df.head(), use_container_width=True)
        col_summary = []
        for col in df.columns:
            col_summary.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing": int(df[col].isna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
            })
        st.dataframe(pd.DataFrame(col_summary), hide_index=True, use_container_width=True)
    return state


# ============================================================
# Chart helpers
# ============================================================


def line_fig(df: pd.DataFrame, x: str, y_cols: List[Tuple[str, str]], title: str = "", y_scale: float = 1.0, yaxis_title: str = "Value") -> go.Figure:
    fig = go.Figure()
    for col, label in y_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=pd.to_numeric(df[col], errors="coerce") * y_scale, mode="lines", name=label))
    fig.update_layout(title=title, yaxis_title=yaxis_title)
    return apply_finance_layout(fig, height=390)


def render_regime_gauge_html(
    forecast_vol: float | None,
    calm_threshold: float | None,
    median_volatility_line: float | None,
    high_volatility_threshold: float | None,
    stress_threshold: float | None,
    assigned_regime: str,
    high_vol_probability: float | None = None,
) -> None:
    if any(v is None for v in [forecast_vol, calm_threshold, median_volatility_line, high_volatility_threshold, stress_threshold]):
        st.info("Regime thresholds are unavailable until a forecast payload and reference volatility history are loaded.")
        return

    active_colors = {
        "Low-volatility regime": "#14b8a6",
        "Normal-volatility regime": "#3b82f6",
        "Elevated-volatility regime": "#f59e0b",
        "High-volatility regime": "#f97316",
        "Stress-volatility regime": "#ef4444",
    }
    active_color = active_colors.get(assigned_regime, FINANCE_COLORS["slate"])
    x_max = max(stress_threshold * 1.20, forecast_vol * 1.20, 1e-8)
    calm_pct = min(100, max(0, calm_threshold / x_max * 100))
    median_pct = min(100, max(0, median_volatility_line / x_max * 100))
    high_pct = min(100, max(0, high_volatility_threshold / x_max * 100))
    stress_pct = min(100, max(0, stress_threshold / x_max * 100))
    forecast_pct = min(97, max(3, forecast_vol / x_max * 100))

    low_width = max(0, calm_pct)
    normal_width = max(0, median_pct - calm_pct)
    elevated_width = max(0, high_pct - median_pct)
    high_width = max(0, stress_pct - high_pct)
    stress_width = max(0, 100 - stress_pct)

    calm_pos = min(97, max(3, calm_pct))
    median_pos = min(97, max(3, median_pct))
    high_pos = min(97, max(3, high_pct))
    stress_pos = min(97, max(3, stress_pct))

    low_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Low-volatility regime" else ""
    normal_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Normal-volatility regime" else ""
    elevated_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Elevated-volatility regime" else ""
    high_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "High-volatility regime" else ""
    stress_highlight = "box-shadow: inset 0 0 0 2px rgba(255,255,255,0.16);" if assigned_regime == "Stress-volatility regime" else ""

    html = f"""
    <style>
        .regime-gauge-card {{
            --active-regime-color: {active_color};
            background: linear-gradient(180deg, rgba(7, 16, 29, 0.99), rgba(14, 20, 36, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 28px;
            padding: 32px;
            margin: 8px 0;
            box-shadow: 0 28px 60px rgba(0, 0, 0, 0.22);
            transition: border-color 240ms ease, box-shadow 240ms ease, transform 240ms ease;
        }}
        .regime-gauge-card:hover {{
            border-color: var(--active-regime-color);
            box-shadow: 0 32px 72px rgba(0, 0, 0, 0.28), 0 0 0 1px rgba(255, 255, 255, 0.05);
            transform: translateY(-2px);
        }}
        .regime-gauge-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
            margin-bottom: 26px;
        }}
        .regime-metric-box {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 16px 18px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.75), rgba(20, 30, 48, 0.55));
            box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.06), 0 4px 12px rgba(0, 0, 0, 0.12);
            flex: 0 1 auto;
            transition: border-color 180ms ease, box-shadow 180ms ease;
        }}
        .regime-metric-box:hover {{
            border-color: rgba(148, 163, 184, 0.35);
            box-shadow: inset 0 1px 3px rgba(255, 255, 255, 0.08), 0 6px 16px rgba(0, 0, 0, 0.15);
        }}
        .regime-metric-label {{
            font-size: 0.68rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #a8b6cc;
            font-weight: 700;
        }}
        .regime-metric-value {{
            font-size: 1.48rem;
            font-weight: 900;
            color: #f8fafc;
            line-height: 1.1;
        }}
        .regime-pill {{
            display: inline-flex;
            align-items: center;
            padding: 9px 16px;
            border-radius: 999px;
            background: {active_color};
            color: #08111f;
            font-size: 0.85rem;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.2);
            white-space: nowrap;
            flex: 0 0 auto;
        }}
        .regime-gauge-body {{
            position: relative;
            padding-top: 12px;
        }}
        .regime-bar {{
            position: relative;
            display: flex;
            width: 100%;
            height: 94px;
            border-radius: 52px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.3), inset 0 -2px 6px rgba(255, 255, 255, 0.08);
        }}
        .regime-segment {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 800;
            color: #f8fafc;
            white-space: nowrap;
            min-width: 0;
            padding: 0 12px;
        }}
        .regime-segment.low {{ background: linear-gradient(135deg, rgba(20,184,166,0.52), rgba(20,184,166,0.28)); }}
        .regime-segment.normal {{ background: linear-gradient(135deg, rgba(59,130,246,0.50), rgba(59,130,246,0.25)); }}
        .regime-segment.elevated {{ background: linear-gradient(135deg, rgba(245,158,11,0.58), rgba(245,158,11,0.28)); }}
        .regime-segment.high {{ background: linear-gradient(135deg, rgba(249,115,22,0.50), rgba(249,115,22,0.24)); }}
        .regime-segment.stress {{ background: linear-gradient(135deg, rgba(239,68,68,0.48), rgba(127,29,29,0.35)); }}
        .regime-segment.active {{
            filter: brightness(1.18) saturate(1.16);
            box-shadow: inset 0 0 22px rgba(255,255,255,0.15), 0 0 16px rgba(255,255,255,0.08);
        }}
        .forecast-pill {{
            position: absolute;
            top: 50%;
            left: var(--forecast-position);
            transform: translate(-50%, -50%);
            padding: 6px 11px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.24);
            background: {active_color};
            color: #08111f;
            font-size: 0.78rem;
            font-weight: 900;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
            white-space: nowrap;
            z-index: 5;
            min-width: auto;
            text-align: center;
        }}
        .threshold-tick {{
            position: absolute;
            top: 10px;
            bottom: 10px;
            width: 1px;
            background: rgba(226, 232, 240, 0.55);
            z-index: 4;
        }}
        .threshold-label-row {{
            position: relative;
            height: 68px;
            margin-top: 1.2rem;
        }}
        .threshold-label {{
            position: absolute;
            transform: translateX(-50%);
            text-align: center;
            color: #cbd5e1;
            font-size: 0.82rem;
            line-height: 1.4;
            white-space: nowrap;
            max-width: 140px;
            padding: 0 6px;
        }}
        .threshold-label strong {{ display: block; color: #f0f4f8; font-size: 0.92rem; font-weight: 700; margin-bottom: 5px; }}
    </style>
    <div class="regime-gauge-card" style="--active-regime-color: {active_color};">
        <div class="regime-gauge-header">
            <div class="regime-metric-box">
                <div class="regime-metric-label">Forecast Volatility</div>
                <div class="regime-metric-value">{format_pct(forecast_vol)}</div>
            </div>
            <div class="regime-metric-box">
                <div class="regime-metric-label">High-Vol Probability</div>
                <div class="regime-metric-value">{format_pct(high_vol_probability) if high_vol_probability is not None else 'N/A'}</div>
            </div>
            <div class="regime-pill"><span>{assigned_regime}</span></div>
        </div>
        <div class="regime-gauge-body">
            <div class="regime-bar" style="--forecast-position: {forecast_pct}%;">
                <div class="regime-segment low{' active' if assigned_regime == 'Low-volatility regime' else ''}" style="width: {low_width}%; {low_highlight}">Low</div>
                <div class="regime-segment normal{' active' if assigned_regime == 'Normal-volatility regime' else ''}" style="width: {normal_width}%; {normal_highlight}">Normal</div>
                <div class="regime-segment elevated{' active' if assigned_regime == 'Elevated-volatility regime' else ''}" style="width: {elevated_width}%; {elevated_highlight}">Elevated</div>
                <div class="regime-segment high{' active' if assigned_regime == 'High-volatility regime' else ''}" style="width: {high_width}%; {high_highlight}">High</div>
                <div class="regime-segment stress{' active' if assigned_regime == 'Stress-volatility regime' else ''}" style="width: {stress_width}%; {stress_highlight}">Stress</div>
                <div class="threshold-tick" style="left: {calm_pos}%"></div>
                <div class="threshold-tick" style="left: {median_pos}%"></div>
                <div class="threshold-tick" style="left: {high_pos}%"></div>
                <div class="threshold-tick" style="left: {stress_pos}%"></div>
                <div class="forecast-pill">{format_pct(forecast_vol)}</div>
            </div>
            <div class="threshold-label-row">
                <div class="threshold-label" style="left: {calm_pos}%;"><strong>Calm threshold</strong>{format_pct(calm_threshold)}</div>
                <div class="threshold-label" style="left: {median_pos}%;"><strong>Median volatility</strong>{format_pct(median_volatility_line)}</div>
                <div class="threshold-label" style="left: {high_pos}%;"><strong>High threshold</strong>{format_pct(high_volatility_threshold)}</div>
                <div class="threshold-label" style="left: {stress_pos}%;"><strong>Stress threshold</strong>{format_pct(stress_threshold)}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def plot_forecast_vs_actual(payload: Optional[dict]) -> None:
    holdout = get_holdout_df(payload)
    if holdout.empty or not {"Date", "actual", "pred_blend"}.issubset(holdout.columns):
        st.info("Forecast-vs-actual history is not available.")
        return
    holdout["Date"] = pd.to_datetime(holdout["Date"], errors="coerce")
    fig = go.Figure()
    if {"pred_blend_p05", "pred_blend_p95"}.issubset(holdout.columns):
        fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["pred_blend_p95"] * 100, mode="lines", line=dict(width=0), showlegend=False, name="Upper band"))
        fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["pred_blend_p05"] * 100, mode="lines", fill="tonexty", fillcolor="rgba(47,216,201,0.12)", line=dict(width=0), name="90% confidence band"))
    fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["actual"] * 100, mode="lines", name="Actual realized volatility", line=dict(width=2.4)))
    fig.add_trace(go.Scatter(x=holdout["Date"], y=holdout["pred_blend"] * 100, mode="lines", name="Final Macro-Event FX Volatility Engine", line=dict(width=2.7)))
    fig = apply_finance_layout(fig, height=430)
    fig.update_layout(
        title=dict(
            text="Forecast vs Realized USD/TND Volatility",
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=18),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
        margin=dict(l=45, r=35, t=105, b=45),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page renderers
# ============================================================


def render_no_payload_choice() -> None:
    st.markdown(
        """
        <div class="hero-panel">
            <div class="page-kicker">No forecast loaded</div>
            <div class="page-title">Choose how to start the trading risk view</div>
            <p class="small-muted">Load the latest saved run for instant monitoring, or run a new forecast from the Data & Market Inputs page.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>Load latest saved result</div><div class='panel-subtitle'>Open the most recent local artifact without rerunning the model.</div>", unsafe_allow_html=True)
        latest = find_latest_run_dir()
        if latest:
            st.caption(f"Latest saved run: {latest.name}")
        else:
            st.caption("No saved local run detected.")
        if st.button("Load latest risk view", type="primary", use_container_width=True, key="load_latest_from_empty"):
            try:
                payload = load_latest_local()
                st.session_state.forecast_payload = payload
                st.session_state.selected_run_id = payload.get("run_id")
                st.session_state.artifact_dir = payload.get("artifact_dir")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>Run new forecast</div><div class='panel-subtitle'>Validate market data and generate a fresh 3-day USD/TND volatility outlook.</div>", unsafe_allow_html=True)
        if st.button("Go to Data & Market Inputs", use_container_width=True, key="go_data_inputs_from_empty"):
            st.session_state.selected_page = "Data & Market Inputs"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_executive_fx_risk_cockpit(payload: Optional[dict]) -> None:
    # Cockpit-specific layout overrides for compactness
    st.markdown(
        """
        <style>
        /* Cockpit compact layout overrides */
        .hero-panel {
            padding: 1.1rem;
            margin-bottom: 0.8rem;
        }
        .kpi-card {
            min-height: 100px;
            padding: 0.9rem;
        }
        .decision-card {
            min-height: 115px;
            padding: 0.9rem;
            margin-bottom: 0.7rem;
        }
        .decision-card.decision-strip {
            min-height: 64px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.6rem;
        }
        .decision-card.decision-strip .decision-title {
            font-size: 0.72rem;
        }
        .decision-card.severity-stable {
            border-left-color: var(--finance-green);
        }
        .decision-card.severity-watch {
            border-left-color: var(--finance-gold);
        }
        .decision-card.severity-defensive {
            border-left-color: var(--finance-red);
        }
        .decision-card.trading-desk-takeaway p {
            margin: 0.35rem 0;
            line-height: 1.55;
            max-width: 1500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    page_header("Executive FX Risk Cockpit", "Decision layer", payload)
    if payload is None:
        render_no_payload_choice()
        return

    classification = classify_forecast_volatility_regime(payload)
    forecast_vol = classification.get("forecast_volatility")
    high_prob = classification.get("high_vol_probability")
    p05, p95 = get_confidence_band(payload)
    event_context = get_current_event_context(payload)
    stance_payload = get_trading_stance_payload(classification.get("regime_label", "Unavailable"), high_prob)
    decision_summary = build_trader_decision_summary(classification.get("regime_label", "Unavailable"), high_prob)
    interpretation = build_risk_interpretation(classification.get("regime_label", "Unavailable"), forecast_vol, classification)
    confidence_note = build_confidence_band_text(p05, p95, forecast_vol)
    band = f"{format_pct(p05)} - {format_pct(p95)}" if p05 is not None or p95 is not None else "n/a"
    event_note = "Calendar-known macro-event context" if event_context["is_classified"] else "No active macro-event regime assigned"

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="page-kicker">Final Macro-Event FX Volatility Engine</div>
            <div class="page-title">USD/TND 3-Day Volatility Risk Signal</div>
            <p class="small-muted">Official production forecast for hedge timing, exposure limits, and short-term FX risk monitoring.</p>
            <div class="page-chip">Latest macro-event context: {html.escape(event_context['label'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    high_vol_note = "Probability signal unavailable"
    if high_prob is not None:
        if high_prob >= 0.60:
            high_vol_note = "High-volatility alert"
        elif high_prob >= 0.40:
            high_vol_note = "Watch zone"
        else:
            high_vol_note = "Low high-volatility probability"

    render_kpi_grid([
        metric_card("3-Day Volatility Forecast", format_pct(forecast_vol), "Official short-term USD/TND volatility outlook", "teal"),
        metric_card("Forecast Volatility Regime", str(classification.get("regime_label", "Unavailable")), "Empirical realized-volatility threshold classification", "gold"),
        metric_card("High-Volatility Signal", format_pct(high_prob), high_vol_note, "purple"),
        metric_card("Current Macro-Event Context", event_context["label"], event_note, "primary"),
    ])

    st.markdown(
        f"""
        <div class="decision-card decision-strip severity-{stance_payload['severity']}">
            <div class="decision-title">Decision Summary</div>
            <div class="decision-text">{html.escape(decision_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="decision-card severity-{stance_payload['severity']}">
                <div class="decision-title">Recommended Trading Stance</div>
                <div class="decision-text"><strong>{html.escape(stance_payload['stance'])}</strong><br>{html.escape(stance_payload['guidance'])}</div>
            </div>
            <div class="decision-card">
                <div class="decision-title">Risk interpretation</div>
                <div class="decision-text">{html.escape(interpretation)}</div>
            </div>
            <div class="decision-card">
                <div class="decision-title">Forecast Confidence Band</div>
                <div class="decision-text">90% forecast interval: <strong>{html.escape(band)}</strong>. {html.escape(confidence_note)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_recent_volatility_pulse(payload)
    with right:
        render_market_watchlist(payload, event_context)
        render_action_checklist()

    takeaway_paragraphs = build_trading_desk_takeaway(
        forecast_vol,
        classification.get("regime_label", "Unavailable"),
        high_prob,
        p05,
        p95,
        event_context,
        classification,
    )
    takeaway_html = "".join(f"<p class='decision-text'>{html.escape(paragraph)}</p>" for paragraph in takeaway_paragraphs)
    st.markdown(
        f"""
        <div class='decision-card trading-desk-takeaway severity-{stance_payload['severity']}'>
            <div class='decision-title'>Trading Desk Takeaway</div>
            {takeaway_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    render_navigation_cards()


def render_data_market_inputs(payload: Optional[dict]) -> None:
    page_header("Data & Market Inputs", "Workflow layer", payload)
    schema = fetch_schema()

    st.markdown("<div class='method-box'><strong>Workflow choice.</strong> Load the latest saved risk view for monitoring, or run a new forecast after validating the input dataset.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>Saved local results</div><div class='panel-subtitle'>Open the latest generated model artifacts.</div>", unsafe_allow_html=True)
        latest = find_latest_run_dir()
        if latest:
            st.write(f"Latest saved run: **{latest.name}**")
        else:
            st.info("No saved run detected yet.")
        if st.button("Load latest saved results", type="primary", use_container_width=True):
            try:
                loaded = load_latest_local()
                st.session_state.forecast_payload = loaded
                st.session_state.selected_run_id = loaded.get("run_id")
                st.session_state.artifact_dir = loaded.get("artifact_dir")
                st.success("Latest saved risk view loaded.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>Official model run</div><div class='panel-subtitle'>Run the validated macro-event engine with the current forecasting pipeline.</div>", unsafe_allow_html=True)
        st.write("Default input file:")
        resolved_default_input = default_input_path()
        st.code(str(RAW_FILE_PATH))
        if resolved_default_input.exists():
            st.caption("Default dataset found in the repository.")
        else:
            st.caption("Default dataset not found. Upload an Excel dataset before running in Community Cloud.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    render_expected_schema(schema)
    uploaded_file = st.file_uploader("Upload Excel market dataset", type=["xlsx", "xls"], key="upload_file")
    upload_state = render_upload_preview(uploaded_file, schema)

    st.markdown("---")
    st.subheader("Run official forecast")
    st.info("The official run uses the forecasting configuration embedded in `forecasting.py`. Scenario Laboratory is visual-only in this app version.")
    no_input_available = uploaded_file is None and not default_input_path().exists()
    run_disabled = bool(upload_state.get("missing") or upload_state.get("error") or no_input_available)
    if no_input_available:
        st.warning("Upload the Excel market dataset before running the forecast on Community Cloud.")
    progress_box = st.empty()
    if st.button("Run Final Macro-Event FX Volatility Engine", type="primary", disabled=run_disabled, use_container_width=True):
        try:
            def streamlit_progress(message: str) -> None:
                progress_box.markdown(f"**{html.escape(message)}**")
            with st.spinner("Running the USD/TND macro-event volatility engine..."):
                new_payload = run_forecast(uploaded_file, progress_callback=streamlit_progress)
            st.session_state.forecast_payload = new_payload
            st.session_state.selected_run_id = new_payload.get("run_id")
            st.session_state.artifact_dir = new_payload.get("artifact_dir")
            st.success("Forecast completed and saved to artifacts.")
        except Exception as exc:
            st.error("Forecast failed. See technical details below.")
            with st.expander("Technical details"):
                st.write(str(exc))


def render_forecast_configuration(payload: Optional[dict]) -> None:
    page_header("Forecast Configuration", "Calibration layer", payload)
    st.markdown(
        """
        <div class="method-box">
            <strong>Production principle.</strong> The official forecast is the validated model configuration. User-defined windows and event selections are displayed as scenario assumptions only in this version, not as separate production forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div class='panel'><div class='panel-title'>Official Validated Forecast</div><div class='panel-subtitle'>Configuration used for the production trading signal.</div>", unsafe_allow_html=True)
        official_rows = pd.DataFrame([
            {"Setting": "Model", "Value": "Final Macro-Event FX Volatility Engine"},
            {"Setting": "Forecast horizon", "Value": f"Next {VOL_TARGET_WINDOW} trading days"},
            {"Setting": "Calibration window", "Value": f"{ROLLING_WINDOW} trading days (validated baseline)"},
            {"Setting": "Event features", "Value": "Enabled"},
            {"Setting": "Event feature forcing", "Value": "Enabled"},
            {"Setting": "Classification", "Value": "Empirical quantile thresholds"},
        ])
        st.dataframe(official_rows, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-title'>Scenario Calibration Preview</div><div class='panel-subtitle'>Visual-only assumptions. They do not retrain the model in this app version.</div>", unsafe_allow_html=True)
        preview_window_options = calibration_window_options("Custom window")
        window = st.selectbox(
            "Calibration window assumption",
            preview_window_options,
            index=default_calibration_window_index(preview_window_options),
            key="config_preview_calibration_window_v2",
        )
        events = st.multiselect(
            "Macro-event regimes to emphasize",
            ["COVID shock", "Russia-Ukraine shock", "U.S. tariff shock", "Post-Tariff Normalization", "Iran geopolitical shock"],
            default=[],
            key="config_preview_event_focus_v2",
        )
        mode = st.selectbox("Scenario interpretation", ["All data with event features", "Crisis-regime overweighting", "Selected-event sample - exploratory"])
        st.markdown(f"<div class='warning-box'>Scenario selected: <strong>{html.escape(window)}</strong>, mode: <strong>{html.escape(mode)}</strong>. This is a research view only and does not replace the official forecast.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_high_volatility_calibration_diagnostics(payload: Optional[dict]) -> None:
    summary = get_high_vol_calibration_df(payload, "summary")
    reliability = get_high_vol_calibration_df(payload, "reliability_curve")
    threshold_table = get_high_vol_calibration_df(payload, "threshold_sensitivity")

    st.markdown(
        "<div class='panel'><div class='panel-title'>High-Volatility Classifier Calibration</div>"
        "<div class='panel-subtitle'>Checks whether the model's high-volatility probabilities are reliable and how alert thresholds affect precision and recall.</div>",
        unsafe_allow_html=True,
    )

    if summary.empty and reliability.empty and threshold_table.empty:
        st.info("High-volatility calibration diagnostics are not available for this run.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not summary.empty:
        def summary_value(metric_name: str) -> Optional[float]:
            if "metric" not in summary.columns or "value" not in summary.columns:
                return None
            match = summary[summary["metric"].astype(str).str.lower() == metric_name.lower()]
            if match.empty:
                return None
            return safe_float(match.iloc[0]["value"])

        brier = summary_value("Brier score")
        roc_auc = summary_value("ROC-AUC")
        pr_auc = summary_value("PR-AUC")
        realized_freq = summary_value("Realized high-volatility frequency")
        render_kpi_grid(
            [
                metric_card("Brier Score", format_number(brier, 4), "Lower probability error is better", "teal"),
                metric_card("ROC-AUC", format_number(roc_auc, 3), "Ranking quality across thresholds", "primary"),
                metric_card("PR-AUC", format_number(pr_auc, 3), "Average precision for high-volatility alerts", "gold"),
                metric_card("Realized High-Vol Frequency", format_pct(realized_freq), "Holdout rate above the development q75 cutoff", "purple"),
            ]
        )
        with st.expander("Calibration summary table", expanded=False):
            st.dataframe(summary, hide_index=True, use_container_width=True)

    if not reliability.empty:
        reliability_plot = reliability.copy()
        required_cols = {"mean_predicted_probability", "observed_high_volatility_frequency"}
        if required_cols.issubset(reliability_plot.columns):
            reliability_plot["mean_predicted_probability"] = pd.to_numeric(
                reliability_plot["mean_predicted_probability"], errors="coerce"
            )
            reliability_plot["observed_high_volatility_frequency"] = pd.to_numeric(
                reliability_plot["observed_high_volatility_frequency"], errors="coerce"
            )
            reliability_plot = reliability_plot.dropna(
                subset=["mean_predicted_probability", "observed_high_volatility_frequency"]
            ).sort_values("mean_predicted_probability")
            if not reliability_plot.empty:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=reliability_plot["mean_predicted_probability"],
                        y=reliability_plot["observed_high_volatility_frequency"],
                        mode="lines+markers",
                        name="Model calibration",
                        line=dict(color=FINANCE_COLORS["teal"], width=3),
                        marker=dict(size=8),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Perfect calibration",
                        line=dict(color=FINANCE_COLORS["slate"], dash="dash", width=2),
                    )
                )
                fig.update_layout(
                    title="High-Volatility Reliability Diagram",
                    xaxis_title="Mean predicted high-volatility probability",
                    yaxis_title="Observed high-volatility frequency",
                )
                fig.update_xaxes(range=[0, 1])
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(apply_finance_layout(fig, height=420), use_container_width=True)
                st.caption("A curve close to the diagonal means the predicted probabilities are well calibrated.")

    if not threshold_table.empty:
        threshold_display = threshold_table.copy()
        percent_columns = [
            "alert_threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "false_alert_rate",
            "missed_high_vol_rate",
        ]
        for column in percent_columns:
            if column in threshold_display.columns:
                threshold_display[column] = threshold_display[column].apply(format_pct)
        st.dataframe(threshold_display, hide_index=True, use_container_width=True)
        st.markdown(
            """
            <div class='method-box'>
                Lower thresholds usually increase recall but create more false alerts. Higher thresholds usually increase selectivity but may miss more high-volatility periods.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_volatility_regime_classification(payload: Optional[dict]) -> None:
    page_header("Volatility Regime Classification", "EMPIRICAL THRESHOLD AND VALIDATION LAYER", payload)
    if payload is None:
        render_no_payload_choice()
        return
    st.markdown(
        """
        <div class="method-box">
            <strong>Validation focus.</strong> This page validates how the forecast is translated into an empirical volatility regime.
            The high-volatility probability is evaluated with calibration diagnostics. The Brier score measures probability forecast error, while the reliability diagram compares predicted probabilities with observed high-volatility frequencies. The threshold sensitivity table shows how alert quality changes across different probability cutoffs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    classification = classify_forecast_volatility_regime(payload)
    st.markdown(
        """
        <div style="background: linear-gradient(180deg, rgba(20, 30, 50, 0.6), rgba(15, 23, 42, 0.5)); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 14px; padding: 18px 24px; margin: 8px 0 16px 0; display: flex; justify-content: center; align-items: center;">
            <div style="text-align: center;">
                <div style="font-size: 0.70rem; letter-spacing: 0.15em; text-transform: uppercase; color: #cbd5e1; font-weight: 800; margin-bottom: 4px;">Classification Method</div>
                <div style="font-size: 1.05rem; font-weight: 900; color: #f8fafc; margin-bottom: 2px;">Historical realized-volatility thresholds</div>
                <div style="font-size: 0.78rem; color: #a8b6cc; font-weight: 700;">Empirical quantiles</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_regime_gauge_html(
        forecast_vol=classification.get("forecast_volatility"),
        calm_threshold=classification.get("q25"),
        median_volatility_line=classification.get("q50"),
        high_volatility_threshold=classification.get("q75"),
        stress_threshold=classification.get("q90"),
        assigned_regime=classification.get("regime_label", "Unavailable"),
        high_vol_probability=classification.get("high_vol_probability"),
    )
    st.markdown(
        """
        <div class="method-box">
            <strong>Methodology note.</strong> Regime labels are assigned by comparing the 3-day USD/TND volatility forecast with historical realized-volatility thresholds. These thresholds divide past USD/TND volatility into calm, normal, elevated, high, and stress zones.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical threshold mapping"):
        st.write("Calm threshold = 25th percentile, Median volatility line = 50th percentile, High-volatility threshold = 75th percentile, Stress threshold = 90th percentile.")

    holdout = get_holdout_df(payload)
    if not holdout.empty and {"actual", "pred_blend"}.issubset(holdout.columns):
        actual = pd.to_numeric(holdout["actual"], errors="coerce")
        pred = pd.to_numeric(holdout["pred_blend"], errors="coerce")
        q75 = actual.quantile(0.75)
        actual_high = actual >= q75
        pred_high = pred >= q75
        valid = actual_high.notna() & pred_high.notna()
        if valid.any():
            tp = int((actual_high[valid] & pred_high[valid]).sum())
            fp = int((~actual_high[valid] & pred_high[valid]).sum())
            tn = int((~actual_high[valid] & ~pred_high[valid]).sum())
            fn = int((actual_high[valid] & ~pred_high[valid]).sum())
            accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            accuracy_interpretation = interpret_classification_accuracy(accuracy)
            recall_interpretation = interpret_recall(recall)
            f1_interpretation = interpret_f1_score(f1)

            # Three standalone dynamic validation cards
            col1, col2, col3 = st.columns(3)
            with col1:
                card1_html = (
                    '<div style="background: linear-gradient(135deg, rgba(20, 184, 166, 0.15), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(20, 184, 166, 0.4); border-radius: 12px; padding: 20px; margin: 5px 0; text-align: center; box-shadow: 0 4px 12px rgba(20, 184, 166, 0.1);">'
                    f'<div style="font-size: 2.2rem; font-weight: 900; color: #14b8a6; margin-bottom: 8px;">{format_metric_percent(accuracy)}</div>'
                    '<div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">Classification Accuracy</div>'
                    '<div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">Overall classification hit rate</div>'
                    f'<div class="validation-card-interpretation">{accuracy_interpretation}</div>'
                    '</div>'
                )
                st.markdown(card1_html, unsafe_allow_html=True)
            with col2:
                card2_html = (
                    '<div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 12px; padding: 20px; margin: 5px 0; text-align: center; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);">'
                    f'<div style="font-size: 2.2rem; font-weight: 900; color: #f59e0b; margin-bottom: 8px;">{format_metric_percent(recall)}</div>'
                    '<div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">Recall</div>'
                    '<div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">High-volatility capture rate</div>'
                    f'<div class="validation-card-interpretation">{recall_interpretation}</div>'
                    '</div>'
                )
                st.markdown(card2_html, unsafe_allow_html=True)
            with col3:
                card3_html = (
                    '<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 12px; padding: 20px; margin: 5px 0; text-align: center; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);">'
                    f'<div style="font-size: 2.2rem; font-weight: 900; color: #3b82f6; margin-bottom: 8px;">{format_metric_percent(f1)}</div>'
                    '<div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px;">F1 Score</div>'
                    '<div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">Balanced alert quality</div>'
                    f'<div class="validation-card-interpretation">{f1_interpretation}</div>'
                    '</div>'
                )
                st.markdown(card3_html, unsafe_allow_html=True)

            if precision is None or pd.isna(precision):
                interp_p1 = "The matrix summarizes how many high-volatility periods were captured, missed, or falsely flagged."
            elif recall > precision + 0.05:
                interp_p1 = "The signal prioritizes capturing high-volatility episodes over avoiding every false alert, which is appropriate for risk monitoring because missing stress periods is usually more costly than issuing cautious alerts."
            elif precision > recall + 0.05:
                interp_p1 = "The signal is relatively selective: high-volatility alerts are more reliable, but some stress periods may be missed."
            else:
                interp_p1 = "The signal is balanced between high-volatility capture and alert precision."

            interp_p2 = f"The model captured {tp} high-volatility periods and missed {fn} high-volatility periods in the evaluated sample."

            # Premium 2x2 confusion matrix
            matrix_html = (
                '<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9)); border: 1px solid rgba(96, 165, 250, 0.4); border-radius: 12px; padding: 20px; margin: 15px 0; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);">'
                '<div style="font-size: 1.2rem; color: #f8fafc; font-weight: 800; margin-bottom: 15px; text-align: center;">High-Volatility Signal Validation Matrix</div>'
                '<div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 20px; text-align: center;">Predicted signal versus realized USD/TND volatility outcome.</div>'
                '<div class="validation-matrix-content">'
                '<div class="matrix-left">'
                '<div style="display: grid; grid-template-columns: 120px 1fr 1fr; grid-template-rows: 50px 1fr 1fr; gap: 2px; max-width: 500px; margin: 0 auto;">'
                '<!-- Header row -->'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem; border-radius: 6px 0 0 0;"></div>'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem;">Actual High Vol</div>'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem; border-radius: 0 6px 0 0;">Actual Normal</div>'
                '<!-- Predicted High Vol row -->'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem;">Predicted High Vol</div>'
                '<div style="background: linear-gradient(135deg, rgba(20, 184, 166, 0.25), rgba(20, 184, 166, 0.15)); border: 2px solid rgba(20, 184, 166, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{tp}<br><span style="font-size: 0.7rem; font-weight: 600; color: #14b8a6;">True high-vol alerts</span>'
                '</div>'
                '<div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(239, 68, 68, 0.15)); border: 2px solid rgba(239, 68, 68, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{fp}<br><span style="font-size: 0.7rem; font-weight: 600; color: #ef4444;">False high-vol alerts</span>'
                '</div>'
                '<!-- Predicted Normal row -->'
                '<div style="background-color: rgba(51, 65, 85, 0.8); color: #f8fafc; padding: 12px; text-align: center; font-weight: 700; font-size: 0.85rem;">Predicted Normal</div>'
                '<div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.25), rgba(245, 158, 11, 0.15)); border: 2px solid rgba(245, 158, 11, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{fn}<br><span style="font-size: 0.7rem; font-weight: 600; color: #f59e0b;">Missed high-vol periods</span>'
                '</div>'
                '<div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(59, 130, 246, 0.15)); border: 2px solid rgba(59, 130, 246, 0.5); color: #f8fafc; padding: 15px; text-align: center; font-weight: 800; font-size: 1.4rem; border-radius: 8px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);">'
                f'{tn}<br><span style="font-size: 0.7rem; font-weight: 600; color: #3b82f6;">Correct normal signals</span>'
                '</div>'
                '</div>'
                '</div>'
                '<div class="matrix-right">'
                '<div class="validation-read-card">'
                '<div class="validation-read-title">Validation read</div>'
                f'<div class="validation-read-text">{interp_p1} {interp_p2}</div>'
                '<div style="margin-top: 1rem; display: flex; gap: 10px; flex-wrap: wrap;">'
                f'<div style="background: rgba(20, 184, 166, 0.15); border: 1px solid rgba(20, 184, 166, 0.3); padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; color: #14b8a6; font-weight: 600;">Captured: {tp}</div>'
                f'<div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; color: #f59e0b; font-weight: 600;">Missed: {fn}</div>'
                '</div>'
                '</div>'
                '</div>'
                '</div>'
                '</div>'
            )
            st.markdown(matrix_html, unsafe_allow_html=True)

            with st.expander("Technical mapping"):
                st.markdown(
                    "- True high-vol alerts = True Positives\\n"
                    "- False high-vol alerts = False Positives\\n"
                    "- Missed high-vol periods = False Negatives\\n"
                    "- Correct normal signals = True Negatives"
                )
        st.markdown(
            """
            <div class='method-box'>
                Forecast vs realized validation compares the model signal with realized USD/TND volatility to assess whether regime classification remains consistent through time.
            </div>
            """,
            unsafe_allow_html=True,
        )
        plot_forecast_vs_actual(payload)
        # Regime Classification Conclusion
        forecast_vol = classification.get("forecast_volatility", 0)
        regime = classification.get("regime_label", "Unavailable")
        q50 = safe_float(classification.get("q50"))
        q75 = safe_float(classification.get("q75"))
        high_vol_prob = safe_float(classification.get("high_vol_probability", 0))
        relation = ""
        if forecast_vol < q50:
            relation = f"below the <strong>Median volatility line of {format_pct(q50)}</strong> and below the <strong>High-volatility threshold of {format_pct(q75)}</strong>"
        elif forecast_vol < q75:
            relation = f"above the <strong>Median volatility line of {format_pct(q50)}</strong> and below the <strong>High-volatility threshold of {format_pct(q75)}</strong>"
        else:
            relation = f"above the <strong>Median volatility line of {format_pct(q50)}</strong> and above the <strong>High-volatility threshold of {format_pct(q75)}</strong>"
        
        regime_display = regime if "regime" in regime.lower() else f"{regime} regime"
        conclusion_text = f"The current 3-day USD/TND volatility forecast is <strong>{format_pct(forecast_vol)}</strong>, placing the pair in an <strong>{regime_display}</strong>. The forecast is {relation}."
        conclusion_text += f" Classification performance remains acceptable, with an <strong>overall hit rate of {accuracy:.1%}</strong> and a <strong>high-volatility capture rate of {recall:.1%}</strong>. This supports using the regime signal as a validation layer for short-term USD/TND risk monitoring."
        if regime in ["Elevated-volatility regime", "High-volatility regime", "Stress-volatility regime"] and high_vol_prob < 0.1:
            conclusion_text += f" Although the current regime is {regime.lower()}, the high-volatility probability remains contained at <strong>{format_pct(high_vol_prob)}</strong>, indicating a monitoring stance rather than a full stress alert."
        st.markdown(
            f"""
            <div style="background-color: rgba(15, 23, 42, 0.92); border: 1px solid rgba(96, 165, 250, 0.35); border-radius: 8px; padding: 15px; margin: 20px 0;">
                <div style="font-size: 16px; color: #f8fafc; font-weight: bold; margin-bottom: 10px;">Regime Classification Conclusion</div>
                <div style="font-size: 14px; color: #cbd5e1; line-height: 1.5;">
                    {conclusion_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_high_volatility_calibration_diagnostics(payload)


# ============================================================
# Macro-Event Audit Helpers
# ============================================================


def get_current_event_regime_from_payload(payload: Optional[dict]) -> Optional[str]:
    """Extract the current event regime code from the forecast payload."""
    if not payload:
        return None
    forecast = payload.get("forecast", {}) or {}
    for key in ("event_regime", "macro_event_regime", "current_macro_regime"):
        value = forecast.get(key)
        if value and isinstance(value, str):
            return value.strip().lower()
    return None


def get_current_event_metrics(payload: Optional[dict]) -> Optional[dict]:
    """
    Fetch the descriptive metrics for the current macro-event regime.
    
    Returns a dict with all event metrics or None if not found.
    """
    if not payload:
        return None
    
    # Get current event regime
    current_regime = get_current_event_regime_from_payload(payload)
    if not current_regime:
        return None
    
    # Try to load the event descriptive stats Excel file
    try:
        event_stats_path = artifact_path(payload, "event_descriptive_stats", prefer_freshest=True)
        if not event_stats_path:
            # Fallback to the standard filename if not in artifacts
            if Path("event_regime_descriptive_stats_3d.xlsx").exists():
                event_stats_path = Path("event_regime_descriptive_stats_3d.xlsx")
            else:
                return None
        
        df = pd.read_excel(str(event_stats_path))
        
        # Match by event_regime column (exact match with current regime code)
        match = df[df["event_regime"].astype(str).str.lower().eq(current_regime)]
        
        if not match.empty:
            return match.iloc[0].to_dict()
    except Exception:
        pass
    
    return None


def is_event_ongoing(event_metrics: dict) -> bool:
    """
    Determine if the current event is ongoing (not yet completed).
    
    An event is considered ongoing if:
    - It has no end_date, or
    - The reliability_label is "Early / low sample"
    """
    if not event_metrics:
        return False
    
    # Check if end_date is NaT or NaN
    end_date = event_metrics.get("end_date")
    if end_date is None or pd.isna(end_date):
        return True
    
    # Check reliability label
    reliability = str(event_metrics.get("reliability_label", "")).strip()
    if "Early" in reliability or "low sample" in reliability:
        return True
    
    return False


def render_macro_event_regime_analysis(payload: Optional[dict]) -> None:
    page_header("Macro-Event Regime Analysis", "Event methodology", payload)
    st.markdown(
        f"<div style='color:{FINANCE_COLORS['muted']};font-size:1rem;margin:-0.55rem 0 1rem;'>Descriptive USD/TND volatility behavior across documented macro-financial regimes.</div>",
        unsafe_allow_html=True,
    )
    event_df = get_event_descriptive_df(payload)
    if event_df.empty:
        if payload is None:
            render_no_payload_choice()
        st.info("Macro-event descriptive statistics are not available. Run the latest forecasting pipeline to generate event artifacts.")
        return

    event_source = event_df.attrs.get("source", "unknown")
    event_artifact_path = artifact_path(payload, "event_descriptive_stats", prefer_freshest=True)

    event_df = event_df.copy()
    if "event_regime" in event_df.columns:
        event_df["event_label"] = event_df["event_regime"].map(pretty_event_name)
    else:
        event_df["event_label"] = event_df.index.astype(str)

    # Use parsed dates for chronological sorting in both the chart and table.
    if "start_date" in event_df.columns:
        event_df["_sort_start_date"] = pd.to_datetime(event_df["start_date"], errors="coerce")
        event_df = event_df.sort_values(["_sort_start_date", "event_regime_code"] if "event_regime_code" in event_df.columns else ["_sort_start_date"]).reset_index(drop=True)
    elif "event_regime_code" in event_df.columns:
        event_df = event_df.sort_values("event_regime_code").reset_index(drop=True)

    if "event_regime" in event_df.columns:
        # Keep analytical datetime columns clean for sorting and filtering.
        # Display columns will be created later for formatting.
        completed_event_df = event_df[event_df["event_regime"].astype(str).ne(ONGOING_EVENT_REGIME)].copy()
        ongoing_event_df = event_df[event_df["event_regime"].astype(str).eq(ONGOING_EVENT_REGIME)].copy()
    else:
        completed_event_df = event_df.copy()
        ongoing_event_df = pd.DataFrame(columns=event_df.columns)

    robust_cols_detected = [c for c in EVENT_DESCRIPTIVE_ROBUST_COLUMNS if c in event_df.columns]
    metadata = get_run_metadata(payload)

    def _fmt_ratio_pct(value: Any, digits: int = 2) -> str:
        x = safe_float(value)
        return f"{x * 100:.{digits}f}%" if x is not None else "n/a"

    def _fmt_point_pct(value: Any) -> str:
        x = safe_float(value)
        return f"{x:.2f}%" if x is not None else "n/a"

    def _fmt_share_pct(value: Any) -> str:
        x = safe_float(value)
        return f"{x * 100:.2f}%" if x is not None else "n/a"

    st.markdown(
        """
        <div class='method-box'><strong>Methodology note.</strong> Descriptive macro-event volatility statistics are computed from a stable USD/TND event-analysis dataset, not from the final ML feature matrix. This prevents changes in optional engineered model features from changing historical event-volatility comparisons. Forecast performance by event regime is still computed from the walk-forward model outputs.</div>
        """,
        unsafe_allow_html=True,
    )

    audit_values = [
        (
            "Source",
            metadata.get("event_descriptive_stats_source", event_df.get("source_dataset", pd.Series(["n/a"])).iloc[0] if "source_dataset" in event_df.columns and not event_df.empty else "n/a"),
            "Event-analysis dataset",
        ),
        ("Rows", metadata.get("event_analysis_row_count", "n/a"), "Stable event rows"),
        (
            "Start",
            pd.to_datetime(metadata.get("event_analysis_start_date"), errors="coerce").date().isoformat() if metadata.get("event_analysis_start_date") else "n/a",
            "Event-analysis start",
        ),
        (
            "End",
            pd.to_datetime(metadata.get("event_analysis_end_date"), errors="coerce").date().isoformat() if metadata.get("event_analysis_end_date") else "n/a",
            "Event-analysis end",
        ),
    ]
    audit_html = "".join(
        (
            "<div style='background:rgba(16,26,44,0.86);border:1px solid rgba(168,179,204,0.16);"
            "border-radius:8px;padding:0.82rem 0.9rem;min-height:88px;'>"
            f"<div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;color:{FINANCE_COLORS['muted']};font-weight:700;'>{html.escape(label)}</div>"
            f"<div style='font-size:1.08rem;color:{FINANCE_COLORS['text']};font-weight:800;margin-top:0.28rem;'>{html.escape(str(value))}</div>"
            f"<div style='font-size:0.78rem;color:{FINANCE_COLORS['muted']};margin-top:0.3rem;'>{html.escape(note)}</div>"
            "</div>"
        )
        for label, value, note in audit_values
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.8rem;margin:0.85rem 0 1rem;'>{audit_html}</div>",
        unsafe_allow_html=True,
    )

    # Current Macro-Event Audit: Dynamically display the latest event regime metrics
    current_event_metrics = get_current_event_metrics(payload)
    if current_event_metrics:
        current_regime = get_current_event_regime_from_payload(payload)
        event_label = format_event_context_label(current_regime) or pretty_event_name(current_regime)
        
        # Determine if event is ongoing
        is_ongoing = is_event_ongoing(current_event_metrics)
        
        # Build metric cards for the current event
        current_event_data = [
            ("Mean annualized volatility", _fmt_ratio_pct(current_event_metrics.get("mean_target_vol_annualized"))),
            ("Change vs Pre-COVID", _fmt_point_pct(current_event_metrics.get("annualized_target_vol_vs_pre_covid_pct"))),
            ("q90 annualized volatility", _fmt_ratio_pct(current_event_metrics.get("q90_target_vol_annualized"))),
            ("Max annualized volatility", _fmt_ratio_pct(current_event_metrics.get("max_target_vol_annualized"))),
            ("Share high-volatility days", _fmt_share_pct(current_event_metrics.get("share_high_vol_days"))),
            ("Observations", f"n={int(safe_float(current_event_metrics.get('observations')) or 0)}"),
            ("Reliability", str(current_event_metrics.get("reliability_label", "n/a"))),
        ]
        
        event_metric_html = "".join(
            (
                "<div style='background:rgba(18,31,52,0.82);border:1px solid rgba(168,179,204,0.14);"
                "border-radius:8px;padding:0.72rem 0.75rem;'>"
                f"<div style='font-size:0.72rem;color:{FINANCE_COLORS['muted']};font-weight:700;text-transform:uppercase;letter-spacing:0.05em;'>{html.escape(label)}</div>"
                f"<div style='font-size:1.18rem;color:{FINANCE_COLORS['text']};font-weight:850;margin-top:0.26rem;'>{html.escape(value)}</div>"
                "</div>"
            )
            for label, value in current_event_data
        )
        
        # Select wording based on whether event is ongoing
        if is_ongoing:
            event_wording = (
                "The current macro-event window is ongoing, so these statistics are preliminary and reflect observed USD/TND volatility so far. "
                "Average volatility may understate short-lived stress episodes; q90 volatility, maximum volatility, and high-volatility-day share "
                "are shown to capture peak-stress intensity."
            )
        else:
            event_wording = (
                "These statistics summarize the completed event window. Peak-stress metrics complement average volatility because "
                "short-lived shocks can be diluted in broad event windows."
            )
        
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,rgba(16,26,44,0.94),rgba(18,31,52,0.86));border:1px solid rgba(79,123,255,0.28);border-radius:10px;padding:1rem;margin:1rem 0 1.1rem;'>
                <div style='font-size:0.8rem;text-transform:uppercase;letter-spacing:0.09em;color:{FINANCE_COLORS["primary"]};font-weight:800;'>Current Macro-Event Audit</div>
                <div style='font-size:1.15rem;color:{FINANCE_COLORS["text"]};font-weight:750;margin-top:0.5rem;'>{html.escape(event_label)}</div>
                <div style='display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:0.72rem;margin-top:0.75rem;'>{event_metric_html}</div>
                <div style='font-size:0.9rem;color:{FINANCE_COLORS["muted"]};line-height:1.45;margin-top:0.8rem;'>
                    {html.escape(event_wording)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    chart_df = completed_event_df.copy()
    if "event_regime_code" in chart_df.columns:
        chart_df = chart_df.sort_values("event_regime_code")
    elif "_sort_start_date" in chart_df.columns:
        chart_df = chart_df.sort_values("_sort_start_date")
    chart_df = chart_df.reset_index(drop=True)
    regime_labels = chart_df["event_label"].astype(str).tolist()
    regime_order = regime_labels
    muted_crisis = "#7E6AE6"
    muted_normal = "#6F8098"
    muted_positive = "#3FA7A0"
    muted_negative = "#A96A72"
    baseline_color = "#C1C9D8"
    bar_colors = [
        muted_normal if str(regime) == "pre_covid_normal"
        else muted_crisis if int(safe_float(crisis, 0) or 0) == 1
        else "#4F6F9F"
        for regime, crisis in zip(
            chart_df.get("event_regime", pd.Series("", index=chart_df.index)),
            chart_df.get("crisis_flag", pd.Series(0, index=chart_df.index)),
        )
    ]

    def _baseline_from_pct(value_col: str, pct_col: str) -> Optional[float]:
        if value_col not in chart_df.columns or pct_col not in chart_df.columns:
            return None
        values = pd.to_numeric(chart_df[value_col], errors="coerce")
        pct = pd.to_numeric(chart_df[pct_col], errors="coerce")
        valid = values.notna() & pct.notna() & (pct != -100)
        if not valid.any():
            return None
        return float((values[valid].iloc[0] * 100.0) / (1.0 + pct[valid].iloc[0] / 100.0))

    st.markdown(
        f"<div style='color:{FINANCE_COLORS['muted']};font-size:0.86rem;margin:0.35rem 0 0.55rem;'>Views: Average Volatility · Change vs Baseline · Peak-Stress Metrics</div>",
        unsafe_allow_html=True,
    )
    avg_tab, change_tab, peak_tab = st.tabs(["Average Volatility", "Change vs Baseline", "Peak-Stress Metrics"])

    with avg_tab:
        avg_pct = pd.to_numeric(chart_df.get("mean_target_vol_annualized", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        avg_fig = go.Figure()
        avg_fig.add_trace(go.Bar(
            x=avg_pct,
            y=regime_labels,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(234,240,250,0.12)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in avg_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Mean annualized volatility: %{x:.2f}%<extra></extra>",
            name="Mean annualized volatility",
        ))
        broad_baseline = _baseline_from_pct(
            "mean_target_vol_annualized",
            "mean_target_vol_annualized_vs_broad_normal_pct",
        )
        if broad_baseline is not None and np.isfinite(broad_baseline):
            avg_fig.add_vline(
                x=broad_baseline,
                line_width=1.2,
                line_dash="dot",
                line_color=baseline_color,
                annotation_text="Broad normal baseline",
                annotation_position="top right",
                annotation_font=dict(color=baseline_color, size=11),
            )
        avg_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        avg_fig.update_xaxes(range=[0, max(float(np.nanmax(avg_pct)) * 1.22 if avg_pct.notna().any() else 10, 10)])
        apply_institutional_plotly_layout(
            avg_fig,
            title="Average Volatility by Event Regime",
            subtitle="Mean annualized 3-day USD/TND volatility; descriptive statistics sourced from event_analysis_df.",
            height=470,
            xaxis_title="Annualized volatility",
        )
        st.plotly_chart(avg_fig, use_container_width=True, config={"displayModeBar": False})

    with change_tab:
        change_col = next(
            (c for c in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_vs_pre_covid_pct"] if c in chart_df.columns),
            None,
        )
        change_values = pd.to_numeric(chart_df.get(change_col, pd.Series(np.nan, index=chart_df.index)), errors="coerce") if change_col else pd.Series(np.nan, index=chart_df.index)
        change_colors = [muted_positive if pd.notna(v) and v >= 0 else muted_negative for v in change_values]
        change_fig = go.Figure()
        error_kwargs: Dict[str, Any] = {}
        if {
            "annualized_target_vol_ci_low_vs_pre_covid_pct",
            "annualized_target_vol_ci_high_vs_pre_covid_pct",
        }.issubset(chart_df.columns):
            lower_ci = pd.to_numeric(chart_df["annualized_target_vol_ci_low_vs_pre_covid_pct"], errors="coerce")
            upper_ci = pd.to_numeric(chart_df["annualized_target_vol_ci_high_vs_pre_covid_pct"], errors="coerce")
            error_kwargs["error_x"] = dict(
                type="data",
                array=np.nan_to_num((upper_ci - change_values).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0),
                arrayminus=np.nan_to_num((change_values - lower_ci).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0),
                visible=True,
                thickness=1,
                width=3,
                color="rgba(193,201,216,0.55)",
            )
        change_fig.add_trace(go.Bar(
            x=change_values,
            y=regime_labels,
            orientation="h",
            marker=dict(color=change_colors, line=dict(color="rgba(234,240,250,0.12)", width=1)),
            text=[f"{v:+.2f}%" if pd.notna(v) else "" for v in change_values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Change vs Pre-COVID: %{x:+.2f}%<extra></extra>",
            name="Change vs Pre-COVID",
            **error_kwargs,
        ))
        change_fig.add_vline(x=0, line_width=1.2, line_color="rgba(234,240,250,0.45)")
        change_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        max_abs = float(np.nanmax(np.abs(change_values))) if change_values.notna().any() else 10.0
        change_fig.update_xaxes(range=[-max_abs * 1.35, max_abs * 1.35])
        apply_institutional_plotly_layout(
            change_fig,
            title="Change vs Pre-COVID Baseline",
            subtitle="Horizontal diverging view of average annualized 3-day USD/TND volatility change.",
            height=470,
            xaxis_title="Change vs Pre-COVID baseline",
        )
        st.plotly_chart(change_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<div class='method-box'><strong>Chart note.</strong> Bars show average annualized 3-day USD/TND volatility change versus the Pre-COVID baseline. Confidence intervals are bootstrap-based and descriptive.</div>",
            unsafe_allow_html=True,
        )

    with peak_tab:
        q90_pct = pd.to_numeric(chart_df.get("q90_target_vol_annualized", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        max_pct = pd.to_numeric(chart_df.get("max_target_vol_annualized", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        peak_fig = go.Figure()
        peak_fig.add_trace(go.Bar(
            x=q90_pct,
            y=regime_labels,
            orientation="h",
            name="q90 volatility",
            marker=dict(color="#D2A84A", line=dict(color="rgba(234,240,250,0.10)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in q90_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>q90 volatility: %{x:.2f}%<extra></extra>",
        ))
        peak_fig.add_trace(go.Bar(
            x=max_pct,
            y=regime_labels,
            orientation="h",
            name="Max volatility",
            marker=dict(color="#9D5E68", line=dict(color="rgba(234,240,250,0.10)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in max_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Maximum volatility: %{x:.2f}%<extra></extra>",
        ))
        peak_fig.update_layout(barmode="group")
        peak_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        peak_fig.update_xaxes(range=[0, max(float(np.nanmax(max_pct)) * 1.22 if max_pct.notna().any() else 10, 10)])
        apply_institutional_plotly_layout(
            peak_fig,
            title="Peak-Stress Volatility by Event Regime",
            subtitle="q90 and maximum annualized 3-day USD/TND volatility.",
            height=500,
            xaxis_title="Annualized volatility",
            hovermode="closest",
        )
        st.plotly_chart(peak_fig, use_container_width=True, config={"displayModeBar": False})

        share_pct = pd.to_numeric(chart_df.get("share_high_vol_days", pd.Series(np.nan, index=chart_df.index)), errors="coerce") * 100
        share_fig = go.Figure()
        share_fig.add_trace(go.Bar(
            x=share_pct,
            y=regime_labels,
            orientation="h",
            marker=dict(color="#5F87B9", line=dict(color="rgba(234,240,250,0.10)", width=1)),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in share_pct],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Regime: %{y}<br>Share of high-volatility days: %{x:.2f}%<extra></extra>",
            name="Share of high-volatility days",
        ))
        share_fig.update_yaxes(categoryorder="array", categoryarray=regime_order, autorange="reversed")
        share_fig.update_xaxes(range=[0, max(float(np.nanmax(share_pct)) * 1.25 if share_pct.notna().any() else 10, 10)])
        apply_institutional_plotly_layout(
            share_fig,
            title="Share of High-Volatility Days",
            subtitle="Uses the fixed high-volatility threshold produced by the backend.",
            height=430,
            xaxis_title="Share of days",
            hovermode="closest",
        )
        st.plotly_chart(share_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            "<div class='method-box'><strong>Peak-stress note.</strong> Peak-stress metrics complement average volatility because short crisis spikes can be diluted in broad event windows.</div>",
            unsafe_allow_html=True,
        )

    detail_cols = [
        "event_regime",
        "observations",
        "reliability_label",
        "mean_target_vol_annualized",
        "annualized_target_vol_vs_pre_covid_pct",
        "mean_target_vol_annualized_vs_broad_normal_pct",
        "q90_target_vol_annualized",
        "max_target_vol_annualized",
        "share_high_vol_days",
        "variance_test_result",
        "source_dataset",
    ]
    detail_df = event_df[[c for c in detail_cols if c in event_df.columns]].copy()
    for col in ["mean_target_vol_annualized", "q90_target_vol_annualized", "max_target_vol_annualized", "share_high_vol_days"]:
        if col in detail_df.columns:
            detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "n/a")
    for col in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_annualized_vs_broad_normal_pct"]:
        if col in detail_df.columns:
            detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "n/a")
    with st.expander("Detailed event-regime statistics", expanded=False):
        st.dataframe(
            detail_df.rename(columns={
                "event_regime": "Event regime",
                "observations": "Observations",
                "reliability_label": "Reliability",
                "mean_target_vol_annualized": "Mean annualized volatility",
                "annualized_target_vol_vs_pre_covid_pct": "Change vs Pre-COVID",
                "mean_target_vol_annualized_vs_broad_normal_pct": "Change vs broad normal",
                "q90_target_vol_annualized": "q90 annualized volatility",
                "max_target_vol_annualized": "Max annualized volatility",
                "share_high_vol_days": "Share high-volatility days",
                "variance_test_result": "Variance test",
                "source_dataset": "Source dataset",
            }),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Artifact source: {event_source}; path: {event_artifact_path or 'Unavailable'}")

    return

    impact_col = next(
        (c for c in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_vs_pre_covid_pct"] if c in event_df.columns),
        None,
    )
    if impact_col:
        top_df = completed_event_df.copy()
        top_df["_impact_value"] = pd.to_numeric(top_df[impact_col], errors="coerce")
        top_df = top_df[
            top_df["_impact_value"].gt(0)
            & ~top_df.get("event_regime", pd.Series("", index=top_df.index)).astype(str).eq("pre_covid_normal")
        ].copy()
        crisis_candidates = top_df
        if "crisis_flag" in crisis_candidates.columns:
            crisis_candidates = crisis_candidates[pd.to_numeric(crisis_candidates["crisis_flag"], errors="coerce").fillna(0).eq(1)]
        selected = crisis_candidates.sort_values("_impact_value", ascending=False).head(3)
        if len(selected) < 3:
            fill = top_df.drop(index=selected.index, errors="ignore").sort_values("_impact_value", ascending=False).head(3 - len(selected))
            selected = pd.concat([selected, fill], axis=0)
        cards = []
        for _, row in selected.head(3).iterrows():
            impact = safe_float(row.get("_impact_value"), 0) or 0
            observations = row.get("observations", None)
            obs_note = f"n={int(observations):,}" if safe_float(observations) is not None else "n=n/a"
            reliability = str(row.get("reliability_label", "") or "").strip()
            variance_result = str(row.get("variance_test_result", "") or "").strip()
            
            # Build footer with observation count, reliability, and variance test status
            footer_parts = [obs_note]
            if reliability:
                footer_parts.append(reliability)
            if variance_result:
                footer_parts.append(variance_result)
            technical_note = " · ".join(footer_parts)
            
            # Create custom card with two-line footer for top shock cards
            regime_name = str(row.get("event_label", "Event"))
            value_text = f"+{impact:.2f}%" if impact > 0 else f"{impact:.2f}%"
            
            card_html = (
                '<div class="kpi-card" style="border-top: 3px solid #FF5F68;">'
                f'<div class="kpi-label">{html.escape(regime_name)}</div>'
                f'<div class="kpi-value">{html.escape(value_text)}</div>'
                '<div class="kpi-note" style="display: flex; flex-direction: column; gap: 0.25rem;">'
                '<div style="font-size: 0.85rem; font-weight: 600; color: #A8B3CC;">Volatility uplift vs Pre-COVID baseline</div>'
                f'<div style="font-size: 0.8rem; color: #7F8EA7;">{html.escape(technical_note)}</div>'
                '</div>'
                '</div>'
            )
            cards.append(card_html)
        if cards:
            st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    avg_tab, peak_tab = st.tabs(["Average Volatility by Event Regime", "Peak-Stress Metrics by Event Regime"])
    with avg_tab:
        avg_cols = [
            "event_label",
            "mean_target_vol_annualized",
            "annualized_target_vol_vs_pre_covid_pct",
            "mean_target_vol_annualized_vs_broad_normal_pct",
            "observations",
            "reliability_label",
            "sample_size_note",
            "source_dataset",
        ]
        avg_available = [c for c in avg_cols if c in completed_event_df.columns]
        if "mean_target_vol_annualized" in completed_event_df.columns:
            avg_fig = go.Figure()
            avg_fig.add_trace(go.Bar(
                x=completed_event_df["event_label"],
                y=pd.to_numeric(completed_event_df["mean_target_vol_annualized"], errors="coerce") * 100,
                marker_color=[EVENT_COLORS.get(str(r), FINANCE_COLORS["teal"]) for r in completed_event_df.get("event_regime", pd.Series([None] * len(completed_event_df)))],
                name="Mean annualized 3-day volatility",
                customdata=np.column_stack([
                    pd.to_numeric(completed_event_df.get("annualized_target_vol_vs_pre_covid_pct", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce"),
                    pd.to_numeric(completed_event_df.get("mean_target_vol_annualized_vs_broad_normal_pct", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce"),
                    pd.to_numeric(completed_event_df.get("observations", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce").fillna(0).astype(int),
                    completed_event_df.get("reliability_label", pd.Series("Unavailable", index=completed_event_df.index)).fillna("Unavailable").astype(str),
                ]),
                hovertemplate=(
                    "Regime: %{x}<br>"
                    "Mean annualized volatility: %{y:.2f}%<br>"
                    "Change vs Pre-COVID: %{customdata[0]:.2f}%<br>"
                    "Change vs broad normal: %{customdata[1]:.2f}%<br>"
                    "Observations: %{customdata[2]}<br>"
                    "Reliability: %{customdata[3]}<extra></extra>"
                ),
            ))
            avg_fig.update_layout(
                title="Average Annualized 3-Day USD/TND Volatility by Completed Event Regime",
                yaxis_title="Mean annualized volatility (%)",
                xaxis_title="",
            )
            st.plotly_chart(apply_finance_layout(avg_fig, height=420), use_container_width=True)
        if avg_available:
            avg_table = completed_event_df[avg_available].copy()
            for col in ["mean_target_vol_annualized"]:
                if col in avg_table.columns:
                    avg_table[col] = pd.to_numeric(avg_table[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
            for col in ["annualized_target_vol_vs_pre_covid_pct", "mean_target_vol_annualized_vs_broad_normal_pct"]:
                if col in avg_table.columns:
                    avg_table[col] = pd.to_numeric(avg_table[col], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
            if "observations" in avg_table.columns:
                avg_table["observations"] = pd.to_numeric(avg_table["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
            st.dataframe(avg_table.rename(columns={
                "event_label": "Macro-event regime",
                "mean_target_vol_annualized": "Mean annualized 3-day volatility",
                "annualized_target_vol_vs_pre_covid_pct": "Change vs Pre-COVID baseline",
                "mean_target_vol_annualized_vs_broad_normal_pct": "Change vs broad normal baseline",
                "observations": "Observations",
                "reliability_label": "Reliability",
                "sample_size_note": "Sample-size note",
                "source_dataset": "Source dataset",
            }), hide_index=True, use_container_width=True)

    with peak_tab:
        peak_cols = [
            "event_label",
            "q90_target_vol_annualized",
            "max_target_vol_annualized",
            "share_high_vol_days",
            "q90_target_vol_annualized_vs_broad_normal_pct",
            "observations",
            "reliability_label",
            "sample_size_note",
            "source_dataset",
        ]
        peak_available = [c for c in peak_cols if c in completed_event_df.columns]
        if {"q90_target_vol_annualized", "max_target_vol_annualized"}.issubset(completed_event_df.columns):
            peak_fig = go.Figure()
            peak_fig.add_trace(go.Bar(
                x=completed_event_df["event_label"],
                y=pd.to_numeric(completed_event_df["q90_target_vol_annualized"], errors="coerce") * 100,
                name="q90 annualized volatility",
                marker_color=FINANCE_COLORS["gold"],
            ))
            peak_fig.add_trace(go.Bar(
                x=completed_event_df["event_label"],
                y=pd.to_numeric(completed_event_df["max_target_vol_annualized"], errors="coerce") * 100,
                name="maximum annualized volatility",
                marker_color=FINANCE_COLORS["red"],
            ))
            peak_fig.update_layout(
                title="Peak-Stress USD/TND Volatility by Completed Event Regime",
                yaxis_title="Annualized volatility (%)",
                xaxis_title="",
                barmode="group",
            )
            st.plotly_chart(apply_finance_layout(peak_fig, height=420), use_container_width=True)
        if peak_available:
            peak_table = completed_event_df[peak_available].copy()
            for col in ["q90_target_vol_annualized", "max_target_vol_annualized"]:
                if col in peak_table.columns:
                    peak_table[col] = pd.to_numeric(peak_table[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
            if "share_high_vol_days" in peak_table.columns:
                peak_table["share_high_vol_days"] = pd.to_numeric(peak_table["share_high_vol_days"], errors="coerce").apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")
            if "q90_target_vol_annualized_vs_broad_normal_pct" in peak_table.columns:
                peak_table["q90_target_vol_annualized_vs_broad_normal_pct"] = pd.to_numeric(peak_table["q90_target_vol_annualized_vs_broad_normal_pct"], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
            if "observations" in peak_table.columns:
                peak_table["observations"] = pd.to_numeric(peak_table["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
            st.dataframe(peak_table.rename(columns={
                "event_label": "Macro-event regime",
                "q90_target_vol_annualized": "q90 annualized 3-day volatility",
                "max_target_vol_annualized": "Maximum annualized 3-day volatility",
                "share_high_vol_days": "Share of high-volatility days",
                "q90_target_vol_annualized_vs_broad_normal_pct": "q90 change vs broad normal baseline",
                "observations": "Observations",
                "reliability_label": "Reliability",
                "sample_size_note": "Sample-size note",
                "source_dataset": "Source dataset",
            }), hide_index=True, use_container_width=True)
        st.markdown(
            "<div class='method-box'><strong>Peak-stress reading.</strong> Average regime volatility, peak-stress volatility, high-volatility day share, and sample-size reliability answer different questions. Short crisis spikes can be visible in q90, maximum volatility, and high-volatility-day share even when the full-window mean is diluted by later stabilization.</div>",
            unsafe_allow_html=True,
        )

    fig = go.Figure()
    value_col = next(
        (
            c
            for c in [
                "annualized_target_vol_vs_pre_covid_pct",
                "mean_target_vol_vs_pre_covid_pct",
                "mean_target_vol_vs_full_sample_pct",
            ]
            if c in event_df.columns
        ),
        None,
    )
    colors = [EVENT_COLORS.get(str(r), FINANCE_COLORS["teal"]) for r in completed_event_df.get("event_regime", pd.Series([None] * len(completed_event_df)))]
    y_values = pd.to_numeric(completed_event_df[value_col], errors="coerce") if value_col else pd.Series(np.nan, index=completed_event_df.index)
    ann_vol = pd.to_numeric(completed_event_df.get("mean_target_vol_annualized", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    ci_low = pd.to_numeric(completed_event_df.get("annualized_target_vol_ci_low", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    ci_high = pd.to_numeric(completed_event_df.get("annualized_target_vol_ci_high", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    pvalues = pd.to_numeric(completed_event_df.get("brown_forsythe_pvalue_vs_pre_covid", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce")
    customdata = np.column_stack([
        pd.to_numeric(completed_event_df.get("observations", pd.Series(np.nan, index=completed_event_df.index)), errors="coerce").fillna(0).astype(int).astype(str),
        completed_event_df.get("reliability_label", pd.Series("Unavailable", index=completed_event_df.index)).fillna("Unavailable").astype(str),
        ann_vol.apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "n/a"),
        [
            f"{lo * 100:.2f}% - {hi * 100:.2f}%" if pd.notna(lo) and pd.notna(hi) else "n/a"
            for lo, hi in zip(ci_low, ci_high)
        ],
        pvalues.apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a"),
        completed_event_df.get("variance_test_result", pd.Series("Unavailable", index=completed_event_df.index)).fillna("Unavailable").astype(str),
    ])
    bar_kwargs = {}
    if {
        "annualized_target_vol_ci_low_vs_pre_covid_pct",
        "annualized_target_vol_ci_high_vs_pre_covid_pct",
    }.issubset(completed_event_df.columns):
        lower_ci_pct = pd.to_numeric(completed_event_df["annualized_target_vol_ci_low_vs_pre_covid_pct"], errors="coerce")
        upper_ci_pct = pd.to_numeric(completed_event_df["annualized_target_vol_ci_high_vs_pre_covid_pct"], errors="coerce")
        err_plus = np.nan_to_num((upper_ci_pct - y_values).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0)
        err_minus = np.nan_to_num((y_values - lower_ci_pct).clip(lower=0).values, nan=0.0, posinf=0.0, neginf=0.0)
        # Suppress error bars for Pre-COVID baseline (first bar at 0%) to reduce visual confusion
        if len(completed_event_df) > 0:
            err_plus[0] = 0.0
            err_minus[0] = 0.0
        bar_kwargs["error_y"] = dict(array=err_plus, arrayminus=err_minus, visible=True, thickness=1.4, width=4)
    fig.add_trace(go.Bar(
        x=completed_event_df["event_label"],
        y=y_values,
        marker_color=colors,
        name="Volatility impact",
        customdata=customdata,
        hovertemplate=(
            "Regime: %{x}<br>"
            "Change vs Pre-COVID: %{y:.2f}%<br>"
            "Observations: n=%{customdata[0]}<br>"
            "Reliability: %{customdata[1]}<br>"
            "Annualized volatility: %{customdata[2]}<br>"
            "95% CI: %{customdata[3]}<br>"
            "Brown-Forsythe p-value: %{customdata[4]}<br>"
            "Variance test: %{customdata[5]}<extra></extra>"
        ),
        **bar_kwargs,
    ))
    fig.update_layout(
        title="Completed Regimes: USD/TND Volatility Change vs Pre-COVID",
        yaxis_title="Volatility change vs Pre-COVID baseline (%)",
        xaxis_title="",
        xaxis=dict(categoryorder="array", categoryarray=completed_event_df["event_label"].tolist()),
    )
    st.plotly_chart(apply_finance_layout(fig, height=430), use_container_width=True)
    st.markdown(
        "<div class='method-box'><strong>Chart explanation.</strong> Bars show the average annualized 3-day USD/TND volatility change versus the Pre-COVID baseline; vertical lines show bootstrap confidence intervals.</div>",
        unsafe_allow_html=True,
    )

    if not ongoing_event_df.empty:
        row = ongoing_event_df.iloc[0]

        def _fmt_ratio_pct(value: Any, digits: int = 2) -> str:
            x = safe_float(value)
            return f"{x * 100:.{digits}f}%" if x is not None else "n/a"

        def _fmt_point_pct(value: Any) -> str:
            x = safe_float(value)
            return f"{x:.2f}%" if x is not None else "n/a"

        def _fmt_pvalue(value: Any) -> str:
            x = safe_float(value)
            return f"{x:.4f}" if x is not None else "n/a"

        # Prepare metrics for the premium summary panel
        regime_name = str(row.get("event_label", "Iran Geopolitical Shock"))
        observations = safe_float(row.get("observations"))
        observations_text = f"n = {int(observations):,}" if observations is not None else "n = n/a"
        ci_low_text = _fmt_ratio_pct(row.get("annualized_target_vol_ci_low"))
        ci_high_text = _fmt_ratio_pct(row.get("annualized_target_vol_ci_high"))
        ci_text = f"{ci_low_text} – {ci_high_text}" if ci_low_text != "n/a" and ci_high_text != "n/a" else "n/a"
        reliability = str(row.get("reliability_label", "Unavailable") or "Unavailable").strip()
        mean_vol = _fmt_ratio_pct(row.get("mean_target_vol_annualized"))
        impact = _fmt_point_pct(row.get("annualized_target_vol_vs_pre_covid_pct", row.get("mean_target_vol_vs_pre_covid_pct")))
        variance_result = str(row.get("variance_test_result", "Unavailable") or "Unavailable").strip()
        pvalue = _fmt_pvalue(row.get("brown_forsythe_pvalue_vs_pre_covid"))

        # Premium summary panel with better visual hierarchy
        st.markdown(
            f"""
            <div class='decision-card' style='border-left: 4px solid {FINANCE_COLORS["gold"]};'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>
                    <div>
                        <div class='decision-title' style='margin:0;'>{html.escape(regime_name)}</div>
                        <div style='font-size:0.85rem; color:{FINANCE_COLORS["muted"]}; margin-top:0.25rem;'>Current Ongoing Macro-Risk Regime</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["gold"]}; color:{FINANCE_COLORS["page_bg"]}; padding:0.35rem 0.7rem; border-radius:3px; font-size:0.8rem; font-weight:600;'>Ongoing / Preliminary</div>
                </div>
                <div class='decision-text' style='margin: 0.75rem 0;'>
                    Iran Geopolitical Shock is active and ongoing. It is excluded from the completed-regime comparison because the event window is still open and the realized-volatility sample remains preliminary. Current volatility statistics represent evidence observed so far, not a final regime impact.
                </div>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["teal"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Official start</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>2026-02-28</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["teal"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>End date</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["teal"]}; margin-top:0.25rem;'>Ongoing</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["primary"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Observations</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(observations_text)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["primary"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Reliability</div>
                        <div style='font-size:0.95rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(reliability)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["gold"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Mean annualized 3-day volatility</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["gold"]}; margin-top:0.25rem;'>{html.escape(mean_vol)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["gold"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Change vs Pre-COVID</div>
                        <div style='font-size:1.1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(impact)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["green"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>95% CI</div>
                        <div style='font-size:0.95rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(ci_text)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["green"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>Variance test</div>
                        <div style='font-size:0.9rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(variance_result)}</div>
                    </div>
                    <div style='background-color:{FINANCE_COLORS["panel_bg_2"]}; padding:0.75rem; border-radius:3px; border-left:2px solid {FINANCE_COLORS["slate"]};'>
                        <div style='font-size:0.75rem; color:{FINANCE_COLORS["muted"]}; text-transform:uppercase; letter-spacing:0.5px;'>p-value</div>
                        <div style='font-size:1rem; font-weight:600; color:{FINANCE_COLORS["text"]}; margin-top:0.25rem;'>{html.escape(pvalue)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Add Key reading box
        if not completed_event_df.empty:
            # Get highest and second-highest completed regimes
            completed_copy = completed_event_df.copy()
            completed_copy["_impact"] = pd.to_numeric(completed_copy.get(impact_col, pd.Series()), errors="coerce")
            completed_sorted = completed_copy[completed_copy["_impact"].notna()].sort_values("_impact", ascending=False)
            
            key_reading_html = "<strong>Key reading.</strong> "
            if len(completed_sorted) > 0:
                regime1 = str(completed_sorted.iloc[0].get("event_label", "Unknown regime"))
                impact1 = safe_float(completed_sorted.iloc[0].get("_impact"), 0) or 0
                key_reading_html += f"{html.escape(regime1)} shows the strongest USD/TND volatility uplift (<strong>+{impact1:.2f}%</strong>)"
                
                if len(completed_sorted) > 1:
                    regime2 = str(completed_sorted.iloc[1].get("event_label", "Unknown regime"))
                    impact2 = safe_float(completed_sorted.iloc[1].get("_impact"), 0) or 0
                    key_reading_html += f", followed by {html.escape(regime2)} (<strong>+{impact2:.2f}%</strong>)"
                key_reading_html += ". "
            
            # Check for significant variance regimes
            has_significant_variance = False
            if "variance_test_result" in completed_event_df.columns:
                significant = completed_event_df[
                    completed_event_df["variance_test_result"].astype(str).str.contains("significant", case=False, na=False)
                ]
                if len(significant) > 0:
                    has_significant_variance = True
                    key_reading_html += "These regimes show statistically significant variance differences versus Pre-COVID. "
            
            if not has_significant_variance and len(completed_sorted) > 0:
                key_reading_html += "The strongest completed-regime uplift shows variance differences versus Pre-COVID. "
            
            key_reading_html += f"{html.escape(regime_name)} is active but excluded from the completed-regime ranking because the event window remains open and the sample is preliminary."
            
            st.markdown(
                f"<div class='method-box'>{key_reading_html}</div>",
                unsafe_allow_html=True,
            )

    display_df = completed_event_df.drop(columns=["_sort_start_date"], errors="ignore").copy()
    if "start_date" in display_df.columns:
        display_df["start_date"] = pd.to_datetime(display_df["start_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "end_date" in display_df.columns:
        display_df["end_date"] = pd.to_datetime(display_df["end_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("Ongoing")
    if "event_regime" in display_df.columns:
        # Display the official methodology bounds while preserving empirical calculations from available observations.
        if "start_date" in display_df.columns:
            display_df.loc[display_df["event_regime"] == "post_tariff_normalization", "start_date"] = "2025-06-13"
            display_df.loc[display_df["event_regime"] == "iran_geopolitical_shock", "start_date"] = "2026-02-28"
        if "end_date" in display_df.columns:
            display_df.loc[display_df["event_regime"] == "post_tariff_normalization", "end_date"] = "2026-02-27"
            display_df.loc[display_df["event_regime"] == "iran_geopolitical_shock", "end_date"] = "Ongoing"

    # Main business-readable table columns
    main_cols = [
        "event_label",
        "observations",
        "reliability_label",
        "start_date",
        "end_date",
        "mean_target_vol_annualized",
        "q90_target_vol_annualized",
        "max_target_vol_annualized",
        "share_high_vol_days",
        "annualized_target_vol_vs_pre_covid_pct",
        "mean_target_vol_annualized_vs_broad_normal_pct",
        "q90_target_vol_annualized_vs_broad_normal_pct",
        "variance_test_result",
        "crisis_flag",
        "sample_size_note",
        "source_dataset",
    ]
    
    # Technical robustness columns for expander
    technical_cols = [
        "event_label",
        "median_target_vol_annualized",
        "annualized_return_vol",
        "annualized_target_vol_ci_low",
        "annualized_target_vol_ci_high",
        "brown_forsythe_pvalue_vs_pre_covid",
        "target_vol_iqr",
        "target_vol_mad_annualized",
    ]
    
    # Fallback columns if robust columns are missing
    fallback_cols = [
        "event_label",
        "observations",
        "start_date",
        "end_date",
        "mean_target_vol",
        "mean_target_vol_vs_pre_covid_pct",
        "mean_target_vol_vs_full_sample_pct",
        "crisis_flag",
    ]
    
    # Determine which columns exist
    main_display_cols = [c for c in main_cols if c in display_df.columns]
    technical_display_cols = [c for c in technical_cols if c in display_df.columns]
    
    # Use fallback if robust columns are missing
    if not any(c in display_df.columns for c in ["mean_target_vol_annualized", "annualized_target_vol_vs_pre_covid_pct"]):
        main_display_cols = [c for c in fallback_cols if c in display_df.columns]
        technical_display_cols = []
    
    # Format main table
    main_table_df = display_df[main_display_cols].copy()
    ratio_cols_main = [c for c in ["mean_target_vol_annualized", "q90_target_vol_annualized", "max_target_vol_annualized"] if c in main_table_df.columns]
    for col in ratio_cols_main:
        main_table_df[col] = pd.to_numeric(main_table_df[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
    
    if "share_high_vol_days" in main_table_df.columns:
        main_table_df["share_high_vol_days"] = pd.to_numeric(main_table_df["share_high_vol_days"], errors="coerce").apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "n/a")

    for col in [
        "annualized_target_vol_vs_pre_covid_pct",
        "mean_target_vol_vs_pre_covid_pct",
        "mean_target_vol_vs_full_sample_pct",
        "mean_target_vol_annualized_vs_broad_normal_pct",
        "q90_target_vol_annualized_vs_broad_normal_pct",
    ]:
        if col in main_table_df.columns:
            main_table_df[col] = pd.to_numeric(main_table_df[col], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    
    if "observations" in main_table_df.columns:
        main_table_df["observations"] = pd.to_numeric(main_table_df["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
    
    if "crisis_flag" in main_table_df.columns:
        main_table_df["crisis_flag"] = pd.to_numeric(main_table_df["crisis_flag"], errors="coerce").fillna(0).apply(lambda x: "Crisis" if int(x) == 1 else "Non-crisis")
    
    rename_map_main = {
        "event_label": "Macro-event regime",
        "observations": "Observations",
        "reliability_label": "Reliability",
        "start_date": "Start date",
        "end_date": "End date",
        "mean_target_vol_annualized": "Mean annualized 3-day volatility",
        "q90_target_vol_annualized": "q90 annualized 3-day volatility",
        "max_target_vol_annualized": "Maximum annualized 3-day volatility",
        "share_high_vol_days": "Share of high-volatility days",
        "annualized_target_vol_vs_pre_covid_pct": "Change vs Pre-COVID baseline",
        "mean_target_vol_annualized_vs_broad_normal_pct": "Mean change vs broad normal baseline",
        "q90_target_vol_annualized_vs_broad_normal_pct": "q90 change vs broad normal baseline",
        "variance_test_result": "Variance test result",
        "crisis_flag": "Crisis regime",
        "sample_size_note": "Sample-size note",
        "source_dataset": "Source dataset",
    }
    
    if not ongoing_event_df.empty:
        st.markdown(
            "<div class='method-box'><strong>Completed Regime Comparison.</strong> Iran Geopolitical Shock is excluded from this table because it is ongoing and shown separately above.</div>",
            unsafe_allow_html=True,
        )
    
    st.dataframe(main_table_df.rename(columns=rename_map_main), hide_index=True, use_container_width=True)
    
    # Technical robustness details expander
    if len(technical_display_cols) > 0:
        with st.expander("Technical robustness details", expanded=False):
            technical_table_df = display_df[technical_display_cols].copy()
            ratio_cols_tech = [
                "median_target_vol_annualized",
                "annualized_return_vol",
                "annualized_target_vol_ci_low",
                "annualized_target_vol_ci_high",
                "target_vol_iqr",
                "target_vol_mad_annualized",
            ]
            for col in ratio_cols_tech:
                if col in technical_table_df.columns:
                    technical_table_df[col] = pd.to_numeric(technical_table_df[col], errors="coerce").apply(lambda x: f"{x * 100:.4f}%" if pd.notna(x) else "n/a")
            
            if "brown_forsythe_pvalue_vs_pre_covid" in technical_table_df.columns:
                technical_table_df["brown_forsythe_pvalue_vs_pre_covid"] = pd.to_numeric(technical_table_df["brown_forsythe_pvalue_vs_pre_covid"], errors="coerce").apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
            
            rename_map_tech = {
                "event_label": "Macro-event regime",
                "median_target_vol_annualized": "Median annualized 3-day volatility",
                "annualized_return_vol": "Annualized daily-return volatility",
                "annualized_target_vol_ci_low": "95% CI low",
                "annualized_target_vol_ci_high": "95% CI high",
                "brown_forsythe_pvalue_vs_pre_covid": "Brown-Forsythe p-value",
                "target_vol_iqr": "3-day volatility IQR",
                "target_vol_mad_annualized": "Annualized robust dispersion",
            }
            st.dataframe(technical_table_df.rename(columns=rename_map_tech), hide_index=True, use_container_width=True)
    
    st.markdown(
        "<div class='method-box'><strong>Economic reading.</strong> Completed regimes are compared using average regime volatility, peak-stress volatility, share of high-volatility days, and sample-size reliability. The Pre-COVID baseline remains visible, and the broad non-crisis baseline provides a more balanced normal reference. Ongoing regimes are excluded from the completed-regime ranking and shown separately as preliminary live risk context.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='method-box'><strong>Methodology note.</strong> Descriptive macro-event volatility statistics are computed from a stable USD/TND event-analysis dataset, not from the final ML feature matrix. This prevents changes in optional engineered model features from changing historical event-volatility comparisons. Forecast performance by event regime is still computed from the walk-forward model outputs.</div>",
        unsafe_allow_html=True,
    )
    
    # Developer diagnostics (collapsed by default)
    with st.expander("Developer diagnostics: macro-event artifacts", expanded=False):
        st.write({
            "source": event_source,
            "artifact_path": str(event_artifact_path) if event_artifact_path else "Unavailable",
            "robust_columns_detected": bool(robust_cols_detected),
            "detected_robust_columns": robust_cols_detected,
            "completed_regime_count": int(len(completed_event_df)),
            "ongoing_regime_count": int(len(ongoing_event_df)),
            "columns": event_df.columns.tolist(),
        })

def render_model_performance_by_event(payload: Optional[dict]) -> None:
    page_header("Model Performance by Event", "Regime validation", payload)
    if payload is None:
        render_no_payload_choice()
        return
    period = st.radio("Evaluation period", ["Holdout", "Development"], horizontal=True)
    df = get_event_metrics_df(payload, period.lower())
    if df.empty:
        st.info("Event-specific forecast metrics are not available.")
        return
    df = df.copy()
    if "event_regime" in df.columns:
        df["event_label"] = df["event_regime"].map(pretty_event_name)
    if "model" in df.columns:
        df["financial_model"] = df["model"].map(financial_model_label)

    blend = df[df.get("financial_model", pd.Series(dtype=str)).eq("Final Macro-Event FX Volatility Engine")].copy()
    if blend.empty and "model" in df.columns:
        blend = df[df["model"].astype(str).str.contains("blend", case=False, na=False)].copy()

    def _is_ongoing_event(row: pd.Series) -> bool:
        regime = str(row.get("event_regime", "")).lower()
        label = str(row.get("event_label", "")).lower()
        return "iran_geopolitical_shock" in regime or "iran" in regime or "iran" in label

    def _regime_type(row: pd.Series) -> str:
        if _is_ongoing_event(row):
            return "Ongoing / preliminary"
        crisis = safe_float(row.get("crisis_flag"))
        if crisis == 1:
            return "Crisis"
        if crisis == 0:
            return "Non-crisis"
        return "Non-crisis"

    def _format_event_metric(value: Any, metric_name: str = "RMSE_red_pct_vs_naive", digits: int = 2) -> str:
        x = safe_float(value)
        if x is None:
            return "n/a"
        if metric_name == "R2_vs_naive":
            return f"{x:.4f}"
        return f"+{x:.{digits}f}%" if x >= 0 else f"{x:.{digits}f}%"
    
    if not blend.empty:
        # Determine the metric column for comparison
        metric_col = "RMSE_red_pct_vs_naive" if "RMSE_red_pct_vs_naive" in blend.columns else "R2_vs_naive" if "R2_vs_naive" in blend.columns else None
        
        if metric_col:
            # Sort by metric DESCENDING (highest value first)
            blend_sorted = blend.sort_values(metric_col, ascending=False, na_position="last").copy()
            blend_sorted["_metric_value"] = pd.to_numeric(blend_sorted[metric_col], errors="coerce")
            blend_sorted = blend_sorted.sort_values("_metric_value", ascending=False, na_position="last").copy()
            metric_is_rmse_reduction = metric_col == "RMSE_red_pct_vs_naive"
            
            # Create panel title
            panel_title = f"Final Model Performance by Event - {period}"
            panel_subtitle = (
                "RMSE reduction versus the naïve benchmark. Higher values indicate stronger forecast improvement."
                if metric_is_rmse_reduction
                else "R-squared improvement versus the naïve benchmark. Higher values indicate stronger benchmark-relative fit."
            )
            
            st.markdown(
                f"""
                <style>
                .event-performance-shell {{
                    margin: 0 0 1rem 0;
                    padding: 1rem 1.05rem;
                    border: 1px solid rgba(79, 123, 255, 0.20);
                    border-radius: 8px;
                    background: linear-gradient(145deg, rgba(16, 26, 44, 0.98), rgba(10, 18, 32, 0.96));
                    box-shadow: 0 18px 44px rgba(0,0,0,0.22);
                }}
                .event-performance-title {{
                    font-size: 1.05rem;
                    font-weight: 850;
                    color: {FINANCE_COLORS["text"]};
                    margin-bottom: 0.18rem;
                }}
                .event-performance-subtitle {{
                    color: {FINANCE_COLORS["muted"]};
                    font-size: 0.86rem;
                    line-height: 1.45;
                }}
                .event-rank-card {{
                    position: relative;
                    overflow: hidden;
                    margin-bottom: 0.72rem;
                    padding: 0.86rem 0.9rem 0.82rem 0.9rem;
                    border: 1px solid rgba(168,179,204,0.14);
                    border-left: 3px solid var(--rank-accent);
                    border-radius: 8px;
                    background: linear-gradient(145deg, rgba(18,31,52,0.98), rgba(11,20,35,0.98));
                    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
                    transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
                }}
                .event-rank-card:hover {{
                    transform: translateY(-2px);
                    border-color: rgba(47,216,201,0.32);
                    box-shadow: 0 18px 34px rgba(0,0,0,0.26);
                }}
                .event-rank-card.rank-one {{
                    padding-top: 1rem;
                    border-color: rgba(47,216,201,0.36);
                    box-shadow: 0 18px 42px rgba(47,216,201,0.08), 0 20px 44px rgba(0,0,0,0.26);
                }}
                .event-rank-topline {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.7rem;
                    margin-bottom: 0.52rem;
                }}
                .event-rank-badge {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 2.2rem;
                    height: 1.55rem;
                    border-radius: 999px;
                    color: #06101D;
                    background: var(--rank-accent);
                    font-size: 0.78rem;
                    font-weight: 900;
                }}
                .event-rank-pill {{
                    border: 1px solid var(--pill-border);
                    border-radius: 999px;
                    padding: 0.18rem 0.48rem;
                    color: var(--pill-color);
                    background: var(--pill-bg);
                    font-size: 0.68rem;
                    font-weight: 800;
                    white-space: nowrap;
                }}
                .event-rank-name {{
                    color: {FINANCE_COLORS["text"]};
                    font-size: 0.92rem;
                    line-height: 1.25;
                    font-weight: 750;
                    margin-bottom: 0.42rem;
                }}
                .event-rank-value {{
                    color: var(--rank-accent);
                    font-size: 1.55rem;
                    line-height: 1;
                    font-weight: 900;
                    margin-bottom: 0.4rem;
                }}
                .event-rank-card.rank-one .event-rank-value {{
                    font-size: 1.86rem;
                }}
                .event-rank-meta {{
                    color: {FINANCE_COLORS["muted"]};
                    font-size: 0.76rem;
                    letter-spacing: 0.01em;
                }}
                .event-interpretation {{
                    margin-top: 0.95rem;
                    padding: 0.95rem 1.05rem;
                    border-radius: 8px;
                    border: 1px solid rgba(47,216,201,0.18);
                    background: linear-gradient(145deg, rgba(16, 26, 44, 0.98), rgba(11, 20, 35, 0.98));
                    color: {FINANCE_COLORS["text"]};
                    box-shadow: 0 14px 34px rgba(0,0,0,0.18);
                    line-height: 1.55;
                }}
                .event-interpretation strong {{
                    color: {FINANCE_COLORS["teal"]};
                }}
                </style>
                <div class="event-performance-shell">
                    <div class="event-performance-title">{html.escape(panel_title)}</div>
                    <div class="event-performance-subtitle">{html.escape(panel_subtitle)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Two-column layout: left summary, right chart
            left_col, right_col = st.columns([0.35, 0.65], gap="large")
            
            with left_col:
                st.markdown(
                    f"<div style='font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; color:{FINANCE_COLORS['muted']}; font-weight:850; margin:0 0 0.62rem 0;'>Event ranking</div>",
                    unsafe_allow_html=True,
                )
                for idx, (_, row) in enumerate(blend_sorted.iterrows(), 1):
                    event_name = str(row.get("event_label", "Unknown"))
                    metric_text = _format_event_metric(row.get("_metric_value"), metric_col)
                    obs_value = safe_float(row.get("observations"))
                    obs = int(obs_value) if obs_value is not None else 0
                    regime_type = _regime_type(row)
                    meta_parts = [f"n={obs:,}"]
                    meta_parts.append(regime_type)
                    meta_text = " | ".join(meta_parts)

                    if regime_type == "Ongoing / preliminary":
                        accent = FINANCE_COLORS["gold"]
                        pill_color = FINANCE_COLORS["gold"]
                        pill_bg = "rgba(243,185,82,0.12)"
                        pill_border = "rgba(243,185,82,0.32)"
                    elif regime_type == "Crisis":
                        accent = "#FF8A8F"
                        pill_color = "#FFB0B4"
                        pill_bg = "rgba(255,95,104,0.12)"
                        pill_border = "rgba(255,95,104,0.28)"
                    else:
                        accent = FINANCE_COLORS["teal"] if (safe_float(row.get("_metric_value")) or 0) >= 0 else FINANCE_COLORS["red"]
                        pill_color = FINANCE_COLORS["teal"]
                        pill_bg = "rgba(47,216,201,0.10)"
                        pill_border = "rgba(47,216,201,0.26)"

                    rank_class = "event-rank-card rank-one" if idx == 1 else "event-rank-card"
                    st.markdown(
                        f"""
                        <div class="{rank_class}" style="--rank-accent:{accent}; --pill-color:{pill_color}; --pill-bg:{pill_bg}; --pill-border:{pill_border};">
                            <div class="event-rank-topline">
                                <span class="event-rank-badge">#{idx}</span>
                                <span class="event-rank-pill">{html.escape(regime_type)}</span>
                            </div>
                            <div class="event-rank-name">{html.escape(event_name)}</div>
                            <div class="event-rank-value">{html.escape(metric_text)}</div>
                            <div class="event-rank-meta">{html.escape(meta_text)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
            with right_col:
                # Create horizontal bar chart
                fig = go.Figure()
                
                # Determine colors: positive=teal, negative=red, ongoing=amber
                colors = []
                for _, row in blend_sorted.iterrows():
                    val = safe_float(row.get("_metric_value"))
                    if _is_ongoing_event(row):
                        colors.append(FINANCE_COLORS["gold"])  # Amber for ongoing
                    elif val is None:
                        colors.append(FINANCE_COLORS["slate"])
                    elif val >= 0:
                        colors.append(FINANCE_COLORS["teal"])
                    else:
                        colors.append(FINANCE_COLORS["red"])
                
                # Format hover text
                hover_data = []
                for _, row in blend_sorted.iterrows():
                    rmse = safe_float(row.get("RMSE"))
                    mae = safe_float(row.get("MAE"))
                    corr = safe_float(row.get("Corr"))
                    r2 = safe_float(row.get("R2_vs_naive"))
                    obs = int(safe_float(row.get("observations")) or 0)
                    regime_type = _regime_type(row)
                    metric_label = "RMSE reduction" if metric_is_rmse_reduction else "R-squared vs naïve"
                    
                    hover_parts = [
                        f"Event: {row.get('event_label', 'n/a')}",
                        f"{metric_label}: {_format_event_metric(row.get('_metric_value'), metric_col)}",
                        f"RMSE: {rmse:.6f}" if rmse is not None else "RMSE: n/a",
                        f"MAE: {mae:.6f}" if mae is not None else "MAE: n/a",
                        f"Correlation: {corr:.4f}" if corr is not None else "Correlation: n/a",
                        f"R² vs naïve: {r2:.4f}" if r2 is not None else "R² vs naïve: n/a",
                        f"Observations: n={obs}",
                        f"Regime: {regime_type}",
                    ]
                    hover_data.append("<br>".join(hover_parts))
                
                # Format bar labels
                bar_labels = [_format_event_metric(row.get("_metric_value"), metric_col, digits=1) for _, row in blend_sorted.iterrows()]
                
                fig.add_trace(go.Bar(
                    x=blend_sorted["_metric_value"],
                    y=blend_sorted["event_label"],
                    orientation="h",
                    text=bar_labels,
                    textposition="outside",
                    textfont=dict(color=FINANCE_COLORS["text"], size=13, family="Arial Black"),
                    marker_color=colors,
                    marker_line=dict(color="rgba(255,255,255,0.20)", width=1),
                    opacity=0.95,
                    hovertext=hover_data,
                    hoverinfo="text",
                    cliponaxis=False,
                    showlegend=False,
                ))
                
                # Update layout - NO TITLE to avoid undefined
                fig.update_layout(
                    title=None,
                    xaxis_title="RMSE reduction vs naïve (%)" if metric_is_rmse_reduction else "R² vs naïve",
                    yaxis_title="",
                    xaxis=dict(
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor="rgba(168,179,204,0.35)",
                        gridcolor="rgba(168,179,204,0.10)",
                        tickfont=dict(color=FINANCE_COLORS["muted"]),
                    ),
                    yaxis=dict(
                        categoryorder="array",
                        categoryarray=blend_sorted["event_label"].tolist(),
                        autorange="reversed",
                        tickfont=dict(color=FINANCE_COLORS["text"], size=12),
                    ),
                    plot_bgcolor="rgba(7,16,29,0.18)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=FINANCE_COLORS["text"], family="Arial"),
                    margin=dict(l=160, r=70, t=6, b=44),
                    height=max(320, 70 * len(blend_sorted)),
                    hovermode="closest",
                    bargap=0.36,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Compact interpretation strip using the sorted dataframe
            best_event = blend_sorted.iloc[0] if not blend_sorted.empty else None
            hardest_event = blend.sort_values("RMSE", ascending=False).iloc[0] if "RMSE" in blend.columns and not blend.empty else None
            has_iran = any(_is_ongoing_event(r) for _, r in blend.iterrows())
            
            interp_parts = []
            if best_event is not None:
                best_name = str(best_event.get("event_label", "Unknown"))
                best_val = safe_float(best_event.get("_metric_value"))
                if best_val is not None:
                    best_metric_text = _format_event_metric(best_val, metric_col)
                    if metric_is_rmse_reduction:
                        interp_parts.append(f"<strong>Best event performance:</strong> {html.escape(best_name)}, where the final model reduced RMSE by <strong>{html.escape(best_metric_text)}</strong> versus the naïve benchmark.")
                    else:
                        interp_parts.append(f"<strong>Best event performance:</strong> {html.escape(best_name)}, with benchmark-relative R-squared of <strong>{html.escape(best_metric_text)}</strong>.")
            
            if hardest_event is not None:
                hard_name = str(hardest_event.get("event_label", "Unknown"))
                hard_rmse = safe_float(hardest_event.get("RMSE"))
                if hard_rmse is not None:
                    interp_parts.append(f"<strong>Hardest regime to forecast:</strong> {html.escape(hard_name)} with RMSE {hard_rmse:.6f}.")
            
            if has_iran:
                interp_parts.append("<strong>Iran Geopolitical Shock</strong> is ongoing, so its event-level performance should be read as preliminary.")
            
            if interp_parts:
                interp_text = " ".join(interp_parts)
                st.markdown(f"<div class='event-interpretation'>{interp_text}</div>", unsafe_allow_html=True)
        else:
            st.warning("No benchmark-relative performance metric available for the selected period.")

    # Improved detailed table
    rename_map = {
        "event_label": "Event regime",
        "financial_model": "Model",
        "observations": "Observations",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "Bias": "Bias",
        "QLIKE": "QLIKE",
        "Corr": "Correlation",
        "R2_vs_naive": "R² vs naïve",
        "RMSE_red_pct_vs_naive": "RMSE reduction vs naïve",
        "_regime_type": "Regime type",
    }
    
    display_source = df.copy()
    display_source["_regime_type"] = display_source.apply(_regime_type, axis=1)
    display_cols = [c for c in ["event_label", "financial_model", "observations", "RMSE", "MAE", "Bias", "QLIKE", "Corr", "R2_vs_naive", "RMSE_red_pct_vs_naive", "_regime_type"] if c in display_source.columns]
    table_df = display_source[display_cols].copy()
    
    # Format numeric columns
    for col in ["RMSE", "MAE", "Bias", "QLIKE"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").apply(lambda x: f"{x:.6f}" if pd.notna(x) else "n/a")
    
    for col in ["Corr", "R2_vs_naive"]:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors="coerce").apply(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
    
    if "RMSE_red_pct_vs_naive" in table_df.columns:
        table_df["RMSE_red_pct_vs_naive"] = pd.to_numeric(table_df["RMSE_red_pct_vs_naive"], errors="coerce").apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    if "observations" in table_df.columns:
        table_df["observations"] = pd.to_numeric(table_df["observations"], errors="coerce").apply(lambda x: f"{int(x):,}" if pd.notna(x) else "n/a")
    
    st.subheader("Detailed Model Metrics")
    st.dataframe(table_df.rename(columns=rename_map), hide_index=True, use_container_width=True)
    
    # Metrics explainer expander
    with st.expander("Metric definitions", expanded=False):
        st.markdown("""
        - **RMSE**: Root Mean Squared Error. Lower is better. Penalizes large forecast errors.
        - **MAE**: Mean Absolute Error. Lower is better. Measures average absolute forecast error.
        - **Bias**: Average forecast error. Positive means overforecasting; negative means underforecasting.
        - **QLIKE**: Volatility-forecast loss metric. Lower is better.
        - **Correlation**: Shows whether forecasts move with realized volatility. Higher is better.
        - **R² vs naïve**: Improvement in squared-error terms versus the naïve benchmark. Positive is better.
        - **RMSE reduction vs naïve**: Percentage reduction in RMSE versus the naïve benchmark. Higher is better.
        """)


def render_global_fx_context(payload: Optional[dict]) -> None:
    page_header("USD/TND vs Global FX Risk Context", "Market context", payload)
    if payload is None:
        render_no_payload_choice()
        return
    gfx = (payload.get("dashboard", {}) or {}).get("global_fx_comparison", {}) if payload else {}
    latest = gfx.get("latest", {}) or {}
    ratio = safe_float(latest.get("ratio"))
    classification = latest.get("classification", "Unavailable")
    render_kpi_grid([
        metric_card("USD/TND Volatility", format_pct(latest.get("usdtnd_volatility")), "Local FX risk", "teal"),
        metric_card("Global FX Average", format_pct(latest.get("global_fx_mean")), "Major FX proxy basket", "primary"),
        metric_card("Local/Global Ratio", format_ratio(ratio), "USD/TND vs global FX", "gold"),
        metric_card("Context Signal", str(classification), "Relative volatility state", "purple"),
    ])
    if ratio is not None:
        if ratio > 1.0:
            insight = "USD/TND volatility is above the global FX basket. Traders should investigate local liquidity, regional risk, and USD/TND-specific pressure before relying on passive hedge timing."
        elif ratio < 1.0:
            insight = "USD/TND volatility is below the global FX basket. Local conditions appear contained relative to major FX, but event headlines should remain monitored."
        else:
            insight = "USD/TND volatility is aligned with global FX conditions. Standard risk protocols are appropriate."
        st.markdown(f"<div class='method-box'><strong>Market read:</strong> {html.escape(insight)}</div>", unsafe_allow_html=True)
    history = records_df(gfx.get("history", []) if gfx else [])
    if not history.empty:
        st.plotly_chart(line_fig(history, "Date", [("usdtnd_volatility", "USD/TND volatility"), ("global_fx_mean", "Global FX average")], y_scale=100, yaxis_title="Volatility (%)"), use_container_width=True)
        st.plotly_chart(line_fig(history, "Date", [("ratio", "USD/TND / Global FX")], yaxis_title="Ratio"), use_container_width=True)
    else:
        st.info("Global FX comparison data is not available in the current payload.")


def render_calendar_patterns(payload: Optional[dict]) -> None:
    page_header("Seasonality & Calendar Risk Patterns", "Calendar diagnostics", payload)
    if payload is None:
        render_no_payload_choice()
        return
    calendar = (payload.get("dashboard", {}) or {}).get("calendar_effect", {}) if payload else {}
    heat = pd.DataFrame(calendar.get("heatmap_matrix", []) or [])
    if heat.empty:
        st.info("Calendar-effect diagnostics are not available.")
        return
    matrix = heat.pivot(index="month", columns="day_of_week", values="average_volatility")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    matrix = matrix.reindex(columns=[day for day in day_order if day in matrix.columns])
    day_avg = matrix.mean(axis=0)
    month_avg = matrix.mean(axis=1)
    render_kpi_grid([
        metric_card("Highest-Risk Weekday", str(day_avg.idxmax()), f"Avg vol {format_pct(day_avg.max())}", "gold"),
        metric_card("Lowest-Risk Weekday", str(day_avg.idxmin()), f"Avg vol {format_pct(day_avg.min())}", "green"),
        metric_card("Highest-Risk Month", str(month_avg.idxmax()), f"Avg vol {format_pct(month_avg.max())}", "red"),
        metric_card("Calendar Method", "Descriptive", "Not causal; not standalone forecast", "primary"),
    ])
    fig = go.Figure(go.Heatmap(z=matrix.values * 100, x=matrix.columns, y=matrix.index, colorscale="Turbo", colorbar_title="Avg vol (%)", hovertemplate="%{y}, %{x}<br>Avg vol: %{z:.3f}%<extra></extra>"))
    fig.update_layout(title="Calendar-Based USD/TND Volatility Patterns", xaxis_title="", yaxis_title="")
    st.plotly_chart(apply_finance_layout(fig, height=430), use_container_width=True)
    st.markdown("<div class='method-box'><strong>Methodology note.</strong> Calendar effects are descriptive historical diagnostics. They help monitoring and execution planning but are not treated as causal drivers.</div>", unsafe_allow_html=True)
    if calendar.get("commentary"):
        st.write(calendar["commentary"])


def render_forecast_engine_explainability(payload: Optional[dict]) -> None:
    page_header("Forecast Engine & Benchmark Attribution", "Model architecture and validation", payload)
    if payload is None:
        render_no_payload_choice()
        return
    forecast = payload.get("forecast", {}) or {}
    render_kpi_grid([
        metric_card("Traditional Volatility Benchmark", format_pct(forecast.get("garch_anchor")), "GARCH / EGARCH reference forecast", "gold"),
        metric_card("Linear Macro-Risk Stabilizer", format_pct(forecast.get("pred_ridge")), "Ridge component", "primary"),
        metric_card("Nonlinear Market-Risk Engine", format_pct(forecast.get("pred_xgb")), "XGBoost component", "purple"),
        metric_card("Final Model Output", format_pct(forecast.get("final_forecast_blend")), "Blended engine", "teal"),
    ])

    summary_df = get_summary_df(payload, "holdout")
    if not summary_df.empty:
        df = summary_df.copy()
        model_col = "Model" if "Model" in df.columns else "model" if "model" in df.columns else None
        if model_col:
            df["Financial model label"] = df[model_col].map(financial_model_label)
        keep = [c for c in ["Financial model label", model_col, "RMSE", "MAE", "QLIKE", "R2_vs_naive", "RMSE_red_pct", "Corr"] if c and c in df.columns]
        st.subheader("Holdout model comparison")
        st.dataframe(df[keep], hide_index=True, use_container_width=True)

    dash = payload.get("dashboard", {}) or {}
    weights = dash.get("model_weights", {}) or {}
    if weights:
        st.subheader("Blend weights")
        dfw = pd.DataFrame([
            {"Component": "Linear Macro-Risk Stabilizer", "Weight": weights.get("ridge_weight", 0)},
            {"Component": "Nonlinear Market-Risk Engine", "Weight": weights.get("xgb_weight", weights.get("final_dynamic_xgb_weight", 0))},
        ])
        fig = go.Figure(go.Bar(x=pd.to_numeric(dfw["Weight"], errors="coerce"), y=dfw["Component"], orientation="h", marker_color=[FINANCE_COLORS["primary"], FINANCE_COLORS["teal"]]))
        fig.update_layout(title="Final Model Blend Allocation", xaxis_tickformat=".0%", xaxis_range=[0, 1])
        st.plotly_chart(apply_finance_layout(fig, height=260), use_container_width=True)

    features = (dash.get("feature_importance", {}) or {}).get("top_5", [])
    fdf = pd.DataFrame(features)
    if not fdf.empty and {"feature_name", "importance_score"}.issubset(fdf.columns):
        st.subheader("Top market-risk drivers")
        fdf = fdf.sort_values("importance_score")
        fig = go.Figure(go.Bar(x=fdf["importance_score"], y=fdf["feature_name"], orientation="h", marker_color=FINANCE_COLORS["teal"]))
        fig.update_layout(title="Feature Importance - Nonlinear Market-Risk Engine", xaxis_title="Relative importance")
        st.plotly_chart(apply_finance_layout(fig, height=350), use_container_width=True)


def render_scenario_laboratory(payload: Optional[dict]) -> None:
    page_header("Scenario & Calibration Laboratory", "Calibration sensitivity overlay", payload)
    if payload is None:
        render_no_payload_choice()
        return

    classification = classify_forecast_volatility_regime(payload)
    official_forecast = safe_float(classification.get("forecast_volatility"), 0.0) or 0.0
    official_prob = safe_float(classification.get("high_vol_probability"), 0.0) or 0.0
    p05, p95 = get_confidence_band(payload)
    official_regime = classification.get("regime_label", "Unavailable")
    official_stance = risk_stance_label(classification)

    st.markdown(
        """
        <div class="method-box">
            <strong>Scenario objective.</strong> Use this laboratory to test whether the hedge signal is sensitive to alternative market-memory and macro-event assumptions. The official forecast remains the production reference; scenario outputs are comparison overlays.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Official production reference")
    render_kpi_grid([
        metric_card("Official Forecast", format_pct(official_forecast), "Final Macro-Event FX Volatility Engine", "teal"),
        metric_card("Official Regime", str(official_regime), "Empirical threshold classification", "gold"),
        metric_card("High-Vol Probability", format_pct(official_prob), "Production high-volatility signal", "purple"),
        metric_card("Hedging Stance", official_stance, "Production decision anchor", "primary"),
    ])

    st.subheader("Scenario controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        scenario_name = st.text_input("Scenario name", value="Short-cycle stress overlay")
        scenario_window_options = calibration_window_options("Custom")
        window = st.selectbox(
            "Calibration window assumption",
            scenario_window_options,
            index=default_calibration_window_index(scenario_window_options),
            key="scenario_calibration_window_v2",
        )
    with c2:
        event_focus = st.multiselect(
            "Macro-event focus",
            ["COVID shock", "Russia-Ukraine shock", "U.S. tariff shock", "Iran geopolitical shock", "Post-conflict normalization"],
            default=[],
            key="scenario_event_focus_v2",
        )
        sensitivity = st.select_slider("Event sensitivity", options=["Low", "Moderate", "High"], value="Moderate")
    with c3:
        purpose = st.selectbox("Scenario purpose", ["Short-cycle stress view", "Medium-cycle risk view", "Long-cycle macro view", "Crisis-sensitive overlay"])
        custom_window = st.number_input("Custom window", min_value=250, max_value=1500, value=ROLLING_WINDOW, step=50, disabled=(window != "Custom"))

    window_days = int(custom_window) if window == "Custom" else int(window.split()[0])
    if window_days < 250:
        health = "Insufficient for institutional calibration"
        health_accent = "red"
    elif window_days == ROLLING_WINDOW:
        health = "Production baseline"
        health_accent = "green"
    elif window_days < ROLLING_WINDOW:
        health = "Tactical short-cycle scenario"
        health_accent = "gold"
    elif window_days < ROLLING_WINDOW * 2:
        health = "Medium-cycle scenario"
        health_accent = "primary"
    else:
        health = "Long-cycle macro scenario"
        health_accent = "teal"

    # Scenario overlay is a decision-support sensitivity proxy. It does not rerun
    # forecasting.py until the backend supports run_pipeline(config=...). The
    # official model stays untouched and remains the production reference.
    window_multiplier = scenario_window_multiplier(window_days)
    event_multiplier = 1.0 + 0.025 * len(event_focus)
    sensitivity_multiplier = {"Low": 0.98, "Moderate": 1.00, "High": 1.06}[sensitivity]
    scenario_forecast = official_forecast * window_multiplier * event_multiplier * sensitivity_multiplier
    scenario_prob = min(0.99, max(0.0, official_prob * (scenario_forecast / max(official_forecast, 1e-12))))
    scenario_classification = dict(classification)
    scenario_classification["forecast_volatility"] = scenario_forecast
    scenario_classification["high_vol_probability"] = scenario_prob
    scenario_classification = classify_scenario_against_thresholds(scenario_classification)
    scenario_stance = risk_stance_label(scenario_classification)

    st.subheader("Scenario health check")
    render_kpi_grid([
        metric_card("Selected Window", f"{window_days}d", "Market-memory assumption", health_accent),
        metric_card("Scenario Health", health, "Governance check", health_accent),
        metric_card("Event Focus", str(len(event_focus)), "Selected macro-event regimes", "purple"),
        metric_card("Sensitivity", sensitivity, purpose, "primary"),
    ])

    comparison = pd.DataFrame([
        {"Metric": "3-day volatility forecast", "Official Forecast": format_pct(official_forecast), "Scenario Overlay": format_pct(scenario_forecast), "Difference": f"{(scenario_forecast - official_forecast) * 100:.2f} pp"},
        {"Metric": "Forecast regime", "Official Forecast": official_regime, "Scenario Overlay": scenario_classification.get("regime_label"), "Difference": "Higher risk" if scenario_forecast > official_forecast else "Lower / similar risk"},
        {"Metric": "High-volatility probability", "Official Forecast": format_pct(official_prob), "Scenario Overlay": format_pct(scenario_prob), "Difference": f"{(scenario_prob - official_prob) * 100:.2f} pp"},
        {"Metric": "Confidence band", "Official Forecast": f"{format_pct(p05)} - {format_pct(p95)}", "Scenario Overlay": f"{format_pct((p05 or official_forecast) * window_multiplier)} - {format_pct((p95 or official_forecast) * window_multiplier)}", "Difference": "Wider stress range" if window_multiplier > 1 else "Tighter / slower-moving range"},
        {"Metric": "Hedging stance", "Official Forecast": official_stance, "Scenario Overlay": scenario_stance, "Difference": "More defensive" if scenario_forecast > official_forecast else "Stable / less defensive"},
    ])
    st.subheader("Official vs scenario comparison")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    if scenario_forecast > official_forecast * 1.10:
        interpretation = "The scenario overlay produces a materially higher volatility signal than the official forecast. This points to a more fragile short-term market-memory view and supports tighter exposure limits or partial hedge acceleration."
    elif scenario_forecast < official_forecast * 0.90:
        interpretation = "The scenario overlay produces a lower volatility signal than the official forecast. This suggests recent or selected assumptions may indicate easing pressure, while the official forecast remains the production anchor."
    else:
        interpretation = "The scenario overlay is close to the official forecast. The hedge signal appears stable across the selected calibration assumptions."
    st.markdown(f"<div class='decision-card'><div class='decision-title'>Trader interpretation</div><div class='decision-text'>{html.escape(interpretation)}</div></div>", unsafe_allow_html=True)

    sensitivity_df = pd.DataFrame([
        {
            "Window": f"{days}d",
            "Forecast": official_forecast * scenario_window_multiplier(days),
            "Role": "Official baseline" if days == ROLLING_WINDOW else "Scenario",
        }
        for days in STANDARD_CALIBRATION_WINDOWS
    ])
    fig = go.Figure(go.Bar(
        x=sensitivity_df["Window"],
        y=sensitivity_df["Forecast"] * 100,
        marker_color=[
            FINANCE_COLORS["green"] if role == "Official baseline" else FINANCE_COLORS["teal"]
            for role in sensitivity_df["Role"]
        ],
    ))
    fig.update_layout(title="Calibration-Window Sensitivity View", yaxis_title="Scenario volatility forecast (%)", xaxis_title="Calibration window")
    st.plotly_chart(apply_finance_layout(fig, height=360), use_container_width=True)

    event_df = get_event_descriptive_df(payload)
    if not event_df.empty and "event_regime" in event_df.columns and event_focus:
        event_df = event_df.copy()
        event_df["event_label"] = event_df["event_regime"].map(pretty_event_name)
        selected = event_df[event_df["event_label"].isin(event_focus)]
        if not selected.empty:
            st.subheader("Selected event volatility profile")
            st.dataframe(selected[[c for c in ["event_label", "observations", "mean_target_vol", "mean_target_vol_vs_pre_covid_pct", "mean_target_vol_vs_full_sample_pct"] if c in selected.columns]], hide_index=True, use_container_width=True)


def render_methodology_audit(payload: Optional[dict]) -> None:
    page_header("Methodology & Audit Trail", "Academic governance", payload)
    st.markdown(
        """
        <div class="method-box">
            <strong>Methodology position.</strong> The app forecasts next 3-day USD/TND realized volatility using a macro-event hybrid engine. Forecast regimes are assigned through empirical realized-volatility thresholds, with macro-event channels providing economic context.
        </div>
        """,
        unsafe_allow_html=True,
    )
    methodology_rows = pd.DataFrame([
        {"Area": "Target", "Methodology": "Next 3-day USD/TND realized volatility"},
        {"Area": "Forecast engine", "Methodology": "Ridge + XGBoost + GARCH/EGARCH blended framework"},
        {"Area": "Final model name", "Methodology": "Final Macro-Event FX Volatility Engine"},
        {"Area": "Benchmarks", "Methodology": "FX Volatility Carry-Forward Benchmark and Traditional Volatility Benchmark"},
        {"Area": "Economic regimes", "Methodology": "COVID, Russia-Ukraine, U.S. tariff shock, Iran geopolitical shock"},
        {"Area": "Classification", "Methodology": "q25/q50/q75/q90 realized-volatility thresholds"},
        {"Area": "Decision outputs", "Methodology": "Forecast, regime, probability, confidence band, and hedge stance"},
        {"Area": "Validation", "Methodology": "Chronological walk-forward validation, event-specific diagnostics, and high-volatility probability calibration diagnostics. The Brier score measures probability forecast error, while the reliability diagram compares predicted probabilities with observed high-volatility frequencies. The threshold sensitivity table shows how alert quality changes across different probability cutoffs."},
    ])
    st.dataframe(methodology_rows, hide_index=True, use_container_width=True)

    metadata = payload.get("metadata", {}) if payload else {}
    validation = metadata.get("event_methodology_validation") or {}
    if validation:
        st.subheader("Event methodology validation")
        st.json(validation)
    else:
        st.info("Event methodology validation metadata is not available in this payload.")

    if payload:
        with st.expander("Event context source", expanded=False):
            event_context, event_context_source = get_event_context_details(payload)
            st.markdown(f"**Resolved event context:** {html.escape(event_context)}")
            st.markdown(f"**Source:** `{html.escape(event_context_source)}`")
        st.subheader("Available artifacts")
        artifacts = payload.get("artifacts", {}) or {}
        rows = []
        for key in ARTIFACT_FILENAMES.keys():
            p = artifact_path(payload, key)
            rows.append({"Artifact": ARTIFACT_DISPLAY_NAMES.get(key, key.replace("_", " ").title()), "Key": key, "Available": bool(p), "Path": str(p) if p else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_downloads(payload: Optional[dict]) -> None:
    page_header("Downloads", "Export center", payload)
    if payload is None:
        render_no_payload_choice()
        return
    artifact_rows = []
    for key in ARTIFACT_FILENAMES.keys():
        p = artifact_path(payload, key)
        artifact_rows.append({"Key": key, "File": ARTIFACT_FILENAMES.get(key, ""), "Available": bool(p)})
    st.dataframe(pd.DataFrame(artifact_rows), hide_index=True, use_container_width=True)
    st.markdown("---")
    cols = st.columns(3)
    idx = 0
    for key in ARTIFACT_FILENAMES.keys():
        p = artifact_path(payload, key)
        if not p or not p.exists():
            continue
        with cols[idx % 3]:
            st.download_button(
                ARTIFACT_DISPLAY_NAMES.get(key, key.replace("_", " ").title()),
                data=p.read_bytes(),
                file_name=p.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key=f"download_{key}",
            )
        idx += 1


# ============================================================
# Main app
# ============================================================


st.set_page_config(page_title="VolSight | Macro-Event FX Volatility Engine", layout="wide")
inject_finance_theme()

init_state = {
    "forecast_payload": None,
    "selected_run_id": None,
    "artifact_dir": None,
    "selected_page": PAGE_NAMES[0],
}
for key, value in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("<div class='app-brand'><span class='app-brand-mark'>V</span><span>VolSight</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='app-tagline'>Macro-Event FX Volatility Engine</div>", unsafe_allow_html=True)
    st.session_state.selected_page = st.radio(
        "Dashboard section",
        PAGE_NAMES,
        index=PAGE_NAMES.index(st.session_state.selected_page) if st.session_state.selected_page in PAGE_NAMES else 0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.subheader("Run control")
    if st.button("Load latest saved results", use_container_width=True, key="sidebar_load_latest"):
        try:
            payload_loaded = load_latest_local()
            st.session_state.forecast_payload = payload_loaded
            st.session_state.selected_run_id = payload_loaded.get("run_id")
            st.session_state.artifact_dir = payload_loaded.get("artifact_dir")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.button("Run new forecast", use_container_width=True, key="sidebar_run_new"):
        st.session_state.selected_page = "Data & Market Inputs"
        st.rerun()
    latest_run = find_latest_run_dir()
    if latest_run:
        st.caption(f"Latest local run: {latest_run.name}")
    else:
        st.caption("No saved local runs detected.")
    st.markdown("---")
    st.caption("Official forecast: validated macro-event engine. Scenario Laboratory provides calibration sensitivity overlays.")

payload = st.session_state.forecast_payload
selected_page = st.session_state.selected_page

if selected_page == "Executive FX Risk Cockpit":
    render_executive_fx_risk_cockpit(payload)
elif selected_page == "Data & Market Inputs":
    render_data_market_inputs(payload)
elif selected_page == "Volatility Regime Classification":
    render_volatility_regime_classification(payload)
elif selected_page == "Macro-Event Regime Analysis":
    render_macro_event_regime_analysis(payload)
elif selected_page == "Model Performance by Event":
    render_model_performance_by_event(payload)
elif selected_page == "USD/TND vs Global FX Risk Context":
    render_global_fx_context(payload)
elif selected_page == "Seasonality & Calendar Risk Patterns":
    render_calendar_patterns(payload)
elif selected_page == "Forecast Engine & Benchmark Attribution":
    render_forecast_engine_explainability(payload)
elif selected_page == "Scenario & Calibration Laboratory":
    render_scenario_laboratory(payload)
elif selected_page == "Methodology & Audit Trail":
    render_methodology_audit(payload)
elif selected_page == "Downloads":
    render_downloads(payload)

