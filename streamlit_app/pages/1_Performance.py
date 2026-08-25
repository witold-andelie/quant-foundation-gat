"""Performance page: visible pre/post-GAT comparison plus legacy diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import pick, render_track_selector


st.title("Performance: Before vs After Graph Modelling")
track, is_energy, db, tables = render_track_selector(key="perf_track")
st.caption(
    "Headline comparisons are shown first. The old trading backtest is retained "
    "below as a historical diagnostic, not as the final GAT result."
)

if is_energy:
    st.subheader("European electricity")
    st.warning(
        "**Before:** the cross-sectional trading-alpha hypothesis produced Sharpe "
        "-0.80 and max drawdown -100%. The hypothesis was rejected and the target "
        "was reframed; forecast skill below is not the same unit as Sharpe."
    )
    comparison = pd.DataFrame(
        [
            {
                "Stage": "Before GAT",
                "Model": "Cross-sectional trading alpha",
                "Primary evidence": "Sharpe -0.80; max DD -100%",
                "Conclusion": "Rejected hypothesis",
            },
            {
                "Stage": "After reframe",
                "Model": "No-graph -> uniform graph",
                "Primary evidence": "Price skill 0.224 -> 0.355 (+0.131)",
                "Conclusion": "Physical neighbours help",
            },
            {
                "Stage": "After GAT",
                "Model": "Uniform -> GAT node",
                "Primary evidence": "Skill 0.355 -> 0.347; rank IC 0.584 -> 0.612",
                "Conclusion": "Better ranking, not MSE",
            },
            {
                "Stage": "After GAT",
                "Model": "Edge ridge -> edge GAT",
                "Primary evidence": "Spread skill 0.192 -> 0.248 (+0.056)",
                "Conclusion": "Strongest relational result; 5/5 seeds",
            },
        ]
    )
else:
    st.subheader("US equities")
    st.warning(
        "**Before:** the naive equal-weight composite produced OOS Sharpe -1.39 "
        "and max drawdown -60.1%. It is context, not a capacity-matched GAT control."
    )
    comparison = pd.DataFrame(
        [
            {
                "Stage": "Before GAT",
                "Model": "Naive equal-weight composite",
                "OOS Sharpe": -1.390,
                "Conclusion": "Weak contextual baseline",
            },
            {
                "Stage": "Graph anchor",
                "Model": "Uniform aggregation",
                "OOS Sharpe": -0.001,
                "Conclusion": "Controlled relational anchor",
            },
            {
                "Stage": "After GAT",
                "Model": "Selected GAT composite",
                "OOS Sharpe": 1.420,
                "Conclusion": "Improved anchor; 3/4 gates; 30/30 seeds",
            },
            {
                "Stage": "Reference ceiling",
                "Model": "Best single factor",
                "OOS Sharpe": 2.880,
                "Conclusion": "Still stronger than selected GAT",
            },
        ]
    )

st.dataframe(
    comparison,
    width="stretch",
    hide_index=True,
    column_config={
        "OOS Sharpe": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.caption(
    "Seed repetition measures optimisation stability, not independent statistical "
    "significance. Naive and GAT models are not capacity-matched."
)
if st.button(
    "Open full GAT evidence, controls, and limitations",
    width="stretch",
):
    st.switch_page("pages/8_GAT_Forecasting.py")

backtest = pick(db, *tables["backtest"])
metrics = pick(db, *tables["metrics"])

if backtest.empty:
    st.info("No local legacy backtest data are available for this track.")
    st.stop()

x_col = next((c for c in ("market_ts", "date") if c in backtest.columns), "date")
backtest = backtest.sort_values(x_col).copy()
backtest[x_col] = pd.to_datetime(backtest[x_col])


def _metric(column: str, fmt: str) -> str:
    if metrics.empty or column not in metrics.columns:
        return "-"
    try:
        return fmt.format(float(metrics[column].iloc[0]))
    except Exception:
        return "-"


def _geometric_cagr(frame: pd.DataFrame) -> float | None:
    if "equity_curve" not in frame.columns or len(frame) < 2:
        return None
    nav = pd.to_numeric(frame["equity_curve"], errors="coerce").dropna()
    if nav.empty:
        return None
    elapsed = (
        pd.Timestamp(frame[x_col].iloc[-1]) - pd.Timestamp(frame[x_col].iloc[0])
    ).total_seconds()
    years = elapsed / (365.25 * 24 * 60 * 60)
    if years <= 0:
        return None
    final_nav = float(nav.iloc[-1])
    if final_nav <= 0:
        return -1.0
    return float(np.expm1(np.log(final_nav) / years))


cagr = _geometric_cagr(backtest)
cagr_label = "-" if cagr is None else f"{cagr:.1%}"

with st.expander("Inspect legacy pre-GAT NAV and diagnostics"):
    st.info(
        "This section describes the superseded trading baseline. Its stored "
        "arithmetic annualised mean P&L is intentionally not labelled as annual "
        "investment return; CAGR below is derived geometrically from NAV."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Return", _metric("total_return", "{:.1%}"))
    c2.metric(
        "CAGR",
        cagr_label,
        help="Geometric annual growth from ending NAV and actual elapsed time.",
    )
    c3.metric("Ann. Vol", _metric("annualized_volatility", "{:.1%}"))
    c4.metric("Sharpe", _metric("sharpe", "{:.2f}"))
    c5.metric("Sortino", _metric("sortino", "{:.2f}"))
    c6.metric("Max Drawdown", _metric("max_drawdown", "{:.1%}"))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Legacy equity curve (NAV)", "Drawdown"],
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=backtest[x_col],
            y=backtest["equity_curve"],
            mode="lines",
            name="NAV",
            line={"color": "#2563EB", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=1, line_dash="dot", line_color="gray", row=1, col=1)

    drawdown = backtest["equity_curve"] / backtest["equity_curve"].cummax() - 1
    fig.add_trace(
        go.Scatter(
            x=backtest[x_col],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line={"color": "#EF4444", "width": 1},
            fillcolor="rgba(239,68,68,0.15)",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    if "portfolio_return" in backtest.columns:
        returns = backtest.set_index(x_col)["portfolio_return"]
        periods_per_year = 2190 if is_energy else 252
        rolling = (
            returns.rolling(63).mean()
            / returns.rolling(63).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        period_name = "4-hour observations" if is_energy else "trading days"
        rolling_fig = px.line(
            x=rolling.index,
            y=rolling.values,
            title=f"Rolling 63-period Sharpe ({period_name})",
            labels={"x": "", "y": "Sharpe"},
            color_discrete_sequence=["#7C3AED"],
        )
        rolling_fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(rolling_fig, width="stretch")

        left, right = st.columns(2)
        with left:
            interval = "4-hour" if is_energy else "daily"
            hist = px.histogram(
                backtest,
                x="portfolio_return",
                nbins=60,
                title=f"Legacy {interval} return distribution",
                labels={"portfolio_return": f"{interval.title()} return"},
                color_discrete_sequence=["#2563EB"],
            )
            hist.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(hist, width="stretch")
        with right:
            values = backtest["portfolio_return"].dropna()
            st.metric("Win rate", f"{(values > 0).mean():.1%}")
            st.metric("Skewness", f"{float(values.skew()):.2f}")
            st.metric("Excess kurtosis", f"{float(values.kurt()):.2f}")
            if "observations" in metrics.columns and not metrics.empty:
                st.metric("Observations", int(metrics["observations"].iloc[0]))

    if "long_count" in backtest.columns and "short_count" in backtest.columns:
        st.subheader("Legacy long / short holdings")
        holdings = go.Figure()
        holdings.add_trace(
            go.Scatter(
                x=backtest[x_col],
                y=backtest["long_count"],
                mode="lines",
                name="Longs",
                line={"color": "#22C55E"},
            )
        )
        holdings.add_trace(
            go.Scatter(
                x=backtest[x_col],
                y=-backtest["short_count"],
                mode="lines",
                name="Shorts (inverted)",
                line={"color": "#EF4444"},
            )
        )
        holdings.update_layout(height=250, yaxis_title="Count")
        st.plotly_chart(holdings, width="stretch")
