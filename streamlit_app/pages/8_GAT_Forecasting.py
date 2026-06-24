"""Page 8 — GNN/GAT capstone: relational factors (equity) + energy forecasting.

Reads committed result artifacts under ``docs/results/`` (so the view works on the
Streamlit Cloud demo, where the cached real ENTSO-E data is not shipped) and,
when present, the equity GAT dbt marts from DuckDB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]          # streamlit_app/
REPO = ROOT.parent
RESULTS = REPO / "docs/results"
sys.path.insert(0, str(ROOT))
from common import EQUITY_DB, load_table  # noqa: E402

st.title("🕸️ GNN/GAT Capstone — Relational Factors & Forecasting")
st.caption(
    "Relational factors propagate over a graph (correlation graph for equities, "
    "physical interconnector graph for energy) via one shared GAT kernel, plus a "
    "leakage-controlled energy price/spread forecasting study. "
    "Full record: docs/energy_forecasting.md · docs/gat_experiment_log.md (E14)."
)


def _csv(name: str) -> pd.DataFrame:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _skill_bar(df: pd.DataFrame, scale: str):
    fig = px.bar(
        df.sort_values("skill"), x="skill", y="predictor", orientation="h",
        color="skill", color_continuous_scale=scale, text="skill",
        hover_data=[c for c in ("rank_ic", "note") if c in df.columns],
    )
    fig.update_traces(texttemplate="%{text:.3f}")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      height=300, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="skill vs persistence", yaxis_title="")
    return fig


# ── Energy forecasting ────────────────────────────────────────────────────────
st.header("⚡ Energy price / spread forecasting")
st.caption("Skill = 1 − MSE/MSE(persistence), out-of-sample · real ENTSO-E, 20 zones, 6 months, k=24h.")

node, edge, findings = _csv("energy_forecast_node_skill.csv"), _csv("energy_forecast_edge_skill.csv"), _csv("energy_gnn_findings.csv")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Node-level — next-period price")
    if not node.empty:
        st.plotly_chart(_skill_bar(node, "Blues"), use_container_width=True)
        st.success("Graph lift (uniform − no-graph): **+0.131** skill — validated by a synthetic negative control (~0).")
with c2:
    st.subheader("Edge-level — cross-border spread")
    if not edge.empty:
        st.plotly_chart(_skill_bar(edge, "Greens"), use_container_width=True)
        st.success("Message passing beats the both-endpoint model: **+0.056 skill, 5/5 seeds** — the GNN's value on the irreducibly-relational target.")

st.subheader("Findings (E14) — the robust wins *and* the honest nulls")
if not findings.empty:
    def _hl(row):
        v = str(row.get("verdict", ""))
        bg = {"robust *": "#1b5e2055", "robust": "#1b5e2033"}.get(v, "#80808022" if v in ("wash", "null") else "")
        return [f"background-color:{bg}"] * len(row)
    st.dataframe(findings.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)

# ── Equity relational factors (GAT) ───────────────────────────────────────────
st.divider()
st.header("📈 Equity relational factors (GAT)")
eq = _csv("equity_gat_summary.csv")
if not eq.empty:
    st.dataframe(eq, use_container_width=True, hide_index=True)

scorecard = load_table(EQUITY_DB, "fct_gat_scorecard")
vs_baseline = load_table(EQUITY_DB, "fct_gat_vs_baseline")
if not scorecard.empty:
    st.subheader("GAT scorecard (live, from DuckDB marts)")
    st.dataframe(scorecard, use_container_width=True, hide_index=True)
if not vs_baseline.empty:
    st.subheader("GAT vs baseline anchors")
    st.dataframe(vs_baseline, use_container_width=True, hide_index=True)
if scorecard.empty and vs_baseline.empty:
    st.info("Run `quant-alpha gat-equity --offline --persist` to populate the GAT dbt marts "
            "(`fct_gat_scorecard`, `fct_gat_vs_baseline`) for the live scorecard.")

# ── Honest take ───────────────────────────────────────────────────────────────
st.divider()
st.info(
    "**The honest take.** The interconnector graph helps price-level forecasts modestly (+0.131) "
    "and learned attention improves cross-sectional ranking (5/5 seeds); a congestion edge feature "
    "did **not** robustly add skill (price-spread proxy 2/5, real flow/NTC 1/5 — a documented null). "
    "But on the irreducibly-relational target — cross-border spreads — message passing robustly beats "
    "a both-endpoint model (+0.056, 5/5). The GNN's value concentrates where the target lives on the network."
)
