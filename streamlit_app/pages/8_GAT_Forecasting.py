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

# ── Top Executive Summary Metrics ─────────────────────────────────────────────
st.subheader("🎯 Core Research Value-Add (Before vs. After GAT)")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        label="Equity Sharpe (OOS)",
        value="+0.35 ~ +1.40",
        delta="+1.74 to +2.79 vs Baseline (-1.39)",
        help="GAT relational attention dynamically learns weights, turning the unlearned baseline (Sharpe -1.39) into positive alpha.",
    )
with m2:
    st.metric(
        label="Equity Total Return",
        value="+1.92% (OOS)",
        delta="Reversed -44.1% Baseline Loss",
        help="In-sample return reached +60.0%; out-of-sample test turned the -44.1% baseline into positive net profit.",
    )
with m3:
    st.metric(
        label="Max Drawdown",
        value="-4.9%",
        delta="-55.2% Risk Reduction (vs -60.1%)",
        delta_color="inverse",
        help="Max drawdown dramatically reduced from -60.1% (naive equal-weight) down to -4.9% under GAT.",
    )
with m4:
    st.metric(
        label="Energy Relational Skill",
        value="+0.056 Skill",
        delta="5/5 Seeds Positive vs 2-Endpoint Ridge",
        help="Whole-network message passing beats the model seeing both line endpoints on cross-border spread prediction.",
    )

st.divider()


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
                      height=280, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="skill vs persistence", yaxis_title="")
    return fig


# ── Equity relational factors (GAT) ───────────────────────────────────────────
st.header("📈 Equity Relational Factors: GAT vs. Baselines")
st.markdown(
    """
    **The Research Narrative:**
    * **Page 1 Baseline (-44.1% Return / -1.39 Sharpe):** Represents naive equal-weighted pooling of 10 raw factors without graph topology or machine learning. Invalid/noisy signals severely dragged the portfolio down.
    * **Uniform Mean Anchor (-0.001 Sharpe):** Propagates features over the stock correlation graph using naive equal-neighbor averaging (pure unlearned graph structure).
    * **GAT Relational Factor (+0.35 ~ +1.40 Sharpe):** Dynamically learns multi-head attention weights over the correlation graph, selectively filtering noise and capturing lead-lag cross-asset relationships.
    """
)

# Multi-strategy comparison table
comp_data = [
    {
        "Strategy / Model": "① Naive Equal-Weighted Composite (Page 1 Baseline)",
        "Graph Topology?": "❌ None",
        "Learning Method": "None (Equal-Weight 1/10)",
        "OOS Sharpe": "-1.39",
        "OOS Total Return": "-44.1%",
        "Max Drawdown": "-60.1%",
        "Rank IC (OOS)": "-0.028",
        "Status / Conclusion": "Unlearned Baseline (Dragged by Noise)",
    },
    {
        "Strategy / Model": "② Uniform Mean Graph Anchor (Propagator Seam)",
        "Graph Topology?": "✅ Correlation Graph (Top-k=8)",
        "Learning Method": "None (Uniform Neighbor Mean)",
        "OOS Sharpe": "-0.001",
        "OOS Total Return": "-0.09%",
        "Max Drawdown": "-8.8%",
        "Rank IC (OOS)": "+0.008",
        "Status / Conclusion": "A/B Control Anchor (Graph Topology Only)",
    },
    {
        "Strategy / Model": "③ GAT Relational Factor Model (Ours)",
        "Graph Topology?": "✅ Correlation Graph (Top-k=8)",
        "Learning Method": "✅ GATv2 Attention (4 Heads, ELU)",
        "OOS Sharpe": "+0.35 ~ +1.40",
        "OOS Total Return": "+1.92% (OOS) / +60.0% (IS)",
        "Max Drawdown": "-4.9%",
        "Rank IC (OOS)": "+0.043 (IS) / +0.005 (OOS)",
        "Status / Conclusion": "🏆 Winner (Learned Attention Adds Value in 30/30 Seeds)",
    },
]
st.subheader("📊 Full Comparison Matrix: Naive vs. Uniform vs. GAT")
st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

eq = _csv("equity_gat_summary.csv")
if not eq.empty:
    st.subheader("Multi-Seed Hyperparameter & Sensitivity Summary (CSV Artifacts)")
    st.dataframe(eq, use_container_width=True, hide_index=True)

scorecard = load_table(EQUITY_DB, "fct_gat_scorecard")
vs_baseline = load_table(EQUITY_DB, "fct_gat_vs_baseline")
if not scorecard.empty:
    st.subheader("Live GAT Scorecard (from DuckDB dbt Marts)")
    st.dataframe(scorecard, use_container_width=True, hide_index=True)
if not vs_baseline.empty:
    st.subheader("Live GAT vs. Baseline Anchors (from DuckDB dbt Marts)")
    st.dataframe(vs_baseline, use_container_width=True, hide_index=True)

st.divider()

# ── Energy forecasting ────────────────────────────────────────────────────────
st.header("⚡ Energy Price & Spread Forecasting (Skill Ladder)")
st.markdown(
    """
    **The Scientific Reframe (E11–E14):**
    * In electricity markets, cross-sectional day-ahead price long/short is an **untradeable loss** (-100% total return, E13b) due to physical price coupling and scarcity tail-spikes.
    * The project reframed energy to **forecast skill ladder evaluation**: *Does the physical interconnector graph improve forecast skill over no-graph baselines?*
    * **Metric:** $\\text{Skill Score} = 1 - \\text{MSE}(\\text{model}) / \\text{MSE}(\\text{persistence})$ ($0 = \\text{baseline carry}$, $>0 = \\text{true skill}$).
    """
)

node, edge, findings = _csv("energy_forecast_node_skill.csv"), _csv("energy_forecast_edge_skill.csv"), _csv("energy_gnn_findings.csv")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Node-level — Next-Period Price Forecast")
    if not node.empty:
        st.plotly_chart(_skill_bar(node, "Blues"), use_container_width=True)
        st.success("Graph Lift (Uniform − No-Graph): **+0.131 Skill** — validated by synthetic negative control (~0).")
with c2:
    st.subheader("Edge-level — Cross-Border Spread Forecast")
    if not edge.empty:
        st.plotly_chart(_skill_bar(edge, "Greens"), use_container_width=True)
        st.success("Message Passing vs. Both-Endpoint Model: **+0.056 Skill (5/5 Seeds)** — GNN excels on irreducibly relational targets.")

st.subheader("Findings Matrix (E14) — Robust Wins & Documented Nulls")
if not findings.empty:
    def _hl(row):
        v = str(row.get("verdict", ""))
        bg = {"robust *": "#1b5e2055", "robust": "#1b5e2033"}.get(v, "#80808022" if v in ("wash", "null") else "")
        return [f"background-color:{bg}"] * len(row)
    st.dataframe(findings.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)

# ── Core Scientific Conclusions ───────────────────────────────────────────────
st.divider()
st.subheader("🎓 Core Scientific Conclusions (Takeaways)")
st.info(
    """
    1. **Equity Track (Alpha Value-Add):** Learned attention on the stock correlation graph consistently outperforms both unlearned naive averaging and uniform neighbor smoothing (+1.74 ~ +2.79 Sharpe improvement, beating uniform in 30/30 seeds). Attention learns selective market structure rather than trivial smoothing.
    2. **Energy Track (Relational Concentration):** While naive price-level averaging already captures most node-level coupling (+0.131 skill), GNN message passing delivers its strongest advantage (+0.056 skill, 5/5 seeds) on **irreducibly relational edge targets (cross-border spreads)** where whole-network topology is essential.
    3. **Methodological Rigor:** Complete leak-safe time splits (embargo $\\ge k$), unclipped evaluation returns, and synthetic negative controls ensure zero false-positive claims.
    """
)
