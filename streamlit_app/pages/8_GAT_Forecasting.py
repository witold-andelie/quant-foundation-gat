"""Evidence-first summary of the GAT experiments."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "docs" / "results"


def _csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _finding_card(label, value, title, body):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"## {value}")
        st.markdown(f"**{title}**")
        st.write(body)


def _skill_chart(frame, models, title, colours):
    data = frame[frame["model"].isin(models)].copy()
    labels = {
        "no_graph_ridge": "No graph ridge",
        "uniform_graph_ridge": "Uniform neighbours",
        "gat_node": "GAT node",
        "edge_persistence": "Edge persistence",
        "edge_ridge": "Edge ridge",
        "edge_gat": "Edge GAT",
    }
    data["Model"] = data["model"].map(labels).fillna(data["model"])
    fig = px.bar(
        data,
        x="skill",
        y="Model",
        orientation="h",
        text="skill",
        color="Model",
        color_discrete_map=colours,
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=310,
        margin={"l": 10, "r": 30, "t": 55, "b": 10},
        xaxis_title="Skill versus benchmark (higher is better)",
        yaxis_title="",
    )
    return fig


node_results = _csv("energy_forecast_node_skill.csv").rename(columns={"predictor": "model"})
edge_results = _csv("energy_forecast_edge_skill.csv").rename(columns={"predictor": "model"})
equity_summary = _csv("equity_gat_summary.csv")
findings = _csv("energy_gnn_findings.csv")

st.title("Key Findings: Do relationships add predictive value?")
st.info(
    "**Short answer:** yes, the graph itself adds useful information. Learned "
    "attention helps most on ranking and cross-border spread prediction, but it "
    "does not beat simple neighbour averaging on every metric."
)

left, right = st.columns(2)
with left:
    _finding_card(
        "EQUITY",
        "1.37 +/- 0.39",
        "Tuned GAT OOS Sharpe across 5 CPU seeds",
        "The matched uniform anchor was -1.05 (mean lift +2.42). Attention lift "
        "was positive in 30/30 runs; the best single factor remained 3.07.",
    )
with right:
    _finding_card(
        "ENERGY PRICE",
        "+0.131 skill",
        "Physical connectivity helped",
        "Uniform neighbours improved skill from 0.224 to 0.355 over the "
        "no-graph ridge model.",
    )

left, right = st.columns(2)
with left:
    _finding_card(
        "ATTENTION",
        "0.612 rank IC",
        "Better ranking, slightly worse MSE",
        "GAT ranked zones better than uniform aggregation (0.612 vs 0.584), "
        "while MSE skill was 0.347 vs 0.355.",
    )
with right:
    _finding_card(
        "CROSS-BORDER SPREAD",
        "+0.056 skill",
        "Strongest result on a relational target",
        "Edge GAT improved skill from 0.192 to 0.248 over edge ridge in 5 of 5 seeds.",
    )

st.caption(
    "30/30 and 5/5 describe optimisation-seed stability. They are not independent "
    "market samples and are not statistical-significance tests."
)

st.markdown("### How to read the experiment")
ladder = st.columns(3)
steps = [
    ("No graph", "Own-market features only", "What can the node predict alone?"),
    (
        "Uniform neighbours",
        "Simple neighbour average",
        "Does connected-market information add value?",
    ),
    (
        "Learned attention",
        "Flexible GAT model",
        "Does learned reweighting improve over the graph anchor?",
    ),
]
for column, (name, mechanism, question) in zip(ladder, steps):
    with column:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(mechanism)
            st.write(question)

st.warning(
    "Uniform aggregation and GAT are not a capacity-matched, attention-only "
    "ablation: GAT also adds learned projections, nonlinearities, and parameters. "
    "Results therefore show GAT-over-uniform performance, not the pure causal "
    "effect of attention."
)

with st.expander("Metric definitions"):
    st.markdown(
        """
- **OOS Sharpe:** out-of-sample return divided by volatility.
- **Skill:** improvement in forecast error relative to the stated benchmark.
- **Rank IC:** rank correlation between predictions and realised outcomes.
- **Seed stability:** whether repeated optimisation runs produce the same direction
  of comparison; it is not a substitute for temporal or statistical validation.
"""
    )

equity_tab, node_tab, edge_tab, validity_tab = st.tabs(
    [
        "US equities",
        "Energy prices",
        "Cross-border spreads",
        "Validity and limits",
    ]
)

with equity_tab:
    st.subheader("US equities: matched OOS evidence")
    st.write(
        "The primary table uses one static walk-forward run so every row shares "
        "the same 404-observation OOS window, input panel, costs, and evaluation code."
    )
    diagnostics = _csv("2026-06-10_matrix_static_wf_diagnostics.csv")
    model_order = [
        "alpha_island_mean",
        "alpha_uniform_composite",
        "alpha_gat_composite",
        "alpha_wq_010_gap_quality",
    ]
    labels = {
        "alpha_island_mean": ("No propagation", "Island mean"),
        "alpha_uniform_composite": ("Graph anchor", "Uniform aggregation"),
        "alpha_gat_composite": ("Learned graph", "GAT composite"),
        "alpha_wq_010_gap_quality": ("Reference", "Best single factor"),
    }
    matched = diagnostics.set_index("alpha_name").loc[model_order].reset_index()
    matched["Stage"] = matched["alpha_name"].map(lambda name: labels[name][0])
    matched["Model"] = matched["alpha_name"].map(lambda name: labels[name][1])
    comparison = matched[
        [
            "Stage",
            "Model",
            "oos_total_return",
            "oos_annualized_return",
            "oos_annualized_volatility",
            "oos_sharpe",
            "oos_max_drawdown",
            "oos_observations",
        ]
    ].rename(
        columns={
            "oos_total_return": "Total Return",
            "oos_annualized_return": "Ann. Mean Return",
            "oos_annualized_volatility": "Ann. Vol",
            "oos_sharpe": "OOS Sharpe",
            "oos_max_drawdown": "Max Drawdown",
            "oos_observations": "OOS N",
        }
    )
    for column in ["Total Return", "Ann. Mean Return", "Ann. Vol", "Max Drawdown"]:
        comparison[column] = comparison[column].map(lambda value: f"{value:.1%}")
    comparison["OOS N"] = comparison["OOS N"].astype(int)
    st.dataframe(
        comparison,
        width="stretch",
        hide_index=True,
        column_config={
            "OOS Sharpe": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.success(
        "Tuned 5-seed CPU validation: GAT OOS Sharpe 1.37 +/- 0.39 versus "
        "uniform -1.05, a mean lift of +2.42. Attention lift was positive in "
        "30/30 runs across three validation families."
    )
    st.warning(
        "The best single factor remains stronger at OOS Sharpe 3.07, so the "
        "strict value-added gate still fails. Seed repetition measures optimisation "
        "stability, not independent market significance."
    )
    st.write(
        "Both equity alpha runners share the original PyG GATConv-based kernel. "
        "Uniform aggregation and GAT are still not parameter-count matched, so "
        "the result is GAT-over-uniform performance rather than a pure causal "
        "attention effect."
    )
    st.warning(
        "Remaining point-in-time caveat: the static equity graph is estimated on "
        "the full in-sample window and reused within earlier training dates."
    )
    st.caption(
        "Attention coefficients are diagnostics, not causal feature importance. "
        "Observed weights were near-uniform, consistent with modest reweighting."
    )
    if not equity_summary.empty:
        with st.expander("Earlier first-run summary (historical, not the headline)"):
            st.dataframe(equity_summary, width="stretch", hide_index=True)

with node_tab:
    st.subheader("Energy node prices: graph structure helps; attention is mixed")
    st.write(
        "The forecasting implementation uses a GATv2 node model. It is distinct "
        "from the GATConv alpha runners used in the equity experiment."
    )
    if node_results.empty:
        st.warning("Committed node summary CSV was not found.")
    else:
        st.plotly_chart(
            _skill_chart(
                node_results,
                ["no_graph_ridge", "uniform_graph_ridge", "gat_node"],
                "Node-price forecast skill",
                {
                    "No graph ridge": "#94A3B8",
                    "Uniform neighbours": "#2563EB",
                    "GAT node": "#F59E0B",
                },
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        selected = node_results[
            node_results["model"].isin(
                ["no_graph_ridge", "uniform_graph_ridge", "gat_node"]
            )
        ][["model", "skill", "rank_ic"]]
        st.dataframe(
            selected,
            width="stretch",
            hide_index=True,
            column_config={
                "skill": st.column_config.NumberColumn(format="%.3f"),
                "rank_ic": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.markdown(
        """
- Adding physical neighbours improved MSE skill from **0.224 to 0.355**.
- GAT was slightly lower on MSE skill: **0.347 vs 0.355**.
- GAT was better on cross-sectional ranking: **0.612 vs 0.584**.

The defensible conclusion is metric-specific: physical topology helps, while
learned attention trades a little MSE performance for better ranking.
"""
    )

with edge_tab:
    st.subheader("Cross-border spreads: the clearest relational result")
    st.write(
        "The edge model predicts the price difference across an interconnector. "
        "Its dense GAT encoder and MLP edge head are a separate implementation: "
        "node embeddings from network message passing are concatenated for each "
        "edge before spread prediction."
    )
    if edge_results.empty:
        st.warning("Committed edge summary CSV was not found.")
    else:
        st.plotly_chart(
            _skill_chart(
                edge_results,
                ["edge_persistence", "edge_ridge", "edge_gat"],
                "Cross-border spread forecast skill",
                {
                    "Edge persistence": "#94A3B8",
                    "Edge ridge": "#2563EB",
                    "Edge GAT": "#059669",
                },
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.dataframe(
            edge_results[
                edge_results["model"].isin(
                    ["edge_persistence", "edge_ridge", "edge_gat"]
                )
            ][["model", "skill", "rank_ic"]],
            width="stretch",
            hide_index=True,
            column_config={
                "skill": st.column_config.NumberColumn(format="%.3f"),
                "rank_ic": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.success(
        "Edge GAT improved skill from 0.192 to 0.248 (+0.056) over edge ridge "
        "across 5 of 5 seeds."
    )
    st.caption(
        "Because edge ridge already sees both endpoint features, the lift is "
        "consistent with useful whole-network context. It is not a pure causal "
        "message-passing estimate because model capacities are not matched."
    )

with validity_tab:
    st.subheader("What is controlled, and what remains")
    controlled, remaining = st.columns(2)
    with controlled:
        with st.container(border=True):
            st.markdown("**Implemented controls**")
            st.markdown(
                """
- Out-of-sample evaluation and repeated optimisation seeds
- No-graph and uniform-neighbour anchors
- Forecast-vintage versus realised-feature separation
- Synthetic independent-zone negative control
- Committed result artefacts and reproducible runners
"""
            )
    with remaining:
        with st.container(border=True):
            st.markdown("**Documented limitations**")
            st.markdown(
                """
- Static equity graph uses the full in-sample window
- ENTSO-E resampling includes bidirectional interpolation, so publication-time
  and vintage causality still need a full audit
- Model-capacity matching is incomplete
- Seed repetition does not establish economic significance
- Attention weights are not causal explanations
"""
            )

    st.warning(
        "**Honest negative result:** the original cross-sectional electricity "
        "trading-alpha hypothesis was rejected. The work was reframed as price "
        "and spread forecasting rather than hiding the failed hypothesis."
    )
    st.write(
        "The correct claim is that leakage controls are implemented with documented "
        "remaining limitations - not that the system is completely leak-safe or "
        "that false positives have been eliminated."
    )
    st.caption(
        "The previously reported drawdown change from -60.1% to -4.9% is a "
        "descriptive comparison to an unmatched naive baseline; it should not be "
        "presented as causal risk reduction."
    )
    if not findings.empty:
        with st.expander("Committed energy finding matrix"):
            st.dataframe(findings, width="stretch", hide_index=True)

st.divider()
st.markdown(
    """
**Bottom line:** graph structure provides the most robust lift. Learned attention
is not universally superior, but it is most useful when the target is genuinely
relational - especially cross-border electricity spreads.
"""
)
