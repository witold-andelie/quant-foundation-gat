"""Story-first landing page for the Quant Alpha Foundation dashboard."""

import plotly.graph_objects as go
import streamlit as st


st.title("Can connected markets improve prediction?")
st.markdown(
    """
Traditional models score each stock or power market in isolation. This project
tests whether information from connected neighbours adds predictive value,
then separates the value of the **graph itself** from the value of
**learned attention**.
"""
)
st.caption(
    "A controlled graph-ML study across US equities and European electricity markets."
)

cta_left, cta_right = st.columns(2)
with cta_left:
    if st.button("See the evidence", type="primary", width="stretch"):
        st.switch_page("pages/8_GAT_Forecasting.py")
with cta_right:
    if st.button("Open performance details", width="stretch"):
        st.switch_page("pages/1_Performance.py")


def _network_figure(nodes, edges, colour):
    fig = go.Figure()
    for source, target in edges:
        x0, y0 = nodes[source]
        x1, y1 = nodes[target]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": colour, "width": 3},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[xy[0] for xy in nodes.values()],
            y=[xy[1] for xy in nodes.values()],
            text=list(nodes),
            mode="markers+text",
            textposition="top center",
            marker={
                "size": 30,
                "color": "#F8FAFC",
                "line": {"color": colour, "width": 3},
            },
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=250,
        margin={"l": 10, "r": 10, "t": 15, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


st.markdown("### Two markets, two meanings of connection")
equity_col, energy_col = st.columns(2)
with equity_col:
    with st.container(border=True):
        st.markdown("**US equities**")
        st.caption("Node = stock; edge = return correlation or sector relationship")
        nodes = {
            "AAPL": (0.0, 1.0),
            "MSFT": (1.0, 1.0),
            "GOOG": (0.0, 0.0),
            "NVDA": (1.0, 0.0),
        }
        edges = [
            ("AAPL", "MSFT"),
            ("AAPL", "GOOG"),
            ("MSFT", "NVDA"),
            ("GOOG", "NVDA"),
        ]
        st.plotly_chart(
            _network_figure(nodes, edges, "#2563EB"),
            width="stretch",
            config={"displayModeBar": False},
        )
with energy_col:
    with st.container(border=True):
        st.markdown("**European electricity**")
        st.caption("Node = bidding zone; edge = physical cross-border interconnector")
        nodes = {
            "FR": (0.0, 0.5),
            "BE": (0.8, 1.0),
            "NL": (1.6, 1.0),
            "DE_LU": (1.2, 0.35),
            "AT": (2.0, 0.0),
            "CZ": (2.4, 0.6),
        }
        edges = [
            ("FR", "BE"),
            ("FR", "DE_LU"),
            ("BE", "NL"),
            ("NL", "DE_LU"),
            ("DE_LU", "AT"),
            ("DE_LU", "CZ"),
            ("AT", "CZ"),
        ]
        st.plotly_chart(
            _network_figure(nodes, edges, "#059669"),
            width="stretch",
            config={"displayModeBar": False},
        )

st.markdown("### The controlled comparison")
ladder = st.columns(3)
items = [
    ("1", "No graph", "Use only the market's own features."),
    ("2", "Uniform neighbours", "Add a simple average of connected neighbours."),
    ("3", "Learned attention", "Let GAT learn different neighbour weights."),
]
for column, (step, title, body) in zip(ladder, items):
    with column:
        with st.container(border=True):
            st.caption(f"STEP {step}")
            st.markdown(f"**{title}**")
            st.write(body)
st.caption(
    "This ladder asks two questions: does connectivity help, and does the more "
    "flexible GAT model improve over the uniform-neighbour anchor?"
)

st.markdown("### Four conclusions to remember")


def _finding_card(label, value, title, body):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"## {value}")
        st.markdown(f"**{title}**")
        st.write(body)


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
        "Physical graph structure helped",
        "Uniform neighbours improved skill from 0.224 to 0.355 versus the "
        "no-graph ridge baseline.",
    )

left, right = st.columns(2)
with left:
    _finding_card(
        "ATTENTION",
        "0.612 rank IC",
        "Attention was not universally better",
        "GAT improved ranking over uniform aggregation (0.612 vs 0.584), but "
        "its MSE skill was slightly lower (0.347 vs 0.355).",
    )
with right:
    _finding_card(
        "CROSS-BORDER SPREAD",
        "+0.056 skill",
        "The strongest result was relational",
        "Edge GAT improved skill from 0.192 to 0.248 over edge ridge and won "
        "across 5 of 5 optimisation seeds.",
    )

st.caption(
    "Seed counts measure optimisation stability under repeated training; they are "
    "not independent market samples or a statistical-significance test."
)

with st.container(border=True):
    st.markdown("**Honest negative result**")
    st.write(
        "The original hypothesis - tradable cross-sectional electricity alpha - "
        "was not supported. The energy track was reframed as node-price and "
        "cross-border-spread forecasting, where the target matches the network."
    )

st.markdown("### Choose your reading path")
paths = st.columns(3)
with paths[0]:
    st.page_link(
        "pages/8_GAT_Forecasting.py",
        label="Key findings",
        help="Results, controls, caveats, and interpretation.",
    )
with paths[1]:
    st.page_link(
        "pages/1_Performance.py",
        label="Research evidence",
        help="Backtest and factor-level diagnostics.",
    )
with paths[2]:
    st.page_link(
        "pages/7_Overview.py",
        label="Platform appendix",
        help="Pipelines, warehouse, orchestration, and reproducibility.",
    )

with st.expander("Plain-language glossary"):
    st.markdown(
        """
- **Factor:** a numeric signal used to rank or forecast markets.
- **Graph:** markets represented as nodes and their relationships as edges.
- **GAT:** a graph attention network that learns neighbour weights.
- **Skill:** improvement over a benchmark; higher is better.
- **Rank IC:** how well predicted rankings match realised rankings.
- **Sharpe ratio:** return per unit of volatility; useful only with evaluation context.
"""
    )

st.divider()
st.link_button(
    "View source repository",
    "https://github.com/witold-andelie/quant-foundation-gat",
)
