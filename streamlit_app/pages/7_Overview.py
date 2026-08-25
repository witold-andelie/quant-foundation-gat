"""Platform overview: research outcomes first, engineering evidence second."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    ENERGY_DB,
    ENERGY_TABLES,
    EQUITY_DB,
    EQUITY_TABLES,
    ROOT,
    list_tables,
    pick,
)


RESULTS_DIR = ROOT / "docs" / "results"

st.title("Platform & Reproducibility")
st.caption("Post-GAT research outcomes first; platform evidence and legacy diagnostics second.")

if st.button("Back to Start Here", width="stretch", key="overview_home"):
    st.switch_page("home.py")


def _read_result(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


node = _read_result("energy_forecast_node_skill.csv")
edge = _read_result("energy_forecast_edge_skill.csv")
equity = _read_result("equity_gat_summary.csv")

st.markdown("### Before vs after GAT")
st.write(
    "Each track shows the failed or weak starting point beside the later controlled "
    "graph experiment. Energy changed its prediction target, so its old Sharpe and "
    "new forecast skill are context, not directly comparable units."
)

energy_col, equity_col = st.columns(2)

with energy_col:
    with st.container(border=True):
        st.markdown("#### European electricity")
        st.caption("BEFORE GAT - REJECTED TRADING HYPOTHESIS")
        before_left, before_right = st.columns(2)
        before_left.metric("Legacy Sharpe", "-0.80")
        before_right.metric("Legacy max drawdown", "-100.0%")
        st.error(
            "Cross-sectional electricity alpha was not stable or tradable. "
            "This hypothesis was rejected."
        )

        st.caption("AFTER REFRAME - PRICE AND SPREAD FORECASTING")
        st.markdown(
            """
**Node-price MSE skill**

No graph **0.224** -> Uniform neighbours **0.355** -> GAT **0.347**

**Node ranking**

Uniform rank IC **0.584** -> GAT rank IC **0.612**

**Cross-border spread skill**

Edge ridge **0.192** -> Edge GAT **0.248** (**+0.056**, 5/5 seeds)
"""
        )
        st.success(
            "After: physical topology adds robust information; attention helps "
            "ranking and performs best on the relational spread target."
        )

with equity_col:
    with st.container(border=True):
        st.markdown("#### US equities")
        st.caption("MATCHED SAME-RUN OOS COMPARISON - 404 OBSERVATIONS")
        diagnostics = _read_result("2026-06-10_matrix_static_wf_diagnostics.csv")
        model_order = [
            "alpha_island_mean",
            "alpha_uniform_composite",
            "alpha_gat_composite",
            "alpha_wq_010_gap_quality",
        ]
        labels = {
            "alpha_island_mean": ("Island mean", "No propagation"),
            "alpha_uniform_composite": ("Uniform", "Graph anchor"),
            "alpha_gat_composite": ("GAT", "Learned graph"),
            "alpha_wq_010_gap_quality": ("Best single", "Reference"),
        }
        matched = diagnostics.set_index("alpha_name").loc[model_order].reset_index()
        matched["Model"] = matched["alpha_name"].map(lambda name: labels[name][0])
        matched["Role"] = matched["alpha_name"].map(lambda name: labels[name][1])
        matched["Total Return"] = matched["oos_total_return"].map(
            lambda value: f"{value:.1%}"
        )
        matched["OOS Sharpe"] = matched["oos_sharpe"]
        matched["Max Drawdown"] = matched["oos_max_drawdown"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(
            matched[["Model", "Role", "Total Return", "OOS Sharpe", "Max Drawdown"]],
            width="stretch",
            hide_index=True,
            column_config={
                "OOS Sharpe": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.metric(
            "Tuned GAT OOS Sharpe (5 CPU seeds)",
            "1.37 +/- 0.39",
            delta="+2.42 vs uniform -1.05",
        )
        st.success(
            "Attention lift was positive in 30/30 runs. The best single factor "
            "remains the stricter ceiling at Sharpe 3.07."
        )

st.caption(
    "The equity table is a same-run A/B with identical OOS observations and "
    "evaluation code. Energy changes target after rejecting the trading hypothesis, "
    "so its old Sharpe and new forecast skill are contextual rather than common units. "
    "Seed counts indicate optimisation stability, not significance."
)

st.markdown("### Controlled comparisons")
comparison = pd.DataFrame(
    [
        {
            "Track / target": "Energy node price",
            "Anchor": "No-graph ridge: 0.224 skill",
            "Relational model": "Uniform graph: 0.355 skill",
            "Change": "+0.131",
            "Reading": "Physical neighbour information helps",
        },
        {
            "Track / target": "Energy node price",
            "Anchor": "Uniform: 0.355 skill / 0.584 rank IC",
            "Relational model": "GAT: 0.347 skill / 0.612 rank IC",
            "Change": "-0.008 / +0.028",
            "Reading": "Attention improves ranking, not MSE",
        },
        {
            "Track / target": "Cross-border spread",
            "Anchor": "Edge ridge: 0.192 skill",
            "Relational model": "Edge GAT: 0.248 skill",
            "Change": "+0.056",
            "Reading": "Strongest result on a relational target",
        },
        {
            "Track / target": "US equity composite",
            "Anchor": "Uniform graph: -1.05 OOS Sharpe",
            "Relational model": "Tuned GAT: 1.37 +/- 0.39 OOS Sharpe",
            "Change": "+2.42 mean Sharpe; positive in 30/30 runs",
            "Reading": "Improves anchor; best single factor is 3.07",
        },
    ]
)
st.dataframe(comparison, width="stretch", hide_index=True)

with st.expander("Superseded pre-GAT trading baselines (historical diagnostic only)"):
    st.warning(
        "These figures belong to the original cross-sectional trading-alpha "
        "hypothesis. They are retained to document why that hypothesis was "
        "rejected; they are not the final GAT forecasting results."
    )

    def _legacy_row(db: Path, tables: dict, track: str) -> dict:
        metrics = pick(db, *tables["metrics"])
        registry = pick(db, *tables["registry"])
        diagnostics = pick(db, *tables["diagnostics"])

        def _number(column: str):
            if metrics.empty or column not in metrics.columns:
                return None
            return float(metrics[column].iloc[0])

        passing = None
        sharpe = _number("sharpe")
        max_drawdown = _number("max_drawdown")
        if not diagnostics.empty and "consistency_score" in diagnostics.columns:
            passing = int((diagnostics["consistency_score"] >= 0.5).sum())
        return {
            "Legacy track": track,
            "Sharpe": f"{sharpe:.2f}" if sharpe is not None else None,
            "Max drawdown": (
                f"{max_drawdown:.1%}"
                if max_drawdown is not None
                else None
            ),
            "Raw factors": len(registry) if not registry.empty else None,
            "Consistency gate": (
                f"{passing} / {len(diagnostics)}" if passing is not None else None
            ),
            "Status": "Rejected / contextual baseline",
        }

    legacy = pd.DataFrame(
        [
            _legacy_row(
                ENERGY_DB,
                ENERGY_TABLES,
                "Energy cross-sectional alpha",
            ),
            _legacy_row(
                EQUITY_DB,
                EQUITY_TABLES,
                "Equity naive composite",
            ),
        ]
    )
    st.dataframe(
        legacy,
        width="stretch",
        hide_index=True,
    )
    st.write(
        "The energy research question was subsequently reframed from tradable "
        "cross-sectional alpha to node-price and edge-spread forecasting."
    )

st.divider()
st.subheader("Engineering coverage")

energy_tables = set(list_tables(ENERGY_DB))
equity_tables = set(list_tables(EQUITY_DB))


def _check(condition: bool) -> str:
    return "Ready" if condition else "Not detected"


health = [
    {
        "Module": "Containerisation",
        "Technology": "Docker / Terraform",
        "Status": _check((ROOT / "infra/terraform").exists()),
    },
    {
        "Module": "Orchestration",
        "Technology": "Kestra",
        "Status": _check((ROOT / "flows/kestra").exists()),
    },
    {
        "Module": "Ingestion",
        "Technology": "dlt incremental pipelines",
        "Status": _check(bool(energy_tables | equity_tables)),
    },
    {
        "Module": "Warehouse",
        "Technology": "DuckDB / BigQuery",
        "Status": _check(ENERGY_DB.exists() or EQUITY_DB.exists()),
    },
    {
        "Module": "Analytics engineering",
        "Technology": "dbt",
        "Status": _check((ROOT / "dbt_energy_alpha").exists()),
    },
    {
        "Module": "Data platforms",
        "Technology": "Bruin",
        "Status": _check((ROOT / "bruin/pipelines").exists()),
    },
    {
        "Module": "Batch processing",
        "Technology": "Apache Spark",
        "Status": _check((ROOT / "src/quant_alpha/batch").exists()),
    },
    {
        "Module": "Streaming",
        "Technology": "Redpanda / Avro",
        "Status": _check("live_energy_signals" in energy_tables),
    },
    {
        "Module": "Cloud deployment",
        "Technology": "Helm / Kubernetes",
        "Status": _check((ROOT / "infra/helm").exists()),
    },
    {
        "Module": "CI/CD",
        "Technology": "GitHub Actions",
        "Status": _check((ROOT / ".github/workflows").exists()),
    },
]
st.dataframe(pd.DataFrame(health), width="stretch", hide_index=True)

with st.expander("Factor inventory"):
    energy_registry = pick(ENERGY_DB, *ENERGY_TABLES["registry"])
    equity_registry = pick(EQUITY_DB, *EQUITY_TABLES["registry"])
    chart_cols = st.columns(2)

    def _family_chart(registry: pd.DataFrame, title: str, column):
        with column:
            if registry.empty or "family" not in registry.columns:
                st.info("No registry data available.")
                return
            families = registry["family"].value_counts().reset_index()
            families.columns = ["family", "count"]
            fig = px.pie(
                families,
                names="family",
                values="count",
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

    _family_chart(energy_registry, "Energy factor families", chart_cols[0])
    _family_chart(equity_registry, "Equity factor families", chart_cols[1])

with st.expander("Reproduction commands"):
    st.code(
        """
# Energy pipeline
quant-alpha energy-run

# Equity pipeline
quant-alpha run --offline

# GAT experiment runners
quant-alpha gat-energy
quant-alpha gat-equity
""",
        language="bash",
    )
