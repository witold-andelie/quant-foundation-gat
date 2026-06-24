# Architecture

This project mirrors the Data Engineering Zoomcamp capstone flow while replacing
taxi data with market data and operational dashboards with alpha research and
forecasting outputs. It is extended with a GNN/GAT relational-factor capstone
([gnn_capstone_design.md](gnn_capstone_design.md)) and a leakage-controlled energy
price/spread forecasting track ([energy_forecasting.md](energy_forecasting.md)).

The strategic priority is the Second Foundation-inspired energy track. The US
equities track remains a cleaner portfolio demo and a baseline for
alpha-expression research.

## Layers

1. Ingestion
   - Current: Yahoo Finance daily OHLCV; deterministic synthetic prices; synthetic
     and **live ENTSO-E** power-market data — day-ahead price, load/wind/solar
     forecasts, actual load, generation mix (A75), and **cross-border physical
     flows + NTC** (A11/A61).
   - Next: Polygon, Databento, EPEX, broker feeds, flow-based-coupling capacity.

2. Data lake — local Parquet under `data/raw`/`data/processed`; next GCS/S3/Blob.

3. Warehouse — DuckDB for local reproducibility; next BigQuery or Snowflake.

4. Transformation & factors
   - **Island factors** — Python factor math (WorldQuant-style expressions), dbt
     warehouse marts, Spark batch features.
   - **Relational factors (GNN/GAT capstone)** — a `Factor`/`FactorProvider` seam
     adds factors propagated over a graph. Two heterogeneous graphs (equity
     correlation graph, energy interconnector graph) share one GAT kernel through
     the `Propagator` seam (`UniformMean` baseline | `GAT`, PyG or pure-torch).
     Outputs: composite + four research gates + attention A/B → `fct_gat_*` marts.

5. Orchestration — Kestra daily equity flow and hourly energy flow; next scheduled
   cloud deployment with backfills and stronger data-quality gates.

6. Research applications
   - Alpha backtests (long-short, decay, walk-forward IC) + Streamlit dashboard.
   - **Energy forecasting** (`forecast/`) — a skill-vs-persistence ladder
     (persistence → seasonal → no-graph ridge → uniform-graph → GAT) over real
     ENTSO-E data, for both node-level price and edge-level cross-border spread.
     Findings (E14): the graph improves price-level skill (+0.131) and edge-level
     message passing beats a both-endpoint model on spreads (+0.056, 5/5 seeds);
     congestion-as-edge-feature was an honest null.

## Production Boundary

The skeleton is for research and portfolio demonstration. It does not place
orders, manage broker state, or provide investment advice.
