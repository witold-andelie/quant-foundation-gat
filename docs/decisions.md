# Open Decisions

These are the points where your direction matters most.

## Resolved for the GAT capstone (2026-06)

- **Asset class** — **both tracks** (dual-track, ADR-0006): US equities is the
  genuine value result; European power (ENTSO-E) is the cautionary-methodology
  result (the energy strategy loses money under honest returns, E13b). A-shares
  out of scope.
- **Data vendor** — yfinance (equity, real) + live ENTSO-E (energy, real, E12).
  Both acknowledged-imperfect (survivorship bias / no balancing-spread data);
  see the experiment-log limitations.
- **Runtime target** — local CPU for all paper runs (E8: GPU is slower at this
  graph size). **Deployment deferred**: when deployed, the read-only Streamlit
  dashboard goes to **Streamlit Community Cloud** (the repo's `requirements.txt`
  is already the slim Streamlit-Cloud set), not Render — for a single read-only
  app Streamlit Cloud is zero-config and Render adds setup for no gain.
  Prerequisite before deploying: build the "GAT vs Baseline" Streamlit page and
  bake the GAT marts into the committed DuckDB.
- **Alpha family** — price-volume cross-sectional (equity, 10 alphas) + energy
  fundamentals (8 alphas); the GAT composes them relationally.
- **Backtest style** — equal-weight top/bottom long-short, four research gates +
  attention A/B anchors. (Energy needs market-structure-aware evaluation, E13b.)
- **Dashboard audience** — research reviewer; GAT marts (`fct_gat_vs_baseline`,
  `fct_gat_scorecard`) are queryable, the Streamlit page is pending.

The original open-decision menu is kept below for context.

1. Asset class
   - Default: US liquid equities.
   - Alternative: European power/energy data, commodity futures, crypto, China A-shares, ETFs.

2. Data vendor
   - Default: Yahoo Finance for demo only.
   - Alternative: Polygon, Tiingo, Databento, Nasdaq Data Link, WRDS, exchange files, ENTSO-E, EPEX.

3. Runtime target
   - Default: local portfolio project.
   - Alternative: GCP with Terraform, GCS, BigQuery, Kestra Cloud, and Streamlit Community Cloud.

4. Alpha family
   - Default: price-volume cross-sectional factors.
   - Alternative: fundamental quality/value, news sentiment, analyst revisions, weather-energy forecasts, order-book microstructure.

5. Backtest style
   - Default: equal-weight top/bottom long-short.
   - Alternative: risk model, optimizer, dollar-neutral portfolio, futures volatility targeting, market-making simulator.

6. Dashboard audience
   - Default: research reviewer.
   - Alternative: PM/risk dashboard, data quality monitor, live trading cockpit.
